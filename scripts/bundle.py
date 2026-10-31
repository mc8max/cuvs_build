#!/usr/bin/env python3
"""Package locally built, trusted cuVS ELF files for execution on RunPod."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


MARKER = ".cuvs-runtime-bundle"
MARKER_CONTENT = "cuvs-build-runtime-bundle-v1\n"
SYSTEM_LIBRARY = re.compile(
    r"^(?:linux-vdso\.so|ld-linux[^/]*\.so|lib(?:c|m|pthread|dl|rt|util|resolv|anl)\.so|libnss_[^.]+\.so)"
)
DRIVER_LIBRARY = re.compile(r"^lib(?:cuda|nvidia-ml)\.so(?:\.|$)")
VAMANA_CODEBOOKS = (
    "384_int8_pq_pivots.bin", "384_int8_pq_pivots.bin_rotation_matrix.bin",
    "64_float_pq_pivots.bin", "64_float_pq_pivots.bin_rotation_matrix.bin",
)


def inside(path, root):
    return path == root or root in path.parents


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def is_elf(path):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def dependencies(path):
    """Read direct dependencies without applying the build host's RPATH rules."""
    result = subprocess.run(
        ["readelf", "--dynamic", "--wide", str(path)], text=True, capture_output=True, check=True
    )
    return re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout)


def library_index(paths, include_unbuilt=False):
    result = {}
    for path in paths:
        if include_unbuilt or is_elf(path):
            result.setdefault(path.name, set()).add(path.resolve())
    return result


def active_build_libraries(build, prefix):
    """Ignore old .so files left behind when CMake switches to imported libraries."""
    if not (build / "build.ninja").is_file():
        # Standalone ELF fixture trees have no CMake build graph.
        if (build / "CMakeCache.txt").is_file():
            raise RuntimeError("The configured build must have a Ninja graph to identify active runtime libraries")
        return build.rglob("*.so*")
    ninja = prefix / "bin/ninja"
    if not ninja.is_file():
        found = shutil.which("ninja")
        if not found:
            raise RuntimeError("ninja is required to identify the current build graph's runtime libraries")
        ninja = Path(found)
    result = subprocess.run([str(ninja), "-C", str(build), "-t", "targets", "all"],
                            text=True, capture_output=True, check=True)
    outputs = []
    for line in result.stdout.splitlines():
        name, separator, rule = line.rpartition(": ")
        if not separator or rule == "phony" or not re.search(r"\.so(?:\.|$)", Path(name).name):
            continue
        path = build / name
        if inside(path.resolve(), build):
            outputs.append(path)
    return outputs


def resolve_library(name, indexes):
    for index in indexes:
        sources = index.get(name, set())
        if len(sources) > 1:
            raise RuntimeError(f"Ambiguous dependency {name}: {', '.join(map(str, sorted(sources)))}")
        if sources:
            return next(iter(sources))
    return None


def required_glibc(paths):
    versions = set()
    for path in paths:
        result = subprocess.run(
            ["readelf", "--version-info", str(path)], text=True, capture_output=True, check=True
        )
        versions.update(re.findall(r"\bGLIBC_(\d+(?:\.\d+)+)\b", result.stdout))
    return max(versions, key=lambda value: tuple(map(int, value.split("."))), default=None)


def target_path(build, name):
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", name):
        raise RuntimeError(f"Invalid build target name: {name!r}")
    known = {"cuvs": build / "libcuvs.so", "cuvs_c": build / "c/libcuvs_c.so"}
    if name in known:
        path = known[name]
        if not is_elf(path):
            raise RuntimeError(f"Build target is missing or is not ELF: {path}")
        return path, "lib"
    # Use upstream output locations first, avoiding a full build-tree scan per
    # executable when packaging the complete native suite.
    candidates = [directory / name for directory in
                  (build / "gtests", build / "c/gtests", build / "bench/ann")
                  if is_elf(directory / name) and os.access(directory / name, os.X_OK)]
    if not candidates:
        candidates = [path for path in build.rglob(name) if is_elf(path) and os.access(path, os.X_OK)]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one built executable for {name}; found {len(candidates)}")
    return candidates[0], "bin"


def verify_destination(output, protected):
    if output.is_symlink():
        raise RuntimeError(f"Refusing symlink output directory: {output}")
    if any(inside(path, output.resolve()) for path in protected):
        raise RuntimeError(f"Output directory contains a source input: {output}")
    if output.exists():
        marker = output / MARKER
        if not output.is_dir() or not marker.is_file() or marker.read_text() != MARKER_CONTENT:
            raise RuntimeError(f"Refusing to replace a directory not created by bundle.py: {output}")


def test_data_files(build, targets):
    source = build / "tests/data"
    if "NEIGHBORS_ANN_VAMANA_TEST" in targets:
        for name in VAMANA_CODEBOOKS:
            path = source / "neighbors/ann_vamana/randomized_codebooks" / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Vamana test data is missing or empty: {path}")
    if not source.exists():
        return []
    if not source.is_dir() or not inside(source.resolve(), build):
        raise RuntimeError(f"Test data directory is outside the build tree: {source}")
    files = []
    for path in sorted(source.rglob("*")):
        if not inside(path.resolve(), build):
            raise RuntimeError(f"Test data path escapes the build tree: {path}")
        if path.is_dir():
            if path.is_symlink():
                raise RuntimeError(f"Symlinked test data directories are not supported: {path}")
            continue
        if not path.is_file():
            raise RuntimeError(f"Test data is not a regular file: {path}")
        files.append((path, path.relative_to(build)))
    return files


def load_test_plan(path, targets):
    plan = json.loads(path.read_text())
    if not isinstance(plan, dict) or not isinstance(plan.get("tests"), list) or not plan["tests"]:
        raise RuntimeError("Test plan must contain a nonempty tests list")
    names = []
    for item in plan["tests"]:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_+-]+", name):
            raise RuntimeError(f"Invalid test plan target: {name!r}")
        names.append(name)
    missing = set(names) - set(targets)
    if missing:
        raise RuntimeError(f"Test plan targets are missing from --targets: {', '.join(sorted(missing))}")
    return set(names)


def create_bundle(args):
    build = args.build_dir.resolve(strict=True)
    prefix = args.env_prefix.resolve(strict=True)
    config = args.config.resolve(strict=True)
    inventory = args.inventory.resolve(strict=True)
    test_plan = getattr(args, "test_plan", None)
    test_plan = test_plan.resolve(strict=True) if test_plan is not None else None
    test_names = load_test_plan(test_plan, args.targets) if test_plan else set()
    helpers = Path(__file__).resolve().parent
    if test_plan and not (helpers / "run_tests.py").is_file():
        raise RuntimeError("A test plan requires scripts/run_tests.py in the bundle")
    # Validate metadata before copying any large libraries.
    for path in (config, inventory):
        json.loads(path.read_text())
    artifacts = [(name, *target_path(build, name)) for name in args.targets]
    for name, _, directory in artifacts:
        if name in test_names and directory != "bin":
            raise RuntimeError(f"Test plan entry is not an executable target: {name}")
    data_files = test_data_files(build, args.targets)
    protected_inputs = (build, prefix, config, inventory) + ((test_plan,) if test_plan else ())
    output = args.output.absolute()
    verify_destination(output, protected_inputs)
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to remove local absolute RPATHs from runtime artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Active build-produced SONAMEs win over prebuilt Conda packages. Stale
    # outputs from an earlier configuration must not shadow imported libraries.
    # An active output that is missing must fail when referenced, rather than
    # silently substituting an installed library with the same SONAME.
    build_libraries = library_index(active_build_libraries(build, prefix), include_unbuilt=True)
    library_dirs = (prefix / "lib", prefix / "targets/x86_64-linux/lib", prefix / "lib/ucx")
    toolchain_libraries = library_index(path for directory in library_dirs for path in directory.glob("*.so*"))
    scratch = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (scratch / "lib").mkdir()
        (scratch / "bin").mkdir()
        (scratch / MARKER).write_text(MARKER_CONTENT)
        pending = []
        copied = {}
        destinations = {}
        visited = set()
        system_libraries = set()
        driver_libraries = {"libcuda.so.1"}
        ucx_modules = []
        ucx_modules_added = False

        def copy_elf(source, name, directory=None):
            source = source.resolve(strict=True)
            if directory is None:
                directory = "lib/ucx" if inside(source, prefix / "lib/ucx") else "lib"
            if not inside(source, build) and not inside(source, prefix):
                raise RuntimeError(f"Dependency {name} resolves outside build/toolchain: {source}")
            if "stubs" in source.parts or DRIVER_LIBRARY.match(name) or SYSTEM_LIBRARY.match(name):
                raise RuntimeError(f"Refusing to bundle a driver, stub, or glibc library: {source}")
            if name != Path(name).name or not is_elf(source):
                raise RuntimeError(f"Invalid ELF dependency: {name!r} ({source})")
            dest = scratch / directory / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest in destinations:
                if destinations[dest] != source:
                    raise RuntimeError(f"Conflicting libraries named {name}: {source} and {destinations[dest]}")
                return
            copy_key = (source, directory)
            if copy_key in copied:
                # Dereference external symlinks; internal aliases share bytes through hardlinks.
                os.link(copied[copy_key], dest)
            else:
                shutil.copy2(source, dest)
                rpath = {"lib": "$ORIGIN", "bin": "$ORIGIN/../lib", "lib/ucx": "$ORIGIN:$ORIGIN/.."}[directory]
                subprocess.run([patchelf, "--set-rpath", rpath, str(dest)], check=True, capture_output=True)
                # Stable timestamps let rsync --link-dest reuse unchanged files.
                status = source.stat()
                os.utime(dest, ns=(status.st_atime_ns, status.st_mtime_ns))
                copied[copy_key] = dest
            destinations[dest] = source
            pending.append(source)

        for _, path, directory in artifacts:
            copy_elf(path, path.name, directory)

        # These libraries are dlopened and do not necessarily appear in ldd output.
        for pattern in ("libnvrtc-builtins*.so*", "libcufile.so*"):
            matches = sorted((prefix / "lib").glob(pattern))
            if not matches:
                raise RuntimeError(f"Required runtime library missing from toolchain: {pattern}")
            for path in matches:
                copy_elf(path, path.name)

        while pending:
            source = pending.pop()
            if source in visited:
                continue
            visited.add(source)
            if not ucx_modules_added and re.match(r"libuc(?:p|s|t|m|xx)\.so", source.name):
                ucx_modules_added = True
                modules = sorted({path for pattern in ("libuct*.so*", "libucm*.so*")
                                  for path in (prefix / "lib/ucx").glob(pattern) if is_elf(path)})
                if not modules:
                    raise RuntimeError("UCX runtime is required but its transport modules are missing from lib/ucx")
                for module in modules:
                    copy_elf(module, module.name, "lib/ucx")
                    ucx_modules.append("lib/ucx/" + module.name)
            for name in dependencies(source):
                if DRIVER_LIBRARY.match(name):
                    driver_libraries.add(name)
                elif SYSTEM_LIBRARY.match(name):
                    system_libraries.add(name)
                else:
                    location = resolve_library(name, (build_libraries, toolchain_libraries))
                    if location is None:
                        raise RuntimeError(f"Unresolved dependency {name}, required by {source}")
                    copy_elf(location, name)

        shutil.copy2(config, scratch / "build-config.json")
        shutil.copy2(inventory, scratch / "remote-inventory.json")
        if test_plan:
            shutil.copy2(test_plan, scratch / "test-plan.json")
        for source, relative in data_files:
            destination = scratch / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        (scratch / "scripts").mkdir()
        for name in ("make_smoke_dataset.py", "remote-run.sh"):
            shutil.copy2(helpers / name, scratch / "scripts" / name)
        if (helpers / "run_tests.py").is_file():
            shutil.copy2(helpers / "run_tests.py", scratch / "scripts/run_tests.py")
        launcher = scratch / "run"
        launcher.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'bundle_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)\n'
            'export LD_LIBRARY_PATH="$bundle_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            'export CI=true\n'
            'export RAPIDS_DATASET_ROOT_DIR="$bundle_root/tests/data/"\n'
            'if [[ -d "$bundle_root/lib/ucx" ]]; then export UCX_MODULE_DIR="$bundle_root/lib/ucx"; fi\n'
            'if (($# == 0)); then echo "Usage: $0 EXECUTABLE [ARGUMENT ...]" >&2; exit 2; fi\n'
            'case "$1" in\n'
            '  /*) exec "$@" ;;\n'
            '  */*) executable="$1"; shift; exec "$bundle_root/$executable" "$@" ;;\n'
            '  *) executable="$1"; shift; exec "$bundle_root/bin/$executable" "$@" ;;\n'
            'esac\n'
        )
        launcher.chmod(0o755)
        manifest = {
            "format": 1,
            "targets": args.targets,
            "system_libraries": sorted(system_libraries),
            "driver_libraries": sorted(driver_libraries),
            "max_required_glibc": required_glibc(visited),
            "test_plan_sha256": digest(test_plan) if test_plan else None,
            "ucx_modules": ucx_modules,
            "files": {
                str(path.relative_to(scratch)): {"sha256": digest(path), "size": path.stat().st_size}
                for path in sorted(scratch.rglob("*")) if path.is_file()
            },
        }
        (scratch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            verify_destination(output, protected_inputs)
            # Preserve the old bundle until the complete replacement has been prepared.
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-old-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
            try:
                scratch.rename(output)
            except BaseException:
                backup.rename(output)
                raise
            shutil.rmtree(backup)
        else:
            scratch.rename(output)
        print(f"Packaged {len(visited)} ELF files for {', '.join(args.targets)} in {output}")
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build-dir", "env-prefix", "output", "config", "inventory"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, help="Test selection and exclusions to deploy verbatim")
    parser.add_argument("--targets", nargs="+", required=True)
    args = parser.parse_args()
    try:
        create_bundle(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"bundle: {error}\n")


if __name__ == "__main__":
    main()
