#!/usr/bin/env python3
"""Discover native cuVS tests and make explicit the supported build/run coverage."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys


MG_TARGETS = {"NEIGHBORS_MG_TEST", "CLUSTER_KMEANS_MG_TEST", "KMEANS_MG_C_TEST", "MG_C_TEST"}
MNMG_TARGETS = {"CLUSTER_KMEANS_MNMG_TEST"}
COMPILE_ONLY = {
    "cuvs_c_neighbors_test": (
        "C API compilation check: source explicitly says its main must not run; "
        "it calls compile-only functions with uninitialized API arguments."
    ),
}
KEYWORDS = {"NAME", "PATH", "GPUS", "PERCENT", "ADDITIONAL_DEP", "NOCUDA", "FETCH_CODEBOOKS", "RUN_SERIAL"}


def configure_test_calls(contents):
    """Read literal ConfigureTest calls, failing on syntax we cannot safely classify."""
    contents = re.sub(r"#[^\n]*", "", contents)
    for match in re.finditer(r"\bConfigureTest\s*\(", contents, re.IGNORECASE):
        position, depth, quote = match.end(), 1, False
        start = position
        while position < len(contents) and depth:
            char = contents[position]
            if char == '"' and (position == 0 or contents[position - 1] != "\\"):
                quote = not quote
            elif not quote:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
            position += 1
        if depth:
            raise ValueError("Unterminated ConfigureTest call")
        yield shlex.split(contents[start:position - 1])


def discover_tests(source_dir):
    source = Path(source_dir).resolve()
    discovered = []
    seen = set()
    for relative in ("cpp/tests/CMakeLists.txt", "c/tests/CMakeLists.txt"):
        cmake = source / relative
        for arguments in configure_test_calls(cmake.read_text()):
            try:
                name = arguments[arguments.index("NAME") + 1]
                offset = arguments.index("PATH") + 1
            except (ValueError, IndexError) as error:
                raise ValueError(f"Unsupported ConfigureTest declaration in {relative}") from error
            if "NOCUDA" in arguments:
                name += "_NOCUDA"
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in seen:
                raise ValueError(f"Invalid or duplicate native test target: {name}")
            seen.add(name)
            paths = []
            for token in arguments[offset:]:
                if token in KEYWORDS:
                    break
                path = cmake.parent / token
                if not path.is_file():
                    raise ValueError(f"Test {name} references unsupported or missing source {path}")
                paths.append(path)
            if not paths:
                raise ValueError(f"Test {name} has no source files")
            own_main = any(re.search(r"\bint\s+main\s*\(", path.read_text()) for path in paths)
            entry = {
                "name": name,
                "kind": "command" if own_main else "gtest",
                "suite": "cpp" if relative.startswith("cpp/") else "c",
                "sources": [str(path.relative_to(source)) for path in paths],
            }
            if entry["kind"] == "gtest":
                entry["gtest_filter"] = "*"
            else:
                entry["command"] = ["bin/" + name]
            if "FETCH_CODEBOOKS" in arguments:
                entry["data_requirement"] = "Vamana randomized codebooks from the configured tests/data directory"
            discovered.append(entry)
    return discovered


def check_headers(source_dir):
    include = Path(source_dir).resolve() / "c/include"
    aggregate = include / "cuvs/core/all.h"
    contents = re.sub(r"/\*.*?\*/|//[^\n]*", "", aggregate.read_text(), flags=re.DOTALL)
    includes = set(re.findall(r"#\s*include\s*<([^>]+)>", contents))
    headers = sorted(str(path.relative_to(include)) for path in include.rglob("*.h") if path != aggregate)
    missing = [name for name in headers if name not in includes]
    if missing:
        raise ValueError("cuvs/core/all.h does not include: " + ", ".join(missing))
    return {"name": "cuvs_c_verify_install_headers", "status": "passed", "headers_checked": len(headers)}


def create_plan(source_dir, cuda_version, inventory, ctest_json=None, *, enable_mg=True, enable_mnmg=True):
    source = Path(source_dir).resolve()
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", str(cuda_version))
    if not match:
        raise ValueError(f"Invalid CUDA version: {cuda_version!r}")
    cuda = tuple(map(int, match.groups()))
    if enable_mnmg and not enable_mg:
        raise ValueError("MNMG tests require enable_mg=True")
    plan = {
        "schema_version": 1,
        "scope": "Native C++ and C API tests",
        "cuda_version": str(cuda_version),
        "gpu_count_in_inventory": len(inventory.get("gpus", [])),
        "tests": [], "excluded": [], "build_only": [], "build_targets": [],
        "capability_notes": [
            "MG/MNMG tests use available CUDA ranks; a single visible CUDA device does not validate communication between different devices or hosts.",
        ],
        "required_cmake": {
            "BUILD_TESTS": "ON", "BUILD_C_TESTS": "ON", "BUILD_C_LIBRARY": "ON",
            "BUILD_CAGRA_HNSWLIB": "ON", "BUILD_SHARED_LIBS": "ON",
            "BUILD_MG_ALGOS": "ON" if enable_mg else "OFF",
            "BUILD_MNMG_TESTS": "ON" if enable_mnmg else "OFF",
        },
        "local_checks": [{
            "name": "cuvs_c_verify_install_headers", "kind": "command",
            "command": [sys.executable, str(Path(__file__).resolve()), "--source-dir", str(source), "--check-headers"],
            "description": "Check every source C header is included by cuvs/core/all.h.",
        }],
        "notes": [
            "Run executables sequentially; individual native cases retain their upstream parameter sizes.",
            "Runtime GoogleTest skips are recorded separately and are not counted as passed cases.",
            "CLUSTER_TEST includes 10,000,000 x 128 balanced-kmeans inputs; memory failures must be reported, not silently filtered.",
            "Some NEIGHBORS_TEST cases legitimately skip unsupported cuSparse half inputs or unavailable top_1_nn backends.",
            "MG target names alone do not imply a minimum of two GPUs; upstream tests can use one rank when the required libraries are enabled.",
        ],
        "source_sha256": {
            path: hashlib.sha256((source / path).read_bytes()).hexdigest()
            for path in ("cpp/tests/CMakeLists.txt", "c/tests/CMakeLists.txt")
        },
    }
    if cuda < (13, 0):
        plan["capability_notes"].append(
            f"CUDA {cuda_version} cannot build cuTile kernels (CUDA 13.0+ required). "
            "CUTILE_SMOKE_TEST remains selected so its upstream DisabledBuild runtime skip is recorded."
        )
    for entry in discover_tests(source):
        name = entry["name"]
        reason = None
        if name in MNMG_TARGETS and not enable_mnmg:
            reason = "Requires BUILD_MNMG_TESTS=ON, BUILD_MG_ALGOS=ON, and NCCL/UCX/UCXX; this profile does not enable the distributed toolchain."
        elif name in MG_TARGETS and not enable_mg:
            reason = "Requires BUILD_MG_ALGOS=ON and NCCL; this profile does not enable the multi-GPU toolchain."
        if reason:
            plan["excluded"].append({"name": name, "reason": reason, "suite": entry["suite"]})
        elif name in COMPILE_ONLY:
            plan["build_only"].append({**entry, "reason": COMPILE_ONLY[name]})
            plan["build_targets"].append(name)
        else:
            plan["tests"].append(entry)
            plan["build_targets"].append(name)
    if ctest_json is not None:
        configured = {test["name"] for test in ctest_json.get("tests", [])}
        required = {entry["name"] for entry in plan["tests"] + plan["build_only"]}
        missing = sorted(required - configured)
        if missing:
            raise ValueError("Selected native tests are not configured: " + ", ".join(missing))
        plan["configured_tests_validated"] = True
        plan["other_configured_tests"] = sorted(configured - required)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--cuda-version")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--ctest-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disable-mg", action="store_true", help="Explicitly omit MG and MNMG targets")
    parser.add_argument("--disable-mnmg", action="store_true", help="Explicitly omit distributed-toolchain targets")
    parser.add_argument("--check-headers", action="store_true")
    args = parser.parse_args()
    try:
        if args.check_headers:
            print(json.dumps(check_headers(args.source_dir), indent=2))
            return
        if not args.cuda_version or not args.inventory or not args.output:
            parser.error("planning requires --cuda-version, --inventory, and --output")
        plan = create_plan(
            args.source_dir, args.cuda_version, json.loads(args.inventory.read_text()),
            json.loads(args.ctest_json.read_text()) if args.ctest_json else None,
            enable_mg=not args.disable_mg, enable_mnmg=not (args.disable_mg or args.disable_mnmg),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Planned {len(plan['tests'])} remote tests, {len(plan['build_only'])} build-only checks, "
              f"{len(plan['local_checks'])} local checks, {len(plan['excluded'])} explicit exclusions: {args.output}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"test-plan: {error}\n")


if __name__ == "__main__":
    main()
