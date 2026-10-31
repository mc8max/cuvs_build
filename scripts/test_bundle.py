"""Exercise dependency discovery and relocation with tiny real ELF fixtures."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import bundle


@unittest.skipUnless(shutil.which("gcc") and shutil.which("readelf") and shutil.which("patchelf"),
                     "requires local ELF build tools")
class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cuvs-bundle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / "build"
        self.prefix = self.root / "env"
        (self.build / "gtests").mkdir(parents=True)
        (self.prefix / "lib").mkdir(parents=True)
        self.source = self.root / "dep.c"
        self.source.write_text("int answer(void) { return 42; }\n")
        self.library = self.prefix / "lib/libfixture.so.1"
        subprocess.run(["gcc", "-shared", "-fPIC", str(self.source), "-o", str(self.library),
                        "-Wl,-soname,libfixture.so.1"], check=True)
        for name in ("libnvrtc-builtins.so.12.9", "libcufile.so.0"):
            shutil.copy2(self.library, self.prefix / "lib" / name)
        (self.prefix / "lib/libnvrtc-builtins.so").symlink_to("libnvrtc-builtins.so.12.9")
        (self.prefix / "lib/libcufile.so").symlink_to("libcufile.so.0")
        self.source.write_text("extern int answer(void); int main(void) { return answer() == 42 ? 0 : 1; }\n")
        subprocess.run(["gcc", str(self.source), str(self.library),
                        f"-Wl,-rpath,{self.prefix / 'lib'}", "-o", str(self.build / "gtests/SMOKE_TEST")],
                       check=True)
        self.config = self.root / "config.json"
        self.inventory = self.root / "inventory.json"
        self.config.write_text("{}\n")
        self.inventory.write_text("{}\n")
        self.helpers = self.root / "helpers"
        self.helpers.mkdir()
        for name in ("make_smoke_dataset.py", "remote-run.sh", "run_tests.py"):
            (self.helpers / name).write_text("# test fixture\n")
        self.args = argparse.Namespace(build_dir=self.build, env_prefix=self.prefix,
                                       output=self.root / "output", config=self.config,
                                       inventory=self.inventory, targets=["SMOKE_TEST"])

    def package(self):
        with patch.object(bundle, "__file__", str(self.helpers / "bundle.py")):
            with contextlib.redirect_stdout(io.StringIO()):
                bundle.create_bundle(self.args)

    def test_bundle_relocates_and_uses_no_external_symlinks(self):
        self.package()
        output = self.args.output
        self.prefix.rename(self.root / "original-env-unavailable")
        self.build.rename(self.root / "original-build-unavailable")
        subprocess.run([str(output / "run"), "SMOKE_TEST"], cwd="/", check=True)
        subprocess.run([str(output / "run"), "bin/SMOKE_TEST"], cwd="/", check=True)
        self.assertFalse(any(path.is_symlink() for path in output.rglob("*")))
        self.assertFalse((output / "lib/libc.so.6").exists())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertIn("libc.so.6", manifest["system_libraries"])
        self.assertIsNotNone(manifest["max_required_glibc"])
        for name, record in manifest["files"].items():
            self.assertEqual(bundle.digest(output / name), record["sha256"])

    def test_missing_dependency_preserves_previous_bundle(self):
        self.package()
        previous = (self.args.output / "manifest.json").read_bytes()
        self.library.unlink()
        with self.assertRaisesRegex(RuntimeError, "Unresolved dependency libfixture.so.1"):
            self.package()
        self.assertEqual(previous, (self.args.output / "manifest.json").read_bytes())

    def test_refuses_unowned_output_and_driver_stubs(self):
        self.args.output.mkdir()
        (self.args.output / "keep.txt").write_text("user file")
        with self.assertRaisesRegex(RuntimeError, "not created by bundle.py"):
            self.package()
        self.assertEqual("user file", (self.args.output / "keep.txt").read_text())
        shutil.rmtree(self.args.output)
        self.library.rename(self.prefix / "lib/libreal.so.1")
        stub_dir = self.prefix / "lib/stubs"
        stub_dir.mkdir()
        shutil.copy2(self.prefix / "lib/libreal.so.1", stub_dir / "libfixture.so.1")
        self.library.symlink_to("stubs/libfixture.so.1")
        with self.assertRaisesRegex(RuntimeError, "driver, stub, or glibc"):
            self.package()

    def test_replaces_owned_bundle(self):
        self.package()
        first_manifest = (self.args.output / "manifest.json").read_bytes()
        first_timestamp = (self.args.output / "bin/SMOKE_TEST").stat().st_mtime_ns
        self.package()
        self.assertEqual((self.args.output / "manifest.json").read_bytes(), first_manifest)
        self.assertEqual((self.args.output / "bin/SMOKE_TEST").stat().st_mtime_ns, first_timestamp)

    def test_test_plan_lowercase_target_and_data_are_relocatable_and_hashed(self):
        (self.build / "c/gtests").mkdir(parents=True)
        shutil.copy2(self.build / "gtests/SMOKE_TEST", self.build / "c/gtests/c_api_smoke")
        self.args.targets.append("c_api_smoke")
        self.args.test_plan = self.root / "test-plan.json"
        self.args.test_plan.write_text(json.dumps({
            "tests": [{"name": "SMOKE_TEST", "kind": "gtest", "gtest_filter": "*"},
                      {"name": "c_api_smoke", "kind": "command"}],
            "excluded": [{"name": "MULTI_GPU_TEST", "reason": "Requires multiple GPUs"}],
        }, indent=2) + "\n")
        relative = Path("tests/data/neighbors/ann_vamana/randomized_codebooks/codebook.bin")
        (self.build / relative).parent.mkdir(parents=True)
        (self.build / relative).write_bytes(b"fixture codebook")
        self.package()
        output = self.args.output
        self.assertEqual((output / "test-plan.json").read_bytes(), self.args.test_plan.read_bytes())
        self.assertEqual((output / relative).read_bytes(), b"fixture codebook")
        self.assertTrue((output / "scripts/run_tests.py").is_file())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["test_plan_sha256"], bundle.digest(self.args.test_plan))
        for name in (str(relative), "test-plan.json", "scripts/run_tests.py"):
            self.assertEqual(manifest["files"][name]["sha256"], bundle.digest(output / name))
        self.prefix.rename(self.root / "original-env-unavailable")
        self.build.rename(self.root / "original-build-unavailable")
        subprocess.run([str(output / "run"), "c_api_smoke"], cwd="/", check=True)
        environment = subprocess.run([str(output / "run"), "/usr/bin/env"], cwd="/",
                                     text=True, capture_output=True, check=True).stdout
        self.assertIn("CI=true", environment.splitlines())
        self.assertIn(f"RAPIDS_DATASET_ROOT_DIR={output}/tests/data/", environment.splitlines())

    def test_test_plan_cannot_reference_unbundled_targets_or_missing_runner(self):
        self.args.test_plan = self.root / "test-plan.json"
        self.args.test_plan.write_text('{"tests": [{"name": "MISSING_TEST"}]}')
        with self.assertRaisesRegex(RuntimeError, "missing from --targets"):
            self.package()
        self.args.test_plan.write_text('{"tests": [{"name": "SMOKE_TEST"}]}')
        (self.helpers / "run_tests.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "requires scripts/run_tests.py"):
            self.package()
        self.assertFalse(self.args.output.exists())

    def test_data_cannot_escape_build_tree(self):
        source = self.build / "tests/data"
        source.mkdir(parents=True)
        (source / "outside.json").symlink_to(self.config)
        with self.assertRaisesRegex(RuntimeError, "escapes the build tree"):
            self.package()
        self.assertFalse(self.args.output.exists())

    def test_vamana_requires_nonempty_downloaded_codebooks(self):
        target = "NEIGHBORS_ANN_VAMANA_TEST"
        shutil.copy2(self.build / "gtests/SMOKE_TEST", self.build / "gtests" / target)
        self.args.targets.append(target)
        with self.assertRaisesRegex(RuntimeError, "Vamana test data is missing or empty"):
            self.package()
        source = self.build / "tests/data/neighbors/ann_vamana/randomized_codebooks"
        source.mkdir(parents=True)
        for name in bundle.VAMANA_CODEBOOKS:
            (source / name).write_bytes(b"fixture codebook")
        (source / bundle.VAMANA_CODEBOOKS[0]).write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "Vamana test data is missing or empty"):
            self.package()

    def test_build_library_wins_over_conflicting_environment_and_legacy_rpath(self):
        built_library = self.build / "_deps/fixture-build/libfixture.so.1"
        built_library.parent.mkdir(parents=True)
        shutil.copy2(self.library, built_library)
        alternate = self.root / "alternate.c"
        alternate.write_text("int answer(void) { return 7; }\n")
        subprocess.run(["gcc", "-shared", "-fPIC", str(alternate), "-o", str(self.library),
                        "-Wl,-soname,libfixture.so.1"], check=True)
        executable = self.build / "gtests/SMOKE_TEST"
        subprocess.run(["gcc", str(self.source), str(self.library),
                        f"-Wl,--disable-new-dtags,-rpath,{self.prefix / 'lib'}", "-o", str(executable)],
                       check=True)
        self.assertEqual(subprocess.run([str(executable)]).returncode, 1)
        original_digest = bundle.digest(executable)
        self.package()
        output = self.args.output
        # Keep the conflicting original environment in place: RUNPATH relocation
        # must work even when the baked absolute path still exists locally.
        subprocess.run([str(output / "run"), "SMOKE_TEST"], cwd="/", check=True)
        self.assertEqual(bundle.digest(executable), original_digest)
        self.assertEqual((output / "lib/libfixture.so.1").stat().st_mtime_ns,
                         built_library.stat().st_mtime_ns)
        rpath = subprocess.run(["patchelf", "--print-rpath", str(output / "bin/SMOKE_TEST")],
                               text=True, capture_output=True, check=True).stdout.strip()
        self.assertEqual(rpath, "$ORIGIN/../lib")

    def test_current_graph_keeps_active_build_library_but_ignores_stale_logger(self):
        def shared(path, code, dependencies=()):
            path.parent.mkdir(parents=True, exist_ok=True)
            source = path.with_suffix('.c')
            source.write_text(code)
            subprocess.run(["gcc", "-shared", "-fPIC", str(source), *map(str, dependencies),
                            "-Wl,-soname," + path.name, "-o", str(path)], check=True)

        installed_logger = self.prefix / "lib/liblogger_fixture.so.1"
        shared(installed_logger, "int logger_value(void) { return 42; }\n")
        stale_logger = self.build / "_deps/logger-build/liblogger_fixture.so.1"
        shared(stale_logger, "int logger_value(void) { return 99; }\n")
        active_library = self.build / "_deps/fixture-build/libfixture.so.1"
        shared(active_library, "extern int logger_value(void); int answer(void) { return logger_value(); }\n",
               (installed_logger,))
        shared(self.library, "int answer(void) { return 7; }\n")
        executable = self.build / "gtests/SMOKE_TEST"
        subprocess.run(["gcc", str(self.source), str(active_library), str(installed_logger),
                        "-o", str(executable)], check=True)
        (self.build / "build.ninja").write_text("# Fixture graph queried by the read-only Ninja stub\n")
        ninja = self.prefix / "bin/ninja"
        ninja.parent.mkdir()
        targets = "_deps/fixture-build/libfixture.so.1: CXX_SHARED_LIBRARY_LINKER\n" + \
                  "libfixture.so.1: phony\ngtests/SMOKE_TEST: CXX_EXECUTABLE_LINKER"
        ninja.write_text("#!/usr/bin/env python3\nimport sys\n" +
                         f"assert sys.argv[1:] == {['-C', str(self.build), '-t', 'targets', 'all']!r}\n" +
                         f"print({targets!r})\n")
        ninja.chmod(0o755)
        self.package()
        output = self.args.output
        # The only combination producing 42 is active build library + installed
        # logger. Either stale logger or installed library would fail this run.
        subprocess.run([str(output / "run"), "SMOKE_TEST"], cwd="/", check=True)
        self.assertEqual((output / "lib/libfixture.so.1").stat().st_mtime_ns,
                         active_library.stat().st_mtime_ns)
        self.assertEqual((output / "lib/liblogger_fixture.so.1").stat().st_mtime_ns,
                         installed_logger.stat().st_mtime_ns)
        self.assertTrue(stale_logger.is_file())
        previous_manifest = (output / "manifest.json").read_bytes()
        active_library.unlink()
        with self.assertRaises(FileNotFoundError):
            self.package()
        self.assertEqual((output / "manifest.json").read_bytes(), previous_manifest)

    def test_ucx_modules_and_their_dependencies_are_deployed_with_driver_exclusion(self):
        def shared(path, source, dependencies=()):
            path.parent.mkdir(parents=True, exist_ok=True)
            cfile = self.root / (path.name + ".c")
            cfile.write_text(source)
            subprocess.run(["gcc", "-shared", "-fPIC", str(cfile), *map(str, dependencies),
                            "-o", str(path), "-Wl,-soname," + path.name], check=True)

        core = self.prefix / "lib/libucp.so.0"
        shared(core, "int answer(void) { return 42; }\n")
        dependency = self.prefix / "lib/libmoduledep.so.1"
        shared(dependency, "int module_dependency(void) { return 17; }\n")
        driver_stub = self.prefix / "lib/stubs/libnvidia-ml.so.1"
        shared(driver_stub, "int nvmlInit_v2(void) { return 0; }\n")
        module = self.prefix / "lib/ucx/libuct_cuda.so.0"
        shared(module, "extern int module_dependency(void); extern int nvmlInit_v2(void); "
               "int module_probe(void) { return module_dependency() + nvmlInit_v2(); }\n",
               (dependency, driver_stub))
        (module.parent / "libuct_cuda.so").symlink_to(module.name)
        subprocess.run(["gcc", str(self.source), str(core), "-o", str(self.build / "gtests/SMOKE_TEST")],
                       check=True)
        self.package()
        output = self.args.output
        self.assertTrue((output / "lib/ucx/libuct_cuda.so.0").is_file())
        self.assertTrue((output / "lib/libmoduledep.so.1").is_file())
        self.assertFalse((output / "lib/libnvidia-ml.so.1").exists())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertIn("libnvidia-ml.so.1", manifest["driver_libraries"])
        self.assertIn("lib/ucx/libuct_cuda.so.0", manifest["ucx_modules"])
        environment = subprocess.run([str(output / "run"), "/usr/bin/env"], cwd="/",
                                     text=True, capture_output=True, check=True).stdout
        self.assertIn(f"UCX_MODULE_DIR={output}/lib/ucx\n", environment)


if __name__ == "__main__":
    unittest.main()
