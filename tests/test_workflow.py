"""Offline checks for target compatibility and reusable build configuration."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow = load_script("workflow")
smoke = load_script("make_smoke_dataset")


def build_config():
    # Keep compatibility assertions stable when the user edits the active profile.
    return {
        "inventory": "inventory.json",
        "build_type": "Release",
        "toolchain": {"glibc_baseline": "2.28", "minimum_driver": "575.57.08"},
    }


def measured_inventory():
    # Deliberately no remote compiler/toolkit: the deployment carries its runtime.
    return {
        "schema_version": 1,
        "system": "Linux",
        "machine": "x86_64",
        "collected_at": "2026-10-31T00:00:00+00:00",
        "glibc": {"returncode": 0, "output": "glibc 2.35"},
        "gpus": [{
            "name": "Test GPU",
            "compute_capability": "8.6",
            "driver_version": "575.57.08",
            "memory_mib": "24576",
        }],
        "tools": {name: {"path": "/usr/bin/" + name}
                  for name in ("python3", "rsync", "tar", "ldd")},
    }


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.host = mock.patch.object(workflow.platform, "machine", return_value="x86_64")
        self.host.start()
        self.addCleanup(self.host.stop)
        self.config = build_config()
        self.inventory = measured_inventory()

    def validate(self):
        return workflow.validate_inventory(self.inventory, self.config)

    def test_compatible_runtime_without_remote_compiler(self):
        self.assertEqual(self.validate(), ["86-real"])

    def test_gpu_architectures_are_unique_and_numerically_sorted(self):
        template = self.inventory["gpus"][0]
        self.inventory["gpus"] = [
            dict(template, compute_capability=capability)
            for capability in ("12.0", "8.6", "9.0", "8.6")
        ]
        self.assertEqual(self.validate(), ["86-real", "90-real", "120-real"])

    def test_driver_floor_applies_to_every_gpu(self):
        self.inventory["gpus"].append(dict(self.inventory["gpus"][0], driver_version="570.99.00"))
        with self.assertRaisesRegex(ValueError, "driver"):
            self.validate()

    def test_target_glibc_must_meet_sysroot_baseline(self):
        self.inventory["glibc"]["output"] = "glibc 2.27"
        with self.assertRaisesRegex(ValueError, "glibc"):
            self.validate()
        self.inventory["glibc"]["output"] = "glibc 2.28"
        self.assertEqual(self.validate(), ["86-real"])

    def test_non_linux_or_wrong_inventory_schema_is_rejected(self):
        for key, value in (("system", "Darwin"), ("schema_version", 2)):
            with self.subTest(key=key):
                inventory = copy.deepcopy(self.inventory)
                inventory[key] = value
                with self.assertRaisesRegex(ValueError, "inventory"):
                    workflow.validate_inventory(inventory, self.config)

    def test_remote_and_local_cpu_architectures_must_match_supported_toolchain(self):
        self.inventory["machine"] = "aarch64"
        with self.assertRaisesRegex(ValueError, "x86_64"):
            self.validate()
        self.inventory["machine"] = "x86_64"
        with mock.patch.object(workflow.platform, "machine", return_value="aarch64"):
            with self.assertRaisesRegex(ValueError, "x86_64"):
                self.validate()

    def test_absent_or_unknown_gpu_blocks_configuration(self):
        for gpus in ([], [dict(self.inventory["gpus"][0], compute_capability="[N/A]")]):
            with self.subTest(gpus=gpus):
                self.inventory["gpus"] = gpus
                with self.assertRaisesRegex(ValueError, "GPU"):
                    self.validate()

    def test_required_remote_runtime_commands_are_checked(self):
        for name in ("python3", "rsync", "tar", "ldd"):
            with self.subTest(command=name):
                inventory = copy.deepcopy(self.inventory)
                inventory["tools"][name]["path"] = None
                with self.assertRaisesRegex(ValueError, name):
                    workflow.validate_inventory(inventory, self.config)


class ConfigurationFingerprintTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cuvs-workflow-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root_patch = mock.patch.object(workflow, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        host = mock.patch.object(workflow.platform, "machine", return_value="x86_64")
        host.start()
        self.addCleanup(host.stop)
        config = build_config()
        config["inventory"] = "inventory.json"
        config_path = self.root / "build.json"
        config_path.write_text(json.dumps(config))
        self.inventory = measured_inventory()
        self.write_inventory()
        self.flow = workflow.Workflow(config_path)

    def write_inventory(self):
        (self.root / "inventory.json").write_text(json.dumps(self.inventory))

    def record_successful_configuration(self):
        (self.flow.state / "configured.json").write_text(json.dumps({
            "fingerprint": self.flow.fingerprint(), "architectures": ["86-real"],
        }))

    def test_missing_configuration_stamp_blocks_build(self):
        with self.assertRaisesRegex(ValueError, "make configure"):
            self.flow.require_configured()

    def test_build_configuration_change_invalidates_existing_stamp(self):
        self.record_successful_configuration()
        self.flow.require_configured()
        self.flow.config["build_type"] = "Debug"
        with self.assertRaisesRegex(ValueError, "make configure"):
            self.flow.require_configured()

    def test_target_gpu_change_invalidates_existing_stamp(self):
        self.record_successful_configuration()
        self.inventory["gpus"][0]["compute_capability"] = "9.0"
        self.write_inventory()
        with self.assertRaisesRegex(ValueError, "make configure"):
            self.flow.require_configured()

    def test_inventory_refresh_without_build_relevant_change_reuses_stamp(self):
        self.record_successful_configuration()
        previous = self.flow.fingerprint()
        self.inventory["collected_at"] = "2026-11-01T00:00:00+00:00"
        self.inventory["gpus"][0]["memory_mib"] = "24575"
        self.write_inventory()
        self.assertEqual(self.flow.fingerprint(), previous)
        self.flow.require_configured()

    def test_remote_incompatibility_blocks_even_an_existing_stamp(self):
        self.record_successful_configuration()
        self.inventory["glibc"]["output"] = "glibc 2.17"
        self.write_inventory()
        with self.assertRaisesRegex(ValueError, "glibc"):
            self.flow.require_configured()


class NativeBuildHeaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cuvs-native-build-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(mock.patch.object(workflow, "ROOT", self.root))
        self.headers = self.root / "source/c/include/cuvs/core"
        self.headers.mkdir(parents=True)
        (self.headers / "a.h").write_text("#pragma once\n")
        (self.headers / "all.h").write_text("#include <cuvs/core/a.h>\n")
        config_path = self.root / "build.json"
        config_path.write_text(json.dumps({
            "source_dir": "source", "build_dir": "build", "env_prefix": "env",
            "jobs": 1, "targets": [], "tests": {"profile": "all-native"},
        }))
        self.flow = workflow.Workflow(config_path)
        self.plan = {
            "build_targets": ["cuvs_c_neighbors_test"],
            "build_only": [{"name": "cuvs_c_neighbors_test", "reason": "Compile-only check"}],
        }
        self.enterContext(mock.patch.object(self.flow, "require_configured"))
        self.enterContext(mock.patch.object(self.flow, "build_environment", return_value={}))
        self.enterContext(mock.patch.object(self.flow, "native_plan", return_value=self.plan))
        self.enterContext(mock.patch.object(workflow, "run", side_effect=self.run_tool))

    def run_tool(self, args, **kwargs):
        tool = Path(args[0]).name
        if tool == "cmake" and "--build" in args:
            return subprocess.CompletedProcess(args, 0)
        if tool == "ctest" and "--show-only=json-v1" in args:
            configured = {"tests": [{"name": "cuvs_c_neighbors_test"}]}
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(configured))
        raise AssertionError(f"Unexpected external command: {args}")

    def test_build_records_header_and_compile_checks_without_generated_cmake_script(self):
        self.flow.build()

        result = json.loads((self.flow.state / "test-plan.json").read_text())
        self.assertEqual(result["local_check_results"], [{
            "name": "cuvs_c_verify_install_headers", "status": "passed", "headers_checked": 1,
        }])
        self.assertEqual(result["build_only"], [{
            "name": "cuvs_c_neighbors_test", "reason": "Compile-only check", "status": "passed",
        }])

    def test_quick_profile_uses_native_build_and_check_reporting(self):
        self.flow.config['tests']['profile'] = 'quick-native'
        self.flow.build()
        result = json.loads((self.flow.state / 'test-plan.json').read_text())
        self.assertEqual(result['local_check_results'][0]['status'], 'passed')
        self.assertEqual(result['build_only'][0]['status'], 'passed')

    def test_quick_profile_remote_dispatch_preserves_the_native_runner(self):
        self.flow.config['tests'].update(profile='quick-native', timeout_seconds=120)
        with mock.patch.object(self.flow, 'release', return_value='release'), \
                mock.patch.object(self.flow, 'remote_root', return_value='/remote'), \
                mock.patch.object(self.flow, 'remote') as remote:
            self.flow.execute_remote('test')
        self.assertEqual(remote.call_args.args[0], [
            'python3', '/remote/releases/release/scripts/run_tests.py',
            '/remote/releases/release', '--timeout', '120',
        ])

    def test_build_fails_when_aggregate_header_omits_a_source_header(self):
        (self.headers / "missing.h").write_text("#pragma once\n")

        with self.assertRaisesRegex(ValueError, "cuvs/core/missing.h"):
            self.flow.build()

        self.assertFalse((self.flow.state / "test-plan.json").exists())


class ResultFetchTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('rsync'), 'rsync is required for the transfer filter check')
    def test_fetch_keeps_reports_and_benchmarks_but_excludes_only_native_scratch(self):
        with tempfile.TemporaryDirectory(prefix='cuvs-fetch-test-') as temporary:
            root = Path(temporary)
            source = root / 'remote-results'
            retained = (
                '20261031T000000Z-tests.abc/summary.json',
                '20261031T000000Z-tests.abc/test-plan.json',
                '20261031T000000Z-tests.abc/ANN_TEST/run.log',
                '20261031T000000Z-tests.abc/ANN_TEST/tests.xml',
                '20261031T000000Z-tests.abc/ANN_TEST/summary.json',
                '20261031T000000Z-tests.abc/ANN_TEST/exit-status',
                '20261031-benchmark/tmp/keep.bin',
                '20261031-benchmark/work/keep.bin',
                'other/20261031T000000Z-tests.nested/ANN_TEST/tmp/keep.bin',
            )
            excluded = (
                '20261031T000000Z-tests.abc/ANN_TEST/tmp/nested/dataset.bin',
                '20261031T000000Z-tests.abc/ANN_TEST/work/index.bin',
                '20261031T000000Z-tests.abc/cuvs_c_test/tmp/index.bin',
            )
            for name in retained + excluded:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            flow = workflow.Workflow.__new__(workflow.Workflow)
            flow.config = {'ssh': {'host': 'unused-host'}}
            with mock.patch.object(workflow, 'ROOT', root), \
                    mock.patch.object(flow, 'release', return_value='release'), \
                    mock.patch.object(flow, 'remote_root', return_value='/remote'), \
                    mock.patch.object(flow, 'ssh', return_value=['ssh']), \
                    mock.patch.object(workflow, 'run') as captured:
                flow.fetch()
            command = captured.call_args.args[0]
            # Exercise the real generated filters against local fixtures, without SSH.
            command[-2] = str(source) + '/'
            subprocess.run(command, check=True, capture_output=True, text=True)
            destination = root / 'results/release'
            for name in retained:
                self.assertEqual((destination / name).read_text(), name)
            for name in excluded:
                self.assertFalse((destination / name).exists())
                self.assertEqual((source / name).read_text(), name)


class SmokeDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="cuvs-dataset-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            smoke.generate(cls.directory, "cuvs_brute_force", 20261031)

    def matrix(self, name, kind):
        data = (self.directory / name).read_bytes()
        rows, columns = struct.unpack_from("<II", data)
        self.assertEqual(len(data), 8 + rows * columns * 4)
        values = struct.unpack_from("<" + kind * (rows * columns), data, 8)
        return [values[row * columns:(row + 1) * columns] for row in range(rows)]

    def test_ground_truth_matches_serialized_float32_vectors(self):
        base = self.matrix("base.fbin", "f")
        queries = self.matrix("query.fbin", "f")
        neighbors = self.matrix("groundtruth.ibin", "i")
        self.assertEqual((len(base), len(base[0])), (2048, 32))
        self.assertEqual((len(queries), len(queries[0])), (128, 32))
        self.assertEqual((len(neighbors), len(neighbors[0])), (128, 10))
        # Independently rank the stored floats, not the generator's integer data.
        for query, actual in zip(queries, neighbors):
            distances = [sum((left - right) ** 2 for left, right in zip(query, row))
                         for row in base]
            expected = sorted(range(len(base)), key=lambda index: (distances[index], index))[:10]
            self.assertEqual(list(actual), expected)

    def test_same_seed_produces_identical_bundle_files(self):
        with tempfile.TemporaryDirectory(prefix="cuvs-dataset-repeat-") as temporary:
            repeated = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                smoke.generate(repeated, "cuvs_brute_force", 20261031)
            for original in self.directory.iterdir():
                self.assertEqual(original.read_bytes(), (repeated / original.name).read_bytes())


if __name__ == "__main__":
    unittest.main()
