"""Local environment configuration must resolve safely and survive packaging."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "workflow_env_config", Path(__file__).resolve().parents[1] / "scripts/workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class EnvironmentConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cuvs-env-config-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(mock.patch.object(workflow, "ROOT", self.root))
        self.enterContext(mock.patch.dict(os.environ, {}, clear=True))
        self.dotenv = self.root / ".env"
        self.config_path = self.root / "config/build.json"
        self.config_path.parent.mkdir()
        self.config = {
            "source_dir": "${CUVS_SOURCE_DIR}",
            "build_dir": "${CUVS_SOURCE_DIR}/cpp/build-runpod",
            "env_prefix": "${CUVS_SOURCE_DIR}/.conda-env",
            "cpm_cache": "${CUVS_SOURCE_DIR}/.cache/cpm",
            "inventory": "config/runpod.json",
            "targets": [],
        }

    def flow(self):
        self.config_path.write_text(json.dumps(self.config))
        return workflow.Workflow(self.config_path)

    def test_plain_configuration_does_not_require_dotenv(self):
        self.config = {"source_dir": "../existing-cuvs"}
        flow = self.flow()
        self.assertEqual(flow.path("source_dir"), (self.root / "../existing-cuvs").resolve())

    def test_local_root_resolves_all_paths_and_nested_values(self):
        self.config["extra"] = {"paths": ["${CUVS_SOURCE_DIR}/data", 4, True]}
        for source in ("../local cuvs", str(self.root / "absolute checkout")):
            with self.subTest(source=source):
                self.dotenv.write_text(f'CUVS_SOURCE_DIR="{source}"\n')
                flow = self.flow()
                expected = (self.root / source).resolve()
                self.assertEqual(flow.path("source_dir"), expected)
                self.assertEqual(flow.path("build_dir"), expected / "cpp/build-runpod")
                self.assertEqual(flow.path("env_prefix"), expected / ".conda-env")
                self.assertEqual(flow.path("cpm_cache"), expected / ".cache/cpm")
                self.assertEqual(flow.config["extra"], {"paths": [source + "/data", 4, True]})

    def test_process_environment_wins_without_mutating_global_environment(self):
        self.dotenv.write_text("CUVS_SOURCE_DIR=from-dotenv\nDOTENV_ONLY=private\n")
        os.environ["CUVS_SOURCE_DIR"] = "from-process"
        before = dict(os.environ)
        flow = self.flow()
        self.assertEqual(flow.config["source_dir"], "from-process")
        self.assertEqual(dict(os.environ), before)
        self.assertNotIn("DOTENV_ONLY", flow.config)

    def test_missing_or_empty_referenced_variable_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUVS_SOURCE_DIR"):
            self.flow()
        for contents in ("# no source configured\n", "CUVS_SOURCE_DIR=\n", 'CUVS_SOURCE_DIR=""\n'):
            with self.subTest(contents=contents):
                self.dotenv.write_text(contents)
                with self.assertRaisesRegex(ValueError, "CUVS_SOURCE_DIR"):
                    self.flow()
        self.dotenv.write_text("CUVS_SOURCE_DIR=valid-dotenv-value\n")
        os.environ["CUVS_SOURCE_DIR"] = ""
        with self.assertRaisesRegex(ValueError, "CUVS_SOURCE_DIR"):
            self.flow()

    def test_export_quotes_comments_and_shell_text_are_literal(self):
        self.dotenv.write_text('# comment\nexport CUVS_SOURCE_DIR="checkout with spaces # inside" # comment\n')
        self.assertEqual(self.flow().config["source_dir"], "checkout with spaces # inside")
        marker = self.root / "shell-executed"
        literal = f"$(touch {marker})/${{UNREFERENCED}}/$HOME/`touch {marker}`"
        self.dotenv.write_text(f"CUVS_SOURCE_DIR='{literal}'\n")
        self.assertEqual(self.flow().config["source_dir"], literal)
        self.assertFalse(marker.exists())

    def test_malformed_dotenv_is_rejected(self):
        for contents in ("CUVS_SOURCE_DIR missing-equals\n", "CUVS_SOURCE_DIR='unclosed\n",
                         "CUVS_SOURCE_DIR=unquoted spaces\n", "2INVALID=value\n"):
            with self.subTest(contents=contents):
                self.dotenv.write_text(contents)
                with self.assertRaises(ValueError):
                    self.flow()

    def test_changed_dotenv_path_invalidates_configuration_stamp(self):
        self.dotenv.write_text("CUVS_SOURCE_DIR=first-checkout\n")
        first = self.flow()
        with mock.patch.object(first, "target", return_value=({}, ["86-real"])):
            fingerprint = first.fingerprint()
            (first.state / "configured.json").write_text(json.dumps({"fingerprint": fingerprint}))
            first.require_configured()
        self.dotenv.write_text("CUVS_SOURCE_DIR=second-checkout\n")
        second = self.flow()
        with mock.patch.object(second, "target", return_value=({}, ["86-real"])):
            self.assertNotEqual(second.fingerprint(), fingerprint)
            with self.assertRaisesRegex(ValueError, "make configure"):
                second.require_configured()

    def test_package_uses_resolved_snapshot_and_publish_rejects_changed_dotenv(self):
        self.dotenv.write_text("CUVS_SOURCE_DIR=first-checkout\nUNUSED_SECRET=stay-local\n")
        flow = self.flow()

        def package_fixture(args, **kwargs):
            snapshot = Path(args[args.index("--config") + 1])
            self.assertEqual(snapshot, flow.state / "resolved-build-config.json")
            self.assertEqual(json.loads(snapshot.read_text()), flow.config)
            self.assertNotIn("${CUVS_SOURCE_DIR}", snapshot.read_text())
            self.assertNotIn("stay-local", snapshot.read_text())
            output = Path(args[args.index("--output") + 1])
            output.mkdir(parents=True)
            shutil.copyfile(snapshot, output / "build-config.json")
            return subprocess.CompletedProcess(args, 0)

        with mock.patch.object(flow, "require_configured"), \
                mock.patch.object(flow, "build_environment", return_value={}), \
                mock.patch.object(workflow, "run", side_effect=package_fixture):
            flow.package()
        self.assertIn("${CUVS_SOURCE_DIR}", self.config_path.read_text())
        self.dotenv.write_text("CUVS_SOURCE_DIR=second-checkout\n")
        updated = self.flow()
        with mock.patch.object(updated, "inventory") as inventory, \
                mock.patch.object(updated, "remote") as remote:
            with self.assertRaisesRegex(ValueError, "configuration changed since packaging"):
                updated.publish()
            inventory.assert_not_called()
            remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
