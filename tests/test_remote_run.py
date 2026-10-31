"""Exercise the deployed shell runner with a fake ANN binary, without SSH/GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RemoteBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="cuvs runner test ")
        self.addCleanup(directory.cleanup)
        self.bundle = Path(directory.name)
        (self.bundle / "bin").mkdir()
        (self.bundle / "scripts").mkdir()
        shutil.copy2(ROOT / "scripts/make_smoke_dataset.py", self.bundle / "scripts")
        launcher = self.bundle / "run"
        launcher.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'bundle=$(cd -- "$(dirname -- "$0")" && pwd)\n'
            'executable=$1; shift\nexec "$bundle/$executable" "$@"\n'
        )
        launcher.chmod(0o755)
        binary = self.bundle / "bin/CUVS_BRUTE_FORCE_ANN_BENCH"
        binary.write_text('''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
phase = "build" if "--build" in args else "search"
config = json.loads(Path(args[-1]).read_text())
assert config["index"][0]["algo"] == "cuvs_brute_force"
index_root = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--index_prefix=")))
index = index_root / config["index"][0]["file"]
row = {"name": "smoke", "iterations": 1}
if phase == "build":
    # Simulate the two callback invocations used by benchmark calibration.
    # The real brute-force index intentionally serializes to an empty marker.
    for iteration in range(2):
        if index.exists() and "--force" not in args:
            row["error_occurred"] = True
            row["error_message"] = "Index file already exists (use --force)"
            break
        index.touch()
else:
    assert index.exists()
    assert "--force" not in args
    row["Recall"] = 1.0
if os.environ.get("FAKE_BENCH_ERROR"):
    row["error_occurred"] = True
    row["error_message"] = "simulated GPU failure"
output = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--benchmark_out=")))
output.write_text(json.dumps({"benchmarks": [row]}))
''')
        binary.chmod(0o755)

    def execute(self, extra_env=None):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/remote-run.sh"), str(self.bundle), "benchmark"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, **(extra_env or {})}, timeout=30,
        )
        results = list((self.bundle / "results").iterdir())
        self.assertEqual(len(results), 1)
        return result, results[0]

    def test_build_calibration_overwrites_empty_index_then_search_reports_recall(self):
        process, results = self.execute()
        self.assertEqual(process.returncode, 0, process.stdout)
        self.assertEqual((results / "exit-status").read_text().strip(), "0")
        self.assertTrue((results / "build.json").is_file())
        search = json.loads((results / "search.json").read_text())
        self.assertEqual(search["benchmarks"][0]["Recall"], 1.0)
        self.assertEqual((results / "index/cuvs_brute_force.idx").stat().st_size, 0)

    def test_successful_exit_with_benchmark_error_stops_before_search(self):
        process, results = self.execute({"FAKE_BENCH_ERROR": "1"})
        self.assertNotEqual(process.returncode, 0)
        self.assertNotEqual((results / "exit-status").read_text().strip(), "0")
        self.assertIn("simulated GPU failure", process.stdout)
        self.assertFalse((results / "search.json").exists())


if __name__ == "__main__":
    unittest.main()
