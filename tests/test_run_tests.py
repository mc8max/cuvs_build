"""Exercise the deployed native suite runner without requiring CUDA or SSH."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('run_tests', ROOT / 'scripts/run_tests.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class NativeRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='native suite runner ')
        self.addCleanup(temporary.cleanup)
        self.bundle = Path(temporary.name)
        (self.bundle / 'bin').mkdir()
        (self.bundle / 'tests/data').mkdir(parents=True)
        launcher = self.bundle / 'run'
        launcher.write_text('#!/bin/bash\nset -eu\nroot=$(cd -- "$(dirname -- "$0")" && pwd)\n'
                            'exe=$1; shift\nexec "$root/$exe" "$@"\n')
        launcher.chmod(0o755)

    def binary(self, name, xml=None, code=0, extra=''):
        executable = self.bundle / 'bin' / name
        executable.write_text('''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import time
assert os.environ['CI'] == 'true'
data_dir = Path(os.environ['RAPIDS_DATASET_ROOT_DIR'])
assert data_dir == Path(__file__).resolve().parents[1] / 'tests/data'
assert data_dir.is_dir()
assert Path(os.environ['TMPDIR']).is_dir()
''' + extra + '\n' +
            ("output = next(arg.split('xml:', 1)[1] for arg in sys.argv if arg.startswith('--gtest_output='))\n"
             f'Path(output).write_text({xml!r})\n' if xml is not None else '') +
            f'sys.exit({code})\n')
        executable.chmod(0o755)

    def execute(self, tests, excluded=None, timeout=10, **plan_options):
        (self.bundle / 'test-plan.json').write_text(json.dumps(
            {'tests': tests, 'excluded': excluded or [], **plan_options}))
        with contextlib.redirect_stdout(io.StringIO()):
            code, results = runner.run_plan(self.bundle, timeout)
        self.assertEqual((results / 'exit-status').read_text().strip(), str(code))
        return code, json.loads((results / 'summary.json').read_text()), results

    def test_counts_pass_fail_skip_disabled_and_continues_after_failure(self):
        xml = '''<testsuites tests="4" failures="1" disabled="1" errors="0"><testsuite name="Suite">
<testcase classname="Suite" name="Pass" status="run" result="completed"/>
<testcase classname="Suite" name="Fail" status="run"><failure message="mismatch"/></testcase>
<testcase classname="Suite" name="Skip" status="run" result="skipped"><skipped message="GPU unavailable"/></testcase>
<testcase classname="Suite" name="DISABLED_Case" status="notrun" result="suppressed"/>
</testsuite></testsuites>'''
        self.binary('FIRST_TEST', xml, code=1)
        self.binary('cuvs_c_test', extra="Path('ran-second').write_text('yes')")
        code, summary, results = self.execute(
            [{'name': 'FIRST_TEST'}, {'name': 'cuvs_c_test', 'kind': 'command'}],
            excluded=[{'name': 'MULTIGPU', 'reason': 'Only one visible GPU'}])
        self.assertEqual(code, 1)
        self.assertEqual(summary['cases'], {'passed': 1, 'failed': 1, 'skipped': 1, 'disabled': 1})
        self.assertEqual(summary['command_checks'], {'passed': 1, 'failed': 0})
        self.assertEqual(summary['completed_executables'], 2)
        self.assertTrue(summary['completed'])
        self.assertTrue(summary['case_counts_complete'])
        self.assertEqual(summary['excluded_targets'][0]['name'], 'MULTIGPU')
        self.assertTrue((results / 'cuvs_c_test/work/ran-second').is_file())
        self.assertEqual(summary['results'][0]['failed_cases'][0]['name'], 'Suite.Fail')
        self.assertEqual(summary['results'][0]['skipped_cases'][0]['reason'], 'GPU unavailable')

    def test_all_runtime_skipped_cases_are_not_a_failure(self):
        self.binary('SKIPPED_TEST', '<testsuites tests="1" failures="0"><testsuite><testcase name="Skip" result="skipped"><skipped>unsupported GPU feature</skipped></testcase></testsuite></testsuites>')
        code, summary, _ = self.execute([{'name': 'SKIPPED_TEST'}])
        self.assertEqual(code, 0)
        self.assertEqual(summary['cases']['skipped'], 1)
        self.assertEqual(summary['executables']['skipped'], 1)

    def test_zero_or_malformed_xml_is_error_despite_successful_exit(self):
        self.binary('EMPTY_TEST', '<testsuites tests="0"/>')
        self.binary('BAD_TEST', '<not-xml')
        self.binary('MISSING_TEST')
        self.binary('NEXT_TEST', '<testsuites tests="1"><testcase name="Fine"/></testsuites>')
        code, summary, _ = self.execute([{'name': x} for x in ('EMPTY_TEST', 'BAD_TEST', 'MISSING_TEST', 'NEXT_TEST')])
        self.assertEqual(code, 1)
        self.assertEqual(summary['executables']['error'], 3)
        self.assertEqual(summary['executables']['passed'], 1)
        self.assertFalse(summary['case_counts_complete'])
        self.assertTrue(summary['completed'])

    def test_timeout_is_reported_and_later_executable_still_runs(self):
        self.binary('SLOW_TEST', extra='time.sleep(30)')
        self.binary('NEXT_TEST', '<testsuites tests="1"><testcase name="Fine"/></testsuites>')
        code, summary, _ = self.execute([{'name': 'SLOW_TEST', 'timeout_seconds': 0.1}, {'name': 'NEXT_TEST'}])
        self.assertEqual(code, 1)
        self.assertEqual(summary['executables']['timeout'], 1)
        self.assertEqual(summary['executables']['passed'], 1)
        self.assertIsNotNone(summary['results'][0]['returncode'])
        self.assertIn('exceeded', summary['results'][0]['errors'][0])

    def test_failures_in_xml_are_failure_even_when_process_returns_zero(self):
        self.binary('BAD_TEST', '<testsuites tests="1" failures="1"><testcase name="Bad"><failure message="wrong"/></testcase></testsuites>')
        code, summary, _ = self.execute([{'name': 'BAD_TEST'}])
        self.assertEqual(code, 1)
        self.assertEqual(summary['cases']['failed'], 1)
        self.assertEqual(summary['executables']['failed'], 1)

    def test_framework_failure_does_not_invent_a_failed_test_case(self):
        xml = '''<testsuites tests="1" failures="0" errors="0">
<testsuite name="Suite" tests="1"><testcase classname="Suite" name="Skipped" result="skipped"><skipped message="Environment setup failed"/></testcase></testsuite>
<testsuite name="NonTestSuiteFailure" tests="1" failures="1"><testcase classname="" name="" status="run" result="completed"><failure message="Global setup failed"/></testcase></testsuite>
</testsuites>'''
        self.binary('SETUP_TEST', xml, code=1)
        code, summary, _ = self.execute([{'name': 'SETUP_TEST'}])
        self.assertEqual(code, 1)
        self.assertEqual(summary['cases'], {'passed': 0, 'failed': 0, 'skipped': 1, 'disabled': 0})
        self.assertEqual(summary['executables']['error'], 1)
        self.assertEqual(summary['results'][0]['framework_errors'][0]['suite'], 'NonTestSuiteFailure')
        self.assertTrue(summary['case_counts_complete'])

    def test_duplicate_or_unsafe_targets_are_rejected(self):
        for tests in ([{'name': '../escape'}], [{'name': 'SAME'}, {'name': 'SAME'}],
                      [{'name': 'EXE', 'command': ['/bin/true']}],
                      [{'name': 'EXE', 'command': ['../outside']}],
                      [{'name': 'EXE', 'kind': 'unknown'}]):
            with self.subTest(tests=tests):
                with self.assertRaises(ValueError):
                    runner.validate_plan({'tests': tests})

    def test_cpu_threads_override_default_pool_and_preserve_explicit_arguments(self):
        thread_limit = os.environ.get('OMP_THREAD_LIMIT')
        self.binary('cpu_test', extra=("assert os.environ['OMP_NUM_THREADS'] == '4'\n"
                                     f"assert os.environ.get('OMP_THREAD_LIMIT') == {thread_limit!r}\n"
                                     "assert sys.argv[1:] == ['--threads=2']"))
        with mock.patch.dict(os.environ, {'OMP_NUM_THREADS': '128'}):
            code, summary, results = self.execute(
                [{'name': 'cpu_test', 'kind': 'command', 'command': ['bin/cpu_test', '--threads=2']}],
                cpu_threads=4)
        self.assertEqual(code, 0)
        self.assertEqual(summary['cpu_threads'], 4)
        self.assertEqual(summary['results'][0]['omp_num_threads'], '4')
        self.assertEqual(json.loads((results / 'cpu_test/summary.json').read_text())['omp_num_threads'], '4')

    def test_optional_cpu_threads_preserves_and_reports_inherited_environment(self):
        self.binary('cpu_test', extra="assert os.environ['OMP_NUM_THREADS'] == '7'")
        with mock.patch.dict(os.environ, {'OMP_NUM_THREADS': '7'}):
            code, summary, _ = self.execute([{'name': 'cpu_test', 'kind': 'command'}])
        self.assertEqual(code, 0)
        self.assertIsNone(summary['cpu_threads'])
        self.assertEqual(summary['results'][0]['omp_num_threads'], '7')

    def test_cpu_threads_requires_a_positive_integer(self):
        for value in (0, -1, True, False, 4.0, '4', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    runner.validate_plan({'tests': [{'name': 'TEST'}], 'cpu_threads': value})


if __name__ == '__main__':
    unittest.main()
