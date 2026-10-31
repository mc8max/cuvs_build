#!/usr/bin/env python3
"""Run every selected native test on one GPU and preserve complete result counts."""

import argparse
import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def validate_plan(plan):
    if not isinstance(plan.get('tests'), list) or not plan['tests']:
        raise ValueError('test-plan.json must contain a nonempty tests list')
    if 'cpu_threads' in plan and (type(plan['cpu_threads']) is not int or plan['cpu_threads'] <= 0):
        raise ValueError('cpu_threads must be a positive integer')
    names = set()
    for test in plan['tests']:
        name = test.get('name', '')
        if not re.fullmatch(r'[A-Za-z0-9_+-]+', name) or name in names:
            raise ValueError(f'Invalid or duplicate test target: {name!r}')
        names.add(name)
        if test.get('kind', 'gtest') not in ('gtest', 'command'):
            raise ValueError(f'Unsupported test kind for {name}')
        if not isinstance(test.get('gtest_filter', '*'), str):
            raise ValueError(f'Invalid GoogleTest filter for {name}')
        if 'timeout_seconds' in test and float(test['timeout_seconds']) <= 0:
            raise ValueError(f'Invalid timeout for {name}')
        if 'command' in test:
            command = test['command']
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                raise ValueError(f'Invalid command argv for {name}')
            executable = PurePosixPath(command[0])
            if executable.is_absolute() or '..' in executable.parts:
                raise ValueError(f'Command executable must be relative to bundle: {name}')
    excluded = plan.get('excluded', [])
    if not isinstance(excluded, list) or any(not item.get('name') or not item.get('reason') for item in excluded):
        raise ValueError('Every excluded target must have a name and reason')


def empty_cases():
    return {'passed': 0, 'failed': 0, 'skipped': 0, 'disabled': 0}


def parse_gtest_report(path):
    """Count case elements, keeping runtime skips distinct from disabled tests."""
    root = ET.parse(path).getroot()
    if root.tag not in ('testsuites', 'testsuite'):
        raise ValueError(f'Unexpected GoogleTest XML root: {root.tag}')
    counts = empty_cases()
    failures = []
    skips = []
    disabled = []
    framework_errors = []
    cases = list(root.iter('testcase'))
    parents = {id(child): parent for parent in root.iter() for child in parent}
    if not cases:
        raise ValueError('GoogleTest report contains no test cases (empty filter or missing registration)')
    for case in cases:
        name = '.'.join(x for x in (case.get('classname'), case.get('name')) if x)
        problems = list(case.findall('failure')) + list(case.findall('error'))
        skipped = case.find('skipped')
        if not case.get('name') and not case.get('classname'):
            # GoogleTest writes an unnamed synthetic testcase for an assertion
            # outside a test body (suite/environment setup or teardown). It is
            # an executable-level error, not an additional registered case.
            framework_errors.append({
                'suite': parents[id(case)].get('name', ''),
                'messages': [x.get('message') or x.text or '' for x in problems]
                            or ['Unnamed GoogleTest framework result'],
            })
            continue
        if problems:
            counts['failed'] += 1
            failures.append({'name': name, 'messages': [x.get('message') or x.text or '' for x in problems]})
        elif skipped is not None or case.get('result') == 'skipped':
            counts['skipped'] += 1
            skips.append({'name': name, 'reason': (skipped.get('message') or skipped.text or '')
                          if skipped is not None else 'GoogleTest reported skipped'})
        elif case.get('status') == 'notrun' or case.get('result') == 'suppressed':
            counts['disabled'] += 1
            disabled.append(name)
        else:
            counts['passed'] += 1
    # Ad-hoc environment/suite failures can be reported in addition to normal
    # registered cases; retain them rather than requiring exact total equality.
    if int(root.get('tests', len(cases))) > len(cases):
        raise ValueError('GoogleTest XML case count is smaller than its declared total')
    if int(root.get('failures', '0')) > counts['failed'] + len(framework_errors) or int(root.get('errors', '0')):
        raise ValueError('GoogleTest XML reports additional failures/errors without case details')
    return {'cases': counts, 'failed_cases': failures, 'skipped_cases': skips,
            'disabled_cases': disabled, 'framework_errors': framework_errors}


def terminate_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_target(bundle, test, results, timeout, cpu_threads=None):
    name = test['name']
    kind = test.get('kind', 'gtest')
    directory = results / name
    directory.mkdir()
    work = directory / 'work'
    work.mkdir()
    temporary = directory / 'tmp'
    temporary.mkdir()
    xml = directory / 'tests.xml'
    command = [str(bundle / 'run')] + test.get('command', [f'bin/{name}'])
    if kind == 'gtest':
        command += [f"--gtest_filter={test.get('gtest_filter', '*')}", f'--gtest_output=xml:{xml}']
    env = dict(os.environ)
    env.update(LC_ALL='C', CI='true', RAPIDS_DATASET_ROOT_DIR=str(bundle / 'tests/data') + '/',
               TMPDIR=str(temporary))
    if cpu_threads is not None:
        env['OMP_NUM_THREADS'] = str(cpu_threads)
    # The plan, rather than inherited GoogleTest controls, defines the selected cases.
    for key in ('GTEST_FILTER', 'GTEST_OUTPUT', 'GTEST_TOTAL_SHARDS', 'GTEST_SHARD_INDEX',
                'GTEST_SHARD_STATUS_FILE', 'GTEST_REPEAT', 'GTEST_ALSO_RUN_DISABLED_TESTS',
                'GTEST_FAIL_FAST', 'GTEST_BREAK_ON_FAILURE', 'GTEST_THROW_ON_FAILURE'):
        env.pop(key, None)
    record = {'name': name, 'kind': kind, 'command': command, 'started_at': utc_now(),
              'status': 'running', 'returncode': None, 'cases': empty_cases(),
              'log': str((directory / 'run.log').relative_to(results)),
              'xml': str(xml.relative_to(results)) if kind == 'gtest' else None,
              'omp_num_threads': env.get('OMP_NUM_THREADS'),
              'report_complete': None if kind == 'command' else False, 'errors': []}
    limit = float(test.get('timeout_seconds', timeout))
    record['timeout_seconds'] = limit
    start = time.monotonic()
    process = None
    with (directory / 'run.log').open('w') as log:
        try:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record['returncode'] = process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            terminate_process(process)
            record.update(status='timeout', returncode=process.returncode)
            record['errors'].append(f'Executable exceeded {limit:g} seconds')
        except KeyboardInterrupt:
            if process is not None:
                terminate_process(process)
            record.update(status='interrupted', returncode=process.returncode if process else None)
            record['errors'].append('Run interrupted')
        except OSError as exc:
            record['status'] = 'error'
            record['errors'].append(str(exc))
    if kind == 'gtest':
        try:
            record.update(parse_gtest_report(xml))
            record['report_complete'] = True
            for error in record['framework_errors']:
                record['errors'].append('GoogleTest framework failure: ' + '; '.join(error['messages']))
        except (OSError, ValueError, ET.ParseError) as exc:
            record['errors'].append(f'No complete GoogleTest result: {exc}')
    if record['status'] == 'running':
        if record.get('framework_errors'):
            record['status'] = 'error'
        elif record['returncode'] != 0:
            record['status'] = 'failed' if kind == 'command' or record['cases']['failed'] else 'error'
            record['errors'].append(f'Executable exited with status {record["returncode"]}')
        elif record['errors']:
            record['status'] = 'error'
        elif record['cases']['failed']:
            record['status'] = 'failed'
        elif kind == 'gtest' and record['cases']['passed'] == 0:
            record['status'] = 'skipped'
        else:
            record['status'] = 'passed'
    record.update(finished_at=utc_now(), duration_seconds=round(time.monotonic() - start, 3))
    (directory / 'exit-status').write_text(str(record['returncode']) + '\n')
    write_json(directory / 'summary.json', record)
    return record


def summarize(plan, records, started_at, results):
    counts = empty_cases()
    command_checks = {'passed': 0, 'failed': 0}
    statuses = {key: 0 for key in ('passed', 'failed', 'skipped', 'error', 'timeout', 'interrupted')}
    for record in records:
        statuses[record['status']] += 1
        for key, value in record['cases'].items():
            counts[key] += value
        if record['kind'] == 'command':
            command_checks['passed' if record['status'] == 'passed' else 'failed'] += 1
    failures = any(statuses[key] for key in ('failed', 'error', 'timeout', 'interrupted'))
    completed = len(records) == len(plan['tests']) and not statuses['interrupted']
    return {'schema_version': 1, 'started_at': started_at, 'updated_at': utc_now(),
            'results_directory': str(results), 'completed': completed,
            'exit_code': 130 if statuses['interrupted'] else int(failures or not completed),
            'cases': counts, 'command_checks': command_checks, 'executables': statuses,
            'case_counts_complete': all(record['report_complete'] is not False for record in records) and completed,
            'planned_executables': len(plan['tests']), 'completed_executables': len(records),
            'not_run': [test['name'] for test in plan['tests'][len(records):]],
            'excluded_targets': plan.get('excluded', []), 'results': records,
            'build_only': plan.get('build_only', []),
            'cpu_threads': plan.get('cpu_threads'),
            'local_check_results': plan.get('local_check_results', []),
            'capability_notes': plan.get('capability_notes', []),
            'counting_note': 'Excluded targets have no fabricated case count; command checks are separate from GoogleTest cases.'}


def run_plan(bundle, timeout=3600):
    bundle = Path(bundle).resolve(strict=True)
    plan = json.loads((bundle / 'test-plan.json').read_text())
    validate_plan(plan)
    if timeout <= 0:
        raise ValueError('Timeout must be positive')
    if not os.access(bundle / 'run', os.X_OK):
        raise ValueError('Bundle launcher is missing or not executable')
    output_root = bundle / 'results'
    output_root.mkdir(exist_ok=True)
    prefix = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ-tests.')
    results = Path(tempfile.mkdtemp(prefix=prefix, dir=output_root))
    write_json(results / 'test-plan.json', plan)
    started = utc_now()
    records = []
    summary = summarize(plan, records, started, results)
    write_json(results / 'summary.json', summary)
    print(f'Running {len(plan["tests"])} test executables sequentially; results: {results}', flush=True)
    with (results / 'run.log').open('w') as log:
        for index, test in enumerate(plan['tests'], 1):
            message = f'[{index}/{len(plan["tests"])}] {test["name"]}'
            print(message, flush=True)
            print(message, file=log, flush=True)
            record = run_target(bundle, test, results, timeout, plan.get('cpu_threads'))
            records.append(record)
            summary = summarize(plan, records, started, results)
            write_json(results / 'summary.json', summary)
            message = (f"  {record['status']}: {record['cases']}; "
                       f"exit={record['returncode']}, duration={record['duration_seconds']}s")
            print(message, flush=True)
            print(message, file=log, flush=True)
            if record['status'] == 'interrupted':
                break
        message = (f"GoogleTest cases: {summary['cases']}; command checks: {summary['command_checks']}; "
                   f"executable statuses: {summary['executables']}; "
                   f"excluded targets: {len(summary['excluded_targets'])}")
        print(message, flush=True)
        print(message, file=log, flush=True)
    (results / 'exit-status').write_text(str(summary['exit_code']) + '\n')
    print(f'Summary: {results / "summary.json"}', flush=True)
    return summary['exit_code'], results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle_dir', type=Path)
    parser.add_argument('--timeout', type=float, default=3600,
                        help='Per-executable timeout in seconds (default: 3600)')
    args = parser.parse_args()
    try:
        exit_code, _ = run_plan(args.bundle_dir, args.timeout)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f'run_tests: {exc}\n')
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
