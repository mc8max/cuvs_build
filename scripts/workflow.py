#!/usr/bin/env python3
"""One configuration for local compilation and remote cuVS execution."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
_plan_spec = importlib.util.spec_from_file_location('cuvs_native_test_plan', ROOT / 'scripts/test_plan.py')
_plan_module = importlib.util.module_from_spec(_plan_spec)
_plan_spec.loader.exec_module(_plan_module)
create_plan, check_headers = _plan_module.create_plan, _plan_module.check_headers


def load_config(path):
    """Expand configuration variables from a local .env and process overrides."""
    variables = {}
    env_file = ROOT / '.env'
    if env_file.exists():
        for number, line in enumerate(env_file.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            assignment = re.fullmatch(r'(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', line)
            if not assignment:
                raise ValueError(f'Invalid .env assignment at line {number}')
            try:
                values = shlex.split(assignment[2], comments=True, posix=True)
            except ValueError as error:
                raise ValueError(f'Invalid .env value at line {number}') from error
            if len(values) > 1:
                raise ValueError(f'Quote .env values containing spaces at line {number}')
            variables[assignment[1]] = values[0] if values else ''
    variables.update(os.environ)

    def substitute(match):
        name = match[1]
        if not variables.get(name):
            raise ValueError(f'Set {name} to a nonempty value in {env_file} or the environment')
        return variables[name]

    def expand(value):
        if isinstance(value, str):
            return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', substitute, value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return expand(json.loads(Path(path).read_text()))


def run(args, **kwargs):
    print('+ ' + shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def version(text):
    found = re.search(r'\d+(?:\.\d+)+', text)
    if not found:
        raise ValueError(f'Cannot parse version: {text!r}')
    return tuple(map(int, found.group().split('.')))


def validate_inventory(info, config):
    if info.get('schema_version') != 1 or info.get('system') != 'Linux':
        raise ValueError('A successful Linux inventory is required; run make inventory.')
    if info.get('machine') != 'x86_64' or platform.machine() != 'x86_64':
        raise ValueError('This toolchain supports x86_64 Linux hosts and targets only.')
    baseline = config['toolchain']['glibc_baseline']
    if version(info['glibc']['output']) < version(baseline):
        raise ValueError(f'Remote glibc is older than the {baseline} build sysroot.')
    if not info.get('gpus'):
        raise ValueError('Remote inventory contains no usable NVIDIA GPU/compute capability.')
    architectures = []
    for gpu in info['gpus']:
        capability = gpu['compute_capability']
        if not re.fullmatch(r'\d+\.\d+', capability):
            raise ValueError(f'Unknown GPU compute capability: {capability}')
        if version(gpu['driver_version']) < version(config['toolchain']['minimum_driver']):
            raise ValueError('Remote driver is older than the configured full CUDA compatibility floor.')
        architectures.append(capability.replace('.', '') + '-real')
    for name in ('python3', 'rsync', 'tar', 'ldd'):
        if not info['tools'][name]['path']:
            raise ValueError(f'Remote command {name} is missing.')
    return sorted(set(architectures), key=lambda value: int(value.split('-')[0]))


class Workflow:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        self.state = ROOT / '.state'
        self.state.mkdir(exist_ok=True)

    def path(self, key):
        return (ROOT / self.config[key]).resolve()

    def ssh(self):
        cfg = self.config['ssh']
        known = (ROOT / cfg['known_hosts']).resolve()
        known.parent.mkdir(parents=True, exist_ok=True)
        return ['ssh', '-F', str((ROOT / cfg['config']).resolve()),
                '-o', 'IgnoreUnknown=UseKeychain', '-o', f'UserKnownHostsFile={known}',
                '-o', 'StrictHostKeyChecking=accept-new', '-o', 'BatchMode=yes',
                '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
                '-o', 'ServerAliveCountMax=3']

    def remote(self, args, **kwargs):
        return run(self.ssh() + [self.config['ssh']['host'], shlex.join(map(str, args))], **kwargs)

    def inventory(self):
        result = self.remote(['python3', '-'], input=(ROOT / 'scripts/inventory.py').read_text(),
                             text=True, stdout=subprocess.PIPE)
        info = json.loads(result.stdout)
        info['ssh_host'] = self.config['ssh']['host']
        destination = self.path('inventory')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(info, indent=2) + '\n')
        temporary.replace(destination)
        print(f'Saved measured inventory: {destination}')
        for gpu in info.get('gpus', []):
            print(f"GPU: {gpu['name']}, SM {gpu['compute_capability']}, {gpu['memory_mib']} MiB, driver {gpu['driver_version']}")
        return info

    def target(self):
        path = self.path('inventory')
        if not path.is_file():
            raise ValueError('Remote inventory missing. Run make inventory with a working SSH identity.')
        info = json.loads(path.read_text())
        return info, validate_inventory(info, self.config)

    def compilation_target(self):
        """Allow explicit build-only SMs without fabricating remote inventory."""
        if 'compile_architectures' not in self.config:
            return self.target()
        if platform.system() != 'Linux' or platform.machine() != 'x86_64':
            raise ValueError('Explicit compilation requires an x86_64 Linux build host.')
        architectures = self.config['compile_architectures']
        if (not isinstance(architectures, list) or not architectures or
                any(not isinstance(arch, str) or not re.fullmatch(r'[1-9]\d{1,2}-real', arch)
                    for arch in architectures)):
            raise ValueError('compile_architectures must contain explicit SMs, e.g. ["120-real"].')
        return {}, sorted(set(architectures), key=lambda arch: int(arch.split('-')[0]))

    def require_deployment_inventory(self):
        """Build-only configurations still need real compatibility checks to ship."""
        if 'compile_architectures' in self.config:
            _, measured = self.target()
            _, compiled = self.compilation_target()
            if measured != compiled:
                raise ValueError('Compiled GPU architectures differ from measured inventory; rebuild for the pod.')

    def conda(self):
        binary = shutil.which('conda')
        if not binary:
            raise ValueError('conda is required on PATH to create/activate the build environment.')
        return binary

    def planned_native_tests(self):
        return self.config.get('tests', {}).get('profile') in ('all-native', 'quick-native')

    def native_plan(self, ctest_json=None):
        info, _ = self.compilation_target()
        selection = None
        if self.config.get('tests', {}).get('profile') == 'quick-native':
            selection = self.config['tests'].get('selection')
            if not selection:
                raise ValueError('quick-native requires a nonempty tests.selection')
        plan = create_plan(self.path('source_dir'), self.config['toolchain']['cuda'], info,
                           ctest_json, enable_mg=self.config['cmake'].get('BUILD_MG_ALGOS') == 'ON',
                           enable_mnmg=self.config['cmake'].get('BUILD_MNMG_TESTS') == 'ON',
                           selection=selection)
        if 'compile_architectures' in self.config:
            plan['compilation_architectures'] = self.compilation_target()[1]
            plan['capability_notes'].append(
                'Build-only architecture selection; remote GPU, driver, and runtime compatibility are not verified.')
        cpu_threads = self.config.get('tests', {}).get('cpu_threads')
        if cpu_threads is not None:
            if type(cpu_threads) is not int or cpu_threads < 1:
                raise ValueError('tests.cpu_threads must be a positive integer')
            plan['cpu_threads'] = cpu_threads
        return plan

    def build_targets(self):
        targets = self.config['targets'][:]
        if self.planned_native_tests():
            targets += self.native_plan()['build_targets']
        return list(dict.fromkeys(targets))

    def ensure_test_dependencies(self):
        wanted = self.config.get('test_dependencies', [])
        installed = {json.loads(p.read_text())['name']
                     for p in (self.path('env_prefix') / 'conda-meta').glob('*.json')}
        if set(wanted) - installed:
            tc = self.config['toolchain']
            run([self.conda(), 'install', '-y', '--prefix', self.path('env_prefix'),
                 '--override-channels', '-c', 'rapidsai', '-c', 'conda-forge',
                 f"cuda-version={tc['cuda']}", f"gcc_linux-64={tc['gcc_major']}",
                 f"gxx_linux-64={tc['gcc_major']}", f"sysroot_linux-64={tc['glibc_baseline']}"] + wanted)
        if wanted:
            lock = run([self.conda(), 'list', '--prefix', self.path('env_prefix'), '--explicit'],
                       stdout=subprocess.PIPE, text=True).stdout
            (self.state / 'environment.lock.txt').write_text(lock)

    def prepare_environment(self):
        prefix = self.path('env_prefix')
        tc = self.config['toolchain']
        specs = ['python=3.13', 'cmake>=4.0', 'ninja', 'make', 'patchelf',
                 f"gcc_linux-64={tc['gcc_major']}", f"gxx_linux-64={tc['gcc_major']}",
                 f"sysroot_linux-64={tc['glibc_baseline']}", f"cuda-version={tc['cuda']}",
                 'cuda-nvcc', 'cuda-cudart-dev', 'cuda-nvrtc-dev', 'cuda-nvtx-dev',
                 'cuda-profiler-api', 'libcublas-dev', 'libcufile-dev', 'libcurand-dev',
                 'libcusolver-dev', 'libcusparse-dev', 'libnvjitlink-dev', 'dlpack>=0.8,<1.0']
        if not (prefix / 'conda-meta/history').is_file():
            run([self.conda(), 'create', '-y', '--prefix', prefix, '--override-channels',
                 '-c', 'conda-forge'] + specs,
                env={**os.environ, 'CONDA_PKGS_DIRS': str(self.path('source_dir') / '.build-cache/conda')})
        else:
            print(f'Reusing existing build environment: {prefix}')
        lock = run([self.conda(), 'list', '--prefix', prefix, '--explicit'],
                   stdout=subprocess.PIPE, text=True).stdout
        (self.state / 'environment.lock.txt').write_text(lock)

    def build_environment(self):
        prefix = self.path('env_prefix')
        if not (prefix / 'conda-meta/history').is_file():
            raise ValueError('Build environment missing. Run make setup.')
        base = run([self.conda(), 'info', '--base'], stdout=subprocess.PIPE, text=True).stdout.strip()
        # Positional arguments keep filesystem paths out of shell source code.
        raw = run(['bash', '-c', 'source "$1/etc/profile.d/conda.sh" && conda activate "$2" && env -0',
                   'build-env', base, prefix], stdout=subprocess.PIPE).stdout
        env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if b'=' in item)
        tc = self.config['toolchain']
        common = (f"-march={tc['cpu_arch']} -mtune={tc['cpu_tune']} -fPIC "
                  f"-I{prefix / 'include'} -I{prefix / 'targets/x86_64-linux/include'}")
        # CMake and smoke explicitly select the host compiler; avoid Conda's
        # duplicate implicit -ccbin option and hidden environment overrides.
        env.update(CFLAGS=common, CXXFLAGS=common, CUDAFLAGS='',
                   NVCC_PREPEND_FLAGS='', NVCC_APPEND_FLAGS='',
                   CPM_SOURCE_CACHE=str(self.path('cpm_cache')),
                   CMAKE_BUILD_PARALLEL_LEVEL=str(self.config['jobs']))
        return env

    def doctor(self):
        env = self.build_environment()
        prefix = self.path('env_prefix')
        compiler = prefix / 'bin/x86_64-conda-linux-gnu-g++'
        nvcc = prefix / 'bin/nvcc'
        versions = {}
        for name, command in [('cxx', [compiler, '--version']), ('cuda', [nvcc, '--version']),
                              ('cmake', [prefix / 'bin/cmake', '--version']),
                              ('ninja', [prefix / 'bin/ninja', '--version'])]:
            versions[name] = run(command, env=env, stdout=subprocess.PIPE, text=True).stdout
        tc = self.config['toolchain']
        if version(versions['cxx'])[0] != tc['gcc_major']:
            raise ValueError('Installed compiler does not match toolchain.gcc_major.')
        if f"release {tc['cuda']}," not in versions['cuda']:
            raise ValueError('Installed nvcc does not match toolchain.cuda.')
        if version(versions['cmake']) < (4, 0):
            raise ValueError('cuVS requires CMake >= 4.0.')
        sysroots = list((prefix / 'conda-meta').glob('sysroot_linux-64-*.json'))
        if not sysroots or json.loads(sysroots[0].read_text())['version'] != tc['glibc_baseline']:
            raise ValueError('Installed glibc sysroot differs from toolchain.glibc_baseline.')
        info, archs = self.compilation_target()
        supported = run([nvcc, '--list-gpu-code'], env=env, stdout=subprocess.PIPE, text=True).stdout.split()
        for arch in archs:
            if 'sm_' + arch.split('-')[0] not in supported:
                raise ValueError(f'Toolkit cannot compile {arch}; select a supported CUDA toolkit.')
        if info:
            print(f"Compatible: {info['machine']}, glibc {info['glibc']['output']}, GPU {';'.join(archs)}")
        else:
            print(f"Build-only target: Linux x86_64, GPU {';'.join(archs)}; deployment inventory pending.")
        (self.state / 'local-toolchain.json').write_text(json.dumps(versions, indent=2) + '\n')
        return env, archs

    def configure(self):
        self.ensure_test_dependencies()
        env, archs = self.doctor()
        source, build, prefix = self.path('source_dir'), self.path('build_dir'), self.path('env_prefix')
        if build == source or source not in build.parents:
            raise ValueError('build_dir must be a dedicated directory inside source_dir.')
        build.mkdir(parents=True, exist_ok=True)
        if not (build / '.gitignore').exists():
            (build / '.gitignore').write_text('*\n')
        opts = {
            'CMAKE_BUILD_TYPE': self.config['build_type'],
            'CMAKE_INSTALL_PREFIX': str(build / 'install'),
            'CMAKE_C_COMPILER': str(prefix / 'bin/x86_64-conda-linux-gnu-gcc'),
            'CMAKE_CXX_COMPILER': str(prefix / 'bin/x86_64-conda-linux-gnu-g++'),
            'CMAKE_CUDA_COMPILER': str(prefix / 'bin/nvcc'),
            'CMAKE_CUDA_HOST_COMPILER': str(prefix / 'bin/x86_64-conda-linux-gnu-g++'),
            'CUDAToolkit_ROOT': str(prefix),
            'CMAKE_CUDA_ARCHITECTURES': ';'.join(archs),
            'CMAKE_C_FLAGS': env['CFLAGS'], 'CMAKE_CXX_FLAGS': env['CXXFLAGS'],
            'CMAKE_CUDA_FLAGS': f"-Xcompiler=-march={self.config['toolchain']['cpu_arch']},-mtune={self.config['toolchain']['cpu_tune']}",
            'CMAKE_BUILD_RPATH_USE_ORIGIN': 'ON',
        }
        # Upstream enables many external ANN libraries by default. Enable only selections in config.
        bench_cmake = (source / 'cpp/bench/ann/CMakeLists.txt').read_text()
        for key in re.findall(r'option\((CUVS_(?:ANN|KNN)_BENCH_USE_\w+)', bench_cmake):
            opts[key] = 'OFF'
        forbidden = set(self.config['cmake']) & {k for k in opts if k.startswith('CMAKE_') or k == 'CUDAToolkit_ROOT'}
        if forbidden:
            raise ValueError(f'Use the toolchain settings instead of overriding {sorted(forbidden)}')
        opts.update(self.config['cmake'])
        if self.planned_native_tests():
            for key, value in self.native_plan()['required_cmake'].items():
                if opts.get(key) != value:
                    raise ValueError(f'Native test profile requires {key}={value}')
        cache = self.state / 'runpod-cache.cmake'
        cache.write_text('# Generated from config/build.json and measured RunPod inventory.\n' +
                         ''.join(f'set({k} [==[{v}]==] CACHE STRING "Managed by cuvs_build" FORCE)\n'
                                 for k, v in opts.items()))
        run([prefix / 'bin/cmake', '-S', source / 'cpp', '-B', build, '-G', 'Ninja', '-C', cache], env=env)
        if self.planned_native_tests():
            configured = json.loads(run([prefix / 'bin/ctest', '--test-dir', build, '--show-only=json-v1'],
                                        env=env, stdout=subprocess.PIPE, text=True).stdout)
            (self.state / 'test-plan.json').write_text(json.dumps(self.native_plan(configured), indent=2) + '\n')
        (self.state / 'configured.json').write_text(json.dumps({'fingerprint': self.fingerprint(),
                                                               'architectures': archs}, indent=2))

    def fingerprint(self):
        _, archs = self.compilation_target()
        source_tests = self.native_plan()['source_sha256'] if self.planned_native_tests() else None
        return hashlib.sha256(json.dumps([self.config, archs, source_tests], sort_keys=True).encode()).hexdigest()

    def require_configured(self):
        stamp = self.state / 'configured.json'
        if not stamp.exists() or json.loads(stamp.read_text())['fingerprint'] != self.fingerprint():
            raise ValueError('Configuration changed or has not completed. Run make configure.')

    def build(self):
        try:
            self.require_configured()
        except ValueError:
            self.configure()
        run([self.path('env_prefix') / 'bin/cmake', '--build', self.path('build_dir'),
             '--parallel', str(self.config['jobs']), '--target'] + self.build_targets(), env=self.build_environment())
        if self.planned_native_tests():
            configured = json.loads(run([self.path('env_prefix') / 'bin/ctest', '--test-dir',
                                         self.path('build_dir'), '--show-only=json-v1'],
                                        env=self.build_environment(), stdout=subprocess.PIPE, text=True).stdout)
            plan = self.native_plan(configured)
            plan['local_check_results'] = [check_headers(self.path('source_dir'))]
            for check in plan['build_only']:
                check['status'] = 'passed'
            (self.state / 'test-plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    def smoke(self):
        self.inventory()
        env, archs = self.doctor()
        binary = self.state / 'cuda-smoke'
        prefix = self.path('env_prefix')
        command = [prefix / 'bin/nvcc', '-ccbin', prefix / 'bin/x86_64-conda-linux-gnu-g++',
                   '-O2', '--cudart=static', '-Xcompiler=-static-libstdc++,-static-libgcc',
                   ROOT / 'scripts/cuda-smoke.cu', '-o', binary]
        for arch in archs:
            sm = arch.split('-')[0]
            command += ['-gencode', f'arch=compute_{sm},code=sm_{sm}']
        run(command, env=env)
        destination = self.remote_root() + '/smoke'
        self.remote(['mkdir', '-p', destination])
        run(['rsync', '-az', '--no-owner', '--no-group', '--protect-args', '-e', shlex.join(self.ssh()), str(binary),
             self.config['ssh']['host'] + ':' + destination + '/'])
        result = self.remote([destination + '/cuda-smoke'], text=True, stdout=subprocess.PIPE)
        print(result.stdout, end='')
        (self.state / 'cuda-smoke.log').write_text(result.stdout)

    def package(self):
        self.require_deployment_inventory()
        self.require_configured()
        release = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        bundle = ROOT / 'artifacts' / release
        resolved_config = self.state / 'resolved-build-config.json'
        resolved_config.write_text(json.dumps(self.config, indent=2) + '\n')
        args = [sys.executable, ROOT / 'scripts/bundle.py', '--build-dir', self.path('build_dir'),
             '--env-prefix', self.path('env_prefix'), '--output', bundle,
             '--config', resolved_config, '--inventory', self.path('inventory')]
        if self.planned_native_tests():
            args += ['--test-plan', self.state / 'test-plan.json']
        run(args + ['--targets'] + self.build_targets(), env=self.build_environment())
        (self.state / 'latest-bundle').write_text(release + '\n')

    def remote_root(self):
        root = self.config['remote_root']
        if not root.startswith('/') or '..' in PurePosixPath(root).parts or len(PurePosixPath(root).parts) < 3:
            raise ValueError('remote_root must be an absolute dedicated directory, e.g. /workspace/cuvs-build.')
        return root.rstrip('/')

    def release(self, filename):
        path = self.state / filename
        if not path.exists():
            raise ValueError('No release recorded. Run make publish first.')
        release = path.read_text().strip()
        if not re.fullmatch(r'[0-9T]+Z', release):
            raise ValueError('Invalid release identifier.')
        return release

    def publish(self):
        self.require_deployment_inventory()
        release = self.release('latest-bundle')
        bundle = ROOT / 'artifacts' / release
        bundled_config = json.loads((bundle / 'build-config.json').read_text())
        if bundled_config != self.config:
            raise ValueError('Build configuration changed since packaging; run make package before publishing.')
        fresh = self.inventory()
        validate_inventory(fresh, self.config)
        bundled = json.loads((bundle / 'remote-inventory.json').read_text())
        if validate_inventory(bundled, self.config) != validate_inventory(fresh, self.config):
            raise ValueError('RunPod GPU changed since build; reconfigure/rebuild before publishing.')
        manifest = json.loads((bundle / 'manifest.json').read_text())
        required = manifest.get('max_required_glibc')
        if required and version(required) > version(fresh['glibc']['output']):
            raise ValueError(f'Bundle requires glibc {required}; remote system is older.')
        destination = f'{self.remote_root()}/releases/{release}'
        self.remote(['mkdir', '-p', destination])
        reuse = []
        if (self.state / 'published-release').exists():
            previous = self.release('published-release')
            if previous != release:
                reuse = [f'--link-dest={self.remote_root()}/releases/{previous}']
        # Relocating an ELF can change bytes without changing size or preserved
        # mtime; compare contents before reusing an older release's hardlink.
        run(['rsync', '-azH', '--checksum', '--no-owner', '--no-group', '--protect-args', '--partial-dir=.rsync-partial',
             '--info=stats2', '-e', shlex.join(self.ssh())] + reuse + [str(bundle) + '/',
             self.config['ssh']['host'] + ':' + destination + '/'])
        self.remote(['python3', '-c', '''import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
if manifest.get("format") != 1 or not manifest.get("files"):
    raise SystemExit("Missing or unsupported bundle manifest")
for name, expected in manifest["files"].items():
    path = root / name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise SystemExit(f"Bundle integrity check failed: {name}")
print(f"Verified {len(manifest['files'])} bundled files")
''', destination])
        self.remote(['bash', destination + '/scripts/remote-run.sh', destination, 'probe'])
        # Record a release only after upload and runtime loader validation succeed.
        (self.state / 'published-release').write_text(release + '\n')
        print(f'Published {destination}')

    def execute_remote(self, mode):
        release = self.release('published-release')
        destination = f'{self.remote_root()}/releases/{release}'
        args = ['bash', destination + '/scripts/remote-run.sh', destination, mode]
        if mode == 'test':
            if self.planned_native_tests():
                args = ['python3', destination + '/scripts/run_tests.py', destination,
                        '--timeout', str(self.config['tests'].get('timeout_seconds', 7200))]
            else:
                args += [self.config['test_target'], self.config['gtest_filter']]
        elif mode == 'benchmark':
            args += [self.config['benchmark_target']]
        self.remote(args)

    def fetch(self):
        release = self.release('published-release')
        destination = ROOT / 'results' / release
        destination.mkdir(parents=True, exist_ok=True)
        run(['rsync', '-az', '--no-owner', '--no-group', '--protect-args',
             '--exclude=/*-tests.*/*/work/', '--exclude=/*-tests.*/*/tmp/',
             '-e', shlex.join(self.ssh()),
             self.config['ssh']['host'] + ':' + f'{self.remote_root()}/releases/{release}/results/',
             str(destination) + '/'])

    def cycle(self):
        failures = []
        try:
            for phase in ('test', 'benchmark'):
                try:
                    self.execute_remote(phase)
                except subprocess.CalledProcessError as exc:
                    failures.append(f'{phase}: exit {exc.returncode}')
        finally:
            self.fetch()
        if failures:
            raise ValueError('Cycle completed with failures: ' + '; '.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config/build.json'))
    parser.add_argument('command', choices=['inventory', 'doctor', 'setup', 'configure', 'build',
                                           'package', 'publish', 'test', 'benchmark', 'fetch', 'cycle', 'ssh', 'smoke'])
    args = parser.parse_args()
    flow = Workflow(args.config)
    if args.command == 'setup':
        flow.target()
        flow.prepare_environment()
        flow.configure()
    elif args.command in ('test', 'benchmark'):
        flow.execute_remote(args.command)
    elif args.command == 'ssh':
        run(flow.ssh() + [flow.config['ssh']['host']])
    else:
        getattr(flow, args.command)()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        sys.exit(1)
