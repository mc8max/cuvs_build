#!/usr/bin/env python3
"""Check the isolated builder and compile Linux C++ and CUDA probe programs."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile


def run(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f'Command failed: {command!r}\n{result.stdout}\n{result.stderr}')
    return result.stdout.strip()


def check(config=None):
    expected = config['toolchain'] if config else {
        'cuda': os.environ['CUVS_CUDA_VERSION'],
        'gcc_major': int(os.environ['CUVS_GCC_MAJOR']),
        'glibc_baseline': os.environ['CUVS_GLIBC_BASELINE'],
    }
    if (platform.system(), platform.machine()) != ('Linux', 'x86_64'):
        raise RuntimeError('The builder must execute Linux x86_64 tools.')
    prefix = Path(os.environ['CUVS_ENV_PREFIX'])
    tools = ('python', 'cmake', 'ctest', 'ninja', 'make', 'git', 'readelf',
             'patchelf', 'rsync', 'ssh', 'pkg-config', 'nvcc',
             'x86_64-conda-linux-gnu-gcc', 'x86_64-conda-linux-gnu-g++')
    paths = {}
    for name in tools:
        found = shutil.which(name)
        if not found or not Path(found).is_relative_to(prefix):
            raise RuntimeError(f'Missing isolated environment tool: {name}')
        paths[name] = found
    gcc = run([paths['x86_64-conda-linux-gnu-g++'], '-dumpfullversion'])
    if int(gcc.split('.')[0]) != expected['gcc_major']:
        raise RuntimeError(f'Wrong GCC version: {gcc}')
    nvcc = run([paths['nvcc'], '--version'])
    if f"release {expected['cuda']}," not in nvcc:
        raise RuntimeError(f'Wrong CUDA version: {nvcc}')
    cmake = run([paths['cmake'], '--version']).splitlines()[0]
    if tuple(map(int, re.search(r'(\d+)\.(\d+)', cmake).groups())) < (4, 0):
        raise RuntimeError(f'cuVS requires CMake >= 4.0: {cmake}')
    packages = {data['name']: data for p in (prefix / 'conda-meta').glob('*.json')
                for data in [json.loads(p.read_text())]}
    if packages['sysroot_linux-64']['version'] != expected['glibc_baseline']:
        raise RuntimeError('The installed glibc sysroot does not match the configuration.')
    for name in ('nccl', 'libucxx', 'ucx', 'dlpack', 'cuda-nvrtc-dev', 'libcufile-dev'):
        if name not in packages:
            raise RuntimeError(f'Missing build dependency: {name}')

    # This is a compiler probe, not a target choice for cuVS. The real build
    # must continue to select architectures from measured RunPod inventory.
    supported = run([paths['nvcc'], '--list-gpu-code']).split()
    probe_sm = 'sm_80'
    if probe_sm not in supported:
        raise RuntimeError(f'CUDA toolkit cannot compile the probe architecture {probe_sm}')
    env = dict(os.environ, NVCC_PREPEND_FLAGS='', NVCC_APPEND_FLAGS='', CUDAFLAGS='')
    with tempfile.TemporaryDirectory(prefix='cuvs-toolchain-check-') as scratch:
        root = Path(scratch)
        (root / 'host.cpp').write_text('#include <iostream>\nint main() { std::cout << "host-ok\\n"; }\n')
        run([paths['x86_64-conda-linux-gnu-g++'], '-std=c++20', '-march=x86-64',
             '-mtune=generic', str(root / 'host.cpp'), '-o', str(root / 'host')], env=env)
        if run([str(root / 'host')], env=env) != 'host-ok':
            raise RuntimeError('The compiled C++ probe failed.')
        (root / 'kernel.cu').write_text(
            '#include <cuda_runtime.h>\n'
            '__global__ void kernel(int* value) { *value = 42; }\n'
            'int main() { int* p = nullptr; cudaMalloc(&p, sizeof(int)); '
            'kernel<<<1, 1>>>(p); return int(cudaDeviceSynchronize()); }\n'
        )
        run([paths['nvcc'], '-ccbin', paths['x86_64-conda-linux-gnu-g++'],
             '-std=c++20', f'-arch={probe_sm}', str(root / 'kernel.cu'),
             '-o', str(root / 'cuda-probe')], env=env)
        elf = run([paths['readelf'], '--file-header', str(root / 'cuda-probe')])
        if 'Advanced Micro Devices X86-64' not in elf:
            raise RuntimeError('The CUDA probe is not an x86_64 ELF executable.')
        # Do not execute a GPU program on a host without an NVIDIA GPU.
        (root / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 4.0)\n'
            'project(cuvs_toolchain_probe LANGUAGES CXX CUDA)\n'
            'add_executable(host_probe host.cpp)\n'
            'target_compile_features(host_probe PRIVATE cxx_std_20)\n'
            'add_executable(cuda_probe kernel.cu)\n'
            'set_target_properties(cuda_probe PROPERTIES CUDA_STANDARD 20)\n'
        )
        run([paths['cmake'], '-S', str(root), '-B', str(root / 'build'), '-G', 'Ninja',
             '-DCMAKE_CXX_COMPILER=' + paths['x86_64-conda-linux-gnu-g++'],
             '-DCMAKE_CUDA_COMPILER=' + paths['nvcc'],
             '-DCMAKE_CUDA_HOST_COMPILER=' + paths['x86_64-conda-linux-gnu-g++'],
             '-DCUDAToolkit_ROOT=' + str(prefix), '-DCMAKE_CUDA_ARCHITECTURES=80-real'], env=env)
        run([paths['cmake'], '--build', str(root / 'build'), '--parallel', '1'], env=env)

    return {
        'platform': 'Linux x86_64', 'gcc': gcc, 'cmake': cmake,
        'cuda': re.search(r'release ([\d.]+), V([\d.]+)', nvcc).group(0),
        'glibc_sysroot': packages['sysroot_linux-64']['version'],
        'ninja': run([paths['ninja'], '--version']),
        'rsync': run([paths['rsync'], '--version']).splitlines()[0],
        'test_dependencies': {name: packages[name]['version'] for name in ('nccl', 'libucxx', 'ucx')},
        'checks': {'cpp20_compile_and_run': 'passed', 'cuda_compile_and_link': 'passed',
                   'cuda_elf_x86_64': 'passed', 'cmake_ninja_cuda_build': 'passed',
                   'gpu_execution': 'not run'},
        'tools': paths,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Validate versions against this build.json')
    args = parser.parse_args()
    print(json.dumps(check(json.loads(args.config.read_text()) if args.config else None), indent=2))
