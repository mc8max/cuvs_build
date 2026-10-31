#!/usr/bin/env python3
"""Read-only Linux/CUDA inventory; also runs over SSH via python3 stdin."""
import csv
import ctypes
import datetime
import io
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess


def command(*args):
    try:
        p = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=25)
        return {"returncode": p.returncode, "output": p.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": -1, "output": str(exc)}


def cuda_driver_devices():
    """Query CUDA-visible devices without creating contexts or allocating GPU memory.

    Prototypes follow CUDA 12.9 cuda.h: CUdevice/CUresult are int, total memory
    uses size_t*, and cuInit takes unsigned int. Do not load toolkit stubs.
    """
    result = {'library': 'libcuda.so.1', 'calls': {}, 'devices': []}
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        prototypes = {
            'cuInit': [ctypes.c_uint],
            'cuDeviceGetCount': [ctypes.POINTER(ctypes.c_int)],
            'cuDeviceGet': [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            'cuDeviceTotalMem_v2': [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int],
            'cuDeviceGetName': [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
            'cuDeviceGetPCIBusId': [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
        }
        for name, arguments in prototypes.items():
            function = getattr(driver, name)
            function.argtypes, function.restype = arguments, ctypes.c_int

        def invoke(record, name, *args):
            status = int(getattr(driver, name)(*args))
            record[name] = {'returncode': status}
            return status == 0

        if not invoke(result['calls'], 'cuInit', 0):
            return result
        count = ctypes.c_int()
        if not invoke(result['calls'], 'cuDeviceGetCount', ctypes.byref(count)):
            return result
        result['device_count'] = count.value
        for ordinal in range(count.value):
            item = {'ordinal': ordinal, 'calls': {}}
            result['devices'].append(item)
            device = ctypes.c_int()
            if not invoke(item['calls'], 'cuDeviceGet', ctypes.byref(device), ordinal):
                continue
            memory = ctypes.c_size_t()
            if invoke(item['calls'], 'cuDeviceTotalMem_v2', ctypes.byref(memory), device):
                item['memory_bytes'] = memory.value
                item['memory_mib'] = str(memory.value / (1024 * 1024))
            for function, field in (('cuDeviceGetName', 'name'), ('cuDeviceGetPCIBusId', 'pci_bus_id')):
                text = ctypes.create_string_buffer(256)
                if invoke(item['calls'], function, text, len(text), device):
                    item[field] = text.value.decode('utf-8', errors='replace')
    except (OSError, AttributeError) as exc:
        result['error'] = str(exc)
    return result


def pci_identity(value):
    match = re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)', value)
    return tuple(int(part, 16) for part in match.groups()) if match else None


def supplement_gpu_memory(gpus):
    """Associate CUDA/MIG memory by PCI identity, never by enumeration order."""
    if all(re.fullmatch(r'\d+(?:\.\d+)?', gpu['memory_mib']) for gpu in gpus):
        return {'attempted': False}
    result = cuda_driver_devices()
    result['attempted'] = True
    pci_query = command('nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits')
    result['pci_query'] = pci_query
    pci_by_uuid = {}
    if pci_query['returncode'] == 0:
        for row in csv.reader(io.StringIO(pci_query['output'])):
            if len(row) == 2:
                pci_by_uuid[row[0].strip()] = row[1].strip()
    for gpu in gpus:
        if re.fullmatch(r'\d+(?:\.\d+)?', gpu['memory_mib']):
            gpu['memory_mib_source'] = 'nvidia-smi'
            continue
        pci = pci_identity(pci_by_uuid.get(gpu['uuid'], ''))
        matching = [item for item in result['devices']
                    if pci is not None and pci_identity(item.get('pci_bus_id', '')) == pci]
        gpu['cuda_visible_device_ordinals'] = [item['ordinal'] for item in matching]
        if len(matching) == 1 and 'memory_bytes' in matching[0]:
            device = matching[0]
            gpu['memory_mib_nvidia_smi'] = gpu['memory_mib']
            gpu['memory_mib'] = device['memory_mib']
            gpu['memory_bytes'] = device['memory_bytes']
            gpu['memory_mib_source'] = 'cuda_driver.cuDeviceTotalMem_v2'
            gpu['memory_scope'] = 'single_visible_cuda_device_or_MIG_instance'
        else:
            gpu['memory_fallback_error'] = ('No unambiguous CUDA device with readable memory matches '
                                            'this physical GPU; inspect cuda_driver_query.devices.')
    return result


def read_text(path):
    try:
        return {'output': Path(path).read_text().strip()}
    except OSError as exc:
        return {'error': str(exc)}


def cgroup_v2_limits(proc_root=Path('/proc'), cgroup_root=None):
    """Read the current cgroup and visible ancestors, including inherited limits."""
    membership = read_text(proc_root / 'self/cgroup')
    mountinfo = read_text(proc_root / 'self/mountinfo')
    result = {'membership': membership, 'hierarchy': [],
              'scope': 'current_cgroup_and_visible_cgroup_v2_ancestors'}
    relative = next((line[3:] for line in membership.get('output', '').splitlines()
                     if line.startswith('0::')), None)
    if relative is None:
        result['error'] = 'No cgroup v2 membership found.'
        return result
    mount = None
    for line in mountinfo.get('output', '').splitlines():
        before, separator, after = line.partition(' - ')
        if separator and after.split()[0] == 'cgroup2':
            fields = before.split()
            unescape = lambda value: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)
            mount = (PurePosixPath(unescape(fields[3])), Path(unescape(fields[4])))
            break
    if mount is None:
        result['error'] = 'No visible cgroup v2 mount found.'
        result['mountinfo'] = mountinfo
        return result
    mount_root, mount_path = mount
    if cgroup_root is not None:  # Isolated test filesystem, not used by collection.
        mount_path = Path(cgroup_root)
    try:
        suffix = PurePosixPath(relative).relative_to(mount_root)
        if '..' in suffix.parts:
            raise ValueError('membership escapes visible hierarchy')
    except ValueError as exc:
        result['error'] = f'Cannot map cgroup membership into mount: {exc}'
        return result
    current = mount_path.joinpath(*suffix.parts)
    result.update(mount_point=str(mount_path), mount_root=str(mount_root), current_path=str(current))
    cpu_quotas, memory_limits = [], []
    while True:
        files = {name: read_text(current / name)
                 for name in ('cpu.max', 'memory.max', 'cpuset.cpus.effective')}
        result['hierarchy'].append({'path': str(current), 'files': files})
        try:
            quota, period = files['cpu.max'].get('output', '').split()
            if quota != 'max':
                cpu_quotas.append(int(quota) / int(period))
        except (ValueError, ZeroDivisionError):
            pass
        memory = files['memory.max'].get('output', '')
        if memory.isdigit():
            memory_limits.append(int(memory))
        if current == mount_path:
            break
        current = current.parent
    result['cpu_quota_cores'] = min(cpu_quotas) if cpu_quotas else None
    result['memory_max_bytes'] = min(memory_limits) if memory_limits else None
    result['cpuset_cpus_effective'] = result['hierarchy'][0]['files']['cpuset.cpus.effective'].get('output')
    return result


def collect():
    os_release = {}
    try:
        for line in open('/etc/os-release'):
            if '=' in line:
                key, value = line.strip().split('=', 1)
                os_release[key] = value.strip('"')
    except OSError:
        pass
    query = command('nvidia-smi', '--query-gpu=name,uuid,compute_cap,memory.total,driver_version',
                    '--format=csv,noheader,nounits')
    gpus = []
    if query['returncode'] == 0:
        for row in csv.reader(io.StringIO(query['output'])):
            if len(row) == 5:
                gpus.append(dict(zip(('name', 'uuid', 'compute_capability', 'memory_mib',
                                      'driver_version'), (v.strip() for v in row))))
    cuda_query = supplement_gpu_memory(gpus)
    return {
        'schema_version': 1,
        'collected_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'system': platform.system(), 'machine': platform.machine(),
        'hostname': platform.node(), 'kernel': platform.release(),
        'os_release': os_release,
        'glibc': command('getconf', 'GNU_LIBC_VERSION'),
        'cpu': command('lscpu', '-J'),
        'available_cpus': len(os.sched_getaffinity(0)),
        'memory': command('free', '-b'),
        'cgroup_v2': cgroup_v2_limits(),
        'disk': command('df', '-B1', '/', '/workspace', '/tmp'),
        'gpu_query': query, 'gpus': gpus,
        'cuda_driver_query': cuda_query,
        'nvidia_smi': command('nvidia-smi'),
        'tools': {name: {'path': shutil.which(name), 'version': command(name, '--version')}
                  for name in ('python3', 'nvcc', 'gcc', 'g++', 'cmake', 'ninja', 'rsync', 'tar', 'ldd')},
        'runtime_libraries': command('ldconfig', '-p'),
        'workspace_writable': os.path.isdir('/workspace') and os.access('/workspace', os.W_OK),
    }


if __name__ == '__main__':
    print(json.dumps(collect(), indent=2))
