"""Offline checks for CUDA memory fallback and container resource limits."""
import ctypes
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location('inventory', Path(__file__).resolve().parents[1] / 'scripts/inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class DriverInventoryTests(unittest.TestCase):
    def driver(self):
        def write_number(pointer, ctype, value):
            ctypes.cast(pointer, ctypes.POINTER(ctype)).contents.value = value
            return 0

        def write_text(buffer, length, device, text):
            buffer.value = text
            return 0

        return SimpleNamespace(
            cuInit=Mock(return_value=0),
            cuDeviceGetCount=Mock(side_effect=lambda ptr: write_number(ptr, ctypes.c_int, 1)),
            cuDeviceGet=Mock(side_effect=lambda ptr, ordinal: write_number(ptr, ctypes.c_int, ordinal)),
            cuDeviceTotalMem_v2=Mock(side_effect=lambda ptr, device: write_number(ptr, ctypes.c_size_t, 24192 * 1024 * 1024)),
            cuDeviceGetName=Mock(side_effect=lambda *args: write_text(*args, b'Test GPU MIG instance')),
            cuDeviceGetPCIBusId=Mock(side_effect=lambda *args: write_text(*args, b'0000:89:00.0')),
        )

    def test_ctypes_uses_size_t_and_preserves_raw_driver_statuses(self):
        driver = self.driver()
        with patch.object(inventory.ctypes, 'CDLL', return_value=driver):
            result = inventory.cuda_driver_devices()
        self.assertEqual(driver.cuDeviceTotalMem_v2.argtypes[0], ctypes.POINTER(ctypes.c_size_t))
        self.assertEqual(result['devices'][0]['memory_bytes'], 24192 * 1024 * 1024)
        self.assertEqual(result['devices'][0]['pci_bus_id'], '0000:89:00.0')
        self.assertEqual(result['devices'][0]['calls']['cuDeviceTotalMem_v2']['returncode'], 0)

    def test_driver_failure_is_recorded_without_invented_memory(self):
        driver = self.driver()
        driver.cuInit.return_value = 100
        with patch.object(inventory.ctypes, 'CDLL', return_value=driver):
            result = inventory.cuda_driver_devices()
        self.assertEqual(result['calls']['cuInit']['returncode'], 100)
        self.assertEqual(result['devices'], [])
        driver.cuDeviceGetCount.assert_not_called()
        with patch.object(inventory.ctypes, 'CDLL', side_effect=OSError('missing libcuda')):
            self.assertEqual(inventory.cuda_driver_devices()['error'], 'missing libcuda')

    def test_mig_memory_is_matched_by_pci_identity_with_source_and_scope(self):
        gpu = {'uuid': 'GPU-test', 'memory_mib': '[Insufficient Permissions]'}
        devices = {'devices': [{'ordinal': 3, 'pci_bus_id': '0000:89:00.0',
                               'memory_mib': '24192.0', 'memory_bytes': 24192 * 1024 * 1024}]}
        with patch.object(inventory, 'cuda_driver_devices', return_value=devices), \
                patch.object(inventory, 'command', return_value={'returncode': 0, 'output': 'GPU-test, 00000000:89:00.0'}):
            inventory.supplement_gpu_memory([gpu])
        self.assertEqual(gpu['memory_mib'], '24192.0')
        self.assertEqual(gpu['memory_mib_nvidia_smi'], '[Insufficient Permissions]')
        self.assertEqual(gpu['cuda_visible_device_ordinals'], [3])
        self.assertEqual(gpu['memory_mib_source'], 'cuda_driver.cuDeviceTotalMem_v2')
        self.assertIn('MIG', gpu['memory_scope'])

    def test_multiple_mig_instances_are_not_collapsed_to_one_guessed_size(self):
        gpu = {'uuid': 'GPU-test', 'memory_mib': '[Insufficient Permissions]'}
        devices = {'devices': [{'ordinal': ordinal, 'pci_bus_id': '0000:89:00.0',
                               'memory_mib': '24192.0', 'memory_bytes': 24192 * 1024 * 1024}
                              for ordinal in (0, 1)]}
        with patch.object(inventory, 'cuda_driver_devices', return_value=devices), \
                patch.object(inventory, 'command', return_value={'returncode': 0, 'output': 'GPU-test, 00000000:89:00.0'}):
            inventory.supplement_gpu_memory([gpu])
        self.assertEqual(gpu['memory_mib'], '[Insufficient Permissions]')
        self.assertIn('memory_fallback_error', gpu)


class CgroupInventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cuvs-cgroup-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.proc = self.root / 'proc'
        self.cgroup = self.root / 'cgroup'
        (self.proc / 'self').mkdir(parents=True)
        (self.cgroup / 'parent/child').mkdir(parents=True)
        (self.proc / 'self/cgroup').write_text('0::/parent/child\n')
        (self.proc / 'self/mountinfo').write_text('1 0 0:31 / /sys/fs/cgroup ro - cgroup2 cgroup2 rw\n')
        for directory in (self.cgroup, self.cgroup / 'parent', self.cgroup / 'parent/child'):
            (directory / 'cpu.max').write_text('max 100000\n')
            (directory / 'memory.max').write_text('max\n')
            (directory / 'cpuset.cpus.effective').write_text('0-127\n')

    def test_parent_limit_is_included_even_if_current_cgroup_is_unlimited(self):
        (self.cgroup / 'parent/cpu.max').write_text('250000 100000\n')
        (self.cgroup / 'parent/memory.max').write_text('8589934592\n')
        (self.cgroup / 'parent/child/cpuset.cpus.effective').write_text('0-3\n')
        result = inventory.cgroup_v2_limits(self.proc, self.cgroup)
        self.assertEqual(result['cpu_quota_cores'], 2.5)
        self.assertEqual(result['memory_max_bytes'], 8589934592)
        self.assertEqual(result['cpuset_cpus_effective'], '0-3')
        self.assertEqual(len(result['hierarchy']), 3)

    def test_unavailable_file_keeps_raw_error(self):
        (self.cgroup / 'parent/child/memory.max').unlink()
        result = inventory.cgroup_v2_limits(self.proc, self.cgroup)
        self.assertIn('error', result['hierarchy'][0]['files']['memory.max'])
        self.assertIsNone(result['memory_max_bytes'])

    def test_membership_cannot_escape_visible_mount(self):
        (self.proc / 'self/cgroup').write_text('0::/../outside\n')
        result = inventory.cgroup_v2_limits(self.proc, self.cgroup)
        self.assertIn('error', result)
        self.assertEqual(result['hierarchy'], [])


if __name__ == '__main__':
    unittest.main()
