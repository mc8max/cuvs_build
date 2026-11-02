"""Exercise release reuse with real local rsync, without making SSH connections."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    'workflow', Path(__file__).resolve().parents[1] / 'scripts/workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


@unittest.skipUnless(shutil.which('rsync'), 'rsync is required for release reuse checks')
class PublishReuseTests(unittest.TestCase):
    def check_reuse(self, retry, use_cache=False):
        with tempfile.TemporaryDirectory(prefix='cuvs-publish-test-') as temporary:
            root = Path(temporary)
            previous_release = '20261031T110000000000Z'
            release = '20261031T120000000000Z'
            config = {
                'ssh': {'host': 'unused-host'},
                'remote_root': str(root / 'remote/cuvs-build'),
                'toolchain': {'glibc_baseline': '2.28', 'minimum_driver': '575.57.08'},
            }
            if use_cache:
                config['library_cache'] = {'mount': str(root), 'path': str(root / 'volume/cache')}
            inventory = {
                'schema_version': 1, 'system': 'Linux', 'machine': 'x86_64',
                'glibc': {'output': 'glibc 2.39'},
                'gpus': [{'compute_capability': '12.0', 'driver_version': '595.91.07'}],
                'tools': {name: {'path': '/usr/bin/' + name}
                          for name in ('python3', 'rsync', 'tar', 'ldd')},
            }
            config_path = root / 'build-config.json'
            config_path.write_text(json.dumps(config))
            bundle = root / 'artifacts' / release
            previous = Path(config['remote_root']) / 'releases' / previous_release
            destination = previous.parent / release
            (bundle / 'lib').mkdir(parents=True)
            (previous / 'lib').mkdir(parents=True)
            old_bytes, new_bytes = b'old ELF contents', b'new ELF contents'
            self.assertEqual(len(old_bytes), len(new_bytes))
            timestamp_ns = 1700000000123456789
            for directory, payload in ((previous, old_bytes), (bundle, new_bytes)):
                library = directory / 'lib/libfixture.so.1'
                library.write_bytes(payload)
                os.utime(library, ns=(timestamp_ns, timestamp_ns))
                os.link(library, directory / 'lib/libfixture.so')
                unchanged = directory / 'lib/libunchanged.so'
                unchanged.write_bytes(b'unchanged library')
                os.utime(unchanged, ns=(timestamp_ns, timestamp_ns))
            previous_inode = (previous / 'lib/libfixture.so.1').stat().st_ino
            if retry:
                # Reproduce the incomplete upload: the destination already
                # incorrectly shares equal-size/equal-mtime files with the old release.
                (destination / 'lib').mkdir(parents=True)
                for name in ('libfixture.so', 'libfixture.so.1'):
                    os.link(previous / 'lib' / name, destination / 'lib' / name)
            (bundle / 'build-config.json').write_text(json.dumps(config))
            (bundle / 'remote-inventory.json').write_text(json.dumps(inventory))
            manifest = {
                'format': 1, 'max_required_glibc': '2.27',
                'files': {str(path.relative_to(bundle)): {
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'size': path.stat().st_size,
                } for path in bundle.rglob('*') if path.is_file()},
            }
            (bundle / 'manifest.json').write_text(json.dumps(manifest))

            library_uploads = []

            def local_transfer(args, **kwargs):
                command = list(map(str, args))
                self.assertEqual(command[0], 'rsync')
                index = command.index('-e')
                del command[index:index + 2]
                self.assertTrue(command[-1].startswith('unused-host:'))
                command[-1] = command[-1].removeprefix('unused-host:')
                if Path(command[-2].rstrip('/')).name.startswith('library-upload-'):
                    library_uploads.append(sum(path.stat().st_size for path in Path(command[-2]).iterdir()))
                return subprocess.run(command, check=True, capture_output=True, text=True)

            def local_remote(args, **kwargs):
                if args[:2] == ['mkdir', '-p']:
                    Path(args[2]).mkdir(parents=True, exist_ok=True)
                elif args[:2] == ['python3', '-c']:
                    # Run the production SHA-256 verification against the actual transfer.
                    return subprocess.run(args, check=True, capture_output=True, text=True)
                elif args[:2] == ['python3', '-']:
                    # Test directories stand in for a mounted network volume.
                    script = 'import os\nos.path.ismount = lambda path: True\n' + kwargs['input']
                    return subprocess.run(args, input=script, check=True, capture_output=True, text=True)
                elif args[0] == 'bash' and args[-1] == 'probe':
                    return subprocess.CompletedProcess(args, 0)
                else:
                    self.fail(f'Unexpected remote operation: {args}')

            with mock.patch.object(workflow, 'ROOT', root), \
                    mock.patch.object(workflow.platform, 'machine', return_value='x86_64'):
                flow = workflow.Workflow(config_path)
                (flow.state / 'latest-bundle').write_text(release)
                (flow.state / 'published-release').write_text(previous_release)
                with mock.patch.object(flow, 'inventory', return_value=inventory), \
                        mock.patch.object(flow, 'ssh', return_value=['ssh']), \
                        mock.patch.object(flow, 'remote', side_effect=local_remote), \
                        mock.patch.object(workflow, 'run', side_effect=local_transfer), \
                        contextlib.redirect_stdout(io.StringIO()):
                    flow.publish()
                    if use_cache:
                        self.assertEqual(library_uploads, [len(new_bytes)])
                        # A new pod/client can reuse the volume with no previous release or local stamp.
                        shutil.rmtree(destination)
                        (flow.state / 'published-release').unlink()
                        flow.publish()
                        self.assertEqual(library_uploads, [len(new_bytes)])
                self.assertEqual((flow.state / 'published-release').read_text().strip(), release)
            for name in ('libfixture.so', 'libfixture.so.1'):
                self.assertEqual((destination / 'lib' / name).read_bytes(), new_bytes)
                self.assertEqual((previous / 'lib' / name).read_bytes(), old_bytes)
                self.assertEqual((previous / 'lib' / name).stat().st_ino, previous_inode)
                self.assertNotEqual((destination / 'lib' / name).stat().st_ino, previous_inode)
            self.assertEqual((destination / 'lib/libfixture.so').stat().st_ino,
                             (destination / 'lib/libfixture.so.1').stat().st_ino)
            self.assertEqual((destination / 'lib/libunchanged.so').stat().st_ino,
                             (previous / 'lib/libunchanged.so').stat().st_ino)

    def test_new_release_reuses_only_identical_contents(self):
        self.check_reuse(retry=False)

    def test_retry_replaces_incorrect_hardlinks_without_mutating_previous_release(self):
        self.check_reuse(retry=True)

    def test_cache_uploads_only_missing_objects_and_reuses_without_local_release_state(self):
        self.check_reuse(retry=False, use_cache=True)

    def test_cache_retry_replaces_stale_release_libraries(self):
        self.check_reuse(retry=True, use_cache=True)


if __name__ == '__main__':
    unittest.main()
