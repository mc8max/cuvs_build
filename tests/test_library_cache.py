"""Exercise the persistent library cache with actual files and integrity failures."""

import errno
import hashlib
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    'library_cache', Path(__file__).resolve().parents[1] / 'scripts/library_cache.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class LibraryCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cuvs-cache-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {'mount': str(self.root), 'path': str(self.root / 'cache')}
        self.enterContext(mock.patch.object(cache.os.path, 'ismount', return_value=True))
        self.payload = b'relocated library'
        self.entry = {'sha256': hashlib.sha256(self.payload).hexdigest(), 'size': len(self.payload)}
        self.manifest = {'format': 1, 'files': {
            'lib/libtest.so': self.entry, 'lib/libtest.so.1': self.entry,
            'lib/ucx/libtransport.so': self.entry,
            'bin/test': {'sha256': 'a' * 64, 'size': 1},
        }}

    def populate(self, destination=None):
        prepared = cache.prepare(self.config, self.manifest)
        for key in prepared['missing']:
            (Path(prepared['incoming']) / key).write_bytes(self.payload)
        destination = destination or self.root / 'release'
        cache.assemble(self.config, self.manifest, destination, prepared['incoming'])
        return prepared, destination

    def test_reuse_survives_release_removal_and_deduplicates_aliases(self):
        first, destination = self.populate()
        self.assertEqual(first['upload_bytes'], len(self.payload))
        self.assertEqual(len(first['missing']), 1)
        self.assertEqual((destination / 'lib/libtest.so').stat().st_ino,
                         (destination / 'lib/ucx/libtransport.so').stat().st_ino)
        shutil.rmtree(destination)
        second, destination = self.populate(self.root / 'replacement-pod-release')
        self.assertEqual(second['missing'], [])
        self.assertEqual(second['upload_bytes'], 0)
        self.assertEqual(second['reused_bytes'], len(self.payload))
        self.assertEqual((destination / 'lib/libtest.so.1').read_bytes(), self.payload)
        self.assertFalse((destination / 'lib/libtest.so').is_symlink())

    def test_changed_bytes_same_size_get_a_new_object_and_preserve_old_release(self):
        _, old = self.populate()
        self.payload = b'RELOCATED library'
        self.assertEqual(len(self.payload), self.entry['size'])
        self.entry['sha256'] = hashlib.sha256(self.payload).hexdigest()
        prepared, new = self.populate(self.root / 'new-release')
        self.assertEqual(len(prepared['missing']), 1)
        self.assertEqual((old / 'lib/libtest.so').read_bytes(), b'relocated library')
        self.assertEqual((new / 'lib/libtest.so').read_bytes(), self.payload)

    def test_corrupt_cache_is_reuploaded_without_writing_through_old_hardlinks(self):
        _, old = self.populate()
        obj = Path(self.config['path']) / 'objects' / self.entry['sha256']
        obj.write_bytes(b'corrupted library')
        prepared, new = self.populate(self.root / 'repaired-release')
        self.assertEqual(len(prepared['missing']), 1)
        self.assertEqual((new / 'lib/libtest.so').read_bytes(), self.payload)
        self.assertEqual((old / 'lib/libtest.so').read_bytes(), b'corrupted library')

    def test_incomplete_or_bad_upload_is_not_installed_and_retry_succeeds(self):
        prepared = cache.prepare(self.config, self.manifest)
        with self.assertRaisesRegex(ValueError, 'Missing or corrupt'):
            cache.assemble(self.config, self.manifest, self.root / 'release', prepared['incoming'])
        (Path(prepared['incoming']) / self.entry['sha256']).write_bytes(b'incomplete')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            cache.assemble(self.config, self.manifest, self.root / 'release', prepared['incoming'])
        self.assertFalse((self.root / 'release').exists())
        self.populate()

    def test_cross_filesystem_release_copies_libraries(self):
        with mock.patch.object(cache.os, 'link', side_effect=OSError(errno.EXDEV, 'cross-device')):
            _, destination = self.populate()
        self.assertEqual((destination / 'lib/ucx/libtransport.so').read_bytes(), self.payload)

    def test_first_publish_seeds_cache_from_verified_existing_release(self):
        previous = self.root / 'previous'
        (previous / 'lib').mkdir(parents=True)
        (previous / 'lib/libtest.so').write_bytes(self.payload)
        prepared = cache.prepare(self.config, self.manifest, previous)
        self.assertEqual(prepared['missing'], [])
        self.assertEqual(prepared['upload_bytes'], 0)
        cache.assemble(self.config, self.manifest, self.root / 'release', prepared['incoming'])
        self.assertEqual((self.root / 'release/lib/libtest.so').read_bytes(), self.payload)

    def test_absent_mount_fails_before_creating_cache(self):
        with mock.patch.object(cache.os.path, 'ismount', return_value=False):
            with self.assertRaisesRegex(ValueError, 'mount is absent'):
                cache.prepare(self.config, self.manifest)
        self.assertFalse(Path(self.config['path']).exists())

    def test_reject_paths_outside_mount_and_manifest_traversal(self):
        for path in (self.root.parent, self.root, self.root / '../escape'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                cache.prepare({**self.config, 'path': str(path)}, self.manifest)
        for path in ('lib/../../escape', '/lib/escape', 'lib//escape'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                cache.prepare(self.config, {'format': 1, 'files': {path: self.entry}})


if __name__ == '__main__':
    unittest.main()
