"""Check source isolation and build-only/deployment boundaries for Docker builds."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
with mock.patch.object(sys, 'path', [str(ROOT / 'scripts'), *sys.path]):
    spec = importlib.util.spec_from_file_location('docker_launcher', ROOT / 'scripts/docker_build.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
workflow = sys.modules['workflow']


class CompilationTargetTests(unittest.TestCase):
    def setUp(self):
        self.flow = workflow.Workflow.__new__(workflow.Workflow)
        self.flow.config = {'compile_architectures': ['120-real'], 'tests': {'profile': 'none'}}
        self.enterContext(mock.patch.object(workflow.platform, 'system', return_value='Linux'))
        self.enterContext(mock.patch.object(workflow.platform, 'machine', return_value='x86_64'))

    def test_explicit_compilation_does_not_manufacture_or_require_inventory(self):
        with mock.patch.object(self.flow, 'target', side_effect=AssertionError('must not read inventory')):
            self.assertEqual(self.flow.compilation_target(), ({}, ['120-real']))

    def test_invalid_architectures_cannot_be_used(self):
        for value in ([], '120-real', ['native'], ['120-virtual'], ['120;rm-real'], [120]):
            with self.subTest(value=value):
                self.flow.config['compile_architectures'] = value
                with self.assertRaises(ValueError):
                    self.flow.compilation_target()

    def test_deployment_still_requires_inventory_and_matching_architectures(self):
        with mock.patch.object(self.flow, 'target', side_effect=ValueError('Missing measured inventory')):
            with self.assertRaisesRegex(ValueError, 'measured inventory'):
                self.flow.package()
            with self.assertRaisesRegex(ValueError, 'measured inventory'):
                self.flow.publish()
        with mock.patch.object(self.flow, 'target', return_value=({}, ['86-real'])):
            with self.assertRaisesRegex(ValueError, 'differ'):
                self.flow.require_deployment_inventory()
        with mock.patch.object(self.flow, 'target', return_value=({}, ['120-real'])):
            self.flow.require_deployment_inventory()

    def test_image_change_invalidates_the_build_fingerprint(self):
        self.flow.config['builder_image_id'] = 'image-one'
        before = self.flow.fingerprint()
        self.flow.config['builder_image_id'] = 'image-two'
        self.assertNotEqual(before, self.flow.fingerprint())


class MountPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cuvs-docker-mounts-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        (self.source / 'cpp').mkdir(parents=True)
        (self.source / 'cpp/CMakeLists.txt').write_text('project(cuvs)\n')
        self.enterContext(mock.patch.object(launcher, 'ROOT', self.root))
        self.config = {'source_dir': 'source', 'build_dir': 'source/cpp/build-runpod',
                       'env_prefix': 'source/.conda-env', 'inventory': 'config/runpod.json',
                       'jobs': 4, 'tests': {'profile': 'all-native'}}
        self.state = self.root / '.state/docker/runs/one'

    def prepare(self, **kwargs):
        return launcher.prepare(self.config, 'test-image', self.state, ['120-real'], **kwargs)

    def test_outputs_stay_in_the_configured_checkout_and_environment_is_isolated(self):
        _, build, cache, translated = self.prepare()
        self.assertEqual(build, self.source / 'cpp/build-runpod')
        self.assertEqual(translated['build_dir'], '/src/cpp/build-runpod')
        self.assertEqual(translated['env_prefix'], '/opt/cuvs-env')
        self.assertEqual(translated['builder_image_id'], 'test-image')
        self.assertEqual(translated['tests'], {'profile': 'all-native'})
        self.assertTrue(cache.is_relative_to(self.root / '.state'))
        self.assertFalse((self.source / '.conda-env').exists())
        self.assertEqual((build / '.gitignore').read_text(), '*\n')

    def test_existing_developer_build_is_never_claimed(self):
        for name in ('build', 'build-runpod'):
            with self.subTest(name=name):
                directory = self.source / 'cpp' / name
                directory.mkdir()
                keep = directory / 'CMakeCache.txt'
                keep.write_text('developer state\n')
                self.config['build_dir'] = str(directory)
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(keep.read_text(), 'developer state\n')
                self.assertFalse((directory / '.cuvs-docker-build.json').exists())

    def test_symlink_cannot_redirect_output(self):
        other = self.source / 'cpp/other-build'
        other.mkdir()
        (self.source / 'cpp/build-runpod').symlink_to(other, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(list(other.iterdir()), [])

    def test_owned_build_reuses_only_the_same_image_identity(self):
        self.prepare()
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'not owned'):
            launcher.prepare(self.config, 'different-image', self.state, ['120-real'])


if __name__ == '__main__':
    unittest.main()
