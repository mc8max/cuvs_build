"""Verify the macOS Make cycle uses persistent Linux state and scoped mounts."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
with mock.patch.object(sys, 'path', [str(ROOT / 'scripts'), *sys.path]):
    spec = importlib.util.spec_from_file_location('docker_workflow', ROOT / 'scripts/docker_workflow.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)


class DockerWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cuvs-docker-workflow-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / 'source'
        (self.source / 'cpp').mkdir(parents=True)
        (self.source / 'cpp/CMakeLists.txt').write_text('project(cuvs)\n')
        (self.root / 'config').mkdir()
        (self.root / 'config/inventory.json').write_text('{}')
        (self.root / 'runpod.ssh').write_text('Host test-pod\n HostName example.invalid\n')
        self.home = self.root / 'home'
        (self.home / '.ssh').mkdir(parents=True)
        self.config = {
            'source_dir': str(self.source), 'build_dir': str(self.source / 'cpp/build-runpod'),
            'env_prefix': str(self.source / '.conda-env'), 'inventory': 'config/inventory.json',
            'jobs': 6, 'ssh': {'config': 'runpod.ssh', 'host': 'test-pod', 'known_hosts': '.state/known_hosts'},
            'library_cache': {'mount': '/workspace', 'path': '/workspace/cuvs-library-cache'},
        }
        self.enterContext(mock.patch.object(launcher, 'ROOT', self.root))
        self.enterContext(mock.patch.object(sys.modules['docker_build'], 'ROOT', self.root))
        self.enterContext(mock.patch.object(launcher.Path, 'home', return_value=self.home))
        self.enterContext(mock.patch.dict(os.environ, {}, clear=True))

    def command(self, step, **kwargs):
        return launcher.command(self.config, 'image-one', 'image-one', step, **kwargs)

    def translated(self):
        return json.loads((self.root / '.state/config.json').read_text())

    def test_build_package_and_publish_share_configuration_and_publication_state(self):
        build = self.command('build')
        config = self.translated()
        stamp = self.root / '.state/configured.json'
        stamp.write_text('preserve build fingerprint')
        package = self.command('package')
        self.assertEqual(self.translated(), config)
        publish = self.command('publish')
        self.assertEqual(self.translated(), config)
        self.assertEqual(stamp.read_text(), 'preserve build fingerprint')
        for args, step in ((build, 'build'), (package, 'package'), (publish, 'publish')):
            self.assertEqual(args[-1], step)
            self.assertIn(f'type=bind,src={self.root / ".state"},dst=/workflow/.state', args)
            self.assertIn(f'type=bind,src={self.root / "artifacts"},dst=/workflow/artifacts', args)
            self.assertIn(f'type=bind,src={self.root / "results"},dst=/workflow/results', args)
        self.assertEqual(config['env_prefix'], '/opt/cuvs-env')
        self.assertEqual(config['inventory'], '/inventory/inventory.json')
        self.assertEqual(config['library_cache'], self.config['library_cache'])
        self.assertFalse((self.source / '.conda-env').exists())

    def test_inventory_bootstraps_without_existing_inventory_or_invented_architectures(self):
        (self.root / 'config/inventory.json').unlink()
        self.command('inventory')
        self.assertEqual(self.translated()['inventory'], '/inventory/inventory.json')
        self.assertNotIn('compile_architectures', self.translated())

    def test_build_and_package_do_not_mount_ssh_credentials(self):
        (self.root / 'runpod.ssh').unlink()
        for step in ('build', 'package'):
            with self.subTest(step=step):
                args = self.command(step)
                self.assertFalse(any('dst=/ssh-config' in arg or 'dst=/root/.ssh' in arg for arg in args))

    def test_remote_ssh_paths_and_macos_agent_are_available_without_copying_keys(self):
        with mock.patch.object(launcher.platform, 'system', return_value='Darwin'), \
                mock.patch.dict(os.environ, {'SSH_AUTH_SOCK': '/private/host-agent.sock'}):
            args = self.command('publish')
        self.assertIn(f'type=bind,src={self.home / ".ssh"},dst=/root/.ssh,readonly', args)
        self.assertIn(f'type=bind,src={self.home / ".ssh"},dst={self.home / ".ssh"},readonly', args)
        self.assertIn('type=bind,src=/run/host-services/ssh-auth.sock,dst=/ssh-agent', args)
        self.assertEqual(self.translated()['ssh']['config'], '/ssh-config')

    def test_job_override_and_build_only_architectures_are_preserved(self):
        self.command('build', architectures=['120-real'], jobs=2)
        self.assertEqual(self.translated()['jobs'], 2)
        self.assertEqual(self.translated()['compile_architectures'], ['120-real'])

    def test_busy_workflow_rejects_another_step_before_starting_docker(self):
        state = self.root / '.state'
        state.mkdir()
        with (state / 'docker-workflow.lock').open('w') as lock:
            launcher.fcntl.flock(lock, launcher.fcntl.LOCK_EX | launcher.fcntl.LOCK_NB)
            with mock.patch.object(sys, 'argv', ['docker_workflow.py', '--step', 'build']), \
                    mock.patch.object(launcher, 'docker') as docker:
                with self.assertRaisesRegex(ValueError, 'Another Docker workflow'):
                    launcher.main()
                docker.assert_not_called()


class MakeRoutingTests(unittest.TestCase):
    def dry_run(self, platform, target='cycle'):
        return subprocess.run(['make', '-n', target, 'HOST_PLATFORM=' + platform], cwd=ROOT,
                              text=True, capture_output=True, check=True).stdout

    def test_macos_cycle_routes_all_phases_through_docker(self):
        output = self.dry_run('Darwin-arm64')
        self.assertEqual(output.count('scripts/docker_workflow.py'), 4)
        for step in ('build', 'package', 'publish', 'cycle'):
            self.assertIn('--step ' + step, output)
        self.assertNotIn('scripts/workflow.py', output)

    def test_native_linux_cycle_and_interactive_ssh_keep_native_driver(self):
        self.assertNotIn('scripts/docker_workflow.py', self.dry_run('Linux-x86_64'))
        self.assertIn('scripts/workflow.py', self.dry_run('Darwin-arm64', 'ssh'))

    def test_macos_check_uses_linux_tools_without_ssh_mounts(self):
        output = self.dry_run('Darwin-arm64', 'check')
        self.assertIn('linux/amd64', output)
        self.assertIn('readonly', output)
        self.assertNotIn('ssh-agent', output)


class NativePlatformGuardTests(unittest.TestCase):
    def test_macos_native_steps_fail_before_environment_or_configuration_writes(self):
        workflow = sys.modules['workflow']
        for step in ('setup', 'doctor', 'configure', 'build', 'package', 'smoke'):
            with self.subTest(step=step), \
                    mock.patch.object(sys, 'argv', ['workflow.py', step]), \
                    mock.patch.object(workflow.platform, 'system', return_value='Darwin'), \
                    mock.patch.object(workflow, 'Workflow') as flow:
                with self.assertRaisesRegex(ValueError, 'Use make ' + step):
                    workflow.main()
                flow.assert_not_called()


if __name__ == '__main__':
    unittest.main()
