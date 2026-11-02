#!/usr/bin/env python3
"""Run ordinary workflow steps in the Linux builder on macOS or ARM hosts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from docker_build import docker, prepare
from workflow import ROOT, load_config


def command(config, image_id, image, step, architectures=None, jobs=None):
    """Use persistent workflow state and host output folders across Make steps."""
    state = ROOT / '.state'
    source, build, cache, translated = prepare(
        config, image_id, state, architectures, config['jobs'] if jobs is None else jobs,
        require_inventory=step not in ('inventory', 'test', 'benchmark', 'fetch', 'smoke'))
    inventory = (ROOT / config['inventory']).resolve()
    inventory.parent.mkdir(parents=True, exist_ok=True)
    ssh_config = (ROOT / config['ssh']['config']).resolve()
    remote = step in ('inventory', 'setup', 'publish', 'test', 'benchmark', 'fetch', 'cycle', 'smoke')
    if remote and not ssh_config.is_file():
        raise ValueError(f'SSH configuration missing: {ssh_config}; copy runpod.ssh.example first')
    translated['inventory'] = '/inventory/' + inventory.name
    translated['ssh'] = {**config['ssh'], 'config': '/ssh-config',
                         'known_hosts': '/workflow/.state/known_hosts'}
    (state / 'config.json').write_text(json.dumps(translated, indent=2) + '\n')
    args = ['run', '--rm', '--init', '--platform', 'linux/amd64',
            '--mount', f'type=bind,src={source},dst=/src,readonly',
            '--mount', f'type=bind,src={build},dst={translated["build_dir"]}',
            '--mount', f'type=bind,src={ROOT},dst=/workflow,readonly',
            '--mount', f'type=bind,src={state},dst=/workflow/.state',
            '--mount', f'type=bind,src={cache},dst=/cache/cpm',
            '--mount', f'type=bind,src={inventory.parent},dst=/inventory']
    if remote:
        args += ['--mount', f'type=bind,src={ssh_config},dst=/ssh-config,readonly']
    for folder in ('artifacts', 'results'):
        path = ROOT / folder
        path.mkdir(exist_ok=True)
        args += ['--mount', f'type=bind,src={path},dst=/workflow/{folder}']
    # Preserve ~/.ssh Includes and absolute IdentityFile paths. Credentials stay
    # on read-only mounts and are never copied into a bundle or builder image.
    ssh_dir = Path.home() / '.ssh'
    if remote and ssh_dir.is_dir():
        for target in sorted({'/root/.ssh', str(ssh_dir)}):
            args += ['--mount', f'type=bind,src={ssh_dir},dst={target},readonly']
    if remote and os.environ.get('SSH_AUTH_SOCK'):
        socket = ('/run/host-services/ssh-auth.sock' if platform.system() == 'Darwin'
                  else os.environ['SSH_AUTH_SOCK'])
        args += ['--mount', f'type=bind,src={socket},dst=/ssh-agent',
                 '--env', 'SSH_AUTH_SOCK=/ssh-agent']
    args += ['--workdir', '/workflow', image, 'python', '/workflow/scripts/workflow.py',
             '--config', '/workflow/.state/config.json', step]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/build.json')
    parser.add_argument('--image', default='cuvs-build:cuda12.9-gcc14')
    parser.add_argument('--architectures', nargs='+')
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--step', required=True,
                        choices=['inventory', 'doctor', 'setup', 'configure', 'build',
                                 'package', 'publish', 'test', 'benchmark', 'fetch', 'cycle', 'smoke'])
    args = parser.parse_args()
    state = ROOT / '.state'
    state.mkdir(exist_ok=True)
    with (state / 'docker-workflow.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another Docker workflow step is running; wait for it to finish')
        active_file = state / 'docker/active-build.json'
        if active_file.exists():
            active = json.loads(active_file.read_text())
            status = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', active['container']],
                                    capture_output=True, text=True)
            if status.returncode == 0 and status.stdout.strip() == 'true':
                raise ValueError(f'A cuVS build is already running: {active["container"]}')
        image_id = docker(['image', 'inspect', args.image, '--format', '{{.Id}}'],
                          capture_output=True).stdout.strip()
        config = load_config(args.config.resolve())
        docker(command(config, image_id, image_id, args.step, args.architectures, args.jobs))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(1)
