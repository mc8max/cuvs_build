#!/usr/bin/env python3
"""Run the configured native target set in the isolated Linux builder."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

from workflow import ROOT, load_config


def docker(args, **kwargs):
    return subprocess.run(['docker', *map(str, args)], check=True, text=True, **kwargs)


def prepare(config, image_id, state, architectures=None, jobs=1):
    """Resolve mounts and reject collisions before creating build state."""
    source = (ROOT / config['source_dir']).resolve()
    requested_build = ROOT / config['build_dir']
    build = requested_build.resolve()
    if not (source / 'cpp/CMakeLists.txt').is_file():
        raise ValueError(f'cuVS source checkout is missing: {source}')
    if build == source or source not in build.parents or requested_build.is_symlink():
        raise ValueError('build_dir must be a dedicated directory inside the source checkout.')
    if build == source / 'cpp/build':
        raise ValueError('Refusing to use the developer cpp/build directory.')
    if type(jobs) is not int or jobs < 1:
        raise ValueError('Docker jobs must be a positive integer.')
    if architectures is not None and (not architectures or any(
            not re.fullmatch(r'[1-9]\d{1,2}-real', arch) for arch in architectures)):
        raise ValueError('Use explicit architectures such as 120-real.')
    inventory = (ROOT / config['inventory']).resolve()
    if architectures is None and not inventory.is_file():
        raise ValueError('Measured inventory is missing. Supply --architectures for a build-only run.')
    marker = build / '.cuvs-docker-build.json'
    identity = {'source_dir': str(source), 'container_source_dir': '/src',
                'image_id': image_id, 'env_prefix': '/opt/cuvs-env'}
    if build.exists() and any(build.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError(f'Existing build is not owned by this checkout/image combination: {build}')
    state.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    (build / '.gitignore').write_text('*\n')
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    cache = ROOT / '.state/docker/cpm'
    cache.mkdir(parents=True, exist_ok=True)
    translated = dict(config)
    translated.update(source_dir='/src', build_dir='/src/' + str(build.relative_to(source)),
                      env_prefix='/opt/cuvs-env', cpm_cache='/cache/cpm', jobs=jobs,
                      builder_image_id=image_id)
    if architectures is not None:
        translated['compile_architectures'] = architectures
    if inventory.is_file():
        destination = state / 'remote-inventory.json'
        destination.write_bytes(inventory.read_bytes())
        translated['inventory'] = '/workflow/.state/remote-inventory.json'
    else:
        translated['inventory'] = '/workflow/.state/remote-inventory.json'
    (state / 'config.json').write_text(json.dumps(translated, indent=2) + '\n')
    (state / 'host-config.json').write_text(json.dumps(config, indent=2) + '\n')
    return source, build, cache, translated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/build.json')
    parser.add_argument('--image', default='cuvs-build:cuda12.9-gcc14')
    parser.add_argument('--architectures', nargs='+', help='Build-only SMs; publishing still requires real inventory')
    parser.add_argument('--jobs', type=int, help='Compiler jobs; defaults to config jobs')
    parser.add_argument('--detach', action='store_true', help='Keep compiling in Docker after this command exits')
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.jobs is None:
        args.jobs = config['jobs']
    state_root = ROOT / '.state/docker'
    active_file = state_root / 'active-build.json'
    if active_file.exists():
        active = json.loads(active_file.read_text())
        status = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', active['container']],
                                capture_output=True, text=True)
        if status.returncode == 0 and status.stdout.strip() == 'true':
            raise ValueError(f'A cuVS build is already running: {active["container"]}')
    image_id = docker(['image', 'inspect', args.image, '--format', '{{.Id}}'],
                      capture_output=True).stdout.strip()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    state = state_root / 'runs' / stamp
    source, build, cache, translated = prepare(config, image_id, state, args.architectures, args.jobs)
    suffix = hashlib.sha256(str(source).encode()).hexdigest()[:8]
    name = 'cuvs-native-' + suffix + '-' + stamp.lower()
    command = ['run', '--detach', '--init', '--platform', 'linux/amd64', '--name', name,
               '--label', 'cuvs_build.source=' + str(source),
               '--mount', f'type=bind,src={source},dst=/src,readonly',
               '--mount', f'type=bind,src={build},dst={translated["build_dir"]}',
               '--mount', f'type=bind,src={ROOT},dst=/workflow,readonly',
               '--mount', f'type=bind,src={state},dst=/workflow/.state',
               '--mount', f'type=bind,src={cache},dst=/cache/cpm',
               '--workdir', '/workflow', image_id,
               'bash', '/workflow/docker/run-build.sh', '/workflow/.state/config.json']
    container_id = docker(command, capture_output=True).stdout.strip()
    record = {'container': name, 'container_id': container_id, 'image_id': image_id,
              'state_dir': str(state), 'build_dir': str(build), 'jobs': args.jobs,
              'architectures': args.architectures, 'started_at': stamp}
    active_file.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    print('Logs: ' + shlex.join(['docker', 'logs', '--follow', name]), flush=True)
    if not args.detach:
        docker(['logs', '--follow', name])
        code = int(docker(['wait', name], capture_output=True).stdout.strip())
        raise SystemExit(code)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'error: {error}', file=sys.stderr)
        raise SystemExit(1)
