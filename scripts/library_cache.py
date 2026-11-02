#!/usr/bin/env python3
"""Content-addressed runtime library storage on a mounted network volume.

This module is also sent over SSH to run with the remote Python standard library.
"""

import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile


def library_files(manifest):
    if manifest.get('format') != 1 or not manifest.get('files'):
        raise ValueError('Missing or unsupported bundle manifest')
    libraries = {}
    for name, entry in manifest['files'].items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name:
            raise ValueError(f'Invalid bundle path: {name}')
        if path.parts[0] != 'lib':
            continue
        if (len(path.parts) < 2 or not re.fullmatch(r'[a-f0-9]{64}', entry['sha256']) or
                type(entry['size']) is not int or entry['size'] < 0):
            raise ValueError(f'Invalid library manifest entry: {name}')
        libraries[name] = entry
    return libraries


def valid(path, entry):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == entry['sha256']


def cache_root(config):
    mount, root = Path(config['mount']), Path(config['path'])
    if (not mount.is_absolute() or not root.is_absolute() or
            '..' in mount.parts or '..' in root.parts or mount == Path('/') or
            mount.resolve() not in root.resolve().parents):
        raise ValueError('library_cache.path must be a dedicated directory inside library_cache.mount')
    if not os.path.ismount(mount):
        raise ValueError(f'Library cache mount is absent: {mount}; attach the network volume before publishing')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'objects').mkdir(exist_ok=True)
    return root


def materialize(source, target):
    """Install a hardlink or copy atomically, without modifying existing inodes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.library-', dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(config, manifest, seed=None):
    libraries = library_files(manifest)
    root = cache_root(config)
    objects = {entry['sha256']: entry for entry in libraries.values()}
    missing = []
    for key, entry in objects.items():
        target = root / 'objects' / key
        if valid(target, entry):
            continue
        # Migrate the previous release on first use, avoiding a second initial upload.
        candidates = (Path(seed) / name for name, item in libraries.items() if item['sha256'] == key) if seed else ()
        source = next((path for path in candidates if valid(path, entry)), None)
        if source:
            materialize(source, target)
        else:
            missing.append(key)
    incoming = Path(tempfile.mkdtemp(prefix='incoming-', dir=root))
    return {'incoming': str(incoming), 'missing': missing,
            'upload_bytes': sum(objects[key]['size'] for key in missing),
            'reused_bytes': sum(entry['size'] for key, entry in objects.items() if key not in missing)}


def assemble(config, manifest, destination, incoming):
    libraries = library_files(manifest)
    root = cache_root(config)
    incoming = Path(incoming)
    if incoming.parent != root or not incoming.name.startswith('incoming-') or incoming.is_symlink():
        raise ValueError('Invalid library cache upload directory')
    objects = {entry['sha256']: entry for entry in libraries.values()}
    # Validate every uploaded object before exposing any of them to releases.
    for key, entry in objects.items():
        upload = incoming / key
        if upload.exists() and not valid(upload, entry):
            raise ValueError(f'Library cache upload integrity check failed: {key}')
        if not upload.exists() and not valid(root / 'objects' / key, entry):
            raise ValueError(f'Missing or corrupt library cache object: {key}')
    for key in objects:
        upload = incoming / key
        if upload.exists():
            # Atomic replacement never writes through hardlinks held by old releases.
            os.replace(upload, root / 'objects' / key)
    destination = Path(destination)
    for name, entry in libraries.items():
        target = destination / name
        # Hardlinks preserve the existing $ORIGIN layout (including UCX modules).
        # Copy when releases live on a filesystem other than the network volume.
        materialize(root / 'objects' / entry['sha256'], target)
    shutil.rmtree(incoming)
    return {'libraries': len(libraries), 'objects': len(objects)}


def main():
    action, raw = sys.argv[1:]
    request = json.loads(raw)
    if action == 'prepare':
        result = prepare(request['config'], request['manifest'], request.get('seed'))
    elif action == 'assemble':
        result = assemble(request['config'], request['manifest'],
                          request['destination'], request['incoming'])
    else:
        raise ValueError(f'Unknown library cache action: {action}')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
