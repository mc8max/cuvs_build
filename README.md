# cuVS build and RunPod workflow

Build cuVS locally, run tests and a smoke benchmark on RunPod, and download
results. No local GPU is required. The cuVS checkout uses a dedicated
`cpp/build-runpod` directory.

## Setup

- **macOS / ARM:** Python 3, Make, Docker, and the [builder image](docker/README.md).
  Make commands automatically use Linux amd64 Docker.
- **Linux x86_64:** Python 3, Conda, Make, SSH, and GNU rsync.
- **RunPod:** Linux x86_64 with an NVIDIA GPU, driver, Python 3, and rsync.

```sh
cp .env.example .env
cp runpod.ssh.example runpod.ssh
```

Set `CUVS_SOURCE_DIR` in `.env` and the pod address, port, and identity in
`runpod.ssh`. Unlock passphrase-protected keys with `ssh-add`, then run:

```sh
make setup
make cycle
```

## Development

After editing cuVS, run `make cycle`:
**build → package → upload → test → benchmark → download results**.
Reports are saved in `results/`.

| Command | Purpose |
| --- | --- |
| `make build` | Compile selected targets. |
| `make publish` | Build, package, and upload. |
| `make test` / `make benchmark` | Run the last published release. |
| `make fetch` | Download reports. |
| `make inventory` | Refresh hardware information after changing pods. |
| `make check` | Check the workflow without SSH or a GPU. |
| `make help` | Show all commands. |

Edit `config/build.json` for build and test settings. The default runs quick
smoke tests; use `make cycle CONFIG=config/build-all-native.json` for the full
profile. See [Docker options and logs](docker/README.md) for detached builds
and compiler job limits.

## Library cache

Attach a network volume at `/workspace`. Libraries are reused from
`/workspace/cuvs-library-cache`; only missing or changed content is uploaded.
Keep this cache when removing old releases, and attach the same volume to
replacement pods. See [cache configuration and maintenance](docs/library-cache.md).
