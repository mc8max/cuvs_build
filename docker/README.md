# Docker builder

Linux amd64 build environment with CUDA 12.9, GCC 14, and a glibc 2.28 sysroot.
Run these commands from the repository root.

## Create and verify the image

```sh
docker build --platform linux/amd64 \
  -t cuvs-build:cuda12.9-gcc14 -f docker/Dockerfile docker

docker run --rm --platform linux/amd64 \
  --mount "type=bind,src=$PWD,dst=/workflow,readonly" \
  cuvs-build:cuda12.9-gcc14 \
  cuvs-check-toolchain --config /workflow/config/build.json
```

Keep the Dockerfile toolchain versions aligned with `config/build.json`.

## Build and inspect logs

Set `CUVS_SOURCE_DIR` in `.env`, then follow the [build instructions](../README.md).
`DOCKER_JOBS` overrides the configured compiler jobs. Omit `DOCKER_DETACH=1`
to follow output and receive the build's exit code.

Output stays in `cpp/build-runpod`. The source is otherwise mounted read only;
the compiler environment stays inside Docker at `/opt/cuvs-env`.

The container name and run directory are in `.state/docker/active-build.json`.
Each run saves `build.log` and `build.exit-code` under `.state/docker/runs/`;
exit code `0` means success. For live output:

```sh
docker logs --follow CONTAINER
```

## Complete development cycle on macOS

The regular `make build`, `make package`, `make publish`, and `make cycle`
commands automatically use the builder on macOS and ARM hosts. Run
`make inventory` once to collect remote GPU information, then `make cycle`.
`make check` also uses the image so its ELF and GNU rsync checks run on Linux.
Use `DOCKER_JOBS=2` to limit compiler parallelism, or `DOCKER_IMAGE=TAG` to select
another compatible builder image. The full cycle always waits for compilation.

These commands share `.state/` across containers and write local bundles into
`artifacts/` and downloaded reports into `results/`. The source checkout is read
only except for its dedicated `cpp/build-runpod` directory. The host `.conda-env`
is not used or created. The launcher refuses to start while a detached build or
another workflow step is running.

Remote steps mount `runpod.ssh` and `~/.ssh` read only, preserving Includes and
absolute identity paths. Docker Desktop's SSH agent socket is forwarded when
the host has `SSH_AUTH_SOCK`; unlock passphrase-protected keys with `ssh-add`
before running the cycle. Credentials are not mounted for build/package steps,
copied into the image, or included in bundles. Keep the pod hostname, port, and
identity settings in the usual host SSH configuration. `make ssh` opens the
host's interactive SSH client.
