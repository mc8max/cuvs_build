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
