# syntax=docker/dockerfile:1
# The base is the inspected Linux amd64 Miniforge image, pinned after pulling.
ARG BASE_IMAGE=quay.io/condaforge/miniforge3@sha256:790ab49984501fd514cd58ecbe2c9e3f764fccb60fd88e4ec27b59daef5b88ec
FROM ${BASE_IMAGE}

ARG CUDA_VERSION=12.9
ARG GCC_MAJOR=14
ARG GLIBC_BASELINE=2.28

LABEL org.opencontainers.image.title="cuVS compilation toolchain"
ENV CUVS_CUDA_VERSION=${CUDA_VERSION} \
    CUVS_GCC_MAJOR=${GCC_MAJOR} \
    CUVS_GLIBC_BASELINE=${GLIBC_BASELINE} \
    CUVS_ENV_PREFIX=/opt/cuvs-env \
    PYTHONDONTWRITEBYTECODE=1

# Use Conda's installer, matching the existing workflow's environment setup.
# Scope the virtual CUDA override to this solve; runtime checks belong on RunPod.
RUN CONDA_EXECUTE_THREADS=1 CONDA_VERIFY_THREADS=1 \
    CONDA_OVERRIDE_CUDA=${CUDA_VERSION} conda create --yes --solver libmamba \
      --prefix /opt/cuvs-env --override-channels \
      --channel rapidsai --channel conda-forge \
      'python=3.13' 'cmake>=4.0' ninja make patchelf \
      "gcc_linux-64=${GCC_MAJOR}" "gxx_linux-64=${GCC_MAJOR}" \
      "sysroot_linux-64=${GLIBC_BASELINE}" "cuda-version=${CUDA_VERSION}" \
      cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev cuda-nvtx-dev \
      cuda-profiler-api libcublas-dev libcufile-dev libcurand-dev \
      libcusolver-dev libcusparse-dev libnvjitlink-dev 'dlpack>=0.8,<1.0' \
      nccl libucxx ucx git binutils pkg-config rsync openssh \
    && conda list --prefix /opt/cuvs-env --explicit > /opt/cuvs-environment.lock.txt \
    && conda clean --all --yes

# The workflow's ELF fixture tests also invoke these conventional tool names.
RUN set -eu; for tool in gcc g++ readelf; do \
      if [ ! -e "/opt/cuvs-env/bin/${tool}" ]; then \
        test -x "/opt/cuvs-env/bin/x86_64-conda-linux-gnu-${tool}" \
        && ln -s "x86_64-conda-linux-gnu-${tool}" "/opt/cuvs-env/bin/${tool}"; \
      fi; \
    done

COPY entrypoint.sh /usr/local/bin/cuvs-build-env
COPY check-toolchain.py /usr/local/bin/cuvs-check-toolchain
RUN chmod 755 /usr/local/bin/cuvs-build-env /usr/local/bin/cuvs-check-toolchain

ENTRYPOINT ["/usr/local/bin/cuvs-build-env"]
CMD ["bash"]
