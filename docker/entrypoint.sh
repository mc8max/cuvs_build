#!/usr/bin/env bash
set -eo pipefail
# Conda's CUDA activation hook reads optional variables before initializing them.
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/cuvs-env
set -u
exec "$@"
