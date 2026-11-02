#!/usr/bin/env bash
set -uo pipefail
python /workflow/scripts/workflow.py --config "$1" build 2>&1 | tee /workflow/.state/build.log
status=$?
printf '%s\n' "$status" > /workflow/.state/build.exit-code
exit "$status"
