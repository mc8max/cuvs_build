#!/usr/bin/env bash
# Run on the target host after deployment. No compilation or package installation.
set -euo pipefail
export LC_ALL=C

usage() {
  cat >&2 <<'EOF'
Usage: bash remote-run.sh BUNDLE_DIR probe
       bash remote-run.sh BUNDLE_DIR test [TEST_TARGET=UTIL_TEST] [GTEST_FILTER=*]
       bash remote-run.sh BUNDLE_DIR benchmark [BENCH_TARGET=CUVS_BRUTE_FORCE_ANN_BENCH]
EOF
  exit 2
}

[[ $# -ge 2 ]] || usage
bundle=$(cd -- "$1" && pwd -P)
mode=$2
shift 2
case "$mode" in
  probe) [[ $# -eq 0 ]] || usage ;;
  test) [[ $# -le 2 ]] || usage ;;
  benchmark) [[ $# -le 1 ]] || usage ;;
  *) usage ;;
esac
[[ -x "$bundle/run" ]] || { echo "Missing executable bundle launcher: $bundle/run" >&2; exit 1; }
mkdir -p "$bundle/results"
results=$(mktemp -d "$bundle/results/$(date -u +%Y%m%dT%H%M%SZ)-${mode}.XXXXXX")
exec > >(tee "$results/run.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$results/exit-status"; printf "Results: %s (exit %s)\n" "$results" "$status"' EXIT
printf 'Mode: %s\nBundle: %s\nResults: %s\n' "$mode" "$bundle" "$results"
# Make temporary files and default benchmark databases unique to this invocation.
cd "$results"

check_target() {
  [[ "$1" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "Invalid target name: $1" >&2; exit 2; }
  [[ -x "$bundle/bin/$1" ]] || { echo "Target is missing from bundle: bin/$1" >&2; exit 1; }
}

check_benchmark_report() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, mode = sys.argv[1:]
with open(path) as source:
    report = json.load(source)
rows = report.get("benchmarks", [])
if not rows:
    raise SystemExit(f"No benchmarks executed: {path}")
for row in rows:
    if row.get("error_occurred") or row.get("error_message") or row.get("skipped"):
        raise SystemExit(f"Benchmark failed or skipped: {row}")
    if row.get("run_type", "iteration") == "iteration" and row.get("iterations", 0) <= 0:
        raise SystemExit(f"Benchmark ran no iterations: {row}")
if mode == "search" and not any("Recall" in row for row in rows):
    raise SystemExit(f"Search did not report recall against the generated ground truth: {path}")
print(f"Validated {len(rows)} benchmark result(s): {path}")
PY
}

case "$mode" in
  probe)
    nvidia-smi
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
    export LD_LIBRARY_PATH="$bundle/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    shopt -s nullglob
    files=("$bundle"/bin/* "$bundle"/lib/*.so* "$bundle"/lib/ucx/*.so*)
    [[ ${#files[@]} -gt 0 ]] || { echo 'Bundle contains no binaries or libraries.' >&2; exit 1; }
    for file in "${files[@]}"; do
      [[ -f "$file" ]] || continue
      # Ignore non-ELF support files, but check every deployed ELF dependency.
      if ! python3 - "$file" <<'PY'
import sys
with open(sys.argv[1], "rb") as source:
    sys.exit(0 if source.read(4) == b"\x7fELF" else 1)
PY
      then
        continue
      fi
      printf '\nDependencies: %s\n' "$file"
      dependencies=$(ldd "$file" 2>&1) || { printf '%s\n' "$dependencies"; exit 1; }
      printf '%s\n' "$dependencies"
      if [[ "$dependencies" == *'not found'* ]]; then
        echo "Unresolved runtime dependency in $file" >&2
        exit 1
      fi
    done
    ;;
  test)
    target=${1:-UTIL_TEST}
    filter=${2:-*}
    check_target "$target"
    "$bundle/run" "bin/$target" "--gtest_filter=$filter" "--gtest_output=xml:$results/tests.xml"
    python3 - "$results/tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
tests = int(root.get("tests", "0"))
skipped = sum(1 for case in root.iter("testcase") if case.find("skipped") is not None)
if tests <= skipped or int(root.get("failures", "0")) or int(root.get("errors", "0")):
    raise SystemExit("Test report has failures or contains no executed, unskipped tests")
print(f"Validated GoogleTest report: {tests} tests, {skipped} skipped")
PY
    ;;
  benchmark)
    target=${1:-CUVS_BRUTE_FORCE_ANN_BENCH}
    check_target "$target"
    case "$target" in
      CUVS_BRUTE_FORCE_ANN_BENCH) algorithm=cuvs_brute_force ;;
      CUVS_CAGRA_ANN_BENCH) algorithm=cuvs_cagra ;;
      CUVS_IVF_FLAT_ANN_BENCH) algorithm=cuvs_ivf_flat ;;
      *) echo "No smoke configuration for benchmark target $target" >&2; exit 2 ;;
    esac
    python3 "$bundle/scripts/make_smoke_dataset.py" --output "$results/data" --algorithm "$algorithm"
    mkdir -p "$results/index"
    for phase in build search; do
      phase_args=("--$phase")
      # Google Benchmark recalibrates by invoking the build callback repeatedly.
      # Each invocation saves the index, so later invocations must overwrite it.
      if [[ "$phase" == build ]]; then phase_args+=(--force); fi
      "$bundle/run" "bin/$target" "${phase_args[@]}" \
        "--data_prefix=$results/data" "--index_prefix=$results/index" \
        --benchmark_min_time=0.1s --benchmark_repetitions=1 \
        --benchmark_out_format=json "--benchmark_out=$results/$phase.json" \
        "$results/data/benchmark.json" 2>&1 | tee "$results/$phase.log"
      check_benchmark_report "$results/$phase.json" "$phase"
    done
    ;;
esac
