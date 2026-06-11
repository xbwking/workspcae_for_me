#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash install_into_verl.sh /path/to/verl"
  exit 2
fi

VERL_ROOT="$1"
if [[ ! -d "${VERL_ROOT}" ]]; then
  echo "verl root does not exist: ${VERL_ROOT}"
  exit 2
fi
if [[ ! -d "${VERL_ROOT}/verl" || ! -d "${VERL_ROOT}/scripts" ]]; then
  echo "target does not look like a verl repo root: ${VERL_ROOT}"
  exit 2
fi

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -R "${PACKAGE_DIR}/files/." "${VERL_ROOT}/"

echo "Installed Ascend benchmark runtime package into: ${VERL_ROOT}"
echo "Next:"
echo "  cd ${VERL_ROOT}"
echo "  MODEL_PATH=/path/to/model TRAIN_FILES=/path/to/train.parquet VAL_FILES=/path/to/test.parquet OUTPUT_DIR=outputs/ascend_timing_breakdown/run1 bash tests/special_npu/run_ascend_timing_breakdown_bench.sh"
