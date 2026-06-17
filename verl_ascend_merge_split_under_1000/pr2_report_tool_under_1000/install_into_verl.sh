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
if [[ ! -d "${VERL_ROOT}/scripts" ]]; then
  echo "target does not look like a verl repo root: ${VERL_ROOT}"
  exit 2
fi

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -R "${TOOL_DIR}/scripts/." "${VERL_ROOT}/scripts/"

echo "Installed report tool into: ${VERL_ROOT}/scripts"
echo "Example:"
echo "  cd ${VERL_ROOT}"
echo "  python3 scripts/report_ascend_verl_timing.py --run-dir outputs/ascend_timing_breakdown/run1"
