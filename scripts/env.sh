# Source me: puts Vitis HLS (vitis-run) and Vivado on PATH and activates the project venv.
#   source scripts/env.sh            (default install: /tools/Xilinx/2026.1.1)
#   XILINX_ROOT=/opt/Xilinx/2025.2 source scripts/env.sh
XILINX_ROOT="${XILINX_ROOT:-/tools/Xilinx/2026.1.1}"
source "${XILINX_ROOT}/Vitis/settings64.sh"
source "${XILINX_ROOT}/Vivado/settings64.sh" 2>/dev/null || true
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${_here}/.venv/bin/activate"
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=2
