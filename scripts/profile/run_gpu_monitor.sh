#!/usr/bin/env bash
# High-frequency GPU utilization logging for correlating with training logs.
#
# Tier A (preferred): DCGM dmon (dcgmi) — fields 203 GPUTL, 204 MCUTL, 155 POWER, 150 TMPTR
# Fallback: nvidia-smi CSV loop with --loop-ms
#
# Usage:
#   # Log only (run training in another terminal), Ctrl+C to stop:
#   bash scripts/profile/run_gpu_monitor.sh --gpus 0,1 --interval-ms 100
#
#   # Monitor in background, run one command, then stop monitor:
#   bash scripts/profile/run_gpu_monitor.sh --gpus 0,1 --interval-ms 100 -- python scripts/train_weighted_dsm.py --help
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACTS="${REPO_ROOT}/artifacts/evaluation"
GPUS="0"
INTERVAL_MS="100"
OUT_DIR=""
MODE="auto" # auto | dcgmi | nvidia
CMD=()

usage() {
  cat <<'EOF'
run_gpu_monitor.sh — high-rate GPU metrics for utilization workflows

Options:
  -h, --help              Show this help
  --gpus LIST             Comma-separated GPU indices (default: 0)
  --interval-ms N         Sample interval in milliseconds (default: 100)
  --out-dir DIR           Directory for log files (default: artifacts/evaluation)
  --mode MODE             auto | dcgmi | nvidia  (default: auto = dcgmi if available)

Without "--": runs the monitor in the foreground until SIGINT/SIGTERM.

With "-- CMD...": starts the monitor in the background, runs CMD, waits for CMD,
                 then stops the monitor (SIGINT to dcgmi/smi loop).

Log file is printed on startup. Correlate wall-clock timestamps with training_log.csv.

DCGM field IDs: 203 gpu_utilization, 204 mem_copy_utilization, 155 power_usage, 150 gpu_temp
EOF
}

log_header() {
  local f="$1"
  local ts
  ts="$(date -Iseconds 2>/dev/null || date)"
  {
    echo "# gpu_monitor started_at=${ts}"
    echo "# repo_root=${REPO_ROOT}"
    echo "# gpus=${GPUS} interval_ms=${INTERVAL_MS} mode=${MODE_EFFECTIVE}"
    echo "# dcgmi: columns are entity_id + field values in -e order (203 204 155 150)"
    echo "# nvidia-smi: CSV columns from --query-gpu"
  } >>"$f"
}

run_dcgmi() {
  local logf="$1"
  local delay="$2"
  # -c 0 = infinite per dcgmi dmon default when omitted; we omit -c for infinite
  dcgmi dmon -i "${GPUS}" -e 203,204,155,150 -d "${delay}" >>"${logf}" 2>&1
}

run_nvidia_smi() {
  local logf="$1"
  local ms="$2"
  # Timestamped CSV; loop-ms for sub-second sampling
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >>"${logf}" 2>&1 || true
    # sleep fractional seconds
    python3 -c "import time; time.sleep(${ms}/1000.0)"
  done
}

MON_PID=""

stop_monitor() {
  if [[ -n "${MON_PID}" ]] && kill -0 "${MON_PID}" 2>/dev/null; then
    kill -INT "${MON_PID}" 2>/dev/null || true
    wait "${MON_PID}" 2>/dev/null || true
  fi
  MON_PID=""
}

trap stop_monitor EXIT

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --gpus)
        GPUS="$2"
        shift 2
        ;;
      --interval-ms)
        INTERVAL_MS="$2"
        shift 2
        ;;
      --out-dir)
        OUT_DIR="$2"
        shift 2
        ;;
      --mode)
        MODE="$2"
        shift 2
        ;;
      --)
        shift
        CMD=("$@")
        return
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
}

parse_args "$@"

if [[ ${#CMD[@]} -gt 0 && -z "${CMD[0]:-}" ]]; then
  echo "Error: empty command after --" >&2
  exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${ARTIFACTS}"
fi
mkdir -p "${OUT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/gpu_monitor_${TS}.log"
PID_FILE="${OUT_DIR}/.latest_gpu_monitor_pid"
LOG_LINK="${OUT_DIR}/.latest_gpu_monitor_log"

MODE_EFFECTIVE="${MODE}"
if [[ "${MODE}" == "auto" ]]; then
  if command -v dcgmi >/dev/null 2>&1; then
    MODE_EFFECTIVE="dcgmi"
  else
    MODE_EFFECTIVE="nvidia"
  fi
fi

log_header "${LOG}"
echo "${LOG}" >"${LOG_LINK}"
echo "Logging to: ${LOG}"
echo "mode=${MODE_EFFECTIVE} gpus=${GPUS} interval_ms=${INTERVAL_MS}"

if [[ ${#CMD[@]} -eq 0 ]]; then
  echo "Monitor running (foreground). Start training in another shell; Ctrl+C to stop."
  echo "Tip: align wall time with training_log.csv batch rows."
  if [[ "${MODE_EFFECTIVE}" == "dcgmi" ]]; then
    run_dcgmi "${LOG}" "${INTERVAL_MS}"
  else
    run_nvidia_smi "${LOG}" "${INTERVAL_MS}"
  fi
else
  echo "Starting monitor in background, then: ${CMD[*]}"
  if [[ "${MODE_EFFECTIVE}" == "dcgmi" ]]; then
    run_dcgmi "${LOG}" "${INTERVAL_MS}" &
    MON_PID=$!
  else
    run_nvidia_smi "${LOG}" "${INTERVAL_MS}" &
    MON_PID=$!
  fi
  echo "${MON_PID}" >"${PID_FILE}"
  echo "monitor_pid=${MON_PID}"
  set +e
  "${CMD[@]}"
  RC=$?
  set -e
  stop_monitor
  trap - EXIT
  exit "${RC}"
fi
