#!/usr/bin/env bash
# Wrap Stage 2 (or any) Python training under Nsight Systems CLI for a bounded CUDA timeline.
#
# After the capture window, nsys stops collecting but the child process keeps running
# (--stop-on-exit=false --kill none). The nsys CLI may return before training finishes; the
# Python process continues in the background — prefer running inside tmux/screen if you
# need a single attached session for the full run.
#
# Usage:
#   bash scripts/profile/nsys_wrap_train.sh --nsys-duration 120 -- \\
#     python scripts/campaign/02_finetune_boltz2.py --out outputs/campaign --cases 1 --batch-size 4 --device cuda
#
# Environment overrides:
#   NSYS_DURATION   Same as --nsys-duration (default: 120)
#   NSYS_TRACE      Same as --nsys-trace (default: cuda,nvtx)
#   NSYS_OUTPUT     Full output path stem (default: timestamped under artifacts/evaluation)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

DURATION="${NSYS_DURATION:-120}"
TRACE="${NSYS_TRACE:-cuda,nvtx}"
OUT_STEM=""

usage() {
  cat <<'EOF'
nsys_wrap_train.sh — bounded Nsight Systems capture around a training command

Required: a literal "--" followed by the command to run (usually "python ...").

Options:
  -h, --help
  --nsys-duration SEC   Collection duration in seconds (default: 120 or NSYS_DURATION)
  --nsys-trace LIST     Trace domains for nsys (default: cuda,nvtx or NSYS_TRACE)
  --nsys-output STEM    Output path prefix for .nsys-rep (default: auto under artifacts/evaluation)

Example:
  bash scripts/profile/nsys_wrap_train.sh --nsys-duration 90 --nsys-trace cuda,nvtx,cudnn,osrt -- \\
    python scripts/train_weighted_dsm.py --yaml inputs/tps_diagnostic/case1_mek1_fzc_novel.yaml \\
      --data path/to/training_dataset.npz --out /tmp/wdsm_prof --epochs 2 --batch-size 2 --device cuda

For NVTX ranges in the trainer, set:  export GENAI_TPS_NVTX=1
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --nsys-duration)
      DURATION="$2"
      shift 2
      ;;
    --nsys-trace)
      TRACE="$2"
      shift 2
      ;;
    --nsys-output)
      OUT_STEM="$2"
      shift 2
      ;;
    --)
      shift
      ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1 (use -- before the python command)" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "Error: missing command after --" >&2
  usage >&2
  exit 2
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "Error: nsys not found in PATH (install Nsight Systems CLI)." >&2
  exit 127
fi

if [[ -z "${OUT_STEM}" ]]; then
  if [[ -n "${NSYS_OUTPUT:-}" ]]; then
    OUT_STEM="${NSYS_OUTPUT}"
  else
    TS="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${REPO_ROOT}/artifacts/evaluation/nsys_wdsm"
    OUT_STEM="${REPO_ROOT}/artifacts/evaluation/nsys_wdsm/capture_${TS}"
  fi
fi

OUT_DIR="$(dirname "${OUT_STEM}")"
mkdir -p "${OUT_DIR}"
echo "nsys output stem: ${OUT_STEM}"
echo "nsys duration:   ${DURATION}s"
echo "nsys trace:      ${TRACE}"
echo "command:         ${ARGS[*]}"

exec nsys profile \
  --trace="${TRACE}" \
  --stats=true \
  --force-overwrite=true \
  --duration="${DURATION}" \
  --stop-on-exit=false \
  --kill=none \
  -o "${OUT_STEM}" \
  -- "${ARGS[@]}"
