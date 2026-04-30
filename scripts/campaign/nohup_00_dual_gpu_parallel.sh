#!/usr/bin/env bash
# Launch Stage 00 (00_generate_reference_data.py) for cases 1–3 in parallel on two GPUs.
#
# Placement (3 jobs / 2 GPUs — one card runs two Python processes; watch VRAM):
#   • Physical GPU 0 — cases 1 + 3 (CUDA_VISIBLE_DEVICES masks each job to a single logical cuda:0)
#   • Physical GPU 1 — case 2
#
# Usage (from repo root):
#   bash scripts/campaign/nohup_00_dual_gpu_parallel.sh
#   CAMPAIGN_OUT=outputs/my_run N_STEPS=5000000 bash scripts/campaign/nohup_00_dual_gpu_parallel.sh
#
# Logs and PIDs under ${CAMPAIGN_OUT}/nohup_logs/
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MINIFORGE="${MINIFORGE:-/scratch/mh7373/miniforge3}"
PYTHON="${PYTHON:-${MINIFORGE}/envs/genai-tps/bin/python}"

TS="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN_OUT="${CAMPAIGN_OUT:-${REPO_ROOT}/outputs/campaign_stage00_parallel_${TS}}"
SCRATCH_ROOT="${SCRATCH:-/scratch/mh7373}"
CACHE="${BOLTZ_CACHE:-${SCRATCH_ROOT}/.boltz}"
N_STEPS="${N_STEPS:-5000000}"

LOGDIR="${CAMPAIGN_OUT}/nohup_logs"
mkdir -p "${LOGDIR}" "${CAMPAIGN_OUT}"

launcher() {
  local phys_gpu="$1"
  local case_id="$2"
  local log_file="$3"
  # shellcheck disable=SC2016
  nohup bash -c '
set -euo pipefail
module unload anaconda3 2>/dev/null || true
export PATH="$(echo "${PATH:-}" | tr ":" "\n" | grep -v "/share/apps/anaconda3/" | awk "!seen[\$0]++" | paste -sd: -)"
source "'"${MINIFORGE}"'/etc/profile.d/conda.sh"
conda activate genai-tps
export PYTHONUNBUFFERED=1
export SCRATCH="'"${SCRATCH_ROOT}"'"
export PYTHONPATH="'"${REPO_ROOT}"'/src/python"
cd "'"${REPO_ROOT}"'"
PHYS="'"${phys_gpu}"'"
CASE="'"${case_id}"'"
exec env CUDA_VISIBLE_DEVICES="${PHYS}" \
  "'"${PYTHON}"'" scripts/campaign/00_generate_reference_data.py \
    --cache "'"${CACHE}"'" \
    --out "'"${CAMPAIGN_OUT}"'" \
    --cases "${CASE}" \
    --device cuda \
    --openmm-device-index 0 \
    --platform CUDA \
    --n-steps '"${N_STEPS}"' \
    --opes-mode plumed
' >>"${log_file}" 2>&1 &
  echo $!
}

echo "[nohup_00] Repo:        ${REPO_ROOT}"
echo "[nohup_00] Out:         ${CAMPAIGN_OUT}"
echo "[nohup_00] Boltz cache: ${CACHE}"
echo "[nohup_00] N_STEPS:     ${N_STEPS}"
echo "[nohup_00] Logs:        ${LOGDIR}"
echo ""

PID1="$(launcher 0 1 "${LOGDIR}/case1_gpu0.log")"
PID2="$(launcher 1 2 "${LOGDIR}/case2_gpu1.log")"
PID3="$(launcher 0 3 "${LOGDIR}/case3_gpu0.log")"

{
  echo "$(date -Is) started"
  echo "case1 PID ${PID1}  CUDA_VISIBLE_DEVICES=0  log ${LOGDIR}/case1_gpu0.log"
  echo "case2 PID ${PID2}  CUDA_VISIBLE_DEVICES=1  log ${LOGDIR}/case2_gpu1.log"
  echo "case3 PID ${PID3}  CUDA_VISIBLE_DEVICES=0  log ${LOGDIR}/case3_gpu0.log (shares GPU 0 with case 1)"
} | tee "${LOGDIR}/launch_pids.txt"

echo ""
echo "Monitor: tail -f ${LOGDIR}/case1_gpu0.log"
echo "Merged log (when jobs finish): ${CAMPAIGN_OUT}/00_reference_data_log.json"
