#!/usr/bin/env bash
# Staged OneOPES legacy-boltz HREX smokes on a single GPU (e.g. RTX 4070 12GB).
#
# Prerequisites:
#   - conda env with OpenMM, openmm-plumed, genai-tps on PYTHONPATH
#   - PLUMED built with opes module; PLUMED_KERNEL set (see scripts/build_plumed_opes.sh)
#   - Boltz mols cache (default ~/.boltz/mols) containing ligand CCD pickles
#
# Usage (from repo root):
#   export PYTHONPATH="${PWD}/src/python:${PYTHONPATH:-}"
#   bash scripts/smoke/run_oneopes_repex_scaleup.sh
#
# Optional env:
#   TOPO_NPZ, FRAME_NPZ, MOL_DIR — input paths (defaults: case1 campaign artifacts)
#   OUT_BASE — base directory for stage outputs (default: artifacts/evaluation/repex_scaleup_smoke)
#   RUN_REPEX_STAGE7=1 — also run 7-replica stage (VRAM-heavy; may OOM before n8)
#   RUN_REPEX_STAGE8=1 — also run 8-replica stage (heaviest)
#   PYTHON — interpreter (must exist and be executable; invalid values are ignored
#            and ``python3`` from PATH is used — activate conda first if needed)
#   MASTER_PORT — torch.distributed rendezvous port (default: 29581)
#   SKIP_PREFLIGHT=1 — skip scripts/smoke/preflight_oneopes_repex.sh (not recommended)
#
# Default stages are n2→n6 (3–6 replica counts are n3–n6). Enable stage7/8 to push
# until GPU OOM on your hardware.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/python:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-29581}"

if [[ -n "${PYTHON:-}" && ! -x "${PYTHON}" ]]; then
  echo "[repex-scaleup] WARN: PYTHON=${PYTHON} is missing or not executable; using python3 from PATH." >&2
  PYTHON=
fi
if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$(command -v python3 2>/dev/null || true)"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "[repex-scaleup] ERROR: No usable interpreter. Activate conda (genai-tps) or set PYTHON to a real path." >&2
  exit 2
fi
OUT_BASE="${OUT_BASE:-${REPO_ROOT}/artifacts/evaluation/repex_scaleup_smoke}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${OUT_BASE}/${TS}"
mkdir -p "${RUN_ROOT}"

TOPO_NPZ="${TOPO_NPZ:-${REPO_ROOT}/artifacts/evaluation/campaign_stage00_full_20260430/case1_mek1_fzc_novel/openmm_opes_md/boltz_prep/boltz_prep_case1_mek1_fzc_novel/processed/structures/case1_mek1_fzc_novel.npz}"
FRAME_NPZ="${FRAME_NPZ:-${REPO_ROOT}/artifacts/evaluation/campaign_stage00_full_20260430/case1_mek1_fzc_novel/openmm_opes_md/boltz_init_coords.npz}"
MOL_DIR="${MOL_DIR:-${HOME}/.boltz/mols}"

MONITOR="${REPO_ROOT}/scripts/profile/run_gpu_monitor.sh"
DRIVER="${REPO_ROOT}/scripts/run_openmm_oneopes_repex.py"
VALIDATE="${REPO_ROOT}/scripts/smoke/validate_oneopes_repex_smoke.py"

if [[ ! -f "${TOPO_NPZ}" ]]; then
  echo "[repex-scaleup] ERROR: missing TOPO_NPZ=${TOPO_NPZ}" >&2
  exit 2
fi
if [[ ! -f "${FRAME_NPZ}" ]]; then
  echo "[repex-scaleup] ERROR: missing FRAME_NPZ=${FRAME_NPZ}" >&2
  exit 2
fi
if [[ ! -d "${MOL_DIR}" ]]; then
  echo "[repex-scaleup] ERROR: missing MOL_DIR=${MOL_DIR} (Boltz cache mols)" >&2
  exit 2
fi

if [[ -z "${PLUMED_KERNEL:-}" ]]; then
  echo "[repex-scaleup] WARN: PLUMED_KERNEL is unset; OPES may fail. Set to conda libplumedKernel.so." >&2
fi

echo "[repex-scaleup] Checking PLUMED opes module..."
"${PYTHON}" -c "from genai_tps.simulation.plumed_kernel import assert_plumed_opes_metad_available; assert_plumed_opes_metad_available()"

PREFLIGHT="${REPO_ROOT}/scripts/smoke/preflight_oneopes_repex.sh"
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "[repex-scaleup] Preflight (CPU checks; see ${PREFLIGHT})..."
  bash "${PREFLIGHT}"
else
  echo "[repex-scaleup] WARN: SKIP_PREFLIGHT=1 — skipping CPU preflight." >&2
fi

# Shared CLI args for run_openmm_oneopes_repex.py (excluding --out and per-stage flags).
driver_args=(
  --topo-npz "${TOPO_NPZ}"
  --frame-npz "${FRAME_NPZ}"
  --mol-dir "${MOL_DIR}"
  --bias-cv oneopes
  --opes-mode plumed
  --opes-explore
  --platform CUDA
  --devices 0
  --replica-device-map packed
  --evaluator-placement same-device
  --oneopes-protocol legacy-boltz
  --temperature 298
  --pocket-radius 12
  --minimize-steps 1000
  --progress-every 500
  --opes-barrier 80
  --opes-biasfactor 8
  --opes-sigma 0.05,0.35
)

run_stage() {
  local name="$1"
  shift
  local out="${RUN_ROOT}/${name}"
  local logdir="${RUN_ROOT}/gpu_${name}"
  mkdir -p "${out}" "${logdir}"
  echo "======== ${name} OUT=${out} ========"
  bash "${MONITOR}" --gpus 0 --interval-ms 200 --out-dir "${logdir}" -- \
    "${PYTHON}" "${DRIVER}" "${driver_args[@]}" --out "${out}" "$@"
  local gpu_log
  gpu_log="$(ls -t "${logdir}"/gpu_monitor_*.log 2>/dev/null | head -1 || true)"
  if [[ -n "${gpu_log}" ]]; then
    "${PYTHON}" "${VALIDATE}" "${out}" --gpu-monitor-log "${gpu_log}"
  else
    "${PYTHON}" "${VALIDATE}" "${out}"
  fi
}

# PLUMED requires STATE_WSTRIDE (from --save-opes-every) >= OPES PACE on every
# replica; auxiliary hydration OPES uses --oneopes-water-pace (default 40000).
SAVE_OPES_EVERY="${SAVE_OPES_EVERY:-50000}"
DEPOSIT_PACE="${DEPOSIT_PACE:-500}"

run_stage "n2" --n-replicas 2 \
  --n-steps 6000 --exchange-every 3000 --save-every 3000 --deposit-pace "${DEPOSIT_PACE}" \
  --save-opes-every "${SAVE_OPES_EVERY}" --max-active-contexts-per-device 4

run_stage "n3" --n-replicas 3 \
  --n-steps 4000 --exchange-every 2000 --save-every 2000 --deposit-pace "${DEPOSIT_PACE}" \
  --save-opes-every "${SAVE_OPES_EVERY}" --max-active-contexts-per-device 5

run_stage "n4" --n-replicas 4 \
  --n-steps 4000 --exchange-every 2000 --save-every 2000 --deposit-pace "${DEPOSIT_PACE}" \
  --save-opes-every "${SAVE_OPES_EVERY}" --max-active-contexts-per-device 6

run_stage "n5" --n-replicas 5 \
  --n-steps 4000 --exchange-every 2000 --save-every 2000 --deposit-pace "${DEPOSIT_PACE}" \
  --save-opes-every "${SAVE_OPES_EVERY}" --max-active-contexts-per-device 7

run_stage "n6" --n-replicas 6 \
  --n-steps 4000 --exchange-every 2000 --save-every 2000 --deposit-pace "${DEPOSIT_PACE}" \
  --save-opes-every "${SAVE_OPES_EVERY}" --max-active-contexts-per-device 8

if [[ "${RUN_REPEX_STAGE7:-0}" == "1" ]]; then
  run_stage "n7" --n-replicas 7 \
    --n-steps 2000 --exchange-every 1000 --save-every 1000 --deposit-pace "${DEPOSIT_PACE}" \
    --save-opes-every "${SAVE_OPES_EVERY}" --max-active-contexts-per-device 9
else
  echo "[repex-scaleup] Skipping 7-replica stage (set RUN_REPEX_STAGE7=1 to probe VRAM / OOM)."
fi

if [[ "${RUN_REPEX_STAGE8:-0}" == "1" ]]; then
  run_stage "n8" --n-replicas 8 \
    --n-steps 2000 --exchange-every 1000 --save-every 1000 --deposit-pace "${DEPOSIT_PACE}" \
    --save-opes-every "${SAVE_OPES_EVERY}" --max-active-contexts-per-device 10
else
  echo "[repex-scaleup] Skipping 8-replica stage (set RUN_REPEX_STAGE8=1 to enable)."
fi

echo "[repex-scaleup] All requested stages OK under ${RUN_ROOT}"
