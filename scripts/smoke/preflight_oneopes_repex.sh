#!/usr/bin/env bash
# CPU-only checks before launching GPU OneOPES REPEX (driver + distributed helpers).
#
# Usage (from repo root, conda env active):
#   bash scripts/smoke/preflight_oneopes_repex.sh
#
# Optional:
#   PYTHON — interpreter (default: python3 on PATH)
#   SKIP_PREFLIGHT_PYTEST=1 — only compileall + import smoke (faster; less coverage)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/python:${PYTHONPATH:-}"

if [[ -n "${PYTHON:-}" && ! -x "${PYTHON}" ]]; then
  echo "[preflight-repex] WARN: PYTHON=${PYTHON} is not executable; using python3." >&2
  PYTHON=
fi
if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$(command -v python3 2>/dev/null || true)"
fi
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
  echo "[preflight-repex] ERROR: No python interpreter." >&2
  exit 2
fi

echo "[preflight-repex] compileall (genai_tps.simulation + driver)"
"${PYTHON}" -m compileall -q \
  src/python/genai_tps/simulation \
  scripts/run_openmm_oneopes_repex.py

echo "[preflight-repex] import smoke (keyword-only prepare_evaluator_scratch_tree)"
"${PYTHON}" -c "
import inspect
from genai_tps.simulation.repex_distributed import run_distributed_repex
from genai_tps.simulation.oneopes_repex import prepare_evaluator_scratch_tree
sig = inspect.signature(prepare_evaluator_scratch_tree)
for name, p in sig.parameters.items():
    if name in ('production_rep_root', 'scratch_root'):
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (name, p.kind)
assert callable(run_distributed_repex)
"

if [[ "${SKIP_PREFLIGHT_PYTEST:-0}" != "1" ]]; then
  echo "[preflight-repex] pytest tests/test_repex_distributed.py tests/test_oneopes_repex.py"
  if ! "${PYTHON}" -m pytest tests/test_repex_distributed.py tests/test_oneopes_repex.py -q --tb=short; then
    echo "[preflight-repex] ERROR: pytest failed." >&2
    exit 1
  fi
else
  echo "[preflight-repex] SKIP_PREFLIGHT_PYTEST=1 — skipping pytest."
fi

echo "[preflight-repex] OK"
