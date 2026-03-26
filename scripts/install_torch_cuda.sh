#!/usr/bin/env bash
# install_torch_cuda.sh — Install PyTorch with the correct CUDA wheel for this machine.
#
# Usage:
#   bash scripts/install_torch_cuda.sh [cu121|cu128|cpu|auto]
#
# If no argument is given, "auto" is used: the script reads `nvidia-smi` output
# to determine the CUDA version and picks the right wheel automatically.
#
# Must be run inside the activated genai-tps conda environment:
#   conda activate genai-tps && bash scripts/install_torch_cuda.sh
#
# After running this script, re-activate the environment so that the
# LD_LIBRARY_PATH hook takes effect:
#   conda deactivate && conda activate genai-tps

set -euo pipefail

# ── 1. Determine the wheel tag ────────────────────────────────────────────────

TAG="${1:-auto}"

if [[ "$TAG" == "auto" ]]; then
    if ! command -v nvidia-smi &>/dev/null; then
        echo "[install_torch_cuda] nvidia-smi not found — installing CPU-only PyTorch."
        TAG="cpu"
    else
        # nvidia-smi header: "CUDA Version: 12.2" → extract major.minor
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || true)
        if [[ -z "$CUDA_VER" ]]; then
            echo "[install_torch_cuda] Could not parse CUDA version from nvidia-smi; defaulting to cu121."
            TAG="cu121"
        else
            CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
            if (( CUDA_MAJOR > 12 || (CUDA_MAJOR == 12 && CUDA_MINOR >= 8) )); then
                TAG="cu128"
            else
                TAG="cu121"
            fi
            echo "[install_torch_cuda] Detected CUDA ${CUDA_VER} → using ${TAG} wheel."
        fi
    fi
fi

case "$TAG" in
    cu121|cu128|cpu) ;;
    *)
        echo "Unknown tag '$TAG'. Use: cu121 | cu128 | cpu | auto" >&2
        exit 1
        ;;
esac

if [[ "$TAG" == "cpu" ]]; then
    INDEX_URL="https://download.pytorch.org/whl/cpu"
else
    INDEX_URL="https://download.pytorch.org/whl/${TAG}"
fi

# ── 2. Nuclear cleanup of any existing torch installation ─────────────────────
#
# conda-forge installs its own pytorch (cuda126 build) during `conda env create`
# which leaves include/ and lib/ directories under site-packages/torch/.
# pip's --force-reinstall will fail with "No such file or directory: ATen.h"
# if those include trees are partially removed.  Wiping everything first gives
# pip a clean slate.
#
# IMPORTANT: the glob uses "torch-", "torchaudio-", "torchvision-" (with the
# hyphen) to match only the torch dist-info directories and NOT torchmetrics,
# torchgen, etc.

SITE="${CONDA_PREFIX}/lib/python3.11/site-packages"
echo "[install_torch_cuda] Removing any existing torch installation ..."
rm -rf "${SITE}/torch" "${SITE}/torchgen" "${SITE}/torchaudio" "${SITE}/torchvision"
for pkg in "torch-" "torchaudio-" "torchvision-"; do
    rm -rf "${SITE}/${pkg}"*.dist-info 2>/dev/null || true
done

# ── 3. Install torch ──────────────────────────────────────────────────────────

echo "[install_torch_cuda] Installing torch from ${INDEX_URL} ..."
pip install torch torchvision torchaudio --index-url "${INDEX_URL}"

# ── 4. Pin nvjitlink for cu121 ────────────────────────────────────────────────
#
# Two libcusparse versions must be satisfied simultaneously after this install:
#   - torch cu121 bundles libcusparse 12.1 → requires __nvJitLinkAddData_12_1    (≥12.1 API)
#   - conda-forge openmm (cuda12.2 build) brings libcusparse 12.2
#                                         → requires __nvJitLinkGetLinkedCubin_12_2 (≥12.2 API)
#
# nvjitlink 12.2 exports both versioned symbol sets and is compatible with
# driver 535 (CUDA 12.2 capability).  Versions ≥12.3 may generate PTX that
# the driver cannot execute.

if [[ "$TAG" == "cu121" ]]; then
    echo "[install_torch_cuda] Pinning nvidia-nvjitlink-cu12 to 12.2.x ..."
    pip install "nvidia-nvjitlink-cu12>=12.2,<12.3"
fi

# ── 5. Install conda activation hook (LD_LIBRARY_PATH fix) ───────────────────
#
# libcusparse from cu121 links against libnvJitLink.so.12 at runtime.  On
# machines where /usr/local/cuda-12.0/lib64 appears early in LD_LIBRARY_PATH,
# the system 12.0 libnvJitLink is found before the pip 12.2 one, causing:
#   ImportError: undefined symbol __nvJitLinkAddData_12_1
# The activation hook prepends the pip nvjitlink lib so the correct version wins.

if [[ "$TAG" == "cu121" ]]; then
    ACTIVATE_D="${CONDA_PREFIX}/etc/conda/activate.d"
    DEACTIVATE_D="${CONDA_PREFIX}/etc/conda/deactivate.d"
    mkdir -p "$ACTIVATE_D" "$DEACTIVATE_D"

    cat > "${ACTIVATE_D}/nvjitlink_priority.sh" << 'HOOK'
#!/bin/sh
# Prepend pip nvidia/nvjitlink/lib so it shadows the system CUDA 12.0 copy.
_L="$(python -c 'import nvidia.nvjitlink, os; print(os.path.dirname(nvidia.nvjitlink.__file__))' 2>/dev/null)/lib"
[ -d "$_L" ] && export LD_LIBRARY_PATH="${_L}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _L
HOOK

    cat > "${DEACTIVATE_D}/nvjitlink_priority.sh" << 'HOOK'
#!/bin/sh
_L="$(python -c 'import nvidia.nvjitlink, os; print(os.path.dirname(nvidia.nvjitlink.__file__))' 2>/dev/null)/lib"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH#${_L}:}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH#${_L}}"
export LD_LIBRARY_PATH
unset _L
HOOK

    chmod +x "${ACTIVATE_D}/nvjitlink_priority.sh" "${DEACTIVATE_D}/nvjitlink_priority.sh"
    echo "[install_torch_cuda] LD_LIBRARY_PATH activation hook written to ${ACTIVATE_D}."
fi

# ── 6. Pin pytorch in conda so `conda install` can never overwrite pip torch ──
#
# Any subsequent `conda install <anything>` re-runs the full dependency solver
# and will reinstall conda-forge's pytorch (cuda126 build), breaking the pip
# torchvision ABI.  The pinned file marks these as immovable.

PINNED="${CONDA_PREFIX}/conda-meta/pinned"
for pkg in pytorch torchvision torchaudio; do
    grep -q "^${pkg}" "$PINNED" 2>/dev/null || echo "${pkg} >=0" >> "$PINNED"
done
echo "[install_torch_cuda] conda-meta/pinned updated — conda will not overwrite pip torch."

# ── 7. Enforce version anchors that frequently drift ─────────────────────────
#
# Several packages can be silently upgraded by transitive pip/conda deps:
#   numpy:        jax[cuda12] pulls 2.4.x; boltz requires <2.3
#                 openmm xtc_utils Cython ext compiled against numpy 2.x ABI
#                 (dtype size 96); numpy 1.x (size 88) causes ValueError
#   scipy:        conda-forge floats to 1.17.x; boltz pins ==1.13.1 exactly;
#                 openmm xtcfile compiled against one scipy ABI, mismatch errors
#   torchmetrics: pytorch-lightning==2.5.0 dep; gets wiped when torch dirs are
#                 cleaned (the "torch*" glob is now "torch-*" but belt-and-suspenders)

echo "[install_torch_cuda] Enforcing numpy / scipy / torchmetrics pins ..."
pip install "numpy>=2.0,<2.3" "scipy==1.13.1" "torchmetrics>=1.0"

# ── 8. Verify ────────────────────────────────────────────────────────────────

echo ""
echo "[install_torch_cuda] Verifying installation ..."
python - << 'PYCHECK'
import sys

def check(label, fn):
    try:
        result = fn()
        print(f"  {label:<28} {result}")
    except Exception as exc:
        print(f"  {label:<28} FAILED: {exc}", file=sys.stderr)

import torch
check("torch", lambda: torch.__version__)
check("CUDA available", lambda: torch.cuda.is_available())
if torch.cuda.is_available():
    check("GPU", lambda: torch.cuda.get_device_name(0))

import torchvision
check("torchvision", lambda: torchvision.__version__)

import numpy as np
check("numpy", lambda: np.__version__)

import scipy
check("scipy", lambda: scipy.__version__)

import torchmetrics
check("torchmetrics", lambda: torchmetrics.__version__)

try:
    import openmm as mm
    from openmm import Platform
    names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    check("OpenMM platforms", lambda: str(names))
except Exception as exc:
    print(f"  {'OpenMM platforms':<28} FAILED: {exc}", file=sys.stderr)

PYCHECK

echo ""
echo "[install_torch_cuda] Done."
echo ""
echo "  *** Re-activate the environment for the LD_LIBRARY_PATH hook to fire:"
echo "      conda deactivate && conda activate genai-tps"
