"""Smoke-test every public module and script for importability.

Catches SyntaxError, ImportError, and NameError from typos before
any real computation runs.

This test file should be kept fast: it only checks that modules can be
imported and scripts can be compiled, not that they produce correct output.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

# All genai_tps submodules to verify
_MODULES = [
    "genai_tps.backends.boltz.collective_variables",
    "genai_tps.backends.boltz.engine",
    "genai_tps.backends.boltz.gpu_core",
    "genai_tps.backends.boltz.path_probability",
    "genai_tps.backends.boltz.snapshot",
    "genai_tps.backends.boltz.states",
    "genai_tps.backends.boltz.tps_sampling",
    "genai_tps.enhanced_sampling.openmm_cv",
    "genai_tps.enhanced_sampling.opes_bias",
    "genai_tps.enhanced_sampling.exponential_tilting",
    "genai_tps.enhanced_sampling.mbar_analysis",
    "genai_tps.analysis.boltz_npz_export",
]


@pytest.mark.parametrize("mod", _MODULES)
def test_import_module(mod: str) -> None:
    """Every listed module must import without error."""
    importlib.import_module(mod)


# ---------------------------------------------------------------------------
# Key callables must be callable
# ---------------------------------------------------------------------------

def test_collective_variables_callables() -> None:
    """contact_order, clash_count, lddt_to_reference, ramachandran_outlier_fraction
    must all be callable after import."""
    from genai_tps.backends.boltz.collective_variables import (
        contact_order,
        clash_count,
        lddt_to_reference,
        ramachandran_outlier_fraction,
        radius_of_gyration,
        rmsd_to_reference,
        make_rg_cv,
        make_plddt_proxy_cv,
        make_boltz_plddt_predictor,
    )
    for fn in [
        contact_order, clash_count, lddt_to_reference,
        ramachandran_outlier_fraction, radius_of_gyration,
        rmsd_to_reference, make_rg_cv, make_plddt_proxy_cv,
        make_boltz_plddt_predictor,
    ]:
        assert callable(fn), f"{fn} is not callable"


def test_gpu_core_boltz_sampler_core_signature() -> None:
    """BoltzSamplerCore.__init__ must accept compile_model, n_fixed_point, inference_dtype."""
    import inspect
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
    sig = inspect.signature(BoltzSamplerCore.__init__)
    params = sig.parameters
    assert "compile_model" in params, "compile_model missing from BoltzSamplerCore.__init__"
    assert "n_fixed_point" in params, "n_fixed_point missing from BoltzSamplerCore.__init__"
    assert "inference_dtype" in params, "inference_dtype missing from BoltzSamplerCore.__init__"


def test_run_tps_path_sampling_has_diagnostic_cv_param() -> None:
    """run_tps_path_sampling must accept diagnostic_cv_functions parameter."""
    import inspect
    from genai_tps.backends.boltz.tps_sampling import run_tps_path_sampling
    sig = inspect.signature(run_tps_path_sampling)
    assert "diagnostic_cv_functions" in sig.parameters, (
        "diagnostic_cv_functions missing from run_tps_path_sampling"
    )


def test_openmm_cv_has_persistent_context_attrs() -> None:
    """OpenMMLocalMinRMSD.__init__ must initialize _simulation and _ligand_smiles_cache."""
    import inspect
    from genai_tps.enhanced_sampling.openmm_cv import OpenMMLocalMinRMSD
    src = inspect.getsource(OpenMMLocalMinRMSD.__init__)
    assert "_simulation" in src
    assert "_ligand_smiles_cache" in src


# ---------------------------------------------------------------------------
# Script syntax compilation
# ---------------------------------------------------------------------------

_SCRIPTS = list((Path(__file__).parent.parent / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_script_syntax(script: Path) -> None:
    """Every script in scripts/ must compile without SyntaxError."""
    code = script.read_text(encoding="utf-8")
    compile(code, str(script), "exec")  # raises SyntaxError on bad syntax


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_script_importable(script: Path) -> None:
    """Every script must be spec-importable (catches hard ImportError at top level)."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            f"import importlib.util; "
            f"spec = importlib.util.spec_from_file_location('s', r'{script}'); "
            f"mod = importlib.util.module_from_spec(spec)",
        ],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Script {script.name} failed spec import:\n{result.stderr.decode()}"
    )
