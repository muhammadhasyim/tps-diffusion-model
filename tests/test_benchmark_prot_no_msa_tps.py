"""Benchmark-style TPS contract using Boltz's public ``prot_no_msa`` YAML example.

We anchor the test to the **Boltz repository** example
``examples/prot_no_msa.yaml``: a single protein chain with ``msa: empty``
(single-sequence mode), as described in Boltz's prediction documentation
(see the Boltz repo ``docs/prediction.md`` and ``examples/``).

This does **not** run full Boltz-2 inference or download checkpoints. It checks
that (1) the checked-in YAML still matches the documented sequence fingerprint,
and (2) the :class:`~genai_tps.backends.boltz.gpu_core.BoltzSamplerCore` plus
:func:`~genai_tps.backends.boltz.path_probability.compute_log_path_prob`
pipeline stays numerically well-defined at a toy atom count proportional to
that example's sequence length—serving as a regression target for the TPS
sampling idea tied to a real, citable input from upstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
from genai_tps.backends.boltz.path_probability import compute_log_path_prob

from tests.mock_boltz_diffusion import MockDiffusion

_REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_YAML = _REPO_ROOT / "boltz" / "examples" / "prot_no_msa.yaml"

# Golden fingerprints — update if upstream Boltz changes the example file.
EXPECTED_PROT_NO_MSA_LEN = 117
EXPECTED_PROT_NO_MSA_PREFIX = "QLEDSEVEAVAKGLEEMYANG"
EXPECTED_PROT_NO_MSA_SUFFIX = "LALWAVQCG"


def _read_sequence_from_boltz_yaml(path: Path) -> str:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("sequence:"):
            return stripped.split("sequence:", 1)[1].strip()
    raise ValueError(f"No sequence: line in {path}")


@pytest.fixture(scope="module")
def prot_no_msa_example() -> Path:
    if not EXAMPLE_YAML.is_file():
        pytest.skip(
            "Boltz tree missing: add the boltz submodule so boltz/examples/prot_no_msa.yaml exists."
        )
    return EXAMPLE_YAML


@pytest.mark.benchmark
def test_prot_no_msa_yaml_matches_boltz_upstream_example(prot_no_msa_example: Path) -> None:
    seq = _read_sequence_from_boltz_yaml(prot_no_msa_example)
    assert len(seq) == EXPECTED_PROT_NO_MSA_LEN
    assert seq.startswith(EXPECTED_PROT_NO_MSA_PREFIX)
    assert seq.endswith(EXPECTED_PROT_NO_MSA_SUFFIX)
    assert "msa: empty" in prot_no_msa_example.read_text()


@pytest.mark.benchmark
def test_tps_path_logprob_finite_at_prot_no_msa_scale(prot_no_msa_example: Path) -> None:
    """Mock diffusion trajectory at O(n_res) atoms matching the doc example length."""
    seq = _read_sequence_from_boltz_yaml(prot_no_msa_example)
    n_res = len(seq)
    n_atoms = n_res
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    diff = MockDiffusion(device)
    atom_mask = torch.ones(1, n_atoms, device=device)
    core = BoltzSamplerCore(diff, atom_mask, {"multiplicity": 1}, multiplicity=1)
    core.build_schedule(3)
    x = core.sample_initial_noise()
    traj, eps_list, _, _, meta_list = core.generate_segment(x, 0, core.num_sampling_steps)
    lp = compute_log_path_prob(
        eps_list,
        meta_list,
        initial_coords=x,
        sigma0=float(core.schedule[0].sigma_tm),
        include_jacobian=True,
        n_atoms=n_atoms,
    )
    assert torch.isfinite(lp)
    assert traj[-1].shape == (1, n_atoms, 3)
