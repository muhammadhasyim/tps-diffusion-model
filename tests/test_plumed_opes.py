"""Tests for PLUMED-backed OPES-MD helpers.

These tests are intentionally split into pure string/index checks and optional
runtime checks.  The pure tests protect the atom-index and PLUMED-deck contract
without requiring a compiled ``openmm-plumed`` plugin.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest


def test_boltz_to_plumed_indices_converts_to_one_based_openmm_indices() -> None:
    from genai_tps.simulation.openmm_boltz_bridge import boltz_to_plumed_indices

    omm_idx_map = np.array([10, 20, 30, 40], dtype=np.int64)

    assert boltz_to_plumed_indices([0, 2, 3], omm_idx_map) == [11, 31, 41]


def test_boltz_to_plumed_indices_rejects_negative_mapping() -> None:
    from genai_tps.simulation.openmm_boltz_bridge import boltz_to_plumed_indices

    omm_idx_map = np.array([10, -1, 30], dtype=np.int64)

    with pytest.raises(ValueError, match="negative OpenMM atom index"):
        boltz_to_plumed_indices([1], omm_idx_map)


def test_generate_plumed_opes_script_contains_real_opes_actions(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    script = generate_plumed_opes_script(
        ligand_plumed_idx=[5, 6, 7],
        pocket_ca_plumed_idx=[1, 2, 3],
        rmsd_reference_pdb=ref_path,
        sigma=(0.3, 0.5),
        pace=500,
        barrier=5.0,
        biasfactor=10.0,
        temperature=300.0,
        save_opes_every=50_000,
        progress_every=10_000,
        out_dir=tmp_path,
    )

    assert "UNITS LENGTH=A" in script
    assert "lig_rmsd: RMSD TYPE=OPTIMAL" in script
    assert f"REFERENCE={ref_path}" in script
    assert "lig_com: CENTER NOPBC ATOMS=5,6,7" in script
    assert "pocket_com: CENTER NOPBC ATOMS=1,2,3" in script
    assert "lig_dist: DISTANCE NOPBC ATOMS=lig_com,pocket_com" in script
    assert "opes: OPES_METAD ..." in script
    assert "ARG=lig_rmsd,lig_dist" in script
    assert "SIGMA=0.3,0.5" in script
    assert "PACE=500" in script
    assert "BARRIER=5" in script
    assert "BIASFACTOR=10" in script
    assert "NLIST" in script
    assert "PRINT STRIDE=10000" in script
    m = re.search(r"KERNEL_CUTOFF=([0-9.eE+-]+)", script)
    assert m is not None
    assert float(m.group(1)) >= 3.5
    assert "opes.zed" in script and "opes.neff" in script
    assert "ARG=lig_rmsd,lig_dist,opes.bias,opes.rct,opes.nker,opes.zed,opes.neff" in script


def test_generate_plumed_opes_script_explicit_kernel_cutoff(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    script = generate_plumed_opes_script(
        ligand_plumed_idx=[5, 6, 7],
        pocket_ca_plumed_idx=[1, 2, 3],
        rmsd_reference_pdb=ref_path,
        sigma=(0.3, 0.5),
        pace=500,
        barrier=5.0,
        biasfactor=10.0,
        temperature=300.0,
        save_opes_every=50_000,
        progress_every=10_000,
        out_dir=tmp_path,
        kernel_cutoff=2.0,
    )
    m = re.search(r"KERNEL_CUTOFF=([0-9.eE+-]+)", script)
    assert m is not None
    assert float(m.group(1)) == pytest.approx(2.0)


def test_generate_plumed_opes_script_nlist_parameters(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    script = generate_plumed_opes_script(
        ligand_plumed_idx=[5, 6, 7],
        pocket_ca_plumed_idx=[1, 2, 3],
        rmsd_reference_pdb=ref_path,
        sigma=(0.3, 0.5),
        pace=500,
        barrier=5.0,
        biasfactor=10.0,
        temperature=300.0,
        save_opes_every=50_000,
        progress_every=10_000,
        out_dir=tmp_path,
        nlist_parameters=(4.0, 0.4),
    )
    m_nl = re.search(r"NLIST_PARAMETERS=([\d.eE+-]+),([\d.eE+-]+)", script)
    assert m_nl is not None
    assert float(m_nl.group(1)) == pytest.approx(4.0)
    assert float(m_nl.group(2)) == pytest.approx(0.4)


def test_generate_plumed_opes_script_rejects_wrong_sigma_length(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    with pytest.raises(ValueError, match="two sigma values"):
        generate_plumed_opes_script(
            ligand_plumed_idx=[5, 6, 7],
            pocket_ca_plumed_idx=[1, 2, 3],
            rmsd_reference_pdb=tmp_path / "reference_pose.pdb",
            sigma=(0.3,),
            pace=500,
            barrier=5.0,
            biasfactor=10.0,
            temperature=300.0,
            save_opes_every=50_000,
            progress_every=10_000,
            out_dir=tmp_path,
        )


def test_build_md_simulation_from_pdb_exposes_extra_forces_hook() -> None:
    import compute_cv_rmsd

    sig = inspect.signature(compute_cv_rmsd.build_md_simulation_from_pdb)

    assert "extra_forces" in sig.parameters


def test_add_plumed_opes_to_system_optional_runtime() -> None:
    pytest.importorskip("openmm")
    pytest.importorskip("openmmplumed")

    import openmm
    from genai_tps.simulation.plumed_opes import add_plumed_opes_to_system

    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    script = "d: DISTANCE ATOMS=1,2\nBIASVALUE ARG=d"

    force, force_index = add_plumed_opes_to_system(
        system,
        script,
        temperature=300.0,
        force_group=30,
    )

    assert force_index == system.getNumForces() - 1
    assert force.getForceGroup() == 30


def test_plumed_opes_bias_energy_becomes_queryable_after_deposition(
    tmp_path: Path,
) -> None:
    pytest.importorskip("openmm")
    pytest.importorskip("openmmplumed")

    import openmm
    import openmm.unit as unit
    from genai_tps.simulation.plumed_opes import add_plumed_opes_to_system

    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    restraint = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    restraint.addGlobalParameter("k", 100.0)
    restraint.addGlobalParameter("r0", 0.2)
    restraint.addBond(0, 1, [])
    system.addForce(restraint)

    kernels = tmp_path / "KERNELS"
    colvar = tmp_path / "COLVAR"
    script = f"""
UNITS LENGTH=A
d: DISTANCE NOPBC ATOMS=1,2
opes: OPES_METAD ARG=d SIGMA=0.1 PACE=1 BARRIER=5 TEMP=300 FILE={kernels}
PRINT STRIDE=1 FILE={colvar} ARG=d,opes.bias
"""
    add_plumed_opes_to_system(system, script, temperature=300.0, force_group=30)

    integrator = openmm.LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        1.0 * unit.femtosecond,
    )
    context = openmm.Context(system, integrator)
    context.setPositions(
        [
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(0.2, 0.0, 0.0),
        ]
        * unit.nanometer
    )
    integrator.step(5)
    state = context.getState(getEnergy=True, groups={30})
    bias_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    assert np.isfinite(bias_energy)
    assert abs(bias_energy) > 1e-12
