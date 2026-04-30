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
    assert "HEAVY_FLUSH" not in script


def test_generate_plumed_opes_script_colvar_heavy_flush(tmp_path: Path) -> None:
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
        print_colvar_heavy_flush=True,
    )
    assert "PRINT STRIDE=10000" in script
    assert "opes.neff HEAVY_FLUSH" in script


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

    with pytest.raises(ValueError, match="requires 2 sigma"):
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
            cv_mode="2d",
        )


def test_generate_plumed_opes_script_3d_coordination_and_three_sigmas(
    tmp_path: Path,
) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    script = generate_plumed_opes_script(
        ligand_plumed_idx=[5, 6, 7],
        pocket_ca_plumed_idx=[1, 2, 3],
        rmsd_reference_pdb=ref_path,
        sigma=(0.3, 0.5, 1.0),
        pace=500,
        barrier=40.0,
        biasfactor=10.0,
        temperature=300.0,
        save_opes_every=50_000,
        progress_every=10_000,
        out_dir=tmp_path,
        cv_mode="3d",
        pocket_heavy_plumed_idx=[10, 11, 12],
        coordination_r0=4.5,
    )
    assert "lig_contacts: COORDINATION GROUPA=5,6,7 GROUPB=10,11,12" in script
    assert "R_0=4.5" in script and "NN=6" in script and "MM=12" in script
    assert "ARG=lig_rmsd,lig_dist,lig_contacts" in script
    m_sig = re.search(r"^\s*SIGMA=([\d.,+-eE]+)\s*$", script, re.MULTILINE)
    assert m_sig is not None
    sig_parts = [float(x) for x in m_sig.group(1).split(",")]
    assert len(sig_parts) == 3
    assert sig_parts == pytest.approx([0.3, 0.5, 1.0])
    assert "lig_rmsd,lig_dist,lig_contacts,opes.bias" in script


def test_explore_with_wall_emits_opes_metad_not_explore(tmp_path: Path) -> None:
    """EXTRA_BIAS is unsupported on OPES_METAD_EXPLORE; deck must use OPES_METAD."""
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
        opes_variant="explore",
        upper_wall_dist=12.0,
    )
    assert "opes: OPES_METAD ..." in script
    assert "opes: OPES_METAD_EXPLORE ..." not in script
    assert "EXTRA_BIAS=lig_dist_uwall.bias" in script
    assert "OPES_METAD_EXPLORE has no EXTRA_BIAS" in script


def test_generate_plumed_opes_script_opes_expanded_multithermal(tmp_path: Path) -> None:
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
        opes_expanded_temp_max=600.0,
        opes_expanded_pace=50,
    )
    assert "ene: ENERGY" in script
    assert "opes_ecv: ECV_MULTITHERMAL ARG=ene TEMP=300" in script
    assert "TEMP_MAX=600" in script
    assert "opes_expanded: OPES_EXPANDED ..." in script
    assert "ARG=opes_ecv.*" in script
    assert "FILE=" in script and "OPES_EXPANDED_DELTAFS" in script
    assert "STATE_WFILE=" in script and "STATE_EXPANDED" in script
    assert "ene,opes_expanded.bias" in script


def test_generate_plumed_opes_script_opes_expanded_with_explore(tmp_path: Path) -> None:
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
        opes_variant="explore",
        opes_expanded_temp_max=550.0,
        opes_expanded_pace=40,
    )
    assert "opes: OPES_METAD_EXPLORE ..." in script
    assert "opes_expanded: OPES_EXPANDED ..." in script


def test_generate_plumed_opes_script_opes_expanded_rejects_temp_max_leq_t(
    tmp_path: Path,
) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    with pytest.raises(ValueError, match="opes_expanded_temp_max"):
        generate_plumed_opes_script(
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
            opes_expanded_temp_max=300.0,
        )


def test_generate_plumed_opes_script_opes_metad_explore(tmp_path: Path) -> None:
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
        opes_variant="explore",
    )
    assert "opes: OPES_METAD_EXPLORE ..." in script


def test_generate_plumed_opes_script_use_pbc_requires_whole_molecule(
    tmp_path: Path,
) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    with pytest.raises(ValueError, match="whole_molecule"):
        generate_plumed_opes_script(
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
            use_pbc=True,
            whole_molecule_plumed_idx=None,
        )


def test_generate_plumed_opes_script_use_pbc_wholemolecules_no_nopbc(
    tmp_path: Path,
) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    whole = list(range(1, 21))
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
        use_pbc=True,
        whole_molecule_plumed_idx=whole,
    )
    assert "WHOLEMOLECULES ENTITY0=" in script
    assert "NOPBC" not in script
    assert "lig_com: CENTER ATOMS=5,6,7" in script
    assert "lig_dist: DISTANCE ATOMS=lig_com,pocket_com" in script


def test_default_opes_upper_wall_dist_angstrom() -> None:
    from genai_tps.simulation.openmm_md_runner import (
        default_opes_upper_wall_dist_angstrom,
    )

    assert default_opes_upper_wall_dist_angstrom(5.2) == pytest.approx(15.2)
    assert default_opes_upper_wall_dist_angstrom(0.0, margin_angstrom=5.0) == pytest.approx(
        5.0
    )


def test_generate_plumed_opes_script_upper_wall_extra_bias(tmp_path: Path) -> None:
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
        upper_wall_dist=15.0,
        upper_wall_kappa=180.0,
    )
    assert "lig_dist_uwall: UPPER_WALLS ARG=lig_dist AT=15" in script
    assert "KAPPA=180" in script
    assert "  EXTRA_BIAS=lig_dist_uwall.bias" in script


def test_generate_plumed_opes_script_3d_requires_pocket_heavy(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    with pytest.raises(ValueError, match="pocket_heavy"):
        generate_plumed_opes_script(
            ligand_plumed_idx=[5, 6, 7],
            pocket_ca_plumed_idx=[1, 2, 3],
            rmsd_reference_pdb=ref_path,
            sigma=(0.3, 0.5, 1.0),
            pace=500,
            barrier=40.0,
            biasfactor=10.0,
            temperature=300.0,
            save_opes_every=50_000,
            progress_every=10_000,
            out_dir=tmp_path,
            cv_mode="3d",
            pocket_heavy_plumed_idx=[],
        )


def test_generate_plumed_opes_script_oneopes_projection_and_contactmap(
    tmp_path: Path,
) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    script = generate_plumed_opes_script(
        ligand_plumed_idx=[5, 6, 7],
        pocket_ca_plumed_idx=[1, 2, 3],
        rmsd_reference_pdb=ref_path,
        sigma=(0.25, 0.4),
        pace=500,
        barrier=40.0,
        biasfactor=10.0,
        temperature=300.0,
        save_opes_every=50_000,
        progress_every=10_000,
        out_dir=tmp_path,
        cv_mode="oneopes",
        oneopes_axis_p0_plumed_idx=[1, 2],
        oneopes_axis_p1_plumed_idx=[3],
        oneopes_contactmap_pairs_plumed=[(10, 5), (11, 6)],
        upper_wall_dist=8.0,
    )
    assert "oneopes_axis_p0: CENTER NOPBC ATOMS=1,2" in script
    assert "oneopes_axis_p1: CENTER NOPBC ATOMS=3" in script
    assert "pp: PROJECTION_ON_AXIS AXIS_ATOMS=oneopes_axis_p0,oneopes_axis_p1" in script
    assert "cmap: CONTACTMAP SUM ATOMS1=10,5 ATOMS2=11,6" in script
    assert "SWITCH={RATIONAL R_0=5.5" in script
    assert "pp_ext_uwall: UPPER_WALLS ARG=pp.ext" in script
    assert "EXTRA_BIAS=pp_ext_uwall.bias" in script
    assert "ARG=pp.proj,cmap" in script
    assert "lig_rmsd,lig_dist,pp.proj,pp.ext,cmap,opes.bias" in script


def test_generate_plumed_opes_script_oneopes_water_auxiliary_opes(tmp_path: Path) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    script = generate_plumed_opes_script(
        ligand_plumed_idx=[5, 6, 7],
        pocket_ca_plumed_idx=[1, 2, 3],
        rmsd_reference_pdb=ref_path,
        sigma=(0.2, 0.35),
        pace=500,
        barrier=40.0,
        biasfactor=10.0,
        temperature=300.0,
        save_opes_every=50_000,
        progress_every=10_000,
        out_dir=tmp_path,
        cv_mode="oneopes",
        oneopes_axis_p0_plumed_idx=[1],
        oneopes_axis_p1_plumed_idx=[2],
        oneopes_contactmap_pairs_plumed=[(9, 5)],
        oneopes_hydration_spot_plumed_idx=[5, 20],
        water_oxygen_plumed_idx=[100, 101, 102],
        oneopes_water_pace=20000,
        oneopes_water_barrier=4.0,
    )
    assert "WO: GROUP ATOMS=100,101,102" in script
    assert "hydr_0: COORDINATION GROUPA=5 GROUPB=WO" in script
    assert "hydr_1: COORDINATION GROUPA=20 GROUPB=WO" in script
    assert "opes_hydr_0: OPES_METAD_EXPLORE ..." in script
    assert "opes_hydr_1: OPES_METAD_EXPLORE ..." in script
    assert "FILE=" in script and "KERNELS_HYDR_0" in script
    assert "opes_hydr_0.bias" in script and "opes_hydr_1.bias" in script


def test_generate_plumed_opes_script_oneopes_rejects_hydration_without_water(
    tmp_path: Path,
) -> None:
    from genai_tps.simulation.plumed_opes import generate_plumed_opes_script

    ref_path = tmp_path / "reference_pose.pdb"
    with pytest.raises(ValueError, match="water_oxygen"):
        generate_plumed_opes_script(
            ligand_plumed_idx=[5, 6, 7],
            pocket_ca_plumed_idx=[1, 2, 3],
            rmsd_reference_pdb=ref_path,
            sigma=(0.2, 0.3),
            pace=500,
            barrier=40.0,
            biasfactor=10.0,
            temperature=300.0,
            save_opes_every=50_000,
            progress_every=10_000,
            out_dir=tmp_path,
            cv_mode="oneopes",
            oneopes_axis_p0_plumed_idx=[1],
            oneopes_axis_p1_plumed_idx=[2],
            oneopes_contactmap_pairs_plumed=[(9, 5)],
            oneopes_hydration_spot_plumed_idx=[5],
            water_oxygen_plumed_idx=None,
        )


def test_compute_oneopes_pp_ext_and_default_axis() -> None:
    import numpy as np

    from genai_tps.simulation.plumed_opes import (
        compute_oneopes_pp_ext_angstrom,
        default_oneopes_axis_boltz_indices,
    )

    ref = np.zeros((20, 3), dtype=np.float64)
    ref[1] = (0.0, 0.0, 0.0)
    ref[2] = (0.0, 0.0, 10.0)
    ref[5] = (1.0, 2.0, 5.0)
    ext = compute_oneopes_pp_ext_angstrom(
        ref,
        axis_p0_boltz=[1],
        axis_p1_boltz=[2],
        ligand_boltz=[5],
    )
    assert ext == pytest.approx(np.sqrt(1.0**2 + 2.0**2))

    p0, p1 = default_oneopes_axis_boltz_indices(
        pocket_ca_boltz=np.array([10, 11, 12, 13], dtype=np.int64),
        ref_coords_angstrom=ref,
        ligand_boltz=np.array([5], dtype=np.int64),
    )
    assert len(p0) >= 1 and len(p1) >= 1
    assert set(p0.tolist()) | set(p1.tolist()) == {10, 11, 12, 13}


def test_build_md_simulation_from_pdb_exposes_extra_forces_hook() -> None:
    import compute_cv_rmsd

    sig = inspect.signature(compute_cv_rmsd.build_md_simulation_from_pdb)

    assert "extra_forces" in sig.parameters
    assert "platform_properties" in sig.parameters


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
