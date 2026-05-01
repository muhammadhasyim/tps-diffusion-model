"""Unit tests for OneOPES Hamiltonian replica exchange scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from genai_tps.simulation.oneopes_repex import (
    LITERATURE_MULTITHERMAL_OPES_EXPANDED_PACE_STEPS,
    assert_context_budget,
    assign_replicas_to_devices,
    build_stratified_replica_plumed_kwargs,
    copy_opes_state_snapshot,
    count_active_contexts_per_device,
    default_expanded_temp_max_for_replica,
    literature_oneopes_replica_specs,
    multithermal_temp_max_k_for_replica,
    log_acceptance_two_replica_hrex,
    metropolis_accept_two_replica_hrex,
    minimum_max_active_contexts_per_device,
    opes_states_fingerprint,
    paper_host_guest_replica_strats,
    plan_evaluator_device_assignments,
    plan_evaluator_devices_for_neighbor_pair,
    prepare_evaluator_scratch_tree,
    register_oneopes_repex_arguments,
    relocate_plumed_deck_paths,
    repex_active_context_counts_for_json,
    validate_oneopes_repex_cli_args,
    write_repex_config_json,
)
from genai_tps.simulation.oneopes_repex_smoke_validate import (
    validate_exchange_log_multi_rows,
    validate_exchange_log_two_replica_rows,
    validate_repex_smoke_directory,
)
from genai_tps.simulation.oneopes_upstream_reference import (
    HREX_NEIGHBOR_PAIRS_PHASE_A,
    HREX_NEIGHBOR_PAIRS_PHASE_B,
    PEFEMA_AUXILIARY_CV_LABELS,
    PEFEMA_MULTITHERMAL_TEMP_MAX_K,
    neighbor_hrex_pairs_for_phase,
    neighbor_hrex_pairs_for_phase_n,
    pefema_auxiliary_labels_for_replica,
    pefema_multithermal_temp_max_k,
)
from genai_tps.simulation.plumed_opes import OpesPlumedDeckConfig, generate_plumed_opes_script_from_config


def test_literature_two_replica_specs() -> None:
    specs = literature_oneopes_replica_specs(n_replicas=2)
    assert len(specs) == 2
    assert specs[0].force_empty_oneopes_hydration is True
    assert specs[1].oneopes_hydration_site_cap == 1


@pytest.mark.parametrize("flag", ["--dry-run", "--dry-run-use-cpu", "--no-dry-run-use-cpu"])
def test_oneopes_repex_cli_rejects_dry_run_flags(flag: str) -> None:
    parser = argparse.ArgumentParser()
    register_oneopes_repex_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([flag])


def test_oneopes_repex_requires_gpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import genai_tps.simulation.plumed_kernel as plumed_kernel

    monkeypatch.setattr(plumed_kernel, "assert_plumed_opes_metad_available", lambda: None)
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(
        bias_cv="oneopes",
        opes_mode="plumed",
        exchange_every=1,
        oneopes_protocol="legacy-boltz",
        n_replicas=2,
        platform="CPU",
        opes_expanded_temp_max=None,
        temperature=300.0,
        oneopes_hydration_max_sites=1,
    )

    with pytest.raises(SystemExit):
        validate_oneopes_repex_cli_args(args, parser)


def test_literature_eight_replica_multithermal_flags() -> None:
    specs = literature_oneopes_replica_specs(n_replicas=8)
    assert len(specs) == 8
    assert specs[3].multithermal is False
    assert specs[4].multithermal is True
    assert default_expanded_temp_max_for_replica(4, thermostat_k=298.15) == pytest.approx(
        multithermal_temp_max_k_for_replica(4, thermostat_k=298.15, hottest_t_max_k=None)
    )


def test_literature_replica_specs_truncates_full_ladder() -> None:
    full = literature_oneopes_replica_specs(n_replicas=8)
    for n in (3, 4, 5, 6, 7):
        cut = literature_oneopes_replica_specs(n_replicas=n)
        assert len(cut) == n
        assert cut == full[:n]


def test_literature_replica_specs_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="2..8"):
        literature_oneopes_replica_specs(n_replicas=1)
    with pytest.raises(ValueError, match="2..8"):
        literature_oneopes_replica_specs(n_replicas=9)


def test_neighbor_hrex_pairs_for_phase_n_matches_eight() -> None:
    assert neighbor_hrex_pairs_for_phase_n(0, 8) == HREX_NEIGHBOR_PAIRS_PHASE_A
    assert neighbor_hrex_pairs_for_phase_n(1, 8) == HREX_NEIGHBOR_PAIRS_PHASE_B
    assert neighbor_hrex_pairs_for_phase(0) == HREX_NEIGHBOR_PAIRS_PHASE_A
    assert neighbor_hrex_pairs_for_phase(1) == HREX_NEIGHBOR_PAIRS_PHASE_B


def test_neighbor_hrex_pairs_for_phase_n_three_replicas() -> None:
    assert neighbor_hrex_pairs_for_phase_n(0, 3) == ((0, 1),)
    assert neighbor_hrex_pairs_for_phase_n(1, 3) == ((1, 2),)


@pytest.mark.parametrize(
    ("n", "phase", "expected"),
    [
        (4, 0, ((0, 1), (2, 3))),
        (4, 1, ((1, 2),)),
        (5, 0, ((0, 1), (2, 3))),
        (5, 1, ((1, 2), (3, 4))),
        (6, 0, ((0, 1), (2, 3), (4, 5))),
        (6, 1, ((1, 2), (3, 4))),
        (7, 0, ((0, 1), (2, 3), (4, 5))),
        (7, 1, ((1, 2), (3, 4), (5, 6))),
    ],
)
def test_neighbor_hrex_pairs_for_phase_n_four_to_seven(
    n: int, phase: int, expected: tuple[tuple[int, int], ...]
) -> None:
    assert neighbor_hrex_pairs_for_phase_n(phase, n) == expected


def test_repex_active_context_counts_for_json_two_vs_multi() -> None:
    rep3 = [0, 0, 0]
    ev3 = plan_evaluator_device_assignments(rep3, placement="same-device")
    assert repex_active_context_counts_for_json(3, rep3, ev3) == {0: 5}
    assert minimum_max_active_contexts_per_device(3, rep3, ev3) == 5

    rep2 = [0, 0]
    ev2 = plan_evaluator_device_assignments(rep2, placement="same-device")
    assert repex_active_context_counts_for_json(2, rep2, ev2) == count_active_contexts_per_device(
        rep2, ev2
    )
    assert repex_active_context_counts_for_json(2, rep2, ev2)[0] == 4


def test_write_repex_config_json_uses_peak_for_three_replicas(tmp_path: Path) -> None:
    import json

    rep = [0, 0, 0]
    ev = plan_evaluator_device_assignments(rep, placement="same-device")
    path = tmp_path / "repex_config.json"
    write_repex_config_json(
        path,
        replica_devices=rep,
        evaluator_devices=ev,
        max_active_contexts_per_device=5,
        exchange_every=2000,
        n_replicas=3,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["active_context_counts_by_device"]["0"] == 5
    assert data["max_active_contexts_per_device"] == 5


def test_validate_exchange_log_two_replica_rows_ok() -> None:
    row = {
        "md_step": "3000",
        "u00": "-1.0",
        "u01": "-2.0",
        "u10": "-3.0",
        "u11": "-4.0",
        "log_accept": "0.0",
        "accepted": "1",
        "rng_u": "0.5",
        "md_elapsed_s": "1",
        "exchange_elapsed_s": "1",
        "elapsed_s": "2",
    }
    assert validate_exchange_log_two_replica_rows([row]) == []


def test_validate_exchange_log_two_replica_rows_rejects_nan() -> None:
    row = {
        "md_step": "3000",
        "u00": "nan",
        "u01": "0.0",
        "u10": "0.0",
        "u11": "0.0",
        "log_accept": "0.0",
        "accepted": "0",
        "rng_u": "0.5",
        "md_elapsed_s": "1",
        "exchange_elapsed_s": "1",
        "elapsed_s": "2",
    }
    errs = validate_exchange_log_two_replica_rows([row])
    assert any("non-finite" in e for e in errs)


def test_validate_repex_smoke_directory_minimal_two_replica(tmp_path: Path) -> None:
    import json

    root = tmp_path / "smoke"
    root.mkdir()
    (root / "repex_config.json").write_text(
        json.dumps({"n_replicas": 2, "exchange_every": 3000}),
        encoding="utf-8",
    )
    header = (
        "md_step,u00,u01,u10,u11,log_accept,accepted,rng_u,"
        "md_elapsed_s,exchange_elapsed_s,elapsed_s\n"
    )
    row = "3000,-1.0,-2.0,-3.0,-4.0,0.0,1,0.5,1.0,1.0,3.0\n"
    (root / "exchange_log.csv").write_text(header + row, encoding="utf-8")
    (root / "barrier_timing.jsonl").write_text(
        json.dumps({"md_step": 3000, "elapsed_s": 3.0, "n_replicas": 2}) + "\n",
        encoding="utf-8",
    )
    rep = root / "rep000"
    rep.mkdir()
    opes = rep / "opes_states"
    opes.mkdir()
    (opes / "KERNELS").write_text("k", encoding="utf-8")
    (opes / "STATE").write_text("s", encoding="utf-8")

    errs = validate_repex_smoke_directory(root)
    assert errs == []


def test_validate_exchange_log_multi_rows_ok() -> None:
    row = {
        "md_step": "2000",
        "phase_mod_2": "0",
        "pair_i": "0",
        "pair_j": "1",
        "u_ii": "-1.0",
        "u_jj": "-2.0",
        "u_ij": "-1.5",
        "u_ji": "-1.5",
        "log_accept": "0.0",
        "accepted": "0",
        "rng_u": "0.9",
        "md_elapsed_s": "1",
        "exchange_elapsed_s": "0.5",
        "elapsed_s": "2",
    }
    assert validate_exchange_log_multi_rows([row]) == []


def test_minimum_max_active_contexts_three_on_one_gpu() -> None:
    rep = [0, 0, 0]
    ev = plan_evaluator_device_assignments(rep, placement="same-device")
    assert minimum_max_active_contexts_per_device(3, rep, ev) == 5


def test_assign_replicas_round_robin_and_packed() -> None:
    assert assign_replicas_to_devices(3, [0, 1], "round-robin") == [0, 1, 0]
    assert assign_replicas_to_devices(3, [0, 1], "packed") == [0, 0, 1]


def test_assign_replicas_explicit() -> None:
    assert assign_replicas_to_devices(
        2, [0, 1], "explicit", explicit_map=[1, 0]
    ) == [1, 0]


def test_assign_replicas_errors() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        assign_replicas_to_devices(1, [], "round-robin")
    with pytest.raises(ValueError, match="explicit"):
        assign_replicas_to_devices(2, [0], "explicit", explicit_map=None)


def test_context_budget_oversubscription() -> None:
    counts = {0: 5}
    with pytest.raises(RuntimeError, match="Device 0"):
        assert_context_budget(counts, max_active_contexts_per_device=4)


def test_count_active_contexts() -> None:
    c = count_active_contexts_per_device([0, 1], [0, None])
    assert c[0] == 2
    assert c[1] == 1


def test_minimum_max_active_contexts_two_replica_same_device() -> None:
    rep = [0, 0]
    ev = plan_evaluator_device_assignments(rep, placement="same-device")
    assert minimum_max_active_contexts_per_device(2, rep, ev) == 4


def test_minimum_max_active_contexts_eight_on_one_gpu() -> None:
    rep = [0] * 8
    ev = plan_evaluator_device_assignments(rep, placement="same-device")
    assert minimum_max_active_contexts_per_device(8, rep, ev) == 10


def test_minimum_max_active_contexts_eight_split_gpus() -> None:
    rep = [0, 0, 0, 0, 1, 1, 1, 1]
    ev = plan_evaluator_device_assignments(rep, placement="same-device")
    assert minimum_max_active_contexts_per_device(8, rep, ev) == 6


def test_plan_evaluator_devices() -> None:
    assert plan_evaluator_device_assignments([0, 1], placement="same-device") == [0, 1]
    assert plan_evaluator_device_assignments([2, 2], placement="serial") == [2, 2]
    assert plan_evaluator_device_assignments([0, 1, 2, 3], placement="same-device") == [
        0,
        1,
        2,
        3,
    ]


def test_plan_evaluator_devices_neighbor_pair() -> None:
    assert plan_evaluator_devices_for_neighbor_pair(
        [0, 1, 2, 3], 1, 2, placement="same-device"
    ) == (1, 2)


def test_pefema_upstream_constants() -> None:
    assert PEFEMA_AUXILIARY_CV_LABELS == ("L4", "V6", "L1", "V8", "V4", "V10", "V2")
    assert pefema_auxiliary_labels_for_replica(0) == ()
    assert pefema_auxiliary_labels_for_replica(3) == ("L4", "V6", "L1")
    assert pefema_multithermal_temp_max_k(4) == 310.0
    assert PEFEMA_MULTITHERMAL_TEMP_MAX_K[7] == 370.0
    assert neighbor_hrex_pairs_for_phase(0) == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert neighbor_hrex_pairs_for_phase(1) == ((1, 2), (3, 4), (5, 6))


def test_paper_host_guest_strats_multithermal_temps() -> None:
    rows = paper_host_guest_replica_strats()
    assert rows[0].auxiliary_cv_count == 0
    assert rows[3].auxiliary_cv_count == 3
    assert rows[3].multithermal is False
    assert rows[4].multithermal is True
    assert rows[4].multithermal_temp_max_k == pytest.approx(310.0)


def test_stratified_paper_host_guest_kwargs() -> None:
    k7 = build_stratified_replica_plumed_kwargs(
        7,
        n_replicas=8,
        user_expanded_temp_max=None,
        user_expanded_pace=None,
        thermostat_temperature_k=298.15,
        oneopes_protocol="paper_host_guest",
    )
    assert k7["oneopes_protocol"] == "paper_host_guest"
    assert k7["opes_expanded_temp_max_override"] == pytest.approx(370.0)
    assert k7["opes_expanded_pace_override"] == LITERATURE_MULTITHERMAL_OPES_EXPANDED_PACE_STEPS


def test_stratified_paper_host_guest_can_disable_energy_multithermal() -> None:
    k7 = build_stratified_replica_plumed_kwargs(
        7,
        n_replicas=8,
        user_expanded_temp_max=None,
        user_expanded_pace=None,
        thermostat_temperature_k=298.15,
        oneopes_protocol="paper_host_guest",
        enable_energy_multithermal=False,
    )
    assert k7["oneopes_protocol"] == "paper_host_guest"
    assert k7["opes_expanded_temp_max_override"] is None
    assert k7["opes_expanded_pace_override"] is None


def test_opes_states_fingerprint_stable_on_copy(tmp_path: Path) -> None:
    src = tmp_path / "opes_states"
    src.mkdir()
    (src / "STATE").write_text("state", encoding="utf-8")
    (src / "KERNELS_AUX_L4").write_text("k", encoding="utf-8")
    h_before = opes_states_fingerprint(src)
    dst = tmp_path / "dst" / "opes_states"
    copy_opes_state_snapshot(src, dst)
    assert h_before == opes_states_fingerprint(src)
    assert opes_states_fingerprint(dst)["STATE"] == h_before["STATE"]


def test_metropolis_rejects_nonfinite_energies() -> None:
    rng = __import__("random").Random(0)
    ok, log_a, _ = metropolis_accept_two_replica_hrex(
        float("nan"), 0.0, 0.0, 0.0, rng
    )
    assert ok is False
    assert not __import__("math").isfinite(log_a)


def test_metropolis_symmetric_zero_swap() -> None:
    rng = __import__("random").Random(0)
    ok, log_a, u = metropolis_accept_two_replica_hrex(1.0, 1.0, 1.0, 1.0, rng)
    assert log_a == 0.0
    assert ok is True


def test_metropolis_manual_asymmetric() -> None:
    class _NearOneRng:
        def random(self) -> float:
            return 1.0 - 1e-15

    u00, u01, u10, u11 = 0.0, 10.0, 10.0, 0.0
    la = log_acceptance_two_replica_hrex(u00, u01, u10, u11)
    assert la == pytest.approx(-20.0)
    ok, _, _ = metropolis_accept_two_replica_hrex(u00, u01, u10, u11, _NearOneRng())
    assert ok is False


def test_relocate_and_prepare_evaluator_script(tmp_path: Path) -> None:
    prod = tmp_path / "rep000"
    prod.mkdir()
    opes = prod / "opes_states"
    opes.mkdir()
    (opes / "STATE").write_text("state", encoding="utf-8")
    (opes / "STATE_AUX_L4").write_text("state aux", encoding="utf-8")
    (opes / "KERNELS").write_text("k", encoding="utf-8")
    raw = "\n".join(
        [
            f"REF={prod / 'plumed_rmsd_reference.pdb'}",
            f"FILE={opes / 'KERNELS'}",
            f"  STATE_WFILE={opes / 'STATE'}",
            f"  STATE_WFILE={opes / 'STATE_AUX_L4'}",
            "",
        ]
    )
    (prod / "plumed_opes.dat").write_text(raw, encoding="utf-8")
    (prod / "plumed_rmsd_reference.pdb").write_text("HEADER\n", encoding="utf-8")

    scratch = tmp_path / "scratch_eval"
    (scratch / "opes_states").mkdir(parents=True)
    copy_opes_state_snapshot(opes, scratch / "opes_states")
    sp = prepare_evaluator_scratch_tree(production_rep_root=prod, scratch_root=scratch)
    text = sp.read_text(encoding="utf-8")
    assert str(scratch) in text
    assert str(prod) not in text
    assert f"STATE_RFILE={scratch / 'opes_states' / 'STATE'}" in text
    assert f"STATE_RFILE={scratch / 'opes_states' / 'STATE_AUX_L4'}" in text


def test_prepare_evaluator_script_skips_empty_state_rfile(tmp_path: Path) -> None:
    prod = tmp_path / "rep000"
    prod.mkdir()
    opes = prod / "opes_states"
    opes.mkdir()
    (opes / "STATE").write_text("", encoding="utf-8")
    raw = "\n".join(
        [
            f"FILE={opes / 'KERNELS'}",
            f"  STATE_WFILE={opes / 'STATE'}",
            "",
        ]
    )
    (prod / "plumed_opes.dat").write_text(raw, encoding="utf-8")

    scratch = tmp_path / "scratch_eval"
    (scratch / "opes_states").mkdir(parents=True)
    copy_opes_state_snapshot(opes, scratch / "opes_states")
    sp = prepare_evaluator_scratch_tree(production_rep_root=prod, scratch_root=scratch)
    text = sp.read_text(encoding="utf-8")

    assert f"STATE_WFILE={scratch / 'opes_states' / 'STATE'}" in text
    assert "STATE_RFILE=" not in text


def test_copy_snapshot_leaves_source_mtime(tmp_path: Path) -> None:
    src = tmp_path / "opes_states"
    src.mkdir()
    p = src / "STATE"
    p.write_text("x", encoding="utf-8")
    m0 = p.stat().st_mtime_ns
    dst = tmp_path / "dst"
    copy_opes_state_snapshot(src, dst)
    assert p.read_text() == "x"
    assert p.stat().st_mtime_ns == m0


def test_stratified_kwargs_two_replicas() -> None:
    k0 = build_stratified_replica_plumed_kwargs(
        0,
        n_replicas=2,
        user_expanded_temp_max=500.0,
        user_expanded_pace=999,
        thermostat_temperature_k=300.0,
    )
    assert k0["oneopes_protocol"] == "legacy_boltz"
    assert k0["force_empty_oneopes_hydration"] is True
    assert k0["opes_expanded_temp_max_override"] is None
    assert k0["opes_expanded_pace_override"] is None
    k1 = build_stratified_replica_plumed_kwargs(
        1,
        n_replicas=2,
        user_expanded_temp_max=None,
        user_expanded_pace=None,
        thermostat_temperature_k=300.0,
    )
    assert k1["oneopes_hydration_site_cap"] == 1
    assert k1["opes_expanded_temp_max_override"] is None


def test_stratified_multithermal_pace_and_interpolation() -> None:
    k4 = build_stratified_replica_plumed_kwargs(
        4,
        n_replicas=8,
        user_expanded_temp_max=None,
        user_expanded_pace=None,
        thermostat_temperature_k=298.15,
    )
    assert k4["opes_expanded_pace_override"] == LITERATURE_MULTITHERMAL_OPES_EXPANDED_PACE_STEPS
    assert k4["opes_expanded_temp_max_override"] == pytest.approx(318.15)
    k7 = build_stratified_replica_plumed_kwargs(
        7,
        n_replicas=8,
        user_expanded_temp_max=398.15,
        user_expanded_pace=55,
        thermostat_temperature_k=298.15,
    )
    assert k7["opes_expanded_pace_override"] == 55
    assert k7["opes_expanded_temp_max_override"] == pytest.approx(398.15)
    k5 = build_stratified_replica_plumed_kwargs(
        5,
        n_replicas=8,
        user_expanded_temp_max=398.15,
        user_expanded_pace=None,
        thermostat_temperature_k=298.15,
    )
    assert k5["opes_expanded_temp_max_override"] == pytest.approx(348.15)


def test_deck_rep0_vs_rep1_auxiliary_line(tmp_path: Path) -> None:
    """Replica 1 deck should mention auxiliary OPES lines when hydration sites exist."""
    ref = tmp_path / "ref.pdb"
    ref.write_text("END\n", encoding="utf-8")
    common = dict(
        ligand_plumed_idx=[1],
        pocket_ca_plumed_idx=[2, 3, 4],
        rmsd_reference_pdb=ref,
        sigma=(0.2, 0.3),
        pace=100,
        barrier=40.0,
        biasfactor=8.0,
        temperature=300.0,
        save_opes_every=1000,
        progress_every=500,
        cv_mode="oneopes",
        pocket_heavy_plumed_idx=[2, 3, 4],
        oneopes_axis_p0_plumed_idx=[2],
        oneopes_axis_p1_plumed_idx=[3],
        oneopes_contactmap_pairs_plumed=[(10, 1)],
    )
    c0 = OpesPlumedDeckConfig(
        **common,
        out_dir=tmp_path / "opes0",
        oneopes_hydration_spot_plumed_idx=None,
        water_oxygen_plumed_idx=None,
    )
    c1 = OpesPlumedDeckConfig(
        **common,
        out_dir=tmp_path / "opes1",
        oneopes_hydration_spot_plumed_idx=(9,),
        water_oxygen_plumed_idx=tuple(range(100, 110)),
    )
    s0 = generate_plumed_opes_script_from_config(c0)
    s1 = generate_plumed_opes_script_from_config(c1)
    assert "opes_hydr_0" not in s0
    assert "opes_hydr_0" in s1


def test_relocate_plumed_deck_paths_simple(tmp_path: Path) -> None:
    a = tmp_path / "foo" / "rep"
    b = tmp_path / "bar" / "scratch"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    t = f"X {a}/plumed Y"
    assert relocate_plumed_deck_paths(t, a, b) == f"X {b}/plumed Y"


@pytest.mark.parametrize(
    ("u00", "u01", "u10", "u11", "expected"),
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0, 4.0, -(2.0 + 3.0 - 1.0 - 4.0)),
    ],
)
def test_log_acceptance_param(
    u00: float, u01: float, u10: float, u11: float, expected: float
) -> None:
    assert log_acceptance_two_replica_hrex(u00, u01, u10, u11) == pytest.approx(expected)
