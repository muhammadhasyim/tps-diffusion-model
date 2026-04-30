"""Unit tests for OneOPES Hamiltonian replica exchange scaffolding."""

from __future__ import annotations

import argparse
import threading
import time
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
    opes_states_fingerprint,
    paper_host_guest_replica_strats,
    plan_evaluator_device_assignments,
    plan_evaluator_devices_for_neighbor_pair,
    prepare_evaluator_scratch_tree,
    register_oneopes_repex_arguments,
    relocate_plumed_deck_paths,
    resolve_replica_step_workers,
    step_replicas_parallel,
    validate_oneopes_repex_cli_args,
)
from genai_tps.simulation.oneopes_upstream_reference import (
    PEFEMA_AUXILIARY_CV_LABELS,
    PEFEMA_MULTITHERMAL_TEMP_MAX_K,
    neighbor_hrex_pairs_for_phase,
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


def test_resolve_replica_step_workers_auto_and_explicit() -> None:
    assert resolve_replica_step_workers(0, n_replicas=8) == 8
    assert resolve_replica_step_workers(None, n_replicas=8) == 8
    assert resolve_replica_step_workers(3, n_replicas=8) == 3
    with pytest.raises(ValueError, match="positive"):
        resolve_replica_step_workers(-1, n_replicas=8)


def test_step_replicas_parallel_runs_concurrently() -> None:
    class _FakeSimulation:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def step(self, chunk: int) -> None:
            time.sleep(0.05)
            self.calls.append(chunk)

    sims = [_FakeSimulation() for _ in range(8)]
    t0 = time.perf_counter()
    elapsed = step_replicas_parallel(sims, 17, max_workers=8)
    wall = time.perf_counter() - t0

    assert all(sim.calls == [17] for sim in sims)
    assert elapsed <= wall
    assert wall < 0.20


def test_step_replicas_parallel_propagates_failures() -> None:
    started: set[int] = set()
    lock = threading.Lock()

    class _FakeSimulation:
        def __init__(self, replica_index: int) -> None:
            self.replica_index = replica_index

        def step(self, chunk: int) -> None:
            del chunk
            with lock:
                started.add(self.replica_index)
            if self.replica_index == 3:
                raise RuntimeError("replica failed")
            time.sleep(0.01)

    sims = [_FakeSimulation(i) for i in range(8)]
    with pytest.raises(RuntimeError, match="replica failed"):
        step_replicas_parallel(sims, 5, max_workers=8)
    assert started == set(range(8))


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
    (opes / "KERNELS").write_text("k", encoding="utf-8")
    raw = f"REF={prod / 'plumed_rmsd_reference.pdb'}\nFILE={opes / 'KERNELS'}\n"
    (prod / "plumed_opes.dat").write_text(raw, encoding="utf-8")
    (prod / "plumed_rmsd_reference.pdb").write_text("HEADER\n", encoding="utf-8")

    scratch = tmp_path / "scratch_eval"
    (scratch / "opes_states").mkdir(parents=True)
    copy_opes_state_snapshot(opes, scratch / "opes_states")
    sp = prepare_evaluator_scratch_tree(production_rep_root=prod, scratch_root=scratch)
    text = sp.read_text(encoding="utf-8")
    assert str(scratch) in text
    assert str(prod) not in text


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
