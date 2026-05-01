"""GPU-native OneOPES replica exchange infrastructure.

This module uses one Python process per GPU.  Each process owns a
``GPUReplicaGroup`` containing all replicas assigned to that GPU.  Local
exchanges stay inside one process and use the optional OpenMM CUDA tensor bridge
for device-to-device copies.  Remote exchanges use ``torch.distributed`` with
NCCL when more than one GPU process is active.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from genai_tps.simulation.oneopes_repex import (
    ExchangeAttempt,
    _bias_energy_kj_mol,
    _parse_devices_csv,
    _reduced_u,
    _serialize_system,
    assign_replicas_to_devices,
    build_stratified_replica_plumed_kwargs,
    copy_opes_state_snapshot,
    kbt_kj_per_mol,
    metropolis_accept_two_replica_hrex,
    prepare_evaluator_scratch_tree,
    refresh_evaluator_context,
    write_repex_config_json,
)
from genai_tps.simulation.oneopes_upstream_reference import neighbor_hrex_pairs_for_phase_n
from genai_tps.utils.nvtx_util import nvtx_range

try:  # Optional accelerator: tests and CPU fallback do not require it.
    import openmm_cuda_bridge as _cuda_bridge
except Exception:  # pragma: no cover - import depends on local CUDA build.
    _cuda_bridge = None


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError("distributed REPEX requires PyTorch") from exc
    return torch


def _dist() -> Any:
    torch = _require_torch()
    return torch.distributed


def _has_bridge(context: Any) -> bool:
    if _cuda_bridge is None:
        return False
    try:
        return str(context.getPlatform().getName()) == "CUDA"
    except Exception:
        return False


def _torch_tensor_from_positions(context: Any) -> Any:
    torch = _require_torch()
    if _has_bridge(context):
        return _cuda_bridge.positions_to_tensor(context)
    import openmm.unit as unit

    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return torch.as_tensor(np.asarray(positions, dtype=np.float64))


def _torch_tensor_from_velocities(context: Any) -> Any:
    torch = _require_torch()
    if _has_bridge(context):
        return _cuda_bridge.velocities_to_tensor(context)
    import openmm.unit as unit

    state = context.getState(getVelocities=True)
    velocities = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    return torch.as_tensor(np.asarray(velocities, dtype=np.float64))


def _set_positions_from_tensor(context: Any, tensor: Any) -> None:
    if _has_bridge(context) and getattr(tensor, "is_cuda", False):
        _cuda_bridge.tensor_to_positions(context, tensor.contiguous())
        return
    import openmm as mm
    import openmm.unit as unit

    host = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    positions = [mm.Vec3(*map(float, row)) for row in host]
    context.setPositions(positions * unit.nanometer)


def _set_velocities_from_tensor(context: Any, tensor: Any) -> None:
    if _has_bridge(context) and getattr(tensor, "is_cuda", False):
        _cuda_bridge.tensor_to_velocities(context, tensor.contiguous())
        return
    import openmm as mm
    import openmm.unit as unit

    host = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    velocities = [mm.Vec3(*map(float, row)) for row in host]
    context.setVelocities(velocities * unit.nanometer / unit.picosecond)


@dataclass(frozen=True)
class ReplicaAssignment:
    """Mapping of one replica index to one GPU rank/device."""

    replica_index: int
    rank: int
    device: int


def assign_replica_groups(replica_devices: Sequence[int]) -> tuple[list[int], dict[int, int], dict[int, list[int]]]:
    """Return unique devices, ``replica -> rank``, and ``rank -> replicas`` mapping."""
    devices = sorted({int(dev) for dev in replica_devices})
    rank_for_device = {dev: rank for rank, dev in enumerate(devices)}
    replica_to_rank: dict[int, int] = {}
    rank_to_replicas: dict[int, list[int]] = {rank: [] for rank in range(len(devices))}
    for replica, dev in enumerate(replica_devices):
        rank = rank_for_device[int(dev)]
        replica_to_rank[int(replica)] = int(rank)
        rank_to_replicas[int(rank)].append(int(replica))
    return devices, replica_to_rank, rank_to_replicas


def setup_per_replica_logging(rank: int, out_root: Path, replica_indices: Sequence[int]) -> None:
    """Route process logs to rank and replica-specific files."""
    out_root.mkdir(parents=True, exist_ok=True)
    rank_log = out_root / f"rank{rank:03d}.log"
    formatter = logging.Formatter(f"[rank{rank:03d}] %(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(rank_log, mode="a", encoding="utf-8")
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if rank == 0:
        stream = logging.StreamHandler(sys.__stdout__)
        stream.setFormatter(formatter)
        root.addHandler(stream)
    for rep in replica_indices:
        rep_dir = out_root / f"rep{int(rep):03d}"
        rep_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(rank_log, "a", buffering=1, encoding="utf-8")
    sys.stderr = open(out_root / f"rank{rank:03d}.err", "a", buffering=1, encoding="utf-8")


class GPUReplicaGroup:
    """Manage all replicas assigned to one GPU rank."""

    def __init__(
        self,
        *,
        rank: int,
        gpu_id: int | None,
        replica_indices: Sequence[int],
        args: Any | None = None,
        out_root: Path | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.rank = int(rank)
        self.gpu_id = None if gpu_id is None else int(gpu_id)
        self.replica_indices = [int(r) for r in replica_indices]
        self.args = args
        self.out_root = Path(out_root) if out_root is not None else Path(".")
        self.rng = rng if rng is not None else random.Random(0)
        self.sims: dict[int, Any] = {}
        self.eval_ctxs: dict[int, Any] = {}
        self.packs: dict[int, dict[str, Any]] = {}
        self.plumed_contexts: dict[int, dict[str, Any]] = {}
        self.replica_dirs: dict[int, Path] = {
            rep: self.out_root / f"rep{rep:03d}" for rep in self.replica_indices
        }
        temperature = 300.0 if args is None else float(getattr(args, "temperature", 300.0))
        self.kbt_kjmol = kbt_kj_per_mol(temperature)

    def add_pack(self, replica: int, pack: dict[str, Any]) -> None:
        """Register an initialized OpenMM simulation bundle."""
        rep = int(replica)
        self.packs[rep] = pack
        self.sims[rep] = pack["sim"]
        self.plumed_contexts[rep] = dict(pack.get("plumed_context", {}))
        self.replica_dirs[rep] = Path(pack.get("out", self.out_root / f"rep{rep:03d}"))

    def step_all(self, n_steps: int) -> float:
        """Step all local replicas in parallel."""
        if int(n_steps) < 1:
            raise ValueError("n_steps must be >= 1")
        t0 = time.time()
        with nvtx_range(f"repex_group_step rank={self.rank} reps={len(self.sims)} steps={n_steps}"):
            with ThreadPoolExecutor(max_workers=max(1, len(self.sims))) as executor:
                futures = [executor.submit(sim.step, int(n_steps)) for sim in self.sims.values()]
                for future in futures:
                    future.result()
        return time.time() - t0

    def get_positions(self, replica: int) -> Any:
        return _torch_tensor_from_positions(self.sims[int(replica)].context)

    def set_positions(self, replica: int, tensor: Any) -> None:
        _set_positions_from_tensor(self.sims[int(replica)].context, tensor)

    def get_velocities(self, replica: int) -> Any:
        return _torch_tensor_from_velocities(self.sims[int(replica)].context)

    def set_velocities(self, replica: int, tensor: Any) -> None:
        _set_velocities_from_tensor(self.sims[int(replica)].context, tensor)

    def get_bias_energy(self, replica: int) -> float:
        plumed_ctx = self.plumed_contexts[int(replica)]
        return _bias_energy_kj_mol(
            self.sims[int(replica)].context,
            int(plumed_ctx.get("force_group", 30)),
        )

    def evaluate_on_positions(self, replica: int, positions: Any) -> float:
        """Evaluate this replica's Hamiltonian on external positions."""
        rep = int(replica)
        ctx = self.eval_ctxs.get(rep)
        if ctx is None:
            ctx = self.sims[rep].context
        _set_positions_from_tensor(ctx, positions)
        force_group = int(self.plumed_contexts.get(rep, {}).get("force_group", 30))
        return _bias_energy_kj_mol(ctx, force_group)

    def local_exchange(self, i: int, j: int, *, md_step: int = 0) -> ExchangeAttempt:
        """Attempt exchange between two replicas owned by this GPU process."""
        pos_i = self.get_positions(i)
        pos_j = self.get_positions(j)
        vel_i = self.get_velocities(i)
        vel_j = self.get_velocities(j)
        u_ii = _reduced_u(self.get_bias_energy(i), kbt_kjmol=self.kbt_kjmol)
        u_jj = _reduced_u(self.get_bias_energy(j), kbt_kjmol=self.kbt_kjmol)
        u_ij = _reduced_u(self.evaluate_on_positions(i, pos_j), kbt_kjmol=self.kbt_kjmol)
        u_ji = _reduced_u(self.evaluate_on_positions(j, pos_i), kbt_kjmol=self.kbt_kjmol)
        accepted, log_accept, rng_u = metropolis_accept_two_replica_hrex(
            u_ii, u_ij, u_ji, u_jj, rng=self.rng
        )
        if accepted:
            self.set_positions(i, pos_j)
            self.set_positions(j, pos_i)
            self.set_velocities(i, vel_j)
            self.set_velocities(j, vel_i)
        return ExchangeAttempt(
            md_step=int(md_step),
            u00=float(u_ii),
            u01=float(u_ij),
            u10=float(u_ji),
            u11=float(u_jj),
            log_accept=float(log_accept),
            accepted=bool(accepted),
            rng_uniform=float(rng_u),
        )

    def remote_exchange(
        self,
        *,
        local_replica: int,
        peer_replica: int,
        peer_rank: int,
        lower_replica: int,
        md_step: int,
    ) -> ExchangeAttempt | None:
        """Attempt exchange against a replica owned by another GPU process."""
        torch = _require_torch()
        dist = _dist()
        local_replica = int(local_replica)
        peer_rank = int(peer_rank)
        send_pos = self.get_positions(local_replica).contiguous()
        send_vel = self.get_velocities(local_replica).contiguous()
        recv_pos = torch.empty_like(send_pos)
        recv_vel = torch.empty_like(send_vel)

        if self.rank < peer_rank:
            dist.send(send_pos, dst=peer_rank)
            dist.recv(recv_pos, src=peer_rank)
            dist.send(send_vel, dst=peer_rank)
            dist.recv(recv_vel, src=peer_rank)
        else:
            dist.recv(recv_pos, src=peer_rank)
            dist.send(send_pos, dst=peer_rank)
            dist.recv(recv_vel, src=peer_rank)
            dist.send(send_vel, dst=peer_rank)

        own_u = _reduced_u(self.get_bias_energy(local_replica), kbt_kjmol=self.kbt_kjmol)
        cross_u = _reduced_u(
            self.evaluate_on_positions(local_replica, recv_pos),
            kbt_kjmol=self.kbt_kjmol,
        )
        local_values = torch.tensor(
            [own_u, cross_u], dtype=torch.float64, device=send_pos.device
        )
        peer_values = torch.empty_like(local_values)
        if self.rank < peer_rank:
            dist.send(local_values, dst=peer_rank)
            dist.recv(peer_values, src=peer_rank)
        else:
            dist.recv(peer_values, src=peer_rank)
            dist.send(local_values, dst=peer_rank)

        decision = torch.zeros(1, dtype=torch.int64, device=send_pos.device)
        attempt: ExchangeAttempt | None = None
        if int(local_replica) == int(lower_replica):
            u_ii = float(local_values[0].item())
            u_ij = float(local_values[1].item())
            u_jj = float(peer_values[0].item())
            u_ji = float(peer_values[1].item())
            accepted, log_accept, rng_u = metropolis_accept_two_replica_hrex(
                u_ii, u_ij, u_ji, u_jj, rng=self.rng
            )
            decision[0] = 1 if accepted else 0
            attempt = ExchangeAttempt(
                md_step=int(md_step),
                u00=u_ii,
                u01=u_ij,
                u10=u_ji,
                u11=u_jj,
                log_accept=float(log_accept),
                accepted=bool(accepted),
                rng_uniform=float(rng_u),
            )
            dist.send(decision, dst=peer_rank)
        else:
            dist.recv(decision, src=peer_rank)

        if int(decision.item()) == 1:
            self.set_positions(local_replica, recv_pos)
            self.set_velocities(local_replica, recv_vel)
        return attempt


def distributed_exchange_step(
    group: GPUReplicaGroup,
    *,
    replica_to_rank: dict[int, int],
    phase: int,
    n_replicas: int,
    md_step: int,
) -> list[tuple[int, int, ExchangeAttempt]]:
    """Attempt all neighbor exchanges for one phase."""
    attempts: list[tuple[int, int, ExchangeAttempt]] = []
    for i, j in neighbor_hrex_pairs_for_phase_n(int(phase), int(n_replicas)):
        rank_i = int(replica_to_rank[int(i)])
        rank_j = int(replica_to_rank[int(j)])
        if rank_i == rank_j:
            if group.rank == rank_i:
                attempts.append((int(i), int(j), group.local_exchange(i, j, md_step=md_step)))
            continue
        if group.rank == rank_i:
            attempt = group.remote_exchange(
                local_replica=i,
                peer_replica=j,
                peer_rank=rank_j,
                lower_replica=min(i, j),
                md_step=md_step,
            )
            if attempt is not None:
                attempts.append((int(i), int(j), attempt))
        elif group.rank == rank_j:
            attempt = group.remote_exchange(
                local_replica=j,
                peer_replica=i,
                peer_rank=rank_i,
                lower_replica=min(i, j),
                md_step=md_step,
            )
            if attempt is not None:
                attempts.append((int(i), int(j), attempt))
    return attempts


def _write_exchange_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "md_step",
                "phase_mod_2",
                "pair_i",
                "pair_j",
                "u_ii",
                "u_jj",
                "u_ij",
                "u_ji",
                "log_accept",
                "accepted",
                "rng_u",
                "md_elapsed_s",
                "exchange_elapsed_s",
                "elapsed_s",
            ],
        )
        writer.writeheader()


def _append_exchange_rows(
    path: Path,
    *,
    phase: int,
    attempts: Sequence[tuple[int, int, ExchangeAttempt]],
    md_elapsed_s: float,
    exchange_elapsed_s: float,
    elapsed_s: float,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "md_step",
                "phase_mod_2",
                "pair_i",
                "pair_j",
                "u_ii",
                "u_jj",
                "u_ij",
                "u_ji",
                "log_accept",
                "accepted",
                "rng_u",
                "md_elapsed_s",
                "exchange_elapsed_s",
                "elapsed_s",
            ],
        )
        for i, j, attempt in attempts:
            writer.writerow(
                {
                    "md_step": attempt.md_step,
                    "phase_mod_2": int(phase) % 2,
                    "pair_i": int(i),
                    "pair_j": int(j),
                    "u_ii": f"{attempt.u00:.10g}",
                    "u_jj": f"{attempt.u11:.10g}",
                    "u_ij": f"{attempt.u01:.10g}",
                    "u_ji": f"{attempt.u10:.10g}",
                    "log_accept": f"{attempt.log_accept:.10g}",
                    "accepted": 1 if attempt.accepted else 0,
                    "rng_u": f"{attempt.rng_uniform:.10g}",
                    "md_elapsed_s": f"{float(md_elapsed_s):.6f}",
                    "exchange_elapsed_s": f"{float(exchange_elapsed_s):.6f}",
                    "elapsed_s": f"{float(elapsed_s):.6f}",
                }
            )


def _gather_attempts(
    local_attempts: list[tuple[int, int, ExchangeAttempt]],
    *,
    world_size: int,
    rank: int,
) -> list[tuple[int, int, ExchangeAttempt]]:
    if int(world_size) <= 1:
        return local_attempts
    dist = _dist()
    gathered: list[Any] | None = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_attempts, object_gather_list=gathered, dst=0)
    if rank != 0:
        return []
    out: list[tuple[int, int, ExchangeAttempt]] = []
    assert gathered is not None
    for item in gathered:
        out.extend(item or [])
    return sorted(out, key=lambda row: (row[2].md_step, row[0], row[1]))


def _barrier(world_size: int) -> None:
    if int(world_size) > 1:
        _dist().barrier()


def _init_process_group(rank: int, world_size: int, backend: str) -> None:
    if int(world_size) <= 1:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    _dist().init_process_group(backend=backend, rank=int(rank), world_size=int(world_size))


def _destroy_process_group(world_size: int) -> None:
    if int(world_size) > 1:
        _dist().destroy_process_group()


def _enable_energy_multithermal(args: Any) -> bool:
    proto = str(getattr(args, "oneopes_protocol", "legacy-boltz")).replace("-", "_")
    return False if proto == "paper_host_guest" else bool(getattr(args, "opes_expanded_temp_max", None) is not None)


def _replica_stratification(args: Any, replica: int, user_temp_max: float | None) -> dict[str, Any]:
    proto = str(getattr(args, "oneopes_protocol", "legacy-boltz")).replace("-", "_")
    return build_stratified_replica_plumed_kwargs(
        int(replica),
        n_replicas=int(args.n_replicas),
        user_expanded_temp_max=user_temp_max,
        user_expanded_pace=args.opes_expanded_pace,
        thermostat_temperature_k=float(args.temperature),
        oneopes_protocol=proto,
        paper_multithermal_pace=int(getattr(args, "paper_oneopes_multithermal_pace", 100)),
        enable_energy_multithermal=_enable_energy_multithermal(args),
    )


def _initialize_replica(
    *,
    args: Any,
    replica: int,
    device: int,
    out_root: Path,
    shared_pdb: Path | None,
    user_temp_max: float | None,
) -> dict[str, Any]:
    from genai_tps.simulation.openmm_md_runner import run_opes_md

    replica_args = copy.copy(args)
    replica_args.openmm_device_index = int(device)
    if shared_pdb is not None and int(replica) != 0:
        replica_args.openmm_prepared_pdb = shared_pdb
    pack = run_opes_md(
        replica_args,
        md_out_dir=out_root / f"rep{int(replica):03d}",
        plumed_factory_extra_kwargs=_replica_stratification(args, replica, user_temp_max),
        return_after_initialization=True,
    )
    if not isinstance(pack, dict):
        raise RuntimeError("run_opes_md did not return an initialization bundle")
    return pack


def _write_shared_pdb(pack: dict[str, Any], path: Path) -> None:
    import openmm.app

    path.parent.mkdir(parents=True, exist_ok=True)
    state = pack["sim"].context.getState(getPositions=True)
    with path.open("w", encoding="utf-8") as handle:
        openmm.app.PDBFile.writeFile(pack["sim"].topology, state.getPositions(), handle)


def _broadcast_shared_pdb_path(rank: int, world_size: int, path: Path | None) -> Path | None:
    if int(world_size) <= 1:
        return path
    payload: list[str | None] = [str(path) if path is not None else None]
    _dist().broadcast_object_list(payload, src=0)
    return Path(payload[0]) if payload[0] is not None else None


def repex_worker(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    devices_by_rank: Sequence[int],
    rank_to_replicas: dict[int, list[int]],
    replica_to_rank: dict[int, int],
) -> None:
    """Worker entry point for one GPU process."""
    torch = _require_torch()
    rank = int(rank)
    world_size = int(world_size)
    gpu_id = int(devices_by_rank[rank]) if devices_by_rank else None
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    backend = "nccl" if world_size > 1 and torch.cuda.is_available() else "gloo"
    _init_process_group(rank, world_size, backend)
    out_root = Path(args.out).expanduser().resolve()
    my_replicas = [int(r) for r in rank_to_replicas.get(rank, [])]
    setup_per_replica_logging(rank, out_root, my_replicas)
    rng = random.Random(0 + rank)
    group = GPUReplicaGroup(
        rank=rank,
        gpu_id=gpu_id,
        replica_indices=my_replicas,
        args=args,
        out_root=out_root,
        rng=rng,
    )
    user_temp_max = getattr(args, "paper_oneopes_temp_max", None)
    if user_temp_max is None:
        user_temp_max = getattr(args, "opes_expanded_temp_max", None)

    shared_pdb = out_root / "shared_solvated_template.pdb"
    if rank == 0 and 0 in my_replicas:
        pack0 = _initialize_replica(
            args=args,
            replica=0,
            device=int(gpu_id or 0),
            out_root=out_root,
            shared_pdb=None,
            user_temp_max=user_temp_max,
        )
        group.add_pack(0, pack0)
        _write_shared_pdb(pack0, shared_pdb)
    shared_pdb = _broadcast_shared_pdb_path(rank, world_size, shared_pdb if rank == 0 else None)
    _barrier(world_size)

    for rep in my_replicas:
        if rep in group.sims:
            continue
        pack = _initialize_replica(
            args=args,
            replica=rep,
            device=int(gpu_id or 0),
            out_root=out_root,
            shared_pdb=shared_pdb,
            user_temp_max=user_temp_max,
        )
        group.add_pack(rep, pack)

    _prepare_evaluator_contexts(group, args)
    _barrier(world_size)

    exchange_log = out_root / "exchange_log.csv"
    barrier_log = out_root / "barrier_timing.jsonl"
    if rank == 0:
        _write_exchange_header(exchange_log)
        barrier_log.write_text("", encoding="utf-8")
    _barrier(world_size)

    t_start = time.time()
    completed = 0
    phase = 0
    while completed < int(args.n_steps):
        chunk = min(int(args.exchange_every), int(args.n_steps) - completed)
        md_elapsed = group.step_all(chunk)
        completed += chunk
        _barrier(world_size)
        t_exchange0 = time.time()
        local_attempts = distributed_exchange_step(
            group,
            replica_to_rank=replica_to_rank,
            phase=phase,
            n_replicas=int(args.n_replicas),
            md_step=completed,
        )
        exchange_elapsed = time.time() - t_exchange0
        all_attempts = _gather_attempts(local_attempts, world_size=world_size, rank=rank)
        if rank == 0:
            elapsed = time.time() - t_start
            _append_exchange_rows(
                exchange_log,
                phase=phase,
                attempts=all_attempts,
                md_elapsed_s=md_elapsed,
                exchange_elapsed_s=exchange_elapsed,
                elapsed_s=elapsed,
            )
            with barrier_log.open("a", encoding="utf-8") as handle:
                import json

                handle.write(
                    json.dumps(
                        {
                            "md_step": int(completed),
                            "phase_mod_2": int(phase) % 2,
                            "elapsed_s": elapsed,
                            "n_replicas": int(args.n_replicas),
                            "backend": backend,
                        }
                    )
                    + "\n"
                )
        phase += 1
        _barrier(world_size)
    _destroy_process_group(world_size)


def _prepare_evaluator_contexts(group: GPUReplicaGroup, args: Any) -> None:
    """Create one scratch evaluator context per local replica."""
    if not group.sims:
        return
    import openmm as mm
    from genai_tps.utils.compute_device import openmm_device_index_properties

    platform = mm.Platform.getPlatformByName(str(args.platform))
    props = openmm_device_index_properties(str(args.platform), group.gpu_id)
    scratch_root = group.out_root / "evaluator_scratch" / f"rank{group.rank:03d}"
    for rep, sim in group.sims.items():
        plumed_ctx = group.plumed_contexts[rep]
        prod_dir = group.replica_dirs[rep]
        scratch_dir = scratch_root / f"rep{rep:03d}"
        opes_states = prod_dir / "opes_states"
        if opes_states.is_dir():
            copy_opes_state_snapshot(opes_states, scratch_dir / "opes_states")
        script = prepare_evaluator_scratch_tree(
            production_rep_root=prod_dir,
            scratch_root=scratch_dir,
        )
        state = sim.context.getState(getPositions=True)
        group.eval_ctxs[rep] = refresh_evaluator_context(
            template_system_xml=_serialize_system(sim.system),
            plumed_force_index=int(plumed_ctx["force_index"]),
            plumed_script_path=script,
            temperature_k=float(args.temperature),
            force_group=int(plumed_ctx.get("force_group", 30)),
            platform=platform,
            platform_properties=props,
            positions=state.getPositions(),
            box_vectors=state.getPeriodicBoxVectors(),
        )


def run_distributed_repex(args: argparse.Namespace) -> None:
    """Public launcher for GPU-native distributed OneOPES REPEX."""
    torch = _require_torch()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    devices = _parse_devices_csv(str(getattr(args, "devices", "0")))
    explicit_map = None
    if str(getattr(args, "replica_device_map", "round-robin")) == "explicit":
        explicit_map = _parse_devices_csv(str(getattr(args, "replica_devices")))
    replica_devices = assign_replicas_to_devices(
        int(args.n_replicas),
        devices,
        getattr(args, "replica_device_map", "round-robin"),
        explicit_map=explicit_map,
    )
    devices_by_rank, replica_to_rank, rank_to_replicas = assign_replica_groups(replica_devices)
    write_repex_config_json(
        out_root / "repex_config.json",
        replica_devices=replica_devices,
        evaluator_devices=[None] * int(args.n_replicas),
        max_active_contexts_per_device=int(getattr(args, "max_active_contexts_per_device", 0)),
        exchange_every=int(args.exchange_every),
        n_replicas=int(args.n_replicas),
        extra={
            "devices": devices,
            "replica_device_map": str(getattr(args, "replica_device_map", "round-robin")),
            "replica_to_rank": {str(k): int(v) for k, v in sorted(replica_to_rank.items())},
            "rank_to_replicas": {str(k): [int(x) for x in v] for k, v in sorted(rank_to_replicas.items())},
            "engine": "gpu-native-repex",
        },
    )
    world_size = len(devices_by_rank)
    if world_size == 1:
        repex_worker(0, 1, args, devices_by_rank, rank_to_replicas, replica_to_rank)
        return
    torch.multiprocessing.spawn(
        repex_worker,
        nprocs=world_size,
        args=(world_size, args, devices_by_rank, rank_to_replicas, replica_to_rank),
    )
