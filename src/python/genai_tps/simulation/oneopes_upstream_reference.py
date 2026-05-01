"""Frozen reference metadata for the Febrer Martinez *et al.* OneOPES host–guest ladder.

These constants are distilled from the public automation in
`Pefema/OneOpes_protocol` (not vendored here), especially
``scripts/automated_protocol/9_multireplica.py``, and cross-checked against the
PLUMED-NEST egg `plumID:23.011` (OneOPES reference inputs; PLUMED 2.9 actions
``OPES_METAD_EXPLORE``, ``ECV_MULTITHERMAL``, ``OPES_EXPANDED``).

See also :mod:`genai_tps.simulation.oneopes_repex` for OpenMM scheduling parity.
"""

from __future__ import annotations

from typing import Final

# --- Pefema / paper pacing (MD steps; barriers in kJ/mol) ---
PEFEMA_MAIN_OPES_METAD_EXPLORE_PACE: Final[int] = 10_000
PEFEMA_MAIN_OPES_METAD_EXPLORE_BARRIER_KJMOL: Final[float] = 100.0
PEFEMA_AUXILIARY_OPES_PACE: Final[int] = 20_000
PEFEMA_AUXILIARY_OPES_BARRIER_KJMOL: Final[float] = 3.0
PEFEMA_MULTITHERMAL_OPES_EXPANDED_PACE: Final[int] = 100
PEFEMA_HREX_ATTEMPT_STRIDE_STEPS: Final[int] = 1000

# --- Auxiliary MultiCV / coordination labels (replica 1 → first only, …, rep 7 → all) ---
PEFEMA_AUXILIARY_CV_LABELS: Final[tuple[str, ...]] = (
    "L4",
    "V6",
    "L1",
    "V8",
    "V4",
    "V10",
    "V2",
)

# --- Multithermal tier at 298 K thermostat (Febrer Martinez automated recipe) ---
PEFEMA_MULTITHERMAL_TEMP_MAX_K: Final[dict[int, float]] = {
    4: 310.0,
    5: 330.0,
    6: 350.0,
    7: 370.0,
}

# --- Paper / Pefema water coordination switching (Å; PLUMED rational NN/MM; neighbor list) ---
PAPER_WL_COORD_NN: Final[int] = 6
PAPER_WL_COORD_MM: Final[int] = 10
PAPER_WH_COORD_NN: Final[int] = 2
PAPER_WH_COORD_MM: Final[int] = 6
PAPER_WATER_COORD_R0_ANGSTROM: Final[float] = 2.5
PAPER_WATER_COORD_D_MAX_ANGSTROM: Final[float] = 0.8
PAPER_WATER_COORD_NL_CUTOFF_ANGSTROM: Final[float] = 1.5
PAPER_WATER_COORD_NL_STRIDE: Final[int] = 20

# GROMACS-style neighbor Hamiltonian exchange: alternate even/odd bond pairs.
HREX_NEIGHBOR_PAIRS_PHASE_A: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
)
HREX_NEIGHBOR_PAIRS_PHASE_B: Final[tuple[tuple[int, int], ...]] = (
    (1, 2),
    (3, 4),
    (5, 6),
)


def neighbor_hrex_pairs_for_phase_n(
    phase_index: int, n_replicas: int
) -> tuple[tuple[int, int], ...]:
    """Alternating even/odd bond pairs for *n_replicas* (GROMACS ``-hrex`` schedule).

    Even *phase_index*: ``(0,1),(2,3),…``; odd *phase_index*: ``(1,2),(3,4),…``.
    Indices are skipped when they would exceed ``n_replicas - 1`` (same pattern as
    replica 7 idle in phase B on an eight-replica ring).
    """
    n = int(n_replicas)
    if n < 2:
        raise ValueError("n_replicas must be >= 2.")
    pairs: list[tuple[int, int]] = []
    if int(phase_index) % 2 == 0:
        k = 0
        while k + 1 < n:
            pairs.append((k, k + 1))
            k += 2
    else:
        k = 1
        while k + 1 < n:
            pairs.append((k, k + 1))
            k += 2
    return tuple(pairs)


def neighbor_hrex_pairs_for_phase(phase_index: int) -> tuple[tuple[int, int], ...]:
    """Same as :func:`neighbor_hrex_pairs_for_phase_n` with eight replicas."""
    return neighbor_hrex_pairs_for_phase_n(phase_index, 8)


def pefema_auxiliary_labels_for_replica(replica_index: int) -> tuple[str, ...]:
    """Return the ordered auxiliary CV labels active on replica *replica_index*.

    Replica **0** has none. Replicas **1–7** include the first *replica_index*
    entries of :data:`PEFEMA_AUXILIARY_CV_LABELS`.
    """
    r = int(replica_index)
    if r <= 0:
        return ()
    if r > len(PEFEMA_AUXILIARY_CV_LABELS):
        raise ValueError("replica_index must be in 0..7 for the eight-replica ladder.")
    return PEFEMA_AUXILIARY_CV_LABELS[:r]


def pefema_multithermal_temp_max_k(replica_index: int) -> float | None:
    """Return fixed ``TEMP_MAX`` (K) for replica 4–7, else ``None``."""
    return PEFEMA_MULTITHERMAL_TEMP_MAX_K.get(int(replica_index))


__all__ = [
    "HREX_NEIGHBOR_PAIRS_PHASE_A",
    "HREX_NEIGHBOR_PAIRS_PHASE_B",
    "neighbor_hrex_pairs_for_phase_n",
    "PAPER_WATER_COORD_D_MAX_ANGSTROM",
    "PAPER_WATER_COORD_NL_CUTOFF_ANGSTROM",
    "PAPER_WATER_COORD_NL_STRIDE",
    "PAPER_WATER_COORD_R0_ANGSTROM",
    "PAPER_WH_COORD_MM",
    "PAPER_WH_COORD_NN",
    "PAPER_WL_COORD_MM",
    "PAPER_WL_COORD_NN",
    "PEFEMA_AUXILIARY_CV_LABELS",
    "PEFEMA_AUXILIARY_OPES_BARRIER_KJMOL",
    "PEFEMA_AUXILIARY_OPES_PACE",
    "PEFEMA_HREX_ATTEMPT_STRIDE_STEPS",
    "PEFEMA_MAIN_OPES_METAD_EXPLORE_BARRIER_KJMOL",
    "PEFEMA_MAIN_OPES_METAD_EXPLORE_PACE",
    "PEFEMA_MULTITHERMAL_OPES_EXPANDED_PACE",
    "PEFEMA_MULTITHERMAL_TEMP_MAX_K",
    "neighbor_hrex_pairs_for_phase",
    "pefema_auxiliary_labels_for_replica",
    "pefema_multithermal_temp_max_k",
]
