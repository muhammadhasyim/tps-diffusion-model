"""OpenPathSampling TPS loop for Boltz diffusion (fixed-length, sigma-based states)."""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, TextIO

import numpy as np
import torch
import openpathsampling as paths
from openpathsampling.engines.trajectory import Trajectory
from openpathsampling.pathmover import (
    BackwardShootMover,
    ForwardShootMover,
    OneWayShootingMover,
    SpecializedRandomChoiceMover,
)
from openpathsampling.high_level import move_strategy

from genai_tps.backends.boltz.path_probability import (
    backward_shooting_metropolis_bias,
    min_metropolis_acceptance_path,
    trajectory_log_path_prob,
)
from genai_tps.backends.boltz.snapshot import BoltzSnapshot
from genai_tps.backends.boltz.states import state_volume_high_sigma, state_volume_low_sigma

if TYPE_CHECKING:
    from genai_tps.backends.boltz.engine import BoltzDiffusionEngine
    from genai_tps.backends.boltz.gpu_core import BoltzSamplerCore
    from genai_tps.enhanced_sampling import EnhancedSamplingBias

_tps_logger = logging.getLogger(__name__)


def trajectory_has_finite_coords(traj: Trajectory) -> bool:
    """Return True only if every frame in *traj* has all-finite coordinates.

    Checks GPU tensors when available (avoids a device transfer) and falls
    back to the NumPy coordinate array.
    """
    for snap in traj:
        tc = getattr(snap, "_tensor_coords_gpu", None)
        if tc is not None:
            if not torch.isfinite(tc).all():
                return False
        else:
            c = getattr(snap, "coordinates", None)
            if c is not None and not np.all(np.isfinite(c)):
                return False
    return True


def _rep0_trajectory(sample_set: Any) -> Trajectory | None:
    rep0 = [s for s in sample_set.samples if s.replica == 0]
    if not rep0:
        rep0 = sample_set.samples
    return rep0[0].trajectory if rep0 else None


def _extract_metropolis_acceptance(change: Any) -> float | None:
    """Best-effort extraction of Metropolis acceptance probability from nested MoveChange."""
    seen: set[int] = set()
    cur: Any = change
    for _ in range(12):
        if cur is None:
            return None
        oid = id(cur)
        if oid in seen:
            break
        seen.add(oid)
        det = getattr(cur, "details", None)
        if det is not None:
            p = getattr(det, "metropolis_acceptance", None)
            if p is not None:
                return float(p)
        sub = getattr(cur, "subchange", None)
        if sub is not None:
            cur = sub
            continue
        subs = getattr(cur, "subchanges", None)
        if subs and len(subs) == 1:
            cur = subs[0]
            continue
        break
    return None


def sigma_tps_state_volumes(core: "BoltzSamplerCore") -> tuple[Any, Any]:
    """Build state A (high :math:`\\sigma`) and B (low :math:`\\sigma`) from the sampler schedule.

    Frame 0 of a forward trajectory uses ``schedule[0].sigma_tm``; the last frame uses the
    final ``sigma_t`` from the forward map. Thresholds are chosen so a typical noise-to-
    structure rollout satisfies the fixed-length TPS ensemble.
    """
    sig0 = float(core.schedule[0].sigma_tm)
    # Last step reaches sigma_t near 0; keep a generous upper band for state B.
    sigma_last = float(core.schedule[-1].sigma_t)
    margin_b = max(1.0, abs(sigma_last) * 50.0 + 1e-3)
    state_a = state_volume_high_sigma(sigma_min=0.45 * sig0, sigma_max=1.0e9)
    state_b = state_volume_low_sigma(sigma_max=margin_b, sigma_min=0.0)
    return state_a, state_b


def build_fixed_length_tps_network(
    state_a: Any,
    state_b: Any,
    path_length: int,
) -> paths.FixedLengthTPSNetwork:
    """``path_length`` = number of frames (``num_sampling_steps + 1`` for a full rollout)."""
    return paths.FixedLengthTPSNetwork(
        initial_states=[state_a],
        final_states=[state_b],
        length=path_length,
    )


def tps_ensemble(network: paths.FixedLengthTPSNetwork) -> Any:
    """Single sampling ensemble for the A→B transition."""
    return network.sampling_transitions[0].ensembles[0]


class _NaNRejectingShootMixin:
    """Mixin: set trial bias to 0 when the trial trajectory contains non-finite coordinates."""

    def _reject_if_nan(self, trial, trial_trajectory):
        if not trajectory_has_finite_coords(trial_trajectory):
            _tps_logger.warning(
                "Trial trajectory has non-finite coordinates; forcing rejection."
            )
            trial.bias = 0.0

    def _apply_enhanced_bias(self, trial, input_sample, trial_trajectory):
        """Multiply trial.bias by the enhanced sampling acceptance factor.

        The enhanced bias and CV function are stored on ``self.engine`` by
        :func:`run_tps_path_sampling` when enhanced sampling is active.
        """
        es_bias = getattr(self.engine, "_enhanced_bias", None)
        cv_fn = getattr(self.engine, "_cv_function", None)
        if es_bias is None or cv_fn is None:
            return
        try:
            cv_old = cv_fn(input_sample.trajectory)
            cv_new = cv_fn(trial_trajectory)
            factor = es_bias.compute_acceptance_factor(cv_old, cv_new)
            trial.bias *= factor
        except Exception as exc:
            _tps_logger.warning(
                "Enhanced bias evaluation failed (%s); leaving trial.bias unchanged.", exc
            )


class BoltzDiffusionForwardShootMover(_NaNRejectingShootMixin, ForwardShootMover):
    """Forward shooting that auto-rejects trajectories with NaN coordinates."""

    def _build_sample(self, input_sample, shooting_index, trial_trajectory,
                      stopping_reason=None, run_details=None):
        trial, trial_details = super()._build_sample(
            input_sample, shooting_index, trial_trajectory,
            stopping_reason, run_details,
        )
        self._apply_enhanced_bias(trial, input_sample, trial_trajectory)
        self._reject_if_nan(trial, trial_trajectory)
        return trial, trial_details


class BoltzDiffusionBackwardShootMover(_NaNRejectingShootMixin, BackwardShootMover):
    """Backward shooting with re-noising (``direction=-1``) and diffusion Hastings bias."""

    def _make_backward_trajectory(self, trajectory, shooting_index):
        initial_snapshot = trajectory[shooting_index].reversed
        run_f = paths.SuffixTrajectoryEnsemble(
            self.target_ensemble,
            trajectory[shooting_index + 1:],
        ).can_prepend
        partial = self.engine.generate(
            initial_snapshot,
            running=[run_f],
            direction=-1,
        )
        return partial + trajectory[shooting_index + 1 :]

    def _build_sample(
        self,
        input_sample,
        shooting_index,
        trial_trajectory,
        stopping_reason=None,
        run_details=None,
    ):
        if run_details is None:
            run_details = {}
        trial, trial_details = super()._build_sample(
            input_sample,
            shooting_index,
            trial_trajectory,
            stopping_reason,
            run_details,
        )
        if stopping_reason is None:
            core = getattr(self.engine, "core", None)
            if core is not None:
                trial.bias *= backward_shooting_metropolis_bias(
                    input_sample.trajectory,
                    trial_trajectory,
                    shooting_index,
                    core,
                )
        self._apply_enhanced_bias(trial, input_sample, trial_trajectory)
        self._reject_if_nan(trial, trial_trajectory)
        return trial, trial_details


class GlobalReshuffleMover(paths.PathMover):
    """Full-path resample: draw fresh ``x_0`` and all noise variables, then forward-integrate.

    This move complements forward and backward shooting to ensure full ergodicity on
    the reactive path ensemble.  The acceptance ratio is identically 1 for any
    reactive trial path because old and new paths are drawn i.i.d. from the same
    Gaussian prior:

    .. math::

        \\frac{\\pi(X^{\\mathrm{new}})}{\\pi(X^{\\mathrm{old}})}
        \\cdot \\frac{q(X^{\\mathrm{old}})}{q(X^{\\mathrm{new}})}
        = \\frac{\\rho(x_0^{\\mathrm{new}}) \\prod_i \\pi_i(\\xi_i^{\\mathrm{new}})}
               {\\rho(x_0^{\\mathrm{old}}) \\prod_i \\pi_i(\\xi_i^{\\mathrm{old}})}
        \\cdot \\frac{\\rho(x_0^{\\mathrm{old}}) \\prod_i \\pi_i(\\xi_i^{\\mathrm{old}})}
               {\\rho(x_0^{\\mathrm{new}}) \\prod_i \\pi_i(\\xi_i^{\\mathrm{new}})}
        = 1.

    Non-reactive trial paths (those that do not end in state B) are rejected.

    Parameters
    ----------
    ensemble
        The TPS ensemble (fixed-length A→B).
    engine
        A :class:`~genai_tps.backends.boltz.engine.BoltzDiffusionEngine` with a
        ``generate_n_frames`` method.  The engine is used to forward-integrate a
        full new path from a fresh initial noise draw.
    """

    def __init__(self, ensemble, engine: "BoltzDiffusionEngine"):
        super().__init__()
        self.ensemble = ensemble
        self.engine = engine

    def move(self, sample_set: paths.SampleSet) -> paths.PathMoveChange:
        """Propose a globally resampled reactive path."""
        rep0 = [s for s in sample_set.samples if s.replica == 0]
        if not rep0:
            rep0 = list(sample_set.samples)
        input_sample = rep0[0]
        old_traj = input_sample.trajectory
        n_frames = len(old_traj)

        try:
            # Reset engine to a fresh initial noise configuration before generating
            x0 = self.engine.core.sample_initial_noise()
            snap0 = BoltzSnapshot.from_gpu_batch(
                x0,
                step_index=0,
                sigma=float(self.engine.core.schedule[0].sigma_tm),
            )
            self.engine.current_snapshot = snap0
            new_traj = self.engine.generate_n_frames(n_frames - 1)
        except Exception as exc:  # noqa: BLE001
            _tps_logger.warning("GlobalReshuffleMover: engine error (%s); rejecting.", exc)
            trial = paths.Sample(
                replica=input_sample.replica,
                trajectory=old_traj,
                ensemble=self.ensemble,
                bias=0.0,
            )
            return paths.RejectedSampleMoveChange(
                [trial],
                mover=self,
                input_samples=[input_sample],
            )

        reactive = bool(self.ensemble(new_traj, candidate=True))
        has_finite = trajectory_has_finite_coords(new_traj)
        bias = 1.0 if (reactive and has_finite) else 0.0

        if not reactive or not has_finite:
            _tps_logger.debug(
                "GlobalReshuffleMover: trial not reactive or has NaN; rejecting."
            )

        if bias > 0.0:
            es_bias = getattr(self.engine, "_enhanced_bias", None)
            cv_fn = getattr(self.engine, "_cv_function", None)
            if es_bias is not None and cv_fn is not None:
                try:
                    cv_old = cv_fn(old_traj)
                    cv_new = cv_fn(new_traj)
                    bias *= es_bias.compute_acceptance_factor(cv_old, cv_new)
                except Exception as exc:
                    _tps_logger.warning(
                        "GlobalReshuffleMover: enhanced bias failed (%s).", exc
                    )

        trial = paths.Sample(
            replica=input_sample.replica,
            trajectory=new_traj,
            ensemble=self.ensemble,
            bias=bias,
        )
        if bias > 0.0:
            return paths.AcceptedSampleMoveChange(
                [trial],
                mover=self,
                input_samples=[input_sample],
            )
        return paths.RejectedSampleMoveChange(
            [trial],
            mover=self,
            input_samples=[input_sample],
        )


class BoltzDiffusionOneWayShootingMover(OneWayShootingMover):
    """50/50 forward / Boltz-corrected backward shooting, or forward-only when requested.

    Parameters
    ----------
    forward_only
        When ``True``, only forward shooting moves are used (no backward re-noising).
        Forward shooting always accepts reactive paths (uniform selector, path weight
        ratio = 1 for identical suffix), making it a guaranteed-correct baseline.
    reshuffle_probability
        Fraction of MC steps that use a global reshuffle move (resample ``x_0`` and
        all noise draws fresh, then forward-integrate the full path).  The reshuffle
        always accepts when the trial path is reactive (path-weight ratio = 1 under
        i.i.d. Gaussian prior).  Default ``0.1`` (10 %).  Set to ``0`` to disable.
    """

    def __init__(
        self,
        ensemble,
        selector,
        engine=None,
        forward_only: bool = False,
        reshuffle_probability: float = 0.1,
    ):
        fwd = BoltzDiffusionForwardShootMover(ensemble=ensemble, selector=selector, engine=engine)
        reshuffle_prob = 0.0 if forward_only else max(0.0, min(1.0, float(reshuffle_probability)))

        if reshuffle_prob > 0.0 and engine is not None:
            bwd = BoltzDiffusionBackwardShootMover(ensemble=ensemble, selector=selector, engine=engine)
            rsh = GlobalReshuffleMover(ensemble=ensemble, engine=engine)
            # forward + backward + reshuffle
            shooting_weight = (1.0 - reshuffle_prob) / 2.0
            movers = [fwd, bwd, rsh]
            weights = [shooting_weight, shooting_weight, reshuffle_prob]
            SpecializedRandomChoiceMover.__init__(self, movers=movers, weights=weights)
        elif forward_only:
            SpecializedRandomChoiceMover.__init__(self, movers=[fwd])
        else:
            bwd = BoltzDiffusionBackwardShootMover(ensemble=ensemble, selector=selector, engine=engine)
            SpecializedRandomChoiceMover.__init__(self, movers=[fwd, bwd])


class BoltzDiffusionOneWayShootingStrategy(move_strategy.OneWayShootingStrategy):
    """Like :class:`OneWayShootingStrategy` but uses :class:`BoltzDiffusionOneWayShootingMover`."""

    MoverClass = BoltzDiffusionOneWayShootingMover

    def __init__(self, selector=None, ensembles=None, engine=None,
                 group="shooting", replace=True, forward_only: bool = False,
                 reshuffle_probability: float = 0.1):
        super().__init__(selector=selector, ensembles=ensembles, engine=engine,
                         group=group, replace=replace)
        self.forward_only = forward_only
        self.reshuffle_probability = float(reshuffle_probability)

    def make_movers(self, scheme):
        parameters = self.get_parameters(scheme=scheme,
                                         list_parameters=[self.selector],
                                         nonlist_parameters=[self.engine])
        return [
            self.MoverClass(
                ensemble=ens,
                selector=sel,
                engine=eng,
                forward_only=self.forward_only,
                reshuffle_probability=self.reshuffle_probability,
            ).named(self.MoverClass.__name__ + " " + ens.name)
            for (ens, sel, eng) in parameters
        ]


class BoltzDiffusionOneWayShootingMoveScheme(paths.MoveScheme):
    """One-way shooting for Boltz diffusion (backward = re-noising + Hastings factor).

    Parameters
    ----------
    forward_only
        When ``True``, only forward shooting moves are generated.  See
        :class:`BoltzDiffusionOneWayShootingMover`.
    reshuffle_probability
        Fraction of MC steps devoted to global reshuffle moves.  A reshuffle draws
        a completely fresh path from the prior (always accepts when reactive).
        Default ``0.1``.  Set to ``0`` to disable.
    """

    def __init__(self, network, selector=None, ensembles=None, engine=None,
                 forward_only: bool = False, reshuffle_probability: float = 0.1):
        super().__init__(network)
        self.append(
            BoltzDiffusionOneWayShootingStrategy(
                selector=selector,
                ensembles=ensembles,
                engine=engine,
                forward_only=forward_only,
                reshuffle_probability=reshuffle_probability,
            )
        )
        self.append(move_strategy.OrganizeByMoveGroupStrategy())


def build_one_way_shooting_scheme(
    network: paths.FixedLengthTPSNetwork,
    engine: "BoltzDiffusionEngine",
    *,
    forward_only: bool = False,
    reshuffle_probability: float = 0.1,
) -> paths.MoveScheme:
    """TPS move scheme: one-way shooting with Boltz backward dynamics and bias.

    Parameters
    ----------
    forward_only
        When ``True``, only forward shooting moves are generated—useful as a
        guaranteed-correct baseline because forward shooting always accepts reactive paths.
    reshuffle_probability
        Fraction of MC steps that use a global reshuffle move (always accepts when reactive).
        Default ``0.1``.  Set to ``0`` to disable global reshuffles.
    """
    return BoltzDiffusionOneWayShootingMoveScheme(
        network, engine=engine, forward_only=forward_only,
        reshuffle_probability=reshuffle_probability,
    )


def assert_trajectory_in_ensemble(traj: Trajectory, ensemble: Any, *, label: str = "") -> None:
    ok = ensemble(traj, candidate=True)
    if not ok:
        msg = f"Trajectory does not satisfy TPS ensemble{(': ' + label) if label else ''}."
        raise ValueError(msg)


def sample_from_trajectory(
    traj: Trajectory,
    ensemble: Any,
    *,
    replica: int = 0,
) -> paths.Sample:
    return paths.Sample(
        replica=replica,
        trajectory=traj,
        ensemble=ensemble,
    )


def run_tps_path_sampling(
    engine: "BoltzDiffusionEngine",
    init_traj: Trajectory,
    n_rounds: int,
    log_path: Path,
    *,
    progress_every: int = 0,
    progress_file: TextIO | None = None,
    periodic_callback: Callable[[int, Trajectory], None] | None = None,
    periodic_every: int = 0,
    periodic_callbacks: Sequence[tuple[Callable[[int, Trajectory], None], int]]
    | None = None,
    periodic_step_callbacks: Sequence[
        tuple[Callable[[int, dict[str, Any]], None], int]
    ]
    | None = None,
    log_path_prob_every: int = 1,
    forward_only: bool = False,
    reshuffle_probability: float = 0.1,
    enhanced_bias: "EnhancedSamplingBias | None" = None,
    cv_function: "Callable[[Trajectory], float | np.ndarray] | None" = None,
    diagnostic_cv_functions: "dict[str, Callable[[Trajectory], float]] | None" = None,
) -> tuple[Trajectory, list[dict[str, Any]]]:
    """Run OPS :class:`PathSampling` with one-way shooting; return final trajectory and step log.

    Parameters
    ----------
    engine
        Must share a :class:`~genai_tps.backends.boltz.gpu_core.BoltzSamplerCore` whose
        schedule was used to build the initial path.
    init_traj
        Forward diffusion trajectory (length ``num_sampling_steps + 1``).
    progress_every
        If positive, print a progress line to ``progress_file`` every this many MC steps
        (rate and ETA). ``0`` disables.
    progress_file
        Stream for progress lines (default: ``sys.stderr``).
    periodic_callback
        If set with ``periodic_every > 0``, invoked as ``callback(step_index, trajectory)``
        every ``periodic_every`` MC steps with replica-0 trajectory **after** that step
        (current accepted path).
    periodic_every
        Interval for ``periodic_callback``; ``0`` disables.
    periodic_callbacks
        Additional ``(callback, every_n_steps)`` pairs; each ``callback(mc_step, trajectory)``
        runs when ``mc_step`` is a multiple of ``every_n_steps``. Use for checkpoints without
        coupling to ``periodic_every`` (e.g. save NPZ every 500 steps while cartoons run every 100).
    periodic_step_callbacks
        ``(callback, every_n_steps)`` pairs; ``callback(mc_step, step_entry)`` runs after the
        step is appended to the internal log, before the shooting log line is written.
        ``step_entry`` is the same dict as in the returned ``step_log`` (includes ``cv_value``
        when enhanced sampling is active).  Keep callbacks lightweight or use ``every_n_steps > 1``.
    log_path_prob_every
        If ``> 0``, compute ``trajectory_log_path_prob`` (expensive on long paths) only
        every this many MC steps for diagnostics. If ``0``, skip path log-probabilities
        entirely (much faster for long runs; ``min_1_r`` in the step log will be ``na``).
    forward_only
        When ``True``, use only forward shooting moves (no backward re-noising).
        Forward shooting always accepts reactive paths (acceptance = 1) so this is a
        guaranteed-correct baseline useful for validating that the TPS ensemble and engine
        are correctly wired before enabling the full Metropolis-Hastings correction.
    reshuffle_probability
        Fraction of MC steps that use a global reshuffle move (draw a completely fresh
        path from the prior; always accepts when reactive).  Default ``0.1``.
        Set to ``0`` to disable.
    enhanced_bias
        Optional :class:`~genai_tps.enhanced_sampling.EnhancedSamplingBias` object
        that modifies the Metropolis acceptance probability.  When set alongside
        ``cv_function``, each mover multiplies ``trial.bias`` by the enhanced
        sampling acceptance factor, and the bias is updated after each MC step.
    cv_function
        Callable that takes a :class:`Trajectory` and returns a scalar CV value
        (e.g. RMSD of the last frame).  Required when ``enhanced_bias`` is set.
    diagnostic_cv_functions
        Optional dict mapping diagnostic CV names to callables
        ``(trajectory) -> float``.  These CVs are evaluated on the **current
        accepted path** after each MC step and logged in the step entry under
        the key ``"diag_<name>"``.  They do **not** feed into the OPES bias.
        Useful for monitoring memorization / mode-collapse / unphysical geometry
        without perturbing the sampling distribution.
    """
    if enhanced_bias is not None and cv_function is None:
        raise ValueError("cv_function is required when enhanced_bias is set.")

    # Attach enhanced sampling context to the engine so movers can access it.
    engine._enhanced_bias = enhanced_bias  # type: ignore[attr-defined]
    engine._cv_function = cv_function  # type: ignore[attr-defined]

    core = engine.core
    state_a, state_b = sigma_tps_state_volumes(core)
    n_frames = len(init_traj)
    network = build_fixed_length_tps_network(state_a, state_b, n_frames)
    ensemble = tps_ensemble(network)
    assert_trajectory_in_ensemble(init_traj, ensemble, label="initial path")

    scheme = build_one_way_shooting_scheme(
        network, engine, forward_only=forward_only,
        reshuffle_probability=reshuffle_probability,
    )
    scheme.move_decision_tree()

    sample = sample_from_trajectory(init_traj, ensemble)
    sample_set = paths.SampleSet([sample])

    sampler = paths.PathSampling(storage=None, move_scheme=scheme, sample_set=sample_set)
    # Default OPS hooks (PathSamplingOutputHook) expect ``before_simulation`` to run
    # before ``before_step``; ``run_one_step`` does not call ``before_simulation``.
    sampler.hooks = sampler.empty_hooks()

    step_log: list[dict[str, Any]] = []
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    log_file = log_path.open("a", encoding="utf-8")

    prog_out = progress_file if progress_file is not None else sys.stderr
    show_prog = progress_every > 0
    if show_prog:
        print(
            f"[TPS] starting {n_rounds} MC steps (log: {log_path})",
            file=prog_out,
            flush=True,
        )

    hook_state = None
    t0 = time.perf_counter()
    try:
        for i in range(n_rounds):
            step_info = (i, n_rounds)
            hook_state, mcstep = sampler.run_one_step(step_info, hook_state)
            ch = mcstep.change
            accepted = bool(ch.accepted)
            mover_name = getattr(getattr(ch, "mover", None), "__class__", type(None)).__name__
            metro_p = _extract_metropolis_acceptance(ch)

            old_traj = _rep0_trajectory(mcstep.previous)
            trials = ch.trials
            new_traj = trials[0].trajectory if trials else None
            reactive = bool(ensemble(new_traj, candidate=True)) if new_traj is not None else False
            do_lp = (
                log_path_prob_every > 0
                and (i + 1) % int(log_path_prob_every) == 0
            )
            if do_lp:
                path_lp_old = (
                    trajectory_log_path_prob(old_traj, core) if old_traj is not None else None
                )
                path_lp_new = (
                    trajectory_log_path_prob(new_traj, core) if new_traj is not None else None
                )
                if path_lp_old is not None and path_lp_new is not None:
                    min_1_r = min_metropolis_acceptance_path(
                        path_lp_old, path_lp_new, reactive=reactive
                    )
                else:
                    min_1_r = None
            else:
                path_lp_old = path_lp_new = min_1_r = None

            entry = {
                "step": i + 1,
                "accepted": accepted,
                "metropolis_acceptance": metro_p,
                "mover": mover_name,
                "path_logp_old": path_lp_old,
                "path_logp_new": path_lp_new,
                "path_reactive": reactive,
                "min_1_r": min_1_r,
            }

            cv_val: "float | np.ndarray | None" = None
            if enhanced_bias is not None and cv_function is not None:
                cur_traj = _rep0_trajectory(sampler.sample_set)
                if cur_traj is not None:
                    try:
                        cv_val = cv_function(cur_traj)
                        if cv_val is not None and np.all(np.isfinite(np.atleast_1d(cv_val))):
                            enhanced_bias.update(cv_val, i + 1)
                        elif cv_val is not None:
                            _tps_logger.warning(
                                "CV returned non-finite value at step %d; "
                                "skipping enhanced bias update.",
                                i + 1,
                            )
                    except Exception as exc:
                        _tps_logger.warning(
                            "Enhanced bias update failed at step %d: %s", i + 1, exc
                        )
                # Serialize cv_val for JSON: convert ndarray to list
                if isinstance(cv_val, np.ndarray):
                    entry["cv_value"] = cv_val.tolist()
                else:
                    entry["cv_value"] = cv_val

            # Evaluate diagnostic CVs (observation-only: do not affect acceptance)
            if diagnostic_cv_functions:
                _diag_traj = _rep0_trajectory(sampler.sample_set)
                for diag_name, diag_fn in diagnostic_cv_functions.items():
                    diag_val: float | None = None
                    try:
                        if _diag_traj is not None:
                            diag_val = diag_fn(_diag_traj)
                    except Exception as exc:
                        _tps_logger.warning(
                            "Diagnostic CV '%s' failed at step %d: %s",
                            diag_name, i + 1, exc,
                        )
                    entry[f"diag_{diag_name}"] = diag_val

            step_log.append(entry)

            if periodic_step_callbacks:
                step_1 = i + 1
                for cb, every in periodic_step_callbacks:
                    if every > 0 and step_1 % every == 0:
                        cb(step_1, entry)

            # OPS does not always set metropolis_acceptance on nested MoveChange; then metro_p
            # is None (log as "na", not a blank field).
            mp = "na" if metro_p is None else f"{float(metro_p):.6g}"
            m1 = "na" if min_1_r is None else f"{float(min_1_r):.6g}"
            if cv_val is None:
                cv_str = "na"
            elif isinstance(cv_val, np.ndarray):
                cv_str = "[" + ",".join(f"{v:.6g}" for v in cv_val.ravel()) + "]"
            elif isinstance(cv_val, (list, tuple)):
                cv_str = "[" + ",".join(f"{float(v):.6g}" for v in cv_val) + "]"
            else:
                cv_str = f"{float(cv_val):.6g}"
            log_file.write(
                f"step {entry['step']} accepted {accepted} "
                f"metropolis_acceptance {mp} min_1_r {m1} mover {mover_name}"
                + (f" cv {cv_str}" if enhanced_bias is not None else "")
                + "\n"
            )
            log_file.flush()

            # Progress before periodic callbacks (e.g. PyMOL cartoons) so stderr shows
            # rate/ETA immediately; expensive callbacks would otherwise look like a hang.
            if show_prog and (i + 1) % progress_every == 0:
                elapsed = time.perf_counter() - t0
                done = i + 1
                rate = done / elapsed if elapsed > 0 else 0.0
                left = n_rounds - done
                eta_s = left / rate if rate > 0 else float("nan")
                print(
                    f"[TPS] {done}/{n_rounds} steps  {rate:.2f} steps/s  "
                    f"~{eta_s / 60.0:.1f} min remaining",
                    file=prog_out,
                    flush=True,
                )

            cbs: list[tuple[Callable[[int, Trajectory], None], int]] = []
            if periodic_callback is not None and periodic_every > 0:
                cbs.append((periodic_callback, periodic_every))
            if periodic_callbacks:
                cbs.extend(list(periodic_callbacks))
            fire_periodic = any(
                every > 0 and (i + 1) % every == 0 for _, every in cbs
            )
            if fire_periodic:
                cur = _rep0_trajectory(sampler.sample_set)
                if cur is not None:
                    step_1 = i + 1
                    for cb, every in cbs:
                        if every > 0 and step_1 % every == 0:
                            cb(step_1, cur)
    finally:
        log_file.close()
        # Clean up engine attributes set for enhanced sampling
        engine._enhanced_bias = None  # type: ignore[attr-defined]
        engine._cv_function = None  # type: ignore[attr-defined]

    # Avoid SampleSet[int] which uses random.choice per replica_dict API.
    rep0 = [s for s in sampler.sample_set.samples if s.replica == 0]
    if not rep0:
        rep0 = sampler.sample_set.samples
    final_traj = rep0[0].trajectory
    return final_traj, step_log
