"""Unit tests for observer MD batching and checkpoint scheduling."""

from genai_tps.simulation.openmm_md_runner import _next_observer_event_step


def test_next_observer_event_step_first_boundary():
    assert (
        _next_observer_event_step(
            0, 10_000, (500, 1000, 10_000, 50_000)
        )
        == 500
    )


def test_next_observer_event_step_prefers_earliest_period():
    assert _next_observer_event_step(499, 10_000, (500, 1000)) == 500
    assert _next_observer_event_step(500, 10_000, (500, 1000)) == 1000


def test_next_observer_event_step_tail_to_n_steps_when_no_boundary():
    assert _next_observer_event_step(0, 250, (500, 1000, 10_000, 50_000)) == 250
    assert _next_observer_event_step(100, 199, (500,)) == 199


def test_next_observer_event_step_when_already_at_n_steps():
    assert _next_observer_event_step(1000, 1000, (500, 1000)) == 1000
