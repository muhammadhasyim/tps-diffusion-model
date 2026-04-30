"""Unit tests for observer MD batching and checkpoint scheduling."""

import sys
import types

from genai_tps.simulation.openmm_md_runner import (
    _diagnostic_energy_and_large_forces,
    _next_observer_event_step,
)


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


def test_diagnostic_energy_and_large_forces_honors_groups(monkeypatch) -> None:
    class _Unit:
        def __truediv__(self, _other):
            return self

    openmm_mod = types.ModuleType("openmm")
    unit_mod = types.ModuleType("openmm.unit")
    unit_mod.kilojoule_per_mole = _Unit()
    unit_mod.kilojoules_per_mole = _Unit()
    unit_mod.nanometer = _Unit()
    openmm_mod.unit = unit_mod
    monkeypatch.setitem(sys.modules, "openmm", openmm_mod)
    monkeypatch.setitem(sys.modules, "openmm.unit", unit_mod)

    class _Quantity:
        def __init__(self, value):
            self.value = value

        def value_in_unit(self, _unit):
            return self.value

    class _Force:
        x = 0.0
        y = 0.0
        z = 0.0

    class _State:
        def getPotentialEnergy(self):
            return _Quantity(1.0)

        def getForces(self):
            return _Quantity([_Force()])

    class _Context:
        def __init__(self) -> None:
            self.groups = None

        def getState(self, *, getEnergy: bool, getForces: bool, groups=None):
            assert getEnergy is True
            assert getForces is True
            self.groups = groups
            return _State()

    class _Simulation:
        def __init__(self) -> None:
            self.context = _Context()

    sim = _Simulation()
    _diagnostic_energy_and_large_forces(sim, "unit_test", groups={0, 1, 2})

    assert sim.context.groups == {0, 1, 2}
