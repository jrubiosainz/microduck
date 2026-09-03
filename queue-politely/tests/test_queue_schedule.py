#!/usr/bin/env python3
"""The people schedule: deterministic, continuous, and blind to the robot.

The central test is ``test_the_queue_never_waits_for_the_duck``, which parses
the schedule module's AST with docstrings stripped and requires the word "duck"
to appear nowhere in its executable code.  Grepping the raw source would fail on
the module's own prose, which legitimately explains why the robot is not an
input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from queue_path import PATH  # noqa: E402
from queue_people import (  # noqa: E402
    DEPARTURE_TIMES,
    QUEUE_NAMES,
    departures,
    max_visible_jump,
    people_at,
)

TRUTH = list(QUEUE_NAMES)


# -------------------------------------------------------------- the schedule
def test_the_queue_never_waits_for_the_duck():
    """The schedule is a closed-form function of time and nothing else.

    Checked against the module's EXECUTABLE CODE, with comments and docstrings
    stripped: the prose legitimately discusses the duck, and grepping the raw
    source would fail on its own explanation.  Replaying the schedule cannot
    depend on the robot, because the robot is not an input to it.
    """
    import ast
    import inspect

    import queue_people

    tree = ast.parse(inspect.getsource(queue_people))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            node.body = [
                statement for statement in node.body
                if not (isinstance(statement, ast.Expr)
                        and isinstance(statement.value, ast.Constant)
                        and isinstance(statement.value.value, str))]
    code = ast.unparse(tree).lower()
    assert "duck" not in code
    # And it is deterministic: the same time always gives the same positions.
    first = [tuple(people_at(t)[n].pos) for t in np.arange(0, 50, 0.5)
             for n in QUEUE_NAMES]
    second = [tuple(people_at(t)[n].pos) for t in np.arange(0, 50, 0.5)
              for n in QUEUE_NAMES]
    assert first == second


def test_nobody_teleports():
    jump, name, when = max_visible_jump(56.0)
    assert jump < 0.05, f"{name} jumped {jump:.4f} m at t={when}"


def test_services_complete_in_queue_order():
    events = departures(56.0)
    assert [e["person"] for e in events] == TRUTH
    assert [e["served_at_s"] for e in events] == list(DEPARTURE_TIMES)


def test_the_straggler_gap_closes_as_the_queue_advances():
    """The hole is real at decision time and gone once the queue moves up."""
    early = people_at(0.0)
    gap_early = (PATH.arc_of(early["eriksson"].pos)
                 - PATH.arc_of(early["dubois"].pos))
    assert gap_early == pytest.approx(0.90, abs=0.02)
    late = people_at(DEPARTURE_TIMES[0] + 6.0)
    gap_late = (PATH.arc_of(late["eriksson"].pos)
                - PATH.arc_of(late["dubois"].pos))
    assert gap_late < gap_early - 0.25
