"""Shared fixtures.  The expensive ones are session-scoped and built once.

The design rule here is the same one the behavior follows: anything that can be
tested WITHOUT MuJoCo is, because a test that has to build a scene to check a
threshold is a test nobody runs.  The physics fixtures are marked ``slow`` and
carry the full rollout that the acceptance gate grades.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

POLICY = REPO / "onnx" / "alpha_walking.onnx"
# The seconds the validated run uses.  A test that graded a different duration
# would not be grading the artifact.
VALIDATED_SECONDS = 95.0


@pytest.fixture(scope="session")
def policy_path() -> Path:
    if not POLICY.is_file():
        pytest.skip(f"policy not found: {POLICY}")
    return POLICY


@pytest.fixture(scope="session")
def model():
    """The built scene, compiled once."""
    from policy_runtime import load_scene
    return load_scene()


@pytest.fixture(scope="session")
def data(model):
    import mujoco
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)
    return d


@pytest.fixture(scope="session")
def plan_at_request():
    """The plan the duck would search at the request instant.

    Pure planner plus actors, no physics, so the planning tests are fast.
    """
    from guide_actors import actors_at
    from guide_cast import FOLLOWER
    from guide_layout import DESTINATION_BY_KEY
    from guide_planner import Planner, tubes_from_states
    from guide_states import DUCK_START_XY, REQUEST_T_S, REQUESTED_DESTINATION

    people = actors_at(REQUEST_T_S)
    tubes = tubes_from_states(people, FOLLOWER.name)
    return Planner().plan(DUCK_START_XY,
                          DESTINATION_BY_KEY[REQUESTED_DESTINATION], tubes)


@pytest.fixture(scope="session")
def rollout(policy_path):
    """THE validated rollout.  Built once; every integration test reads it.

    This is the same object ``scripts/validate_guide.py`` grades, at the same
    duration, so a test that passes here describes the artifact rather than a
    convenient shorter run.
    """
    from rollout_guide import GuideRollout
    run = GuideRollout(str(policy_path), VALIDATED_SECONDS)
    run.run()
    return run


@pytest.fixture(scope="session")
def summary(rollout):
    from guide_metrics import summarize
    return summarize(rollout)


@pytest.fixture(scope="session")
def gate_results(summary):
    from guide_metrics import report
    passed, results = report(summary)
    return passed, results


# -- helpers shared by several test modules --------------------------------

def wrap_deg(angle: float) -> float:
    """Wrap degrees to [-180, 180)."""
    return math.degrees(math.atan2(math.sin(math.radians(angle)),
                                   math.cos(math.radians(angle))))


def straight_route(corners, speed=1.0, radius=0.62):
    from guide_route import Route
    return Route("test", tuple(corners), speed, radius=radius)


def fake_follower(pos=(0.0, 0.0), speed=0.0, trail_gap=1.0, walked=0.0):
    """A minimal stand-in for the follower, for record-building tests."""
    class _F:
        def __init__(self):
            self.pos = np.asarray(pos, dtype=np.float64)
            self.speed = speed
            self.trail_gap_m = trail_gap
            self.walked_m = walked
            self.stall_label = ""
    return _F()
