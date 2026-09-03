#!/usr/bin/env python3
"""Shared fixtures: the scene, the committed run, and one short real rollout.

Compiling the plaza and running 190 s of physics with an ONNX policy in the
loop is expensive, so everything expensive is session-scoped.

THE RUN THESE TESTS GRADE IS THE COMMITTED ONE
------------------------------------------------
:data:`SUMMARY_PATH` points at ``media/protective-personal-space-metrics.json``
- the artifact the README quotes and the video was rendered from - rather than
at a rollout the suite performs for itself.  That is deliberate: a suite that
re-runs the behavior grades a re-enactment, and a re-enactment that drifted
from the shipped artifact would still pass.  Pinning the shipped file means a
behavior change has to update the artifact, and the pin tests then say exactly
what moved.

The per-tick trace is NOT committed (it is 10 MB), so it comes from a
validation run and its fixture SKIPS when absent.  Every claim that needs the
trace is therefore an optional deepening of a claim the summary already makes,
never the only place an invariant is checked.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

POLICY = REPO / "onnx" / "alpha_walking.onnx"
SCENE = REPO / "assets" / "scene_protective_personal_space.xml"

# The COMMITTED metrics of the validated run.  Overridable so the suite can be
# pointed at a fresh run without editing it.
SUMMARY_PATH = Path(os.environ.get(
    "PPS_SUMMARY", str(REPO / "media" / "protective-personal-space-metrics.json")))
# The per-tick trace, which is far too large to commit.
TRACE_PATH = Path(os.environ.get("PPS_TRACE", "/tmp/pps_trace.json"))


@pytest.fixture(scope="session")
def model():
    """The compiled plaza scene, with the stock walking robot included."""
    from policy_runtime import load_scene
    return load_scene()


@pytest.fixture(scope="session")
def data(model):
    """Data at the STAND keyframe, which is the pose the behavior runs from.

    NOT a bare ``mj_forward`` on a fresh ``MjData``: that leaves every joint at
    zero, which is a pose the robot never adopts.  A geometry constant has to
    be measured at the pose it describes, and ``DUCK_PLANAR_RADIUS`` is one.
    """
    import mujoco
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("STAND").id)
    mujoco.mj_forward(model, d)
    return d


@pytest.fixture(scope="session")
def summary():
    """The committed summary of the validated 190 s run."""
    if not SUMMARY_PATH.is_file():
        pytest.skip(f"no rollout summary at {SUMMARY_PATH}")
    return json.loads(SUMMARY_PATH.read_text())


@pytest.fixture(scope="session")
def trace():
    """The per-tick record stream of a validation run, when one is on disk."""
    if not TRACE_PATH.is_file():
        pytest.skip(
            f"no rollout trace at {TRACE_PATH}; run "
            f"scripts/validate_pps.py --seconds 190 --trace {TRACE_PATH}")
    return json.loads(TRACE_PATH.read_text())


@pytest.fixture(scope="session")
def short_rollout():
    """A genuinely short rollout, for the claims that need real physics.

    Three seconds is enough to prove the policy steps, the camera copies rather
    than drives the physical state, and the contact probes return finite
    distances - and short enough that the suite stays usable.
    """
    from rollout_pps import PpsRollout
    if not POLICY.is_file():
        pytest.skip(f"no policy at {POLICY}")
    rollout = PpsRollout(str(POLICY), 3.0)
    rollout.run()
    return rollout


@pytest.fixture(scope="session")
def camera(model, data):
    """A PpsCamera bound to the STAND pose, for isolation and LOS tests."""
    from pps_camera import PpsCamera
    from policy_runtime import actuator_indices
    qpos_idx, _ = actuator_indices(model)
    return PpsCamera(model, data, qpos_idx, model.body("trunk_base").id)
