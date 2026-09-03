#!/usr/bin/env python3
"""Shared fixtures: the scene, a short rollout, and the cached full summary.

Building a MuJoCo model and running 92 s of physics is expensive, so both are
session-scoped.  The full-run fixtures read the artifacts a validation run
already wrote rather than re-running the rollout, which keeps the whole suite
under a minute while still grading the REAL run rather than a re-enactment.
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
# Where a validation run leaves its artifacts.  Overridable so the suite can be
# pointed at a fresh run without editing it.
SUMMARY_PATH = Path(os.environ.get("SLALOM_SUMMARY", "/tmp/sl_final.json"))
TRACE_PATH = Path(os.environ.get("SLALOM_TRACE", "/tmp/sl_trace.json"))


@pytest.fixture(scope="session")
def model():
    """The compiled depot scene."""
    from policy_runtime import load_scene
    return load_scene()


@pytest.fixture(scope="session")
def data(model):
    """Data at the STAND keyframe, which is the pose the behavior runs from.

    NOT a bare ``mj_forward`` on a fresh ``MjData``: that leaves every joint at
    zero, which is a pose the robot never adopts and which measures 0.1303 m of
    planar radius against the 0.1162 m the STAND pose actually produces.  A
    geometry constant has to be measured at the pose it describes.
    """
    import mujoco
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("STAND").id)
    mujoco.mj_forward(model, d)
    return d


@pytest.fixture(scope="session")
def summary():
    """The summary of a real validation run.

    Skips rather than fails when absent: the pure-logic tests are the bulk of
    the suite and must run without a 92 s rollout on disk.
    """
    if not SUMMARY_PATH.is_file():
        pytest.skip(
            f"no rollout summary at {SUMMARY_PATH}; run "
            f"scripts/validate_slalom.py --json {SUMMARY_PATH}")
    return json.loads(SUMMARY_PATH.read_text())


@pytest.fixture(scope="session")
def trace():
    """The full per-tick record stream of a real validation run."""
    if not TRACE_PATH.is_file():
        pytest.skip(
            f"no rollout trace at {TRACE_PATH}; run "
            f"scripts/validate_slalom.py --trace {TRACE_PATH}")
    return json.loads(TRACE_PATH.read_text())


@pytest.fixture(scope="session")
def short_rollout():
    """A genuinely short rollout, for tests that need real physics."""
    from rollout_slalom import SlalomRollout
    rollout = SlalomRollout(str(POLICY), 3.0)
    rollout.run()
    return rollout
