#!/usr/bin/env python3
"""Shared fixtures.  The expensive ones are session-scoped and built once."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tools"))

POLICY = REPO / "onnx" / "alpha_walking.onnx"
# The full rollout is 110 s of physics at 50 Hz and takes minutes, so the
# integration tests share ONE of them.  Anything that needs a different length
# builds its own and is marked slow.
FULL_SECONDS = 110.0


@pytest.fixture(scope="session")
def model():
    from policy_runtime import load_scene
    return load_scene()


@pytest.fixture(scope="session")
def route():
    from etiquette_path import build_route
    return build_route()


@pytest.fixture(scope="session")
def bounds(route):
    from etiquette_path import leg_bounds
    return leg_bounds(route)


@pytest.fixture(scope="session")
def rollout():
    """One full rollout, run once and shared by every integration test.

    Session-scoped deliberately: the behavior is deterministic, so a second run
    would produce byte-identical records at several minutes of cost.
    """
    from rollout_etiquette import EtiquetteRollout
    run = EtiquetteRollout(str(POLICY), FULL_SECONDS)
    run.run()
    return run


@pytest.fixture(scope="session")
def summary(rollout):
    from etiquette_summary import summarize
    return summarize(rollout)


@pytest.fixture(scope="session")
def results(summary):
    from etiquette_metrics import report
    return report(summary)


def gate_named(results, fragment: str):
    """The (label, passed, evidence) triple whose label contains ``fragment``.

    Used by the counterexample suite so a mutation can name the gate it must
    break, rather than merely asserting that SOMETHING failed - a mutation that
    tripped an unrelated gate would otherwise look like a pass.
    """
    _, entries = results
    matches = [entry for entry in entries if fragment in entry[0]]
    if not matches:
        raise AssertionError(
            f"no gate matching {fragment!r}; gates are: "
            + "; ".join(entry[0] for entry in entries))
    if len(matches) > 1:
        raise AssertionError(
            f"{fragment!r} matches {len(matches)} gates: "
            + "; ".join(entry[0] for entry in matches))
    return matches[0]
