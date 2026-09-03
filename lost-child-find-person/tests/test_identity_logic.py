#!/usr/bin/env python3
"""Identity: scoring, the readability rules, and refusing a look-alike.

Pure logic on hand-built inputs.  No MuJoCo, no ONNX, no physics — which is the
point of keeping ``lost_identity`` and ``lost_cast`` separate from the camera
that feeds them.

The tracker bookkeeping that consumes these verdicts — confirmation duration,
cooldowns and the wrong-accept counter — is graded in ``test_identity_tracker``.

THE CLAIM UNDER TEST
--------------------
The two authored look-alikes must sit ABOVE the candidate threshold and BELOW
the accept threshold.  A distractor that scores 0.4 never reaches the identity
layer and proves nothing; one that scores 0.95 would be accepted and the
behavior would be broken.  The measured scores are pinned here so a weight tweak
that quietly collapses the tension fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lost_cast import (  # noqa: E402
    BASE_SAMPLE_DZ,
    CROWD_NAMES,
    FEATURE_SAMPLES,
    GUARDIAN,
    LOOKALIKE_NAMES,
    PEOPLE,
    WEIGHTS,
    BY_NAME,
    dominant_mismatch,
    match_score,
    rejection_reason,
    term_penalties,
)
from lost_constants import (  # noqa: E402
    ACCEPT_SCORE,
    CANDIDATE_SCORE,
    READ_CONE_DEG,
)
from lost_identity import evaluate  # noqa: E402

REFERENCE = GUARDIAN.descriptor()


def _entry(person, *, visible=True, readable=None, off_axis=4.0, range_m=1.2):
    """A camera record for ``person`` with a chosen readable-feature set."""
    readable = tuple(sorted(WEIGHTS if readable is None else readable))
    observed = {}
    if "shirt" in readable:
        observed["shirt"] = person.shirt
    if "stature" in readable:
        observed["stature"] = person.height_m
    if "cap" in readable:
        observed["cap"] = person.cap
    if "satchel" in readable:
        observed["satchel"] = person.satchel
    return {"visible": visible, "readable": list(readable),
            "observed": observed, "off_axis_deg": off_axis, "range_m": range_m}


# ----------------------------------------------------------------- weights
def test_the_descriptor_weights_sum_to_one():
    """So a score reads directly as 'fraction of the appearance that matched'."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert set(WEIGHTS) == set(FEATURE_SAMPLES)


def test_the_guardian_scores_exactly_one_against_herself():
    score, penalties = match_score(REFERENCE, GUARDIAN.descriptor())
    assert score == pytest.approx(1.0)
    assert all(v == 0.0 for v in penalties.values())


def test_both_authored_look_alikes_sit_between_the_two_thresholds():
    """The whole point of the cast: tempting, and refusable."""
    for name in LOOKALIKE_NAMES:
        score, _ = match_score(REFERENCE, BY_NAME[name].descriptor())
        assert CANDIDATE_SCORE <= score < ACCEPT_SCORE, f"{name} scored {score}"


def test_the_look_alike_scores_are_the_documented_ones():
    """Pinned, so a weight tweak that collapses the tension fails loudly."""
    assert match_score(REFERENCE, BY_NAME["mira"].descriptor())[0] == \
        pytest.approx(0.8615, abs=5e-4)
    assert match_score(REFERENCE, BY_NAME["sofia"].descriptor())[0] == \
        pytest.approx(0.8193, abs=5e-4)


def test_each_look_alike_is_refused_on_exactly_one_feature():
    """Shirt and bag are tied by construction, so one feature must decide."""
    mira = term_penalties(REFERENCE, BY_NAME["mira"].descriptor())
    assert mira["cap"] == 1.0
    assert mira["satchel"] == 0.0
    assert mira["shirt"] < 0.05 and mira["stature"] < 0.10

    sofia = term_penalties(REFERENCE, BY_NAME["sofia"].descriptor())
    assert sofia["cap"] == 0.0
    assert sofia["satchel"] == 0.0
    assert sofia["shirt"] < 0.05
    assert sofia["stature"] > 0.40


def test_no_crowd_member_ever_reaches_the_accept_threshold():
    for name in CROWD_NAMES:
        score, _ = match_score(REFERENCE, BY_NAME[name].descriptor())
        assert score < ACCEPT_SCORE, f"{name} scored {score}"


def test_the_named_rejection_reason_matches_the_dominant_feature():
    for name, expected in (("mira", "cap"), ("sofia", "stands")):
        person = BY_NAME[name]
        penalties = term_penalties(REFERENCE, person.descriptor())
        reason = rejection_reason(name, penalties, set(WEIGHTS),
                                  person.descriptor())
        assert expected in reason


# -------------------------------------------------------------- readability
def test_an_unreadable_feature_costs_its_whole_weight():
    """Not renormalised away: a half-seen candidate must not score as a match."""
    full, _ = match_score(REFERENCE, GUARDIAN.descriptor())
    torso_only = {"shirt": GUARDIAN.shirt, "satchel": GUARDIAN.satchel}
    partial, _ = match_score(REFERENCE, torso_only)
    assert full == pytest.approx(1.0)
    assert partial == pytest.approx(1.0 - WEIGHTS["stature"] - WEIGHTS["cap"])


def test_the_guardian_seen_only_from_the_waist_up_cannot_be_confirmed():
    """'Everything I could see matched' is the classic re-id failure."""
    entry = _entry(GUARDIAN, readable=("shirt", "satchel"))
    sighting = evaluate(GUARDIAN.name, 5.0, REFERENCE, entry)
    assert sighting.complete is False
    assert sighting.verdict != "accept"
    assert "incomplete descriptor" in sighting.reason


def test_stature_needs_the_knees_and_the_head_together():
    """You cannot judge somebody's height from their shoulders up."""
    assert FEATURE_SAMPLES["stature"] == (0, 3)
    assert FEATURE_SAMPLES["cap"] == (3, 4)
    assert FEATURE_SAMPLES["shirt"] == FEATURE_SAMPLES["satchel"] == (1, 2)


def test_the_sample_heights_are_ordered_from_knees_to_crown():
    assert list(BASE_SAMPLE_DZ) == sorted(BASE_SAMPLE_DZ)
    assert len(BASE_SAMPLE_DZ) == 5


def test_sample_offsets_scale_with_a_persons_own_stature():
    """A shorter adult is genuinely sampled lower, so stature is geometry."""
    tall = BY_NAME["priya"].sample_dz
    short = BY_NAME["sofia"].sample_dz
    assert short[-1] < tall[-1]
    assert BY_NAME["sofia"].height_m < BY_NAME["priya"].height_m - 0.10


def test_an_unreadable_dominant_feature_is_named_as_unreadable():
    penalties = {"shirt": 0.0, "satchel": 0.0}
    feature, _ = dominant_mismatch(penalties, {"shirt", "satchel"})
    assert feature.endswith("_unreadable")


# ------------------------------------------------------------- the verdicts
def test_a_body_the_camera_cannot_see_is_ignored_not_scored():
    entry = _entry(BY_NAME["mira"], visible=False)
    sighting = evaluate("mira", 3.0, REFERENCE, entry)
    assert sighting.verdict == "ignore"
    assert sighting.reason == "not visible"
    assert sighting.score == 0.0


def test_a_smear_at_the_edge_of_the_frame_is_ignored():
    """A body too far off the optical axis is not evidence either way."""
    entry = _entry(GUARDIAN, off_axis=READ_CONE_DEG + 1.0)
    sighting = evaluate(GUARDIAN.name, 3.0, REFERENCE, entry)
    assert sighting.verdict == "ignore"
    assert "optical axis" in sighting.reason


def test_the_guardian_fully_readable_and_centred_is_accepted():
    sighting = evaluate(GUARDIAN.name, 3.0, REFERENCE, _entry(GUARDIAN))
    assert sighting.verdict == "accept"
    assert sighting.score == pytest.approx(1.0)
    assert sighting.complete is True


def test_both_look_alikes_fully_readable_are_candidates_never_accepts():
    for name in LOOKALIKE_NAMES:
        sighting = evaluate(name, 3.0, REFERENCE, _entry(BY_NAME[name]))
        assert sighting.verdict == "candidate", name
        assert sighting.score < ACCEPT_SCORE


def test_a_low_scoring_crowd_member_never_becomes_a_candidate():
    sighting = evaluate("dahl", 3.0, REFERENCE, _entry(BY_NAME["dahl"]))
    assert sighting.verdict == "ignore"
    assert "too dissimilar" in sighting.reason


# ------------------------------------------------------------ the cast itself
def test_every_person_has_a_role_and_the_guardian_is_unique():
    roles = [p.role for p in PEOPLE]
    assert roles.count("guardian") == 1
    assert roles.count("lookalike") == 2
    assert roles.count("crowd") == 6
    assert len({p.name for p in PEOPLE}) == len(PEOPLE)


def test_the_sighting_record_round_trips_every_graded_field():
    record = evaluate(GUARDIAN.name, 12.34, REFERENCE,
                      _entry(GUARDIAN)).as_record()
    for key in ("name", "t", "score", "penalties", "readable",
                "complete_descriptor", "range_m", "off_axis_deg", "verdict",
                "reason"):
        assert key in record
    assert record["name"] == GUARDIAN.name
    assert record["t"] == pytest.approx(12.34)
