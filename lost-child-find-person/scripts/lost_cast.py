#!/usr/bin/env python3
"""The cast: one guardian, two look-alikes, and six other adults.

WHAT "IDENTITY" MEANS HERE, STATED PLAINLY
------------------------------------------
The duck does **not** run an RGB person recognizer.  It reads an APPEARANCE
DESCRIPTOR out of the simulator — shirt colour, standing height, headwear,
shoulder bag — for whichever body its camera can currently see, and compares it
against the descriptor it recorded of its guardian before the loss.  That is a
*semantic proxy* for person re-identification, and it is labelled as such
everywhere it appears.

What makes the proxy non-trivial is that **each feature is only readable when
the camera can actually see the part of the body that carries it**.  Headwear
needs the head sample; stature needs the knees *and* the head, because you
cannot judge somebody's height from their shoulders up.  A candidate seen
half-behind a column therefore yields an INCOMPLETE descriptor and cannot be
confirmed, however well the visible half matches.  That is why the confirmation
gate is a duration and not an instant.

THE TWO LOOK-ALIKES ARE DESIGNED TO SCORE HIGH
-----------------------------------------------
A distractor in a different-coloured shirt proves nothing: the shirt term alone
rejects it and the identity layer never has to do any work.  Both look-alikes
here wear a shirt within a hair of the guardian's teal and carry the same
shoulder bag, so the shirt and bag terms are effectively tied and the decision
falls entirely on ONE remaining feature each:

* ``mira``  — the same shirt, the same bag, the same height, **but a cap**.
* ``sofia`` — the same shirt, the same bag, no cap, **but 12 cm shorter**.

Both score above the candidate threshold and below the accept threshold, which
is exactly what a tempting false positive is.  Everybody else in the hall wears
an unmistakably different colour and never becomes a candidate at all, so the
count of rejected candidates is the count of genuine near-misses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Nominal standing height of an adult in this hall, before the stature factor.
BASE_HEIGHT_M = 1.72
# Mocap origin height, and the camera's sample points relative to it, both
# scaled by a person's stature.  Five samples: knees, waist, chest, head, crown.
BASE_ORIGIN_Z = 0.36
BASE_SAMPLE_DZ: tuple[float, ...] = (-0.10, 0.02, 0.16, 0.28, 0.34)

# Which sample indices must be visible before a feature can be read at all.
FEATURE_SAMPLES: dict[str, tuple[int, ...]] = {
    "shirt": (1, 2),          # the torso
    "satchel": (1, 2),        # worn on the torso
    "cap": (3, 4),            # the head and crown
    "stature": (0, 3),        # you need the bottom AND the top
}

# Weights of the four descriptor terms.  They sum to 1.0 so a score is directly
# readable as "fraction of the appearance that matched".
WEIGHTS: dict[str, float] = {
    "shirt": 0.45, "stature": 0.35, "cap": 0.10, "satchel": 0.10}
# Normalisers: the largest difference each term is expected to express.
SHIRT_NORM = 1.00      # RGB euclidean distance
STATURE_NORM = 0.25    # metres


@dataclass(frozen=True)
class Person:
    """One adult, with the appearance the identity layer reads from the camera."""

    name: str
    shirt: tuple[float, float, float]
    stature: float
    cap: bool
    satchel: bool
    role: str                  # "guardian" | "lookalike" | "crowd"
    label: str = ""

    @property
    def height_m(self) -> float:
        return BASE_HEIGHT_M * self.stature

    @property
    def origin_z(self) -> float:
        return BASE_ORIGIN_Z * self.stature

    @property
    def sample_dz(self) -> tuple[float, ...]:
        return tuple(dz * self.stature for dz in BASE_SAMPLE_DZ)

    @property
    def rgba(self) -> str:
        r, g, b = self.shirt
        return f"{r:.3f} {g:.3f} {b:.3f} 1"

    def descriptor(self) -> dict:
        return {"shirt": self.shirt, "stature": self.height_m,
                "cap": self.cap, "satchel": self.satchel}


# ---------------------------------------------------------------------------
# THE GUARDIAN, and the two adults built to be mistaken for her.
GUARDIAN = Person(
    "priya", (0.150, 0.620, 0.600), 1.000, False, True, "guardian",
    "guardian - teal shirt, shoulder bag, no cap")
LOOKALIKES: tuple[Person, ...] = (
    Person("mira", (0.170, 0.600, 0.615), 0.990, True, True, "lookalike",
           "look-alike - same teal, same bag, WEARS A CAP"),
    Person("sofia", (0.140, 0.640, 0.585), 0.930, False, True, "lookalike",
           "look-alike - same teal, same bag, 12 cm SHORTER"),
)
# Six more adults, all continuously moving, none of them near the guardian's
# appearance.  They are what makes the hall a crowd rather than a stage.
CROWD: tuple[Person, ...] = (
    Person("arun", (0.880, 0.420, 0.180), 1.020, False, False, "crowd"),
    Person("bekele", (0.180, 0.240, 0.560), 0.970, True, False, "crowd"),
    Person("costa", (0.640, 0.190, 0.260), 1.010, False, True, "crowd"),
    Person("dahl", (0.900, 0.780, 0.240), 0.940, False, False, "crowd"),
    Person("eze", (0.520, 0.260, 0.700), 1.030, True, True, "crowd"),
    Person("faruq", (0.400, 0.440, 0.400), 0.980, False, False, "crowd"),
)

PEOPLE: tuple[Person, ...] = (GUARDIAN, *LOOKALIKES, *CROWD)
BY_NAME: dict[str, Person] = {p.name: p for p in PEOPLE}
ALL_NAMES: tuple[str, ...] = tuple(p.name for p in PEOPLE)
LOOKALIKE_NAMES: tuple[str, ...] = tuple(p.name for p in LOOKALIKES)
CROWD_NAMES: tuple[str, ...] = tuple(p.name for p in CROWD)
# LEGACY NOMINAL REFERENCE, duplicated from ``lost_geometry`` for the cast's
# own convenience.  NOT a measurement of this scene and NOT used by any gate:
# MEASURED here, an adult's exact planar half-extent runs 0.1375 m at pose zero
# to 0.2629 m mid-stride (0.2709 m for the widest adult over the full rollout).
# See ``lost_geometry.ADULT_HALF_EXTENT_M`` for the full note.  Clearance is
# measured every tick by ``ContactProbe`` against the real geoms.
ADULT_HALF_EXTENT_M: float = 0.1647


def shirt_distance(a: tuple[float, float, float],
                   b: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def term_penalties(reference: dict, observed: dict) -> dict[str, float]:
    """Per-feature mismatch in [0, 1], for the features present in ``observed``.

    Only features the camera could actually READ appear in ``observed``, so a
    partially seen candidate produces a partial penalty set.  Scoring that as a
    match would be the classic re-identification failure — "everything I could
    see matched" is not "it is her" — so :func:`match_score` refuses to return
    an accept-grade score unless every feature was readable.
    """
    penalties: dict[str, float] = {}
    if "shirt" in observed:
        penalties["shirt"] = min(
            shirt_distance(reference["shirt"], observed["shirt"]) / SHIRT_NORM,
            1.0)
    if "stature" in observed:
        penalties["stature"] = min(
            abs(reference["stature"] - observed["stature"]) / STATURE_NORM, 1.0)
    if "cap" in observed:
        penalties["cap"] = 0.0 if reference["cap"] == observed["cap"] else 1.0
    if "satchel" in observed:
        penalties["satchel"] = (
            0.0 if reference["satchel"] == observed["satchel"] else 1.0)
    return penalties


def match_score(reference: dict, observed: dict) -> tuple[float, dict]:
    """Score in [0, 1] plus the per-term penalties, weighted by :data:`WEIGHTS`.

    UNREADABLE FEATURES COUNT AS FULLY UNMATCHED.  A candidate whose head is
    behind a column cannot be confirmed by their torso, so the missing term
    contributes its whole weight as penalty rather than being renormalised away.
    """
    penalties = term_penalties(reference, observed)
    total = 0.0
    for feature, weight in WEIGHTS.items():
        total += weight * penalties.get(feature, 1.0)
    return max(0.0, 1.0 - total), penalties


def dominant_mismatch(penalties: dict[str, float],
                      readable: set[str]) -> tuple[str, float]:
    """The feature that did the most to reject a candidate, and its penalty."""
    contributions = {
        feature: WEIGHTS[feature] * penalties.get(feature, 1.0)
        for feature in WEIGHTS
    }
    feature = max(contributions, key=lambda k: (contributions[k], k))
    if feature not in readable:
        return f"{feature}_unreadable", contributions[feature]
    return feature, contributions[feature]


def rejection_reason(name: str, penalties: dict[str, float],
                     readable: set[str], observed: dict) -> str:
    """A human sentence naming why this candidate is not the guardian."""
    feature, _ = dominant_mismatch(penalties, readable)
    if feature.endswith("_unreadable"):
        return f"{feature.split('_')[0]} not readable from this viewpoint"
    if feature == "cap":
        worn = observed.get("cap")
        return ("wears a cap; guardian does not" if worn
                else "bare-headed mismatch")
    if feature == "stature":
        return (f"stands {observed.get('stature', float('nan')):.2f} m, "
                f"guardian {GUARDIAN.height_m:.2f} m")
    if feature == "shirt":
        return "shirt colour differs from the guardian's teal"
    if feature == "satchel":
        return "shoulder bag mismatch"
    return f"{feature} mismatch"
