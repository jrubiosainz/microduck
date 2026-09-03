#!/usr/bin/env python3
"""Emit assets/scene_queue_politely.xml.

A SERVICE COUNTER with a rope-barrier queue folded into a hairpin: a leg out
from the counter, a 180 deg fold, and a return leg.  The bend is not decoration
- it is the reason the ordering problem is real.  With a straight queue, "the
person furthest from the counter" and "the person furthest back along the aisle"
both give the right answer.  With this fold they give two DIFFERENT wrong
answers (measured in ``queue_path``), and only projection onto the path gets it
right.

The scene is GENERATED from ``scripts/queue_path.py`` and
``scripts/queue_people.py`` rather than hand-written, so the painted lane, the
barrier posts and the people's stations cannot drift apart from the geometry the
duck reasons about.  The lane paint IS the path: every stripe is
``PATH.point_at(s)`` for an ``s`` the decision layer also uses.

Everything except the robot is ``contype="0" conaffinity="0"``: kinematic
scenery that cannot touch or push the robot.  The adults are mocap bodies posed
analytically each tick, so they add no degrees of freedom to the floating base
and the walking policy sees exactly the robot it was trained on.  An advance the
duck makes can therefore never be the result of somebody pushing it forward.

THE BARRIERS ARE NON-COLLIDING TOO, deliberately.  If they collided, "the duck
stayed in the lane and followed the bend" would be enforced by the contact
solver rather than demonstrated by the controller, and a duck that scraped
around the fold against a rope would still pass.  With non-colliding barriers
the lane is a constraint the CONTROLLER has to respect, and the acceptance gate
measures the real surface distance to every post, rope, counter and person on
every control tick.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from queue_geometry import DUCK_START_XY, DUCK_START_YAW_DEG  # noqa: E402
from queue_people import ALL_NAMES, PERSON_RGBA  # noqa: E402

from scene_parts import (  # noqa: E402
    _yaw_quat,
    barriers,
    hall,
    lane,
    markers,
    person_block,
)
from scene_template import FOOTER_OPEN, HEADER  # noqa: E402


def build() -> str:
    materials = "".join(
        f'        <material name="{name}_shirt" rgba="{PERSON_RGBA[name]}" />\n'
        for name in ALL_NAMES
    )
    parts = [HEADER + materials + FOOTER_OPEN,
             hall(), lane(), barriers(), markers(),
             "\n        <!-- people: mocap, non-colliding, scripted -->\n"]
    parts.extend(person_block(name) for name in ALL_NAMES)
    parts.append("    </worldbody>\n\n")

    quat = _yaw_quat(math.radians(DUCK_START_YAW_DEG))
    base = (f"{DUCK_START_XY[0]:.4f} {DUCK_START_XY[1]:.4f} 0.12 "
            f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}")
    hinge_pad = " ".join(["0"] * (4 * len(ALL_NAMES)))
    stand = (
        f"        {base}\n"
        "        0 -0.08726646259971647 -0.457924 -0.004940 0.452984\n"
        "        0.3490658503988659 0.3490658503988659 0 0\n"
        "        0 0.08726646259971647 0.457924 0.004940 -0.452984\n"
        f"        {hinge_pad}"
    )
    ctrl = (
        "        0 -0.08726646259971647 -0.457924 -0.004940 0.452984\n"
        "        0.3490658503988659 0.3490658503988659 0 0\n"
        "        0 0.08726646259971647 0.457924 0.004940 -0.452984"
    )
    init_pad = " ".join(["0"] * (14 + 4 * len(ALL_NAMES)))
    parts.append(
        "    <keyframe>\n"
        f'        <key name="INIT" qpos="{base} {init_pad}"\n'
        '             ctrl="0 0 0 0 0 0 0 0 0 0 0 0 0 0" />\n'
        f'        <key name="STAND" qpos="\n{stand}"\n'
        f'             ctrl="\n{ctrl}" />\n'
        "    </keyframe>\n</mujoco>\n"
    )
    return "".join(parts)


def main() -> int:
    out = (Path(__file__).resolve().parents[1] / "assets"
           / "scene_queue_politely.xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
