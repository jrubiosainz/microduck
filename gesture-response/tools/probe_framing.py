#!/usr/bin/env python3
"""Choose the wide camera by MEASUREMENT, replaying the real recorded trace.

Framing is the one part of this behavior that is a judgement call, so it is made
the same way every other constant is: by scoring candidates on the run that
actually happened rather than by looking at one frame and adjusting.

Each candidate is flown through ``render_frames``' OWN easing and distance ramp,
tick by tick over the recorded trace, and scored on five things that matter for
THIS behavior specifically:

* **duck on screen** and **duck unoccluded** - the robot must be visible;
* **instructor on screen** - unlike a patrol, the person giving the commands is
  half the subject: a frame without her cannot show what the duck responded to;
* **arm elevation legibility** - the angle between the camera's view direction
  and the instructor's raised arm.  A camera looking steeply down foreshortens a
  raised arm into the body, which is exactly the thing this video exists to
  show, and it is why the elevation here is shallower than a patrol's;
* **duck clear of the HUD panels** - the left column and the right column are
  opaque, so a duck drawn under them is not visible however well framed.

Run:
    ../../microduck_rl/.venv/bin/python tools/probe_framing.py \
        --trace /tmp/gr_trace.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from gest_arena import FLOOR_HALF  # noqa: E402
from policy_runtime import load_scene  # noqa: E402
import render_frames as RF  # noqa: E402

WIDTH, HEIGHT = 960, 640
# The opaque HUD regions, from ``video_overlay.compose``.  A duck drawn inside
# either is not visible, so a candidate that parks it there scores zero on this.
HUD_LEFT = (10, 44, 278, HEIGHT - 84)
HUD_RIGHT = (WIDTH - 310, 44, WIDTH - 10, HEIGHT - 76)
HUD_BOTTOM = (10, HEIGHT - 70, WIDTH - 10, HEIGHT - 8)


def in_box(px, py, box) -> bool:
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


class Fly:
    """Replays the render module's own easing for one candidate."""

    def __init__(self, azimuth, elevation, near, far, start, bias=None):
        self.azimuth = azimuth
        self.elevation = elevation
        self.near = near
        self.far = far
        self.bias = RF.LOOKAT_SUBJECT_BIAS if bias is None else bias
        self.lookat = np.array([start[0], start[1], RF.LOOKAT_Z])
        self.distance = near

    def step(self, duck, instructor, t):
        if instructor is None:
            target = np.array([duck[0], duck[1], RF.LOOKAT_Z])
            separation = 0.0
        else:
            target = np.array([
                self.bias * duck[0] + (1.0 - self.bias) * instructor[0],
                self.bias * duck[1] + (1.0 - self.bias) * instructor[1],
                RF.LOOKAT_Z])
            separation = float(np.linalg.norm(
                np.asarray(duck) - np.asarray(instructor)))
        self.lookat += RF.LOOKAT_EASE * (target - self.lookat)
        wanted = self.near + (self.far - self.near) * min(
            max(separation / RF.SEPARATION_FOR_FAR_M, 0.0), 1.0)
        self.distance += RF.LOOKAT_EASE * (wanted - self.distance)

        azimuth = self.azimuth + RF.AZIMUTH_SWING_DEG * math.sin(
            t / RF.AZIMUTH_SWING_PERIOD_S)
        elevation = math.radians(self.elevation)
        eye_z = self.lookat[2] + self.distance * abs(math.sin(elevation))
        lookat = self.lookat.copy()
        if eye_z < RF.EYE_CLEARS_SCENE_Z:
            heading = math.radians(azimuth)
            shift_x = math.cos(elevation) * math.cos(heading) * self.distance
            shift_y = math.cos(elevation) * math.sin(heading) * self.distance
            half_x = FLOOR_HALF[0] - RF.EYE_WALL_MARGIN_M
            half_y = FLOOR_HALF[1] - RF.EYE_WALL_MARGIN_M
            lookat[0] = float(np.clip(lookat[0], -half_x + shift_x,
                                      half_x + shift_x))
            lookat[1] = float(np.clip(lookat[1], -half_y + shift_y,
                                      half_y + shift_y))
        return lookat, self.distance, azimuth


def camera_basis(lookat, distance, azimuth_deg, elevation_deg):
    """Eye position and the orthonormal basis MuJoCo's free camera would use."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    forward = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation)])
    eye = np.asarray(lookat) - forward * distance
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)
    return eye, right, up, forward


def project(point, eye, right, up, forward, fovy_deg=45.0):
    """World point to pixel, or ``None`` when behind the camera."""
    delta = np.asarray(point, dtype=np.float64) - eye
    depth = float(delta @ forward)
    if depth <= 1e-6:
        return None
    tan_v = math.tan(math.radians(fovy_deg) * 0.5)
    tan_h = (WIDTH / HEIGHT) * tan_v
    px = WIDTH * 0.5 * (1.0 + float(delta @ right) / (depth * tan_h))
    py = HEIGHT * 0.5 * (1.0 - float(delta @ up) / (depth * tan_v))
    return px, py, depth


def gesture_at(t: float) -> str:
    """What the instructor is doing at ``t``, from the scenario itself.

    The probe is allowed to read the choreography - it is a rendering tool, not
    a decision layer, and it needs to know which arm pose to measure.
    """
    from gest_arm import REST
    from gest_cast import INSTRUCTOR
    from gest_script import active_cue

    cue = active_cue(INSTRUCTOR, t)
    return cue.gesture if cue is not None else REST


class ArmProbe:
    """World positions of the instructor's gesturing arm, posed on the model.

    Legibility is measured as the PROJECTED PIXEL LENGTH of the shoulder-to-hand
    segment, which is the honest form of "can a viewer see the gesture": an arm
    pointing at the camera projects to almost nothing however well lit it is.
    """

    def __init__(self, model, data, person: str = "mira",
                 yaw_deg: float = -90.0):
        from gest_arm import JOINT_KEYS

        self.model = model
        self.data = data
        self.person = person
        self.yaw = math.radians(yaw_deg)
        self.joint_keys = JOINT_KEYS
        self.joints = {k: model.joint(f"{person}_{k}").id for k in JOINT_KEYS}
        self.mocap = int(model.body_mocapid[model.body(f"actor_{person}").id])
        self._cache: dict[tuple, tuple] = {}

    def arm_segment(self, gesture: str, held_s: float, position):
        """Shoulder and hand world points of the more-raised arm."""
        from gest_arm import arm_targets

        key = (gesture, round(held_s, 2))
        if key not in self._cache:
            targets, _ = arm_targets(gesture, held_s, 6.0, 0.0)
            self.data.mocap_pos[self.mocap] = (float(position[0]),
                                               float(position[1]), 0.36)
            self.data.mocap_quat[self.mocap] = np.array(
                [math.cos(self.yaw / 2.0), 0.0, 0.0, math.sin(self.yaw / 2.0)])
            for name, joint in self.joints.items():
                self.data.qpos[int(self.model.jnt_qposadr[joint])] = \
                    targets[name]
            mujoco.mj_forward(self.model, self.data)
            best = None
            for side in ("l", "r"):
                shoulder = self.data.xpos[
                    self.model.body(f"{self.person}_shoulder_{side}").id].copy()
                hand = self.data.xpos[
                    self.model.body(f"{self.person}_hand_{side}").id].copy()
                if best is None or hand[2] > best[1][2]:
                    best = (shoulder, hand)
            self._cache[key] = best
        return self._cache[key]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="/tmp/gr_trace.json")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    trace = json.loads(Path(args.trace).read_text())
    records = trace[::args.stride]
    model = load_scene()
    data = mujoco.MjData(model)
    arms = ArmProbe(model, data)

    candidates = []
    # THE DISTANCE RANGE MATTERS AS MUCH AS THE ANGLE, and the first sweep was
    # too narrow to see it.  The two subjects sit ~300 px apart on a 960 px
    # frame whose clear middle band - between the two opaque HUD columns - is
    # only about 370 px wide, so at close range one of them is always behind a
    # panel.  Pulling back shrinks their pixel separation and lets both sit in
    # the clear band at once, at the cost of a smaller arm; the sweep is what
    # decides where that trade lands rather than an opinion about it.
    # ONLY CAMERAS THAT FLY ABOVE THE WALLS ARE CONSIDERED, AND THAT IS THE
    # MEASUREMENT WHICH EXPLAINS THE TWO WRONG ANSWERS BEFORE IT.
    #
    # ``render_frames._aim_camera`` clamps the LOOK-AT whenever the eye would
    # sit below ``EYE_CLEARS_SCENE_Z``, to stop the camera being placed inside
    # a wall.  The sibling patrol flies at -52 deg, where the eye is 3.2 m up
    # and the clamp never engages, so it was invisible there.  At the shallow
    # elevations this behavior wants for arm legibility the eye drops to about
    # 1.5 m, the clamp engages on nearly every tick, and it shifts the look-at
    # by a full camera distance - which is what dragged the instructor out to
    # x = 32 px, behind the opaque left column, for the whole COME and STOP.
    # VERIFIED against MuJoCo's own segmentation renderer: the analytic
    # projection below agrees with the rendered centroid to about 10 px, so the
    # probe was right and the eye was wrong.
    #
    # Rather than fight the clamp, the sweep only offers candidates whose eye
    # clears the walls at the NEAR distance, so the clamp is provably inert and
    # the frame really is centred on the look-at:
    #     eye_z = LOOKAT_Z + distance * sin(|elevation|)
    def clears(elevation: float, near: float) -> bool:
        eye_z = RF.LOOKAT_Z + near * math.sin(math.radians(abs(elevation)))
        return eye_z >= RF.EYE_CLEARS_SCENE_Z

    for azimuth in (0.0, 20.0, 38.0, 58.0, 74.0, 90.0, 105.0, 125.0, 145.0):
        for elevation in (-20.0, -24.0, -28.0, -32.0, -36.0):
            for near, far in ((3.60, 4.40), (4.20, 5.00), (4.80, 5.60),
                              (5.40, 6.20), (6.00, 6.80)):
                if not clears(elevation, near):
                    continue
                for bias in (0.50, 0.62):
                    candidates.append((azimuth, elevation, near, far, bias))

    print("=" * 110)
    print(f"WIDE CAMERA, SCORED ON {len(records)} TICKS OF THE REAL RUN "
          f"({len(candidates)} candidates)")
    print("arm_px   = projected shoulder-to-hand length of the gesturing arm, "
          "in pixels: this is the gesture's legibility")
    print("instr_ok = instructor on screen AND clear of the opaque HUD columns")
    print("gap_px   = duck-to-instructor pixel separation, so the two subjects "
          "do not stack on top of each other")
    print("=" * 110)
    print(f"{'az':>6} {'elev':>6} {'near':>5} {'far':>5} {'bias':>5} | "
          f"{'duck_on':>8} {'duck_free':>9} {'instr_ok':>9} {'arm_px':>7} "
          f"{'gap_px':>7} {'hud_free':>9} {'duck_px':>7} | {'score':>6}")

    start = records[0]["duck_xy"]
    results = []
    for azimuth, elevation, near, far, bias in candidates:
        fly = Fly(azimuth, elevation, near, far, start, bias)
        duck_on = duck_free = instr_on = instr_free = hud_free = 0
        arm_lengths = []
        gaps = []
        widths = []
        for record in records:
            duck = record["duck_xy"]
            instructor = record["bodies"][record["instructor"]]["xy"]
            lookat, distance, az = fly.step(duck, instructor, record["t"])
            eye, right, up, forward = camera_basis(lookat, distance, az,
                                                   elevation)

            duck_pt = project([duck[0], duck[1], 0.11], eye, right, up, forward)
            if duck_pt is not None and 0 <= duck_pt[0] <= WIDTH \
                    and 0 <= duck_pt[1] <= HEIGHT:
                duck_on += 1
                if not (in_box(duck_pt[0], duck_pt[1], HUD_LEFT)
                        or in_box(duck_pt[0], duck_pt[1], HUD_RIGHT)
                        or in_box(duck_pt[0], duck_pt[1], HUD_BOTTOM)):
                    hud_free += 1
                tan_h = (WIDTH / HEIGHT) * math.tan(math.radians(45.0) * 0.5)
                widths.append(WIDTH * 0.165 / (2.0 * duck_pt[2] * tan_h))
                span = np.array([duck[0], duck[1], 0.11]) - eye
                dist = float(np.linalg.norm(span))
                geom = np.zeros(1, dtype=np.int32)
                hit = mujoco.mj_ray(model, data, eye, span / dist, None, 1, -1,
                                    geom)
                if geom[0] < 0 or hit < 0.0 or hit >= dist - 0.25:
                    duck_free += 1

            instr_pt = project([instructor[0], instructor[1], 0.55],
                               eye, right, up, forward)
            if instr_pt is not None and 0 <= instr_pt[0] <= WIDTH \
                    and 0 <= instr_pt[1] <= HEIGHT:
                instr_on += 1
                # THE INSTRUCTOR MUST ALSO BE CLEAR OF THE OPAQUE HUD COLUMNS,
                # and leaving this out was a real miss.  The first probe scored
                # only "on screen", chose azimuth 20, and the resulting preview
                # put her half behind the left column during both turns - on
                # screen by the metric, invisible to a viewer.  A gesture drawn
                # under a panel is not a gesture anybody can read.
                if not (in_box(instr_pt[0], instr_pt[1], HUD_LEFT)
                        or in_box(instr_pt[0], instr_pt[1], HUD_RIGHT)
                        or in_box(instr_pt[0], instr_pt[1], HUD_BOTTOM)):
                    instr_free += 1
                if duck_pt is not None:
                    gaps.append(math.hypot(instr_pt[0] - duck_pt[0],
                                           instr_pt[1] - duck_pt[1]))

            # THE GESTURE'S OWN LEGIBILITY, measured rather than assumed.
            gesture = gesture_at(record["t"])
            if gesture != "REST":
                shoulder, hand = arms.arm_segment(gesture, 1.6, instructor)
                a = project(shoulder, eye, right, up, forward)
                b = project(hand, eye, right, up, forward)
                if a is not None and b is not None:
                    arm_lengths.append(math.hypot(b[0] - a[0], b[1] - a[1]))

        total = len(records)
        # The arm term is normalised against 46 px - MEASURED as the best any
        # candidate that keeps BOTH subjects clear of the HUD actually achieves
        # - and capped, because past that the gesture is legible and more
        # pixels buy nothing.  Normalising against an unreachable 60 px let the
        # arm term dominate and selected a camera that hid the instructor
        # behind the left column for a fifth of the run.
        arm_px = float(np.mean(arm_lengths)) if arm_lengths else 0.0
        gap_px = float(np.mean(gaps)) if gaps else 0.0
        # VISIBILITY IS A HARD FLOOR, NOT A TERM.  A candidate that hides either
        # subject behind a panel for more than 5 % of the run is rejected
        # outright rather than allowed to trade that away against a longer arm.
        visible = min(duck_free / total, instr_free / total, hud_free / total)
        score = 0.0 if visible < 0.95 else (
            visible * min(arm_px / 46.0, 1.0)
            * (duck_on / total) * (instr_on / total))
        results.append((score, azimuth, elevation, near, far, bias,
                        duck_on / total, duck_free / total,
                        instr_free / total, arm_px, gap_px, hud_free / total,
                        float(np.median(widths)) if widths else 0.0))

    results.sort(reverse=True)
    for entry in results[:16]:
        (score, azimuth, elevation, near, far, bias, d_on, d_free, i_ok,
         arm_px, gap_px, h_free, px) = entry
        print(f"{azimuth:6.0f} {elevation:6.0f} {near:5.2f} {far:5.2f} "
              f"{bias:5.2f} | {d_on:8.3f} {d_free:9.3f} {i_ok:9.3f} "
              f"{arm_px:7.1f} {gap_px:7.1f} {h_free:9.3f} {px:7.1f} | "
              f"{score:6.3f}")

    best = results[0]
    if best[0] <= 0.0:
        print("\nNO CANDIDATE KEEPS BOTH SUBJECTS CLEAR OF THE HUD ON 95 % OF "
              "TICKS")
        return 1
    print()
    print("BEST:")
    print(f"  CAM_AZIMUTH           = {best[1]:.1f}")
    print(f"  CAM_ELEVATION         = {best[2]:.1f}")
    print(f"  CAM_DISTANCE_NEAR     = {best[3]:.2f}")
    print(f"  CAM_DISTANCE_FAR      = {best[4]:.2f}")
    print(f"  LOOKAT_SUBJECT_BIAS   = {best[5]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
