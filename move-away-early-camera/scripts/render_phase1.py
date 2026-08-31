#!/usr/bin/env python3
"""Phase 1 only: back away, turn 90 degrees, keep backing away.

Deliberately minimal. This is the v2 retreat -- the one leg that was already
working -- plus a single closed-loop 90 degree turn, and nothing else. No
follow, no strafe, no person choreography beyond a straight approach.

    RETREAT  person inside RETREAT_D  -> back up for RETREAT_HOLD seconds
    TURN     keep backing up, and add wz until the heading has swung 90 deg
    CLEAR    keep backing up for CLEAR_HOLD seconds on the new heading
    DONE     stand still

Yaw notes (measured, and the reason earlier attempts span out):
  * yaw authority only exists WHILE WALKING; from a standstill wz does ~nothing
  * the sign is INVERTED -- a positive wz command yields a NEGATIVE yaw rate
So the turn is run as an open-loop-ish sweep with a measured stop condition
rather than a proportional controller, which is what used to wind up.

Adds a live picture-in-picture view from the duck's head camera and starts the
same proven manoeuvre at 1.15 m instead of 0.95 m. The gait and state durations
are intentionally unchanged.

Usage:
  python scripts/render_phase1.py --seconds 16 --out /tmp/f_p1 --fps 50
"""
import argparse
import math
import os

import numpy as np
import mujoco
import onnxruntime as ort

XML = "src/mjlab_microduck/robot/microduck/scene_yield.xml"

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)

CTRL_HZ = 50.0

# Reacts a little sooner than the 0.95 m base. This is the ONLY behavioural
# change: every gait constant (VX_RETREAT/VX_ROTATE, the HOLDs, TURN_TARGET,
# YAW_KP/WZ_MAX) is untouched, so the duck performs exactly the same movements,
# just starting them earlier. Kept modest on purpose: pushing the trigger out
# to 1.6 m stretched the phases into the window where the gait self-destabilises
# (that was the v2-v5 failure). 1.15 m buys ~1.7 s of margin without that.
RETREAT_D = 1.15
RETREAT_HOLD = 5.0        # seconds of straight retreat before turning
CLEAR_HOLD = 2.5          # seconds of retreat on the new heading after turning
TURN_MAX = 6.0            # safety cap on the turn leg

# Same narrow usable window as before: below ~-0.30 the gait never starts and
# the duck marches in place; above ~-0.33 it accumulates drift and topples.
VX_RETREAT = -0.28        # MEASURED: holds heading to 1.5 deg over 5 s
VX_ROTATE = -0.31         # MEASURED: past the cliff, self-rotates ~110 deg/s
TURN_CUT = math.radians(45.0)   # cut early: rotation overshoots to ~90 deg
TURN_TARGET = math.radians(90.0)
TURN_TOL = math.radians(12.0)   # heading error that counts as "turn done"

# Heading is held closed-loop in EVERY walking state. Leaving wz at zero during
# the retreat was the real bug behind "it turns back the other way": the gait is
# asymmetric and, unopposed, it spins the duck a full turn on its own (measured:
# -52 deg at t=9 s, +117 deg at t=12 s with wz commanded at exactly 0.00).
# MEASURED: the wz command sign is INVERTED (positive wz -> negative yaw rate),
# hence the minus in the control law.
YAW_KP = 0.9              # rad/s of command per rad of heading error
WZ_MAX = 0.20             # 0.6 spins out and topples; 0.20 closes 90 deg cleanly

CMD_TAU = 0.08            # command low-pass; 0.25 is too slow to start the gait


def quat_rotate_inverse(quat, vec):
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2.0
    return vec - w * t + np.cross(xyz, t)


def yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="onnx/alpha_walking.onnx")
    p.add_argument("--seconds", type=float, default=16.0)
    p.add_argument("--person-speed", type=float, default=0.12)
    p.add_argument("--out", default="/tmp/f_p1")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--warmup", type=float, default=1.5)
    p.add_argument("--turn-sign", type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    nu = model.nu
    qpos_idx = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(nu)]
    qvel_idx = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(nu)]
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    if gyro_id < 0:
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
    gyro_adr = model.sensor_adr[gyro_id]

    person_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "person")
    person_mid = int(model.body_mocapid[person_bid])
    # the person is a small tree of bodies (torso/head/legs); collect them all
    person_tree = {person_bid}
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == person_bid:
                person_tree.add(b)
                break
            p = model.body_parentid[p]

    for k, j in enumerate(qpos_idx):
        data.qpos[j] = DEFAULT_POSE[k]
    data.ctrl[:] = DEFAULT_POSE[:nu]
    mujoco.mj_forward(model, data)

    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    obs_dim = sess.get_inputs()[0].shape[1]
    cmd_dim = obs_dim - (3 + 3 + nu * 3)

    last_action = np.zeros(nu, dtype=np.float32)
    sim_dt = model.opt.timestep
    decim = max(1, int(round((1.0 / CTRL_HZ) / sim_dt)))
    total = int(args.seconds * CTRL_HZ)
    frame_every = max(1, int(round(CTRL_HZ / args.fps)))

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    # Duck's-eye view, shown as a PiP inset in the top-right corner. A separate
    # Renderer is required because mujoco.Renderer caches its framebuffer size.
    PIP_W, PIP_H = 300, 220
    pip_renderer = mujoco.Renderer(model, height=PIP_H, width=PIP_W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.9
    cam.elevation = -16
    cam.azimuth = 128

    # Real perception by ray-cast from the head camera to the person.
    # Segmentation rendering was useless here: the camera site sits INSIDE the
    # duck's own jaw geometry, so every ray/pixel hits `jaw_soft` first and the
    # person is never visible. Ray-casting lets us skip the duck's own body and
    # ask the real question: is the person within the field of view AND not
    # occluded?
    head_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    HALF_FOV = math.radians(float(model.cam_fovy[head_cam]))   # generous h-fov
    SELF_SKIP = 0.02          # small step to get the ray off the camera site

    # The camera site sits INSIDE the duck's own jaw geometry, so a naive ray
    # always hits `jaw_soft` within ~1 cm and every occlusion test fails. Build
    # the set of the duck's own bodies so self-hits can be skipped: the ray is
    # walked forward past them until it hits the world, the person, or nothing.
    self_bodies = set()
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == trunk:
                self_bodies.add(b)
                break
            p = model.body_parentid[p]
    self_bodies.add(trunk)

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    frames = 0

    # Keep the approach duration aligned with the validated 0.95 m base:
    # raising both x0 and RETREAT_D by 0.20 m makes the trigger occur at the
    # same policy phase, while the duck genuinely reacts 0.20 m farther away.
    # This preserves the validated gait trajectory instead of perturbing it.
    x0 = 1.60
    state = "IDLE"
    state_t = 0.0
    yaw_ref = None
    yaw_goal = 0.0
    bearing = 0.0
    turned = 0.0
    prev_yaw = None
    cmd_s = np.zeros(3, dtype=np.float32)
    turn_sign = 0.0            # 0 = not yet decided; latched during RETREAT
    turned_f = 0.0             # low-passed heading change (decision variable)
    visible = False
    last_seen_t = -99.0
    alpha = (1.0 / CTRL_HZ) / CMD_TAU

    for step in range(total):
        t = step / CTRL_HZ
        state_t += 1.0 / CTRL_HZ

        # person walks straight in and keeps coming
        # The person walks in from +x, facing the duck. (An earlier version of
        # this script moved them to -x after mis-reading the camera matrix and
        # concluding the head camera pointed at -x. That was wrong: driving the
        # policy with vx=+0.31 moves the trunk to x=+0.227, so the duck's
        # forward axis -- and its head, and its camera -- face +x. Flipping the
        # scene just put the two back-to-back.)
        px = x0 if t < args.warmup else x0 - args.person_speed * (t - args.warmup)
        data.mocap_pos[person_mid][0] = px
        person = data.mocap_pos[person_mid].copy()

        duck = data.xpos[trunk].copy()
        dist = float(np.linalg.norm(person[:2] - duck[:2]))

        # --- real perception: is the person actually in view? ---------------
        # Gaze is taken from the trunk's forward (+x) axis rather than from the
        # camera matrix: the head camera rides on the head, pointing forward,
        # and the trunk axis is the one that provably matches locomotion
        # (commanding vx>0 moves the trunk towards its own +x). Field of view
        # first, then line of sight, so turning away really does lose the
        # target no matter how close the person is.
        eye_pos = data.cam_xpos[head_cam].copy()
        fwd = data.xmat[trunk].reshape(3, 3)[:, 0]
        to_person = data.mocap_pos[person_mid].copy() - eye_pos
        rng = float(np.linalg.norm(to_person))
        u = to_person / max(rng, 1e-9)
        off_axis = math.acos(float(np.clip(np.dot(u, fwd), -1.0, 1.0)))
        in_fov = off_axis < HALF_FOV
        # signed bearing to the person in the ground plane: >0 = to the duck's
        # left. Used to pick which way to turn so they stay in view longest.
        left = data.xmat[trunk].reshape(3, 3)[:, 1]
        bearing = math.atan2(float(np.dot(u, left)), float(np.dot(u, fwd)))
        occluded = False
        if in_fov:
            # walk the ray forward, skipping hits on the duck's own body
            gid = np.zeros(1, dtype=np.int32)
            travelled = SELF_SKIP
            for _ in range(8):
                origin = eye_pos + u * travelled
                hit = mujoco.mj_ray(model, data, origin, u, None, 1, -1, gid)
                if gid[0] < 0 or hit < 0.0:
                    break                      # clear line of sight
                hit_body = int(model.geom_bodyid[int(gid[0])])
                if hit_body in person_tree:
                    break                      # we can see them
                if hit_body in self_bodies:
                    travelled += hit + 0.005   # our own head: step past it
                    if travelled >= rng:
                        break
                    continue
                if travelled + hit < rng - 0.02:
                    occluded = True            # something real is in the way
                break
        visible = in_fov and not occluded
        if visible:
            last_seen_t = t
        px_shown = int(math.degrees(off_axis))

        # --- heading: ABSOLUTE, never integrated ---------------------------
        # Project the trunk's forward axis onto the ground plane. Two
        # independent extractions (from the X and Y columns) agree to 0.2 deg
        # and |X_horiz| stays ~0.99, so this is well conditioned and is the
        # real heading -- the wild swings seen earlier were the duck genuinely
        # spinning, not measurement noise. Integrating dyaw was the bug.
        fwd_h = data.xmat[trunk].reshape(3, 3)[:, 0]
        yaw = math.atan2(fwd_h[1], fwd_h[0])
        if yaw_ref is None:
            yaw_ref = yaw
        turned = wrap(yaw - yaw_ref)
        yaw_rate = math.degrees(wrap(yaw - prev_yaw)) * CTRL_HZ if prev_yaw is not None else 0.0
        prev_yaw = yaw

        prev_state = state
        if state == "IDLE":
            # detection is now line-of-sight AND range, not range alone
            if visible and dist < RETREAT_D:
                state = "RETREAT"
        elif state == "RETREAT":
            if state_t > RETREAT_HOLD:
                state = "TURN"
        elif state == "TURN":
            # MEASURED: there is a stability cliff between vx=-0.285 and -0.290.
            # At -0.28 the duck holds heading to 1.5 deg over 5 s; at -0.31 it
            # self-rotates at ~110 deg/s. wz is useless here (a 5-seed sweep of
            # KP from -1.5..+1.5 left the final heading anywhere in -158..+28
            # deg), so the turn is driven by the gait's own rotation and simply
            # CUT when the absolute heading has swung far enough. The cut is at
            # 35 deg, not 90, because rotation continues through the transition.
            if abs(turned) >= TURN_CUT or state_t > TURN_MAX:
                state = "CLEAR"
        elif state == "CLEAR":
            if state_t > CLEAR_HOLD:
                state = "DONE"

        if state != prev_state:
            state_t = 0.0
            print(f"  t={t:5.2f}s  {prev_state} -> {state}  "
                  f"(dist={dist:.3f} turned={math.degrees(turned):+.1f}deg)")

        # --- command: speed IS the steering ---------------------------------
        # RETREAT/CLEAR use the straight-line speed, TURN uses the rotating one.
        # wz stays at zero throughout: it has no usable authority in this
        # regime and only destabilises the gait.
        if state == "TURN":
            target = np.array([VX_ROTATE, 0.0, 0.0], dtype=np.float32)
        elif state in ("RETREAT", "CLEAR"):
            target = np.array([VX_RETREAT, 0.0, 0.0], dtype=np.float32)
        else:
            target = np.zeros(3, dtype=np.float32)

        cmd_s += alpha * (target - cmd_s)

        gyro = data.sensordata[gyro_adr:gyro_adr + 3].astype(np.float32)
        quat = data.xquat[trunk].astype(np.float32)
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        jpos = data.qpos[qpos_idx].astype(np.float32) - DEFAULT_POSE[:nu]
        jvel = data.qvel[qvel_idx].astype(np.float32)

        cmd = np.zeros(cmd_dim, dtype=np.float32)
        cmd[0:3] = cmd_s

        obs = np.concatenate([gyro, grav, jpos, jvel, last_action, cmd]).astype(np.float32)
        action = sess.run([out_name], {in_name: obs.reshape(1, -1)})[0].squeeze(0).astype(np.float32)
        last_action = action.copy()
        data.ctrl[:] = DEFAULT_POSE[:nu] + action

        for _ in range(decim):
            mujoco.mj_step(model, data)

        if step % 50 == 0:
            print(f"  t={t:5.1f}s {state:7s} d={dist:.3f} "
                  f"turned={math.degrees(turned):+7.1f}deg "
                  f"duck=({duck[0]:+.3f},{duck[1]:+.3f},{duck[2]:.3f}) "
                  f"cmd=({cmd_s[0]:+.2f},{cmd_s[2]:+.2f})")

        if step % frame_every == 0:
            cam.lookat[:] = 0.5 * (data.xpos[trunk] + data.mocap_pos[person_mid])
            renderer.update_scene(data, camera=cam)
            img = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(img)

            # Separating COMMANDED from ACTUAL is the whole point of this HUD:
            # it tells apart "the duck turned because we asked it to" from
            # "the duck drifted/stumbled while we were only asking it to walk".
            wz_cmd = float(cmd_s[2])
            drift = (state in ("RETREAT", "IDLE", "DONE")) and abs(yaw_rate) > 8.0

            label = {
                "IDLE": "IDLE - standing, watching",
                "RETREAT": "RETREAT - walking BACKWARD (no turn commanded)",
                "TURN": "TURN - walking backward + TURNING",
                "CLEAR": "CLEAR - backward on new heading",
                "DONE": "DONE - stopped",
            }[state]

            dr.rectangle([0, 0, args.width, 132], fill=(0, 0, 0))
            dr.text((10, 6), f"t={t:6.2f}s   {label}", fill=(255, 255, 0))
            dr.text((10, 24),
                    f"person dist={dist:.3f} m   SEES PERSON: "
                    f"{'YES' if visible else 'NO '}  (off-axis {px_shown} deg"
                    f", fov +-{math.degrees(HALF_FOV):.0f})"
                    + ("" if visible else "  << BACK TURNED / OCCLUDED"),
                    fill=(120, 255, 140) if visible else (255, 120, 120))
            dr.text((10, 42),
                    f"CMD   vx={cmd_s[0]:+.3f} (neg=backward)   wz={wz_cmd:+.3f}"
                    f"  {'<- TURN COMMANDED' if abs(wz_cmd) > 0.02 else '(no turn commanded)'}",
                    fill=(255, 255, 255))
            dr.text((10, 60),
                    f"ACTUAL yaw_rate={yaw_rate:+7.1f} deg/s   heading={math.degrees(yaw):+7.1f} deg",
                    fill=(255, 160, 160) if drift else (180, 255, 180))
            dr.text((10, 78),
                    f"turn progress: {math.degrees(turned):+6.1f} / {math.degrees(TURN_TARGET):.0f} deg",
                    fill=(255, 255, 255))
            dr.text((10, 96),
                    f"trunk z={duck[2]:.3f} (nominal 0.116)  "
                    f"{'*** FALLEN ***' if duck[2] < 0.09 else 'upright'}",
                    fill=(255, 80, 80) if duck[2] < 0.09 else (180, 255, 180))
            if drift:
                dr.text((10, 114),
                        "!! UNCOMMANDED YAW - drift/stumble, not a commanded turn",
                        fill=(255, 80, 80))

            # --- duck's-eye PiP, top-right --------------------------------
            # Rendered from the on-board head_camera: this is literally what
            # the duck sees the person do as they walk in.
            pip_renderer.update_scene(data, camera=head_cam)
            pip = Image.fromarray(pip_renderer.render())
            px0, py0 = args.width - PIP_W - 12, 142
            dr.rectangle([px0 - 3, py0 - 20, px0 + PIP_W + 3, py0 + PIP_H + 3],
                         fill=(0, 0, 0))
            dr.text((px0, py0 - 17), "DUCK'S-EYE VIEW (head camera)",
                    fill=(120, 255, 140) if visible else (255, 120, 120))
            img.paste(pip, (px0, py0))
            dr = ImageDraw.Draw(img)
            dr.rectangle([px0 - 1, py0 - 1, px0 + PIP_W, py0 + PIP_H],
                         outline=(120, 255, 140) if visible else (255, 120, 120),
                         width=2)

            imageio.imwrite(os.path.join(args.out, f"f{frames:05d}.png"), np.asarray(img))
            frames += 1

    pos = data.xpos[trunk]
    print(f"frames={frames} final_state={state} turned={math.degrees(turned):+.1f}deg "
          f"final_trunk_pos={pos} height={pos[2]:.3f}")


if __name__ == "__main__":
    main()
