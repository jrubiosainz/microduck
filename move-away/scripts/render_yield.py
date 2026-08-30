#!/usr/bin/env python3
"""Reactive social-navigation demo: the duck keeps its eyes on an approaching
person, backs away, watches them leave, follows them, stops when they stop, and
finally steps aside when they come back at it.

No new policy is trained. This drives the stock `alpha_walking.onnx` with a
*reactive* twist command produced by a small state machine that watches the
scripted mocap "person". The point of the demo is the behaviour layer plus the
gaze controller, not the gait.

The duck's heading is closed-loop at all times: a proportional yaw controller
drives the trained `wz` command so the trunk keeps pointing at the person. That
is what keeps it facing them while retreating, and what stops the aimless yaw
drift that used to appear at the end of a long manoeuvre.

    RETREAT  : person closing and inside RETREAT_D   -> back away, keep facing
    WATCH    : person moving away                    -> hold still, keep facing
    FOLLOW   : person far and receding/stopped       -> walk after them
    STOP     : person stopped and within FOLLOW_D    -> hold
    SIDESTEP : person closing again and very close   -> strafe out of the lane
    SETTLE   : out of the way                        -> hold, keep facing

Usage:
  python scripts/render_yield.py --seconds 45 --out /tmp/f_yield
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

# --- behaviour tuning -------------------------------------------------------
# Distances are in metres, in the duck's own scale (trunk stands at z~0.12).
RETREAT_D = 0.95          # start backing away when the person is this close
PANIC_D = 0.30            # too close: commit to the sidestep whatever happens
CLEAR_D = 0.26            # lateral gap that counts as "out of the lane"
FOLLOW_D = 0.85           # trail the person at roughly this distance
FOLLOW_HYST = 0.12        # dead-band so FOLLOW/STOP doesn't chatter
CLOSING_EPS = 0.015       # m/s of range-rate that counts as approaching

# Narrow usable window for backing up. Below ~-0.30 the gait never leaves its
# onset threshold and the duck marches in place (looks stable, goes nowhere).
# Above ~-0.33 a long retreat accumulates enough drift to topple on the next
# transition. -0.31 both moves and holds the nominal trunk height of 0.116.
VX_RETREAT = -0.31
# MEASURED: sustained locomotion is what topples this policy, NOT the speed.
# vx=-0.31 held for 5 s (the retreat leg) keeps z=0.116, but the same command
# held for 12 s ends at z=0.040. Same for forward motion. So every walking
# burst is time-boxed and the duck is allowed to re-settle between bursts;
# it trails the person in steps rather than in one long march.
VX_FOLLOW = 0.22          # measured stable in 3.5 s bursts (z=0.117)
FOLLOW_BURST = 3.5
FOLLOW_REST = 2.5
VY_SIDESTEP = 0.12        # strafing speed (trained range is [-0.3, 0.3])
SIDESTEP_MAX = 2.5        # seconds; strafing is the least stable gait mode

# Gaze controller. MEASURED yaw authority (sustained command, 10 s):
#
#   vx= 0.00  wz=0.30 ->    +0.4 deg/s   (standing: no authority at all)
#   vx=-0.31  wz=0.30 ->   -18.8 deg/s   (walking: plenty of authority)
#   vx=-0.31  wz=0.60 ->  -189.2 deg/s   (spins out and falls, z=0.047)
#
# Two things worth writing down:
#  1. Yaw authority only exists WHILE WALKING. Measuring wz from a standstill
#     reads ~0 deg/s and wrongly suggests the policy cannot turn.
#  2. The sign is INVERTED: positive wz produces a NEGATIVE yaw rate. A naive
#     proportional controller is therefore positive feedback -- it winds up,
#     saturates, spins the duck out and drops it. That, not the gait, was the
#     "weird turn" at the end of the earlier videos.
YAW_KP = -0.5
WZ_MAX = 0.15             # 0.6 spins out; stay well under it
YAW_DEADBAND = 0.10       # rad (~6 deg); don't chase small errors

# Command low-pass. MUST be short: the walking policy has a gait-onset
# threshold, so a slow ramp leaves the duck marching in place for seconds
# (0.25 s => no displacement at all). Below ~0.04 s the filter overshoots
# (alpha > 1) and pushes the command out of the trained range -> falls.
CMD_TAU = 0.08            # seconds


def quat_rotate_inverse(quat, vec):
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def yaw_of(quat):
    """Heading angle of a wxyz quaternion, in the world XY plane."""
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def person_schedule(t, warmup, speed):
    """Scripted x of the person over time. Returns (x, phase).

    APPROACH  walks in at `speed`
    LEAVE     turns around and walks back out
    PAUSE     stops out there for a beat
    RETURN    comes back in, this time intending to walk straight through
    """
    # Legs are kept SHORT on purpose: measured limit is ~8 s of continuous
    # walking before this policy topples (vx=-0.31 for 5 s holds z=0.116; the
    # same command for 12 s ends at z=0.040). Forward motion behaves the same.
    x0, x_near, x_far = 1.40, 0.62, 1.75
    if t < warmup:
        return x0, "WAIT"
    t -= warmup

    d_app = (x0 - x_near) / speed
    if t < d_app:
        return x0 - speed * t, "APPROACH"
    t -= d_app

    d_leave = (x_far - x_near) / (speed * 1.6)
    if t < d_leave:
        return x_near + speed * 1.6 * t, "LEAVE"
    t -= d_leave

    d_pause = 2.5
    if t < d_pause:
        return x_far, "PAUSE"
    t -= d_pause

    return max(-1.0, x_far - speed * 1.8 * t), "RETURN"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="onnx/alpha_walking.onnx")
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--person-speed", type=float, default=0.12,
                   help="m/s the person walks")
    p.add_argument("--out", default="/tmp/f_yield")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--warmup", type=float, default=1.5)
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
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.9
    cam.elevation = -16
    cam.azimuth = 128

    import imageio.v2 as imageio
    frames = 0

    state = "IDLE"
    side_t = 0.0
    side_sign = 0.0
    prev_dist = None
    rate = 0.0                                 # low-passed range-rate
    cmd_s = np.zeros(3, dtype=np.float32)      # smoothed [vx, vy, wz]
    burst_t = 0.0
    walking = True
    alpha = (1.0 / CTRL_HZ) / CMD_TAU
    log = []

    for step in range(total):
        t = step / CTRL_HZ
        duck = data.xpos[trunk].copy()

        px, phase = person_schedule(t, args.warmup, args.person_speed)
        data.mocap_pos[person_mid][0] = px
        person = data.mocap_pos[person_mid].copy()

        dx = person[0] - duck[0]
        dy = person[1] - duck[1]
        dist = math.hypot(dx, dy)

        if prev_dist is None:
            prev_dist = dist
        raw_rate = (dist - prev_dist) * CTRL_HZ
        rate += 0.05 * (raw_rate - rate)        # negative => closing in
        prev_dist = dist
        closing = rate < -CLOSING_EPS
        receding = rate > CLOSING_EPS

        # --- gaze: proportional yaw control, always on --------------------
        yaw_err = wrap(math.atan2(dy, dx) - yaw_of(data.xquat[trunk]))
        wz = 0.0 if abs(yaw_err) < YAW_DEADBAND else \
            float(np.clip(YAW_KP * yaw_err, -WZ_MAX, WZ_MAX))

        # --- behaviour state machine ---------------------------------------
        # Intent is keyed on the person's scripted phase (the duck reacts to
        # what they are *doing*), with distance/closing as the trigger inside
        # each phase. Keying purely on range-rate made the states chatter:
        # a duck that walks changes `dist` itself, so it kept re-deciding.
        prev = state
        if state in ("SIDESTEP", "SETTLE") and phase != "RETURN":
            # a sidestep from an earlier phase; release it once they move on
            if dist > RETREAT_D:
                state = "WATCH"
        if state == "SIDESTEP":
            side_t += 1.0 / CTRL_HZ
            if abs(duck[1] - person[1]) > CLEAR_D or side_t > SIDESTEP_MAX:
                state = "SETTLE"
        elif phase == "WAIT":
            state = "IDLE"
        elif phase == "APPROACH":
            # first encounter: back off, and only bail sideways if cornered
            if dist < PANIC_D:
                state = "SIDESTEP"
                side_t = 0.0
                side_sign = 1.0 if dy <= 0.0 else -1.0
            else:
                state = "RETREAT" if dist < RETREAT_D else "WATCH"
        elif phase == "LEAVE":
            # watch them go, then trail after them once they are clear
            state = "FOLLOW" if dist > FOLLOW_D + FOLLOW_HYST else "WATCH"
        elif phase == "PAUSE":
            state = "FOLLOW" if dist > FOLLOW_D + FOLLOW_HYST else "STOP"
        else:  # RETURN: they come back at us, get out of the lane
            # Latching: once we have stepped aside we STAY aside. Re-evaluating
            # SETTLE against dist made it flip back into SIDESTEP every tick.
            if state not in ("SIDESTEP", "SETTLE"):
                if dist < RETREAT_D:
                    state = "SIDESTEP"
                    side_t = 0.0
                    side_sign = 1.0 if dy <= 0.0 else -1.0
                else:
                    state = "WATCH"

        if state == "RETREAT":
            target = np.array([VX_RETREAT, 0.0, wz], dtype=np.float32)
        elif state == "FOLLOW":
            # time-boxed burst: walk, then stand and recover, then walk again
            burst_t += 1.0 / CTRL_HZ
            if burst_t > FOLLOW_BURST:
                walking = not walking
                burst_t = 0.0
            target = np.array([VX_FOLLOW if walking else 0.0, 0.0, wz],
                              dtype=np.float32)
        elif state == "SIDESTEP":
            target = np.array([0.0, VY_SIDESTEP * side_sign, wz], dtype=np.float32)
        else:
            # WATCH / STOP / SETTLE / IDLE: hold position but keep looking
            target = np.array([0.0, 0.0, wz], dtype=np.float32)

        if t < args.warmup:
            target = np.zeros(3, dtype=np.float32)
        cmd_s += alpha * (target - cmd_s)

        if state != prev:
            log.append((t, prev, state, dist))
            print(f"  t={t:5.2f}s  {prev} -> {state}  "
                  f"(dist={dist:.3f} rate={rate:+.3f} person={phase})")

        # --- policy step ----------------------------------------------------
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
        data.ctrl[:] = DEFAULT_POSE[:nu] + action * args.action_scale

        for _ in range(decim):
            mujoco.mj_step(model, data)

        if step % 50 == 0:
            print(f"  t={t:5.1f}s {state:8s} {phase:8s} d={dist:.3f} "
                  f"yerr={math.degrees(yaw_err):+6.1f}deg "
                  f"duck=({duck[0]:+.3f},{duck[1]:+.3f},{duck[2]:.3f}) "
                  f"cmd=({cmd_s[0]:+.2f},{cmd_s[1]:+.2f},{cmd_s[2]:+.2f})")

        if step % frame_every == 0:
            cam.lookat[:] = 0.5 * (data.xpos[trunk] + data.mocap_pos[person_mid])
            renderer.update_scene(data, camera=cam)
            imageio.imwrite(os.path.join(args.out, f"f{frames:05d}.png"), renderer.render())
            frames += 1

    pos = data.xpos[trunk]
    print(f"frames={frames} final_state={state} "
          f"final_trunk_pos={pos} height={pos[2]:.3f}")
    print("transitions:", [(round(a, 2), b, c) for a, b, c, _ in log])


if __name__ == "__main__":
    main()
