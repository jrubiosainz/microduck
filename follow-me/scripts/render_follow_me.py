#!/usr/bin/env python3
"""Render and measure the Microduck follow-me choreography.

Sequence: forward follow -> left turn -> stop -> right sidestep -> backward.
Locomotion is the stock alpha_walking ONNX policy. Head gaze is rendered from
an isolated MjData copy so it cannot destabilize the validated walking physics.
"""
import argparse
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from camera_tracking import HeadCameraTracker
from follow_motion import (CTRL_HZ, FollowController, animate_person,
                           person_trajectory, wrap)
from video_overlay import PIP_H, PIP_W, compose

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)


def yaw_of_body(data, body_id):
    forward = data.xmat[body_id].reshape(3, 3)[:, 0]
    return math.atan2(float(forward[1]), float(forward[0]))


def make_observation(data, model, qpos_idx, qvel_idx, trunk_id, gyro_adr,
                     last_action, command, command_dim):
    quat = data.xquat[trunk_id].astype(np.float32)
    w, xyz = quat[0], quat[1:4]
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cross = np.cross(xyz, gravity_world) * 2.0
    gravity = gravity_world - w * cross + np.cross(xyz, cross)
    gyro = data.sensordata[gyro_adr:gyro_adr + 3].astype(np.float32)
    joint_pos = data.qpos[qpos_idx].astype(np.float32) - DEFAULT_POSE
    joint_vel = data.qvel[qvel_idx].astype(np.float32)
    policy_command = np.zeros(command_dim, dtype=np.float32)
    policy_command[:3] = command
    return np.concatenate([
        gyro, gravity, joint_pos, joint_vel, last_action, policy_command
    ]).astype(np.float32)


def phase_summary(records):
    result = {}
    for phase in sorted({r["phase"] for r in records}):
        rows = [r for r in records if r["phase"] == phase]
        result[phase] = {
            "samples": len(rows),
            "follow_rmse_m": math.sqrt(sum(r["follow_error_m"] ** 2 for r in rows) / len(rows)),
            "follow_max_m": max(r["follow_error_m"] for r in rows),
            "person_range_mean_m": sum(r["person_range_m"] for r in rows) / len(rows),
            "person_range_min_m": min(r["person_range_m"] for r in rows),
            "person_range_max_m": max(r["person_range_m"] for r in rows),
            "min_trunk_z_m": min(r["trunk_z_m"] for r in rows),
            "camera_visible_pct": 100.0 * sum(r["visible"] for r in rows) / len(rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="src/mjlab_microduck/robot/microduck/scene_follow_me.xml")
    parser.add_argument("--policy", default="onnx/alpha_walking.onnx")
    parser.add_argument("--seconds", type=float, default=33.0)
    parser.add_argument("--out", default="/tmp/follow-me-frames")
    parser.add_argument("--metrics", default="/tmp/follow-me-metrics.json")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if not args.no_render:
        os.makedirs(args.out, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    actuator_count = model.nu
    if actuator_count != 14:
        raise RuntimeError(f"expected 14 policy actuators, got {actuator_count}")
    qpos_idx = np.array([
        int(model.jnt_qposadr[model.actuator_trnid[i, 0]])
        for i in range(actuator_count)
    ])
    qvel_idx = np.array([
        int(model.jnt_dofadr[model.actuator_trnid[i, 0]])
        for i in range(actuator_count)
    ])
    for index, qpos_address in enumerate(qpos_idx):
        data.qpos[qpos_address] = DEFAULT_POSE[index]
    data.ctrl[:] = DEFAULT_POSE

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    person_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "person")
    person_mocap = int(model.body_mocapid[person_id])
    gyro_id = -1
    for sensor_name in ("imu_gyro", "gyro", "imu_ang_vel", "angular-velocity"):
        gyro_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if gyro_id >= 0:
            break
    if gyro_id < 0:
        raise RuntimeError("no angular-velocity sensor found in model")
    gyro_adr = int(model.sensor_adr[gyro_id])

    initial_person = person_trajectory(0.0)
    animate_person(model, data, initial_person, 0.0)
    mujoco.mj_forward(model, data)

    session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    observation_dim = int(session.get_inputs()[0].shape[1])
    command_dim = observation_dim - (3 + 3 + actuator_count * 3)
    if command_dim != 13:
        raise RuntimeError(f"expected 13 command dimensions, got {command_dim}")

    sim_dt = model.opt.timestep
    decimation = max(1, int(round((1.0 / CTRL_HZ) / sim_dt)))
    total_steps = int(args.seconds * CTRL_HZ)
    frame_every = max(1, int(round(CTRL_HZ / args.fps)))
    controller = FollowController()
    tracker = HeadCameraTracker(model, data, qpos_idx, trunk_id, (PIP_W, PIP_H))
    last_action = np.zeros(actuator_count, dtype=np.float32)
    previous_yaw = yaw_of_body(data, trunk_id)
    min_height = float(data.xpos[trunk_id, 2])
    records = []
    frames = 0

    if not args.no_render:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        pip_renderer = mujoco.Renderer(model, height=PIP_H, width=PIP_W)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 1.85
        camera.elevation = -20
        # Opposite-side presentation makes the leader's RIGHT leg travel
        # screen-right while retaining a clear three-quarter view of the duck.
        camera.azimuth = -45
        camera_lookat = np.array([0.4, 0.0, 0.15], dtype=np.float64)

    print(f"follow-me: {total_steps} steps, decimation={decimation}, render={not args.no_render}")
    last_phase = None
    for step in range(total_steps):
        t = step / CTRL_HZ
        person = person_trajectory(t)
        animate_person(model, data, person, t)
        mujoco.mj_forward(model, data)
        duck_pos_before = data.xpos[trunk_id].copy()
        duck_yaw_before = yaw_of_body(data, trunk_id)
        command, follow = controller.update(person, duck_pos_before, duck_yaw_before)

        observation = make_observation(
            data, model, qpos_idx, qvel_idx, trunk_id, gyro_adr,
            last_action, command, command_dim)
        action = session.run(
            [output_name], {input_name: observation.reshape(1, -1)}
        )[0].squeeze(0).astype(np.float32)
        last_action = action.copy()
        # 0.9 is the shipped walking action scale. 1.0 crosses the measured
        # stability boundary on long forward legs.
        data.ctrl[:] = DEFAULT_POSE + 0.9 * action
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        display_t = min(t + 1.0 / CTRL_HZ, args.seconds)
        display_person = person_trajectory(display_t)
        animate_person(model, data, display_person, display_t)
        mujoco.mj_forward(model, data)
        duck_pos = data.xpos[trunk_id].copy()
        duck_yaw = yaw_of_body(data, trunk_id)
        yaw_rate = math.degrees(wrap(duck_yaw - previous_yaw)) * CTRL_HZ
        previous_yaw = duck_yaw
        min_height = min(min_height, float(duck_pos[2]))
        camera_state = tracker.update(data)

        target_pos = follow["target_pos"]
        follow_error = float(np.linalg.norm(target_pos - duck_pos[:2]))
        person_range = float(np.linalg.norm(
            data.mocap_pos[person_mocap, :2] - duck_pos[:2]))
        follow.update({"error": follow_error, "person_range": person_range})
        row = {
            "t": display_t,
            "phase": display_person.phase,
            "follow_error_m": follow_error,
            "person_range_m": person_range,
            "yaw_error_deg": math.degrees(follow["yaw_error"]),
            "trunk_z_m": float(duck_pos[2]),
            "visible": bool(camera_state["visible"]),
            "off_axis_deg": math.degrees(camera_state["off_axis"]),
            "command": command.tolist(),
        }
        records.append(row)

        if display_person.phase != last_phase:
            print(f"  t={display_t:5.2f}s -> {display_person.phase:9s} "
                  f"err={follow_error:.3f} z={duck_pos[2]:.3f}")
            last_phase = display_person.phase
        if step % 50 == 0:
            print(f"  t={display_t:5.1f}s {display_person.phase:9s} "
                  f"err={follow_error:.3f} range={person_range:.3f} "
                  f"yaw={math.degrees(duck_yaw):+6.1f} z={duck_pos[2]:.3f} "
                  f"cmd=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f})")

        if not args.no_render and step % frame_every == 0:
            center = 0.5 * (duck_pos + data.mocap_pos[person_mocap])
            center[2] = 0.15
            camera_lookat += 0.08 * (center - camera_lookat)
            camera.lookat[:] = camera_lookat
            renderer.update_scene(tracker.gaze_data, camera=camera)
            pip_renderer.update_scene(tracker.gaze_data, camera=tracker.camera_id)
            image = compose(
                renderer.render(), pip_renderer.render(), t=display_t,
                total_seconds=args.seconds, person=display_person,
                duck_pos=duck_pos, duck_yaw=duck_yaw, follow=follow,
                command=command, camera=camera_state, yaw_rate=yaw_rate,
                min_height=min_height)
            imageio.imwrite(Path(args.out) / f"f{frames:05d}.png", np.asarray(image))
            frames += 1

    errors = [r["follow_error_m"] for r in records]
    summary = {
        "duration_s": args.seconds,
        "control_steps": total_steps,
        "frames": frames,
        "follow_distance_target_m": 0.50,
        "follow_rmse_m": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "follow_mean_m": sum(errors) / len(errors),
        "follow_max_m": max(errors),
        "person_range_mean_m": sum(r["person_range_m"] for r in records) / len(records),
        "person_range_min_m": min(r["person_range_m"] for r in records),
        "person_range_max_m": max(r["person_range_m"] for r in records),
        "min_trunk_z_m": min(r["trunk_z_m"] for r in records),
        "final_trunk_z_m": records[-1]["trunk_z_m"],
        "fallen_steps": sum(r["trunk_z_m"] < 0.09 for r in records),
        "camera_visible_steps": total_steps - tracker.lost_steps,
        "camera_lost_steps": tracker.lost_steps,
        "camera_visible_pct": 100.0 * (total_steps - tracker.lost_steps) / total_steps,
        "camera_rms_off_axis_deg": math.degrees(tracker.rms_off_axis),
        "camera_max_off_axis_deg": math.degrees(tracker.max_off_axis),
        "phases": phase_summary(records),
    }
    Path(args.metrics).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
