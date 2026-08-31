#!/usr/bin/env python3
"""Render SEARCH→FOUND→FOLLOW→STOP among moving pedestrians."""
import argparse
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from camera_search import CrowdCameraSearch
from crowd_motion import (COLORS, CTRL_HZ, TARGET_SEQUENCE, TRAIL_DISTANCE,
                          CrowdFollowController, FootstepTrail,
                          SearchFollowStateMachine, animate_crowd,
                          crowd_trajectory, wrap)
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
    observation = np.concatenate([
        gyro, gravity, joint_pos, joint_vel, last_action, policy_command
    ]).astype(np.float32)
    if observation.shape != (61,):
        raise RuntimeError(f"expected 61-D observation, got {observation.shape}")
    return observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="src/mjlab_microduck/robot/microduck/scene_follow_me_among_others.xml")
    parser.add_argument("--policy", default="onnx/alpha_walking.onnx")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", default="/tmp/follow-me-among-others-frames")
    parser.add_argument("--metrics", default="/tmp/follow-me-among-others-metrics.json")
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
    if model.nu != 14:
        raise RuntimeError(f"expected 14 policy actuators, got {model.nu}")
    qpos_idx = np.array([
        int(model.jnt_qposadr[model.actuator_trnid[i, 0]])
        for i in range(model.nu)
    ])
    qvel_idx = np.array([
        int(model.jnt_dofadr[model.actuator_trnid[i, 0]])
        for i in range(model.nu)
    ])
    for index, address in enumerate(qpos_idx):
        data.qpos[address] = DEFAULT_POSE[index]
    data.ctrl[:] = DEFAULT_POSE

    trunk_id = model.body("trunk_base").id
    people_mocap = {
        color: int(model.body_mocapid[model.body(f"person_{color.lower()}").id])
        for color in COLORS
    }
    target_mocap = int(model.body_mocapid[model.body("trail_target").id])
    gyro_id = -1
    for sensor_name in ("imu_gyro", "gyro", "imu_ang_vel", "angular-velocity"):
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if gyro_id >= 0:
            break
    if gyro_id < 0:
        raise RuntimeError("no angular-velocity sensor found in model")
    gyro_adr = int(model.sensor_adr[gyro_id])

    initial_crowd = crowd_trajectory(0.0)
    animate_crowd(model, data, initial_crowd, 0.0)
    mujoco.mj_forward(model, data)
    trails = {color: FootstepTrail(initial_crowd[color]) for color in COLORS}

    session = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    observation_dim = int(session.get_inputs()[0].shape[1])
    command_dim = observation_dim - (3 + 3 + model.nu * 3)
    if observation_dim != 61 or command_dim != 13:
        raise RuntimeError(
            f"policy shape mismatch: observation={observation_dim}, command={command_dim}")

    decimation = max(1, int(round((1.0 / CTRL_HZ) / model.opt.timestep)))
    total_steps = int(args.seconds * CTRL_HZ)
    frame_every = max(1, int(round(CTRL_HZ / args.fps)))
    controller = CrowdFollowController()
    machine = SearchFollowStateMachine()
    camera_search = CrowdCameraSearch(
        model, data, qpos_idx, trunk_id, (PIP_W, PIP_H))
    last_action = np.zeros(model.nu, dtype=np.float32)
    records = []
    transitions = []
    frames = 0
    min_height = float(data.xpos[trunk_id, 2])

    if not args.no_render:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        pip_renderer = mujoco.Renderer(model, height=PIP_H, width=PIP_W)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 2.85
        camera.elevation = -27
        camera.azimuth = -48
        camera_lookat = np.array([0.75, 0.0, 0.16], dtype=np.float64)

    print(f"follow-me-among-others: {total_steps} steps, targets="
          f"{'->'.join(TARGET_SEQUENCE)}, lag={TRAIL_DISTANCE:.2f}m, "
          f"decimation={decimation}, render={not args.no_render}")
    last_state_key = None
    for step in range(total_steps):
        t = step / CTRL_HZ
        crowd = crowd_trajectory(t)
        animate_crowd(model, data, crowd, t)
        mujoco.mj_forward(model, data)
        trail_states = {color: trails[color].update(crowd[color]) for color in COLORS}

        state = machine.state if not machine.done else "DONE"
        target = machine.target
        selection = min(machine.index + 1, len(TARGET_SEQUENCE))
        state_elapsed = t - machine.state_since
        duck_pos_before = data.xpos[trunk_id].copy()
        duck_yaw_before = yaw_of_body(data, trunk_id)
        active_trail = trail_states[target]
        command, follow = controller.update(
            state == "FOLLOW", active_trail, duck_pos_before, duck_yaw_before)
        if state == "FOLLOW":
            data.mocap_pos[target_mocap] = np.array([
                active_trail.pos[0], active_trail.pos[1], 0.012])
        else:
            data.mocap_pos[target_mocap] = np.array([0.0, 0.0, -0.10])

        observation = make_observation(
            data, model, qpos_idx, qvel_idx, trunk_id, gyro_adr,
            last_action, command, command_dim)
        action = session.run(
            [output_name], {input_name: observation.reshape(1, -1)}
        )[0].squeeze(0).astype(np.float32)
        last_action = action.copy()
        data.ctrl[:] = DEFAULT_POSE + 0.9 * action
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        display_t = min(t + 1.0 / CTRL_HZ, args.seconds)
        display_crowd = crowd_trajectory(display_t)
        animate_crowd(model, data, display_crowd, display_t)
        mujoco.mj_forward(model, data)
        duck_pos = data.xpos[trunk_id].copy()
        duck_yaw = yaw_of_body(data, trunk_id)
        min_height = min(min_height, float(duck_pos[2]))
        camera_state = camera_search.update(
            data, target_color=target, mode=state,
            mode_elapsed=state_elapsed, duck_yaw=duck_yaw)

        follow_error = float(np.linalg.norm(active_trail.pos - duck_pos[:2]))
        follow.update({"error": follow_error})
        person_pos = data.mocap_pos[people_mocap[target], :2]
        person_range = float(np.linalg.norm(person_pos - duck_pos[:2]))
        row = {
            "t": display_t,
            "state": state,
            "selection": selection,
            "target": target,
            "state_elapsed_s": state_elapsed,
            "target_visible": bool(camera_state["target_visible"]),
            "target_off_axis_deg": math.degrees(camera_state["target_off_axis"]),
            "visible_colors": camera_state["visible_colors"],
            "trail_target_xy": active_trail.pos.tolist(),
            "follow_error_m": follow_error,
            "person_range_m": person_range,
            "yaw_error_deg": math.degrees(follow["yaw_error"]),
            "duck_yaw_deg": math.degrees(duck_yaw),
            "duck_xy": duck_pos[:2].tolist(),
            "trunk_z_m": float(duck_pos[2]),
            "command": command.tolist(),
        }
        records.append(row)

        next_state, next_target, changed = machine.update(display_t, camera_state)
        if changed:
            event = {
                "t": display_t,
                "from": state,
                "to": next_state,
                "target": next_target,
                "selection": min(machine.index + 1, len(TARGET_SEQUENCE)),
            }
            transitions.append(event)
            print(f"  t={display_t:5.2f}s {state:10s} -> {next_state:10s} "
                  f"target={next_target}")
        state_key = (selection, target, state)
        if state_key != last_state_key:
            print(f"  t={display_t:5.2f}s CYCLE {selection}/4 target={target:5s} "
                  f"state={state}")
            last_state_key = state_key
        if step % 50 == 0:
            print(f"  t={display_t:5.1f}s {state:10s} target={target:5s} "
                  f"seen={camera_state['target_visible']} "
                  f"off={math.degrees(camera_state['target_off_axis']):5.1f} "
                  f"err={follow_error:.3f} range={person_range:.3f} "
                  f"duck=({duck_pos[0]:+.2f},{duck_pos[1]:+.2f}) "
                  f"trail=({active_trail.pos[0]:+.2f},{active_trail.pos[1]:+.2f}) "
                  f"yaw={math.degrees(duck_yaw):+.1f} "
                  f"z={duck_pos[2]:.3f} cmd="
                  f"({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f})")

        if not args.no_render and step % frame_every == 0:
            all_positions = np.vstack([
                duck_pos[:2], *[display_crowd[color].pos for color in COLORS]
            ])
            center = np.array([
                float(np.mean(all_positions[:, 0])),
                float(np.mean(all_positions[:, 1])), 0.16,
            ])
            camera_lookat += 0.05 * (center - camera_lookat)
            camera.lookat[:] = camera_lookat
            renderer.update_scene(camera_search.gaze_data, camera=camera)
            pip_renderer.update_scene(
                camera_search.gaze_data, camera=camera_search.camera_id)
            image = compose(
                renderer.render(), pip_renderer.render(), t=display_t,
                total_seconds=args.seconds, state=state,
                state_elapsed=state_elapsed, selection=selection, target=target,
                duck_pos=duck_pos, follow=follow, command=command,
                camera=camera_state, min_height=min_height,
                completed_cycles=len(machine.cycles))
            imageio.imwrite(Path(args.out) / f"f{frames:05d}.png", np.asarray(image))
            frames += 1

    if not machine.done:
        raise RuntimeError(
            f"sequence incomplete after {args.seconds:.1f}s: "
            f"selection={machine.index + 1}, state={machine.state}")
    follow_rows = [r for r in records if r["state"] == "FOLLOW"]
    follow_by_selection = []
    for selection_index, target_color in enumerate(TARGET_SEQUENCE, start=1):
        rows = [r for r in follow_rows if r["selection"] == selection_index]
        positions = [np.asarray(r["duck_xy"], dtype=np.float64) for r in rows]
        path_distance = sum(float(np.linalg.norm(b - a))
                            for a, b in zip(positions, positions[1:]))
        net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        follow_by_selection.append({
            "selection": selection_index,
            "target": target_color,
            "path_distance_m": path_distance,
            "net_displacement_m": net_displacement,
            "start_error_m": rows[0]["follow_error_m"],
            "end_error_m": rows[-1]["follow_error_m"],
            "min_error_m": min(r["follow_error_m"] for r in rows),
            "start_yaw_error_deg": rows[0]["yaw_error_deg"],
            "end_yaw_error_deg": rows[-1]["yaw_error_deg"],
        })
    summary = {
        "duration_s": args.seconds,
        "control_steps": total_steps,
        "frames": frames,
        "target_sequence_requested": list(TARGET_SEQUENCE),
        "target_sequence_completed": [c["target"] for c in machine.cycles],
        "pattern": ["SEARCH", "FOUND", "FOLLOW", "STOP"],
        "cycles_completed": len(machine.cycles),
        "cycles": machine.cycles,
        "follow_by_selection": follow_by_selection,
        "all_follow_segments_moved": all(
            segment["path_distance_m"] >= 0.40
            and segment["net_displacement_m"] >= 0.30
            for segment in follow_by_selection),
        "all_follow_segments_approached": all(
            segment["min_error_m"] <= segment["start_error_m"] - 0.05
            for segment in follow_by_selection),
        "transitions": transitions,
        "trail_distance_m": TRAIL_DISTANCE,
        "follow_rmse_m": math.sqrt(sum(
            r["follow_error_m"] ** 2 for r in follow_rows) / len(follow_rows)),
        "follow_max_m": max(r["follow_error_m"] for r in follow_rows),
        "person_range_follow_mean_m": sum(
            r["person_range_m"] for r in follow_rows) / len(follow_rows),
        "person_range_follow_min_m": min(r["person_range_m"] for r in follow_rows),
        "person_range_follow_max_m": max(r["person_range_m"] for r in follow_rows),
        "search_duration_mean_s": sum(
            c["search_duration_s"] for c in machine.cycles) / len(machine.cycles),
        "search_duration_max_s": max(
            c["search_duration_s"] for c in machine.cycles),
        "found_while_target_visible": all(
            any(r["target_visible"] for r in records
                if r["target"] == cycle["target"]
                and abs(r["t"] - cycle["found_s"]) < 0.04)
            for cycle in machine.cycles),
        "wrong_color_locks": sum(
            cycle["target"] != TARGET_SEQUENCE[index]
            for index, cycle in enumerate(machine.cycles)),
        "camera_target_visible_follow_pct": 100.0 * sum(
            r["target_visible"] for r in follow_rows) / len(follow_rows),
        "stationary_state_command_max": max(
            float(np.linalg.norm(r["command"])) for r in records
            if r["state"] in ("SEARCH", "FOUND", "STOP", "DONE")),
        "min_trunk_z_m": min(r["trunk_z_m"] for r in records),
        "final_trunk_z_m": records[-1]["trunk_z_m"],
        "fallen_steps": sum(r["trunk_z_m"] < 0.09 for r in records),
        "camera_target_visible_steps": camera_search.target_visible_steps,
        "camera_search_steps": camera_search.search_steps,
        "camera_search_target_visible_steps": camera_search.search_target_visible_steps,
    }
    Path(args.metrics).write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["all_follow_segments_moved"]:
        raise RuntimeError("one or more FOLLOW segments did not produce locomotion")
    if not summary["all_follow_segments_approached"]:
        raise RuntimeError("one or more FOLLOW segments never approached its target")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
