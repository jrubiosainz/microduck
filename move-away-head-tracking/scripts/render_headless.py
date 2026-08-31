#!/usr/bin/env python3
"""Headless ONNX policy rollout -> mp4 frames. No viewer, no keyboard.

Usage:
  python scripts/render_headless.py --policy onnx/alpha_walking.onnx \
      --vx 0.15 --seconds 10 --out /tmp/duck_frames
"""
import argparse, os, math
import numpy as np
import mujoco
import onnxruntime as ort

XML = "src/mjlab_microduck/robot/microduck/scene.xml"

DEFAULT_POSE = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
], dtype=np.float32)

CTRL_HZ = 50.0


def quat_rotate_inverse(quat, vec):
    w = quat[0]
    xyz = quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="onnx/alpha_walking.onnx")
    p.add_argument("--vx", type=float, default=0.15)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--out", default="/tmp/duck_frames")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--warmup", type=float, default=1.0,
                   help="seconds with zero command before applying the twist")
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

    # start from the default pose so the policy sees an in-distribution state
    for k, j in enumerate(qpos_idx):
        data.qpos[j] = DEFAULT_POSE[k]
    data.ctrl[:] = DEFAULT_POSE[:nu]
    mujoco.mj_forward(model, data)

    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    obs_dim = sess.get_inputs()[0].shape[1]
    cmd_dim = obs_dim - (3 + 3 + nu * 3)
    print(f"obs_dim={obs_dim} nu={nu} -> command dim {cmd_dim}")

    last_action = np.zeros(nu, dtype=np.float32)
    sim_dt = model.opt.timestep
    decim = max(1, int(round((1.0 / CTRL_HZ) / sim_dt)))
    total_ctrl_steps = int(args.seconds * CTRL_HZ)
    frame_every = max(1, int(round(CTRL_HZ / args.fps)))
    print(f"sim_dt={sim_dt} decimation={decim} ctrl_steps={total_ctrl_steps}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 0.85
    cam.elevation = -12
    cam.azimuth = 135

    import imageio.v2 as imageio
    frames_written = 0
    warm_steps = int(args.warmup * CTRL_HZ)

    for step in range(total_ctrl_steps):
        gyro = data.sensordata[gyro_adr:gyro_adr + 3].astype(np.float32)
        quat = data.xquat[trunk].astype(np.float32)
        grav = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        jpos = data.qpos[qpos_idx].astype(np.float32) - DEFAULT_POSE[:nu]
        jvel = data.qvel[qvel_idx].astype(np.float32)

        cmd = np.zeros(cmd_dim, dtype=np.float32)
        if step >= warm_steps:
            cmd[0], cmd[1], cmd[2] = args.vx, args.vy, args.wz

        obs = np.concatenate([gyro, grav, jpos, jvel, last_action, cmd]).astype(np.float32)
        action = sess.run([out_name], {in_name: obs.reshape(1, -1)})[0].squeeze(0).astype(np.float32)
        last_action = action.copy()
        data.ctrl[:] = DEFAULT_POSE[:nu] + action * args.action_scale

        for _ in range(decim):
            mujoco.mj_step(model, data)

        if step % 50 == 0:
            tp = data.xpos[trunk]
            print(f"  t={step/CTRL_HZ:5.1f}s x={tp[0]:+.4f} y={tp[1]:+.4f} z={tp[2]:.4f}")

        if step % frame_every == 0:
            cam.lookat[:] = data.xpos[trunk]
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            imageio.imwrite(os.path.join(args.out, f"f{frames_written:05d}.png"), img)
            frames_written += 1

    pos = data.xpos[trunk]
    print(f"frames={frames_written} final_trunk_pos={pos} height={pos[2]:.3f}")


if __name__ == "__main__":
    main()
