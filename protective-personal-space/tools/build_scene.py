#!/usr/bin/env python3
"""Generate the Protective Personal Space MuJoCo plaza.

People and fixtures are non-colliding kinematic proxies.  Avoidance therefore
comes from the controller and is graded by exact geometric clearance, never by
the contact solver.  The PiP rig is rendering-only.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pps_cast import PEOPLE
from pps_plaza import DUCK_START, DUCK_START_YAW_DEG, FIXTURES, FLOOR_HALF, WALL_HALF_Z, WALL_T

HEADER = '''<mujoco model="scene_protective_personal_space">
  <include file="robot_walk.xml" />
  <visual><headlight diffuse=".8 .8 .8" ambient=".45 .45 .45" specular="0 0 0"/>
    <global azimuth="145" elevation="-30" offwidth="1600" offheight="1200"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".18 .21 .27" rgb2=".04 .05 .08" width="512" height="3072"/>
    <texture type="2d" name="ground" builtin="checker" rgb1=".28 .29 .31" rgb2=".22 .23 .25" width="256" height="256"/>
    <material name="ground" texture="ground" texuniform="true" texrepeat="18 15"/>
    <material name="floor" rgba=".34 .35 .38 1"/><material name="wall" rgba=".72 .71 .68 1"/>
    <material name="kioskmat" rgba=".25 .39 .57 1"/><material name="columnmat" rgba=".68 .68 .66 1"/>
    <material name="plantmat" rgba=".25 .48 .28 1"/><material name="benchmat" rgba=".48 .35 .24 1"/>
    <material name="bollardmat" rgba=".85 .58 .16 1"/><material name="skin" rgba=".86 .66 .52 1"/>
    <material name="trouser" rgba=".12 .15 .20 1"/><material name="shoe" rgba=".04 .05 .06 1"/>
    <material name="buffer" rgba=".20 .78 .98 .18"/><material name="escort" rgba=".24 .95 .55 .55"/>
    <material name="threat" rgba=".98 .30 .24 .48"/><material name="escape" rgba=".98 .78 .22 .48"/>
'''

WORLD = '''  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="ground_plane" type="plane" size="0 0 .05" material="ground"/>
    <body name="pps_rig" mocap="true" pos="0 0 .2"><camera name="pps_camera" fovy="140"/></body>
'''


def box(name, x, y, z, hx, hy, hz, material):
    return (f'    <geom name="{name}" type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
            f'size="{hx:.4f} {hy:.4f} {hz:.4f}" material="{material}" contype="0" conaffinity="0"/>\n')


def cyl(name, x, y, z, radius, half_h, material):
    return (f'    <geom name="{name}" type="cylinder" pos="{x:.4f} {y:.4f} {z:.4f}" '
            f'size="{radius:.4f} {half_h:.4f}" material="{material}" contype="0" conaffinity="0"/>\n')


def fixture_block():
    out = [box("plaza_floor", 0, 0, .002, FLOOR_HALF[0], FLOOR_HALF[1], .002, "floor")]
    out += [box("wall_e", FLOOR_HALF[0] + WALL_T, 0, WALL_HALF_Z, WALL_T, FLOOR_HALF[1] + .12, WALL_HALF_Z, "wall"),
            box("wall_w", -FLOOR_HALF[0] - WALL_T, 0, WALL_HALF_Z, WALL_T, FLOOR_HALF[1] + .12, WALL_HALF_Z, "wall"),
            box("wall_n", 0, FLOOR_HALF[1] + WALL_T, WALL_HALF_Z, FLOOR_HALF[0] + .12, WALL_T, WALL_HALF_Z, "wall"),
            box("wall_s", 0, -FLOOR_HALF[1] - WALL_T, WALL_HALF_Z, FLOOR_HALF[0] + .12, WALL_T, WALL_HALF_Z, "wall")]
    for f in FIXTURES:
        z = .5 * f.height_m
        if f.kind == "cylinder":
            out.append(cyl(f.name, f.center[0], f.center[1], z, f.radius, z, f.material))
        else:
            out.append(box(f.name, f.center[0], f.center[1], z, f.half[0], f.half[1], z, f.material))
    return "".join(out)


def actor_block(person):
    n, s = person.name, person.stature
    lines = [f'    <body name="actor_{n}" mocap="true" pos="0 0 -3">',
             f'      <geom name="{n}_torso" type="capsule" fromto="0 0 {-0.10*s:.4f} 0 0 {0.16*s:.4f}" size="{0.078*s:.4f}" material="{n}_shirt" contype="0" conaffinity="0"/>',
             f'      <geom name="{n}_head" type="sphere" pos="0 0 {0.255*s:.4f}" size="{0.064*s:.4f}" material="skin" contype="0" conaffinity="0"/>']
    for side, sy in (("l", .036), ("r", -.036)):
        lines.append(f'      <body name="{n}_leg_{side}" pos="0 {sy*s:+.4f} {-0.10*s:.4f}"><joint name="{n}_hip_{side}" type="hinge" axis="0 1 0" range="-42 42" damping="1"/><geom type="capsule" fromto="0 0 0 0 0 {-0.26*s:.4f}" size="{0.033*s:.4f}" material="trouser" contype="0" conaffinity="0"/><geom type="capsule" fromto="0 0 {-0.26*s:.4f} {0.07*s:.4f} 0 {-0.272*s:.4f}" size="{0.035*s:.4f}" material="shoe" contype="0" conaffinity="0"/></body>')
    lines.append("    </body>")
    return "\n".join(lines) + "\n"


def marker(name, material, radius):
    return (f'    <body name="{name}" mocap="true" pos="0 0 -3"><geom name="{name}_disc" '
            f'type="cylinder" size="{radius:.3f} .004" material="{material}" contype="0" conaffinity="0"/></body>\n')


def build():
    mats = "".join(f'    <material name="{p.name}_shirt" rgba="{p.rgba}"/>\n' for p in PEOPLE)
    parts = [HEADER, mats, WORLD, fixture_block()]
    parts += [marker("ward_buffer", "buffer", 1.95), marker("escort_target", "escort", .10),
              marker("interpose_target", "threat", .10), marker("escape_target", "escape", .10),
              marker("active_threat", "threat", .08)]
    for i in range(16): parts.append(marker(f"prediction_{i}", "threat", .035))
    parts.extend(actor_block(p) for p in PEOPLE)
    parts.append("  </worldbody>\n")

    yaw = math.radians(DUCK_START_YAW_DEG)
    base = f"{DUCK_START[0]:.4f} {DUCK_START[1]:.4f} .12 {math.cos(yaw/2):.6f} 0 0 {math.sin(yaw/2):.6f}"
    robot = "0 -0.0872664626 -0.457924 -0.004940 0.452984 0.3490658504 0.3490658504 0 0 0 0.0872664626 0.457924 0.004940 -0.452984"
    pads = " ".join(["0"] * (2 * len(PEOPLE)))
    parts.append(f'''  <keyframe>
    <key name="INIT" qpos="{base} {" ".join(["0"]*(14+2*len(PEOPLE)))}" ctrl="{" ".join(["0"]*14)}"/>
    <key name="STAND" qpos="{base} {robot} {pads}" ctrl="{robot}"/>
  </keyframe>
</mujoco>
''')
    return "".join(parts)


def main():
    out = ROOT / "assets" / "scene_protective_personal_space.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size} bytes), people={len(PEOPLE)}, fixtures={len(FIXTURES)}")


if __name__ == "__main__": main()
