#!/usr/bin/env python3
"""HUD and duck-camera PiP composition for the follow-me video."""
import math

from PIL import Image, ImageDraw, ImageFont

PIP_W, PIP_H = 225, 165
PHASES = ["READY", "FORWARD", "LEFT TURN", "STOP", "RIGHT TURN", "BACKWARD", "DONE"]
PHASE_COLORS = {
    "READY": (180, 210, 255),
    "FORWARD": (100, 235, 145),
    "LEFT TURN": (255, 205, 80),
    "STOP": (255, 120, 120),
    "RIGHT TURN": (120, 215, 255),
    "BACKWARD": (215, 155, 255),
    "DONE": (180, 255, 180),
    "STOPPED": (255, 120, 120),
}


def _font(size=14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", size)
    except OSError:
        return ImageFont.load_default()


def compose(main_rgb, pip_rgb, *, t, total_seconds, person, duck_pos,
            duck_yaw, follow, command, camera, yaw_rate, min_height):
    image = Image.fromarray(main_rgb)
    pip = Image.fromarray(pip_rgb)
    draw = ImageDraw.Draw(image)
    font = _font(14)
    small = _font(12)
    width, height = image.size
    phase_color = PHASE_COLORS[person.phase]

    draw.rectangle([0, 0, width, 126], fill=(4, 8, 14))
    replay = (follow['replay_phase'] if person.moving else 'STOPPED')
    replay_color = PHASE_COLORS.get(replay, (235, 240, 245))
    draw.text((12, 8), f"FOLLOW-ME · TRUE LEFT / RIGHT   t={t:05.2f}s", fill=phase_color, font=font)
    draw.text((12, 31), f"LEADER: {person.phase:<9}   DUCK REPLAYS: {replay:<9}",
              fill=replay_color, font=small)
    draw.text((12, 51),
              f"world-path lag={follow['spatial_lag']:.3f} m   "
              f"trail error={follow['error']:.3f} m   person range={follow['person_range']:.3f} m",
              fill=(235, 240, 245), font=small)
    draw.text((12, 71),
              f"command vx={command[0]:+.3f} vy={command[1]:+.3f} wz={command[2]:+.3f}   "
              f"heading error={math.degrees(follow['yaw_error']):+5.1f} deg",
              fill=(235, 240, 245), font=small)
    stable = duck_pos[2] >= 0.09
    draw.text((12, 91),
              f"trunk z={duck_pos[2]:.3f} m   min={min_height:.3f} m   "
              f"{'UPRIGHT' if stable else 'FALLEN'}",
              fill=(145, 255, 165) if stable else (255, 90, 90), font=small)

    visible = camera["visible"]
    draw.text((12, 111),
              f"head camera: {'PERSON VISIBLE' if visible else 'TARGET LOST'}   "
              f"off-axis={math.degrees(camera['off_axis']):.1f} deg",
              fill=(120, 255, 140) if visible else (255, 110, 110), font=small)

    px0, py0 = width - PIP_W - 12, 138
    border = (100, 255, 130) if visible else (255, 90, 90)
    draw.rectangle([px0 - 4, py0 - 22, px0 + PIP_W + 4, py0 + PIP_H + 4],
                   fill=(2, 5, 8))
    draw.text((px0, py0 - 19), "DUCK VIEW · STABILIZED HEAD CAM",
              fill=border, font=small)
    image.paste(pip, (px0, py0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([px0 - 1, py0 - 1, px0 + PIP_W, py0 + PIP_H],
                   outline=border, width=3)
    cx, cy = px0 + PIP_W // 2, py0 + PIP_H // 2
    draw.line([cx - 8, cy, cx + 8, cy], fill=(255, 235, 80), width=1)
    draw.line([cx, cy - 8, cx, cy + 8], fill=(255, 235, 80), width=1)

    # Timeline makes the requested choreography readable at a glance.
    bar_y = height - 34
    x0, x1 = 12, width - 12
    draw.rectangle([x0, bar_y, x1, bar_y + 12], fill=(25, 34, 46))
    boundaries = [0, 2, 7, 15, 18, 35, 41, total_seconds]
    for index, phase in enumerate(PHASES):
        left = x0 + (x1 - x0) * boundaries[index] / total_seconds
        right = x0 + (x1 - x0) * boundaries[index + 1] / total_seconds
        color = PHASE_COLORS[phase]
        draw.rectangle([left, bar_y, right, bar_y + 12], fill=color)
        if right - left > 55:
            draw.text((left + 3, bar_y - 15), phase, fill=color, font=small)
    marker = x0 + (x1 - x0) * min(t / total_seconds, 1.0)
    draw.line([marker, bar_y - 4, marker, bar_y + 17], fill=(255, 255, 255), width=2)
    return image
