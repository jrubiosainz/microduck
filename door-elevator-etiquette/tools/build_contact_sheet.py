#!/usr/bin/env python3
"""Build a contact sheet: one frame per phase, selected BY TIME.

Selection goes through the render's own ``frames.json`` manifest rather than
assuming ``frame == t * fps``.  That assumption is false whenever the control
rate is not an exact multiple of the frame rate, and a contact sheet built on it
silently shows the wrong moment - which is worse than no contact sheet, because
it looks like evidence.

Run:
    ../../microduck_rl/.venv/bin/python tools/build_contact_sheet.py \\
        --frames /tmp/dee-final \\
        --metrics media/door-elevator-etiquette-metrics.json \\
        --out media/door-elevator-etiquette-contact-sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from PIL import Image, ImageDraw  # noqa: E402

from hud_style import DIM, F09, F11, INK, PANEL  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# One row per act of the behavior.  Each entry names the STATE whose first
# appearance is wanted, so the sheet follows the machine rather than a clock.
WANTED = [
    ("APPROACH_DOOR", "walking up to the automatic door"),
    ("YIELD_EXITERS", "stopped outside the threshold: two people coming out"),
    ("FOLLOW_THROUGH", "through the doorway, behind her"),
    ("APPROACH_LIFT", "crossing the lobby"),
    ("WAIT_SIDE", "waiting BESIDE the doors, not in front"),
    ("LET_OCCUPANTS_EXIT", "letting all three occupants out first"),
    ("FOLLOW_GUARDIAN_IN", "boarding after her"),
    ("POSITION_INSIDE", "moving to the side of the car"),
    ("RIDE", "riding: exactly still, sealed car"),
    ("DOORS_OPEN_TARGET", "arrived; she steps out first"),
    ("FOLLOW_OUT", "following her off the lift"),
    ("DONE", "on the target floor"),
]

COLUMNS = 3
TILE_W = 480
LABEL_H = 34


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--metrics", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quality", type=int, default=86)
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    manifest_path = frames_dir / "frames.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no manifest at {manifest_path}; render first")
    manifest = json.loads(manifest_path.read_text())

    # First frame of each wanted state, by TIME through the manifest.
    chosen: list[tuple[dict, str]] = []
    for state, caption in WANTED:
        entry = next((m for m in manifest if m["state"] == state), None)
        if entry is None:
            print(f"  warning: no frame for {state}")
            continue
        chosen.append((entry, caption))

    if not chosen:
        raise SystemExit("nothing to draw")

    first = Image.open(frames_dir / f"f{chosen[0][0]['frame']:05d}.png")
    scale = TILE_W / first.width
    tile_h = int(first.height * scale)
    rows = (len(chosen) + COLUMNS - 1) // COLUMNS
    sheet = Image.new(
        "RGB", (COLUMNS * TILE_W, rows * (tile_h + LABEL_H)), PANEL)
    draw = ImageDraw.Draw(sheet)

    for index, (entry, caption) in enumerate(chosen):
        path = frames_dir / f"f{entry['frame']:05d}.png"
        tile = Image.open(path).convert("RGB").resize(
            (TILE_W, tile_h), Image.LANCZOS)
        column = index % COLUMNS
        row = index // COLUMNS
        x = column * TILE_W
        y = row * (tile_h + LABEL_H)
        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + tile_h + 4),
                  f"{entry['t']:6.2f}s  {entry['state']}", font=F11, fill=INK)
        draw.text((x + 8, y + tile_h + 19), caption, font=F09, fill=DIM)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=args.quality, optimize=True)
    print(f"wrote {out} ({out.stat().st_size} bytes), "
          f"{len(chosen)} tiles from {len(manifest)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
