#!/usr/bin/env python3
"""Build the contact sheet: one tile per named moment of the behavior.

FRAMES ARE SELECTED BY TIME THROUGH THE MANIFEST, NEVER BY ``t * fps``.
``render_frames.FrameWriter`` emits every ``round(50 / fps)``-th control tick, so
frame number equals ``t * fps`` only when the control rate is an exact multiple
of the output rate.  At 4 fps the writer emits every 12th tick and frame 200 is
t = 48.02 s rather than t = 50.00 s.  The manifest records the real tick time of
every written frame, so the sheet reads it and picks the nearest.

Each tile is labelled with the moment it shows, the state the machine was in,
and the wall-clock time, so the sheet is readable without the video.

Run:
    ../../microduck_rl/.venv/bin/python tools/build_contact_sheet.py \\
        --frames /tmp/wbm-final --out media/walk-beside-me-contact-sheet.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# The moments the sheet has to show, in order.  These are the phases the
# behavior claims, so a reader can check the claim against a picture.
MOMENTS: tuple[tuple[float, str], ...] = (
    (0.50, "right slot refused: hedge"),
    (3.00, "joining her left side"),
    (6.50, "beside her, on her left"),
    (8.90, "left lane blocked: kiosk"),
    (11.00, "falling back astern"),
    (15.00, "crossing behind her"),
    (18.50, "committed to the far side"),
    (22.00, "coming up her right side"),
    (29.60, "beside her, on her right"),
    (37.50, "bend 1: left 42.9 deg"),
    (53.50, "bend 2: right -42.9 deg"),
    (71.00, "bend 3: left 81.3 deg"),
)

COLUMNS = 3
TILE_W = 480
LABEL_H = 34
PAD = 6
BACKGROUND = (14, 16, 22)
INK = (232, 236, 244)
DIM = (150, 158, 172)


def _font(size: int):
    for path in ("/System/Library/Fonts/SFNSMono.ttf",
                 "/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Supplemental/Courier New.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="/tmp/wbm-final")
    parser.add_argument("--out",
                        default="media/walk-beside-me-contact-sheet.jpg")
    parser.add_argument("--quality", type=int, default=88)
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    manifest = json.loads((frames_dir / "frames.json").read_text())
    if not manifest:
        raise SystemExit("empty manifest; render frames first")

    picks = []
    for target, caption in MOMENTS:
        entry = min(manifest, key=lambda m: abs(m["t"] - target))
        picks.append((entry, caption))
        print(f"  {caption:<28} asked {target:6.2f}s  "
              f"got t={entry['t']:6.2f}s  {entry['state']:<16} "
              f"frame {entry['frame']}")

    probe = Image.open(frames_dir / f"f{picks[0][0]['frame']:05d}.png")
    tile_h = round(TILE_W * probe.size[1] / probe.size[0])
    rows = (len(picks) + COLUMNS - 1) // COLUMNS
    sheet_w = COLUMNS * TILE_W + (COLUMNS + 1) * PAD
    sheet_h = rows * (tile_h + LABEL_H) + (rows + 1) * PAD

    sheet = Image.new("RGB", (sheet_w, sheet_h), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    caption_font = _font(15)
    detail_font = _font(13)

    for index, (entry, caption) in enumerate(picks):
        column, row = index % COLUMNS, index // COLUMNS
        x = PAD + column * (TILE_W + PAD)
        y = PAD + row * (tile_h + LABEL_H + PAD)
        tile = Image.open(frames_dir / f"f{entry['frame']:05d}.png")
        sheet.paste(tile.resize((TILE_W, tile_h), Image.LANCZOS), (x, y))
        draw.text((x + 4, y + tile_h + 4), caption, font=caption_font, fill=INK)
        draw.text((x + 4, y + tile_h + 19),
                  f"t = {entry['t']:.2f} s    {entry['state']}",
                  font=detail_font, fill=DIM)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=args.quality, optimize=True)
    print(f"wrote {out}  {sheet.size[0]}x{sheet.size[1]}  "
          f"{out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
