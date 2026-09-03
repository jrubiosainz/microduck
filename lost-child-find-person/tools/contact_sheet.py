#!/usr/bin/env python3
"""Build a labelled contact sheet from rendered frames.

Selects frames by WALL-CLOCK TIME using the frame manifest the renderer writes,
not by assuming ``frame == t * fps``.  That assumption is false whenever the
50 Hz control rate is not an exact multiple of the frame rate: at 4 fps the
writer emits every 12th tick, so 250 frames span 60 s at 4.167 fps of simulated
time and frame 200 is t = 48.02 s rather than t = 50.00 s.  An earlier sheet
built on that assumption captioned a SEARCH_SWEEP frame as "REJOIN cycle 2".

Each tile is captioned with the time it actually came from and with the state
the rollout was in at that tick, so a mislabelled tile is self-evident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for path in ("/System/Library/Fonts/SFNSMono.ttf",
                 "/System/Library/Fonts/Menlo.ttc",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build(frames_dir: Path, moments, out: Path, columns: int = 3,
          tile_w: int = 640) -> Path:
    paths = sorted(frames_dir.glob("f*.png"))
    if not paths:
        raise SystemExit(f"no frames in {frames_dir}")
    manifest_path = frames_dir / "frames.json"
    if not manifest_path.exists():
        raise SystemExit(f"no frame manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    tiles = []
    for t, label in moments:
        entry = min(manifest, key=lambda row: abs(row["t"] - t))
        image = Image.open(paths[entry["frame"]]).convert("RGB")
        scale = tile_w / image.width
        tile = image.resize((tile_w, int(image.height * scale)), Image.LANCZOS)
        draw = ImageDraw.Draw(tile)
        caption = f"t={entry['t']:5.2f}s  {entry['state']:<13} {label}"
        font = _font(17)
        box = draw.textbbox((0, 0), caption, font=font)
        draw.rectangle([0, 0, box[2] + 16, box[3] + 12], fill=(10, 12, 18))
        draw.text((8, 4), caption, font=font, fill=(240, 244, 250))
        tiles.append(tile)
        if abs(entry["t"] - t) > 0.30:
            print(f"  WARNING: wanted t={t:.2f}s, nearest frame is "
                  f"t={entry['t']:.2f}s")

    tile_h = tiles[0].height
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w + (columns + 1) * 8,
                              rows * tile_h + (rows + 1) * 8), (24, 26, 32))
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet.paste(tile, (8 + column * (tile_w + 8), 8 + row * (tile_h + 8)))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    print(f"wrote {out}  {sheet.width}x{sheet.height}  {len(tiles)} tiles")
    return out


# The nine moments the render has to prove, at the times the METRICS report
# them.  Refusals are placed a beat after the recorded rejection instant so the
# tile lands inside the REJECT hold rather than on its first tick.
MOMENTS = [
    (8.00, "following the guardian in the open"),
    (20.00, "kiosk geometrically occludes the guardian"),
    (18.80, "sofia refused - same teal, same bag, 12 cm shorter"),
    (21.70, "mira refused - same teal, same bag, WEARS A CAP"),
    (23.40, "faruq refused - shirt colour differs from the teal"),
    (25.06, "priya reacquired - score 1.000, confirmed 0.90 s"),
    (30.00, "rejoin cycle 1 - planned route, 2.744 m walked"),
    (50.40, "rejoin cycle 2 - second loss resolved, 0.432 m"),
    (59.98, "final standoff 0.7057 m, guardian visible"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile-width", type=int, default=640)
    args = parser.parse_args()
    build(Path(args.frames), MOMENTS, Path(args.out), tile_w=args.tile_width)
