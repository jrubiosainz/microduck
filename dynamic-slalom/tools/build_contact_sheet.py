#!/usr/bin/env python3
"""Build a contact sheet: one frame per act of the behavior, selected BY TIME.

Selection goes through the render's own manifest rather than assuming
``frame == t * fps``.  That assumption is false whenever the control rate is not
an exact multiple of the frame rate, and a contact sheet built on it silently
shows the wrong moment - which is worse than no contact sheet, because it looks
like evidence.

EACH TILE IS AN ENCOUNTER, NOT A CLOCK TICK.  The sheet follows the machine:
the first frame of each pass and each wait, so a reader sees the five decisions
and the three refusals in order rather than twelve evenly-spaced stills.

Run:
    ../../microduck_rl/.venv/bin/python tools/build_contact_sheet.py \\
        --frames /tmp/sl-stills --manifest /tmp/sl-final-manifest.json \\
        --metrics media/dynamic-slalom-metrics.json \\
        --out media/dynamic-slalom-contact-sheet.jpg
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

# The acts of the behavior, in order.  Each entry is a (state, caption) pair
# whose Nth occurrence is wanted: ``build_selection`` walks the manifest and
# takes each state's successive ENTRIES, so the five CHOOSE tiles are five
# different encounters rather than five frames of the first one.
OPENING = [
    ("PLAN", "planning a line to the arrival band"),
    ("ADVANCE", "walking the planned line"),
]
CLOSING = [
    ("GOAL", "arrived: standing in the band, command exactly zero"),
]

COLUMNS = 3
TILE_W = 460
LABEL_H = 34


def entries_of(manifest, state):
    """The first frame of each CONTIGUOUS run of ``state`` in the manifest.

    Contiguous runs rather than every frame, because a state the duck enters
    five times should yield five tiles, not five hundred.
    """
    out = []
    previous = None
    for entry in manifest:
        if entry["state"] == state and previous != state:
            out.append(entry)
        previous = entry["state"]
    return out


def build_selection(manifest):
    """One tile per act: the opening, every decision and refusal, the arrival."""
    chosen = []
    for state, caption in OPENING:
        found = entries_of(manifest, state)
        if found:
            chosen.append((found[0], caption))

    # Every WAIT and every committed corridor, interleaved in time so the sheet
    # reads as the run does.
    decisions = []
    for entry in entries_of(manifest, "WAIT"):
        decisions.append((entry, "WAIT: neither corridor predicted safe"))
    for entry in entries_of(manifest, "CHOOSE_LEFT"):
        decisions.append((entry, "committed LEFT, having scored both hands"))
    for entry in entries_of(manifest, "CHOOSE_RIGHT"):
        decisions.append((entry, "committed RIGHT, having scored both hands"))
    decisions.sort(key=lambda pair: pair[0]["t"])
    chosen.extend(decisions)

    for state, caption in CLOSING:
        found = entries_of(manifest, state)
        if found:
            chosen.append((found[0], caption))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--manifest", default="",
                        help="defaults to <frames>/frames.json")
    parser.add_argument("--metrics", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quality", type=int, default=84)
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    manifest_path = (Path(args.manifest) if args.manifest
                     else frames_dir / "frames.json")
    if not manifest_path.is_file():
        raise SystemExit(f"no manifest at {manifest_path}; render first")
    manifest = json.loads(manifest_path.read_text())

    chosen = build_selection(manifest)
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
