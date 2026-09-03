#!/usr/bin/env python3
"""Build a contact sheet: one frame per act of the behavior, selected BY TIME.

Selection goes through the render's own manifest rather than assuming
``frame == t * fps``.  That assumption is false whenever the control rate is not
an exact multiple of the frame rate, and a contact sheet built on it silently
shows the wrong moment - which is worse than no contact sheet, because it looks
like evidence.

EACH TILE IS AN ACT, NOT A CLOCK TICK.  The sheet follows the machine: the
search, the first reading, each of the six commands being carried out, the
moment the STOP cuts into the approach, and the goodbye - so a reader sees the
session in order rather than twelve evenly-spaced stills.

Run:
    ../../microduck_rl/.venv/bin/python tools/build_contact_sheet.py \\
        --frames /tmp/gr_frames --manifest /tmp/gr_manifest.json \\
        --out media/gesture-response-contact-sheet.jpg
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

COLUMNS = 3
TILE_W = 460
LABEL_H = 34

# The acts, in the order they happen.  Each entry names a state and the caption
# for its Nth contiguous occurrence; ``build_selection`` walks the manifest and
# takes each state's successive ENTRIES, so the two ACK tiles are two different
# acknowledgments rather than two frames of the first one.
#
# THE SHEET IS THE ARGUMENT IN TWELVE PICTURES: it has to show the six commands
# being carried out AND the two refusals, because a robot that only ever says
# yes has not demonstrated judgment.
ACTS = [
    ("READY", "sweeping the head, looking for its instructor"),
    ("OBSERVE", "locked onto mira; watching, command exactly zero"),
    ("CONFIRM", "reading a gesture: timing it before acting on it"),
    ("EXECUTE_APPROACH", "COME: closing on her, 1.75 m of real walking"),
    ("EXECUTE_STOP", "STOP given MID-WALK: command cut to an exact zero"),
    ("ACK", "acknowledging, standing still"),
    ("EXECUTE_TURN_LEFT", "TURN LEFT: a real +64.5 deg walked arc"),
    ("EXECUTE_TURN_RIGHT", "TURN RIGHT: the mirror, -64.3 deg"),
    ("EXECUTE_BACK_UP", "BACK UP: reversing along its own heading"),
    ("GOODBYE", "waved off: the session is complete"),
    ("DONE", "standing down, six commands carried out"),
]


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
    """One tile per act, in the order the acts actually happened."""
    chosen = []
    for state, caption in ACTS:
        found = entries_of(manifest, state)
        if not found:
            continue
        chosen.append((found[0], caption))
        # ACK happens after every command; the SECOND one is shown too so the
        # sheet reads as a session of repeated commands rather than one.
        if state == "ACK" and len(found) > 1:
            chosen.append((found[1], "acknowledging the next command"))
    chosen.sort(key=lambda pair: pair[0]["t"])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--manifest", default="",
                        help="defaults to <frames>/frames.json")
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
        if not path.is_file():
            continue
        tile = Image.open(path).convert("RGB").resize(
            (TILE_W, tile_h), Image.LANCZOS)
        column = index % COLUMNS
        row = index // COLUMNS
        x = column * TILE_W
        y = row * (tile_h + LABEL_H)
        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + tile_h + 4),
                  f"{entry['t']:7.2f}s  {entry['state']}", font=F11, fill=INK)
        draw.text((x + 8, y + tile_h + 19), caption, font=F09, fill=DIM)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=args.quality, optimize=True)
    print(f"wrote {out} ({out.stat().st_size} bytes), "
          f"{len(chosen)} tiles from {len(manifest)} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
