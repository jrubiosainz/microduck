#!/usr/bin/env python3
"""Build a contact sheet: one frame per ACT of the behavior, selected BY TIME.

Selection goes through the render's own manifest rather than assuming
``frame == t * fps``.  That assumption is false whenever the control rate is not
an exact multiple of the frame rate, and a contact sheet built on it silently
shows the wrong moment - which is worse than no contact sheet, because it looks
like evidence.

EACH TILE IS A MOMENT THE SCENARIO REQUIRED, NOT A CLOCK TICK.  The sheet walks
the machine's own episode log, so it shows the four intrusion cycles from their
alternating bearings, the false near-pass being watched and dismissed, the yield
to the protected person, and the squeeze - in the order they happened.

Run:
    ../../microduck_rl/.venv/bin/python tools/build_pps_contact_sheet.py \\
        --frames /tmp/pps_frames --manifest /tmp/pps_manifest.json \\
        --summary /tmp/pps_render.json \\
        --out media/protective-personal-space-contact-sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from PIL import Image, ImageDraw  # noqa: E402

from hud_style import F09, F11  # noqa: E402
from pps_hud_style import DIM, INK, PANEL  # noqa: E402

COLUMNS = 3
TILE_W = 470
LABEL_H = 34

# The acts, in the order they happen.  Each entry is a state and the caption for
# its Nth contiguous ENTRY into that state, so four separate INTERPOSE tiles are
# four different encounters rather than four frames of the first one.
ACTS: tuple[tuple[str, int, str], ...] = (
    ("ESCORT", 0, "joining the escort slot beside and behind Aina"),
    ("MONITOR", 0, "escort established: watching everyone, acting on nobody"),
    ("INTERPOSE", 0, "cycle 1 - Dario from the EAST: walking to the station"),
    ("HOLD_BUFFER", 0, "cycle 1 - on station between them, EXACT zero"),
    ("INTERPOSE", 1, "cycle 2 - Noor from the WEST: the opposite bearing"),
    ("HOLD_BUFFER", 1, "cycle 2 - holding the line on the far side"),
    ("INTERPOSE", 2, "cycle 3 - Yara from the NORTH-EAST"),
    ("PERSON_APPROACH", 0, "Aina walks AT the duck: it yields, never blocks"),
    ("RETREAT", 0, "reversing along its own heading to give her room"),
    ("MULTI_THREAT", 0, "the SQUEEZE: two converging at once, no station works"),
    ("ESCAPE_GAP", 0, "leaving the pinch through the measured safe gap"),
    ("DONE", 0, "escort restored, session complete, standing down"),
)


def entries_of(manifest, state):
    """The first frame of each CONTIGUOUS run of ``state`` in the manifest."""
    out, previous = [], None
    for entry in manifest:
        if entry["state"] == state and previous != state:
            out.append(entry)
        previous = entry["state"]
    return out


def build_selection(manifest):
    """One tile per act, in the order the acts actually happened."""
    chosen = []
    for state, occurrence, caption in ACTS:
        found = entries_of(manifest, state)
        if len(found) <= occurrence:
            continue
        chosen.append((found[occurrence], caption))
    chosen.sort(key=lambda pair: pair[0]["t"])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--manifest", default="",
                        help="defaults to <frames>/frames.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quality", type=int, default=82)
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

    drawn = 0
    for index, (entry, caption) in enumerate(chosen):
        path = frames_dir / f"f{entry['frame']:05d}.png"
        if not path.is_file():
            continue
        tile = Image.open(path).convert("RGB").resize(
            (TILE_W, tile_h), Image.LANCZOS)
        column, row = index % COLUMNS, index // COLUMNS
        x, y = column * TILE_W, row * (tile_h + LABEL_H)
        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + tile_h + 4),
                  f"{entry['t']:7.2f}s  {entry['state']}", font=F11, fill=INK)
        draw.text((x + 8, y + tile_h + 19), caption, font=F09, fill=DIM)
        drawn += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=args.quality, optimize=True)
    print(f"wrote {out} ({out.stat().st_size} bytes), "
          f"{drawn} tiles from {len(manifest)} frames")
    for entry, caption in chosen:
        print(f"  {entry['t']:7.2f}s  {entry['state']:<18} {caption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
