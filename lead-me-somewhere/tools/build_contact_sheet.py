#!/usr/bin/env python3
"""Build a contact sheet from the rendered frames, selecting BY TIME.

Frames are chosen through the render manifest rather than by assuming
``frame == t * fps``, which is false whenever the control rate is not an exact
multiple of the frame rate.  Each tile is captioned with its timestamp and the
state the rollout was actually in, so the sheet is evidence rather than
decoration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from PIL import Image, ImageDraw  # noqa: E402

from hud_style import BAD, DIM, F10, F11, INK, STATE_COLORS  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# The moments the sheet must show, one per phase of the scenario.  Chosen by
# WHAT THEY DEMONSTRATE, not by even spacing.  The captions are checked against
# the manifest's own state and corrected from it — an earlier draft captioned
# t=86 s "indicating" when the rollout was still in LEAD, which is exactly the
# kind of caption that makes a contact sheet a claim instead of evidence.
MOMENTS = [
    (1.0, "asked, standing still"),
    (3.6, "searching the route"),
    (12.0, "leading"),
    (25.5, "lag detected"),
    (30.0, "waiting, command 0.0"),
    (36.5, "she caught up; resuming"),
    (48.5, "second lag"),
    (58.0, "waiting again"),
    (70.0, "leading the last leg"),
    (84.0, "closing on the lifts"),
    (87.0, "arrived; indicating"),
    (92.0, "delivered"),
]

# The state each moment is REQUIRED to be in.  A caption that disagrees with the
# manifest is a caption that is wrong, so it fails loudly rather than shipping.
EXPECTED_STATE = {
    1.0: "RECEIVE_DESTINATION",
    3.6: "PLAN",
    12.0: "LEAD",
    25.5: ("CHECK_FOLLOWER", "WAIT_FOR_PERSON"),
    30.0: "WAIT_FOR_PERSON",
    36.5: ("RESUME", "LEAD"),
    48.5: ("CHECK_FOLLOWER", "WAIT_FOR_PERSON", "LEAD"),
    58.0: "WAIT_FOR_PERSON",
    70.0: ("LEAD", "RESUME"),
    84.0: "LEAD",
    87.0: ("INDICATE", "ARRIVE"),
    92.0: "DONE",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="/tmp/lms-final")
    parser.add_argument("--out",
                        default=str(REPO / "media"
                                    / "lead-me-somewhere-contact-sheet.jpg"))
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--quality", type=int, default=86)
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    manifest_path = frames_dir / "frames.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no manifest at {manifest_path}; render first")
    manifest = json.loads(manifest_path.read_text())

    def nearest(t: float) -> dict:
        return min(manifest, key=lambda entry: abs(entry["t"] - t))

    chosen = [(nearest(t), caption) for t, caption in MOMENTS]

    # A caption that disagrees with the state the rollout was actually in makes
    # the sheet a claim rather than evidence, so it fails here.
    mismatches = []
    for (requested_t, caption), (entry, _) in zip(MOMENTS, chosen):
        expected = EXPECTED_STATE.get(requested_t)
        if expected is None:
            continue
        allowed = (expected,) if isinstance(expected, str) else expected
        if entry["state"] not in allowed:
            mismatches.append(
                f"t={requested_t}s captioned {caption!r} but the rollout was "
                f"in {entry['state']}, not {allowed}")
    if mismatches:
        raise SystemExit("contact sheet captions disagree with the run:\n  "
                         + "\n  ".join(mismatches))

    first = Image.open(frames_dir / f"f{chosen[0][0]['frame']:05d}.png")
    scale = args.tile_width / first.width
    tile_w = args.tile_width
    tile_h = int(first.height * scale)
    caption_h = 34
    cols = args.cols
    rows = (len(chosen) + cols - 1) // cols

    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + caption_h)),
                      (10, 12, 16))
    draw = ImageDraw.Draw(sheet)

    for index, (entry, caption) in enumerate(chosen):
        path = frames_dir / f"f{entry['frame']:05d}.png"
        if not path.is_file():
            continue
        tile = Image.open(path).resize((tile_w, tile_h), Image.LANCZOS)
        col, row = index % cols, index // cols
        x, y = col * tile_w, row * (tile_h + caption_h)
        sheet.paste(tile, (x, y))
        state = entry["state"]
        draw.text((x + 8, y + tile_h + 5),
                  f"{entry['t']:6.2f}s  {state}", font=F11,
                  fill=STATE_COLORS.get(state, INK))
        draw.text((x + 8, y + tile_h + 19), caption, font=F10, fill=DIM)
        draw.rectangle([x, y, x + tile_w - 1, y + tile_h + caption_h - 1],
                       outline=(38, 44, 56))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=args.quality, optimize=True)
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out} ({size / 1024:.0f} KB, {sheet.width}x{sheet.height})")
    for entry, caption in chosen:
        print(f"  t={entry['t']:6.2f}s  frame {entry['frame']:5d}  "
              f"{entry['state']:<20} {caption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
