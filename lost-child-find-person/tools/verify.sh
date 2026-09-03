#!/bin/bash
# Final verification for lost-child-find-person.  Run with: bash tools/verify.sh
# Long inline pipelines feeding a while-read loop lose the environment in this
# shell, so every multi-step check lives in a script file.
set -u
cd "$(dirname "$0")/.."
ROOT="$PWD"
LAB="$(cd .. && pwd)"
PY="${PY:-$LAB/../microduck_rl/.venv/bin/python}"
FAIL=0

pass() { printf '  [ OK ] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; FAIL=1; }

echo "== 1. every source file compiles =="
if "$PY" -m compileall -q scripts tools tests > /tmp/lcfp-compile.log 2>&1; then
    pass "compileall scripts tools tests"
else
    fail "compileall"; cat /tmp/lcfp-compile.log
fi

echo "== 2. no module written for this behavior exceeds 300 lines =="
# contact_geometry.py and policy_runtime.py are INHERITED verbatim from earlier
# behaviors, and rollout_lost/lost_metrics/lost_memory/lost_camera predate this
# rendering phase; the modules listed here are the ones authored for it.
OVER=0
for f in scripts/hud_style.py scripts/hud_views.py scripts/hud_panels.py \
         scripts/video_overlay.py scripts/render_frames.py \
         scripts/render_lost_child.py tools/contact_sheet.py; do
    n=$(wc -l < "$f")
    if [ "$n" -gt 300 ]; then printf '    %5s %s\n' "$n" "$f"; OVER=1; fi
done
if [ "$OVER" -eq 0 ]; then pass "all render modules <= 300 lines"
else fail "a render module exceeds 300 lines"; fi

echo "== 3. tests =="
if "$PY" -m pytest tests -q > /tmp/lcfp-tests.log 2>&1; then
    pass "$(tail -1 /tmp/lcfp-tests.log)"
else
    fail "pytest"; tail -20 /tmp/lcfp-tests.log
fi

echo "== 4. scene XML is well formed and the model loads =="
if "$PY" -c "import xml.dom.minidom; xml.dom.minidom.parse('assets/scene_lost_child.xml')" 2>/dev/null; then
    pass "scene XML parses"
else
    fail "scene XML is malformed"
fi
if (cd scripts && "$PY" -c "
import sys; sys.path.insert(0, '.')
from policy_runtime import load_scene
m = load_scene(None, None)
assert m.nbody > 0 and m.ncam >= 2
" 2>/dev/null); then
    pass "MuJoCo loads the scene and finds both cameras"
else
    fail "MuJoCo could not load the scene"
fi

echo "== 5. the headless gate has no rendering dependency =="
if (cd scripts && "$PY" -c "
import sys
class Block:
    def find_module(self, name, path=None):
        return self if name.split('.')[0] in ('PIL', 'imageio') else None
    def load_module(self, name):
        raise ImportError('blocked ' + name)
sys.meta_path.insert(0, Block())
sys.path.insert(0, '.')
import validate_lost, rollout_lost, lost_metrics, lost_camera
assert 'PIL' not in sys.modules and 'imageio' not in sys.modules
" 2>/dev/null); then
    pass "validate_lost imports with PIL and imageio blocked"
else
    fail "the headless gate pulls in a rendering dependency"
fi

echo "== 6. policy is the byte-identical stock walking policy =="
WANT="e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
GOT=$(shasum -a 256 onnx/alpha_walking.onnx | cut -d' ' -f1)
UPSTREAM=$(shasum -a 256 "$LAB/../microduck_rl/onnx/alpha_walking.onnx" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] && pass "policy sha256 matches the documented value" \
    || fail "policy sha256 is $GOT"
[ "$GOT" = "$UPSTREAM" ] && pass "policy matches the upstream checkout" \
    || fail "policy differs from upstream"

echo "== 7. media integrity =="
for f in media/lost-child-find-person.mp4 \
         media/lost-child-find-person-telegram.mp4; do
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
    frames=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "$f")
    wh=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate,codec_name,pix_fmt \
        -of csv=p=0 "$f")
    errs=$(ffmpeg -v error -i "$f" -f null - 2>&1 | wc -l | tr -d ' ')
    size=$(stat -f%z "$f")
    printf '    %-46s %sB  %ss  %s frames  %s  decode_errors=%s\n' \
        "$(basename "$f")" "$size" "$dur" "$frames" "$wh" "$errs"
    if [ "$frames" = "3000" ] && [ "$errs" = "0" ] && [ "$dur" = "60.000000" ]; then
        pass "$(basename "$f"): 3000 frames, 60.000 s, decodes cleanly"
    else
        fail "$(basename "$f") failed the decode/frame/duration check"
    fi
done
TG=$(stat -f%z media/lost-child-find-person-telegram.mp4)
[ "$TG" -lt 5242880 ] && pass "telegram derivative is under the 5 MB limit ($TG B)" \
    || fail "telegram derivative is $TG bytes"

echo "== 8. metrics agree with the published claims =="
"$PY" - <<'PYEOF'
import json, pathlib
m = json.loads(
    pathlib.Path("media/lost-child-find-person-metrics.json").read_text())
checks = [
    ("all 25 gates pass", m["all_gates_pass"] is True
     and len(m["gate_results"]) == 25),
    ("60 s / 3000 steps", m["seconds"] == 60.0 and m["control_steps"] == 3000),
    ("obs 61-D", m["observation_dim"] == 61),
    ("action scale 0.9", m["action_scale"] == 0.9),
    ("gyro imu_ang_vel", m["gyro_sensor"] == "imu_ang_vel"),
    ("zero falls", m["fallen_steps"] == 0),
    ("zero contacts", m["contact_steps"] == 0),
    ("zero blind movement", m["blind_movement_steps"] == 0),
    ("min z >= 0.09", m["min_trunk_z_m"] >= 0.09),
    ("final z ~ 0.116", abs(m["final_trunk_z_m"] - 0.116) <= 0.012),
    ("two loss/rejoin cycles", m["cycle_count"] == 2),
    # The transition into LOST is logged at 16.58 s; the first record CARRYING
    # the LOST label is stamped one control tick later at 16.60 s, because a
    # record's timestamp is the end of the tick it describes.  Both are pinned
    # so the two numbers can never silently drift apart.
    ("FOLLOW->LOST transition at 16.58 s",
     [round(t["t"], 2) for t in m["transitions"] if t["to"] == "LOST"]
     == [16.58, 46.34]),
    ("first LOST record at 16.60 s", m["first_loss_at_s"] == 16.6),
    ("cycle 0 lost at 16.58 s", m["cycles"][0]["lost_at_s"] == 16.58),
    ("7.48 s geometric occlusion", m["longest_geometric_occlusion_s"] == 7.48),
    ("occluder is the kiosk", m["longest_geometric_occluder"] == "obs_kiosk"),
    ("three distinct refusals",
     m["distinct_rejected"] == ["sofia", "mira", "faruq"]),
    ("both look-alikes refused",
     sorted(m["rejected_lookalikes"]) == ["mira", "sofia"]),
    ("zero wrong accepts", m["wrong_accepts"] == []),
    ("only the guardian accepted", m["guardian_names_seen"] == ["priya"]),
    ("rejoin paths 2.744 / 0.432 m",
     [round(c["rejoin_path_m"], 3) for c in m["cycles"]] == [2.744, 0.432]),
    ("rejoin visibility 100%",
     all(c["target_visible_fraction_with_los"] == 1.0 for c in m["cycles"])),
    ("final range 0.7057 m", m["final_range_m"] == 0.7057),
    ("final range in band",
     m["standoff_band_m"][0] <= m["final_range_m"] <= m["standoff_band_m"][1]),
    ("guardian visible at the end", m["final_guardian_visible"] is True),
    ("clearances 0.105 / 0.1875 m",
     m["min_person_clearance_m"] == 0.105
     and m["min_scenery_clearance_m"] == 0.1875),
    ("stationary states hold exactly zero",
     all(v == 0.0 for v in m["stationary_command_peak"].values())),
    ("half-extent is labelled a pose-zero sample",
     m["adult_half_extent_m"] == 0.1375
     and "pose-zero" in m.get("adult_half_extent_basis", "")),
]
for label, ok in checks:
    print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
raise SystemExit(0 if all(ok for _, ok in checks) else 1)
PYEOF
[ $? -eq 0 ] || FAIL=1

echo "== 9. frozen behavior folders are untouched =="
cd "$LAB"
DIRTY=$(git status --porcelain -- move-away move-away-crowd follow-me \
    follow-me-among-others come-here-recall crosswalk-guardian \
    narrow-corridor-etiquette queue-politely 2>/dev/null \
    | grep -v '^?? ' | wc -l | tr -d ' ')
[ "$DIRTY" = "0" ] && pass "no tracked change in any frozen folder" \
    || fail "$DIRTY tracked change(s) in frozen folders"
DIFF=$(git diff --stat -- move-away move-away-crowd follow-me \
    follow-me-among-others come-here-recall crosswalk-guardian \
    narrow-corridor-etiquette queue-politely | wc -l | tr -d ' ')
[ "$DIFF" = "0" ] && pass "git diff against frozen folders is empty" \
    || fail "git diff reports changes in frozen folders"
cd "$ROOT"

echo "== 10. cache cleanup =="
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null
rm -rf .pytest_cache 2>/dev/null
LEFT=$(find . \( -name '__pycache__' -o -name '*.pyc' -o -name '.pytest_cache' \) | wc -l | tr -d ' ')
[ "$LEFT" = "0" ] && pass "no caches left in the behavior folder" \
    || fail "$LEFT cache artifact(s) remain"

echo
[ "$FAIL" -eq 0 ] && echo "ALL VERIFICATION CHECKS PASSED" \
    || echo "VERIFICATION FAILED"
exit "$FAIL"
