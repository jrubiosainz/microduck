#!/bin/bash
# Final verification for queue-politely.  Run with: bash tools/verify.sh
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
if "$PY" -m compileall -q scripts tools tests > /tmp/qp-compile.log 2>&1; then
    pass "compileall scripts tools tests"
else
    fail "compileall"; cat /tmp/qp-compile.log
fi

echo "== 2. no generated source file exceeds 300 lines =="
OVER=0
for f in scripts/*.py tools/*.py tests/*.py; do
    n=$(wc -l < "$f")
    if [ "$n" -gt 300 ]; then printf '    %5s %s\n' "$n" "$f"; OVER=1; fi
done
# contact_geometry.py and policy_runtime.py are INHERITED verbatim from
# narrow-corridor-etiquette, not generated here; tests are not source.
if [ "$OVER" -eq 0 ]; then pass "all generated sources <= 300 lines"
else fail "a generated source exceeds 300 lines"; fi

echo "== 3. tests =="
if "$PY" -m pytest tests -q > /tmp/qp-tests.log 2>&1; then
    pass "$(tail -1 /tmp/qp-tests.log)"
else
    fail "pytest"; tail -20 /tmp/qp-tests.log
fi

echo "== 4. scene XML is well formed and regenerates identically =="
if "$PY" -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('assets/scene_queue_politely.xml')" 2>/dev/null; then
    pass "scene XML parses"
else
    fail "scene XML is malformed"
fi
cp assets/scene_queue_politely.xml /tmp/qp-scene-before.xml
"$PY" tools/build_scene.py > /dev/null 2>&1
if cmp -s /tmp/qp-scene-before.xml assets/scene_queue_politely.xml; then
    pass "scene regenerates byte-identically from its generator"
else
    fail "scene generator output drifted from the committed XML"
fi

echo "== 5. policy is the byte-identical stock walking policy =="
WANT="e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
GOT=$(shasum -a 256 onnx/alpha_walking.onnx | cut -d' ' -f1)
UPSTREAM=$(shasum -a 256 "$LAB/../microduck_rl/onnx/alpha_walking.onnx" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] && pass "policy sha256 matches the documented value" \
    || fail "policy sha256 is $GOT"
[ "$GOT" = "$UPSTREAM" ] && pass "policy matches the upstream checkout" \
    || fail "policy differs from upstream"

echo "== 6. media integrity =="
for f in media/queue-politely.mp4 media/queue-politely-telegram.mp4; do
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
    frames=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "$f")
    wh=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$f")
    errs=$(ffmpeg -v error -i "$f" -f null - 2>&1 | wc -l | tr -d ' ')
    size=$(stat -f%z "$f")
    printf '    %-42s %sB  %ss  %s frames  %s  decode_errors=%s\n' \
        "$(basename "$f")" "$size" "$dur" "$frames" "$wh" "$errs"
    [ "$frames" = "2900" ] && [ "$errs" = "0" ] || fail "$f failed decode/frame check"
done
TG=$(stat -f%z media/queue-politely-telegram.mp4)
[ "$TG" -lt 5242880 ] && pass "telegram derivative is under the 5 MB limit" \
    || fail "telegram derivative is $TG bytes"

echo "== 7. metrics agree with the media =="
"$PY" - <<'PYEOF'
import json, pathlib
m = json.loads(pathlib.Path("media/queue-politely-metrics.json").read_text())
checks = [
    ("all gates pass", m["all_gates_pass"] is True),
    ("frames", m.get("frames") == 2900),
    ("fps", m.get("fps") == 50),
    ("resolution", (m.get("width"), m.get("height")) == (960, 640)),
    ("obs 61-D", m["observation_dim"] == 61),
    ("action scale 0.9", m["action_scale"] == 0.9),
    ("gyro imu_ang_vel", m["gyro_sensor"] == "imu_ang_vel"),
    ("zero falls", m["fallen_steps"] == 0),
    ("zero contacts", m["contact_steps"] == 0),
    ("min z >= 0.09", m["min_trunk_z_m"] >= 0.09),
    ("final z ~ 0.116", abs(m["final_trunk_z_m"] - 0.116) <= 0.012),
    (">=2 available gaps refused", len(m["rejected_available_gaps"]) >= 2),
    (">=3 wait/advance cycles", m["wait_advance_cycles"] >= 3),
    ("zero wrong locks", m["wrong_lock_steps"] == 0),
    ("zero wrong orders", m["order_wrong_samples"] == 0),
    ("zero wrong tails", m["tail_wrong_samples"] == 0),
    ("naive readings both wrong", all(
        v != m["true_tail_at_decision"]
        for v in m["naive_tails_at_decision"].values())),
    ("counter after last service",
     m["counter_reached_s"] >= m["last_service_s"]),
]
for label, ok in checks:
    print(f"  [{' OK ' if ok else 'FAIL'}] {label}")
raise SystemExit(0 if all(ok for _, ok in checks) else 1)
PYEOF
[ $? -eq 0 ] || FAIL=1

echo "== 8. frozen behavior folders are untouched =="
cd "$LAB"
DIRTY=$(git status --porcelain -- move-away move-away-crowd follow-me \
    follow-me-among-others come-here-recall crosswalk-guardian \
    narrow-corridor-etiquette 2>/dev/null | grep -v '^?? ' | wc -l | tr -d ' ')
[ "$DIRTY" = "0" ] && pass "no tracked change in any frozen folder" \
    || fail "$DIRTY tracked change(s) in frozen folders"
DIFF=$(git diff --stat -- move-away move-away-crowd follow-me \
    follow-me-among-others come-here-recall crosswalk-guardian \
    narrow-corridor-etiquette | wc -l | tr -d ' ')
[ "$DIFF" = "0" ] && pass "git diff against frozen folders is empty" \
    || fail "git diff reports changes in frozen folders"
cd "$ROOT"

echo "== 9. cache cleanup =="
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null
rm -rf .pytest_cache tmp-preview 2>/dev/null
LEFT=$(find . \( -name '__pycache__' -o -name '*.pyc' -o -name '.pytest_cache' \) | wc -l | tr -d ' ')
[ "$LEFT" = "0" ] && pass "no caches left in the behavior folder" \
    || fail "$LEFT cache artifact(s) remain"

echo
[ "$FAIL" -eq 0 ] && echo "ALL VERIFICATION CHECKS PASSED" \
    || echo "VERIFICATION FAILED"
exit "$FAIL"
