#!/bin/bash
# A/B one PROSPER_* switch against a routed live run and report the compute-call cost.
#
# WHY THIS EXISTS: a timing number carries no evidence of the conditions it was taken under. This
# machine is shared by several agents and a human, so a benchmark can silently run against a busy
# GPU and produce a number nobody can attribute. ctest has an exit code; performance does not --
# so this harness REFUSES to run when another prosper-app is alive, and records the conditions
# alongside the result.
#
# Usage: ab_compute.sh <switch=value> <route.pad> <dump-dir> [reps] [seconds]
#   e.g. ab_compute.sh PROSPER_NO_DIRECT_RTT_BIND=1 scripts/blasphemous2/title-screen-idle.pad \
#          <REPO_ROOT>/PPSA13579-app0 3 60
# The switch names the OFF (baseline) condition; ON is the same run with the switch unset.
set -u
SWITCH="${1:?switch=value naming the OFF condition}"
ROUTE="${2:?route .pad file}"
DUMP="${3:?game dump dir}"
REPS="${4:-3}"
SECS="${5:-60}"
cd "$(dirname "$0")/../.."
# Resolve AFTER the cd so a relative AB_OUT_DIR is created where it is later written.
OUT="${AB_OUT_DIR:-$(mktemp -d)}"
mkdir -p "$OUT" || { echo "cannot create output dir: $OUT" >&2; exit 2; }

# --- refuse to measure against a contended GPU -------------------------------------------------
# Match the EXECUTABLE, not any command line containing the string -- a -f match also catches the
# wrapper shells that invoked this script, which would make the guard fire against itself.
others=$(pgrep -x prosper-app | wc -l)
if [ "$others" -gt 0 ]; then
  echo "REFUSING TO MEASURE: $others prosper-app process(es) already running." >&2
  echo "Another agent or an interactive session is using the GPU; timings would be unattributable." >&2
  pgrep -ax prosper-app >&2
  exit 2
fi
if [ ! -d "$DUMP" ]; then echo "dump not found: $DUMP" >&2; exit 2; fi
if [ ! -f "$ROUTE" ]; then echo "route not found: $ROUTE" >&2; exit 2; fi

echo "# ab_compute  commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
echo "# route=$ROUTE  reps=$REPS  seconds=$SECS  off_switch=$SWITCH"
echo "# started=$(date -u +%Y-%m-%dT%H:%M:%SZ)  logs=$OUT"

run() { # $1=label  $2=extra env ("" for ON)
  env SDL_AUDIO_DRIVER=dummy PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
      PROSPER_RENDER=1 PROSPER_VULKAN_LIB=libvulkan.so.1 PROSPER_RENDER_TIMING=1 $2 \
      PROSPER_PAD_SCRIPT="@$ROUTE" \
      timeout "$SECS" ./build-linux/prosper-app --dump "$DUMP" > "$OUT/$1.log" 2>&1
  # run-wide mean; for a long route prefer the rolling-window tail (see --tail below)
  grep '\[render-timing\] compute calls=' "$OUT/$1.log" | tail -1 \
    | sed -E 's/.*avg_ms=([0-9.]+).*/\1/'
}
tail_window() { # mean of the last N rolling windows = the route's tail (skips menus/loading)
  grep '\[render-window\] compute' "$OUT/$1.log" | tail -"${AB_TAIL_WINDOWS:-20}" \
    | sed -E 's/.*avg_ms=([0-9.]+).*/\1/' | awk '{s+=$1;n++} END {if(n) printf "%.2f", s/n}'
}
# FRAME RATE is the metric a user actually feels, and it is NOT interchangeable with the compute
# cost above: a large win on one stage converts to a much smaller win on the frame if that stage is
# a minority of frame time. Always report both, or a change looks better than it is.
tail_fps() {
  grep -oE '^\[app\] [0-9.]+ fps' "$OUT/$1.log" | grep -oE '[0-9.]+' \
    | tail -"${AB_TAIL_WINDOWS:-20}" | awk '{s+=$1;n++} END {if(n) printf "%.1f", s/n}'
}

bad=0
valid=0
for rep in $(seq 1 "$REPS"); do
  # Re-check contention every rep: a competing process can start mid-sweep, and a startup-only
  # check would let it silently skew the remaining reps.
  if [ "$(pgrep -x prosper-app | wc -l)" -gt 0 ]; then
    echo "ABORTING at rep$rep: another prosper-app started mid-sweep" >&2; exit 2
  fi
  # ALTERNATE the arm order by rep parity. Running one arm consistently first hands the second arm
  # every warm-up benefit there is -- on-disk pipeline caches, GPU clock ramp, page cache -- which
  # systematically favours it. With a fixed order that bias is indistinguishable from the effect
  # being measured, and it flatters whichever arm is scheduled second.
  if [ $((rep % 2)) -eq 1 ]; then
    off=$(run "off_$rep" "$SWITCH"); on=$(run "on_$rep" "")
  else
    on=$(run "on_$rep" ""); off=$(run "off_$rep" "$SWITCH")
  fi
  offt=$(tail_window "off_$rep");  offf=$(tail_fps "off_$rep")
  ont=$(tail_window "on_$rep");    onf=$(tail_fps "on_$rep")
  order=$([ $((rep % 2)) -eq 1 ] && echo "OFF-first" || echo "ON-first")
  echo "rep$rep ($order)  compute ms/call  OFF run=${off:-n/a} tail=${offt:-n/a}   ON run=${on:-n/a} tail=${ont:-n/a}"
  echo "rep$rep ($order)  frame rate       OFF tail=${offf:-n/a} fps            ON tail=${onf:-n/a} fps"
  # A blind timed route can desync and leave a run stuck in menus (cheap frames, ~180 fps, almost
  # no compute) or produce no timing at all. Neither is a valid sample of the gameplay workload.
  # Require a floor of compute calls in BOTH arms; a rep failing it is DISCARDED (not counted) and
  # the sweep continues -- one desync must not fail an otherwise-good sweep on a flaky route.
  floor="${AB_MIN_COMPUTE_CALLS:-2000}"
  rep_ok=1
  for arm in "off_$rep" "on_$rep"; do
    cc=$(grep -oE '\[render-timing\] compute calls=[0-9]+' "$OUT/$arm.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    if [ -z "$cc" ] || [ "$cc" -lt "$floor" ]; then
      echo "rep$rep  DISCARDED: $arm reached only ${cc:-0} compute calls (< $floor) -- route desynced, not gameplay" >&2
      rep_ok=0
    fi
  done
  [ -n "$off" ] && [ -n "$on" ] || rep_ok=0
  [ "$rep_ok" -eq 1 ] && valid=$((valid+1))
done
echo "# finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)  valid_reps=$valid/$REPS"
need="${AB_MIN_VALID_REPS:-3}"
if [ "$valid" -lt "$need" ]; then
  echo "MEASUREMENT FAILED: only $valid/$REPS reps reached the gameplay workload (need $need); see $OUT" >&2
  exit 1
fi
