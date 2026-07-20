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
#          /home/mattias800/repos/ps5ys/PPSA13579-app0 3 60
# The switch names the OFF (baseline) condition; ON is the same run with the switch unset.
set -u
SWITCH="${1:?switch=value naming the OFF condition}"
ROUTE="${2:?route .pad file}"
DUMP="${3:?game dump dir}"
REPS="${4:-3}"
SECS="${5:-60}"
OUT="${AB_OUT_DIR:-$(mktemp -d)}"
cd "$(dirname "$0")/../.."

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

for rep in $(seq 1 "$REPS"); do
  off=$(run "off_$rep" "$SWITCH");  offt=$(tail_window "off_$rep")
  on=$(run  "on_$rep"  "");         ont=$(tail_window "on_$rep")
  echo "rep$rep  OFF run=${off:-n/a} tail=${offt:-n/a}   ON run=${on:-n/a} tail=${ont:-n/a}  (ms/compute call)"
done
echo "# finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
