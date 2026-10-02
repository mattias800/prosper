---
name: perf-change
description: Make, measure and land a prosper performance change - a default-off switch, a same-binary A/B on reference workloads, and a recorded verdict either way.
---

# Make a performance change

Preserve task authorization, resource coordination and `LOCAL.md`; this skill grants no additional
authority. It does not authorize GPU runs, claiming a GPU lease, pushing or merging — follow the
task and `prosper/docs/GAME_COMPAT_ORCHESTRATION.md` for those.

1. **Read first.** `CLAUDE.md` § *Architecture and performance ratchets* (invariants P1-P6) and its
   FPS-harness warnings; `prosper/src/diagnostics/AGENTS.md` for the `[perf-alarm]` lines; the
   `## Ruled out` section of the status doc for each workload below. Name the invariant the change
   moves toward, or say that it moves none.
2. **Make the change switchable, default off.** One `PROSPER_*` switch selects the new path, so the
   same binary runs both arms and the control arm is the change disabled. Classify it in the PR (a
   selector needs an issue that will delete it). Cache-off controls must leave results identical.
3. **Reference workloads** — one change is measured on all three, not on the title it was written
   for:
   - *Grand Theft Auto V* `PPSA04263`, `prosper/scripts/gta5/reach-performance-story.pad` (the world
     renders only in the game's Performance mode; `prosper/docs/GTA5_STATUS.md`).
   - A second 3D title, e.g. *Outer Wilds* `PPSA08102`,
     `prosper/scripts/outer-wilds-PPSA08102/reach-first-person.pad`
     (`prosper/docs/OUTER_WILDS_STATUS.md`).
   - *The Messenger* `PPSA24651`, `prosper/scripts/messenger/reach-first-level.pad`, as the flat 2D
     control a change should not regress.
4. **Measure the shipped path.** `prosper-app` (built with `-DPROSPER_APP=ON`) with GPU present —
   the log must say `[app] GPU present: adopted the renderer's device`, since an adoption failure
   falls back to a private device and CPU pixels and the run still looks normal — plus `--present-mode immediate` and `--fps`; quote `distinct`, not `presented`, and drop the samples next
   to a capture. A `tools/screenshot` rate is the forced-readback path. Comparing small changes
   needs a peer-clean GPU: ask for an exclusive lease (`GAME_COMPAT_ORCHESTRATION.md`
   § *GPU scheduling*).
5. **Read before optimising.** The `[perf-alarm] summary` lines at exit first. Then an F8 `.prperf`
   (`PROSPER_PERF_CAPTURE_AFTER_MS` or `PROSPER_PERF_CAPTURE_AT_FRAME`, written under
   `PROSPER_CAPTURE_DIR`) read with `prosper/tools/perf/performance_capture_report.py` — and check
   that the capture can see the leaf you are about to change.
6. **Compare honestly.** `prosper/tools/perf/compare_runs.py` refuses mismatched runs but reads only
   `tools/screenshot` manifests; `prosper-app` writes none. For a `prosper-app` A/B, state the
   frontend, present mode, route, build, both arms and the sample counts by hand, with each arm run
   more than once so run-to-run noise is visible.
7. **Record the verdict either way.** A win lands with its A/B in the PR. A loss or a null is a
   result: add a line to the relevant `## Ruled out` (the title's status doc, or the area doc for a
   cross-title one) with the evidence and the PR or issue link, and delete the switch if the change is abandoned.
8. **Run the ratchet in delta mode** —
   `python3 prosper/tools/ci/check_arch_ratchet.py --root . --base origin/main` — and fix any count
   your change raised, or justify a raised baseline row in the same PR. The commit hook runs the
   same check but only warns, so do not read a commit that went through as a pass.
