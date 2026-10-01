# src/diagnostics — observing prosper's own startup

This folder holds the **observer-only** layer for prosper's boot: the seven `BootPhase`
transitions `boot_program()` records, the in-memory history of those events, and the JSON report
written from it. Nothing here participates in loading, linking, mapping or running the guest — if
code in this folder can change what the guest sees, it is in the wrong folder.

The split that matters:

- **`boot_phase_log.{hpp,cpp}` is reachable in the DEFAULT build.** One stderr line per phase,
  gated on the runtime environment variable `PROSPER_BOOTPHASE`. It exists because
  `boot_program()` ends by calling `run_guest_inits()` — real guest code — so a title can be
  inside `boot_program()` for minutes or forever, during which every frontend
  (`tools/screenshot`, `tools/boot_trace`, `prosper-app`) prints nothing and
  `screenshot --timeout` cannot fire, because that deadline lives in a sampling loop `boot_program()`
  has not returned to yet. The last `[bootphase]` line names the phase a stalled boot is inside.
- **`core/` and `storage/` are compiled out unless `PROSPER_DIAGNOSTICS` is defined** — that is what
  the source-glob filter in `CMakeLists.txt` excludes, and the filter is scoped to those two
  subfolders by name so a future file beside `boot_phase_log.cpp` is not silently dropped from the
  default build. They are the
  history, event bus and JSON writer. `diagnostics.hpp` is the single include and selects between
  the real types and stubs.

**The failure this folder already had once, so do not rebuild it:** three independent things each
made the phases unreachable, and any one of them was sufficient. `CMakeLists.txt` excluded the whole
of `src/diagnostics/` from `prosper_core` whenever `PROSPER_DIAGNOSTICS` was off, which is the
default; `DiagnosticContext::enable()` was never called from anywhere in the tree; and no subscriber
was ever attached to the bus. Seven instrumentation points existed and **no build the project
shipped could print any of them** — the instrument was unavailable in exactly the
situation it was written for, because a title that hangs is by definition one nobody had already
instrumented for. That is why the phase *line* is a runtime switch on a build-time-optional
subsystem: the asymmetry is deliberate, not an oversight.

So when adding anything here, ask which half it belongs to. A new *report* or aggregation is
`core/`/`storage/` work and may stay behind `PROSPER_DIAGNOSTICS`. A new signal that answers
"where did it stop?" must be reachable without a rebuild, or it will be missing on the day it
matters.

`diag_clock.hpp` is the second thing on the always-reachable side, and it is here rather than in a
subsystem because its whole purpose is to be shared. Diagnostics in different subsystems write to
one stderr stream from different threads, so their line order carries no ordering information —
comparing two of them by position in a log is unsound, and doing exactly that produced a wrong
published conclusion on #3142. `diag_now_us()` gives them a common monotonic reading; steady_clock
is process-wide, so independently taken values in unrelated translation units compare directly with
no anchor or registry. Anything that needs to be time-ordered against a diagnostic in another
subsystem should stamp it rather than grow a private clock.

`env_cache.hpp`, `env_numeric.hpp` and `env_submit.hpp` are the remaining always-reachable files, and they are here for
the same reason as `diag_clock.hpp`: it is shared, and it belongs to no one subsystem. It holds `PROSPER_ENV_ON` /
`PROSPER_ENV_VALUE`, the one-shot reads of a `PROSPER_*` switch. They lived in
`hle/dispatch/dispatch.hpp` until #3094, which is where they were first needed and not where they
belong — the renderer backend, the draw executor and the live frontend all evaluate diagnostic
gates on per-draw and per-resource paths, and none of them should pull in the HLE dispatch
registry to ask whether a switch is on. `dispatch.hpp` includes this header, so its existing users
did not change.

The thing to know before using them: **caching a diagnostic switch changes its semantics**, and it
fails in the quiet direction. The variable is sampled at first use, so a process that arms it later
never observes the write — and a test that arms a diagnostic and then asserts on the behaviour does
not go red when the read is cached, it goes **vacuous** and keeps printing `[ok]` (#2214).
`tools/env/check_cached_env.py` is the gate: it refuses any name something in the tree arms at
runtime, and separately pins the hot sites #3094 converted so they cannot regrow a live `getenv`.
Run it before caching a new name. Note it can only see arming through a *literal* name —
`tools/gpu_replay` re-applies `render_env[]` (`gpu/capture/gpu_capture.cpp`) per bundle submit
through a variable, which is invisible to it; that comment lives in `env_cache.hpp` itself.

`env_submit.hpp` is the third visibility contract, and it exists because the gate above leaves the
hottest names with nowhere to go. A name a test arms cannot be process-cached, so it stays a live
`getenv` — and several of those are read once per draw per stage. `PROSPER_ENV_ON_PER_SUBMIT`
re-samples at each `SubmitEnvScope` and reads **live outside every scope**, which is the part that
makes it safe to adopt one site at a time: a path with no scope behaves exactly as it did before, so
forgetting a scope costs speed and never correctness. Measured on a 240 s routed *Grand Theft Auto
V* window with diagnostics in their default state, converting four names took the process from
33.9 M `getenv` calls to 17.6 M (#3407).

Which of the three to reach for: **`PROSPER_ENV_ON`** when nothing arms the name at runtime —
`check_cached_env.py` will tell you. **`PROSPER_ENV_ON_PER_SUBMIT`** when something does and the
site is hot enough to matter; check the arming sites first and confirm each arms BETWEEN operations
rather than inside one, because that is the contract, not an assumption. **A live `getenv`**
otherwise. The gate refuses a name read two of those ways at once, since which answer you get would
then depend on which call site ran.

`env_numeric.hpp` answers the neighbouring question — not "was this set?" but "what NUMBER is that
text?" — and exists because the obvious spelling is quietly wrong. `strtoull(value, nullptr, 10)`
discards the end pointer, so the parse cannot fail; it just answers something, and measured on glibc
it answers **three** different wrong things: `0` for text it cannot start on, the leading digits for
`8mb`, and `UINT64_MAX` for `-1`. On a knob where 0 means "off" that is a lost experiment. On the
write-watch family it is worse, because 0 there is a *meaningful and maximally aggressive* setting, so
a typo silently ran a different experiment than the one asked for (#3253, after #3155's retracted
measurement). `env_u64_or_default*` refuses loudly and keeps the default; the accepted grammar is
exactly `[0-9]+`.

`env_tristate_or_unset` in the same header is the one non-numeric member, and it exists because an
A/B **lever** has a third answer that a number cannot express: `PROSPER_POST_SUBMIT_VISIBILITY` is
`1` forced on, `0` forced off, and UNSET meaning "follow the SDK version the guest asked for". Read
with `strtol` that collapsed -- `=on`, `=true`, `=yes` and `=enabled` all parse to 0, so every word
spelling ran the FORCED-OFF arm and the site announced it as deliberate (#3304). It accepts
`1`/`on`/`true`/`yes`/`enabled` and `0`/`off`/`false`/`no`/`disabled` in either case, plus `0x1` and
`0x0` because the site it was written for read base 0; anything else -- including a number that is
neither 0 nor 1 -- is UNSET with a line naming the value, never a silent third arm.

Both headers take the variable's NAME as a literal argument, and `check_cached_env.py` treats any
callee containing `env` as an environment WRITE — so a new reader here has to be added to that
script's `ENV_READERS` set or it reads as an arming. That coupling is the one non-obvious thing about
adding a function to this pair.

`exit_reports.{hpp,cpp}` is always-reachable for the same reason. It is the registry for **end-of-run
reports**: anything that summarises a run when it ends registers there instead of calling
`std::atexit` directly. Almost no run here returns from `main()`. `prosper-app`, `tools/screenshot`,
`boot_trace`'s host-exception path and the guest's own exit all terminate with `_Exit`/`_exit`, which
skip atexit handlers, and each of them calls `flush_exit_reports()` first. Before #3353 a bare atexit
report was silent on all of those paths, and the silence read as a zero. SIGTERM and signal-context
exits are still not covered, so a number that must survive a killed run needs a periodic report too.

`exit_census.{hpp,cpp}` sits on top of that registry and is what a new **census** should use:
`register_census(disable_env, report)` owns the end-of-run hook and the opt-out variable, and hands
formatting back to the caller because a per-reason distribution, a bucket series and a per-site
table are not the same report. The censuses themselves —
`transfer_pressure`, `readback_reason_census`, `worker_spawn_census`, `persistent_target_census` —
are separate files rather than one shared counter type on purpose; what was worth de-duplicating is
the part where a mistake is silent, not the counting.

**Register on first use, never at namespace-scope static initialisation**, and this is the one that
bites somebody else rather than you. `register_exit_report` installs its `std::atexit` fallback on
the *first* registration in the process, and `atexit` runs LIFO against `__cxa_atexit`'s static
destructors — so a census that registers during static init becomes the earliest atexit entry,
which makes the flush the **latest** thing to run, after the function-local statics that other
censuses read. Adding one file with a namespace-scope registration took this suite from green to
eleven failures: a SIGSEGV in `game_compute_exec` *after* its `[storage-materialize]` line printed,
plus three timeouts, none of them in the new census and none in code it touches. Pair it with the
other half: any object a report reads should be **never-destroyed** (the no-destroy idiom
`exit_reports.cpp` uses for its own registry), because no registration site controls that ordering.

The rule that decides whether a census belongs here at all: **it must be able to distinguish "this
did not happen" from "this was not counted".** Both of those print zero, and the second one reads
as a finding. `persistent_target_census` exists because a colour-target cache with no entry for a
key looks identical whether it evicted one or never made one, and the env-var A/B that would
separate them cannot show its own lever moved (instrument trap 288) — so it reports residency and
evictions directly instead, and says in one clause which of the two states it is in.

`perf/` is the always-on performance-alarm layer (#3891), and it answers a different question from
everything above: not "what happened?" but "what is this run paying for, and is that a problem?".
`perf_ledger.hpp` is the accumulation side and is **header-only on purpose** — render_runner.h is
compiled straight into dozens of Vulkan tests that name their own translation units, so a ledger
that needed a `.cpp` would need a CMake edit per test. Hooks feeding it must stay coarse: relaxed
counters per event, clock reads only around a whole readback, lock wait, present or pass group —
never per draw (PROSPER_RENDER_TIMING's per-draw clocks cost ~9% of the render thread, which is why
that census is opt-in and this one is not). `perf_alarm_rules.{hpp,cpp}` is a pure function from one
window of ledger deltas to the alarms it raises; every threshold is a named constant beside the
measurement that set it, and every rule has a hand-built positive and negative window in
`tests/diagnostics/test_perf_alarms.cpp`. `perf_alarms.{hpp,cpp}` is the engine: windowing per guest
flip (the frame budget is the guest's own SetFlipRate), rate-limited `[perf-alarm]` lines, JSONL and
the exit summary. A new rule belongs here only if it can name the next instrument to reach for in its
hint and was validated against a run where the cost really existed; a rule that fires on a healthy
run gets tuned or removed, because an alarm people learn to ignore is worse than none.

**Reading the alarms** — do this first on any run you did not expect to be slow or wrong. Every run
log carries them without a flag: `[perf-alarm] #<ordinal> rule=<name> ... value=<v> <unit>
threshold=<t> <detail> hint=<next instrument>`, rate-limited per rule (the ordinal, not the line
count, says how many windows fired), then `[perf-alarm] summary` lines at exit: windows fired, the
worst window, and for a correctness rule the per-reason breakdown summed over the run. A summary's
`NO DATA` list names rules that could not have fired, which is not the same as quiet. A run killed
with SIGTERM skips the summary, so `PROSPER_PERF_ALARM_LOG=<path>` (JSONL: every firing plus every
window's raw quantities, flushed as written) is the record to keep for anything scripted. Under
`prosper-app --fps` the HUD adds one `! alarm: <rules>` line while a rule is active in the latest
window.

**Correctness rules carry the SITE, not just a count.** `dropped-draws` names the drop site per
window (`reasons=render-array-reject/depth-unavailable:276`) from `perf::DropReason`: one code per
`built.reject(...)` site in the live renderer's resource builder, `contract-mismatch`,
`shader-recompile/{vertex,fragment,geometry}` from `realize_draw_item` (a stage with no SPIR-V: the
draw never reaches the pass loop, so nothing downstream could see it -- #3951; realize_draw_item's
other exits -- missing program, no-effect, indirect arguments, zero vertex count -- are still NOT
counted), and
`backend/*` mirroring `gpu::DrawDrop` (`draw_disposition.cpp` static_asserts the mirror).
`skipped-dispatches` does the same with `perf::DispatchSkip`, and its exclusions are code, not
prose: a deliberate backend decline (the `PROSPER_COMPUTE_SKIP_PROGRAM` selector calls
`note_deliberate_dispatch_decline()`, which `BackendDispatchOutcome` checks), the parent-walk
diagnostic, indirect-dependency skips after either in the same submit, and every dispatch of a
process with no compute backend are not counted — each pinned by an arm in `test_gpu_execute`.
`gpu-memory-off-device` breaks down by renderer allocation class (`classes=depth-target:40`),
names supplied by `gpu/diagnostics/memory_placement_log.hpp`; it fires on any GPU-only allocation
placed off device-local memory on a device that has some, and `oom-fallbacks=` in its detail says
whether that was the #3897 out-of-memory retry or a resource that allowed no device-local type.
On this project's APU it should never fire; `PROSPER_GPU_MEM_FORCE_OOM` exercises it.
`host-copy-pressure`'s breakdown is whole MiB per `[transfer-pressure]` category, not a count. A new drop site must pass a reason;
one that sets `complete = false` directly shows up as `unattributed`, which is the instrument naming
its own blind spot, not a finding about the title. Re-realizations that are not live execution (an
F9 capture) wrap themselves in `SuppressDispatchSkipCounting` (and, for draws, `SuppressDrawDropCounting`)
so that capturing a frame cannot raise the alarm being investigated. `host-copy-pressure` reads the `transfer_pressure` census's totals at
window close rather than adding hooks of its own — reuse an existing always-on counter that way
before adding a parallel one.

**The 2026-09-28 queue rules** (#3891) each hang off an event path that already existed, so none
costs anything on an accepted draw, reference or present: `unaccounted-draws` is the
`draw_disposition` census's own `UNACCOUNTED` blind spot, added once per pass; `unimplemented-hle-calls`
counts the dispatcher's `prosper_on_unimpl` and fires on the FIRST call of an unregistered NID after
the first flip (boot-time ones are before the engine's baseline); `diagnostic-path-active` is a
gauge the live renderer sets once, naming the switch that turned GPU-resident colour targets off;
`present-path-fallback` compares prosper-app's CPU-fallback presents against the GPU presents
`present-cpu-overhead` already counts, and its `declines=` names WHY the renderer did not publish
the front buffer (counted per final render span, so not a count of presents) (`no-render-target`, `not-gpu-resident`, `compute-scanout-stale`, ... -- the
`GpuPresentOutcome` names in `frontends/shared/present/present_blit_policy.hpp`, #3915); `pipeline-cache-thrash` counts evictions from the pipeline,
pipeline-layout and descriptor-set-layout caches; `texture-validation-churn` counts route-specific
prefix bytes charged to failed exact decode-cache validations. Direct comparisons count fully
matching chunks and omit the first differing chunk (at most 64 KiB); a first-chunk mismatch can
therefore report zero. Depth comparisons accumulate matching guest-face prefixes, while the
scratch-copy route counts copied readable bytes. An incomplete prefix can also fail validation,
so the counter proves neither source mutation nor total comparison traffic. Late differences can
still charge most of the source before a re-decode. The separate direct-validation observer below
keeps this counter's original meaning. Proposals needing a new expensive signal were declined on
the issue with a reason rather than approximated.

**Direct texture-validation observations** (#3891) cover only the retained-pixel and complete
encoded-prefix branches of the live image cache's exact validation. Each actual helper invocation
records an attempt, its number of `memcmp` calls, and the sum of their argument extents, including
the first differing chunk. Extents do not measure physical reads or bandwidth. The original helper
result and caller's required-prefix check classify accepted-prefix, bytes-differ, expected-missing
or incomplete-prefix outcomes. A zero-readable prefix never claims the unreached null-expected
refusal; short equal prefixes still obey the caller's original length requirement. Depth-source
validation, scratch copying, watch-only refusal and watch/journal shortcuts are outside this scope.

The seven `texture_direct_validation_*` JSONL quantities and the exit observer summary cover
closed windows after the first-flip baseline, excluding boot and the trailing partial window.
No attempts and no other quantities means NO DATA; positive work/outcomes without an attempt
means PARTIAL; an observed attempt can have zero comparison work. Independent relaxed snapshots
do not form a coherent outcome partition or certify guest quiescence. This adds no alarm threshold
and leaves `texture-validation-churn`, F8 populations and all cache/watch admission policy intact.

**Surface readback separates attempt volume from duration** (#3891, #3948). Its detail appends
`attempts/flip` beside the existing mean time per attempt. The outermost readback scope includes
early returns, so these are attempts rather than completed physical copies. Compare both fields
across matching routes: a higher mean with unchanged attempt volume also warrants checking host
CPU/memory contention. The measured GTA V host-load control reproduced the low-frame-rate state
with deferred waits off. The hint suggests an investigation; it does not attribute a window's
cost to host contention. The rule reuses existing quantities and keeps its thresholds.

**Host copy has two denominators** (#3891, 2026-09-29). `host-copy-per-flip` divides the
`transfer_pressure` bytes by guest flips and leads; `host-copy-pressure` (per second) reports only
what the per-flip form cannot see -- a window with fewer flips than the per-flip floor (a stall, a
load), or a small per-frame copy at a high frame rate -- and is not reported in a window where the
per-flip rule is, so one cause raises one line. That deferral is applied at REPORT time
(`apply_reporting_deferrals`, after the engine's sustain streaks), never by withholding the
candidate: a rule with no candidate has its streak reset, and a per-flip value alternating across
its threshold then reset both rules' streaks so that neither ever printed (the review of #3928). The per-second form is the one that moves with the frame rate (an A/B that changed only
the fps flipped its verdict), which is why it does not lead; the JSONL window keeps both figures
(`host_copy_mib_per_flip`, `host_copy_mib_per_s`) and `host_copy_calls_by_site`, and the per-flip
line names each top site's MiB per call, because "bigger copies" and "more copies" are different
defects. `guest-scanout` is its own `Transfer` site: `videoout_read_front_linear` charges both its
read out of guest memory and its de-swizzle there through a `TransferAttributionScope` (the
de-swizzle goes through `detile_surface`, which would otherwise report it as `detile`).
`present-slot-trouble` counts only the two GPU-present declines that mean the scanout path itself
failed (`publish-failed`, `compute-scanout-unwatched`), per guest flip -- the ledger has no count of
spans that published, so a flip is the denominator; `test_present_blit_policy` pins the names it
matches.

**Compute results that miss the renderer image** (#3891, 2026-09-29). `rtt-destination-refused`
counts `live_compute`'s one destination-borrow refusal site (the same site the exit-only
`[rtt-destination-refused]` census notes) per guest flip, with MiB by refusal reason: GTA V's heavy
pre-#3949 host-copy regime was one 14 MiB result refused `destination-creation-refused` per flip, and this is
the rule that names it live. `color-target-count-ceiling` reads the persistent colour-target
cache's per-window PEAKS (`Peak`, raised where `persistent_target_census` already samples residency,
reset at every window close) against its two bounds: at the entry-count bound, under half the byte
budget, and either evicting or refusing a compute creation. The refusal arm is not optional: GTA V's
pre-fix costly state sat at exactly 256 of 256 entries with zero evictions. Compute creation now
evicts eligible idle targets (#3949), and a count-bound cache can churn without refusing it.
`destination-creation-refused` also covers admission checks such as an unproven submission or
partial/pinned target state. That reason alone does not prove the cache had no room; the combined
count/byte/churn-or-refusal observation suggests count pressure without attributing every refusal
to that bound. Eviction counts do not prove a physical readback either: valid targets need a sink
and successful readback to publish CPU pixels; invalid targets or targets without a sink also count.
When the rule fires on the refusal arm it
reports `creation-refusals`, not a 0.00 evictions/s. The refusal it counts is matched by name
(`kDestinationCreationRefused`, which `live_compute.cpp` static_asserts against
`live_target_import_refusal_name`). A Vulkan failure to create the image is a separate reason,
`destination-allocation-failed`, and does not count toward the ceiling. The exit summary also prints a census line (not a rule): each tested compute result's
FIRST failing field of the exact full-overwrite shape test (`ExactResultDecline`, in
`exact_full_result`'s order), then the post-shape format declines, then `accepted`; the JSONL window
carries it as `exact_result_verdicts`. `exact_full_result` is defined as "the verdict is accepted",
so a new condition in the shape test is added to `exact_result_verdict` with a verdict of its own.

**Synchronous GPU waits** (#3948 stage 0). `Cost::GpuWaitCompute` / `GpuWaitGraphics` are one clock
pair around each executor fence wait (a compute dispatch's, a graphics submission batch's), and
`Counter::GpuDevice*` carry the GPU time inside them from one timestamp pair per dispatch or batch,
armed whenever the ledger is (per dispatch/batch, never per draw; the graphics envelope's query
pools are reused through a free list, not created per batch). `gpu-sync-wait` fires when the waits
exceed 75% of the frame budget per flip while prosper's own GPU work is under half of wall time:
the cost asynchronous submission would recover. `gpu-busy` counts only the work prosper submitted,
so a GPU saturated by something else reads as headroom, and wait minus device includes time queued
behind other submissions on the shared queue (present blits), not only submission latency.
**It deliberately fires on every GTA V and Sonic Frontiers gameplay window**: it is a standing
signal of the synchronous architecture, not a defect of those runs, and the general rule that a
rule firing on a healthy run gets tuned or removed does not apply to it until #3948's later stages
land. Retune or retire it then; do not "fix" it before.

`gpu-device-time-coverage` checks timestamp samples against fence waits separately for compute
and graphics. Fewer than 80% sampled over at least 100 waits in either stream, sustained over
two windows, means the GPU-busy estimate is incomplete; `gpu-sync-wait` then has **NO DATA**.
Complete compute samples cannot hide missing graphics envelopes. Corrected #3964 gameplay
windows cover about 89% of graphics waits; the broken deferred-envelope arms cover about
68-71%. This uses the existing counters at window close and adds no event hook. Check timestamp
support, query results and envelope closure before drawing performance conclusions from it.

**Fresh direct-fold WaitRegMem observations** (#3891) retain the predicate's existing raw and
pending-overlay value before the warning's later raw reread. The decision sample carries value
validity, overlay-touched status and comparison support; invalid sample values are placeholders,
not measured zeroes. Its mask applies only to the effective value, not the reference. The six
`wait_regmem_direct_*` JSONL quantities count only actual direct evaluations and their false
reason/action. Ordered retained-effect evaluations, waits queued without evaluation, invalid
packets and deferred rechecks are outside that denominator. Warning suppression does not suppress
counting. The q label is the submit entry, not a physical Vulkan queue identity.

This is a non-rule observer without a chosen threshold or HUD alarm. The summary covers closed
windows after the first-flip baseline, excluding boot and the trailing partial window. Zero
evaluations and zero reason/action samples means NO DATA; positive reason/action counts with no
evaluation in a relaxed snapshot means PARTIAL. Observed direct evaluations without false samples
describe quiet only in this scope. Separate relaxed counter loads do not form a coherent partition
or prove guest quiescence. Existing no-engine/no-window reporting behavior is preserved. Neither
a false predicate nor a later matching printed value identifies writer causality, a hardware
timeout or an ordering regression. Real packet-route calibration is required separately from the
native sampler/ledger/window controls.

**Colourless CPU pass publication** (#3891, #3907). `rtt-colorless-publication` counts an
actual slot-0 CPU pass-readback publication without any colour-write mask. A single violation
reports immediately, independent of sensitivity. The candidate count covers guarded decisions,
including depth-only candidates that correctly publish nothing; no candidates or violations is
**NO DATA**, whereas observed candidates with zero violations are quiet. The predicate accepts
both mask representations exactly as the existing guard does, and is evaluated only when that
CPU fallback guard would inspect masks. The hook changes no publication policy and adds no guest
reads, clocks or entry state. Compute snapshots, resolve copies and GPU materialization are
outside this site's contract. Extent reversals alone were falsified by actual GTA controls:
restoring the old guard produced540 reversals, but the fixed guard still produced310 in matching
300-second CPU-readback routes, with32/55 versus30/55 windows firing. The generic alias signal
was withdrawn rather than assigning a threshold that did not separate the two populations.

**A window nobody saw** (#3891, #3951). `gpu-present-stalled` fires when a GPU-present frontend
(`Gauge::GpuPresentActive`, set by `set_gpu_present_active`) presented nothing in a window, neither
a GPU scanout (`Cost::PresentCpu` events) nor a CPU fallback, while the guest flipped at least 20
times. The window then keeps showing its last frame (or black), whatever the other rules say.
#3951's recompiler regression is the case: the guest flipped at ~30/s with no `[app] fps` line, and
only a downstream host-copy-per-flip fired. It has no data in `tools/screenshot`/`boot_trace` (no
consumer), in a window where the guest barely flips (boot, loading), or in a window the window
system made unpresentable (`Counter::PresentWindowUnavailable`: a zero-extent surface, or SDL
reporting the window minimized, hidden or occluded -- the gauge stays set across those, so without
this it read a hidden window as a stall), and it needs two windows. A skipped or out-of-date present
on a VISIBLE window deliberately does not count: that is a broken present path, which is exactly
what the rule is for.
