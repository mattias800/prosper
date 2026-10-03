# Overlapping graphics buffer uploads (#3407)

The renderer now copies connected overlapping direct guest storage-buffer inputs once per
render pass, then binds each original descriptor's exact slice. In one same-binary pair,
Blue Prince's host presentation rate rose from **11.34 to 14.74/s**. GTA remained at **6.17/s**;
Sonic measured **7.77 to 8.15/s** with its existing black-world rendering defect. These are
observed presentation rates, not newly rendered FPS or a repeatable library-wide speedup.

## Contract

Admission is generic: explicit direct guest backing, at least 4 KiB, complete backing identity,
and a complete negative storage-buffer-write proof over the final VS/FS/GS for the whole pass.
A later writer, atomic operation or unresolved write provenance rejects sharing for that pass.
Hosted copies, texture/GDS/table data and incomplete direct backing retain ordinary uploads.

The Vulkan-free planner deduplicates exact spans and merges overlapping spans with compatible
address residues for the device's storage-buffer offset alignment. It does not bridge gaps.
Each union is bounded by the device's maximum storage-buffer range and 64 MiB, and must save
at least 4 KiB against distinct inputs. Upload occurs lazily on the first resolved binding.
The existing submission arena owns the union; child descriptors retain their original length
and an aligned offset into that arena. No guest pointer or new cache owner survives the call.
Allocation refusal falls back to ordinary uploads. Queued passes retain separate snapshots.

`PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE=1` disables the new path. Existing resource-sharing,
buffer-pool, arena and buffer-verification controls also veto it. Generic cross-call renderer
buffer residency remains disabled by default. There are no title/address/shader-specific rules,
new optional Vulkan features or experimental driver flags. NVIDIA/Windows runtime testing was
not performed on this Linux/RADV machine.

F8 records actual union uploads, resolved unique descriptor slices after memoization, their
respective byte counts and `res_buffer_range_plan_ms`. They are not GPU-completion counters.
Planning includes the shared write proof, is included in backend setup, and sits outside the
per-binding buffer timer. The reporter preserves unavailable older fields and signed
`bound_bytes - upload_bytes`: unused planned members can make that difference negative.

## Same-binary comparisons

All six accepted runs use source `68b8a96f5d8292f457541bda8f07aed6ea305dfe`, based on main
`267406bcda792ea0b2966a10d6c131a26a926f54`, and executable SHA-256
`6d607cc7cafc97d8da0d3348af4503f5146b676181bca19565367aeddaa4bae2`.
The subsequent merge of main `c26e8e44aa62a953057cd0c60bc64f57c420a9df` adds fiber rename and
its test; it does not change the measured renderer. Captures precede that integration.

Normal windowed prosper-app, AMD Radeon 8060S/RADV, device Vulkan 1.4.354, immediate presentation,
native resolution and full render cadence. Each arm starts with fresh save, application cache
and driver caches. The control sets only `PROSPER_NO_BACKEND_BUFFER_RANGE_SHARE=1`; all other
effective settings, source, binary, route and capture-tool hashes agree within each pair.
Blue Prince runs control then enabled, GTA enabled then control, Sonic control then enabled.
There is one accepted pair per title, with no statistical confidence interval.

CPU (`perf record -F 199 -g`) and GPU (`radeontop -t 10`) sampling start at 250 seconds for
20 seconds; both instruments finish before the F8 window at 300 seconds. F9 follows at
310 seconds for GTA and 330 seconds for Blue Prince/Sonic. Process censuses reject competing
builds, games and profilers. Two earlier Blue Prince attempts aborted on competing builds
before F8 and are excluded. Source and executable identity, capture completeness, profiler
completion and process isolation are checked independently at the end of each accepted run.

All table arrows are **control → enabled**. Bytes use decimal GB. Times are totals over the
approximately five-second F8 window unless explicitly normalized.

| Measurement | Blue Prince | GTA Performance Story | Sonic Frontiers |
|---|---:|---:|---:|
| F8 seconds | 5.025 → 5.021 | 5.021 → 5.021 | 5.017 → 5.029 |
| Host presentations/s | 11.343 → 14.739 | 6.174 → 6.174 | 7.774 → 8.152 |
| Renderer records | 58 → 76 | 775 → 816 | 638 → 668 |
| Actual buffer upload GB | 52.876 → 7.504 | 7.072 → 1.781 | 10.962 → 3.937 |
| Buffer copy ms | 1112.934 → 259.243 | 153.472 → 61.427 | 224.106 → 92.041 |
| Planning ms | 0.080 → 53.013 | 0.132 → 17.467 | 0.117 → 11.037 |
| Copy + planning ms/renderer record | 19.190 → 4.109 | 0.1982 → 0.0967 | 0.3514 → 0.1543 |
| Backend setup ms/renderer record | 27.776 → 10.908 | 0.6163 → 0.5177 | 0.5598 → 0.3467 |
| Frontend + backend resources ms | 2293.173 → 1698.661 | 1354.745 → 1367.156 | 770.072 → 659.944 |
| Graphics GPU wait ms | 213.299 → 259.271 | 508.873 → 516.200 | 656.072 → 650.696 |
| Graphics device bracket ms | 97.675 → 115.755 | 322.641 → 322.554 | 537.648 → 530.925 |
| Compute total ms | 16.401 → 21.153 | 1200.892 → 1271.111 | 1568.123 → 1587.685 |
| Process CPU cores | 2.035 → 2.289 | 1.431 → 1.439 | 1.252 → 1.264 |

Copy plus planning falls **78.6%, 51.2% and 56.1% per renderer record**, respectively. Record
populations differ and renderer records are not frames; normalization is a workload indicator,
not identical-work or critical-path proof. Copy/planning are already in backend setup, and
resource/compute/wait/device scopes can nest or overlap: do not sum columns into a frame budget.
Guest flip rates equal host presentation rates here; newly rendered FPS is unavailable.

In the separate CPU profiles, the weighted memcpy/memmove leaf share falls 15.62%→4.67% for
Blue Prince, 8.28%→6.80% for GTA and 12.70%→9.43% for Sonic. These are event-period weights,
not wall milliseconds. Blue Prince consumes more process CPU while completing more work.
Mean sampled GPU load is 8.5%→9.5%, 13.0%→16.5% and 13.5%→17.5%, respectively. Radeontop's
coarse 10 Hz samples are separate from F8, and its unsupported per-engine columns are unused;
these readings do not establish a GPU saturation ceiling or explain a critical path by themselves.

## Correctness and audio observations

Both Blue Prince F9 images show the coherent furnished entrance hall, including lighting,
chandelier, table and busts. Both GTA images show the lit bank, characters, glass/counters,
world and radar; actor poses differ. Sonic retains the known black world behind the white HUD
(#2790) in both arms. These are paired visual inspections, not pixel identity or full-movie
equivalence. No title/menu defect is claimed fixed, and no snapshot baseline was changed.

The active main output's cumulative callback shortfalls stay flat across conservative
approximately 6–8-second brackets encompassing each F8 window: Blue Prince output 1/generation 1
stays 0→0 in control and 1→1 enabled; GTA output 17/1 stays 1→1 and 2→2; Sonic output 17/1
stays 3→3 and 1→1. All six bracket deltas are zero. This is no observed increase in those
brackets, not evidence of improved audio, complete output coverage or absent hardware XRUNs.

The analysis brackets each SDL callback timestamp with strictly earlier/later SDL lifecycle
anchors and uses their steady-clock before/after bounds. It does not extrapolate a constant
offset: SDL's Linux CLOCK_MONOTONIC_RAW and steady_clock need not run at the same rate.
Unbracketed outputs remain unavailable. Independent buffers/channels and audio production
are unchanged; startup and post-F8 shortfalls are outside this limited claim.

## Validation and reproduction

At the measured source, all **523 CTests**, **46 Python reporter tests** and **seven focused
Vulkan tests under strict synchronization validation** pass. The strict scan uses an empty
allowlist and observes zero messages; positive controls prove layer loading and a deliberate
write-after-write hazard proves synchronization checking is armed.

Guards cover planner bounds/residues/gaps/caps, actual shifted descriptor reads, nonzero arena
base, original descriptor bounds, hosted refusal, later fragment atomics and incomplete write
proof, diagnostic controls, changed guest contents queued before completion, actual live F8
split-span ownership and exact large counters. The enabled Vulkan arm asserts reduced actual
upload bytes, so forcing sharing off makes that arm fail while preserving its visual checks.
No forced union-allocation failure or actual geometry-shader writer fixture was added; those
fallback/stage paths are reviewed but not exhaustively executed by the new tests.

From the checkout root, with the normal Linux/container dependencies and game root configured:

```sh
cmake -S prosper -B prosper/build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DPROSPER_APP=ON
cmake --build prosper/build-linux -j6
ctest --test-dir prosper/build-linux --no-tests=error --output-on-failure
python3 prosper/tools/perf/test_performance_capture_report.py
: > prosper/build-linux/empty-vkval-allowlist.txt
python3 prosper/tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux \
  --allowlist prosper/build-linux/empty-vkval-allowlist.txt --sync \
  --ctest-arg=-R --ctest-arg='^(render_buffer_range_.*|render_buffer_residency|render_buffer_capture)$'
```

The retained private archive `issue-3407-buffer-range-sharing-20260913` contains original
F8/F9, direct images, logs, profiles, run manifests and the three capture scripts. Restoring
`range-capture.py`, `snapshot-child.py` and `snapshot-validation.py` under
`prosper/build-linux/evidence` reproduces the guarded launches below after configuring the
local dump root in the runner. Each launch requires exclusive measurement access:

```sh
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/range-capture.py disabled blue-prince control
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/range-capture.py enabled blue-prince enabled
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/range-capture.py enabled gta enabled
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/range-capture.py disabled gta control
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/range-capture.py disabled sonic control
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/range-capture.py enabled sonic enabled
```

GTA uses `scripts/gta5/reach-performance-story.pad`; Sonic uses
`scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`; Blue Prince uses IME autokey. The runner
enables render timing, compute phase/buffer/image timing and per-output audio queue/demand/flow/
lifecycle/WAV diagnostics in both arms. The [structured results](RENDERER_BUFFER_RANGE_SHARING_2026_09.json)
preserve source/tool hashes, exact counters, nested CPU/GPU scopes and conservative audio bounds.

## Ruled out and remaining work

An offline census found no overlap between captured graphics **shader storage-buffer input
descriptors** and compute-written storage-buffer spans within the same captured frame in these
three titles, even ignoring order. This rules out a useful same-frame bridge for that observed
population. It did not census index buffers, indirect arguments or previous-frame producers;
it does not reject compute-to-graphics ownership sharing in general.

#3407 remains open. Large source copies, retained texture refresh/conversion, graphics result
writeback and synchronous compute/writeback still need ownership work. GTA's larger compute
and frontend resource scopes are consistent with its unchanged presentation throughput; these
measurements do not establish critical-path causality. Next, census index/indirect/prior-frame producer populations and stable
union reuse before choosing another cache. Union sharing reduces the number of distinct copied spans substantially
in Blue Prince; this is not a measurement of allocation owners. Any cross-call retention must
still demonstrate reuse, content invalidation,
safe pinning and lower enclosing CPU cost. Do not re-enable the previously rejected generic
per-binding residency policy without that new evidence.
