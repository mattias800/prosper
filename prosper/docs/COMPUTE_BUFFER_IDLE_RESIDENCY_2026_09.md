# Compute buffer residency: reclaim idle owners for exact comparisons

Continuation of #3407, based on main including #3621. This change keeps the existing 256 MiB
requested-residency budget and guest writeback contract. It admits useful exact comparison
baselines by retiring sufficiently old, unpinned cache owners. It also provides a reusable cache
census so later admission decisions can start from the whole population.

## What occupied the cache

The earlier timing rows described only owners bound by observed dispatches. Those five active keys
charged 223.125 MiB, while the complete cache charged 255 MiB. Lowering the comparison threshold
alone admitted nothing: every eligible baseline was refused by the budget.

A fresh normal-windowed Sonic run at source `14386e1d1`, with the existing 8 MiB threshold override,
recorded 1,110 complete cache snapshots. They contained exactly six keys and summed to 255 MiB:

| Population | Primary charge | Observed use |
|---|---:|---|
| Three large owners | 63.75 MiB each | Repeatedly acquired; maximum access age 4 |
| Two medium owners | 15.9375 MiB each | Repeatedly acquired; maximum access age 4 |
| One other owner | 31.875 MiB | No pins or last-use changes; access age 4,329–4,526 |

The cache clock advanced from 16,406 to 16,603; the unused owner's last-use remained 12,077.
These are insertion/acquisition counts, including failed validations, not seconds or proof of
future deadness. All six owners had no comparison baseline. The census copies only metadata,
before cleanup releases pins, and reports complete keys, separate primary/baseline charges,
content-valid state and completion status. Stored validity is not a fresh content proof.

Enable `PROSPER_COMPUTE_BUFFER_TIMING=1` and `PROSPER_COMPUTE_BUFFER_CACHE_CENSUS=1`, using the
existing program/F8 selectors. Analyze the runtime log with:

```sh
python3 prosper/tools/perf/compute_buffer_cache_report.py <RUN>/run.log > <RUN>/cache.json
```

The reader refuses incomplete snapshots, missing/duplicate owners, broken charge totals, excess
budget, backwards clocks and invalid flags. It keeps complete materialization keys, including
same-address entries with different semantics. It reports observed access changes, not hit counts.
Census collection itself adds work inside cleanup timing; it is disabled in the performance pairs.

## Admission and ownership

Optional result baselines may use spare capacity or reclaim owners unused for at least 256 primary
insertions/acquisitions. Pins and recent accesses exclude an owner from both the capacity proof and
the eviction loop. If enough eligible capacity cannot be found, admission changes nothing. Otherwise
the existing baseline-first/LRU eviction and accounting machinery reclaims enough space. Primary
admission retains its existing policy. The result is still an exact byte baseline, never a hash.

`PROSPER_NO_IDLE_COMPUTE_BUFFER_RECLAIM=1` restores spare-capacity-only baseline admission.
The default comparison threshold is 8 MiB; `PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB` remains
configurable. All current output-overlap vetoes, content validation, watches, host-read barriers,
completion pins and architectural notifications remain in force. No guest publication is deferred.
Requested cache charges are Vulkan resource requirements; pooled physical allocations may be larger.

The runtime guards preserve exact shader output and untouched tails. They check recent-primary hits,
refusal through dispatch 257, first admission at 258, both steady GPU comparisons, later exact
rematerialization of the retired owner, and an insufficient-idle-capacity case that must preserve
the old primary and baseline. The original production policy from `14386e1d1` fails the new guard
at dispatch 258 (`baseline=budget`, expected `created`). Restoring the candidate restores the pass.

## Measurements

The first Sonic pair isolates idle reclamation at source `1584fff34`: both arms explicitly use the
8 MiB threshold, one executable, fresh save/cache directories, the same route and the same budget.
Enabled ran before the control. The ordinary F8 reader reports spans of 5.018 and 5.012 seconds.

| First Sonic pair | Original admission | Idle reclamation |
|---|---:|---:|
| Observed medium owners | 78 | 82 |
| Medium CPU result comparison | 32.215 ms | 0 ms |
| Medium writeback, enclosing comparison | 34.687 ms | 0.970 ms |
| Selected program GPU comparison bracket | 0.034 ms | 11.082 ms |
| Selected GPU-timed callback mean | 2.369 ms | 2.095 ms |
| Selected GPU-timed callback records | 193 | 202 |
| All compute recorded total | 1,595.160 ms | 1,610.945 ms |
| All graphics recorded total | 2,093.257 ms | 1,984.401 ms |

Every medium owner used its retained exact GPU result in the enabled window. Every larger primary
remained a hit in both arms, and requested residency stayed at 255 MiB. The selected program also has 115/120 records without GPU timestamps;
those are excluded from the callback means above. Inner execution-phase timers have a narrower
scope than the F8 callback total. Different record counts mean aggregate totals are not equal-work comparisons. CPU phase totals are nested, GPU intervals
are separately measured, and subtracting one from another is not a projected frame-time saving.

The second Sonic pair uses source `e518caeaa`, reverses the order (control first), and tests the
actual new defaults with no threshold override. Its control restores the former 16 MiB threshold
and spare-only admission. The ordinary F8 spans are 5.011 and 5.012 seconds.

| Actual-default Sonic pair | Former defaults | New defaults |
|---|---:|---:|
| Observed medium owners | 78 | 76 |
| Medium CPU result comparison | 32.756 ms | 0 ms |
| Medium writeback, enclosing comparison | 35.225 ms | 1.107 ms |
| Selected program GPU comparison bracket | 0.032 ms | 10.635 ms |
| Selected GPU-timed callback mean | 2.343 ms | 2.239 ms |
| Selected GPU-timed callback records | 195 | 188 |
| All compute recorded total | 1,614.571 ms | 1,577.119 ms |
| All graphics recorded total | 1,971.834 ms | 1,987.914 ms |

The 15.9375 MiB owners were disabled by the old threshold and use retained exact GPU comparisons
with the new defaults. All active large owners remain hits; their CPU comparison still costs
166.771 / 160.046 ms over 117 / 112 observations. The selected program's other 116 / 111 callbacks
lack GPU timestamps and are excluded from the callback means. Counts again differ between arms.

Across the two pairs the local saving repeats, but presentation rates move in opposite directions.
The first pair's control/enabled rates are 7.382 / 7.971 presentations per second; the reverse-order
actual-default pair gives 7.782 / 7.382. These are host presentations, not newly rendered frames.
The recording does not provide a newly rendered-frame counter. There is no supported FPS gain.
The 8 MiB size threshold and 256-access grace are generic, configurable/bounded policy choices,
not proof that every eligible workload or GPU benefits. No new Vulkan feature is required.

The actual-default GTA pair uses the same `e518caeaa` executable and the Performance Story route,
control first. Both show the lit bank, characters and HUD. Its observed buffers are all below
8 MiB (1,608 / 1,590 owner rows); it checks the broader route but does not stress the newly eligible
baseline population. The F8 spans differ: 5.018910 / 4.767412 seconds, with 19+21 / 20+20 periodic
samples. Compare reported spans and record counts, never assume each file covers exactly five seconds.

| Actual-default GTA pair | Former defaults | New defaults |
|---|---:|---:|
| Host presentations/s | 6.177 | 6.083 |
| Guest flips/s | 5.977 | 6.083 |
| Process CPU cores | 1.472 | 1.466 |
| Graphics recorded total | 2,577.776 ms | 2,531.304 ms |
| Compute recorded total | 1,137.899 ms | 1,161.800 ms |
| Compute dispatch/wait, enclosing GPU work | 410.252 ms | 414.862 ms |
| Compute GPU device brackets | 307.570 ms | 310.308 ms |
| Graphics GPU wait | 514.617 ms | 511.398 ms |
| Graphics GPU device brackets | 327.753 ms | 319.708 ms |
| Renderer / compute records | 761 / 2,060 | 748 / 2,008 |

This pair establishes neither a GTA speedup nor a causal regression. Its unequal spans and guest
states limit small aggregate differences. The ordinary report has no independent renderer-submit
CPU field; compute dispatch/wait includes command work and waiting, while the graphics wait and
GPU intervals are separate nested measurements. Do not sum overlapping scopes.

## Audio, completion and visual checks

All seven runs completed with fixed source, binary, route and helper hashes unchanged during each
run; no competing game/build/profiler was observed. All independently gated compute phase rows
report success. CPU profiles, folded stacks and flame graphs are retained, with clean conversion
logs. The machine identifies AMD Radeon 8060S Graphics / RADV STRIX_HALO, Vulkan 1.4. These results
do not measure NVIDIA or Windows behavior.

| Capture | Successful phase rows | Earlier GPU load mean | Main audio shortfalls, first→last |
|---|---:|---:|---:|
| Sonic census | 1,110 | 13.5% | 1→1 |
| Sonic policy control | 1,073 | 17.0% | 0→2 |
| Sonic policy enabled | 1,127 | 15.5% | 0→0 |
| Sonic former defaults | 1,092 | 19.0% | 0→0 |
| Sonic new defaults | 1,047 | 20.0% | 1→1 |
| GTA former defaults | 1,040 | 14.5% | 1→2 |
| GTA new defaults | 1,022 | 14.0% | 0→0 |

GPU load is the mean of 20 supported `radeontop` samples in the earlier profiling window, not the
F8 interval. Main-output callbacks advance with flat shortfall counts in each observed band:
60–240 s, 295–307 s and 345–355 s. GTA's separate early output reaches 40 cumulative shortfalls
in both arms and has no later-band coverage. These are per-output/per-generation demand counters,
not physical XRUNs or an audible-quality assessment. No audio behavior changes or improvements
are claimed; independent output/channel buffering remains intact.

The census Sonic run and GTA control each log one `submit=-4` only after shutdown begins. The
closing gate can return this without calling the driver; these logs do not distinguish the two
sources. The other five runs have no such status. No pre-shutdown occurrence was observed.

Every retained screenshot is published in the [progress blog](../../BLOG.md). The five Sonic
frames retain the known black world and visible HUD, at 55.45 / 53.63 / 56.28 / 53.28 / 51.05 s
in the table's order. Clock and HUD tint differ; this is not pixel equivalence or a rendering fix.
GTA retains the same bank setting with different character animation poses. These captures do
not establish full-library compatibility or newly rendered-frame FPS.

## Capture identities

The table uses original F8 SHA256 hashes. Retained manifests and the audit retain original frame/capture hashes and all frozen
helper/route identities alongside source and binary hashes. Publication only scales and encodes
the screenshots as WebP; original frames remain retained.

| Capture | Source | F8 SHA256 |
|---|---|---|
| Sonic census | `14386e1d1da6` | `b8f13b72bc2901720995a46e2f81e096ade607c31b5b84d1d32ae88d56b3e1b3` |
| Sonic policy control | `1584fff34723` | `105e97f48b7fba553c6e86a15a8a3212604bf30e89757321d3a1de2ae2a0c01d` |
| Sonic policy enabled | `1584fff34723` | `9b85814163dd3d8d914a6a7188cdfdff7baa58789195a301a2da2f99e21dd668` |
| Sonic former defaults | `e518caeaa8d5` | `753e09ad158cef9a26adabf639e176af8da0b1ec072fe3bf36b0937f9768a5b8` |
| Sonic new defaults | `e518caeaa8d5` | `51efff189cbdb8a67a7b91f95448fac890541d46fe7c3177cc9d6c7f0fe65158` |
| GTA former defaults | `e518caeaa8d5` | `8468f7704cb9fb14d3bc2ed12e76ed9d83a2e47dcac8cedf6d465353d9c09614` |
| GTA new defaults | `e518caeaa8d5` | `e627e5768499558264d38f57f97a82c3d286e82a6354e88f04b9e8ac8ac9ffd1` |

Executable SHA256 by captured source:

- `14386e1d1da6`: `7f583a230f71a53d2692d08d60f4800bd13d42c317bd85c2d1731c291045980c`
- `1584fff34723`: `669969561b62caaee9df24670fb1d38435d4e0df6338b4bf0cc03f0803d17f26`
- `e518caeaa8d5`: `eab609c163768dba5e85f21871bb21fa4b236d99b3c4fadd77a9cc0ec6685c90`

Subsequent documentation, image and diagnostic-baseline updates do not change the tested or
captured implementation.


## Reproduction and checks

Build with the real dump root so dump-gated tests remain present:

```sh
PROSPER_GAME_ROOT=<DUMP_ROOT> cmake -S prosper -B prosper/build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPROSPER_APP=ON -DPROSPER_AUDIO_SDL3=ON \
  -DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0
cmake --build prosper/build-linux -j6
ctest --test-dir prosper/build-linux --no-tests=error --output-on-failure
python3 prosper/tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux --sync \
  --allowlist <EMPTY_ALLOWLIST> --ctest-arg=-R \
  --ctest-arg='^(compute_buffer_timing_runtime|compute_buffer_hosted_overlap|compute_buffer_residency_runtime|storage_output_conflicts.*|live_compute_host_read_barrier)$'
```

At source `e518caeaa`: 509/509 tests passed in 126.06 s. The strict synchronization selection passed
9/9 in 2.12 s with zero messages after successful layer-insertion and deliberate-WAW controls.
The census guard includes an actually resident/unpinned owner; its offline controls corrupt real
records and separately construct same-address/different-materialization entries. Truncation and
failed-completion snapshot paths were source-reviewed, not claimed as runtime coverage.

Captures use the ordinary `prosper-app` frontend, native resolution, immediate presentation,
`PROSPER_RENDER=1`, `PROSPER_GUEST_ARGS=-force-gfx-direct`, fresh `PROSPER_SAVE0` and cache directories.
Sonic uses `prosper/scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`; GTA uses
`prosper/scripts/gta5/reach-performance-story.pad`. CPU sampling at 199 Hz and Radeon GPU load at
10 Hz cover seconds 250–270, before F8 at 300 s. F9 is at 330 s for Sonic and 310 s for GTA;
normal shutdown is requested at 380/360 s. The capture driver strips inherited graphics/guest
controls and rejects observed competing game, build or profiler processes. Original F8/F9,
BMP frames, CPU samples/folded stacks/SVGs, logs and per-output audio evidence are retained.

## Remaining scope

Retiring only the 31.875 MiB unused owner cannot fit a 63.75 MiB baseline beside the 223.125 MiB
active primary set under 256 MiB. The large-buffer CPU comparisons remain a bottleneck; raising
the budget or displacing that active set is not part of this change. #3407 remains open.
No newly rendered-frame FPS, full-library speedup or audio improvement is established by these
measurements. Sonic's known black world/HUD remains separate (#2790), as do its title-menu defects.


## Ruled out

- **Lowering the size threshold alone makes room for the medium comparisons.** With all six
  primaries resident, the first control still refuses every eligible baseline. The complete census
  identifies the previously unaccounted 31.875 MiB owner, and the matched policy pair admits both
  medium baselines only when idle reclamation is enabled.
- **A lower local comparison cost is already an FPS improvement.** Two Sonic pairs repeat the
  medium-buffer CPU saving while their presentation-rate differences have opposite signs. The
  available counter also includes repeated presentations. Retain the local claim only.


## Next performance milestone

Use the complete profiles to group the next resource-preparation changes before another full
before/after capture cycle. A local timer improvement is useful evidence, but it is not a
frame-throughput result. The current enabled windows still record 761 / 1,357 ms of renderer
resource preparation for Sonic / GTA, including 212 / 148 ms of backend buffer copying. Those
nested totals are over 5.012 / 4.767 seconds and must not be summed with their parents.

One bounded capacity experiment can first test the three large Sonic owners with the existing
`PROSPER_COMPUTE_BUFFER_CACHE_MB` override: require successful baseline admission, record exact GPU
comparison and enclosing dispatch costs, and check primary misses/uploads as well as throughput.
It would test a budget explanation; it would not justify a larger default without evidence.
GTA's current buffers do not justify lowering the threshold again.

The separate CPU profiles put 8.54–12.23% of weighted leaf periods in libc memmove across the four
actual-default captures. About 45% / 33% of that copy weight in enabled Sonic / GTA has no caller
beyond its thread label. The known backend-copy stacks account for 4.38% / 3.02% of total periods.
Do not assign missing-callchain work to a guessed resource. Use existing byte/timing counters and
improve attribution where needed before extending retention. In particular, the 512×512×6 FP16
texture witness does not describe every texture in the remaining category, and the prior non-BC
cube-cache regression still constrains any broader admission change.
