# Resource preparation: cheaper keys and retained GPU comparisons (#3407)

The combined change reduces warm shader-key preparation work and raises the persistent compute
buffer budget from 256 to 512 MiB. The additional capacity lets large exact-result baselines stay
on the GPU. Descriptor-range and dispatch-count checks preserve the exact CPU fallback on devices
that cannot compare an extent in one dispatch. No experimental driver switch or new Vulkan feature
is required.

The final Sonic capture reaches 7.77 host presentations/s, versus 6.99 and 7.38 in the two earlier
256 MiB observations with the new key implementation. Large-owner CPU comparisons fall from
156–159 ms per recorded window to zero. GTA remains around six presentations/s with no established
speedup. These are limited routed observations, not newly rendered FPS or a library-wide gain.

## Implementation boundaries


- Hash each existing semantic field a word at a time, then avalanche the result for bucket
  selection. Full key equality remains unchanged. Shader-code hashes, shader dump names, compiled
  module identities and persisted cache formats are unchanged.
- Rebuild resource semantics on every lookup using reusable vector storage. Each active call owns
  its vector. No guest data, descriptor result or resource-generation decision is memoized.
- Keep at most 64 KiB of idle vector capacity per thread. Active keys and cache entries have their
  own allocations. Larger scratch is discarded. A smaller cache miss restores the existing fresh
  vector allocation behavior before insertion, returning oversized scratch before cache eviction
  or accounting changes. Stored keys do not participate in scratch recycling.
- Freeze each policy once per process. `PROSPER_NO_SHADER_KEY_WORD_HASH=1` restores the original
  byte mixer; `PROSPER_NO_SHADER_KEY_SCRATCH=1` restores allocation on each resource-bearing lookup.
  Both controls are for comparison and bisection. No Vulkan feature or platform-specific memory
  tracking is introduced.

The compute budget remains configurable through `PROSPER_COMPUTE_BUFFER_CACHE_MB`: 256 restores
the previous capacity, zero disables retention, and malformed input falls back to the new 512 MiB
default. Admission, idle grace, pins, content invalidation, output-overlap vetoes and guest writeback
ordering are unchanged. This is a generic capacity change, with no game-name or shader-hash policy.

`compute_result_compare_group_count` checks the full byte extent against the device's
`maxStorageBufferRange` and the required group count against `maxComputeWorkGroupCount[0]` before
narrowing to Vulkan's 32-bit arguments. Both owned and adopted devices populate those limits.
Otherwise eligible image extents rejected by this device-limit guard retain a current host
snapshot; unsupported buffers keep exact CPU comparison. A 256 MiB extent needs 65,536 groups, beyond the conformant limit of 65,535 on some
devices, so increasing retention without this guard would expose an invalid dispatch.

The requested cache charge in Sonic rises from 255 to **510 MiB**. That is not physical GPU memory
usage: the separate allocation pool may supply larger allocations and retain evicted allocations.
Its default budget is separately 640 MiB. GTA's maximum observed requested cache charge also rises
from 11.25 to 267.25 MiB; the additional retained population yields no established GTA speedup. This PR does not introduce a total process-memory cap or
an adaptive memory-budget policy. Baseline allocation failures retain the existing fallback.

## CPU benchmark

Two balanced forward/reverse process observations per policy, 64 contexts and 100,000 warm API
calls per resource population, using one Release executable. Median nanoseconds per lookup:

| Resources | Original hash + allocation | New hash + scratch | Original hash + scratch | New hash + allocation |
|---|---:|---:|---:|---:|
| 8 | 2043.5 | 1032.7 | 2050.7 | 1000.2 |
| 32 | 6704.7 | 2851.3 | 6779.5 | 2855.5 |
| 64 | 12932.4 | 5328.3 | 13127.7 | 5306.1 |

The combined path takes 49–59% less time in these three synthetic populations, principally from
hashing. Scratch removes warm scalar C++ allocation requests, but has no clear isolated timing
benefit in this experiment. Timing includes the allocation counter and module/identity checks;
these are two observations per policy, not a confidence interval or game frame-time result.

The game profiles independently reduce the hash function's share of sampled leaf cycle-period
weight from 1.125% to 0.468% in Sonic, and 2.789% to 0.996% in GTA. Sampled cycles are not elapsed
CPU time. Global allocation samples do not isolate the scratch optimization.

## Routed game measurements

All captures use Linux `prosper-app`, AMD Radeon 8060S / RADV, native resolution and full cadence,
normal windowed presentation, fresh save/cache directories and the committed gameplay routes.
The machine was reserved for these runs; the known game/build/profiler census found no competing
workloads. Capture inputs, source cleanliness, executable hashes and route hashes were checked
before and after each run. F8 reports are complete, and every captured compute phase reports success.

The first six runs use source `47420495b85482ede8e0d31af3408a8a94c689fd`, which includes the key
changes and the old 256 MiB default. They use one executable with SHA-256
`51cded865993ff0c0fc2f29f8ef7e5974b89648822897d7a7c16a893cd4bdaa9`.
The final default confirmations use `1a28926a1d8c39613d36d762fd2cddafdbf5f9db`, including the new
budget and device-limit guards, with executable SHA-256
`eb2bc32005c891e260217682c82dfb3a67f0ac2a9e2d4b619ee5e10472362abd`.
Both include main `ac4dc1c2d735aeabde136fdc22e21eea40a35679`.

| Run, in collection order | Budget MiB | F8 seconds | Host presentations/s | Process CPU cores | Large-owner CPU comparisons |
|---|---:|---:|---:|---:|---:|
| Sonic legacy keys | 256 | 5.013 | 6.782 | 1.417 | 144.904 ms / 103 observations |
| Sonic new keys | 256 | 5.010 | 6.986 | 1.464 | 159.080 ms / 110 |
| GTA new keys | 256 | 5.022 | 6.173 | 1.453 | no matching population |
| GTA legacy keys | 256 | 5.020 | 5.976 | 1.462 | no matching population |
| Sonic new keys, capacity override | 512 | 5.021 | 7.768 | 1.224 | 0 ms / 120 |
| Sonic new keys, 256 repeat | 256 | 5.015 | 7.377 | 1.449 | 156.495 ms / 111 |
| Sonic final defaults | 512 | 5.020 | 7.768 | 1.237 | 0 ms / 118 |
| GTA final defaults | 512 | 5.028 | 5.967 | 1.439 | no matching population |

The capacity override is a same-binary comparison. Final defaults are a separate rebuilt
confirmation. All large-owner observations in both 512 MiB runs are GPU-proven unchanged results.
They are 66,846,720-byte owners of shader `0x1c413fd58124b291`, named only to identify the measured
population. The previous budget rejects their baseline admission.

Moving comparisons to the GPU costs GPU time: the selected shader's comparison bracket rises
from approximately 10.3 ms in the 256 MiB observations to 97.2 ms with final defaults. Its enclosing
recorded total falls from 503.2/514.1 to 432.4 ms, with different populations (296/291 versus 313).
The removed CPU time is therefore **not** a net frame-time saving. Texture work also changes across
runs: final Sonic RTT preparation is 48.7 ms versus 4.3/8.0 ms, while invalid-texture preparation
falls from 99.2/72.5 ms to zero. Those route differences limit attribution of the presentation delta.

| Final-default F8 scope | Sonic | GTA |
|---|---:|---:|
| Graphics recorded total | 2076.573 ms | 2605.549 ms |
| Renderer resource preparation | 829.423 ms | 1404.761 ms |
| Graphics GPU wait | 660.653 ms | 518.295 ms |
| Graphics GPU device brackets | 542.958 ms | 318.979 ms |
| Graphics wait overhead | 117.695 ms | 199.317 ms |
| Compute recorded total | 1533.785 ms | 1161.671 ms |
| Compute dispatch/wait | 621.036 ms | 404.229 ms |
| Compute writeback | 467.219 ms | 355.022 ms |
| Compute GPU device brackets | 466.362 ms | 303.753 ms |
| Renderer / compute records | 625 / 1329 | 751 / 2010 |

These are nested recorded scopes, not additive frame budgets. The capture has no independent
renderer-submit CPU field and no newly rendered-frame counter. Host presentations can repeat;
`rendered_fps` remains null. Do not convert this result into a newly rendered FPS claim. The earlier
20-second `radeontop -t 10` windows and CPU profiles are separate from F8; their GPU load and
provenance are retained in the [measurement summary](RESOURCE_PREPARATION_2026_09.json).

## Audio and visual checks

Final main-output active-phase cumulative shortfalls stay at 2→2 for Sonic and 1→1 for GTA, with
callbacks advancing. GTA's auxiliary output 18 separately gains 38 shortfalls around 16 seconds.
These counters describe demand on each output/open generation, not physical device XRUNs. Earlier
Sonic increases occur after F8 around 332/376 seconds. No audio improvement follows from these
observations, and this PR changes no audio production, buffering or mixing ownership.

All eight automatic screenshots are included in the [blog](../../BLOG.md). GTA's Performance Story
bank retains lighting, world geometry, characters and HUD; actor poses vary. Sonic retains its
known HUD over an absent world. This PR establishes no Sonic rendering fix or progression milestone.
Screenshots are direct frontend output, resized/re-encoded for publication; original BMPs and F9
bundles remain with the raw captures. No snapshot baseline was changed.

## Reproduction and regression guards


Build `test_shader_recompile_cache` in a Release configuration. The focused CTest matrix runs the
semantic suite in all four policy combinations:

```sh
ctest --test-dir prosper/build-linux --no-tests=error -R '^shader_recompile_cache' --output-on-failure
```

The optional benchmark invokes the real warm shared-module API with synthetic resource contexts:

```sh
prosper/build-linux/test_shader_recompile_cache --benchmark-key-preparation
PROSPER_NO_SHADER_KEY_WORD_HASH=1 prosper/build-linux/test_shader_recompile_cache --benchmark-key-preparation
PROSPER_NO_SHADER_KEY_SCRATCH=1 prosper/build-linux/test_shader_recompile_cache --benchmark-key-preparation
PROSPER_NO_SHADER_KEY_WORD_HASH=1 PROSPER_NO_SHADER_KEY_SCRATCH=1 prosper/build-linux/test_shader_recompile_cache --benchmark-key-preparation
```

Use the same executable and a quiet CPU window; repeat with balanced order. JSON lines report
resource/context/iteration counts, scalar C++ allocation requests, SPIR-V checksums and nanoseconds
per lookup. The shader does not execute these synthetic resources, so this measures API preparation,
not representative shader execution. Compare checksums and exact hit/miss assertions across arms
before comparing times. There are no timing thresholds in the functional tests.

The new functional guards require allocation-free warm resource keys, distinguish resource-only
context changes, return immediately to a large warm key after smaller misses, and check that
storage exceeding the idle limit is not retained. Against original production code, the unchanged
new tests fail six allocation assertions. Removing only miss compaction fails the immediate return
to the large key. To reproduce these negative controls in a disposable checkout, keep the new test
and replace `gpu_executor.cpp` with its version from `ac4dc1c2d735`, then rebuild and run
`test_shader_recompile_cache`. For the second control, retain the candidate and remove its two
`scratch.prepare_for_cache()` calls before rebuilding. Restore the candidate after each control. The existing suite also covers shader changes, diagnostic selectors, compute
configuration, vertex chains, concurrent readers and shared-module lifetime across cache reset.


The comparator guard is covered at exact descriptor/group boundaries, zero, misalignment and
64-bit overflow-sized inputs. Removing the range/group checks from the production helper makes
three of the new boundary assertions fail. This is an arithmetic negative control, not a simulated
small-limit GPU. The existing hardware runtime tests exercise actual exact comparisons and fallback
ownership. Local NVIDIA and Windows game captures were not performed.

Build and verification (Release, game corpus configured through `PROSPER_GAME_ROOT`):

```sh
cmake --build prosper/build-linux -j6
ctest --test-dir prosper/build-linux --no-tests=error --output-on-failure -j6
# Empty allowlist: the focused synchronization run permits no validation messages.
: > prosper/build-linux/strict-allowlist.txt
python3 prosper/tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux --sync \
  --allowlist prosper/build-linux/strict-allowlist.txt --ctest-arg=-R \
  --ctest-arg='^(live_compute_host_read_barrier|storage_output_conflicts.*|compute_buffer_timing_runtime|compute_buffer_hosted_overlap|compute_buffer_residency_runtime)$'
```

Combined production code: **512/512 CTest cases pass** (132.11 s); **9/9 focused synchronization
checks pass** (2.05 s), with zero messages after the runner's layer-load and deliberate WAW hazard
positive controls. Code and measurement interpretation received independent reviews. The original
key implementation and missing-compaction negative controls described above fail as expected.

For game reproduction, use `scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad` or
`scripts/gta5/reach-performance-story.pad` under `prosper/`. Use separate fresh `<RUN>` directories
for save, generic/Mesa/NVIDIA caches and pipeline-cache file. Clear inherited `PROSPER_*`, `VK_*`,
`MESA_*`, `RADV_*` and SDL overrides, retaining the normal desktop environment. Set:

```sh
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_SAVE0=<RUN>/save PROSPER_CAPTURE_DIR=<RUN> \
PROSPER_PAD_SCRIPT=@<REPO_ROOT>/prosper/scripts/<ROUTE> PROSPER_PAD_SCRIPT_LOG=1 \
PROSPER_PERF_CAPTURE_AFTER_MS=300000 PROSPER_GRAB_BUNDLE_AFTER_MS=<F9_MS> \
PROSPER_RENDER_TIMING=1 PROSPER_COMPUTE_PHASE_TIMING=1 \
PROSPER_COMPUTE_TIMING_CAPTURE_ONLY=1 PROSPER_COMPUTE_BUFFER_TIMING=1 \
PROSPER_COMPUTE_IMAGE_TIMING=1 PROSPER_AUDIO_DEBUG=1 PROSPER_AUDIO_QUEUE_TIMELINE=2 \
PROSPER_AUDIO_FLOW=1 PROSPER_AUDIO_DEMAND=1 PROSPER_AUDIO_DUMP_WAV=<RUN>/audio.wav \
PROSPER_GRAPHICS_PIPELINE_CACHE_PATH=<RUN>/graphics.bin \
XDG_CACHE_HOME=<RUN>/cache MESA_SHADER_CACHE_DIR=<RUN>/mesa-cache \
__GL_SHADER_DISK_CACHE_PATH=<RUN>/nvidia-cache TMPDIR=<RUN>/tmp \
prosper/build-linux/prosper-app <DUMP_ROOT>/<TITLE_ID>-app0
```

F9 is 330000 ms for Sonic (380-second run) and 310000 ms for GTA (360-second run). Collect
`perf record -F 199 -g -p <APP_PID> -o <RUN>/cpu.data -- sleep 20` and
`radeontop -t 10 -d <RUN>/gpu.log -l 20` at seconds 250–270; require both to finish before 295.
Do not trigger manual F8/F9 during the run. For the old capacity on the final executable, add only
`PROSPER_COMPUTE_BUFFER_CACHE_MB=256`; for old keys add both key opt-outs. Recheck the actual binary
revision with `tools/revision/check_build_revision.py --binary` before each comparison.

Read F8 with `python3 prosper/tools/perf/performance_capture_report.py <RUN>/<CAPTURE>.prperf --json`.
Buffer timing is emitted as `[compute-buffer-timing]` key/value rows in `<RUN>/run.log`; the
`PROSPER_COMPUTE_TIMING_CAPTURE_ONLY=1` gate confines these to the F8 interval. Parse keys with
`([a-zA-Z0-9_-]+)=([^\s]+)`, select `owner-resolved=1`, and group by `hash` and `bytes` before
summing `result_compare_ms`. Do not sum alias bindings as distinct owners. Convert CPU profiles with:

```sh
perf script -i <RUN>/cpu.data -F comm,pid,tid,time,period,event,ip,sym,dso > <RUN>/cpu.script
stackcollapse-perf.pl --tid <RUN>/cpu.script > <RUN>/cpu.folded
flamegraph.pl <RUN>/cpu.folded > <RUN>/cpu.svg
```

The committed JSON contains exact revisions, binary/route/F8/original-image hashes, timing totals,
cache populations and per-output demand endpoints. Large raw captures, profiles, folded stacks,
flame graphs, WAVs, test logs and immutable-input manifests are retained in the local evidence
archive; the JSON alone is not a replacement for raw-trace reanalysis.

## Remaining work

#3407 remains open. CPU copies, dynamic fetch-resource resolution, texture conversions and guest
writeback are still substantial. The first paired profiles put `memmove` at roughly 7–12% of leaf
cycle weight and dynamic fetch-resource resolution around 2–5%; attribute their callers before
changing ownership or adding caches. Renderer resource preparation still dominates much of GTA's
recorded CPU work. Keep the established content-invalidation and guest-publication contract while
attacking those costs. Broader-library compatibility, Sonic's missing world and independent audio
scheduling/production work remain separate obligations.
