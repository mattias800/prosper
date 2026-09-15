# Packed native storage-image GPU retile

Tracking: [#3407](https://github.com/mattias800/prosper/issues/3407).

## Scope and evidence

The existing GPU tiler accepted word-sized ordinary images and mode-24 native two-byte arrays.
Ordinary R16_UINT and R8/RG8_UNORM images still tiled on the CPU even when their Vulkan transfer
already contained exact native guest pixels. This extension groups adjacent texels into full 32-bit
words and uses the same submission, linear baseline and host-publication contract as existing retile.

The retained GTA Performance-route F8 from source `b785d0cd35dc2129775ae50a5fcb8173320d1bec`
(buffer retention disabled) identifies the following CPU layout work:

| Native format / mode | Writeback records | CPU layout ms | Repeated guest bytes |
| --- | ---: | ---: | ---: |
| R8_UNORM / 27 | 30 | 19.495 | 117,964,800 |
| RG8_UNORM / 27 | 30 | 14.935 | 235,929,600 |
| R16_UINT / 24 | 31 | 17.241 | 97,517,568 |

This is 51.671 ms of targeted CPU work in a 5.0185-second window, not a prediction of net savings.
Packing already takes zero reported milliseconds for these native inputs. The F8 reports 373.427 ms
of total compute writeback; adding GPU work and retaining its output can offset CPU savings. Current
before/after game results must therefore include the enclosing dispatch and presentation intervals.
Freshly rendered frames are not distinguished by these captures. The source, binary and original
capture are identified in [the preceding measurement guide](RENDERER_BUFFER_RESIDENCY_2026_09.md).

## Ownership and layout contract

- Mode 24 admits two-byte texels; mode 27 admits one- or two-byte texels. The selected pipe equation
  must map its low address bits exactly to the horizontal group and exclude those x bits from every
  higher address bit. Each invocation owns a complete aligned output word; no read-modify-write,
  8/16-bit storage capability, atomics or vendor-specific feature is needed.
- Ordinary 2D images, one-layer views and existing exact native multilayer images use independent
  planes. A row must contain complete words. Partial rows, non-native interchange, selected mips,
  ambiguous metadata/pitches and unsupported equations retain CPU layout. Native shader declarations
  are unchanged; RG8/R16F multilayer interchange remains the existing CPU path.
- Padded dispatch and byte bounds are checked before allocation. Out-of-image output words are zeroed
  without reading the source. The linear result remains separate for exact comparison; the tiled
  buffer is owned through submission completion and receives its own host-read barrier.
- Allocation failure retains CPU fallback. Device loss remains a sticky failure; it is not hidden as
  successful fallback. The ordinary guest write-watch and publication sequence is unchanged.

`PROSPER_NO_GPU_RETILE_PACKED_EXTENSION=1` restores the previous packed scope (mode-24 two-byte
multilayer images only), while leaving word/volume GPU tiling enabled. It is a same-binary comparison
control. `PROSPER_NO_GPU_RETILE=1` still disables all GPU retile. Buffer retention remains opt-in.

## Verification

`test_gpu_retile --subword` exercises actual native R8_UNORM, R8_UINT, RG8_UNORM, R16_UINT and R16F
storage-image copies through the production compute backend. It checks exact guest bytes, poisoned
padding, destination guards, three partial updates crossing packed-word boundaries, ordinary and
one-layer array views, incomplete-word fallback, actual GPU recording and host barriers. The existing
multilayer Uint16 cases cover both modes; RG8/R16F arrays retain explicit raw-interchange controls.
Seven separate processes cover `PROSPER_RX_PIPES=1,2,4,8,16,32,64`. Existing allocation/mapping failure
and device-loss controls remain, including new ordinary packed-image cases.

Focused build and 30/30 retile tests passed. With the extension deliberately disabled, the subword
fixture fails 144 per-dispatch and 48 aggregate GPU-recording assertions while every byte-correctness
assertion passes. A collapsed equation bit is separately rejected by the permutation guard.

```sh
cmake --build prosper/build-linux -j6
ctest --test-dir prosper/build-linux --no-tests=error --output-on-failure
python3 prosper/tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux \
  --probe test_gpu_retile --sync --allowlist /dev/null \
  --ctest-arg=-R '--ctest-arg=^gpu_retile' --ctest-arg=--no-tests=error --ctest-arg=-j1
```

The full Release build passed **496/496 tests**. The strict synchronization-validation scan passed
**30/30 retile cases with zero messages**, using an empty allowlist. Its layer-load probe and
deliberate write-after-write hazard proved that validation was active before the real tests.
These checks and all four game runs used source `6de414b799074bada14578358b0d0c350c565fd3`,
app SHA-256 `828d6edcf808454ed4df012afddf65f07b75466b2b8700f2ff48d9e33582b8f4`.

## Matched game captures

Linux `prosper-app`, Radeon 8060S/RADV, normal windowed frontend, native resolution and immediate
presentation. Both arms use one binary, fresh save/cache directories, unchanged helpers and routes,
and default buffer-retention settings. The sole normalized environment difference is the extension
control above. GTA uses `prosper/scripts/gta5/reach-performance-story.pad`; Sonic uses
`prosper/scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`. CPU sampling (`perf record -F 199 -g`)
and GPU utilization sampling run for 20 seconds starting at elapsed 250 seconds and finish before
295 seconds. F8 starts at 300 seconds; F9 follows at 310 seconds for GTA and 330 seconds for Sonic.
The runs stop after 360/380 seconds respectively. No foreign workload was observed in the sampled
process census. All four runs exited normally, with complete F8/F9 files, unchanged binary/helper
hashes and successful logged compute phases. Original BMPs, bundles, profiles and receipts are retained.

| F8 observation | GTA off | GTA on | Sonic off | Sonic on |
| --- | ---: | ---: | ---: | ---: |
| Measured sample span, seconds | 5.018708 | 5.018375 | 4.766231 | 5.013613 |
| Host presentations / second | 6.176888 | 6.177298 | 7.343328 | 7.180450 |
| Average CPU cores in F8 | 1.485 | 1.459 | 1.453 | 1.472 |
| Renderer total, ms | 2691.722 | 2693.100 | 1970.438 | 2042.843 |
| Compute total, ms | 1196.613 | 1172.010 | 1596.750 | 1624.456 |
| Compute GPU time, ms | 312.115 | 327.398 | 359.149 | 352.081 |
| Renderer wait / device time, ms | 536.132 / 341.623 | 547.891 / 350.275 | 633.280 / 514.011 | 642.083 / 522.321 |

Intervals use the actual first-to-last post-trigger sample span, not an assumed five seconds.
The complete Sonic-off capture has 20 post samples; the other three have 21. Counter rates use
that measured span. Counts are presentations, not newly rendered frames; no fresh-frame counter
is available. GTA's reciprocal mean host-presentation interval is 161.894/161.883 ms. These captures
do not supply individual frame-time percentiles. Nested CPU/GPU intervals must not be added together.

### GTA target result

All three ordinary-image targets appear **32 times in each arm**, with identical byte counts and shader
identities. Each changes from 0/32 to 32/32 GPU-retiled writebacks. Historical multilayer R16 images
are excluded from these target counts.

| Target | Mean layout ms, off / on | Mean complete image writeback ms, off / on |
| --- | ---: | ---: |
| R8_UNORM / 27 | 0.610844 / 0.166063 | 0.715219 / 0.249438 |
| RG8_UNORM / 27 | 0.478469 / 0.326375 | 0.660250 / 0.465500 |
| R16_UINT / 24 | 0.555469 / 0.148125 | 0.692844 / 0.293688 |

The remaining layout time on the GPU path includes copying the tiled result to guest memory; it is
not zero-copy publication. Summed target layout cost falls from 52.633 to 20.498 ms. The two enclosing
programs (`0x007b239298ae2f88`, `0x4b8f65de95fb620d`) each have 32 dispatches per arm:
their combined time falls from **144.383 to 118.387 ms**, a **25.996 ms** local saving. Their GPU time
rises from 33.999 to 42.735 ms, already included in that enclosing cost. Aggregation uses every raw
compute record, not the summary's top-ten shader list; a faster shader must not disappear from the
comparison. An independent reader reproduced the raw totals.

Both F9 images show the same intact bank world/HUD and camera, with ordinary character-pose changes.
**No presentation-rate improvement was observed.** This one matched pair supports a targeted cost
reduction, not a stable whole-game speedup or a pixel-identical replay claim.

### Sonic compatibility and audio limits

Sonic has no newly eligible packed-image writeback in either F8. The same format/layout groups retain
their prior GPU/CPU admission. Both images show the known black world with visible HUD (#2790), at
slightly different game-clock times. This is limited compatibility evidence, not correct world rendering
or a performance improvement. Title-screen problems (#2206) remain outside this change.

Per-output/open-generation audio demand remains separate. Main output `17/1/active` has 1/2 observed
shortages in GTA off/on and 1/1 in Sonic off/on, all within the first periodic snapshot near one second.
All four main-output streams have advancing callback coverage and zero additional shortages in the
covering bands 60–240, 295–307 and 345–355 seconds. GTA's separate intro output `18/1/active` has
40/41 observed shortages and stops reporting near 16 seconds; no late coverage is inferred for it.
These are periodic SDL demand observations, not final-close totals, physical XRUN counts or audible
quality evidence. No audio implementation or independently owned channel buffer changed. #3435 remains open.

## Remaining work and reproduction artifacts

The largest GTA renderer component remains resource preparation: 1474.917/1467.993 ms off/on in
these F8 windows. Existing GPU-retiled formats still copy large outputs to guest memory: GTA's
format-20 three-component mode-27 group records 193 writebacks / 2,109,603,840 bytes and 62.314 ms
in the layout/copy timer with the extension enabled. Skipping those writes requires the guest-content
authority and deferred-publication contracts (#3441, #1854), not another tiling shader alone.
Sonic's six-layer Float16x4 mode-27 fallback records 252 writebacks, 3.713 ms packing and 47.318 ms
CPU layout; it is a separate measured candidate, with native-format and layer semantics still to prove.
Do not close #3407 on this extension or reopen the rejected buffer-retention enablement experiment.

Private evidence is retained under `<EVIDENCE_ROOT>/issue-3407-packed-retile-20260912/`.
The capture directories are `packed-off-gta`, `packed-on-gta`, `packed-off-sonic`, `packed-on-sonic`.
Each includes `run.json`, `child-result.json`, `effective-env.json`, `process-census.json`, `run.log`,
the `.prperf`/`.prgbundle`/original BMP, `summary.json`, writeback/phase rows, audio coverage,
`cpu.data`, `cpu-self.txt`, folded stacks and `cpu-flame.svg` (through `stackcollapse-perf.pl`).
`verification.json` binds the full tests and strict scan; `pair-verification.json` checks matched
inputs and successful compute phases. `packed-gta-comparison.json`, `packed-sonic-comparison.json`
and `audio-comparison.json` retain numeric results. The local capture/report helpers are archived
with them so the route can be repeated without reconstructing the protocol from prose.

```sh
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/packed-retile-capture.py disabled gta packed-off
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/packed-retile-capture.py enabled gta packed-on
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/packed-retile-capture.py disabled sonic packed-off
env -u SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS python3 prosper/build-linux/evidence/packed-retile-capture.py enabled sonic packed-on
```

Run one arm at a time after establishing an idle machine; the helper refuses competing workloads
and requires a clean committed checkout whose embedded app revision matches HEAD. Labels must name
fresh directories. The archived helper's private dump paths must be adjusted on another machine.

## Ruled out

- Horizontal packing does not apply to every sub-word mode: mode-24 byte texels put y0 in address
  bit1. The equation proof refuses them, and real runtime cases preserve correct CPU fallback. #3407.
- Reduced CPU layout alone is not an end-to-end speedup. The enclosing GPU/dispatch and presentation
  costs remain the acceptance measurements; the previous buffer-cache experiment remains closed.
- **CPU tiling is not a material cost on the measured Sonic Frontiers route, and `layout_ms` is not a
  measurement of it.** `layout_ms` times the whole publication branch, and both of its arms land in the
  same counter: a GPU-retiled result is memcpy'd out of the retile buffer, a declined one is tiled on
  the CPU. Split on 2026-09-15 over 18,354 dispatch records (`PPSA03831`, native `prosper-app`,
  Linux/RADV, default launch, full cadence): `layout_ms` **6,551 ms** of which `retile_copy_ms`
  **6,541 ms** and CPU tiling proper **10 ms**. So reading `layout_ms` as tiling cost overstated it by
  roughly 650x on this route. `PROSPER_GPU_RETILE_CENSUS` says why the split lands there: on the hot
  program `0x20002fe800` the retile is **admitted 5,961 of 10,001** times, and all 4,040 declines are
  `inexact-bytes` (5,961 + 4,040 = 10,001, so the counters prove the whole bucket). The census's
  recorded shape for that bucket is `2x2x1` — **the first decline seen, not a proven typical one**;
  nothing in the instrument establishes that the other 4,039 share it, and the 10 ms does not depend
  on their shape. Read the "10 ms" as two significant figures at most: it is the difference of two
  ~6,550 ms sums of `%.2f`-quantised per-dispatch values, and per-dispatch copies under 0.005 ms print
  as `0.00`, which biases the residual upward. The conclusion — immaterial — survives that; the digits
  do not. Do not open further CPU-tiling vectorisation work against this route without first
  re-splitting the timer: the cost is publication, and #3683 carries it. #3683.
