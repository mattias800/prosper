# Persistent texture source snapshot ownership

Tracking: [#3407](https://github.com/mattias800/prosper/issues/3407).

After decoding an encoded texture, the renderer reads an authoritative source snapshot for later
cache validation. Previously, admitting the decoded entry copied that already-read snapshot again.
Admission can now transfer its allocation, leaving the outgoing prefix allocation available as
scratch. This removes a CPU copy; it does not remove a guest read, detile operation or GPU upload.

## Ownership and memory bounds

The guest read stays after decode. Cache budget admission, old-entry debit, immutable decoded pixel
retirement, content versions, write watches and journal snapshots retain their existing order.
The handoff runs only after admission succeeds and stores exactly the actual readable prefix.
Earlier draw spans keep their original decoded pixel owners when an interleaved write replaces an
entry. Rejected admission leaves the populated scratch unconsumed.

Global scratch can hold a preceding large texture. Transferring it into many small entries would
retain more memory than their logical-size ledger records. Transfer therefore requires
`scratch.capacity() <= max(actual_prefix_size, inherited.capacity())`; otherwise the existing copy
path runs. The helper returns the actual selected path, which drives the counters. No title or
format-specific threshold is used, and no Vulkan feature or driver option is added.

`PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE=1` restores copying in the same binary.
`PROSPER_NO_TEXTURE_PREFIX_INHERIT=1` independently prevents reusing the outgoing allocation;
setting both restores the previous allocation-and-copy control.

The standard F8 report exposes copied bytes, transferred bytes and handoff time. They describe
admitted CPU snapshots, not guest reads, Vulkan uploads or resident occupancy. Handoff time is
already included in frontend texture preparation; it must not be added to it. Legacy or partially
missing populations report unavailable, while a complete zero population reports zero.

## Verification

The full Release suite passed **500/500 tests**. Seven focused tests cover CPU ownership and short
prefixes, differently sized replacements, bounded capacity, interleaved packed10 guest writes,
earlier/later pixels, and real renderer-to-F8 propagation across split semantic submits.
Five Vulkan cases passed strict synchronization validation with zero messages and an empty
allowlist; a layer-load probe and deliberate write-after-write hazard verified the instrument.
The ownership helper also passed AddressSanitizer and UndefinedBehaviorSanitizer.
The added offline report coverage subsequently passed all 43 reader tests.

Disabling movement while forcing the F8 fixture to expect it produces exactly four byte-counter
failures, with its pixel, callback, timer and warm-cache checks still passing. The independent
no-prefix-inheritance control also passes. These distinguish a missing optimization from corrupt
pixels and verify that reported zero work is observable.

```sh
cmake --build prosper/build-linux -j8
ctest --test-dir prosper/build-linux --no-tests=error --output-on-failure
python3 prosper/tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux \
  --sync --allowlist /dev/null --ctest-arg=-R \
  '--ctest-arg=gpu_capture_render|render_.*capture' --ctest-arg=--no-tests=error

# Expected exit 1: four snapshot byte-counter failures, other checks pass.
PROSPER_NO_LIVE_PERSISTENT_COLOR_TARGETS=1 \
PROSPER_BACKEND_BUFFER_RESIDENCY_MB=256 PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS=4096 \
PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE=1 \
  prosper/build-linux/test_render_buffer_capture --source-snapshot-expect-move
```

## Comparable captures

All game runs use source `99b8fa410444a4146e7f9e9814cd312ab077ea1f`, app SHA-256
`bd3ca4545b5be73196840cb020c45901255a12fd3c4ba640ff9becca1ab43899`.
Linux/RADV on Radeon 8060S, normal windowed frontend, native resolution, immediate presentation,
fresh save/cache directories and the default cache budgets. The sole normalized environment
difference within each pair is `PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE=1` in the off arm.

The [preceding capture protocol](GPU_RETILE_PACKED_2026_09.md#matched-game-captures) is retained:
GTA Performance Story and Sonic gameplay gamepad scripts; 20-second CPU/GPU profiles at elapsed
250 seconds, observed finished before 295; F8 at 300; F9 at 310/330; shutdown requested at 360/380.
Use the launch recipe there with the snapshot control above in place of the packed-retile control.
Original F8/F9 files, unmodified BMPs, profiles, source/binary/helper hashes, effective environments
and process-census receipts are retained. The offline command is:

```sh
python3 prosper/tools/perf/performance_capture_report.py <CAPTURE.prperf>
```

| F8 observation | GTA off | GTA on | Sonic off | Sonic on |
| --- | ---: | ---: | ---: | ---: |
| Actual sample interval (s) | 5.023040 | 5.018951 | 5.012727 | 5.010388 |
| Copied snapshot bytes | 1,120,665,600 | 117,964,800 | 0 | 0 |
| Transferred snapshot bytes | 0 | 969,277,440 | 0 | 0 |
| Snapshot handoff ms | 45.530 | 5.666 | 0.000 | 0.000 |
| Frontend texture ms | 684.829 | 655.465 | 362.294 | 363.991 |
| Frontend build_resources ms | 945.635 | 922.278 | 451.042 | 449.800 |
| Total renderer resource ms | 1421.908 | 1399.776 | 796.171 | 794.532 |
| Graphics GPU device ms | 301.623 | 299.746 | 481.431 | 479.730 |
| Graphics GPU wait ms | 475.634 | 469.492 | 594.298 | 591.008 |
| Total compute ms | 1131.864 | 1140.316 | 1659.362 | 1637.360 |
| CPU cores used | 1.486 | 1.487 | 1.450 | 1.439 |
| Host presentations/s | 5.972 | 5.977 | 7.182 | 6.985 |
| Guest flips/s | 5.972 | 5.778 | 7.182 | 6.985 |
| Newly rendered frames/s | Unavailable | Unavailable | Unavailable | Unavailable |

GTA's enabled window transfers **969,277,440 bytes** without the second copy. The remaining
117,964,800 bytes use the bounded copy fallback. Handoff time falls from 45.530 to 5.666 ms, while
frontend texture preparation falls from 684.829 to 655.465 ms and enclosing resource work from
1,421.908 to1,399.776 ms. Populations differ slightly: these are comparable elapsed windows, not
identical replayed workloads. Host presentation is essentially unchanged; no FPS improvement is
established. Neither capture distinguishes a new rendered frame from a repeated presentation.

Sonic records zero handoffs in both F8 populations, so these windows do not measure this optimization.
Both F9 images show the known black world and visible HUD. GTA retains its bank scene and lighting.
The screenshots are compatibility observations, not pixel-identical replay proofs or evidence of
fixing Sonic's world/title bugs.

The largest named self leaf in the separate CPU profiles is broad memory movement (8.46%/8.18% of sampled cycle
periods for GTA; 10.81%/10.50% for Sonic). Those stacks contain many call sites and do not isolate the
snapshot copy. Average coarse GPU utilization is 11.5%/14.5% for GTA and 21.0%/17.5% for Sonic during
those earlier 20-second windows (`radeontop -t10`); these are not F8 device-time measurements.

Every process exited 0 after completed F8/F9 output and with no observed foreign workload. All four
runs logged one graphics submit=-4 **after** the frontend's shutdown message, consistent with the
existing submit gate's intentional shutdown refusal; it is not a new measured gameplay failure.
The logs do not independently identify whether the gate or driver supplied that return code.

Main audio-output shortfall counters did not advance in any F8 window. Sonic on records two additional
shortfall callbacks around 333.37 seconds, after the scheduled F9 trigger; GTA's auxiliary output
remains a separate observed population. These are application demand shortages, not physical device
XRUN measurements, and no audio improvement is claimed.

## Ruled out

Attributing this captured Sonic slowdown to the second snapshot copy: both current F8 populations
record zero handoffs and zero handoff time (#3407). Other routes or earlier populations may differ.

## Remaining work

This closes only the redundant admitted-snapshot copy step. Guest reads, conversion work, buffer
materialization, compute result/writeback costs, graphics wait overhead and fresh-frame lineage
remain separate bottlenecks or measurement gaps. In particular, Sonic's zero target population
cannot justify another snapshot optimization for its current measured slowdown. Follow the broader
#3407 checklist and #3441 materialization work; preserve #1854's guest-publication contract before
attempting deferred writeback. Existing Linux VMA work in #3399 remains a separate lane.

## Capture identities

- `perf_capture_PPSA04263_20260912-231936-842.prperf` — SHA-256 `7a04ffe71d18b0452ba1940fa6edad39e85a80ceba81b9ee20cde240f01621d7`.
- `perf_capture_PPSA04263_20260912-232606-033.prperf` — SHA-256 `2680a006b6d2df6c60aee83d193a9ccae86f10f6a5adbb964edac0a13b0ec31d`.
- `perf_capture_PPSA03831_20260912-233230-319.prperf` — SHA-256 `40a75bcda0a11d9b807685f07ccf085be56f4490a826e07b65cc743c14484a22`.
- `perf_capture_PPSA03831_20260912-233947-534.prperf` — SHA-256 `0109b32928726342c46be2cd542274fc16dc244c4f070eb592cd7291b4f525d5`.
