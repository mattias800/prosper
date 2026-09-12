# Retained renderer buffer inputs (September 2026)

Tracking: [#3407](https://github.com/mattias800/prosper/issues/3407).
The implementation is under evaluation. A larger automatic budget has not been accepted.

## Ownership and validation

The shipping graphics backend in `tests/fixtures/render_runner.h` can retain complete read-only
storage-buffer uploads across calls. Guest identity selects a candidate; current content establishes
reuse. Each owner holds a whole Vulkan buffer and an ordinary CPU snapshot, and uploads from that
snapshot. Comparisons never read a potentially uncached Vulkan mapping.

Admission requires complete write provenance for every final shader in the pass and no storage-buffer
writer. This whole-pass rule preserves existing aliases between draws. Descriptor tables, internal GDS,
small inputs, ambiguous identities and writable passes keep their ordinary upload paths. Hosted and
copied inputs are validated against their current materialization, not against an old guest address.

A changed idle owner can be refreshed. A changed recorded or in-flight owner is detached and remains
alive through its submission's completion lease; a replacement cannot overwrite it. Byte accounting
includes both actual Vulkan allocation size and the CPU snapshot until the final owner releases them.
A separate live-owner allowance is the smaller of 4096 and one sixteenth of the device's
`maxMemoryAllocationCount`. It includes detached owners and applies with explicit byte overrides.
This bounds the added cache population; device-wide accounting for other allocations remains
[#3491](https://github.com/mattias800/prosper/issues/3491).

At capacity, a bounded search can rekey an idle exact-size owner without allocating another Vulkan
buffer. Each request inspects at most 32 entries, continuing from the prior search rather than becoming
stuck behind an incompatible prefix. Inspection preserves LRU recency. A key cursor can safely disappear
through source replacement or cache clearing. An incompatible miss keeps ordinary uploads.

Only explicitly identified direct guest views can use Linux production write watches. Two exact equal
validations precede promotion, and arming happens before the next authoritative comparison. Alias writes,
host/GPU notifications and remapping invalidate watch authority. Failed coverage falls back to exact
comparison; repeatedly dirty entries stop watching until their source changes or they are rekeyed.
GPU owners never retain a guest pointer or a guest mapping lease.

The current byte budget is 256 MiB on Linux and zero on platforms without production write watches.
`PROSPER_BACKEND_BUFFER_RESIDENCY_MB` explicitly selects 0–2048 MiB on any platform. Zero bypasses
additional pass write-proof work. `PROSPER_NO_BACKEND_BUFFER_RESIDENCY=1` disables retention;
`PROSPER_NO_BACKEND_BUFFER_WRITE_WATCH=1` keeps exact-comparison retention but disables these watches.
No GPU vendor, title identity or experimental driver flag selects admission.

## Measurements and their limits

These same-binary captures use `ae90fca570e3d08e8b0076d5a3c6ded3f815ded5`, before the live-owner
allowance and advancing-search changes. They do not measure those later fixes. Executable SHA-256:
`b01f36fb827c2bb26a2db5be500bf2a12017deeb4c0a0708de41121c3f6a27e4`.

| Title and control | Guest/host presentations per second | Renderer records | Renderer ms | Ordinary buffer-copy ms |
| --- | ---: | ---: | ---: | ---: |
| Blue Prince, 256 MiB | 11.542 | 60 | 3555.561 | 991.502 |
| Blue Prince, explicit 2048 MiB | 13.545 | 70 | 3199.433 | 284.467 |
| Blue Prince, explicit 2048 MiB repeat | 13.342 | 68 | 3137.163 | 273.958 |
| GTA Performance, retention disabled | 5.976 | 784 | 2681.244 | 156.212 |
| GTA Performance, explicit 2048 MiB | 5.177 | 677 | 2683.999 | 132.612 |
| GTA Performance, 2048 MiB with buffer watches disabled | 5.175 | 675 | 2704.254 | 133.074 |
| Sonic, retention disabled | 7.382 | 596 | 1966.187 | 213.101 |
| Sonic, explicit 2048 MiB | 6.383 | 528 | 2014.026 | 143.345 |

Every F8 has complete records and lasts approximately five seconds. Counts and rates describe guest
flips and successful host presentations; newly rendered frames remain unknown. These are different
work populations, so lower raw copy totals cannot establish a speedup. Blue Prince's larger-budget
runs preserve its centered entrance hall and add approximately 727/715 MiB of observed RSS. GTA keeps
its lit bank scene but shows no residency hits, admissions or reuse in either enabled F8. Sonic reuses
2.506 GB of repeated spans through watches, yet its enclosing renderer and compute costs increase.
Its F9 retains the previously documented black-world gameplay HUD at different stage times.

Launches use fresh saves and shader caches, native resolution, normal full cadence and immediate
presentation. GTA uses `scripts/gta5/reach-performance-story.pad`; Sonic uses
`scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`; Blue Prince uses its automatic opening route.
Set `PROSPER_PERF_CAPTURE_AFTER_MS=300000`; F9 uses `PROSPER_GRAB_BUNDLE_AFTER_MS=310000` for GTA
and `330000` for Sonic/Blue Prince. Capture durations are 360/380/420 seconds respectively. Separate
199 Hz CPU and coarse GPU sampling runs before F8, around 250–270 seconds. F9 is later than F8 and
cannot prove identical content in the earlier performance window. Copied executable identities,
source revisions, environment differences and process isolation are checked per arm.

F8 reports actual upload, compared, reused, admitted, refreshed, watched and refused spans. These
are repeated logical byte counts, not unique working-set memory or physical memory traffic. Watch
time is already inside residency time; watched bytes are already inside reused bytes. Timing logs
add sampled cache occupancy, with availability explicit before device initialization. Cleanup can
change owner/byte gauges between reads; they are not one transactional snapshot.

Audio is independently checked per output/generation. These runs do not establish an audio fix or
physical hardware XRUN counts. In particular, neither Sonic arm covers a full writer-plus-15-second
tail, and startup/intro-output shortages must not be mixed with steady gameplay demand.

## Ruled out

- **HOST_COHERENT upload memory is necessarily cheap to compare on the CPU.** The first retained
  candidate (`30d8e611`) spent 56.87% of sampled weighted cycles in `memcmp`, with 3517.344 ms of
  residency work inside a five-second F8, and reached a black Blue Prince checkpoint. It was rejected.
  An owned CPU snapshot replaces mapped-memory comparison. A fixture makes the stored mapping
  inaccessible during equality lookup, and restoring the old comparison fails that guard. #3407.
- **Lower copying alone proves exact-snapshot residency improves performance.** The `94e2110f`
  Blue Prince enabled/disabled pair had 6.27% higher enclosing renderer cost per record with residency,
  despite less copying. The 2048 MiB Sonic pair above also improves buffer costs while slowing the
  enclosing workload. Neither result is discarded. #3407.
- **New renderer buffer watches are necessary for the observed GTA slowdown.** With only
  `PROSPER_NO_BACKEND_BUFFER_WRITE_WATCH=1` changed at the same 2048 MiB budget, the frozen GTA
  binary still records 26 presentations and about 5.18/s. Elevated cleanup/readback costs remain.
  This does not exonerate all shared watch costs or explain the retention slowdown. #3407.

## Regression guards

`render_buffer_residency` uses actual Vulkan vertex fetches to distinguish stale data from valid reuse,
including hosted replacement, direct-memory aliases, host/GPU write notifications, same-address remaps,
queued old/new versions, rounded allocation pressure, discard and idle rekey. Small synthetic allocation
allowances exercise zero-cap refusal, detached-owner accounting and bounded-search progress without
approaching a real device limit. An incompatible prefix must not hide an eligible actual Vulkan buffer;
a pinned candidate must keep its old snapshot. A cold statistics query must not initialize Vulkan.

`render_buffer_capture` exercises the actual live renderer's ordered graphics spans and F8 accumulation.
It guards counters and timer scope through the semantic record, rather than testing only serialization.

```sh
cmake --build <BUILD> --target test_render_buffer_residency test_render_buffer_capture
ctest --test-dir <BUILD> --no-tests=error --output-on-failure \
  -R '^(render_buffer_residency|render_buffer_capture)$'
python3 prosper/tools/vkval/vk_validation_scan.py --build-dir <BUILD> \
  --probe test_render_buffer_residency --sync --allowlist /dev/null \
  --ctest-arg=-R '--ctest-arg=^(render_buffer_residency|render_buffer_capture)$' \
  --ctest-arg=--no-tests=error
```
