# Retained renderer buffer inputs (September 2026)

Tracking: [#3407](https://github.com/mattias800/prosper/issues/3407).
Retention is opt-in on every platform. The measured cross-title results do not justify a default
policy; further tuning of this standalone-allocation approach is deferred.

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

The default byte budget is zero on every platform, including Linux with production write watches.
`PROSPER_BACKEND_BUFFER_RESIDENCY_MB` explicitly selects 0–2048 MiB on any platform. Zero bypasses
additional pass write-proof work. `PROSPER_NO_BACKEND_BUFFER_RESIDENCY=1` disables retention;
`PROSPER_NO_BACKEND_BUFFER_WRITE_WATCH=1` keeps exact-comparison retention but disables these watches.
No GPU vendor, title identity or experimental driver flag selects admission.

`PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS` selects a lower owner allowance (0–4096,
default 4096), still bounded by one sixteenth of the device allocation limit. It permits
comparing allocation populations at the same byte budget. Zero disables retention and
its additional pass write proof; use nonzero allowances to compare populations while
keeping that proof active. Changing this allowance also changes reuse opportunities,
so it does not isolate driver allocation costs by itself.

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
`scripts/sonic-frontiers-PPSA03831/reach-gameplay.pad`; Blue Prince uses IME auto-key with no pad script.
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

`render_buffer_residency_default` unsets the budget and disable switches, renders the actual
vertex-fetch pixel witness, and asserts ordinary uploads with no retained owners or byte charge.
The opt-in fixtures separately pin their budget so they continue exercising retention on all platforms.

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

## Bounded owner-population evaluation (September 12)

After incorporating main through `c7ecd2ad7031`, source `b785d0cd35dc2129775ae50a5fcb8173320d1bec`
was built and passed 485 tests plus 24 strict core/synchronization checks with zero validation messages.
Executable SHA-256: `abd1f53aa82cd6ad06d6b90afe33c49889fe1cd4c45c6a27d39243e63a1c675b`.
These captures precede the final default-off policy change; all enabled arms explicitly select their
budget. Main's pacing and gamepad-duration changes require these fresh controls rather than treating
the earlier measurements as current-main baselines.

All three GTA Performance-route arms use that binary, a 2048 MiB explicit budget, the same
route and capture inputs, fresh saves/caches, and the timing protocol above. The disabled arm sets
`PROSPER_NO_BACKEND_BUFFER_RESIDENCY=1`; the enabled arms set
`PROSPER_BACKEND_BUFFER_RESIDENCY_OWNERS` to 4096 or 256. Independent process censuses are clean,
both profilers finish before F8, and actual review of all three later F9 images shows the same intact
lit bank composition with character-pose variation.

| GTA arm | Presentations / measured seconds | Presentations/s | Renderer records | Residency ms | Cleanup ms | GPU wait ms | Last live owners / charged bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Disabled | 31 / 5.018502270 | 6.177 | 771 | 0 | 139.782 | 511.276 | 0 / 0 |
| 4096 owners | 26 / approximately 5.021 | 5.178 | 656 | 118.402 | 260.198 | 579.010 | 4096 / 309,927,488 |
| 256 owners | 30 / approximately 5.019 | 5.977 | 746 | 34.391 | 147.024 | 510.956 | 256 / 24,158,528 |

The 4096-owner run reuses only 3,294,720 bytes while declining 5,931,985,052 repeated input-span
bytes. The 256-owner run has zero hits/reused bytes, admits 3,284,992 bytes through copying, and
declines 6,760,842,468 repeated input-span bytes. Reducing the owner population removes most of the
observed penalty, but does not establish a useful cache benefit. The one-presentation difference
between disabled and 256 is not a statistically established 3.23% regression; neither is it evidence
for enabling the cache. Owner policy changes reuse opportunities and memory population together,
so this does not prove a specific driver-internal allocation cost. Last occupancy is sampled after
the timing window, not a transactional F8 snapshot. Fresh rendered-frame counts remain unknown.

The original disabled-run validator rejected 19 pre-trigger history samples instead of its hardcoded
20, despite all 21 post-trigger samples and complete renderer/compute records. Offline revalidation
preserves that original rejection and checks actual count consistency, sample continuity and measured
span instead. The measured rate uses the post-trigger endpoints. A known capture contaminated by
another worktree's builds/tests remains rejected by the revised validator. That earlier contaminated
`05ffda25` run is diagnostic evidence only, not an accepted timing comparison.

Decision: keep the tested retention mechanism and its counters available by explicit opt-in, preserve
the normal upload path by default, and stop speculative tuning of this cache. The earlier Blue Prince
benefit remains scoped evidence, not a library-wide recommendation. Remaining #3407 conversion and
writeback work should be prioritized by enclosing costs. This change does not claim an audio fix.

The independent audio audit keeps each sink and measurement interval separate. All three quiet/F8
windows have zero observed shortages. The main sink has one initial shortage in each of the disabled
and 256-owner runs; the 4096-owner run has two later shortages before F9 was requested, so they cannot
be attributed to that request. Its writer-plus-15-second interval is not fully observed. The intro
sink stops receiving deliveries around 15.18 seconds in all three runs and has no later output
coverage. These observations neither establish a hardware XRUN count nor explain the intro sink's
guest-production stop; that investigation remains under #3435.
