# `frontends/shared/live/` — the two live Vulkan backends

What turns a decoded guest submit into real GPU work at runtime, shared by every frontend
(`prosper-app`, `boot_trace`, `tools/screenshot`) so they all drive the *same* backend rather than
three that drift. Built as the `prosper_live_renderer` static library, which exists only where CMake
found Vulkan.

- `live_renderer.cpp` — the graphics half: DrawItem → Vulkan compositor, render-target/RTT authority,
  present. It is a *driver* for the offscreen backend rather than the backend itself; the pipeline,
  pass and readback code it calls lives in `tests/fixtures/render_runner.h` (whose directory name is
  a trap — see that folder's `AGENTS.md`).
- `live_compute.cpp` — the compute half, and a **separate Vulkan backend**, not a caller of the
  render one. It builds its own device (or adopts the renderer's when one is published), its own
  pipeline cache, descriptor pools, memory pool and command buffers, and does not include
  `render_runner.h` at all. Reflected storage buffers and storage images are materialized from guest
  memory, dispatched, and written back into guest memory synchronously.
- `gpu_retile.hpp` — exact 2D and standard 3D storage-image writeback layout conversion in the
  guest compute submission. 3D SW_4KB_S/SW_64KB_S uses a separate pipeline variant and checks
  padded XYZ dispatch limits. A packed mode-24/27 variant groups four native byte texels or two
  native halfword texels into ordinary 32-bit stores across independent 2D planes/array layers.
  Mode-24 bytes interleave rows and retain CPU conversion.
  The selected equation must prove horizontal word ownership; no 8/16-bit storage feature is needed.
  Current multi-layer codegen exposes native Uint16; RG8/R16F arrays retain raw interchange storage
  and CPU conversion. Incomplete source words, ambiguous strides/mips and other sub-word layouts
  retain CPU conversion. Linear comparison baselines stay separate from the tiled host-read buffer.
  Typed R32_UINT/RGBA8_UINT image reads are fused by default with
  both tiled and exact linear output in one submission, including one-layer array views. This is
  a checked core storage-image path. `PROSPER_NO_DIRECT_IMAGE_RETILE=1` restores transfer plus
  retile for comparison without changing publication semantics. Other representations retain
  transfer plus retile. The linear allocation remains required by current/future result baselines,
  CPU fallback and diagnostics; a cold cache miss does not prove it unnecessary. Its producer is
  now a shader write, including for host availability, comparison and baseline-copy dependencies.
  Image mirroring and borrowed-image layout restoration remain independent obligations.
  Eligible retained 2D word outputs also compare/adopt their exact **linear** baseline inside
  retile. Padding is written deterministically but excluded from equality. Shared flags clear
  before retile, and fused targets omit the separate comparison dispatch. The control
  `PROSPER_NO_FUSED_RETILE_COMPARE=1` restores separate comparison. Optional setup refusal retains
  ordinary GPU retile, then image transfer plus CPU tiling if its binding also fails; device loss
  remains fatal. Owners, pins, content invalidation and guest publication are unchanged.
  `gpu_retile_ms` includes fused comparison when selected; compare total storage/dispatch time
  across policies instead of treating a smaller `gpu_compare_ms` alone as savings. Render timing's
  `fused_compare_recorded` is a cumulative command-recording count, not completed dispatches.
  `PROSPER_NO_GPU_RETILE_PACKED_EXTENSION` restores the prior mode-24 multilayer-only packed scope
  for comparable runs while leaving ordinary word and volume GPU tiling enabled.
  `PROSPER_GPU_RETILE_CENSUS=1` reports, per dispatching program, how many storage images this
  admission accepted and how many it declined **for which named reason**, with one representative
  shape per reason. The admission has a dozen `continue`s; before the census, "the CPU layout path is
  hot" and "which condition sent it there" were separate questions with only the first answerable.
- `packed_rtt_conversion.hpp` — device-owned RGBA8→packed-10-bit sampled conversion.
  Records transfers and conversion into the guest compute submission; setup failures retain their
  `VkResult` so optional fallback cannot hide device loss.
- `live_target_format.hpp` — the guest↔Vulkan pixel-format mapping. Compiled with `-Werror=switch`
  on purpose: a silent RGBA8 fallback has cost two titles a whole render layer.
- `decode_scratch.hpp` — the pooled full-surface intermediates that both the texture decode branches
  in `live_renderer.cpp` and the per-dispatch seed/pack buffers in `live_compute.cpp` stage through
  (`ScratchBuffer` is the drop-in shape for the latter's old `std::unique_ptr<uint8_t[]>`).
  Header-only and Vulkan-free. It is here rather than in
  `shared/texture/` because that folder holds *decision* logic that owns no memory, and this owns
  the memory. The one rule it exists to make explicit: a lease arrives holding the PREVIOUS
  surface, so a caller that fills it partially must say so (`zero_tail`) — a fresh
  `std::vector<uint8_t>(n, 0)` used to supply that tail for free, and nothing else would notice it
  had stopped. `PROSPER_DECODE_SCRATCH_POISON=1` is the arm that makes such a caller fail loudly
  instead: it fills every lease before handing it out, so nothing can quietly inherit the zeros a
  fresh mmap used to supply.

`texture_source_snapshot.hpp` owns the encoded validation snapshot handoff at persistent texture
cache admission. Guest reads, admission/accounting, write watches and decoded pixel retirement
stay in the renderer. Oversized global scratch retains the copy path rather than increasing an
entry's capacity beyond its outgoing allocation or exact readable prefix. The same-binary
`PROSPER_NO_TEXTURE_SOURCE_SNAPSHOT_MOVE` control restores copying independently of prefix
allocation inheritance. F8 handoff time is nested in frontend texture time; copied/transferred
bytes describe actual admitted snapshots and exclude the guest read and GPU uploads.

Queued renderer invalidation keeps writes ordered, including their captured origins. Multi-write
drains prepare CPU RTT footprints only for the lifetime of that drain; metadata association must
precede preparation, and erased entries cannot be revisited. Single writes and RTT-watch diagnostics
retain ordinary traversal. `PROSPER_NO_RTT_WRITE_BATCH` selects that control at startup.
`PROSPER_WRITE_DRAIN_CENSUS=1` enables recurring per-thread work counts without changing execution
policy. Logical live-entry visits and physical prepared-record visits are separate; counters are
not timings, and overflow's wholesale invalidation is excluded from traversal totals.

`pass_buffer_lookup_memory.hpp` supplies strictly per-pass CPU allocation storage for the backend's
buffer-reference memo and content-dedup index. Keys, equality, slot memoization and Vulkan upload
owners are unchanged. Four KiB is initial storage, not a cap; larger passes use the ordinary heap,
and replaced hash buckets remain until arena destruction. Every map dies before its arena, and
the arena never enters submission cleanup callbacks. `PROSPER_NO_BUFFER_LOOKUP_ARENA` selects
direct PMR heap allocation; unsupported PMR deployment targets retain ordinary standard maps.
Allocation observations count upstream heap requests/bytes, not nodes or GPU buffer traffic.

## The boundary that is easy to get wrong

The compute backend is where *guest memory* and *device memory* meet twice per dispatch — an upload
before, a writeback after — and both directions have a synchronization contract that no output check
can verify:

- **A host READ of anything the device wrote needs `record_host_read_barrier()`**
  (`src/gpu/execute/host_read_barrier.hpp`). Waiting the dispatch fence orders execution; it does not
  perform the availability operation into the host domain. Every writable storage buffer, every
  storage image's staging buffer, the comparator's flag word and every retained result baseline is
  device-written host-visible memory, so every one of them carries the dependency (#3249).
- **The scope is the ALLOCATION, not the binding.** `allocate_memory`/`release_memory` run a pool
  that recycles host-visible allocations across bindings and submits, and a later binding maps a
  recycled allocation and *reads its retained contents* to decide whether an upload is needed. So a
  buffer this dispatch never maps still needs the barrier if the device wrote it.
- **Neither can be checked by asserting the result.** Every allocation here is `HOST_COHERENT` (both
  tiers of `host_memory_type` require it), and on the drivers this project runs on the unsynchronized
  code returns byte-identical results. The guard is `live_compute_host_read_barrier`, which asserts
  the barrier was *recorded*; synchronization validation cannot see this class either (#3248).

`execute_live_compute_items()` is the one entry point exposed for tests, which is how
`game_compute_exec`, `live_compute_descriptor_array` and `live_compute_host_read_barrier` drive the
production backend without a frontend or a game dump.

## Measuring buffer setup and writeback

The persistent buffer budget defaults to 512 MiB and charges primary allocations and optional
exact-result baselines. `PROSPER_COMPUTE_BUFFER_CACHE_MB` overrides it; zero disables retention.
Primary admission first proves that enough unpinned storage can be reclaimed, then drops unpinned
baselines before evicting primary entries. Baseline admission uses spare capacity or owners that
have been unused for at least 256 cache insertions/acquisitions; it proves enough eligible capacity
exists before any reclamation. Recent and pinned owners remain protected. The diagnostic control
`PROSPER_NO_IDLE_COMPUTE_BUFFER_RECLAIM=1` restores spare-capacity-only baseline admission. One owner
pin protects both handles through completion; a failed submission without completion proof keeps
that pin. Reclaiming a baseline preserves the primary's guest-content validation and write watches.
Unchanged writeback preserves watch-promotion progress already earned by exact source validation;
it neither resets that progress nor counts it a second time. Changed content resets progress.
`compute_result_compare_group_count()` checks the real device's storage-descriptor range and
X-dispatch limit before GPU baseline retention or comparison. Unsupported extents keep exact CPU
comparison; otherwise eligible storage images rejected by the device-limit guard retain the
current host snapshot instead of transferring GPU ownership.

`PROSPER_COMPUTE_BUFFER_TIMING=1` emits `[compute-buffer-timing]` records after cleanup. Use
`PROSPER_COMPUTE_TIMING_CODE` / `PROSPER_COMPUTE_TIMING_HASH` to select a program and
`PROSPER_COMPUTE_TIMING_CAPTURE_ONLY=1` to bound records to F8; the existing `TRACE_ONLY` selector
also applies. No guest-content hashes or extra comparisons are added, and the added timers read no
clocks when disabled. Keep `PROSPER_COMPUTELOG` off when measuring: its additional scans and output
are included in enclosing timers.

Each resolved allocation owner gets one row, including read-only owners. `owner-index` identifies
the flattened descriptor-table entry; `bindings` includes the owner and exact aliases, and `aliases`
counts only the additional entries. Writability is combined across aliases. On setup failure, later
descriptors may have `owner-resolved=0`; exclude those placeholders from allocation counts. Join
records using submit/dispatch/order/program identity. CPU fast paths, proven no-ops and declines
before materialization produce no buffer rows; these records are not the whole F8 population.

`cache`, `validation`, `baseline`, `gpu-compare` and `writeback` report decisions already made by the
backend. `unobserved` means the stage was not reached. GPU comparison progresses through eligible,
prepared, recorded and a completed changed/unchanged result; `no-result` means completion occurred
but the flag was not read, and `setup-failed` retains the CPU fallback. Check `ok` before treating a
row as a completed dispatch. `host-key` is the existing cache-key pointer, whereas `host-backed`
describes effective source selection (a short host mirror falls back to guest memory).

`cache-bytes` and `cache-limit` report end-of-dispatch requested residency. The owner's
`primary-allocation-bytes` and `result-allocation-bytes` report the surviving cache entry's separate
charges (zero if absent). These use Vulkan resource memory requirements, as the existing budget
does; the underlying allocation pool can supply a larger allocation. They are not a physical
device-memory usage measurement or cap.

Byte counters are lengths passed to comparisons and copies, not actual CPU bytes read by an
early-exiting comparison. `compared-bytes` includes the initial exact comparison against pooled
allocation contents. `upload-skipped` is the existing cached-source authority bit: zero upload bytes
on a cold, already-equal pooled allocation does not set it. Atomic image writeback uses
`guest_layout_ms`; `guest-copied-bytes` counts only the ordinary contiguous buffer copy.

`layout_ms` is the one timer here whose name does not describe what it usually measures. It times the
whole publication branch, and the branch has two arms that are different work: a GPU-retiled result is
memcpy'd out of the retile buffer, a declined one is tiled on the CPU. `retile_copy_ms` is a strict
subset of `layout_ms` covering only the memcpy, so `layout_ms - retile_copy_ms` is the CPU tiling
residual — measured 2026-09-15 at 6,551 / 6,541 / ~10 ms on a Sonic Frontiers route, i.e. the name is
misleading by three orders of magnitude on that title. `tools/perf/compute_phase_report.py` models the
copy as layout's child, so the whole branch no longer reads as a targetable leaf. On those figures the
residual does not print at all: that report gates `unattributed` at 1% of its parent and 10 / 6,551 is
0.15%. An absent `unattributed` row under layout therefore means the residual is under ~65 ms on that
route, not that it is zero.

The GPU result-compare ceiling (`max_gpu_compare_image_bytes`) is **derived from the memory
topology, not fixed**: an integrated device gets 128 MiB, everything else keeps 2 MiB. That is not a
performance guess, it is what decides whether the comparison's reads cross a bus — the 2 MiB bound
came from a PCIe measurement on a discrete GPU (#3275), where reading 33-66 MB of staging back per
writeback cost 1,138 ms per window. On unified memory the same comparison reads memory the GPU
already owns, and the bound instead disables the mechanism for every 4K target, which is every
compute target that matters. Above the ceiling there is no comparison and every writeback publishes
unconditionally; below it, an unchanged result skips the map, the CPU retile copy, the layout pass
and the write-watch notify. `PROSPER_MAX_GPU_COMPARE_IMAGE_MB` overrides the derivation; a malformed
value keeps the derived ceiling rather than silently selecting the discrete one.

One consequence of admitting larger results, worth knowing before trusting the skip's proof at every
size: `cold_storage_result_snapshot_can_defer` stops retaining a host source snapshot for write-only
storage images at or above its minimum (16 MiB by default), and `acquire_cached_image`'s exact-memcmp
leg needs that snapshot. So for images between that minimum and the ceiling, `upload_skipped` rests on
the GPU-write journal and the page write-watch alone rather than on all three legs. Those two legs
already carried the upload decision at every size and are unchanged — but if the watch ever misses a
guest write, the failure mode for this population is a STALE RESULT left in guest memory rather than a
fresh result computed from stale input, which is the harder one to notice.

Timers are nested, not additive: `setup_ms` covers owner materialization after resource/alias checks;
`validation_ms` includes upload compare/copy/map/watch on cache hits. `writeback_ms` encloses result
comparison, guest copy/layout, map, host-write-watch notification, baseline creation, architectural
notification, source validation/watch rearming and provenance. Those last two have separate
`source_validation_ms` and `provenance_ms` fields. Use the existing dispatch phase totals for the
remaining setup checks and loop overhead; do not claim the sum of owner timers covers all setup.

`PROSPER_COMPUTE_BUFFER_CACHE_CENSUS=1` additionally snapshots up to 256 complete cache keys
when buffer timing is selected. `compute-buffer-cache` states total/emitted entry counts and
whether the snapshot is complete; `compute-buffer-cache-owner` rows include separate primary and
baseline charges, content validity, pins before cleanup, and the existing last-use cache clock.
The clock counts primary insertions/acquisitions, including failed validations; it is neither time
nor a completed-dispatch count. Compare full keys and last-use values across snapshots to observe
reuse. Pins include this dispatch's owners and prohibit reclamation. Census collection adds work
inside cleanup timing; use it to diagnose residency, disable it for throughput comparisons.

## CPU fill admission

The exact `FillSgprUvec4` shortcut validates the live descriptor and launch before preparing any
write, and completes its prepared host-write scope after the store. Its semantic fill proof can
survive a CPU permission refusal, but does not itself authorize memory access or cached reuse.
Ordinary architectural clear semantics remain required even when all destination bytes already
equal the fill pattern. `cached_compute_fill` exercises this distinction against rendered depth.

`PROSPER_COMPUTE_FAST_PATH_CENSUS=1` reports the first admission/refusal reason for this shortcut
only while F8 detailed timing is active. Broadcast-buffer shortcuts have a separate implementation
and are not included in this reason partition. The census reads no additional guest contents.

Complete fill patterns can be reused on an existing compute-buffer owner by default.
`PROSPER_NO_CACHED_COMPUTE_FILL=1` disables pattern recording and reuse for comparisons.
Full ordinary direct backing, matching emitted extent and current acquired
source authority are required. The pattern is cleared before source refresh or writable dispatch
and published only after all result writebacks succeed. Partial/hosted/tail bindings and applied
shader overrides are excluded. No additional guest-data cache is created. A `cached-fill` buffer
timing row identifies an actual skipped submission; its GPU comparison is `not-recorded`.

Known fills use an explicit clear origin for unchanged GPU writeback and cached reuse, including
when reuse is disabled. A partial fill clears only its written prefix; its untouched tail retains
the ordinary unchanged-output notification. Guest-byte
watches/journal remain valid, while renderer aliases still receive the semantic clear. Generic
refresh operations retain their existing preservation classification. The production fixture also
checks complete-pattern reuse, intervening writers, alias mutations, failed publication, partial
and mismatched shader extents, and a targeted shader replacement followed by an original shader.

## Read-only storage images

Storage descriptor class does not imply an output. Omit output staging, masks, retile, comparisons,
writeback and output authority only when the entire exact alias group is readable, non-writable
and non-atomic, and reflection proves it accounted for every image store in the module. Unknown
image-write provenance or any image texel pointer keeps the conservative output path.
`PROSPER_NO_READONLY_STORAGE_WRITEBACK_SKIP=1` restores that path for comparisons.

A native read-only input may retain validated input content after completion, before other guest
outputs invalidate it. A changed input must discard both CPU and GPU result baselines from previous
writers: otherwise writer A, read-only upload B, writer A can incorrectly leave B in guest memory.
Read-only input retention never publishes a new producer result or resets DCC metadata. The
`storage_readonly` execution test checks bytes, dependent shader reads, architectural notifications,
recorded host dependencies, alias writers, unknown write provenance, and both baseline kinds.

`PROSPER_COMPUTE_IMAGE_TIMING=1` enables `compute-storage-access` records; add
`PROSPER_COMPUTE_TIMING_CAPTURE_ONLY=1` to limit them to F8. They include whole-group access, proof,
writeback obligation and actual staging bytes for each unique materialized storage owner, including owners whose output
comparison later finds an identical result. These are preparation records, not completion or timing
measurements. Join submit/dispatch/order/program identity with completed dispatch records.

## Overlapping output authority

An allocation pin proves lifetime, not current guest content. Build the complete output plan before
recording, following actual writeback order: all unique buffers, then each unique image's pixels and
DCC reset. Earlier pixel or metadata writes veto setup-time equality skips. Later writes to either
dependency, and a reset overlapping its own pixels, veto final retention/publication and revoke
source/consumer authority before recording without releasing handles or pins. Failed completion
skips cleanup. Earlier-only image conflicts keep their source proof; ordinary pre-submit export
revocation and successful writeback govern their next publication. The final writer may retain its
complete result. Buffer GPU comparisons require no earlier overlapping buffer; their CPU fallback
checks current destination bytes. Include effective hosted destinations even for zero-address
private outputs; an absent architectural address contributes no architectural range. Private
outputs still emit no guest notification. The same effective overlap vetoes a later storage image's
unchanged-result skip; input-only image authority is a separate dependency contract.

Input-only retention precedes guest writeback, so later pixel notifications invalidate its original
snapshots naturally. Metadata-only writes require an explicit retention veto because pixel authority
does not cover interpretation metadata. Keep sampled metadata dependencies separate from storage
metadata reset obligations. Include effective hosted and advertised ranges, and fold exact aliases.

Later dispatches can change a separate metadata plane without touching a retained image's pixel
allocation. Graphics imports and compute transfer seeds therefore recheck the consumer's complete
supported DCC plane immediately before borrowing. Unknown metadata kinds, unsupported footprints,
unreadable or truncated planes and non-plain contents decline the borrow. A supplied hosted plane
never falls back to guest metadata. This fresh check applies to numeric aliases too; neither an
earlier all-0xff scan nor the pixel journal proves current metadata. Refusal preserves outstanding
leases and the existing materialization fallback. `storage_metadata_borrow` guards this boundary.

Zero-address internal backing (such as GDS) is not an architectural guest output.
Notify actual hosted guest destinations as well as advertised architectural ranges, including for cached
views absent from a dispatch. Prepare page watches before every actual mutation. The production
`storage_output_conflicts` tests and companion journal/failure/watch cases guard this contract.

## Untouched storage pixels

A previous dispatch's coverage cannot authorize discarding current inputs. Native images retain
exact texels; raw images may retain extra channel precision only for wholly write-only alias groups
whose packed result still matches validated current guest bytes. A source-validation miss requires
an upload. Readable/atomic raw groups need canonical inputs instead.

Raw formats whose seed conversion is non-injective need exact per-store masks. The SPIR-V helper
in `shared/compute/storage_write_mask_spirv.hpp` adds ordinary host-backed storage-buffer bindings;
those buffers follow the same upload, completion, host-read-barrier and writeback contract as guest
buffers. Their completed masks select which original linear guest texels survive packing. Such
repaired raw images cannot advertise ordinary retained packed-content authority. Descriptor binding
allocation includes unreferenced resource-table entries and unused SPIR-V globals.

Exact native RGBA8 storage can seed from a separately pinned renderer image when no earlier sampled
binding supplies one. Admission requires actual Vulkan format, extent, device and transfer-source
usage to match; incompatible views retain CPU snapshots. The borrowed image stays read-only and
returns to its incoming layout (or an earlier sampled borrower's GENERAL layout); the private
destination still writes back guest bytes. Release every acquired pin, including folded aliases,
only after completion is established. `PROSPER_NO_STANDALONE_RTT_SEED` selects the CPU control;
F8-gated `compute-rtt-seed` rows distinguish admission from recorded copies.

A completed private renderer-seeded native RGBA8 image may be retained under the existing image
budget for validated storage-to-sampled transfers. This new result does not authorize graphics
export or create its page watch: Linux uses the ordered journal and declines outside its authority;
Windows retains its exact guest mirror. Ordinary graphics preparation remains the fallback. Input cache
reuse remains excluded for renderer owners: their next dispatch acquires the current renderer image.
Retain only after ordinary guest/DCC writeback; publish authority only after every writeback succeeds.
Refuse a result if a later independent output changes its pixels or interpretation metadata, or its
own metadata reset changes its pixels. Include effective host backing and advertised ranges. Exact folded aliases
share one owner. Pinned replacement refusal and failure invalidation retain their existing contracts.
No separate comparison baseline is created for this consumer source. The control
`PROSPER_NO_RENDERER_SEEDED_RESULT_CACHE=1` disables this promotion; F8 writeback rows report
`renderer-result-retained` separately from input cache hits.

## Texture validation census

`PROSPER_TEXTURE_VALIDATION_CENSUS=1` counts validation invocations in callbacks that observe
F8 detailed timing active. It adds no guest reads or extra clocks and does not change realization
parallelism. Fixed buckets partition exact matches, exact failures (including short/unavailable
sources), and untimed watch-only refusals by watch reason and declared source-size band.

Reports are cumulative per thread across all observed timing intervals. `timing-inactive` means a
later callback observed timing off; `thread-exit` can flush a still-active interval. Neither proves
a complete F8 window or whole-process coverage. Sum bucket calls against the matching header.
Watch queries precede rearming; active flags follow it; stability precedes this validation's update.
All count acquisition observations, not distinct entries. Reported validated bytes retain the
helper's platform-dependent extent semantics, not physical read traffic. Refusals have no measured
comparison work; their zero timer does not mean a zero-cost acquisition. Timers exclude the census
update and remain nested in existing texture preparation.
The thread identifier is the POSIX `pthread_self()` token or Windows thread ID; do not join POSIX
values directly to a native profiler's operating-system TIDs.
