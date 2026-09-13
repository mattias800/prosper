# FP16 texture and storage preparation (#3407)

This change extends the existing GPU cube detiler to ordinary single-mip 2D RG16F/RGBA16F
resources. Sampled images retain exact half-to-UNORM8 conversion. Portable storage images use
four UINT channels carrying exact binary32 representations of the guest binary16 values.
Writable images retain the existing completion-gated guest writeback and invalidation path.

A same-binary GTA pair reduces the affected CPU texture bucket from 0.611 to 0.257 ms per
reference. Host presentation rates remain effectively unchanged; no end-to-end FPS gain is established.

## Scope and ownership

- Admit tile27, two or four FP16 components, one 2D layer and mip, no compression or mip tail,
  and no competing retained target or compute-image owner. Preserve established persistent
  sampled-texture caching rather than replacing a reusable decode with repeated conversion.
- Copy complete padded guest source blocks into an immutable submission-owned mapping.
  Cache the four shader/pipeline variants per device. Their numeric operations are integer-only;
  no optional half arithmetic or experimental driver flag is required.
- Check source/output descriptor ranges, compute limits, dimensions and exact Vulkan image usage.
  Unsupported layouts, short source backing or failed allocation retain CPU fallback.
- Writable storage owns a separate mapped result lease. Backend image-to-buffer copy and completed
  callbacks retain authority over guest writes. Cleanup cannot recycle that mapping while a
  snapshot owner remains alive. Read-only and writable decode identities remain distinct.
- When backend texture sharing is disabled, writable GPU preparation and backend admission both
  decline: separate images must not overwrite one owner's result mapping. Read-only GPU conversion
  remains supported in that diagnostic configuration.

`PROSPER_NO_GPU_DETILE_2D=1` restores CPU conversion for the new 2D population while preserving
the pre-existing cube path. The existing global GPU-detile and layout diagnostics retain their
fallback behavior. There are no title, guest-address or shader-hash admission rules.

## Measurement method

Normal windowed Linux prosper-app, immediate presentation with FPS overlay, AMD Radeon 8060S/RADV,
native resolution and full cadence. Each run starts with fresh save and driver/application caches.
GTA uses the committed Performance Story gamepad sequence. CPU/GPU profiles run at 250 seconds,
F8 records at 300 seconds and F9 follows the timing window. Source, route and executable hashes
are checked before and after; a process census rejects competing builds, games and profilers.

The initial sampled-only implementation at 01c6d0aa9 recorded **zero** GPU preparations in GTA.
F8 had identified a 1280x720 RG16F tile27 resource, but that shape alone did not establish its
binding class. Reading the F9 bundle manifests and reflected descriptor contract showed a writable
storage image. That observation justified the raw UINT ABI extension; the sampled-only run is
not evidence of a GPU-detile speedup.

The F8 fields `frontend_gpu_detile_preparations`, `frontend_gpu_detile_2d_preparations` and
`frontend_gpu_detile_source_bytes` count successful frontend snapshot preparations and padded input
bytes, including the 80-byte header. They do not count dispatches, GPU-completed conversions or
new frames. Older/partially populated captures remain unavailable; real zero counts remain zero.

The report's graphics, compute, resource, wait and device scopes are nested, not additive frame
budgets. Host presentations can repeat; newly rendered FPS remains unavailable. Per-output audio
demand shortfalls are not physical device XRUN measurements. This change does not alter audio
production, buffering or channel ownership.

## Validation

All 517 CTests pass. Seven focused GPU-detile tests also pass strict Vulkan synchronization
validation with an empty allowlist and zero messages; positive controls prove validation loaded
and synchronization checking armed. Representative emitted shader variants pass SPIR-V validation. The offline performance reporter passes 45 Python tests.

Guards cover all 65,536 half encodings in both component counts and output representations,
partial tiles, tail invocations, actual frontend GPU/CPU controls, independent sampled/storage
semantics, writable guest updates, untouched texels, changed-address contents, short backing,
unsupported layout fallback, unshared-image rejection and result-mapping lifetime. Existing F8
split-span ownership carries the counters; recorder/reporter tests preserve missing/zero values
and exact large integer byte counts. Independent read-only reviews covered conversion, ownership,
fallbacks and measurement semantics.

## Ruled out: reusable CPU copy workers

A bounded semaphore worker-pool candidate was tested before integration using the renderer's actual
worker-count policy. Hot repeated buffers sometimes improved, but streaming copies over a 128 MiB
working set did not: at 2/4/8/32 MiB, fresh workers measured roughly .087/.129/.226/.663 ms versus
.097/.159/.297/1.163 ms for the reused pool. These observations reject that implementation and
policy, not all pools. No pool code ships. The separate September 8 reusable comparison-worker
experiment was also rejected; do not confuse the two or repeat them without a new hypothesis.

## Remaining work

Source copying and graphics storage result writeback remain. Large renderer buffer uploads,
retained texture refreshes and synchronous compute/writeback costs still need measured ownership
work. A possible compute-buffer-to-renderer bridge requires exact byte-range/materialization
identity, successful producer completion, guest-authority checks and protection from mutation:
existing compute-cache pins prevent eviction but do not prevent an in-place refresh.
Capture dependency overlap alone is insufficient proof of a safe borrow. Generic renderer buffer
residency remains disabled because its prior comparisons/watch policy regressed measured runs.

## Observed results

The paired GTA runs and both compatibility runs use source
`5603a8821ed168ca0a4fe04a2473db5cc9943049`, including main `3c4e4db9a2e5`, and one executable
with SHA-256 `61dc303b709a107e1590987cc1e65436bea2256678d180203a7dbfa50a2b7d5a`.
The final reporter-only change exposes resource class, image dimension, depth comparison and hosted
backing already present in those raw captures; it changes no renderer behavior. The summary now
identifies the GTA witness as `StorageImage`. Missing older identity fields remain unavailable.

| GTA same-binary run | GPU conversion | CPU control |
|---|---:|---:|
| F8 seconds | 5.019 | 4.770 |
| Host presentations/s | 6.375 | 6.290 |
| Guest flips/s | 6.375 | 6.080 |
| Renderer / compute records | 825 / 2217 | 780 / 2085 |
| New 2D GPU preparations | 33 | 0 |
| Padded GPU input bytes, including headers | 129763920 | 0 |
| Unclassified texture references | 99 | 94 |
| Unclassified texture time | 25.442 ms | 57.468 ms |
| Time per unclassified reference | 0.257 ms | 0.611 ms |
| All frontend texture preparation | 652.921 ms | 665.952 ms |
| Renderer resource preparation | 1422.673 ms | 1406.808 ms |
| Graphics GPU wait / device brackets | 532.137 / 341.286 ms | 521.030 / 337.364 ms |
| Compute total | 1246.895 ms | 1174.354 ms |
| Compute dispatch/wait | 443.966 ms | 417.077 ms |
| Compute writeback | 390.973 ms | 358.543 ms |
| Process CPU cores | 1.434 | 1.450 |

The unclassified bucket takes approximately **58% less CPU time per reference** in this pair.
That bucket includes more references than the 33 new GPU preparations; it is not a per-dispatch
GPU timing. The CPU control has a shorter sample window and fewer records. Presentation rates
are effectively unchanged, and one pair does not establish an end-to-end speedup. The earlier
sampled-only observation also showed 63.792 ms for 99 unclassified references, consistent with
removing CPU work after admitting the actual storage population. It is a different executable,
not another same-binary control.

GTA retains the lit bank, characters, world and HUD in both captures; actor poses differ. Sonic
retains the known HUD over an absent world. Its 8.155 host presentations/s accompanies 267 existing
cube preparations and **zero new 2D preparations**, so no Sonic gain is attributed to this change.
Blue Prince reaches the furnished hall with coherent geometry and lighting at 11.538 host
presentations/s and zero GPU-detile preparations. This is a hall compatibility observation, not
an FMV equivalence test or a measured Blue Prince speedup.

During the F8-adjacent audio records, cumulative active shortfalls stay flat: GTA main output 17
at 2 in both arms, Sonic output 17 at 0, Blue Prince outputs 1/2 at 1/0. GTA auxiliary output 18
increases earlier in startup (1 to 39 GPU arm; 2 to 41 CPU arm). Sonic gains one after the F8 window.
No audio improvement is claimed. Outputs remain independently recorded.

The [measurement summary](FP16_GPU_PREPARATION_2026_09.json) retains the complete relevant scopes,
window populations, hashes and separate GPU-load observations. Original F8/F9, images, logs, CPU
profiles and runner remain in the private evidence archive `issue-3407-fp16-preparation-20260913`.
No snapshot baseline or blog screenshot was changed.

