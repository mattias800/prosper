# Performance roadmap handoff (2026-09-23)

This handoff continues the general performance roadmap after #3770. It records one rejected
optimization, a fresh current-main critical-thread profile, and the next bounded discriminator. It
does not claim an FPS improvement and does not change runtime behavior.

## Current source and workload

The retained measurements used `a076095b311c`, a Release `-O3 -DNDEBUG` build, the established GTA
V Performance Story route, fresh per-run save and cache directories, and continuous peer-process
auditing. The exact executable had SHA-256
`d3a1b92286b49511324d4a0a32c2256aa5a84519280c1331b152dbb862c12f6e` and ELF build ID
`b72f59f447d2cc58823089779d54aeefdfa63e0d`. The route SHA-256 was
`683994c1b5e6fb14b07e90e7d07390d52a1ffb7ebbf1cc496d2b92489c81ed7a`.

Main has since advanced through #3800. Recheck the selected source on the new exact head before
implementing; do not transplant changes by reverting the intervening renderer or recompiler work.

## Rejected descriptor-set reuse

#3770 left a recurring Blue Prince descriptor setup leaf of roughly 1.01--1.13 ms for settled
2,100/2,249-draw batches. A diagnostic census then found:

- 5,908,971 descriptor-set references;
- 1,352,313 exact pass-local repeats (22.886%);
- 988,035 exact adjacent repeats (16.721%);
- zero repeated complete draw bundles.

The comparison covered the set-layout handle and every buffer/image descriptor field. A prototype
reused only an exact adjacent payload within one render pass, after current resource resolution and
upload, under the existing descriptor-pool lifetime. Focused tests covered changed slices, descriptor
arrays, layouts, mixed sets, allocation failure and recovery, plus a same-binary disable control.
Two independent static reviews accepted the corrected lifetime and failure behavior.

Performance rejected it. In one peer-clean same-binary Blue pair, with 20 records per recurring
shape in each arm:

| Accepted draws | Descriptor leaf | Enclosing renderer |
|---:|---:|---:|
| about 2,098 | 1.180 -> 1.230 ms | 50.325 -> 51.405 ms |
| about 2,247 | 1.255 -> 1.290 ms | 53.535 -> 54.275 ms |

Both targeted and enclosing medians moved in the wrong direction. Exact payload comparison and
pending-write construction cost more than the avoided allocation/update work. The prototype was
removed. Do not revive it from repeat counts alone.

## Fresh GTA critical-thread profile

The profiler selected actual OS TID 994723 at the settled bank-world anchor. It accumulated 148 CPU
ticks during selection versus the next TID's 18. The 20-second window retained 576 samples, zero
loss, the matching build ID, and no detected compiler, game, replay or screenshot peer.

Selected-TID self shares were:

| Symbol | Self share |
|---|---:|
| `__memmove_avx512_unaligned_erms` | 11.63% |
| `_int_malloc` | 5.90% |
| live renderer callback | 5.90% |
| `resolve_dynamic_fetch` | 4.69% |
| `__memcmp_evex_movbe` | 4.51% |
| `drain_guest_gpu_writes` | 2.60% |
| `render_draw_pass_rgba` | 2.60% |
| `extract_render_state` | 2.26% |
| `memset` | 2.08% |
| `find_spirv_descriptor_binding` | 1.91% |
| `realize_draw_item` | 1.74% |

These are sampled CPU shares on one TID. They are not elapsed latency, do not include worker CPU,
and must not be added.

## Source-resolved movement and comparison

A separate intrusive run selected actual TID 1138649 (149 selection ticks versus 19). Its complete
peer audit was empty. The probe had no overflow and did not read an invalid TID selector. This run
attributes callers only; its cycle totals and byte rates are not timing evidence.

Leading populations in its 30-second selected-thread interval were:

| Operation | Calls | Bytes | Contract/status |
|---|---:|---:|---|
| compute movement at `execute_item` | 3,198 | 7.38 GB | includes ordinary result/guest movement |
| direct `copy_compute_buffer` | 754 | 2.32 GB | exact copy, often parallelized already |
| renderer exact source validation | 91,021 | 1.63 GB | mutation proof, not removable from volume alone |
| compute CPU renderer publication | 659 | 2.10 GB | apparent removable second materialization |
| `compute_buffers_equal` | 2,445 | 1.77 GB | exact result/source decision |
| renderer upload batches | 40,009 | 2.84 GB | existing overlap sharing still applies |
| shader-resource-key equality | 2,485,440 | 0.04 GB | only 175.4 M intrusive cycles; weak target |

Disassembly binds the 659-copy population exactly: `execute_item` allocates and copies
`layout_source[0..linear_bytes)` into a shared CPU vector, then immediately calls
`notify_live_render_target_image_written`. This is the fallback branch near the completed storage
image writeback publication. It is separate from mandatory guest writeback.

The top compute movement must not be skipped under the current guest-observation contract. Existing
diff-span, parallel copy, repeated-output and GPU result comparison paths already address their
eligible subsets. The CPU renderer-publication copy is the strongest remaining removable population
identified so far, but its formats and refusal reasons still need one bounded census before code is
chosen.

## Next discriminator

Take one peer-clean, capture-window-only GTA run on current main with:

```text
PROSPER_COMPUTE_IMAGE_TIMING=1
PROSPER_COMPUTE_TIMING_CAPTURE_ONLY=1
PROSPER_PERF_CAPTURE_AFTER_MS=<settled anchor>
```

Use the existing shortened Performance Story anchor. Rank `[compute-image-writeback]` rows by
`fmt`, component count, tile mode, exact bytes, dimensions, renderer-result retention and the
measured `cache_ms`/`total_ms`. Join them with `[compute-rtt-destination-check]` refusal fields and
the destination census. Keep this diagnostic run separate from timing comparisons.

The previous attempt was deliberately stopped before launch after the peer detector saw another
lane's compiler. No format conclusion exists yet.

The experiment rejects a format extension when the CPU publication population is mostly
unsupported geometry, aliases, partial writes without an exact source seed, absent renderer images,
or too little enclosing cost. If one exact recurring format accounts for material copy time and an
existing renderer image can safely receive it, extend the destination mirror only for that proven
shape. `Rgba32Float`, `R8Unorm`, `R32Uint`, `R32Float`, `Rg8Unorm`, `Rg16Float` and `R16Float` are
already represented in `LiveTargetPixelFormat`; their presence in the enum is not admission proof.

## Correctness contract for a destination extension

Preserve the contracts established by #3746, #3750, #3753 and #3768:

- exact allocation address, registration lifetime, extent, pixel format and native Vulkan format;
- one layer, one mip, one sample, no unsupported pitch/tail/offset geometry;
- no storage write mask, source collision, prior/final output conflict or unrelated alias;
- same Vulkan device, transfer-destination capability and a restorable non-undefined layout;
- separate source and destination leases when a partial writer needs old pixels;
- no publication until submission completes and all ordinary guest writebacks succeed;
- revoke destination authority before submission and on every failed/unproven completion path;
- preserve ordinary guest-memory writeback and conservative CPU publication for every refusal;
- check required Vulkan format features rather than assuming one vendor;
- keep a same-binary disable control for the new admission.

A format with no exact renderer-to-compute seed may initially admit only a producer proven to
overwrite the complete image. Do not infer full overwrite merely from a dispatch extent or from the
absence of an observed read. If the current write-mask/producer analysis cannot prove it, retain the
fallback.

## Verification and measurement plan

Tests should exercise the production handoff with independent byte expectations:

1. complete write publishes the exact pinned renderer image and exact guest bytes;
2. partial write either seeds exact untouched pixels under a separate lease or is refused;
3. format, extent, layout, image identity and registration-lifetime mismatches fall back;
4. an alias/source collision falls back;
5. injected readback, submit and completion failures revoke authority and release leases;
6. address reuse or renderer-image replacement cannot publish into the replacement;
7. the disable control takes CPU publication, and a targeted mutation makes the critical assertion
   fail.

Run the focused production-backend tests and strict Vulkan synchronization validation. Then compare
same-binary control/candidate GTA windows by exact program hash and dispatch shape. Report enclosing
dispatch, writeback, publication/cache and GPU timestamp intervals separately. GTA producer lineage
is still unknown, so no delivery or FPS claim follows. Use Blue Prince as the contrasting visual and
known-new delivery workload; a neutral delivery result remains a valid local optimization result.

If the census rejects renderer publication, return to the current profile ranking. Allocation needs
caller/size/lifetime evidence before any pool. `resolve_dynamic_fetch` needs the existing exact-input
workbench and a bounded reusable-plan subset; do not cache evaluated tables or current guest values.
The prior descriptor-set reuse, snapshot pools, reflected-binding index and retained-range default
enablement remain rejected or separately blocked.
