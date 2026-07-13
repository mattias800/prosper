# GPU replay

`gpu_replay` inspects, validates, graphs, and renders a local `.prgcap` without booting the guest.
Capsules contain title-derived shaders, resource bytes, addresses, optional rendered RTT pixels, and optional
exact persistent Vulkan depth/stencil checkpoint planes.
They are gitignored local artifacts and must never be committed or shared as project fixtures.

Build from `prosper/`, using the worktree-local Linux build:

```bash
cmake --build build-linux -j8 --target gpu_replay
```

Create a capsule with the native-speed workflow in
[`../gpu_timeline/README.md`](../gpu_timeline/README.md). Then use:

```bash
./build-linux/gpu_replay --inspect-only /tmp/submit.prgcap
./build-linux/gpu_replay --validate /tmp/submit.prgcap
./build-linux/gpu_replay --graph /tmp/submit.prgcap
./build-linux/gpu_replay --graph-json /tmp/graph.json /tmp/submit.prgcap
./build-linux/gpu_replay /tmp/submit.prgcap /tmp/replay.bmp
./build-linux/gpu_replay --prepend /tmp/producer.prgcap /tmp/consumer.prgcap /tmp/closure.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle /tmp/closure.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle --bundle-tail 2 /tmp/tail.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle \
  --bundle-intermediate-through-target 642x362 /tmp/closure.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle \
  --bundle-final-capsule /tmp/final-seeded.prgcap /tmp/closure.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle \
  --bundle-extract-submit 19046 /tmp/submit-19046.prgcap
./build-linux/gpu_replay --bundle /tmp/window.prgbundle --bundle-compact /tmp/compact.prgbundle
./build-linux/gpu_replay --bundle /tmp/window.prgbundle --bundle-zero-boundary /tmp/zero-ab.bmp
```

## Reading graph output

- `edge producer=P consumer=C` means operation `C` reads a range last written by earlier operation `P`.
- `external` means no earlier writer exists inside this capsule. Static assets are normally external.
- `consumers=N first=C` deduplicates one external address/range used by multiple operations.
- `future-writer=N` means the selected submit reads the range before rewriting it. This is a temporal
  version and the strongest signal that an earlier submit must join the replay closure.
- `future-writer=-1` means this capsule never writes the range. It may be static/CPU data or have a producer
  further back in GPU history. Cross-submit resolution is tracked by #595.
- `missing operation` is semantic work whose shader/resource contract could not be realized. A graph or replay
  containing one is explicitly partial; do not treat its rendered image as a faithful oracle.

The JSON contains `nodes`, `edges`, and deduplicated `external_leaves` with the same operation indices,
logical addresses, byte ranges, dimensions, bindings, stages, and future-writer identities.

## Pixel oracles

Renderer-invocation captures record an expected byte count/hash and replay exits nonzero on mismatch.
Timeline-selected captures occur before Vulkan and report `oracle=no`; they render normally but do not invent an
expected hash. `--allow-mismatch` is for an intentional differential such as `--draw N:M`.

## Isolation and extraction

```bash
./build-linux/gpu_replay --draw 12:18 /tmp/submit.prgcap /tmp/pass.bmp
./build-linux/gpu_replay --through-operation 52 /tmp/submit.prgcap /tmp/prefix.bmp
./build-linux/gpu_replay --dump-resource 18:ps:34 /tmp/texture.bin /tmp/submit.prgcap
./build-linux/gpu_replay --dump-shader 18:fs /tmp/fragment.spv /tmp/submit.prgcap
./build-linux/gpu_replay --dump-compute 0 /tmp/compute.spv /tmp/submit.prgcap
./build-linux/gpu_replay --dump-compute-resource 0:2 /tmp/storage.bin /tmp/submit.prgcap
./build-linux/gpu_replay --dump-failed-shader 0:1 /tmp/failed-fragment.bin /tmp/submit.prgcap
```

### Failed operations

Capture v7 and later retain bounded diagnostics for draws and dispatches that semantic PM4 ordering contains but the
executor cannot realize. `--inspect-only` prints the failure reason, decoded target/pipeline or compute-launch
state, every referenced stage address, resource-table presence/count, descriptor issues, recompile coverage,
and the first rejected opcode/format at its exact dword PC. It also prints the raw RDNA2 content hash, byte
count, and whether the retained stream reached `s_endpgm`.

`--dump-failed-shader FAILURE:STAGE PATH` writes that raw stream for `shader_inspect` or a focused recompiler
fixture. Both indices are the zero-based values printed by `--inspect-only`; they are not draw indices. Each raw
stage is fault-safely read once, content-deduplicated, capped at 64 KiB, and stopped at the first decoded
`s_endpgm`/unknown instruction or the cap. Total failed-stage data is capped at 64 MiB, diagnostics cannot
outnumber semantic operations, and every reference/hash is validated while reading. Captures v1-v6 remain
readable and print `failure-diagnostics: unavailable (capture predates v7)` rather than inventing evidence.

Capture v8 adds persistent depth/stencil checkpoints. Each seed stores the renderer's complete guest cache
identity (depth/stencil read/write bases, HTILE base, extent, and D32/D32S8 format), independent depth/stencil
validity, and raw valid-plane bytes. Counts, extents, formats, duplicate identities, per-plane lengths, and a
1 GiB total are validated before allocation or Vulkan upload. Captures v1-v7 remain readable with zero DS
seeds. `--inspect-only` prints every seed's identity, validity, byte counts, and content hashes.

Compute selectors use the realized compute index printed by `--inspect-only`. The resource selector is
`COMPUTE:BINDING`; it writes the captured pre-dispatch storage-buffer bytes, while `--dump-compute` writes
the exact specialized SPIR-V executed by replay. For a long live run, `PROSPER_COMPUTELOG_CODE=0x...` and
`PROSPER_COMPUTELOG_SIZE=N` restrict before/after hash diagnostics to dispatches matching the configured
program address and storage-buffer byte size. Either filter may be used alone.

`--draw` uses realized draw indices, while graphs use mixed semantic operation indices. They are not
interchangeable when dispatches or unrealized operations are present. Replay restores serialized render-target
seeds before operation zero and uses owned resource memory; it must not dereference original guest mappings.
Bundle operation sources are semantic draw IDs and may contain holes; tooling must resolve them through each
realized item's `draw_index`, never treat them as offsets into the compact draw vector.

`--through-operation N` executes the inclusive mixed graphics/compute prefix `0..N`, preserving operation
order and all earlier work. Prefix output uses the last executed draw target's native dimensions. Use it with
a hash-verified seeded final capsule for fast composition bisection; unlike `--draw`, it does not discard
compute dispatches or earlier draws.

`--prepend` materializes and executes one earlier capsule in the same renderer instance before the consumer.
Its rendered targets take precedence over consumer RTT seeds at matching addresses; unrelated seeds are still
restored. The tool rejects a predecessor whose submit number is not earlier. A changed image proves that the
consumer sampled the retained producer output, but it does not prove faithful closure: graph the predecessor
and continue if it also has temporal leaves.

`--bundle` reconstructs each captured submit through the normal versioned validator, executes them in
ascending order through one renderer instance, and releases each materialized submit before the next.
The summary reports logical versus unique bytes, per-submit output hashes, and every temporal image leaf:

- `stop=included-producer` names the earlier bundled submit whose target overlaps the leaf.
- `stop=initialized-seed` means serialized RTT pixels establish the version without another submit.
- `stop=configured-bound` means the earliest bundled submit still needs older history.
- `stop=unresolved-producer` means a later submit has no overlapping earlier bundled target; a contiguous
  bundle should not produce this and the capture/replay evidence is incomplete.

A successful replay with `configured-bound` is an explicitly partial closure, not a faithful pixel oracle.
Bundle files use `.prgbundle`, contain title-derived data, are gitignored, and must not be committed.

`--bundle-tail N` replays only the latest `N` manifests while retaining the bundle's exact shared resource
dictionary. Use it when lifetime evidence proves that earlier submits cannot be target producers; the first
replayed submit becomes the explicit configured boundary and its graph must still be inspected for leaves.

`--bundle-intermediate-through-target WxH` stops each non-final submit after its last realized draw to that
target extent. It avoids rendering later presentation/UI passes when timeline and dependency evidence proves
that only the named target family crosses submit boundaries. The final submit remains complete. Replay aborts
if an included temporal image leaf has another extent or an intermediate submit has no matching draw. This is
an evidence-gated optimization, not automatic dependency closure.

`--bundle-final-capsule PATH` snapshots the complete live color RTT and persistent Vulkan depth/stencil caches
before executing the final submit and writes them as seeds in a standalone capture-v8 capsule. Its standalone
output must match the bundle's final hash before using it for fast `--draw`, operation-prefix, resource, or
shader experiments. Export validates every surface and fails when a required temporal color surface is absent;
it never substitutes zero pixels. The file is installed only after the source final submit renders, and embeds
that output byte count/hash as its replay oracle. A legacy pre-v7 manifest with unrealized operations is migrated explicitly to
`Unknown` failure diagnostics with the original operation kind/source/order, so the v8 writer remains strict.
The DS snapshot uses transfer barriers to return every image to depth/stencil-attachment layout before the final
bundle submit executes; equality therefore also checks that readback did not perturb the source run.

`--bundle-extract-submit N PATH` materializes one run-local submit manifest as a normal capsule and exits
without Vulkan when no image output is requested. Use it to inspect, graph, or validate the exact predecessor
whose frontier or cache behavior is in question.

`--bundle-find-ds ADDR` scans only the compact submit manifests and reports every submit whose depth/stencil
read or write bases match `ADDR`, including draw range, target extent, depth-test/write counts, clear count,
and compare-op bitmask. It does not reconstruct resource payloads or initialize Vulkan. Use it to establish a
guest DS surface's lifetime before expanding a replay window or changing cache behavior.

`--bundle-ds-summary` scans every depth/stencil-active draw and groups it by the renderer's current guest
identity tuple (depth read/write plus stencil read/write bases). It reports lifetime, target extents,
test/write/clear totals, mixed-identity passes, anonymous passes, and identity transitions for each color
target. The query is manifest-only and can be combined with `--bundle-tail N`. Use it before changing cache
identity or invalidation rules: a mixed pass proves the one-attachment backend is collapsing state, while a
stable identity with no clear keeps the investigation focused on backing-memory/HTILE lifetime.

`--legacy-htile-before-stencil` is an explicit migration diagnostic for the preserved pre-v6 Dead Cells
closure. Capture v5 did not serialize `DB_HTILE_DATA_BASE`; that run's allocations and later v6 captures
prove its HTILE block is exactly 64 KiB before stencil. The switch supplies only that missing cache identity
so compute-write invalidation can be A/B tested against the immutable closure. Current captures retain the
real register and must not use this layout inference.

`--bundle-compact PATH` writes a validated copy containing only resources and chunks reachable from retained
submit manifests. It is useful after a rolling semantic capture; with no image output argument, compaction
exits without initializing Vulkan. Combine it with `--bundle-tail N` to write a compact suffix bundle.

`--bundle-zero-boundary` is an A/B diagnostic. It supplies transparent RGBA pixels for unseeded temporal
image leaves at the oldest bundled submit, labels them `diagnostic-zero-seed`, and leaves the bundle unchanged.
It can disprove stale guest backing versus zero initialization as the cause of a mismatch, but it is not a
faithful replacement for the missing producer history.

## Current Dead Cells reference

The preserved #608 playable closure selects exactly 90 semantic draws with the 738x420 pass at draw 79..81. Its
retained submits 18,165..19,047 represent 158.94 GiB logically and 739 MiB uniquely, resolving all 1,764 temporal
image dependencies. Its original pre-#611/#615 replay hash was `5759c125812154dc`; absolute hashes from that old
renderer revision are historical, while source/capsule equality remains the checkpoint invariant.

On current code, normal #611 invalidation produces `fac9ca4cbbba8196` from both the full bundle and standalone
v8 capsule. With `PROSPER_DS_GUEST_WRITE_INVALIDATE=0`, the exact stale-depth A/B produces
`535256588b67a536` from both paths and byte-identical BMPs. The self-contained capsule holds 12 RTT surfaces,
one valid 642x362 D32S8 depth plane (929,616 bytes), the effective invalidation/legacy-HTILE settings, and its
33,177,600-byte output oracle; standalone replay takes about 3.3 seconds instead of roughly 24 minutes. A
tail-two negative control remains `71b84bdfae53933c` in both bundle and capsule because its two color frontiers
are explicitly bounded. This closes #569 without hiding an incomplete history boundary.

Manifest scanning shows the same 642x362 depth address in all 883 submits, always with LESS_OR_EQUAL
tests/writes and never a draw/register clear. Timeline-v5 backing hashes and writer provenance resolved the
hardware boundary: supported compute program `0x401aec200` fills the exact 32 KiB HTILE allocation with
`0xfffffff0` before the scene draw span. The live backend invalidates overlapping persistent DS entries on guest
GPU writeback (#611); capture v8 preserves the state exactly when an investigation deliberately disables that
invalidation.

For new routed captures, timeline v6 replaces that historical endpoint with 91..93 semantic draws, exactly
8 dispatches, and the 738x420 pass at draw 80..82. `gpu_timeline --signatures` derived the conjunction from two
independent routes, and `--select` proves it has no splash/menu/loading or adjacent-index matches before capture.

The #594/#595 gameplay capsule at submit 18,750 has two external 642x362 temporal versions. Timeline-v4
full-run aggregation shows that both surfaces begin around submit 17,400 in current runs and are rewritten
about 1,200-1,350 times before the selected submit. Their first observed graphics writers have programmed
clear state, color mode 0, target mask `0xf`, and color format 10. These are raw register observations, not
proof that hardware performed an implicit clear. Resource addresses, submit numbers, and semantic operation
ordinals vary between runs; recompute them and never hardcode them into renderer behavior.

A same-run depth-16 bundle for submits 18,735..18,750 resolved all 30 internal temporal edges and reported
exactly two 642x362 leaves at the configured lower bound. Content-defined dedup stored 2.883 GiB of logical
capsules in 166.3 MiB (5.8%). No initialized RTT seed appeared in that window, and even/odd depths alternate
between the dark and overbright failure. Supplying transparent zero to those two lower-bound leaves produces
the same final hash (`62a46f15efa52a26`) as unseeded replay, so stale guest backing versus transparent
initialization is not the cause in this window.

Bundle v2 stores exact resource versions in a global content-defined chunk dictionary. On a fixed 1,200-submit
full-state run it folded 122.97 GiB of logical capture data into 301.1 MiB in 169.4 seconds. Reuse is proven by
hash plus byte equality; mutable versions are never reused from address identity. Linux capture reads use
fault-safe `process_vm_readv` rather than per-page pipe probes.

A timing-independent semantic run retained a rolling 1,400-submit window and selected run-local submit 23,350.
The two 642x362 surfaces first appear one submit earlier. Compacting to that exact two-submit suffix stores
142.6 MiB; replay resolves both temporal edges with zero seeded, bounded, or unresolved leaves and finishes at
hash `112e7db0001791cc`. The image is the dark opening vignette with `HOLD Skip`, not playable gameplay, so
extent plus 80 draws is still too broad. #608 tracks a stable playable checkpoint and first bad composition.
