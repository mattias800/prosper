# GPU replay

`gpu_replay` inspects, validates, graphs, and renders a local `.prgcap` without booting the guest.
Capsules contain title-derived shaders, resource bytes, addresses, and optional rendered RTT pixels.
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
./build-linux/gpu_replay --dump-resource 18:ps:34 /tmp/texture.bin /tmp/submit.prgcap
./build-linux/gpu_replay --dump-shader 18:fs /tmp/fragment.spv /tmp/submit.prgcap
```

`--draw` uses realized draw indices, while graphs use mixed semantic operation indices. They are not
interchangeable when dispatches or unrealized operations are present. Replay restores serialized render-target
seeds before operation zero and uses owned resource memory; it must not dereference original guest mappings.

`--prepend` materializes and executes one earlier capsule in the same renderer instance before the consumer.
Its rendered targets take precedence over consumer RTT seeds at matching addresses; unrelated seeds are still
restored. The tool rejects a predecessor whose submit number is not earlier. A changed image proves that the
consumer sampled the retained producer output, but it does not prove faithful closure: graph the predecessor
and continue if it also has temporal leaves.

`--bundle` reconstructs each captured submit through the normal version-5 validator, executes them in
ascending order through one renderer instance, and releases each materialized submit before the next.
The summary reports logical versus unique bytes, per-submit output hashes, and every temporal image leaf:

- `stop=included-producer` names the earlier bundled submit whose target overlaps the leaf.
- `stop=initialized-seed` means serialized RTT pixels establish the version without another submit.
- `stop=configured-bound` means the earliest bundled submit still needs older history.
- `stop=unresolved-producer` means a later submit has no overlapping earlier bundled target; a contiguous
  bundle should not produce this and the capture/replay evidence is incomplete.

A successful replay with `configured-bound` is an explicitly partial closure, not a faithful pixel oracle.
Bundle files use `.prgbundle`, contain title-derived data, are gitignored, and must not be committed.

`--bundle-zero-boundary` is an A/B diagnostic. It supplies transparent RGBA pixels for unseeded temporal
image leaves at the oldest bundled submit, labels them `diagnostic-zero-seed`, and leaves the bundle unchanged.
It can disprove stale guest backing versus zero initialization as the cause of a mismatch, but it is not a
faithful replacement for the missing producer history.

## Current Dead Cells reference

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

Capturing only 642x362-target predecessor draws still costs about 150 MiB per submit because repeated static
fragment resources include 64, 32, and 16 MiB textures. A 1,200-submit brute-force bundle is therefore not a
practical next step. The next tooling boundary is an exact shared-resource dictionary or equivalently proven
unchanged-resource reuse across submits; mutable versions must never be reused from address identity alone.
