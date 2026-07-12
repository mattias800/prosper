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

## Current Dead Cells reference

The #594/#595 gameplay capsule at submit 18,750 has two external 642x362 temporal versions. A timeline-v3
run resolved both to submit 18,749: draw 45 / PM4 order 9,436,927 and draw 41 / PM4 order 9,436,871. The
resource addresses and even semantic operation ordinals are run-local observations, not title constants.
Recompute them from each new capsule and timeline; never hardcode them into renderer behavior.
