---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0030: Descriptors as memory -- `VK_EXT_descriptor_buffer` for the per-draw binding path

## Context

prosper binds every draw's and dispatch's resources through classic descriptor sets. The live
backend still sits in `tests/fixtures/` (ADR 0004 proposes moving it out); on `main` at `aa7d3b57`:

- `tests/fixtures/render_runner.h:10600-10650` sizes one shared `VkDescriptorPool` per batch from
  per-resource counters (storage buffers counted per array element, `:10609`) and creates it with
  `vkCreateDescriptorPool` at `:10647`.
- Per draw, `:13031` calls `vkAllocateDescriptorSets` for the draw's `n_sets` layouts and `:13054`
  writes every binding with `vkUpdateDescriptorSets`; `:14406` and `:15556` bind them with
  `vkCmdBindDescriptorSets`. The compute, NGG and fragment-batch runners
  (`compute_runner.h`, `ngg_subgroup_runner.h`, `fragment_draw_collect_gpu.h`, ...) repeat the same
  allocate / update / bind triple.
- `git grep -i descriptor_buffer prosper/src prosper/tests` finds nothing: the extension is
  unused, and so are push descriptors and update-after-bind.

What the measurements say, quoted with their sources, and they say the cost is **small**:

- `docs/performance/RENDERER_PERFORMANCE_2026_07.md:122,206`: `res.descriptor` was 2.25 ms across
  2,152 draws, about 1 us per draw and 2.0 % of `resources` (July, pre-#1270 renderer).
- `docs/performance/PERFORMANCE_ROADMAP_HANDOFF_2026_09_23.md` § *Rejected descriptor-set reuse*:
  on *Blue Prince* the descriptor setup leaf is 1.01-1.13 ms inside a ~50-54 ms renderer for
  ~2,100-2,250 draws. A census found 22.9 % pass-local and 16.7 % adjacent repeats of set payloads
  and zero repeated full draw bundles; a reuse prototype made both the leaf and the enclosing
  renderer slower (1.180 -> 1.230 ms; 50.3 -> 51.4 ms) and was removed.

So descriptor work is about 2 % of renderer time on the one workload that has been measured. This
ADR does **not** claim that moving it is a frame-rate win; it claims something narrower, below.

The second input is proposed ADR 0027 (open PR #4716). Its dynamic layer proposes
*descriptor-table mirroring*: a host table that mirrors a guest descriptor table index for index,
so a shader whose resource is selected at run time indexes the host table with the guest's own
index instead of the recompiler having to prove a constant provenance. With descriptor sets that
mirror is an array binding that must be (re)written through `vkUpdateDescriptorSets` and is bounded
by pool sizing and `maxPerStageDescriptor*` limits. With descriptor buffers it is plain memory: the
host writes each descriptor at `index * descriptorSize` with `vkGetDescriptorEXT`, the same shape
the guest's own table has.

External reference, **unverified here, from general knowledge**: vkd3d-proton maps D3D12
descriptor heaps -- large, CPU-written, shader-indexed arrays, the closest desktop analogue to a
guest descriptor table -- onto `VK_EXT_descriptor_buffer` where available, binding the heap once and
selecting tables by offset (`vkCmdSetDescriptorBufferOffsetsEXT`), with a descriptor-indexing set
path as fallback. It is cited as evidence that the model carries a heap-shaped API on NVIDIA and
RADV, not as code to follow (charter: external implementations are verification-only).

Driver support, also **unverified on this project's machines**: NVIDIA's proprietary driver and
RADV expose `VK_EXT_descriptor_buffer`; Mesa's lavapipe is believed to implement it in recent
releases, but whether CI's lavapipe version does is unchecked. MoltenVK is not expected to.

## Decision

1. **One binding interface, two backends.** The renderer binds a draw's or dispatch's resources
   through one interface that takes the resolved resources (buffers, image views, samplers per
   binding) and returns nothing the caller must manage. Behind it:
   - **descriptor-buffer backend** when the device reports `descriptorBuffer` and the needed
     limits: per-frame ring buffers of resource and sampler descriptors, written with
     `vkGetDescriptorEXT`, bound with `vkCmdBindDescriptorBuffersEXT` once per command buffer and
     selected per draw with `vkCmdSetDescriptorBufferOffsetsEXT`;
   - **descriptor-set backend**: today's pool / allocate / update / bind path, unchanged in
     behaviour, used when the extension is absent or disabled.
   Pipelines are created with or without `VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT` to match;
   the backend is chosen once per device, never per draw.
2. **The descriptor buffer is the substrate for ADR 0027's mirroring.** If ADR 0027's dynamic
   layer is accepted, a mirrored guest table is a region of the descriptor buffer written index for
   index; on the set backend it stays a descriptor-indexing array. This ADR does not decide 0027.
3. **Selection is a host-capability switch** (`PROSPER_DESCRIPTOR_BUFFER=0` forces the set
   backend). It changes nothing the guest sees, so it is not a guest-behaviour selector; it exists
   for the A/B and for driver bring-up.
4. **Adoption is gated on measurement**, by the procedure in `.claude/skills/perf-change/`: a
   same-binary A/B, the control arm the switch off, on the reference workloads (at least *Blue
   Prince* and *GTA V*'s bank-world anchor). A neutral result is acceptable if ADR 0027 needs the
   substrate; a regression on either workload is not.

No new spec rule is proposed. A candidate (`PERF-P11`, "binding a draw does not allocate from a
pool") would rest on a cost measured at ~2 %; it waits for the A/B in step 4 of the migration.

## Consequences

- **Memory.** Descriptor buffers are host-visible device memory sized per frame in flight; their
  size is `descriptors per frame x descriptorSize` (tens to a few hundred bytes per descriptor,
  device-dependent). Ring exhaustion must stall or grow, never wrap onto in-flight descriptors.
- **Alignment and limits.** Offsets are multiples of `descriptorBufferOffsetAlignment`; the number
  of bound buffers is capped by `maxDescriptorBufferBindings` and sampler descriptors live in a
  separate, smaller buffer (`maxSamplerDescriptorBufferRange`). Combined image samplers need
  `combinedImageSamplerDescriptorSingleArray` care; storage-buffer descriptors still carry ranges,
  so the robustness behaviour of today's ranges must be preserved, not re-derived.
- **Validation.** The Khronos validation layer's coverage of descriptor buffers is thinner than of
  sets; GPU-assisted validation of descriptor contents is weaker. The set backend remains the
  debugging fallback (`PROSPER_DESCRIPTOR_BUFFER=0`), which is one reason it is kept.
- **Capture and replay.** `.prgbundle` records guest commands and resources, so `tools/gpu_replay`
  replays through whichever backend the replaying device selects; this is believed but unverified
  and is checked in migration step 1. RenderDoc support for descriptor buffers is partial, and a
  developer may need the switch off to inspect bindings.
- **Two paths to test.** CI's lavapipe may exercise only one backend; the GPU-execution tests run
  under both where the device allows, and a backend the CI device lacks is stated as untested.

## Alternatives considered

- **Push descriptors** (`VK_KHR_push_descriptor`): removes pool allocation per draw with little
  code, but caps a set at `maxPushDescriptors` (often 32), cannot express a large mirrored table,
  and still costs a write per binding per draw. A reasonable cheaper experiment; it does not
  serve ADR 0027.
- **Descriptor-set caching or reuse**: already tried and **rejected** by measurement (#3770 follow-up,
  roadmap § *Rejected descriptor-set reuse*): payload comparison cost more than the avoided work.
  Not re-proposed in any form keyed on payload equality.
- **Status quo**: costs ~2 % and works on every driver. It remains the fallback; its weakness is
  that it is the wrong shape for index-for-index mirroring, not that it is slow.

## Migration order

1. Introduce the binding interface over today's set path only; pure refactor, no behaviour change,
   GPU-execution tests and one replayed bundle byte-identical.
2. Descriptor-buffer backend for compute dispatches (fewest binding shapes), behind the switch,
   off by default; GPU-execution tests run under both backends.
3. Graphics draws, NGG and fragment batches on the same backend.
4. Same-binary A/B on the reference workloads; record the verdict in the performance docs either
   way. Default on only if step 4 passes.
5. If ADR 0027 is accepted, its mirrored tables are built on this backend.

## Open questions

- Does CI's lavapipe expose `VK_EXT_descriptor_buffer`, and with which limits?
- Is `VK_EXT_descriptor_buffer` the right target, or does a newer cross-vendor descriptor-heap
  extension supersede it on the drivers prosper targets? Check the registry before step 2.
- How large are the per-frame rings on GTA V's heaviest frames, and does the sampler limit bind?
- Windows/NVIDIA and Linux/RADV both need the A/B; macOS (MoltenVK) stays on sets.

## Approval

Requires the project owner's acceptance. Until then nothing changes: the set path is the only
path, and step 1 (a refactor with no behaviour change) is the only part that may land before
acceptance.
