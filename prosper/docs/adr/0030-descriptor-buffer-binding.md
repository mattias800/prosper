---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0030: Descriptors as memory -- `VK_EXT_descriptor_heap` behind one binding interface

## Context

prosper binds every draw's and dispatch's resources through classic descriptor sets. The live
backend still sits in `tests/fixtures/` (ADR 0004 proposes moving it out); on `main` at `000202dbd`:

- `tests/fixtures/render_runner.h:10600-10650` sizes one shared `VkDescriptorPool` per batch from
  per-resource counters (storage buffers counted per array element, `:10609`) and creates it with
  `vkCreateDescriptorPool` at `:10647`.
- Per draw, `:13031` calls `vkAllocateDescriptorSets` for the draw's `n_sets` layouts and `:13054`
  writes every binding with `vkUpdateDescriptorSets`; `:14406` and `:15556` bind them with
  `vkCmdBindDescriptorSets`. The shipping compute path,
  `frontends/shared/live/live_compute.cpp`, keeps its own pool and calls `vkAllocateDescriptorSets`
  the same way; the test-fixture runners (`compute_runner.h`, `ngg_subgroup_runner.h`,
  `fragment_draw_collect_gpu.h`, ...) repeat the same allocate / update / bind triple.
- Across the whole repository (`src/`, `tests/`, `frontends/`) nothing uses descriptor buffers,
  descriptor heaps, push descriptors or update-after-bind.

What the measurements say, quoted with their sources, and they say the cost is **small**. Both
figures are *Blue Prince*:

- `docs/performance/RENDERER_PERFORMANCE_2026_07.md:122,206`, in the section re-measured on
  2026-08-02 (#1284; `c79f742e`, RADV, after #1270): `res.descriptor` was 2.25 ms across 2,152
  draws, about 1 us per draw and 2.0 % **of `resources`** (not of renderer time; the file name says
  July, the section does not).
- `docs/performance/PERFORMANCE_ROADMAP_HANDOFF_2026_09_23.md` § *Rejected descriptor-set reuse*:
  the descriptor setup leaf is 1.01-1.13 ms inside a ~50-54 ms renderer for ~2,100-2,250 draws --
  this is the ~2 % renderer-time share. A census found 22.9 % pass-local and 16.7 % adjacent repeats
  of set payloads and zero repeated full draw bundles; a reuse prototype made both the leaf and the
  enclosing renderer slower (1.180 -> 1.230 ms; 50.3 -> 51.4 ms) and was removed.

So descriptor work is about 2 % of renderer time on the one workload that has been measured. This
ADR does **not** claim that moving it is a frame-rate win; it claims something narrower, below.

The second input is proposed ADR 0027 (open PR #4716). Its dynamic layer proposes
*descriptor-table mirroring*: a host table that mirrors a guest descriptor table index for index,
so a shader whose resource is selected at run time indexes the host table with the guest's own
index instead of the recompiler having to prove a constant provenance. With descriptor sets that
mirror is an array binding that must be (re)written through `vkUpdateDescriptorSets`. With
descriptor indexing and update-after-bind the size bound is not the obstacle: the
`maxPerStageDescriptorUpdateAfterBind*` limits are orders of magnitude above the ordinary
`maxPerStageDescriptor*` ones, and that -- not the ordinary limits -- is the baseline this ADR
argues against. The difference is the update model -- API calls into a pool-allocated set versus
writes into memory. With a descriptor heap it is plain memory: the host writes each descriptor at
an offset with `vkWriteResourceDescriptorsEXT`, the same shape the guest's own table has, and the
shader looks it up by index.

### Which extension (verified 2026-10-07)

Khronos has two "descriptors as memory" extensions, and the newer one supersedes the older in
intent:

- **`VK_EXT_descriptor_heap`** -- announced 2026-01-23 with Roadmap 2026, in the headers from
  Vulkan 1.4.340, cross-vendor EXT with promotion to KHR planned. Khronos states it is intended to
  completely replace the descriptor-set mechanism. One resource heap and one sampler heap, bound
  with `vkCmdBindResourceHeapEXT` / `vkCmdBindSamplerHeapEXT`; descriptors written to memory with
  `vkWriteResourceDescriptorsEXT` / `vkWriteSamplerDescriptorsEXT` (no image-view objects needed);
  no descriptor-set or pipeline layouts -- shaders address descriptors by heap offset, with
  `vkCmdPushDataEXT` replacing push constants. Pipelines are created with
  `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT` and a null layout. Its proposal lists the problems
  it fixes in `VK_EXT_descriptor_buffer`: image views still required, inconsistent constant-data
  paths across vendors, performance portability of mixed buffer/image packing, awkward push
  constants and layouts.
  Sources: <https://www.khronos.org/blog/vulkan-introduces-roadmap-2026-and-new-descriptor-heap-extension>,
  <https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_descriptor_heap.html>,
  <https://docs.vulkan.org/spec/latest/chapters/descriptorheaps.html>.
- **`VK_EXT_descriptor_buffer`** -- the older model (per-set descriptor buffers, still with set
  layouts). Not formally marked deprecated in the registry as far as checked, but no longer the
  direction Khronos is taking.

Driver support, from public reports, **not yet measured on this project's machines**:

- NVIDIA: driver 610 and later, Windows and Linux (NVIDIA developer blog,
  <https://developer.nvidia.com/blog/streamlining-resource-binding-with-end-to-end-support-for-vulkan-descriptor-heaps/>).
  The review cited the R595 branch; NVIDIA's own post says 610, which this ADR uses.
- RADV: merged in Mesa 26.1 behind `RADV_EXPERIMENTAL=heap`, exposed by default from Mesa 26.2
  (<https://www.phoronix.com/news/RADV-Merges-Descriptor-Heap>,
  <https://phoronix.com/news/RADV-Descriptor-Heap-Default>). Intel ANV has also merged it
  (<https://www.phoronix.com/news/Intel-ANV-Descriptor-Heap-Merge>).
- lavapipe (CI): implements `VK_EXT_descriptor_buffer` since Mesa 23.2
  (<https://www.phoronix.com/news/Lavapipe-Vulkan-Descriptors>); **whether it implements
  `VK_EXT_descriptor_heap`, and whether CI's version does, is unconfirmed.**
- MoltenVK: not expected to.

External reference: vkd3d-proton -- D3D12 descriptor heaps are the closest desktop analogue to a
guest descriptor table -- has used `VK_EXT_descriptor_buffer` and has now merged a
`VK_EXT_descriptor_heap` path, off by default behind `VKD3D_CONFIG=descriptor_heap` while NVIDIA
driver bugs are resolved (<https://phoronix.com/news/VKD3D-Proton-Descriptor-Heaps>). It is cited as
evidence that the heap model carries a heap-shaped API, and that early drivers still have bugs --
not as code to follow (charter: external implementations are verification-only).

## Decision

1. **One binding interface, two backends.** The renderer binds a draw's or dispatch's resources
   through one interface that takes the resolved resources (buffers, image views or image
   descriptions, samplers per binding) and returns nothing the caller must manage. Behind it:
   - **descriptor-heap backend** (`VK_EXT_descriptor_heap`) when the device reports the feature
     and the needed limits: per-frame ring regions of the resource and sampler heaps, written with
     `vkWriteResourceDescriptorsEXT` / `vkWriteSamplerDescriptorsEXT`, heaps bound once per command
     buffer, per-draw table offsets delivered with `vkCmdPushDataEXT`;
   - **descriptor-set backend**: today's pool / allocate / update / bind path, unchanged in
     behaviour, used when the extension is absent or disabled.
   `VK_EXT_descriptor_buffer` is **not** a third backend: it is the model Khronos is replacing, and
   every target driver that has it either already has the heap extension (NVIDIA 610+, RADV 26.2+)
   or falls back to sets. A third backend would be a third path to test for no device that needs it.
   The backend is chosen once per device, never per draw, so pipelines are created with or without
   `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT` uniformly. That also satisfies proposed ADR 0029
   (#4722): every library linked into a GPL pipeline must agree on the descriptor model, and a
   per-device choice guarantees it.
2. **The descriptor heap is the substrate for ADR 0027's mirroring.** If ADR 0027's dynamic layer
   is accepted, a mirrored guest table is a region of the resource heap written index for index; on
   the set backend it stays an update-after-bind descriptor-indexing array. This ADR does not decide
   0027.
3. **Selection is a host-capability switch** (`PROSPER_DESCRIPTOR_HEAP=0` forces the set backend).
   It changes nothing the guest sees, so it is not a guest-behaviour selector; it exists for the A/B
   and for driver bring-up -- the vkd3d-proton report above says early heap drivers have bugs.
4. **Adoption is gated on measurement**, by the procedure in `.claude/skills/perf-change/`: a
   same-binary A/B, the control arm the switch off, on the reference workloads (at least *Blue
   Prince* and *GTA V*'s bank-world anchor). A neutral result is acceptable if ADR 0027 needs the
   substrate; a regression on either workload is not.

No new spec rule is proposed. A future PERF rule ("binding a draw does not allocate from a pool")
would rest on a cost measured at ~2 %; it waits for the A/B in step 4 of the migration.

## Consequences

- **Recompiler.** With no pipeline layout, the recompiled SPIR-V addresses descriptors through the
  heap mapping instead of set/binding decorations, so the recompiler emits two binding forms (or
  one form plus the extension's mapping). This is the largest cost of the decision and is why
  step 1 below introduces the interface before any backend changes.
- **Memory.** Heap regions are host-visible device memory sized per frame in flight:
  `descriptors per frame x descriptor size` (device-dependent). Ring exhaustion must stall or grow,
  never wrap onto in-flight descriptors.
- **Alignment and limits.** Heap offsets and sizes obey the extension's alignment and maximum-size
  limits, and samplers live in a separate, smaller heap. Buffer descriptors still carry ranges, so
  the robustness behaviour of today's ranges must be preserved, not re-derived.
- **Driver maturity.** The extension is new (2026); RADV enabled it by default only in 26.2 and
  vkd3d-proton keeps its heap path off by default over NVIDIA bugs. The set backend stays the
  default until the A/B and the GPU-execution tests pass on both vendors.
- **Validation.** Validation-layer coverage of a 2026 extension is thinner than of sets. The set
  backend remains the debugging fallback (`PROSPER_DESCRIPTOR_HEAP=0`), which is one reason it is
  kept.
- **Capture and replay.** `.prgbundle` records guest commands and resources, so `tools/gpu_replay`
  replays through whichever backend the replaying device selects; this is believed but unverified
  and is checked in migration step 1. Nsight Graphics 2026.2 supports heaps; RenderDoc support is
  unchecked, and a developer may need the switch off to inspect bindings.
- **Two paths to test.** CI's lavapipe may exercise only the set backend; the GPU-execution tests
  run under both where the device allows, and a backend the CI device lacks is stated as untested.

## Alternatives considered

- **`VK_EXT_descriptor_buffer`**: the earlier memory model, wider deployed today (including
  lavapipe). Rejected as the target because Khronos is replacing it and its own successor's
  proposal lists portability and performance problems with it; adopting it now means a second
  migration later. Revisit only if the heap extension turns out unusable on a target driver that
  has descriptor buffers.
- **Push descriptors** (`VK_KHR_push_descriptor`): removes pool allocation per draw with little
  code, but caps a set at `maxPushDescriptors` (often 32), cannot express a large mirrored table,
  and still costs a write per binding per draw. A reasonable cheaper experiment; it does not
  serve ADR 0027.
- **Update-after-bind descriptor indexing on sets**: large limits, works widely, and is the set
  backend's form of the mirror. Its weakness is the update model (API calls into a pool-allocated
  set), not size.
- **Descriptor-set caching or reuse**: already tried and **rejected** by measurement (#3770 follow-up,
  roadmap § *Rejected descriptor-set reuse*): payload comparison cost more than the avoided work.
  Not re-proposed in any form keyed on payload equality.
- **Status quo**: costs ~2 % and works on every driver. It remains the fallback; its weakness is
  that it is the wrong shape for index-for-index mirroring, not that it is slow.

## Migration order

1. Introduce the binding interface over today's set path only; pure refactor, no behaviour change,
   GPU-execution tests and one replayed bundle byte-identical.
2. Descriptor-heap backend for compute dispatches first (fewest binding shapes) -- the shipping
   path in `frontends/shared/live/live_compute.cpp` and the `compute_runner.h` fixture -- behind
   the switch, off by default; GPU-execution tests run under both backends.
3. Graphics draws, NGG and fragment batches on the same backend.
4. Same-binary A/B on the reference workloads; record the verdict in the performance docs either
   way. Default on only if step 4 passes.
5. If ADR 0027 is accepted, its mirrored tables are built on this backend.

## Open questions

- Does CI's lavapipe expose `VK_EXT_descriptor_heap` at all, and with which limits? If not, the
  heap backend is untested in CI and must be stated so.
- Which minimum NVIDIA (610+) and Mesa (26.2+) versions do the project's machines run, and are the
  driver bugs vkd3d-proton reports on NVIDIA hit by prosper's binding shapes?
- How does the recompiler express heap lookups -- the extension's descriptor mapping, or direct
  heap indexing in SPIR-V -- and does `spirv-val` cover it?
- How large are the per-frame rings on GTA V's heaviest frames, and does the sampler heap limit bind?
- Windows/NVIDIA and Linux/RADV both need the A/B; macOS (MoltenVK) stays on sets.

## Approval

Requires the project owner's acceptance. Until then nothing changes: the set path is the only
path, and step 1 (a refactor with no behaviour change) is the only part that may land before
acceptance.
