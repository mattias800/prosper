---
kind: adr
status: proposed
date: 2026-10-07
---

# ADR 0033: Read guest buffers in place on the GPU instead of copying them

## Context

The PS5 GPU reads buffers straight out of unified memory. prosper's Vulkan renderer cannot: every
buffer a draw or dispatch reads is first copied by the CPU into Vulkan-owned memory, and a cached
copy is re-validated by reading the guest bytes again. Vulkan has two standard ways to avoid that
copy, and prosper uses neither: a grep of `src/`, `frontends/` and `tests/` for
`external_memory_host`, `HostPointer`, `minImportedHostPointerAlignment` and
`BufferDeviceAddress` finds nothing (`origin/main` at `5fee2f98d`).

**How a buffer reaches the GPU today.**
- The live frontend "borrows" guest memory where it can: `draw_resources.cpp:221-224`
  (`frontends/shared/live/submit_renderer/`) hands the backend a view whose words *are* the guest
  VA (`direct_guest_buffer_addr`, documented at `tests/fixtures/render_runner.h:164-166`). Where it
  cannot, it copies into a vector and counts `Transfer::DrawBufferStage` (`draw_resources.cpp:242`,
  `:254`). "Borrowed" removes only the frontend copy.
- The backend then copies again. A shareable read-only buffer of 4 KiB or more goes to the
  resident buffer cache (`render_runner.h:11702-11716`); anything else is `parallel_render_memcpy`'d
  into a host-visible arena slice (`render_runner.h:11724-11748`), the time charged to
  `res_buffer_copy_ms` (`render_runner.h:1398`).
- A resident hit is validated by write-watch where one is armed, otherwise by a full `memcmp`
  against a retained CPU snapshot. The watch arms only after two equal validations and disables
  itself after two dirty queries (`render_runner.h:4968-4999`) -- the `PERF-P5` violation #3155
  records. `docs/gpu/RENDERER_ARCHITECTURE_GAPS_2026_09_25.md` § 3 counts the hit path as `2 x
  bytes` read plus a second full CPU copy of every resident buffer.
- On Windows there is no write-watch at all (`GuestWriteWatch::create` returns empty; ADR 0032),
  so every validation is the compare.

**What it costs, measured.** *Blue Prince*'s collapsed regime (#2215, charter § F9/F8): with the
backend sub-buckets live, `setup_resources 801.9ms [buffer=673.4 (copy=539.2) ...]` -- buffer work
~84% and the copy alone ~67% of resource setup. *GTA V*, 2026-09-23 profile (gaps doc § 3): renderer
upload batches 2.84 GB and exact source validation 1.63 GB in 30 s; `__memmove_avx512` 11.63%
self. *Black Flag*, Windows/NVIDIA, one ~110 s run (#4681): 5.21 GB of renderer compares and
66.07 GB of compute-cache compares, of which 5 renderer compares found a change. The always-on
`host-copy-per-flip` and `host-copy-pressure` alarms (`src/diagnostics/perf/perf_alarm_rules.cpp`
around `:560-590`) already name `draw-buffer-stage` as a cause.

**What other implementations do** (reference designs, verification-only; nothing is ported).
- KytyPS5 (`038d2257de85`) keeps guest buffers in Vulkan buffers and publishes each page's buffer
  device address in a device-side page table (`src/graphics/host_gpu/renderer/cache/bufferCache.cpp:69-81`,
  `cache/streamBuffer.cpp:92`, `:113`), bound as a `BdaPagetable` descriptor
  (`renderer/pipeline/descriptors.cpp:1137-1144`).
- shadPS4 (`afde63182a38`) has the same shape: a device-local BDA page table
  (`src/video_core/buffer_cache/buffer_cache.cpp:81-87`, `:330`) read by recompiled shaders
  (`src/shader_recompiler/backend/spirv/spirv_emit_context.cpp:1207`). A grep of its current
  `src/video_core` finds no host-pointer import. (Unverified, from memory: an earlier shadPS4
  imported guest memory with `VK_EXT_external_memory_host` and later dropped it; DXVK and
  vkd3d-proton use BDA for their own buffers but do not import application memory.)
  Both reference designs therefore copy into device memory and use BDA for *addressing*, not for
  avoiding the copy. Agreement of two secondary implementations is evidence class (3): a reason to
  test, not a result.

**Constraints any zero-copy path has to meet.**
- *Alignment.* `VkImportMemoryHostPointerInfoEXT` needs pointer and size aligned to
  `minImportedHostPointerAlignment` (commonly 4 KiB; reported larger on some drivers -- query it).
  A guest buffer is not page-aligned, so an import covers the enclosing pages and binds at an offset.
- *Which memory imports.* Windows guest direct memory is a sparse-file section mapped as views
  (`src/hle/memory/hle_kernel_mem.cpp:4966-4985`, `:5244`, `:5522`; ADR 0032), and #4681 found
  100% of compared bytes in section views. Whether NVIDIA and AMD Windows drivers import a
  section-view pointer is not established; `vkGetMemoryHostPointerPropertiesEXT` answers it per
  pointer and must be asked, not assumed.
- *Linux/AMD cannot import guest direct memory as host memory -- this is known, not open.* RADV
  implements `VK_EXT_external_memory_host` through amdgpu userptr, and libdrm's
  `amdgpu_create_bo_from_user_mem` (`amdgpu/amdgpu_bo.c`, primary evidence) always passes
  `AMDGPU_GEM_USERPTR_ANONONLY`; the kernel then refuses (`-EPERM`) any VMA backed by a file.
  prosper's Linux guest direct memory is a memfd (`hle_kernel_mem.cpp:1552`,
  `prosper_memfd_create("prosper-dmem")`) mapped `MAP_SHARED` (`:1617`, `:1681-1707`), so every
  guest mapping has a `vm_file` and a host-pointer import would be refused for essentially every
  guest buffer. Two independent secondary implementations hit the same wall and moved to udmabuf
  (evidence class 3/4 only): AnyPS5 PR #943 (<https://github.com/boykopovar/AnyPS5/pull/943>) and a
  shadPS4 fork issue (<https://github.com/kaaburgh/shadPS4/issues/9>).
  The Linux/AMD route is therefore different: create a `udmabuf` over the existing memfd and import
  it as a dma-buf (`VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf`). Its own
  constraints are **unverified** and must be checked before relying on it: udmabuf requires the
  memfd to be sealed with `F_SEAL_SHRINK`, ranges are page-granular, and the process needs access
  to `/dev/udmabuf`. **Absent that route, Linux/AMD gets only Stage B plus the page-tracking half
  of this ADR**, not the copy removal.
- *macOS/MoltenVK.* `minImportedHostPointerAlignment` follows the host page size, 16 KiB on Apple
  silicon. macOS is out of scope for this ADR; Stage A stays off there.
- *Aliasing.* One physical guest page can appear at several VAs (section views, ADR 0032). An import
  is per VA; a write through one alias must invalidate every alias's consumer.
- *Coherence.* Imported memory is host-coherent only if the chosen type says so; a GPU read of
  host memory crosses PCIe on every access, so a buffer read many times per frame can be slower
  in place than copied once to device-local memory.
- *Lifetime.* Unmapping or remapping guest memory under an import that a submitted command still
  reads is a use-after-free on the GPU. Import lifetime must follow the guest mapping and the
  pipelined retirement of ADR 0009 (`PERF-P8`).
- *Dirty signal.* Deciding "copy again" or "still valid" without a compare needs page tracking:
  `PERF-P9` (ADR 0010) and, on Windows, ADR 0032.

## Decision

1. **Staged, measured, and switchable.** Each stage lands behind a guest-invisible
   host-capability switch, default off, and is promoted only by a same-binary A/B
   (`.claude/skills/perf-change/`) on the reference workloads (ADR 0016) plus *Black Flag* on
   Windows/NVIDIA. A stage that does not move `res_buffer_copy_ms`, the transfer counters or frame
   time is recorded in `## Ruled out` and removed.
2. **Stage A -- host-pointer import for buffers the GPU reads.** Where the device exposes
   `VK_EXT_external_memory_host` and `vkGetMemoryHostPointerPropertiesEXT` accepts the pointer, a
   read-only guest buffer range is imported (page-rounded, cached per host page run) and bound at
   its offset instead of copied. Anything the query rejects keeps today's copy, counted. On
   Linux/AMD the import is the udmabuf/dma-buf route in Context, not a host pointer; until that
   route's constraints are verified, Stage A is off there.
3. **Device-local copies only where measured cheaper.** Once Stage A is promoted, the default is read in place for
   buffers read about once per upload; a buffer whose reads per change exceed a measured threshold
   is copied once to device-local memory and kept there until page tracking says a page changed.
   The threshold is a measurement per vendor, not a constant chosen in review.
4. **One coherence model, tied to page tracking.** Every GPU-visible copy or import of guest memory
   registers its page run with the guest memory tracker (ADR 0010/0032). A tracked write
   invalidates every alias's consumer; a retained CPU snapshot exists only where tracking is
   unavailable, and that fallback is counted.
5. **Lifetime follows the guest mapping and stream order.** An import is released only after the
   last submit that references it retires (ADR 0009); a guest unmap or remap of an imported range
   waits for, or defers behind, that retirement and never frees memory a queued command reads.
6. **Stage B -- buffer device address where provenance is proven.** The recompiler may address a
   buffer by `VK_KHR_buffer_device_address` only when the descriptor's guest VA is proven to lie in
   one imported or resident range for the whole draw -- the descriptor-provenance proof of ADR 0027
   (`GPU-4`); otherwise it keeps a descriptor binding. This does not compete with ADR 0030: the
   descriptor heap remains where descriptors live, and BDA is used only for raw buffer accesses
   whose range is proven in bounds; every other access goes through the heap.
   BDA is an addressing change that removes descriptor churn; it does not by itself remove a copy.
7. **Spec rule.** This ADR adds `PERF-P14` (`docs/spec/performance.md`), proposed.

## Consequences

- The largest measured CPU cost in the collapsed *Blue Prince* regime, and the `memcpy`/`memcmp`
  share in *GTA V*, become removable rather than optimisable, and the retained CPU snapshot per
  resident buffer goes away where tracking exists.
- Harder: lifetime now spans guest memory management and the GPU timeline; alias invalidation
  must be complete; driver import support becomes a per-host capability probed at startup.
- GPU-side cost moves: reads of host memory are PCIe-bound. Item 3 exists because this can make
  a frame slower, and only the A/B can say which way it goes on a given title.
- Enforced by the `host-copy-pressure` / `host-copy-per-flip` alarms and the transfer counters
  (fewer bytes is the visible effect), the ratchet's `vk-object` rule (imports are cached, never
  per draw, `PERF-P2`), and review of lifetime against ADR 0009, which no text scan can see.

## Alternatives considered

- **Status quo: copy, and compare to validate.** Measured above as the dominant setup cost in
  three titles; `PERF-P5` is already recorded as violated by it.
- **Bigger resident caches.** A larger cache raises the hit rate but every hit still pays a full
  compare without tracking (gaps doc § 3), and memory grows with the guest's working set.
- **Persistently mapped staging with diff-copies.** Cheaper than a fresh copy, but still a copy
  plus a compare; it is the fallback this ADR keeps where import is refused, not the target.
- **BDA first, as KytyPS5 and shadPS4 do.** Both still copy into device memory; adopting BDA alone
  changes addressing and leaves the measured copy in place, so it is Stage B, not Stage A.

## Migration order

1. Probe and count: report `VK_EXT_external_memory_host`, `minImportedHostPointerAlignment` and
   per-pointer `vkGetMemoryHostPointerPropertiesEXT` acceptance for the bytes `DrawBufferStage` and
   `res_buffer_copy_ms` cover, on Linux/AMD (udmabuf route) and Windows/NVIDIA. No behaviour change.
2. Page tracking that validates without a compare (ADR 0010, ADR 0032) where it is missing.
3. Stage A behind its switch for read-only graphics buffers; A/B; then compute read-only buffers.
4. Device-local promotion by measured reads-per-change (item 3).
5. Stage B in the recompiler, after the descriptor-provenance proof of ADR 0027 (`GPU-4`) exists.
6. Delete each switch once its default is settled, with the verdict recorded.

## Open questions

- Do NVIDIA and AMD Windows drivers accept pointers into `MapViewOfFile` section views? If not,
  Stage A on Windows needs guest direct memory backed differently, which is its own ADR.
- Does write-protect page tracking (`mprotect` on Linux) interact badly with pinned or imported
  pages -- e.g. a driver MMU-notifier registration invalidating and revalidating the import on
  every arm? Low confidence that it costs anything; step 3's A/B measures it.
- How are GPU *writes* into imported guest memory ordered against guest CPU reads (writeback
  today goes through `guest_write_watch_notify_gpu_write`, `src/gpu/execute/gpu_executor.cpp:12567`)?
  This ADR covers reads only.
- Is `VkPhysicalDeviceVulkan12Features::bufferDeviceAddress` available on every supported target,
  including lavapipe in CI?

## Approval

The project owner accepts or rejects this ADR. Acceptance moves `PERF-P14` to `accepted` and
unblocks migration steps 3-5; step 1 is measurement only and may land while this is proposed.
