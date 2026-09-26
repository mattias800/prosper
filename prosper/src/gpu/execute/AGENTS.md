# `execute` — submission and execution

Where a decoded draw or dispatch becomes real GPU work.

The existing immutable compiled-shader entry also owns its compute trip-witness analysis result.
Return that fact only with the exact selected module; replacement or transformation needs fresh
analysis. Dispatch-specific guest-GDS exclusion remains outside the cache. Cold compilation,
bypass and admission refusal still derive the result from emitted words, and early refusal clears
the output. `PROSPER_NO_COMPUTE_WITNESS_CACHE` restores analysis on requested warm results for
comparisons; `compute_witness_analyses` counts actual cache-entry-point parser invocations.

- `gpu_executor` — the executor: builds each stage's resource table, runs the scalar const-fold that
  recovers descriptors the shader header does not declare, and issues the work. Large; navigate it by
  symbol rather than by reading it.
- `gpu_execute.hpp` — the shared contracts, including **`SrtUse`**: a descriptor use recovered by the
  const-fold, keyed by the `s_load` immediate byte offset. Read this before assuming prosper cannot
  see a descriptor channel.
- `index_expand` — the guest's validated 16-bit index range widened to the 32-bit indices the
  backend uploads, and the maximum that sizes the vertex buffer. Two things about it are easy to
  get wrong and both are load-bearing. The maximum must be reduced from the **same** loaded values
  that are stored, because the guest may be rewriting the buffer concurrently and the caller sizes
  an allocation from the returned maximum. And the read must stop exactly at `count`: the range is
  validated only as `guest_readable(addr, n * esz)`, so a legitimate draw can end on a mapping
  edge, which `tests/gpu/execute/test_index_expand.cpp` drives against a `PROT_NONE` page.
  The AVX2 kernel beside it is **not** an optimization and **nothing in the emulator calls it** —
  the portable loop is already auto-vectorized, and given the same ISA the compiler produces a
  wider loop than the intrinsics do. It survives only so that `test_index_expand --bench` keeps
  the falsifying A/B executable; see `docs/OUTER_WILDS_STATUS.md` § Ruled out before spending any
  time here. `PROSPER_INDEX_EXPAND_STATS=1` reports the index volume that would have to be large
  for any of this to matter.
- `gpu_dependency_graph` — ordering and dependencies between submitted work.
- `host_read_barrier` — the availability half of a GPU→CPU readback: the `HOST_READ`/`HOST_BIT`
  dependency that a fence wait does **not** perform (#2944/#3249). Header-only and deliberately
  backend-agnostic, because both Vulkan backends need it and they live in different trees — the
  offscreen render backend (`tests/fixtures/render_runner.h`, which re-exports it into
  `prosper::test`) and the live compute backend (`frontends/shared/live/live_compute.cpp`). The
  *other* half, `vkInvalidateMappedMemoryRanges`, is not here: it belongs with whichever allocator
  can hand out non-coherent memory, which today is only the render backend's.
- `mb3_freelist` — answers "is this guest pointer a free MallocBinned3 block". **Nothing in this
  folder calls it**: `gpu_executor.cpp` has no reference, and the callers are
  `src/gpu/pm4/command_processor.cpp`, `src/hle/graphics/hle_agc.cpp`,
  `src/hle/kernel/hle_kernel.cpp` and `src/host/image/exec_image_linux.cpp` — the last through the
  weak `prosper_mb3_is_pool_candidate`. Its placement here is therefore historical rather than a
  coupling claim, and it is a candidate for moving.

  Re-derive that list rather than trusting it, and grep the module's declared symbols rather than the
  substring `mb3_`:
  ```bash
  grep -rlE "$(grep -oE '\b(prosper_)?mb3[a-z0-9_]*' src/gpu/execute/mb3_freelist.hpp \
               | sort -u | paste -sd'|')" --include='*.cpp' src/ | grep -v mb3_freelist
  ```
  Two traps that grep avoids. `src/hle/dispatch/dispatch.cpp` looks like a caller under a loose
  `mb3_` pattern but is not one — its only token is `g_mb3_arm_hook`, the `PROSPER_MB3WATCH`
  write-watch hook, a different mechanism. And no reference COUNT is quoted here, because
  `command_processor.cpp` defines its own `mb3_freelist_guard` / `mb3_freelist_report` statics: a
  substring count says 21 where module references number 8, and three defensible counting rules give
  three different answers.

The const-fold is the part most often mistaken for absent. It resolves descriptors loaded with
`s_load_dwordx4/x8 sN, s[ptr:ptr+1], <imm>` from a user-data table and publishes them with
`srt_offset` = that immediate.
