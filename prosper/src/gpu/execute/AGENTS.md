# `execute` — submission and execution

Where a decoded draw or dispatch becomes real GPU work.

- `gpu_executor` — the executor: builds each stage's resource table, runs the scalar const-fold that
  recovers descriptors the shader header does not declare, and issues the work. Large; navigate it by
  symbol rather than by reading it.
- `gpu_execute.hpp` — the shared contracts, including **`SrtUse`**: a descriptor use recovered by the
  const-fold, keyed by the `s_load` immediate byte offset. Read this before assuming prosper cannot
  see a descriptor channel.
- `gpu_dependency_graph` — ordering and dependencies between submitted work.
- `mb3_freelist` — answers "is this guest pointer a free MallocBinned3 block". **Its callers are
  elsewhere**: `src/gpu/pm4/command_processor.cpp` (21 references), `src/hle/graphics/hle_agc.cpp`
  (4), `src/hle/kernel/hle_kernel.cpp` (3) and `src/host/image/exec_image_linux.cpp` (1, through
  the weak `prosper_mb3_is_pool_candidate`) —
  `gpu_executor.cpp` does not reference it. Note `src/hle/dispatch/dispatch.cpp` is **not** a caller: its only
  `mb3` token is `g_mb3_arm_hook`, the `PROSPER_MB3WATCH` write-watch hook, which is a different
  mechanism and catches a loose `mb3_` grep. So
  its placement here is historical rather than a coupling claim, and it is a candidate for moving.

The const-fold is the part most often mistaken for absent. It resolves descriptors loaded with
`s_load_dwordx4/x8 sN, s[ptr:ptr+1], <imm>` from a user-data table and publishes them with
`srt_offset` = that immediate.
