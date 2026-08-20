# `execute` — submission and execution

Where a decoded draw or dispatch becomes real GPU work.

- `gpu_executor` — the executor: builds each stage's resource table, runs the scalar const-fold that
  recovers descriptors the shader header does not declare, and issues the work. Large; navigate it by
  symbol rather than by reading it.
- `gpu_execute.hpp` — the shared contracts, including **`SrtUse`**: a descriptor use recovered by the
  const-fold, keyed by the `s_load` immediate byte offset. Read this before assuming prosper cannot
  see a descriptor channel.
- `gpu_dependency_graph` — ordering and dependencies between submitted work.
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
