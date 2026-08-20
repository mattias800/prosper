# `execute` — submission and execution

Where a decoded draw or dispatch becomes real GPU work.

- `gpu_executor` — the executor: builds each stage's resource table, runs the scalar const-fold that
  recovers descriptors the shader header does not declare, and issues the work. The largest file in
  the tree.
- `gpu_execute.hpp` — the shared contracts, including **`SrtUse`**: a descriptor use recovered by the
  const-fold, keyed by the `s_load` immediate byte offset. Read this before assuming prosper cannot
  see a descriptor channel.
- `gpu_dependency_graph` — ordering and dependencies between submitted work.
- `mb3_freelist` — answers "is this guest pointer a free MallocBinned3 block", which the executor asks
  before trusting a pointer. Here rather than in `resources/` because it is a question about
  *trusting* an address, not about binding one.

The const-fold is the part most often mistaken for absent. It resolves descriptors loaded with
`s_load_dwordx4/x8 sN, s[ptr:ptr+1], <imm>` from a user-data table and publishes them with
`srt_offset` = that immediate.
