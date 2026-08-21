# `src/hle/memory` — the guest's address space

Everything the guest can do to its own memory through libkernel, plus the libSceAmpr command-buffer
constructs that also move memory around. If a Sony entry point answers "where does this live, how
big is it, and who may touch it", it belongs here.

The PS5 memory model this reimplements has three layers and they are not interchangeable:
**reserve** a virtual range (address space, no pages), **allocate** direct memory (physical pages as
an opaque pool offset), then **map** one into the other — or map *flexible* memory, which has no
physical identity at all. `sceKernelVirtualQuery`, `sceKernelDirectMemoryQuery` and the fault
handler's lazy-commit probe all answer from the tracking tables this folder owns, so a mapping that
happens without being tracked is a mapping the rest of prosper cannot see.

## The boundary against its siblings

* `host/memory/` is the *host* side — the write-watch, the readable-mapping interval set, the
  page-protection generation counter. This folder calls into it to publish what it just did; it
  never reimplements it.
* `hle/fs/` owns the APR **file read** half of libSceAmpr. This folder owns the command-buffer
  object itself (construct / set-buffer / reset / cursor) and the **AMM** memory-mapping half. The
  two share `AmprCbState` and the completion bookkeeping, which is why both live above the
  platform split in `hle_kernel_mem.cpp` rather than in either arm.
* `hle/kernel/` owns the event queues an APR completion is posted to.

## What a newcomer gets wrong here

**`hle_kernel_mem.cpp` contains two complete implementations of almost everything** — a POSIX arm
and a Windows arm, thousands of lines apart, separated by one
`#if defined(__linux__) || defined(__APPLE__)` near the top. Nothing local tells you which arm you
are reading, and reading one as though it were the whole file is a mistake this project has made
three times in a single session (#2376, #2384). Grep for a symbol and expect **two** definitions;
when you change a guest-visible answer, change both, because a capacity or a base address is a fact
about the guest's object rather than about the host prosper happens to run on (#1970, #2139).

**Never place a mapping without proving the target is free.** `map_at` and `map_phys_at` refuse a
`MAP_FIXED` over anything except prosper's own *uncommitted reservation* — that discipline is the
only thing standing between a wrong guest address and a silent overwrite of live heap, and both
issues that established it (#88, #107) presented as unrelated crashes hours later in another
subsystem. New mapping code goes through those two functions.

**A "reservation" is a real, tracked object.** `track(..., committed=false, ...)` plus a `PROT_NONE`
mapping is what makes a range later commitable in place; a bare `mmap(PROT_NONE)` that nobody
tracked is indistinguishable from somebody else's memory.
