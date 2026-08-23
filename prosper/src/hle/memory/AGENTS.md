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

**And a tracked reservation is NOT inert — on Linux it is a lazy-commit target.** The SIGSEGV
handler treats a fault above `0x1000000000` inside one as "the guest touched a page it believes it
committed" and backs it with a 64 KiB anonymous read/write page. Several titles' allocator bring-up
depends on that, so it stays the default — but it means a range you reserved to hold a *particular*
kind of memory will be silently filled with a different kind the moment anything touches an unmapped
part of it, and the guest cannot tell. Register such a range as declining lazy commit
(`g_amm_no_lazy_commit_base`, which makes `prosper_reserved_range_state` answer **4** instead of 1)
so the touch faults where it happened. That function's return values are one contract shared by both
platform arms and they are **not** dense — 3 refines "committed" and belongs to the Windows arm,
4 refines "reserved" — so read its table before adding a value. This is the thing a newcomer gets wrong here, and it was found in
review rather than by testing, because every symptom of getting it wrong looks like a bug somewhere
else.

**The pool's physical offsets have to fit inside the size prosper advertises.** `kDmemBase` is where
the direct-memory pool starts and `kDmemTotal` is both its length and what
`sceKernelGetDirectMemorySize()` returns — but a guest's `[searchStart, searchEnd)` window is
expressed in the offset space that *advertised size* defines, and `searchEnd` is routinely the
advertised size itself. So a base of B places the pool's last B bytes at offsets the guest is
structurally unable to name, and a guest that partitions the whole budget loses its tail with a
correct-looking ENOMEM. That is not hypothetical: at B = 0x10000000 it cost *Metaphor: ReFantazio*
its boot, and it cost it in a place nobody would look — the guest handled the ENOMEM by building a
resource reader over a null buffer and then byte-reversing 4 GiB of its own heap (#2934,
`docs/METAPHOR_STATUS.md`). B is now **64 KiB** — small enough that the unreachable tail is
negligible, non-zero so a successful allocation never returns physical offset 0, and a multiple of
the **Windows allocation granularity**, which is not optional: the Windows arm maps its section at
`phys - kDmemBase`, and `map_section_view`'s `MapViewOfFileEx` fallback depends on that offset
staying congruent to `phys` mod 64 KiB.

**Raising `kDmemTotal` does not buy the tail back** — `sceKernelGetDirectMemorySize()` returns
`kDmemTotal` verbatim while the pool is `[kDmemBase, kDmemBase + kDmemTotal)`, so raising it moves
the advertisement and the pool together and leaves exactly the same B bytes out of reach. (An
earlier version of this paragraph said the opposite; the `PROSPER_DMEM_BUDGET_MB` row in
`docs/METAPHOR_STATUS.md`'s `## Ruled out` is the measurement that kills it.) The only ways to
change the reachable amount are to move B or to **decouple** the two — advertise
`kDmemBase + kDmemTotal`, or size the pool at `kDmemTotal + kDmemBase`.
