# UE4 (PPSA17942 "DOLL") — APR / IoStore read bring-up (path to the first frame)

Status as of this doc: the title boots from garbage-byte execution all the way through memory
bootstrap, platform-services init, and into UE's **IoStore asset layer**. The MallocBinned
allocator is deterministically healthy (0 canary fatals) and the engine's own error handler runs.

Boot recipe: `PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 ./boot_trace <PPSA17942-app0>`
(GUEST_FS: UE MallocBinned TLS caches read `%fs`; NULL_PAGE: UE's FP-chain backtrace walker
derefs the null chain terminator).

## What already works (merged / on feat/ue4-apr)

- PS5 SELF-variant loading (magic `0xEEF51454`), the real direct-memory model, `sceKernelBatchMap`,
  memfd physical aliasing, the `libSceAmpr` init/begin/setBuffer trio, `sceKernelGetdents`, RTC
  calendar, `pthread_create_name_np`, etc. (PR #47, #49.)
- **`sceKernelAprResolveFilepathsToIdsAndFileSizes`** (commit 3d51097): the APR entry point.
  ABI (live-captured): `(const char** paths, int count, uint32_t* outIds, uint64_t* outSizes,
  uint32_t* outFlags, int)`. Called once per pak container with `/app0`-translated paths
  (`global.utoc/.ucas`, `pakchunkN-ps5.{pak,utoc,ucas}`). Implemented in `hle_file.cpp`
  (`f_apr_resolve`): stat the host file, assign a 1-based id, record `id -> host path`, exposed as
  `std::string prosper_apr_path_for_id(uint32_t id)`. This eliminated the entire MallocBinned
  corruption class (the corruption was a *cascade* of resolve previously returning EINVAL, which
  made the async-read pipeline run on garbage ids and wild-write over allocator blocks).

## The current wall — precisely located

After resolve, the engine builds an Ampr command buffer and issues a page read, which fails:

```
LowLevelFatalError [File:Unknown] [Line: 93]
Apr read failure 24 at CB offset 40
```
(format string: eboot vaddr `0x8052580`, UTF-16; then `LowLevelFatalError` → `int $0x45` abort at
`libc.prx+0x48e0`.)

- The read executes through a C++ virtual method: `eboot 0x23d7232  call *0x28(%rax)` returns
  `al == 0` (failure). Caller chain: `0x23d7235 ← 0x23d6858 ← 0x23d941b ← 0x23d448f ← 0x23d4442
  ← 0x4552136`.
- "CB offset 40" = the read command sits at byte `0x40` of the Ampr command buffer.
- "24" is APR's own read-command error code (NOT `EMFILE`: only ~11 host fds are open vs a 10240
  limit — fd exhaustion is ruled out).
- Ruled out this session, with measurements: begin-cursor population (no effect), fd exhaustion,
  larger dmem pool, aliasing/mirror machinery (all were cascades of the EINVAL resolve).

## CRUCIAL: the read is DEVICE-LEVEL DMA, not an interceptable import

Confirmed by capturing all 28 `libSceAmpr` import stubs *after* real resolve reaches the read:
the page read invokes **none of them**. Only the 3 setup imports fire per APR object
(`8aI7R7WaOlc` init, `a8uLzYY--tM` begin/cursors, `N-FSPA4S3nI` setBuffer). Nor does it use our
`pread` (no pak file is `pread` before the failure), nor a raw syscall (`int $0x45` at
`libc.prx+0x48e0` is the *abort* path after the failure, not the read).

Therefore the read is executed by the **Ampr DMA engine at the device level**: the guest writes
read command(s) into its command buffer (the failing one is at CB offset `0x40`), rings a
doorbell via a plain memory write to a device-register/MMIO region, and the engine DMAs file
bytes into the `setBuffer`'d destination (e.g. `0x1720df0000`). We back none of that, so the
destination stays empty and the guest's reader returns error `24`.

**Implication for step 1 below:** it cannot be done as import HLE. Two viable routes:
- (A) **Doorbell watchpoint + command-buffer executor.** Identify the doorbell MMIO address (the
  memory write the guest does right before it polls for completion), trap writes to it
  (`PROSPER_HWWATCH`-style, or a guarded page), parse the command buffer (decode the read command
  at offset `0x40`: source id/handle, file offset, dest VA, size), `pread` from
  `prosper_apr_path_for_id(id)` into the dest, then set the command's completion/result field so
  the guest's poll succeeds. This is the faithful model.
- (B) **Patch the guest reader method** (`B.vtable[+0x28]`, reached at `eboot 0x23d7232`) to a
  trampoline that reads the 0x90-byte descriptor (`rsi`/`r12`), performs the `pread`, and returns
  success — bypassing the DMA emulation entirely. Lower-fidelity but far smaller; a good first
  milestone to get past IoStore and expose the next wall (decompression / RHI).

Route (B) is the recommended first step: it needs only the descriptor layout (dump the 0x90 bytes
at the `0x23d7232` call — offset, dest, size, id) and the existing `prosper_apr_path_for_id` hook.

### Read-submit function IDENTIFIED and initial executor implemented

Tracing `readFile` (guest `0x59b6110`, a thin wrapper) → import thunk `0x669ae90` → GOT
`0x409490678` → stub `0x600028810` = **`libSceAmpr::mQ16-QdKv7k`** (stub #1728 — *beyond* the 28
Ampr stubs the earlier sweep covered, which is why it was never seen). NOT one of the setup trio.

Live-captured call: `mQ16(readReq, &cur1, &cur2, count, outDescBuf, descSize=0x90)`. The
read-request object (`readReq`, on stack) layout: `+0x18` = destination buffer VA
(`0x7fffeb400000`), `+0x20` = second buffer, `+0x30` = total byte count (`0x285` = 645 = exactly
`global.utoc`'s size), `+0x28` = post-failure result word (`0x24`=err 36, `0x28`=CB offset 40).
The file id is NOT directly legible in `readReq`.

Initial executor (`hle_file.cpp` `f_apr_read_submit`, registered for `mQ16-QdKv7k`): read
`*(readReq+0x30)` bytes from the resolved host file into `*(readReq+0x18)` via `pread(offset 0)`.
File matched by size (boot-container sizes are distinct; the id isn't in the request). This is the
first milestone — get the (uncompressed) global TOC to load and expose the next wall. Iterate:
non-zero read offsets and the file-id linkage will be needed for the compressed `.ucas` chunk
reads that follow.

### STATUS: read executor working end-to-end — the global TOC now reads AND parses

Four commits on `feat/ue4-apr` implement the read path and get the engine through IoStore TOC
handling entirely:
1. `f_apr_read_submit` (mQ16-QdKv7k): synchronous `pread` of the resolved file.
2. Clear the completion-status word at `req+0x28` on success (else the pre-seeded `0x24` "Apr read
   failure 24 at CB offset 40" still fatals — check decoded at eboot `0x22738a5`, `rbx = req-8`).
3. Destination is `req+0x20` (the mapped guest data buffer, `0x10xxxxxxxx`), NOT `req+0x18` (a
   small stack scratch — the earlier 645-byte write there smashed the stack; crash was a `ret`
   with the TOC magic in `rbp`). `req+0x30` = byte count.

Verified: `read-submit global.utoc -> size=645 got=645 OK`, `aprreadfail=0`, TOC parses, and the
engine advances FAR deeper — new crash at **`eboot+0x2316c91`** via a 17-frame IoStore/IoDispatcher
backtrace (`0x22fce69 <- 0x245f78 <- 0x22dfbb5 <- 0x4551748 <- 0x45529c2 <- 0x4552317`), a static-
init-guard-style function (`rax`=0x2001c10000 a BatchMap buffer, null deref, `rdx`=0x50). Only ONE
read happens before this — the engine crashes *processing* the parsed TOC, before the next read.

Field map of the APR read-request object (live `req` dump):
`+0x00` count(1) · `+0x08` stack ptr · `+0x10` fn/vtable · `+0x18` **begin out-slot 1** ·
`+0x20` **begin out-slot 2 = data pointer** · `+0x28` status(err\|cboff) · `+0x30` **byte count** ·
`+0x38` stack canary · `+0x40` stack ptr.

### RESOLVED: the `eboot+0x2316c91` freelist crash was a stale, never-populated begin-cursor

Root cause (proven by a full-run `PROSPER_HWWATCH_ABS` write history of the "dest" and by
`PROSPER_AMPRLOG` arg/out-slot capture):

- The APR read flow inits its request through the SAME Ampr NIDs as the page-map path:
  `8aI7R7WaOlc(req, cbSize=0x40, 0, pathStruct, allocCtx=0x2001c10060, 3)` then
  `a8uLzYY--tM(req, &req->0x18, &req->0x20, 0, 0x2a, 0)` then `mQ16(req, …)`. The `begin` call's
  two pointer args are OUT-SLOTS the library must populate; the engine reads the file data back
  through `*(req+0x20)` after completion — the library chooses the data location, not the engine.
- Our `begin` HLE returned 0 WITHOUT writing the slots, so `req+0x20` kept stale stack garbage
  that happened to point at a FREED 0x50-byte block (`0x1080e10a50`) of the engine's own live
  path-string pool (freelist head `0x2001c10080`, carve-linker `eboot+0x230f1d8`, push
  `eboot+0x2316e0c` — the push LINKS `node->next := head`, it does not zero). The HW watch showed
  the block sitting free on the freelist at read time; writing the 645-byte TOC over it clobbered
  the free nodes' next pointers with the TOC magic → the later pop at `eboot+0x2316c80` popped
  `next == 0x2d3d3d2d2d3d3d2d` and faulted at `+0x2316c91`.
- Why the TOC still "parsed": both our read HLE (write) and the engine (read-back) dereferenced
  the SAME stale slot, so the data round-tripped consistently — only the pool underneath was
  corrupted. There was never a prosper memory-model overlap: MEMLOG showed the page
  `0x1080e10000` BatchMap-committed exactly once, no alias, no mirror (dmem/batchmap are clean).
- Fix: `k_ampr_begin` now populates both out-slots with a prosper-owned 16 MiB staging buffer
  (modeling the library-owned staging on real HW); a zero-length submit also clears the `req+0x28`
  status. The crash is gone and the engine advances through global.utoc parse, `.ucas` open,
  saved-paks probing, and pakchunk resolve.

### RESOLVED: the "multi-segment" 3rd submit was frame residue — a3 IS the file id

The `count=0x4 / total=36144292` reading of the 3rd submit was wrong: those request-frame slots
are pure residue at that callsite (`req+0x00 = 0x402316d88` and `req+0x08 = 0x402316d50` are
eboot CODE addresses; `req+0x30 = 0x22784a4` — 36,144,292 — recurs there across runs while
`req+0x28` holds a stack pointer). The pak-read wrapper (descSize `0xdd`, a heap desc buffer)
simply doesn't initialize the frame the way the utoc-read wrapper (descSize `0x90`) does.

The real contract, proven over 6 live reads at both callsites (A/B-tested by rewriting the
record and watching the engine follow it):

- **`a3` is the APR file id** from resolve — a3=1 global.utoc, 3 pakchunk2-ps5.utoc,
  4 pakchunk1-ps5.pak (twice), 5 pakchunk1-ps5.utoc, 6 pakchunk0-ps5.pak (twice),
  7 pakchunk0-ps5.utoc. `mQ16(reqFrame, a1=&rec.scratch, a2=&rec.data, a3=fileId, descBuf,
  descSize)`.
- **`a2` points at a completion record the LIBRARY fills**: `[0]` data pointer (library-chosen
  location — begin's staging cursor until submit publishes the final buffer), `[+8]` status
  (0 = success; on failure `{low32 err, high32 CB offset}` — exactly the `24|40` / `2b|48`
  pairs the "Apr read failure" fatal prints; checked at eboot `0x22738a5`), `[+0x10]` bytes.
- **Byte count = the whole resolved file** (the engine got sizes from resolve; nothing in the
  record is a trustworthy input).

`f_apr_read_submit` now resolves by id, reads the whole file into a per-read prosper-owned
buffer (never freed — the engine treats the pages as library-owned), and completes the record
through `a2`. Result: the `eboot+0x22ebe7f` failure-path fault is gone and ALL containers read
clean (global 645, pakchunk1 pak 339 + utoc 32,485, pakchunk0 pak 2,062,854,722 (!) + utoc
14,271,592, empty pakchunk2 completes trivially with size 0). The whole IoStore container-mount
phase finishes; the engine walks on to `globalshadercache-sf_ps5.bin` and a (legitimately
absent) `doll/doll.uproject` probe.

**Current wall (deeper again): `GEngineLoop.PreInit Failed!`** →
`sceKernelDebugRaiseExceptionOnReleaseMode(0xa0020005)`. No fault — the engine reports init
failure itself. Suspects, in order: (1) whole-file-from-offset-0 semantics for the TWO pak reads
per pak (descSize `0xdd`/`0xde` — real HW likely reads the pak index/footer at an offset carried
somewhere we don't decode yet; if the engine expects the published pointer to start AT its
requested offset, the FPakFile index parse fails silently and no containers mount → PreInit
can't find the engine inis); (2) the `resolve MISS` output contract (we write id=0/size=0 —
verify against real errno/flag semantics); (3) `.ucas` chunk reads never happen before the
failure — possibly downstream of (1). Watch item for (1): the 2 GiB pak is read whole TWICE
(two leaked 2 GiB buffers) — finding the offset/size input fixes both correctness and cost.
Then decompression: `.ucas` chunks MAY be Oodle-compressed (check `compressionMethodNameCount`;
`global` was 0/uncompressed).

Leads for (1), captured live (`PROSPER_APR_DIAG` + the now-logged four extra Ampr NIDs):

- `a5` of the submit looks like the intended READ SIZE, not a descriptor size: utoc reads pass
  `0x90` = exactly the utoc TocHeaderSize, pak reads pass `0xdd` = 221 = the FPakInfo footer
  size (61 + 5×32 compression-method names) and then `0xde` (a second footer-version probe) —
  i.e. the pak reads want the LAST `a5` bytes (`filesize-221`), which whole-file-from-0 does not
  deliver at the published pointer.
- For the pakchunk0 pak read the desc buffer (`a4=0x20018f0000`, BatchMap memory) is a
  structured ENGINE-PREWRITTEN array (~0x28-byte records): `{seq 0x456/0x467/0x485/0x4a7, small
  count 0/1/3/4, eboot ptr 0x4058d26b0, dest VA 0x10e0dfff8000, {high32 seq | low32 0xffffffff}
  pending markers}` — very likely the real per-read command/completion descriptors. Decoding
  this array is the next concrete step for offset-correct pak reads.
- The four other Ampr NIDs the flow calls fire AFTER each submit with constant args
  (`baQO9ez2gL4(req,0,0,0,0x40|0,0)`, `ULvXMDz56po(req,stack,0,0,0x45,4|7)`,
  `Qs1xtplKo0U(req,&rec.scratch,&rec.data,0,0x45,0xe)`, `GuchCTefuZw(req,stack,0,0,0x45,0xf)`) —
  teardown/end-of-request, not the offset carrier (arg capture via `PROSPER_AMPRLOG`).

### SOLVED (2026-07-08, feat/ue4-apr-read): pak mounting works — the read is (offset, size) and offset is the 7th arg

Two facts closed the loop:

1. **NID names recovered by brute-force** (`nid_hash(name)` over a generated libSceAmpr corpus —
   the algorithm in `nid.cpp`): the read is **`sceAmprAprCommandBufferReadFile`** (`mQ16-QdKv7k`),
   and the flow is `sceAmprCommandBufferConstructor`(`8aI7R7WaOlc`) →
   `sceAmprAprCommandBufferConstructor`(`a8uLzYY--tM`) → `sceAmprCommandBufferSetBuffer`
   (`N-FSPA4S3nI`) → `sceAmprAprCommandBufferReadFile` → `sceAmprCommandBufferReset`
   (`baQO9ez2gL4`) → `sceAmprAprCommandBufferDestructor`(`Qs1xtplKo0U`) →
   `sceAmprCommandBufferDestructor`(`GuchCTefuZw`). "ReadFile" being a command *builder* means
   **every read parameter is an argument** of that one call.

2. **`ReadFile` has ≥7 args; the FILE OFFSET is the 7th, passed ON THE STACK** — which the plain
   6-arg HleFn signature never saw (the exact blind spot that made pak footer reads look
   offset-less). An asm entry shim (`f_apr_read_submit_entry` in `hle_file.cpp`) snapshots `rsp`
   into a TLS slot so the handler reads stack args; it distinguishes the two stub tails (swap-stub
   `call` vs `jmp`) by whether `[rsp]` is in the stub region `[0x6_0000_0000, 0x7_0000_0000)`.

   Live-verified offsets (exact footer math, 8 reads, 3 file kinds): utoc header reads
   `(off=0, size=0x90)`; the two FPakInfo footer probes per pak `(off=filesize-221, size=221)` and
   `(off=filesize-222, size=222)` — e.g. pakchunk1 pak (339 B): `0x76`/`0x75`; pakchunk0 pak
   (2,062,854,722 B): `0x7af4a965`/`0x7af4a964`. So **`a5` = size, `arg7` = offset**; the read is
   a host `pread(fd, dst, a5, arg7)` clamped to EOF.

3. **The destination is `a4` (arg5), the caller's `dst` buffer — not only the completion record.**
   After fixing offset+size, PreInit *still* failed until the bytes were also written INTO `a4`:
   the utoc callsite consumes data via the record pointer, but the **pak-footer callsite reads its
   own `dst` buffer directly** (with only the side record published, the correct footer bytes were
   never seen, no index read followed, mount failed). `f_apr_read_submit` now
   `process_vm_writev`s into `a4` and publishes `record[0]=a4`; it falls back to a prosper-owned
   staging buffer only if `a4` is absent/unmapped.

**Result: `GEngineLoop.PreInit Failed!` is GONE. All containers mount.** The engine now issues the
real post-footer reads it previously never reached: the FPakFile **index** at the footer's
`IndexOffset`, then chunk/index-entry reads at learned offsets (e.g. pakchunk0 pak reads at
`0x7af0b825`, `0x7af3550c`, `0x78a559dd`, …), and it opens+resolves the `.ucas` (pakchunk0-ps5.ucas
id=10, 17.8 GB) — i.e. IoStore container mounting is complete. CONFIDENCE: HIGH (offset/size/dst all
verified on live reads; PreInit-passes is the end-to-end proof).

### NEW WALL: post-mount engine/RHI init — a null virtual call (racy, two faces)

Boot now reaches early UE engine init and dies in one of two timing-dependent spots (the RHI /
task-graph worker threads are now live, so it is nondeterministic):

- **Main-thread face (representative):** a null vtable call. At `eboot+0x24c75ef`
  `mov 0x28(%r15),%rdi ; mov (%rdi),%rax ; call *0x10(%rax)` the object's vtable slot `+0x10` is 0
  → control jumps to `rip=0` (`RUN ENDED kind=2, rip=0x0`; backtrace
  `eboot+0x24c75f2 ← +0x24c7234 ← +0x95a2 ← +0x7b26 ← +0x9c7 ← +0x508e ← +0xbf`, i.e. reached
  from CRT `_start`/main on the main thread). The neighbourhood registers CVars by name — the
  rodata strings loaded just before are `LoadModule  - `, and the sibling singleton-init block
  (`eboot+0x503b878`) registers `r.Shadow.CacheWPOPrimitivesError` / `r.ShowMaterialDrawEvents` /
  `r.RHIThread…` via `call *0xb0(%rax)` on a CVar-manager object. So this is **RHI / CVar / module
  bring-up**, past PreInit.
- **Worker-thread face:** a fault in host HLE code reading a small address (~`0x2936xx`, grows per
  run) — a worker racing the same half-initialized subsystem.

Next steps (in order):
1. Resolve the null vtable object at `eboot+0x24c75ef`: break there, walk `[[r15+0x28]]` and its
   vtable to see which interface method `+0x10` should be — likely a subsystem/module singleton
   the engine expects a prior init to have populated (an unimplemented Sony import returning 0
   where the engine stored the result as an object pointer, or a module we skip loading:
   `SaveData.prx` / `PSN.prx` / `PS5Util.prx` / `Il2cppUserAssemblies.prx` are all "skipping absent
   module"). Check whether one of those provides the missing vtable.
2. Then the `.ucas` chunk reads proper (asset registry / shader cache). Check each utoc's
   `CompressionMethodNameCount`: `global` was 0/uncompressed; the pakchunk `.ucas` chunks are
   likely Oodle Kraken/Mermaid and will need a decode path.
3. Then RHI/AGC init → first UE draw via the existing `execute_and_present`.

## Remaining work to the first frame (the subsystem)

This is genuine subsystem construction, sized (per `CROSS_ENGINE_UE4.md`) like a second-title
bring-up. In dependency order:

1. **Ampr command-buffer read executor.** Identify the guest read method (`vtable[+0x28]` of the
   reader object at the `0x23d7232` call — resolve it live: break there, read `rax`, deref
   `*(rax)+0x28`) and the Ampr submit/kick it invokes. The read command at CB offset `0x40`
   carries `(source id/handle, file offset, destination buffer VA, byte count)`. Implement the
   submit to parse each read command and `pread(open(prosper_apr_path_for_id(id)), dest, count,
   offset)` — filling the buffer and setting the command's success/result field so the guest's
   `vtable[+0x28]` returns `al != 0`. The `id -> host path` map is already in place.
2. **Decompression.** IoStore `.ucas` chunks are compressed (Oodle Kraken/Mermaid on PS5). The
   `.utoc` (645 bytes for `global`) is the table of contents; chunk entries carry a compression
   method + block layout. Either (a) decode uncompressed/`None`-method chunks first to get the
   engine further, or (b) integrate an Oodle decode path. Start with (a): many boot-critical
   chunks may be stored uncompressed; gate Kraken behind a later milestone.
3. **Config + engine init past IoStore.** Once the global TOC + a chunk read succeed, the engine
   parses `DefaultEngine.ini`/asset registry and proceeds to RHI creation.
4. **RHI / AGC bring-up.** UE's PS5 RHI drives the same `libSceAgc` GPU submission path the
   Messenger already uses. The prosper AGC→Vulkan executor (PM4 decode, RDNA2→SPIR-V, the live
   renderer) is in place and renders Messenger frames; the work is wiring UE's RHI submits into
   `execute_and_present` and handling any UE-specific render-state.
5. **First draw** → present via the existing path (`PROSPER_RENDER=1`, frame BMP dump).

## Instrumentation that works here

- NID identification: hash candidate names (`ps5-payload-sdk/sce_stubs/*.c` corpus) against the
  raw NID; the eboot's own UTF-16 assert/error strings name the failing subsystem (`strings -e l`).
- Live ABI capture: gdb breakpoints on the import stubs (`PROSPER_STUBDUMP` gives offsets;
  base `0x600000000 + off`). Pass all guest signals (`handle SIG… nostop pass`) or the first
  benign SIGSEGV wedges gdb.
- Caveat seen this session: batch-gdb breakpoints on guest RWX addresses sometimes don't fire
  cleanly (worker-thread timing); prefer breaking on a host symbol first, then arming guest bps.
