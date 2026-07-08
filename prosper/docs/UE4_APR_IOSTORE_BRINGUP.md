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
`+0x00` count(1) · `+0x08` stack ptr · `+0x10` fn/vtable · `+0x18` small stack scratch ·
`+0x20` **mapped data buffer (dest)** · `+0x28` status(err\|cboff) · `+0x30` **byte count** ·
`+0x38` stack canary · `+0x40` stack ptr.

Next: disassemble `0x2316c91` / `0x22fce69` — likely another null global/uninitialized IoStore
struct (a missing init call or a container/registry the TOC parse expected to populate). Then the
`.ucas` chunk reads begin (those MAY be Oodle-compressed — check each container's utoc
`compressionMethodNameCount`; `global` was 0/uncompressed).

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
