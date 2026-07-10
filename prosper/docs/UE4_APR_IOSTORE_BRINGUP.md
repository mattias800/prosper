# UE4 (PPSA17942 "DOLL") — APR / IoStore read bring-up (path to the first frame)

Status as of this doc: the title boots from garbage-byte execution all the way through memory
bootstrap, platform-services init, and into UE's **IoStore asset layer**. The MallocBinned
allocator is deterministically healthy (0 canary fatals) and the engine's own error handler runs.

Boot recipe: `PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 ./boot_trace <PPSA17942-app0>`
(GUEST_FS: UE MallocBinned TLS caches read `%fs`; NULL_PAGE: UE's FP-chain backtrace walker
derefs the null chain terminator).

## Frame loop reached; 0 draws = early-load present loop (issue #213, 2026-07-09)

DOLL now boots fully and runs a **stable frame loop**. Two blockers cleared this session:

1. **CRI Atom audio init crash (fixed).** DOLL's CRI Atom (ADX) middleware drives audio through
   **libSceAudioOut2** (PS5-only; no Kyty/shadPS4 surface). The generic unimplemented stub returned
   0 with out-params untouched, so CRI read an UNINITIALIZED context work-memory size, malloc'd it,
   and memset the result — when the stack garbage was unallocatable the MAIN thread died in
   `libc.prx+0x10556` (`mov %rdx,(%rdi)` inside memset) through the CRI region — the intermittent
   "1 flip then crash" (tail frames `0x276f7a0 <- 0xb008 <- ... <- 0xbf`). NIDs recovered by
   nid_hash brute force: `g2tViFIohHE`=sceAudioOut2Initialize, `t5YrizufpQc`=ContextResetParam,
   `pDmme7Bgm6E`=ContextQueryMemory. Contracts recovered by live capture (PROSPER_AUDIO2LOG):
   ResetParam param is 0x40 bytes; QueryMemory writes the work-mem size to `a1` (the guest hands it
   straight to ContextCreate); then UserCreate/PortCreate and a pump loop (PortGetState →
   PortSetAttributes → ContextAdvance → ContextPush) on a dedicated CRI server thread. Implemented
   as a **null-device backend** (Wine null-audio sense): real handle lifecycle + ContextPush pacing
   one grain of wall-clock per call (blocking-when-full HW semantics). hle_audio.cpp. CONFIDENCE MED
   (LOW on PortGetState layout — zero-filled 0x20).

2. **The 0-draws state is diagnosed as EARLY-LOAD, not a render bug.** With audio fixed DOLL runs
   146+ flips / 147 DCB submits over a 480s run. But the steady-state frame DCB is a fixed
   **444-dword pure vblank-sync control stream** — `ReleaseMem` (EOP fence) ×8, `WaitRegMem` ×3,
   `WriteData` ×3, `GetDataPacketPayloadAddress` ×3, plus the AGC patch-address NIDs
   (`0fWWK5uG9rQ`/`3KDcnM3lrcU`/`fPSCdQxgpSw`/`-KRzWekV120`/`tSBxhAPyytQ`). Across the **entire**
   post-flip run there are **0 `DcbSetCxRegistersIndirect` (ZvwO9euwYzc) and 0 `CreateShader`** — all
   2593 shaders are created during init, before the frame loop, and no draw setup happens after.
   The game builds all pipelines at load then parks in a present-only loop showing cleared frames.
   This mirrors The Messenger's earlier "scene activation" gap: the game is NOT idling for input
   (pad is never opened, #213 parallel finding) — it is in early-load / pre-world-activation,
   presenting blank frames while the GameThread has not yet activated a UWorld/level that submits
   geometry. libScePlayGo fires only twice at boot (not polled → not the content gate); PlayGo/
   SaveData/AJM all one-shot. The remaining gate is a GameThread world/level-activation step, not a
   missing GPU event — see #213 for the next-session trace target.

**Next hard blocker: racy worker-thread fault at `eboot+0x59949e4` (issue #222).** A libSceAgc
EUD-ring store (`mov %r14,(%rax)` where `%rax` = `obj+0x10 + idx*4`) faults on a guest worker
thread with a stale/small ring pointer — intermittently, after ~140 flips — terminating long runs.
Not on the draw path (frame DCBs carry no draws). Full disasm + repro in #222.

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

### SOLVED (2026-07-08, issue #88): the null virtual call was a SetBuffer heap clobber

Full causal chain, proven step by step (single-run reg-time vs crash-time dumps + a silent HW
write-watch):

1. The `eboot+0x24c75ef` `call *0x10(%rax)` is `IModuleInterface::StartupModule()` inside
   `FModuleManager::LoadModuleWithFailureReason` (the "LoadModule  - " boot scope) — loading
   **"Engine"** from `SCOPED_BOOT_TIMING("LoadPreInitModules")`. The rbp-chain backtrace hid the
   real fault: it is INSIDE `FEngineModule::StartupModule` at `eboot+0x503b72e/731`
   (`mov (%rbx),%rax ; call *0xb0(%rax)`), where `rbx` = a guarded function-local static caching
   `IConsoleManager::FindConsoleVariable(L"r.Shadow.CacheWPOPrimitives")` — which returned NULL,
   and the code calls `SetOnChangedCallback` on it with no null check (a sibling block does the
   same for `r.ShowMaterialDrawEvents`).
2. The CVar IS registered: the TAutoConsoleVariable static ctor (in the eboot's .ctors table —
   walked backward from `0x4094939e8` to the -1 sentinel by DT_INIT at eboot+0x10; all 1498 ran)
   called `RegisterConsoleVariable` successfully; the object sat in FConsoleManager's 3000+ entry
   map. But by StartupModule time the map entry's FString KEY DATA was ALL ZEROS in place —
   FindConsoleVariable can never match it.
3. The zeroing had NO writer (a HW write-watch on the exact key address never fired): it was a
   MAPPING REPLACED UNDER THE VA. `k_ampr_push_map` (sceAmprCommandBufferSetBuffer HLE) treated
   EVERY SetBuffer as "map fresh phys page at va (+ mirror at va-0x540000000)". The APR read
   flow also issues SetBuffer for an ALREADY-LIVE 0x4000-byte descriptor buffer
   (`va=0x15a0dfc000`), and the MIRROR (`0x1060dfc000`) MAP_FIXED'd fresh zero pages over live
   MallocBinned heap — the exact 16KB holding the console map's key strings.
4. Fix (hle_kernel_mem.cpp): the two SetBuffer flavors are discriminated by their own args —
   captured live: map flavor has `a3 != va, a4 = 0xffffffff sentinel`; existing-buffer flavor has
   `a3 == va, a4 = allocCtx, a5 = small flags` (13/13 and 3/3 in capture). The existing-buffer
   flavor is now a memory no-op. The worker-thread face was the same corruption seen from the
   other side.
5. Companion fix (hle_file.cpp): the read-submit dst write (`process_vm_writev`) cannot fault
   through the lazy-commit SIGSEGV handler, so an untouched reserved dst returned EFAULT and the
   record published the PROSPER-OWNED staging pointer — which the engine later
   `FMemory::Free()`d ("FMallocBinned3 Attempt to free an unrecognized block"). The dst write now
   commits lazy 64K pages exactly like the fault handler and retries.

**Result: boot passes LoadPreInitModules, `FConfigCacheIni::InitializeConfigSystem` runs (the
BinaryConfig probe prints), plugin/localization/ICU pak reads all serve, and the engine reads
`DefaultEngine.ini` from the pak.** Note this build's PreInit really does run LoadPreInitModules
BEFORE InitializeConfigSystem (verified in the binary at `eboot+0x7af3` vs `+0x7eda`).

### NEW WALL (deterministic): FMallocBinned3 "Attempt to free an unrecognized block"

Right after the engine reads the `DefaultEngine.ini` pak entry (off=0x2dcc378, 46135 bytes,
served OK — the command buffer even carries the UTF-16 name), the main thread hits
`LowLevelFatalError [Line: 1228] FMallocBinned3 Attempt to free an unrecognized block` inside the
config-load call chain (`eboot+0x2451492 <- +0x245217a <- +0x245288d <- +0x24561a1 <- +0x24b1153
<- +0x243e8ca <- +0x243ff73 <- +0x24679e8`), then UE's own crash handler prints a backtrace and
aborts (`libc.prx+0x48e0`). 4/4 runs deterministic. The `[nullpage] addr=0x8 rip=eboot+0x22ebe7f`
line right before it is BENIGN — that is UE's rbp-chain backtrace walker hitting the null
terminator during crash reporting, not a fault. Suspects: the freed pointer is likely a pak-read
buffer pointer the engine did not allocate (check the remaining record/descriptor contract of the
0xdd/0xde pak callsites for compressed entries — `PakFileCompressionFormats=Oodle`,
`bCompressed=True`: DefaultEngine.ini's entry is compressed, and the post-read decompress path
is where the bogus free happens). The five still-unimplemented libSceAmpr NIDs
(`vWU-odnS+fU sSAUCCU1dv4 tZDDEo2tE5k GnxKOHEawhk H896Pt-yB4I`) fire during mount and may be part
of the compressed-read contract.

### SOLVED (2026-07-08, issue #107): the Ampr map-flavor MIRROR clobbered live MallocBinned heap

Deep diagnosis of the `FMallocBinned3 Attempt to free an unrecognized block` crash. Every ticket
hypothesis (Oodle decompress; compressed-pak buffer ownership; the 5 unimplemented Ampr NIDs) was
tested and **disproven**. The real fault is a guest-side MallocBinned heap overlap; prosper serves
the config bytes correctly.

- **The freed pointer is `0xd` (13), not a real allocation.** Captured at the free site by both the
  in-process int3 logger (`PROSPER_BP=0x22f527a`) and a gdb HW watchpoint: `rdi=0xd`. The message
  "...unrecognized block d" is literally the `%p` of `0xd`. The free instruction is
  `eboot+0x22f5270  mov 0x18(%r14,%r15,1),%rdi ; test ; call 0x231ece0` — a loop freeing
  `element[idx]+0x18` (0x30-byte stride) of an array whose base `r14 = *object = 0x12dadfcf80`
  (stable across runs). The freed object is a **stack-temp** container (`object r15 = 0x7fffec7fee20`,
  a `0x7fff…` stack VA), idx=0.

- **DECOMPRESSION IS NOT NEEDED — the entry is plaintext.** `dd` of `pakchunk0-ps5.pak` at the exact
  read offset `0x2dcc378` shows ASCII INI for all 46135 bytes: begins `[Core.System]\r\n
  CanUseUnversionedPropertySerialization=True…`, ends `…bCompileBlueprintsInDevelopmentMode=False
  \r\n\r\n\0`. So `bCompressed=True` in the ticket referred to the ini's *directive*, not this
  entry; the FPakEntry data is stored uncompressed. No Oodle/Kraken path is required here.

- **THE READ PATH IS CORRECT.** `PROSPER_DUMPAT=0x2007a4002a` (the read dst `a4`) shows the correct
  plaintext landed in the guest buffer, and the config parse *did* consume it (the crash element's
  SavedValue is a valid 100-char UTF-16 config value `(Material="/Engine/BufferVisualization/…`).
  A gated experiment mirroring the bytes into the begin-cursor staging buffer too
  (`PROSPER_APR_FILL_CUR1`) left the crash bit-identical — the guest reads its data from `a4`, not
  the cursor, so the read contract is not implicated.

- **ROOT CAUSE: two live guest containers share memory 0x10 apart (heap overlap).** A gdb HW
  write-watch on the exact freed slot `0x12dadfcf98` traced who wrote `0xd`: a
  `TSparseArray`/`TSet` element-relocation loop at `eboot+0x2467850…0x2467927`
  (`mov (%rbx),%rax` src = `*mapObject`; per-field 0x30-byte MOVE with FString move-semantics on the
  two FStrings at `+0x08`/`+0x18`; heap object `rbx=0x1620e05148`). That loop writes its element at
  **base `0x12dadfcf70`**, and `0xd` is that element's `+0x28` field (a valid TSparseArray
  hash-next / free-list index). But the stack-temp free loop reads the SAME region at **base
  `0x12dadfcf80`** (0x10 higher), so the neighbour's `+0x28` bookkeeping word aliases the freed
  object's `+0x18` `FConfigValue::ProcessedValue.Data` slot — which on real HW is null (empty,
  skipped by the `test rdi,rdi; je`). Two DISTINCT live objects (heap map @0x…cf70 relocation dest,
  stack-temp array @0x…cf80) with overlapping storage ⇒ a MallocBinned pool corruption. The page
  `0x12dadf0000` is freshly lazy-committed (zeroed) right before, so `0xd` is actively written by
  the guest, not stale residue — the guest's own allocator handed out overlapping blocks.

- **The 5 Ampr NIDs are a dead end for this crash.** Named 2 by brute-forcing `nid_hash` over a
  generated libSceAmpr corpus: `tZDDEo2tE5k = sceAmprCommandBufferGetSize`,
  `ULvXMDz56po = sceAmprCommandBufferClearBuffer` (both now registered in `hle_kernel_mem.cpp`).
  `GetSize` fires during APR read teardown; making it return the real cb size instead of 0 left the
  crash bit-identical, confirming the Ampr teardown functions do not gate it. The other three
  (`vWU-odnS+fU sSAUCCU1dv4 GnxKOHEawhk H896Pt-yB4I`) resisted the corpus (non-`CommandBuffer<Verb>`
  shapes) and are harmless no-ops (returning 0) — they are not on the corruption path.

- **THE CLOBBER, PROVEN BY MEMLOG.** `PROSPER_MEMLOG` shows page `0x11e0df0000` appears TWICE:
  `[lazy-commit] mapped page=0x11e0df0000` (line 321 — the guest wrote there, so prosper committed
  it as a normal anon MallocBinned heap page) and later `[memhle] ampr push-map va=0x1720df0000 …
  mirror=0x11e0df0000` (line 453 — strictly LATER). `k_ampr_push_map`'s map flavor maps the buffer
  at `va` AND `MAP_FIXED`s a "mirror" at `va - 0x540000000` (a LOW-confidence rule pinned on The
  Messenger). For this title `va = 0x1720df0000` ⇒ mirror `0x11e0df0000`, which is exactly the live
  heap page — so `MAP_FIXED` replaced the heap page with a fresh page aliased to the Ampr buffer's
  phys `0x10220000`, destroying the pool bookkeeping there. Downstream, MallocBinned carved two
  blocks 0x10 apart (`0x12dadfcf70` heap-map relocation dest vs `0x12dadfcf80` stack-temp array),
  their fields aliased, and the teardown freed the `0xd` bookkeeping word. Same clobber class as
  issue #88 (SetBuffer over live heap), here via the **map-flavor mirror** instead of the
  existing-buffer flavor.

- **FIX (hle_kernel_mem.cpp, `k_ampr_push_map`):** only create the mirror when its target VA is NOT
  already backed guest memory (`mincore(mirror,1,&vec) == 0` means fully mapped == live ⇒ skip). A
  live target means the guest is using that VA as its own heap, not as a second view of the Ampr
  pool, so the mirror is a false positive of the `0x540000000` heuristic and must not overwrite it.
  Verified: the `FMallocBinned3` crash is GONE and DOLL boots through `InitializeConfigSystem` all
  the way into UE's online/PSN subsystem init (SSL, HTTP, NpWebApi, Json, NpManager, NpTrophy2,
  VoiceQoS, SystemService, Share, NetCtl, GameUpdate, NpUniversalDataSystem, NpGameIntent). The
  Messenger is UNAFFECTED: its mirror targets are unmapped at map time (`mincore` fails), so it takes
  the create-mirror path exactly as before — smoke shows 0 mirror-skips, 30k+ render lines, no crash;
  ctest 50/50. CONFIDENCE: HIGH (collision proven by MEMLOG page-double-listing + strict ordering;
  fix is target-conditional so it cannot regress the create-mirror case).

- **NEW WALL: UE online/PSN subsystem init.** After config, the engine constructs its online
  subsystem and hits a long run of unimplemented online/social NIDs
  (`libSceNpManager qQJfO8HAiaY`, `libSceNpTrophy2 sUXGfNMalIo`, `libSceNpWebApi2`, `libSceHttp/Http2`,
  `libSceSsl`, `libSceJson2 *`, `libSceNetCtl obuxdTiwkF8/iQw3iQPhvUQ`, `libSceSystemService`,
  `libSceShare`, `libSceVoiceQoS`, `libSceGameUpdate`, `libSceNpGameIntent`), all returning 0. The
  boot does not crash but does not visibly advance to RHI/AGC within 200s — likely blocked/polling in
  online init (a service call that must report "signed out / offline" rather than 0, or an init that
  the engine waits on). NEXT: trace which online-init call the main thread blocks on (break after the
  last new NID; `PROSPER_BP`/gdb), and give the NpManager/NetCtl/SystemService "get state" queries a
  real offline/no-network status so the engine proceeds to RHI creation → AGC → first UE draw (the
  same `libSceAgc` path The Messenger renders through).

### SOLVED (2026-07-09, issue #115): the "online init hang" was CreateGlobalShaderMap waiting on
### APR async-read completions that prosper never delivered

The issue-#115 hypothesis (main thread polling an online/PSN "get state" call) was **disproven** by
evidence: every unimplemented online NID fires exactly ONCE (dispatch dedupe counts), and a gdb
all-thread dump at the hang showed every thread parked in waits — no poller. The online subsystem
init completes fine with the return-0 stubs. The real chain, recovered by guest stack scan + live
disassembly (all offsets eboot-relative):

1. Main thread parks in `FEventPThread::Wait` (+0x22eaa2b) ← a blocking "wait for archive" reader
   (+0x24ae6e0: `IAsyncReadRequest::WaitCompletion(0)` then `WaitCompletion(INFINITE)` on
   `*(obj+0x28)`) ← **`CreateGlobalShaderMap`** (+0x4ea8c20 — UTF-16 scope strings "Creating
   Global Shader Map...", "GlobalShaderCache", "VerifyShaderSourceFiles"). The file being read:
   `../../../Engine/GlobalShaderCache-SF_PS5.bin`, 0x107e3a bytes, **inside pakchunk0-ps5.pak**.
2. The async read completes on real HW via the **APREventQueue**: the engine creates that equeue,
   registers APR completions on it, and runs an `FAPREventQueueListener` thread in
   `WaitEqueue(eq, evs, 15, ...)` (+0x22740b0). Each event is decoded with `sceKernelGetEventData`:
   **`data = (ring_index << 58) | completion_counter`**; for every newly completed counter the
   listener calls the completion handler (+0x229dcb0), which matches the token against the tracked
   request's `+0x10` slot, else a hash map keyed by the full token. prosper served the READS
   (synchronously) but never posted any EVENT → the listener never woke → WaitCompletion waited
   forever. That is the whole hang.
3. The previously-unknown NIDs, pinned by live arg capture + thunk→GOT→stub mapping:
   - `libSceAmpr::sSAUCCU1dv4 (eq, id=0x7502, 0, 0, 0x43, 0)` — register APR completion events on
     the equeue.
   - `libSceAmpr::H896Pt-yB4I (cbCtx, eq, id, 0x10000000000003e8, 0, 7)` — bind a command-buffer
     ctx to the equeue.
   - `libSceAmpr::vWU-odnS+fU (fileId=8, dst, size=0x107e3a, off=0x7adec90a, off, 4)` — the DIRECT
     async read: exactly the GlobalShaderCache region of pakchunk0-ps5.pak (fileId from APR
     resolve). Implemented as `f_apr_read_direct` in hle_file.cpp (pread + lazy-commit dst write +
     completion token/event).
   - `libkernel::ASoW5WE-UPo (cb, ring_1based, u64* out1, u64* out2)` — the APR **submit**
     (wrapper +0x59b6420 → thunk +0x669afd0; the completion handler resubmits queued CBs with
     `ring = (data>>58)+1`, proving 1-based). Nonzero return = error (checked at +0x22a1d55);
     the completion token is returned through the OUT slots. Implemented as `k_apr_submit`.
4. Contract subtleties (each crash-verified live, both ways):
   - **No phantom sequence numbers.** Pre-registration (mount-era) submits are consumed by
     polling the completion record; delivering their seqs after registration makes the listener
     hand the handler tokens with no tracking entry, and the handler's hash-miss path is NOT
     null-tolerant (`vmovups` from `0x10`, fault at +0x229df3e). Untracked rings reset to 0 at
     first registration so post-registration submissions start at seq 1 (matching the listener's
     zero-initialized per-ring "last processed").
   - **Completions must be delivered ASYNCHRONOUSLY.** Posting the event inside the submit loses
     the race the real DMA never runs (the engine inserts its {token→request} entry right after
     submit returns); prosper defers each post ~2 ms and posts the ring's counter at post time.
   - **Sync-flow submits (`ring_1based == 6`, hardcoded at +0x22a1d89) get NO events** — their
     completions are record-polled; ring-4 (tracked async) events are consumed correctly. The
     real discriminator is likely the H896 cb↔eq binding; ring-id gating reproduces observed
     behavior (CONFIDENCE MED).
   - `sceKernelGetEventData/Id/Filter/Fflags/UserData/Error` implemented (Kyty field-read
     semantics) — the listener consumes events exclusively through GetEventData.

**Result: the GlobalShaderMap loads, PreInit continues through online/PSN init (which was never
the blocker), pak/precacher async reads flow (180+ served in one run), and the boot reaches the
AGC RHI bring-up — AgcCleanupThread/AgcInterruptThread/AgcSubmissionThread + AgcEqueue with
AgcDriverAddEqEvent(id=0x20, id=0x0) + user event 6144 — then keeps loading content.** ctest
50/50, Messenger smoke unaffected.

Follow-on fixes landed in the same change (each its own live-diagnosed wall):

- **POSIX `pthread_cond_timedwait` (libScePosix 27bAgiJmOh0)** implemented for real (shared
  cond/mutex pointer-slot scheme; FreeBSD ETIMEDOUT=60). The unimplemented-0 stub returned
  "signaled" instantly, spinning UE's `IAsyncReadRequest::WaitCompletion(timeout)` loop at 100%
  CPU (caught live: the only busy thread's RA 0x4022ea954 inside prosper_on_unimpl).
- **`sceRtcSetTick` / `sceRtcGetTick`** (tick <-> UTC datetime, shadPS4 semantics). Unimplemented
  SetTick left the out datetime zeroed and UE spammed
  `LowLevelFatalError: Invalid Date values. Y:0, M:0, D:0...` thousands of times during the
  post-shader-map load.
- **Per-container host fd cache for APR reads.** open()+close() per read against the 2 GB pak
  over the WSL 9p mount cost ~1.7 s/read; the whole load phase now runs ~15x faster.
- **Race-free stack-arg capture for `sceAmprAprCommandBufferReadFile`.** The entry shim now
  passes its %rsp as a real 7th argument instead of a plain global — with completions delivered,
  loader/precacher threads submit APR reads CONCURRENTLY and a cross-thread overwrite of the
  global made the handler read the file OFFSET from another thread's frame (wrong-offset reads
  served as "OK"). No TLS (issue #89 constraint respected).

### SOLVED (2026-07-09, issue #161): the MB3 canary corruption was TWO guest VM spaces handed the
### SAME reserved base — non-fixed sceKernelReserveVirtualRange must SEARCH, not echo the hint

With loading fast and parallel, the boot died ~10 s in spamming
`LowLevelFatalError [Line: 186] MallocBinned3 Corruption Canary was 0xN, will be 0x1` until a
SIGSEGV. Both prior hypotheses (guest-%fs TLS caches for fresh loader threads; a
completion-delivery double-free race) were WRONG. The proven chain (each step captured live
under gdb):

1. Line 186 is `FPoolInfoSmall::CheckCanary` inside MB3's GetOrCreatePoolInfo (eboot+0x230d900;
   entries live in on-demand-committed 64K tables of 16384 4-byte records; fresh tables are
   memset to `0x00020003`, canary bits = 3). At the first fatal the failing entry address was
   **misaligned** (`rbx=0x20015f0011`, ≡1 mod 4) while the aligned table at 0x20015f0000 was
   perfectly healthy — so the STORED TABLE POINTER was corrupt, not the table.
2. The pointer array for size-class 0 lives at **0x1000000000** (the very base of guest VA);
   slot 0 read `0x20015f0001` (real table + 1) and slot 1 read `0xffffffffffffffff`.
3. A HW watchpoint on those two qwords caught the writer: **the guest's own MB3 block-of-blocks
   bit tree** (`bts` loop at eboot+0x231c0b0..0x231c0ce) — its storage is ALSO based at
   0x1000000000. Level-1 qword (base+8) filled bit-by-bit as pools 0..63 of class 0 were
   allocated during the load burst; when it hit all-ones the tree propagated to its ROOT qword
   (base+0): `Bits[0] |= 1` — flipping the low byte of the aliased table pointer. Two distinct
   MB3 bookkeeping structures at one VA.
4. Why they aliased: the MB3 ctor (eboot+0x230db50) carves the per-class metadata (ptr array +
   2 bit trees, 3 × 64K per class) from a **64 MiB metadata VM space** that the guest reserves
   with `sceKernelReserveVirtualRange(hint=0x1000000000, len=0x4000000, flags=0, align=0x4000)`
   — AFTER having reserved the 512 GiB MB3 arena with
   `(hint=0x1000000000, len=0x8000000000, flags=0, align=0x200000)`. **flags=0: neither call is
   MAP_FIXED**, so the hint is only a search start (BSD/PS4/PS5 contract; shadPS4
   MemoryManager::SearchFree). prosper treated every hinted reserve as fixed, and the #115
   "re-reserve-of-own-range → OK" workaround blessed call 2 with the SAME base 0x1000000000 →
   the metadata space overlapped the arena (whose granules the class-0 structures also landed
   on), and the two suballocators handed out the same VAs.

**Fix (`hle_kernel_mem.cpp`, `k_reserve_vrange`):** honor SCE_KERNEL_MAP_FIXED (0x10). Fixed →
old behavior (incl. the #115 idempotent re-reserve of an own uncommitted range). Non-fixed with
a hint → search upward from the hint with MAP_FIXED_NOREPLACE probes, skipping past tracked
mappings (one step past the 512 GiB arena); len==0 → EINVAL; search exhaustion → ENOMEM. The
metadata pool now lands at 0x9000000000 and the canary corruption is gone (0 canary lines over
full runs; previously ~31k lines + SIGSEGV at ~10 s). CONFIDENCE: HIGH.

**Post-fix frontier (measured):** the boot now survives the ENTIRE parallel content load with the
AGC RHI threads live (AgcCleanup/AgcInterrupt/AgcSubmission + FAPREventQueueListener +
IoDispatcher/IoService + full TaskGraph pools) and streams pakchunk0 continuously (256 KiB APR
reads; ~845 MB consumed at the 7-minute mark — the pace is bounded by the WSL 9p mount, not by
prosper). At ~9.5 minutes (PROSPER_GFXLOG run) the engine issued its FIRST real AGC driver work:
`libSceAgc::23LRUSvYu1M` / `libSceAgc::BfBDZGbti7A` plus two AGC `ReleaseMem` end-of-pipe packets
(`addr=0x2012..ffe0, data_sel=3`) right as the timeout expired. No crash, no fatal anywhere in
between. NEXT: run past the load phase (longer wall clock, and/or cut pak IO latency — e.g. host
page-cache warm-up or larger read batching) and follow the AGC submission stream into
`execute_and_present` for the first UE4 draw (the same libSceAgc path The Messenger renders
through).

### CORRECTED (2026-07-09, issue #180 session): the "IO-bound load" reading above is WRONG —
### the boot stalls DETERMINISTICALLY after exactly 90 APR reads, on every IO speed

Two changes of fact, both measured:

1. **IO fast path.** Copying the dump to WSL-native ext4 (`cp -r /mnt/c/.../PPSA17942-app0
   /root/PPSA17942-app0`, ~20 GB) and booting `/root/PPSA17942-app0` cuts time-to-first-AGC-driver
   -call from ~9.5 min (9p) to **~80 s cold / ~13 s warm page cache**. The load was never
   streaming-bound on guest demand — the 845 MB rchar figure was dominated by eboot/module/footer
   reads, and total pak consumption before the wall is only ~45 MB (90 APR reads).
2. **The wall is a correctness gap, not latency.** With identical binaries, both the ext4 boot and
   a 9p control boot stop issuing APR reads after EXACTLY the same 90th read (`pakchunk0-ps5.pak
   off=0x32687c8e size=0x40000` — the first async-archive precache chunk after
   `GlobalShaderCache-SF_PS5.bin`), a few seconds after the `23LRUSvYu1M`/`BfBDZGbti7A` +
   AgcEqueue setup. The earlier 9.5-minute runs were sitting at this SAME stall — there was never
   additional progress to be had by waiting.

**The stall, fully dissected (all live-verified under gdb on the stalled process):**

- MAIN THREAD is parked at `eboot+0x24ae718` — inside the #115-documented blocking archive reader
  (`IAsyncReadRequest::WaitCompletion(INFINITE)` on the FEvent at `*(obj+0x28)`) under
  `CreateGlobalShaderMap` (`eboot+0x4ea8ea0` on the same stack). Its FEvent (state u32 at
  obj+0x14, mutex slot +0x20, cond slot +0x28) is never triggered. Manually setting state=2 +
  `pthread_cond_broadcast` un-wedges the whole machine (the engine then re-reads the chunk and
  streams on through vWU direct reads) — proving the ONLY missing piece is the completion signal.
- One worker spins in `k_cond_timedwait` (~16k futex/s, RA `eboot+0x22ea954` = FEventPS5 timed
  wait) — a second waiter on the same stuck pipeline. IoService/IoDispatcher threads wait in
  `sceKernelWaitEqueue` on their user-event queues (`AddUserEvent id=-1`); force-posting those
  events (gdb) wakes them to EMPTY queues — the request is not queued anywhere, it is tracked and
  waiting for its APR completion event.
- **The guest's APR completion machinery** (handler `eboot+0x229dcb0`, listener loop
  `eboot+0x22740b0`, both disassembled live):
  - The listener object is an **eboot GLOBAL (`eboot+0x95aebd8`)**, NOT the sSAUCCU1dv4 a4 heap
    object. Per-ring in-flight tracking slot at `[ctx+0xa8+ring*0x28]` (a request object whose
    `+0x10` holds the EXPECTED completion token, `+0x00` the cb), queued-batch array pointer at
    `+0xb8` (0x20-byte entries, `+0x18` submitted flag; completion resubmits the next entry via
    the ASoW wrapper `eboot+0x59b6420` with `ring_1b = ring+1`), per-ring last-processed counter
    at `+0xc8+ring*0x28`.
  - **Completion tokens are GUEST-CHOSEN**: the `H896Pt-yB4I` binding's a3 tag IS the token
    ((ring<<58)|n, live captures n = 0x3e8, 0x3e9, ... — the engine's own counter base 1000).
    Read live at the stall: the ring-4 slot expects EXACTLY `0x10000000000003e8` — the first
    H896 tag — while prosper posted an invented `(4<<58)|1`. The engine has therefore considered
    async batch #1 in flight FOREVER, and the whole async IO pipeline (including the archive
    precache the main thread waits on) is jammed behind it.
  - An event whose data does not match: slot-compare fails -> hash walk; with a NON-empty
    tracking hash the walk falls off the -1 chain sentinel and dereferences `0 + 0x10` — the
    `eboot+0x229df3e` fault (re-verified 2/2 this session by posting events for unbound ring-6
    submissions; `eboot+0x229dd21` is the same fault when the ring slot is null —
    PROSPER_NULL_PAGE shims that one).
  - `ASoW5WE-UPo`'s two out-pointers ALIAS the request's completion record: a2/a3 = req+0x28
    (status) / req+0x30 (bytes) — the same fields `sceAmprAprCommandBufferReadFile` completes
    with {0, size}. Writing a nonzero "token" there marks the read FAILED (the
    `eboot+0x22738a5` check) — with tag-echo enabled this turned the shader-map load into
    "GEngineLoop.PreInit Failed!" until the write was skipped for bound cbs.
- **Experiments gated in the code** (all default-OFF; `hle_kernel_mem.cpp` / `hle_kernel_time.cpp`):
  - `PROSPER_APR_TAG_ECHO=1` — bound cbs echo the H896 tag as their token, post the tag verbatim
    on the binding's own equeue (2 ms deferred), leave the record untouched, and run a "slot-echo"
    scan that re-reads registered listener ctxs' tracking slots and posts exactly the expected
    tokens. Current result: the tag event IS consumed (the jam breaks) but the listener's range
    walk over seqs 1..1000 (its last-processed starts at 0; the tag base is 1000) crosses a
    non-empty tracking hash and faults at `eboot+0x229df3e`. THE remaining unknown: how real HW
    seeds the listener's per-ring last-processed to the tag base — the answer is in the listener
    loop body (`eboot+0x2274130..`), un-disassembled past `+0x2274146`.
  - `PROSPER_APR_EVENT_ARG8=1` — mark requests whose ReadFile arg8 points just above the request
    frame as eventful. Disproven as a discriminator (sync flavors pass live arg8 pointers too;
    crashes the listener); kept only for bisection.
- The next session should: (1) disassemble the listener loop body to learn the last-processed
  seeding (or write it to tag_base-1 from the HLE before the first tag post — the listener object
  address is discoverable by scanning eboot .data for the pointer-to-slot whose `[slot+0x10]`
  equals the known H896 tag); (2) then enable tag-echo semantics by default; (3) the ring-6
  async-archive read (#90) completes through the same machinery once the ring-4 channel drains —
  re-verify, then follow the boot toward VideoOut/RHI init and the first SubmitDcb.

Boot recipe (fast): `cp -r` the dump to `/root/PPSA17942-app0` once, then
`PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 [PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1
PROSPER_EVLOG=1 PROSPER_AMPRLOG=1] ./build-linux/boot_trace /root/PPSA17942-app0` — reaches the
stall (the current frontier) in ~15-80 s.

### SOLVED (2026-07-09, issue #208): the APR completion contract is fully recovered — the GUEST
### seeds the listener's range walk; prosper now echoes exactly the guest-chosen tags

The #180 open question ("how does real HW seed the listener's per-ring last-processed to the tag
base?") is answered by static disassembly of the guest (all offsets eboot-relative):

- **The listener-ctx CONSTRUCTOR (+0x22a0670, once-guarded at 0x95aed70, ctx = the eboot global
  0x95aebd8) seeds everything itself**: it creates the APREventQueue ([ctx+0x18]), registers ids
  `0x74fe + ring` for rings 0..5 on it (the six sSAUCCU1dv4 calls), and initializes each ring's
  pair: token counter `[ctx+0xc0+ring*0x28] = 0x3e8` (1000) and listener last-processed
  `[ctx+0xc8+ring*0x28] = 0x3e7` (999). The walk over the first tag event (cnt=1000) covers
  exactly seq 1000 — there was never any library-side seeding to model.
- **The batch submit (+0x22a02b0)** draws `token = (ring<<58) | [ctx+0xc0+ring*0x28]++`, passes it
  to H896Pt-yB4I as the binding tag, stores it at `[slot+0x10]`, and inserts a
  `{token -> completion callback}` entry into the hash at ctx+0x58 (128-byte entries, chain
  sentinel -1). The handler (+0x229dcb0) completes the slot match AND invokes the hash callback
  (that callback is what fires the FEvent the CreateGlobalShaderMap precache blocks on).
- **The listener (+0x22740b0) stores `last := cnt` UNCONDITIONALLY after every event**
  (+0x2274143), even when `cnt < last+1` (no walk). This is why the #180 tag-echo experiment still
  faulted: prosper's own invented-counter events (registration catch-up replays, vWU direct-read
  wakeups) carried cnt << 1000, walked nothing, but REGRESSED the guest's 999 seed — the next real
  tag event then walked the gap seqs, and any walked seq with no slot match and no hash entry
  takes the null-entry path (a 64-byte ymm swap against address 0x10, the +0x229df3e fault —
  fatal on real HW too, so the guest guarantees dense counters from exactly 1000).
- **Fix (now the default; the PROSPER_APR_TAG_ECHO / slot-echo experiments are retired):**
  bound (H896) submits echo NOTHING into the out slots (they alias the completion record) and
  post the binding tag verbatim, deferred ~2 ms, coalesced per ring to the highest counter
  (kqueue "completed up to" semantics). Unbound submits keep returning prosper counter tokens
  through the out slots and post NO event; vWU direct reads post NO event (live-verified: the
  gdb-unwedged engine streamed the whole remaining load through vWU reads with no events).

**Also fixed in the same session — a DOLL boot regression that masked all of the above** (came in
with the master merge, first bad commit fe8e8d7 / issue #183, found by git bisect): the guest
wedged single-threaded ~10 s into boot, self-deadlocked in k_mutex_lock (mutex `__owner` == the
caller). UE4's PS5 lock wrapper (+0x24ca4b6, same shape inside the APR handler at +0x229dcf5)
builds its own recursion on the FreeBSD self-lock contract:
`err = mutex_lock(obj); if (err) /* EDEADLK: already mine */ skip-acquire; depth++;` — and DOLL
creates those mutexes with `pthread_mutexattr_settype(type=4)` (ADAPTIVE_NP, live-captured via
PROSPER_MUTEXLOG). FreeBSD libthr's `mutex_self_lock` returns EDEADLK for ERRORCHECK **and**
ADAPTIVE_NP (adaptive = errorcheck + a spin heuristic); only NORMAL hard-deadlocks. #183 (after
Kyty) mapped 4 -> host NORMAL, which self-deadlocks on glibc. Fixes in hle_kernel.cpp: settype
type 4 -> host ERRORCHECK; a fresh mutexattr defaults to ERRORCHECK (FreeBSD attr default — the
#183 change only covered the no-attr init path); the static ADAPTIVE sentinel (1) also maps to
ERRORCHECK. Kyty is weighted DOWN here per policy: no title it runs exercises adaptive self-lock.

**Measured result (ext4 fast path, 240-480 s runs):** the 90-read wall is GONE — 978 APR reads
served, the guest's own tag counter advanced past 0x458 (112+ batches consumed through the
listener), GlobalShaderMap loads, PreInit completes, the engine initializes VideoOut + the AGC
RHI (VideoOutQueue equeue + AddFlipEvent), loads every plugin assetregistry.bin, submits its
first real DCBs (`SubmitDcb #1: 85 dwords/13 packets`, `#2: 1436 dwords/232 packets` — EventWrite/
WriteData/AcquireMem/ReleaseMem streams, 0 draws yet) and performs its first flip
(`GpuFlip handle=0x1001 bufidx=0 mode=0x1 fliparg=0x1`). ctest 60/60; Messenger smoke renders
3860 frames in 300 s.

**Second wall in the same session — the IoStore append-loop spin — also SOLVED:** with
completions flowing, the boot next parked its IoStore thread in a busy poll (the only running
thread, inside k_ampr_getsize, guest RA +0x227e2eb). The guest's batch-append loop (+0x227e2c0)
polls `GetSize(cb) - <used>(cb) > 0xff` (wrappers 0x59b5dd0/0x59b5e00) before appending the next
command packet; GetSize (tZDDEo2tE5k) is the cb's fixed byte CAPACITY (carried by the batch cb's
init in a5 = 0x720 — a different arg than the APR read-request flavor's a1), and the old global
"last cb size" answer starved it to 0. k_ampr_init now records {cb -> capacity} for both init
flavors and k_ampr_getsize answers per-cb; the spin cleared (the thread parks in eq_wait like the
rest of the pool).

**NEXT (the new frontier, diagnosed to the thread level):** after the first flip and the plugin
assetregistry.bin loads, no further draws/flips arrive (480-600 s runs). Live state: the RHI-side
thread cycles a WaitEqueue(VideoOutQueue) loop (ident=1, filter=-13, one event delivered per
cycle — vblank pump, normal); every IoDispatcher/IoService/TaskGraph worker is parked in
k_eq_wait/k_cond_wait; and the MAIN thread runs HOT (~70% cpu) in a lock-poll-unlock loop over
shared state (sampled: scePthreadGetthreadid + glibc mutex futexes at high rate, FName-machinery
frames on the stack, alternating lock words 0x10a0e4f120 / 0x200ff00628) — the classic UE4
flush-async-loading shape: the game thread polls package-loading state that the idle IO threads
never advance. The question for the next session is which completion/user-event the async-loading
pipeline is missing (the IoDispatcher threads wait on their OWN user-event equeues, AddUserEvent
id=-1 — force-posting those was already shown in #180 to wake them to empty queues). Also worth
wiring: new unimplemented NIDs on this path — libSceAgc dbOlWdppb4o, qj7QZpgr9Uw, bbFueFP+J4k,
xSAR0LTcRKM, w6Dj1VJt5qY; libSceVideoOut MTxxrOCeSig, U2JJtSqNKZI; plus the
sceVideoOutGetOutputStatus struct fill (issue #82) which this boot now hits with a warning.

### The remaining 3 unnamed Ampr NIDs (issue #107, not on the crash path)

`vWU-odnS+fU`, `sSAUCCU1dv4`, `GnxKOHEawhk`, `H896Pt-yB4I` still resist the generated
`sceAmprCommandBuffer<Verb><Noun>` corpus (the same brute-forcer that named the other 8 Ampr NIDs).
They fire during mount, return 0 via the generic unimplemented stub, and are proven irrelevant to the
config crash (fixed without them). Likely non-`CommandBuffer`-shaped names (Amm/Measure/util helpers);
leave as no-ops until a future frontier needs them.

### (superseded) original issue-#107 diagnosis notes

### OLD ANALYSIS (superseded): post-mount engine/RHI init — a null virtual call (racy, two faces)

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

### SOLVED (2026-07-09, issue #213/#222): four post-load walls cleared — DOLL runs deep into engine init with the AGC draw-path wired; the remaining wall is characterized

Building on the stable frame loop (#213), four distinct walls between "empty present loop" and
"scene draws" were RE'd and fixed (all offsets eboot-relative; captures via gdb on the ext4 fast
path). The game now runs 200+ flips / 470+ compute dispatches, deep into post-load engine init
(trophy, error-dialog, telemetry subsystems), with 0 faults and 0 OOM.

1. **The #222 EUD-ring fault (eboot+0x59949e4) was the FIRST scene-draw prep crash, not a race.**
   `+kSrjIVxKFE` (the AGC register-context ctor) is called repeatedly on LIVE contexts (a per-
   frame/post-bind reset; DOLL even builds contexts on the stack), not once at device init. The
   guest's SetSource (eboot+0x5994620) EARLY-OUTS when [sub+0x00] still equals the shader being
   bound; a full bind derives +0x34/+0x38/+0x3c/+0x40/+0x44 from the SAME user_data descriptor,
   so +0x44!=0 always implies direct-table[10]!=0xffff. Our Stage-1 ctor overwrote ONLY [sub+0x08]
   with the empty all-0xffff descriptor; after a reset of a bound context, re-binding the SAME
   shader early-outed, leaving +0x44 stale-nonzero against the sentinel table. The EUD writer
   (eboot+0x5994940, no 0xffff guard in DOLL's build) then computed slot 0xffff -> spill base 0 +
   0xffff*4 - 0x80 = 0x3ff7c (the exact captured fault addr). Fix: model the ctor as the guest's
   own sub-reset (eboot+0x59945e0) — zero [sub+0x00..0x47] + the +0x60/+0x68 cached bank & flags,
   preserve the +0x48/+0x50 allocator pointers & +0x58 mode byte, install the empty descriptor.

2. **The per-draw fence cluster was never wired.** The guest fence builder (eboot+0x59a1780)
   allocates a label inside a Dcb NOP data packet, pre-builds WriteData/ReleaseMem/WaitRegMem with
   NULL placeholder addresses, then patches the label address through three imports that were
   observe-only glog no-ops: `fPSCdQxgpSw`=WriteData-patch-addr, `3KDcnM3lrcU`=WaitRegMem-patch-
   addr, `0fWWK5uG9rQ`=ReleaseMem-patch-addr. With them stubbed, every per-draw GPU fence targeted
   address 0. Implemented all three (each validates the packet header sub-op first; 0 refusals
   across a full run). Also implemented `k3GhuSNmBLU`=sceAgcDcbDispatchDirect(dcb,x,y,z,modifier)
   as a new R_DISPATCH_DIRECT packet (guest wrapper eboot+0x220ede0; Kyty Gen4 DispatchDirect ABI)
   — DOLL's compute prologue issues ~470 of these.

3. **The 34.6 GB OOM / RenderThread-timeout was a trophy success-with-garbage-out.**
   `sceNpTrophy2GetGameInfo` (NID 4IzqhhUQ3nk, nid_hash brute force) + fill sibling y3zHpdZO6ME:
   the guest (eboot+0xdbcb43) reads a u32 count from the out-struct and grows two TArrays from it;
   the unimplemented stub's success-without-write fed heap garbage as the count -> a 34,644,492,288-
   byte array grow ("Ran out of memory allocating 34644492288 bytes"), or (garbage allocatable-huge)
   a multi-minute zero-fill that starved the RenderThread until UE's 120 s watchdog fired. Fix:
   return failure (0x80551500) so the caller takes its clean trophy-unavailable path (eboot+0xdbd239).

4. **sceVideoOutGetOutputStatus (#82) — the ONE consumer is now RE'd.** DOLL's swapchain init
   (eboot+0x2226fea) reads a u32 at status+0x04 and switches to an HDR pixel format when it == 2.
   Success-without-write fed it stack garbage. Fix: write a defined {u32 state=0, u32 mode=0}
   prefix (SDR, 8 bytes — within the caller's 0x50-byte reservation).

**Remaining wall (characterized, next session):** the main GameThread is parked in a ~1 ms
timed-poll loop — guest fn eboot+0x2cc2340 does a vtable poll (`call *0x20(%rax)` at +0x2cc236a)
then a `vucomisd` elapsed-time/timeout compare (eboot+0x2324190), backed by a timed cond-wait
(eboot+0x22ea94f -> the guest's ~1 ms sleep helper). It is polling an async completion that a
worker never delivers — the same "wait-for-completion-with-timeout" scene-activation shape, now
reached far deeper (past PreInit, RHI, plugin load, trophy/telemetry init). Next: resolve the
vtable object at eboot+0x2cc236a's `*0x20(%rax)` (break there, walk the interface) and what the
+0x2324190 double compare gates — likely a streaming-level / world-activation handshake, still on
the FlushAsyncLoading -> TickAsyncLoading axis. Still 0 graphics draws (DrawIndex/DrawIndexAuto);
the compute-only dispatch stream is UE4's persistent per-frame GPU setup, not scene geometry.

### IDENTIFIED (2026-07-09, issue #232): the poll wall is UE4's FRenderCommandFence — the GameThread's GPU-hang watchdog; the RenderThread stalls in the libSceAgc submit ring-drain (intrinsic, intermittent)

The #213/#231 "GameThread ~1 ms timed-poll" wall (eboot+0x2cc2340) is now positively identified by
its own log strings (read out of eboot rodata; VA→file map is `off = 0x66e04d0 + (va − 0x669c000)`
for the r-- segment, anchored on the `"Unknown"` FName at 0x82639c2):

- 0x407e3b518 (UTF-16): **"GameThread timed out waiting for RenderThread after %.02f seconds"**
- 0x407fe886a (UTF-16): **"GPU has hung or crashed!"**
- 0x407e3b474 (UTF-16): **"Rendering thread exception:\r\n%s"**, and the CVar name **"r.GTSyncType"**.

So eboot+0x2cc2340 is UE4's **`FRenderCommandFence::Wait` / GameThread↔RenderThread frame sync with
the GPU-hang watchdog** (`GGameThreadWaitForRenderThreadTimeout`, 120 s — the loop's `vucomisd`
compares elapsed vs a start+120.0 s deadline: start≈4.25 s, deadline≈124.25 s live). The poll
`call *0x20(%rax)` is `FEvent::Wait(1 ms)` on the fence's completion event:
- outer future obj vtable = eboot+0x8e38ee0; inner FEvent obj vtable = eboot+0x8e37ef8.
- FEvent::Wait = eboot+0x22ea830 (reads triggered-state at `event+0x14`: 0=untriggered / 1=auto /
  2=manual; waits on the guest pthread cond at `event+0x28`, mutex at `event+0x20`, via
  `scePthreadCondTimedwait` → our `k_cond_timedwait`).
- FEvent::Trigger = eboot+0x22f4cf0 (locks `event+0x20`, sets `event+0x14`=1/2, `cond_signal`
  (0x6699490) / `cond_broadcast` (0x6699480) on `event+0x28` → our `k_cond_signal`/`broadcast`).
The FEvent primitive itself is faithfully wired (ensure_cond/ensure_mutex CAS the guest slot to one
host object; no signaler/waiter desync). **The completion genuinely never fires within 120 s → the
guest then aborts through the warn path (eboot+0x2cc23dd) and SIGSEGVs in the libc.prx log formatter
(rip=libc.prx+0x48e0) — a *secondary* crash in the hang logger, not the root cause.**

**Root cause is the RenderThread/RHIThread, not the GameThread.** At the freeze (flips stop at
~180–210; all-thread dump): GameThread(T1) parks in `k_cond_timedwait` on the fence FEvent cond;
64 workers idle in `k_cond_wait`; the ONLY live event traffic is the VideoOut vblank pump
(`WaitEqueue ident=1 filter=-13` ra=eboot+0x20001ac14) — **zero GPU-EOP kevents, zero
TriggerUserEvent**. The RHIThread (T3) is deep in the **libSceAgc submit path**
(eboot+0x59a49f2 → 0x59a6700 → 0x599f210 ring/pipe-progress lookup; helper eboot+0x58e0640 builds a
kevent and calls a 0x66xxxxx sync thunk) — it is spin-sleeping in the submit **ring-drain / GPU-
progress poll** that never advances, so the RenderThread waits on the RHIThread and the GameThread
times out. i.e. the GPU submit ring fills and our executor never signals the submit-completion /
ring read-pointer advance the guest's `sceAgcDcbSubmit` waits on.

**It is intermittent** — the same build sails to 22 000+ flips on some runs, freezes ~188 on others
— which fingerprints a **submit-completion / EOP-event delivery race** in the AGC executor, the same
family as #208/#210/#231 (undelivered completion), now on the *steady-state per-frame* submit path.

**The startup-movie hypothesis is DISPROVEN.** The freeze correlates in time with the opening
cutscene (`doll/content/movies/cutscene/sms_opening_en.usm` opens ≈flip 186; only 3 media NIDs ever
fire — `sceVideodec2QueryComputeMemoryInfo`=RnDibcGCPKw, `sceAjmInitialize`=dl+4eHSzUu4,
`sceAjmModuleRegister`=Q3dyFuwGn64, all nid_hash-recovered), and 2 threads actively pace CRI-Atom
audio in `audio2_ctx_push`. But **making every `.usm` fail its stat/access existence check (movie
player sees them absent, all 7 incl. the opening cutscene refused) did NOT move the wall** — DOLL
still froze at the identical FRenderCommandFence timeout + libc-logger SIGSEGV. The movie is a
coincidental co-timing, not the cause.

**Next session (the real fix):** instrument the libSceAgc submit ring model in the executor
(prosper/src/gpu) — confirm each `sceAgcDcbSubmit` marks its ring range complete and advances the
read-pointer / posts the submit-EOP the guest's ring-space poll (eboot+0x59a6700) reads, with no
race under back-to-back compute-only submits. Reproduce with `tools/dbg/gwalk.sh` (freeze-detect +
per-thread guest-RA walk) and compare the AGC submit/EOP path to the Messenger's (which drains
cleanly). Fixing the ring-drain/EOP delivery should let the RenderThread complete the frame, fire
the fence, and advance the GameThread into world activation + the first scene draws. Diagnostic
gdb tooling added this session: `tools/dbg/poll_probe.{sh,gdb}`, `poll_deep.{sh,gdb}`,
`stall_bt2.sh`, `gwalk.sh`.

### SOLVED (2026-07-09, issue #232): the FRenderCommandFence wall was an UNFOLDED submit variant (w1KFAHVqpaU) — DOLL advances 178 → 284 flips past it

Post-#236 (EOP coalesce=false), DOLL still parked: every thread idle except **RenderThread 1**,
which spin-yielded in a timeout poll at **eboot+0x221d6c2** (`mov 0x30(%rbx),%rax ; cmpq $0,(%rax)`
reached from eboot+0x3bf4862) waiting for a GPU **fence label to become nonzero**. Live probe
(`tools/dbg/slot232.py`): the polled label = `*[obj+0x30]` = **0x1180f06760**, value stuck at 0.
That label is the destination of a `ReleaseMem action=0x14 addr=0x1180f06760 data=0x1` the guest
BUILT — but `EOP write [0x1180f0*]` count was **0**: the Dcb carrying those fences was never folded.

**Root cause: DOLL's UE4 RHI submits through TWO driver entry points, and we only folded one.** The
guest `SubmitCommandBuffers` impl (eboot+0x220a9a0) loops the buffer array, submitting buffers
[0..n−2] via `sceAgcDriverSubmitDcb` (UglJIZjGssM — folded) and the **final** buffer (eboot+0x220aace)
via **w1KFAHVqpaU**, which was unimplemented (returned 0). So every batch's final buffer — carrying
the RenderThread's completion fences at 0x1180f0xxxx — was dropped on the floor; the label never went
nonzero, the RenderThread never completed the frame, and the GameThread's 120 s watchdog fired.

**Fix (committed, `hle_agc.cpp` `agc_driver_submit_dcb_variant`):** fold w1KFAHVqpaU's Dcb through the
CommandProcessor and fire EOP, same as the primary path. ABI RE'd from the compiler adapter thunk
(eboot+0x58df3f0) that marshals the internal call into the Sony import — the raw dword-stream address
is stack **arg8**, forwarded by the guest-%fs swap stub to **fp[3]** (the slot ReleaseMem/WaitRegMem
already read arg8 from). The dword count (arg9) is past the 2-arg forward window, so the fold relies on
`decode_pm4`'s type-3 self-termination (observed bounded: 377–10386 packets, well under the
512 Ki-dword runaway cap). The PM4 header is validated before folding (fallback: scan a0..a5; refuse +
log otherwise) so a wrong pointer can never fold garbage.

**Verified:** w1KFAHVqpaU now folds every frame; **2100 EOP writes land in the 0x1180f0 fence region**
(was 0); the frame loop advances **178 → 284 flips** past the wall. ctest 63/63; Messenger unregressed
(153 rendered frames to #980, 0 faults — it never calls this path). CONFIDENCE: MED (buffer-slot ABI +
missing-fence symptom pinned; unknown-length fold empirically bounded, not ABI-confirmed).

**NEW WALL (next session):** past the fence, DOLL hits a deeper **worker-thread fault at
eboot+0x2316c91** — `mov (%rbx),%rcx` on a corrupt free-list node (`rbx≈0x20015f00`, a bad "next"
pointer popped by `mov (%r8),%rbx` at +0x2316c85; this is a memory-pool/free-list alloc). Still **0
scene DrawIndex/DrawIndexAuto** — the ~1500-packet-per-frame stream folded via w1KFAHVqpaU is still
UE4's compute/setup work (dispatches climb to 3500+), not scene geometry. Open questions: (a) is the
0x2316c91 free-list node corrupted by the unknown-length fold over-walking (lower the cap / widen the
swap stub to forward arg9 = the true count and re-check), or is it an independent deeper init-race
(the doc's "worker-thread face")? (b) what populates that pool, and does world/level activation gate
on it? Probes added: `tools/dbg/slot232.py`, `w1k232.py`, `gw232.py`, `waketrace232.py`,
`unimpl232.py`, `gotwho.cpp`, `nid2got.cpp`, `run232{b..h}.sh`. Thread-name adoption
(`pthread_setname_np` in `k_pthread_create`) now labels all guest threads in gdb/procfs — the probe
that made the RenderThread identifiable.

### SOLVED x3 (2026-07-09 late, issue #232 session 3): boot-killer crash zoo cleared (1/6 -> 5/6 stable boots); the GameThread wall is FlushAsyncLoading; the APR batch pipeline now drains clean

Three verified fixes this session (all offsets eboot-relative unless noted):

1. **The #241 free-list fault (0x2316a**, node 0x20015f00) was the w1KFAHVqpaU fold over-walking.**
   The Dcb is carved from a live ring whose stale tail is still valid type-3 PM4 — an unknown-length
   fold re-executes stale WriteData/ReleaseMem fence writes into heap blocks the guest has freed and
   reused (8-byte GPU-address-shaped scribbles -> the deterministic 0x20015f00 free-list node; the
   fault site 0x2316a20/0x2316b00 is Sony libc's per-thread TLS small-block cache, keyed off the
   pthread key at 0x95fa764). The TRUE dword count IS passed by the guest: the callsite
   (0x220aabf) loads {addr @+0x00, dw_count32 @+0x10} from the buffer array and the adapter thunk
   (0x58df3f0) forwards it as the import's stack arg9. arg9 is beyond the swap stub's 2-arg re-push
   window but the ORIGINAL guest frame is intact above the stub frame: handler fp[6/7/8] = orig
   arg7/8/9 (validated fp[7]==fp[3] and fp[5] in guest code before trusting). Fold now uses the
   exact count (`walk==arg9`, 0 fallbacks, 28039/9352/6325-dword folds observed).

2. **The libc.prx+0x48e0 "hang-logger SIGSEGV" + "stack smashing detected" abort was TWO stacked
   artifacts of one guest fatal.** The real event: UE4 LowLevelFatalError **"MallocBinned3
   Corruption Canary was 0x3, should be 0x1"** (found formatted in the core's GErrorHist) raised
   from the RHI's retired-buffer free loop (0x220bd50, GMalloc->Free over a buffer array) ->
   Sony libc abort stub libc.prx+0x48d0: `mov $0xa002000b,%fs:0x28 ; int $0x45`. Root cause:
   prosper fired the GPU EOP kevent INSIDE the submit call (synchronous fold), violating the real
   invariant that the EOP interrupt arrives only AFTER submit returns (doorbell at end of submit)
   — the guest's AgcInterruptThread->AgcCleanupThread chain raced the AgcSubmissionThread's own
   post-submit bookkeeping and double-handled the retired-allocation list. Fix: a single ordered
   FIFO worker delivers EOP kevents ~1 ms after the submit returns (label/fence WRITES stay
   synchronous; #236 distinct+ordered semantics kept; PROSPER_EOP_SYNC=1 restores old behavior).
   The "stack smashing" was OUR fault_handler's stack-protector: canary read from %fs:0x28 (the
   very slot the guest abort stub just wrote) on the guest %fs, checked after the handler switched
   to the host %fs -> spurious SIGABRT. fault_handler is now no_stack_protector.
   Messenger verified unregressed with deferred EOP (1549 presented frames, 0 faults).

3. **APR ptr-tag (id=0) completions now deliver in submission order.** The tag is the guest BATCH
   pointer; the consumer retires its in-flight list FROM THE HEAD up to the tagged batch
   (retire fn 0x227e8e0, in-flight count [disp+0x30]: incl on submit 0x227e565, decl per retired
   node 0x227e92b; the tail-flush gate `[disp+0x30] <= 1` at 0x227e7d3 is an UNSIGNED compare).
   The old per-post detached 2 ms threads made cross-post order a scheduler race. With the FIFO
   worker the previously-never-submitted final partial batch now binds+submits+completes (live:
   the deterministic tail read off=0x328600000 of pakchunk0-ps5.ucas gets its H896+ASoW+event,
   and at the stall ALL THREE rotating batches (0x1200df4980/49f0/4a60, a +0x00-linked ring) are
   free (+0x68=1) and empty (+0x8=0) — the APR channel fully drained, verified by live scan
   `tools/dbg/scan_disp.py`).

**The #232 wall itself (still standing, now precisely characterized):** the GameThread is a
busy FlushAsyncLoading spin — `ProcessAsyncLoading` (0x25b2980, identified by its own
"ProcessAsyncLoading" rodata xref at 0x25b2a89; ProcessLoadedPackages at 0x25a1877/0x25b32d4) in
a tight IsInGameThread/GetCurrentThreadId loop, ~16k gettid syscalls/s, no sleeping, no IO.
IoDispatcher/IoService/all workers idle in k_eq_wait/k_cond_wait. File activity ends after the
localization/assetregistry scans + the .ucas streaming burst (~2500 reads); the LAST reads are
deterministic across runs (pakchunk0-ps5.ucas off 0x328600000 / 0x353640000 / 0x353680000).
The async package(s) being flushed wait on a NON-IO dependency that never fires. Next: identify
what consumes the completed .ucas chunk data (decompression job? FEvent chain? the request
records at the retry list disp+0x198?) and which signal the FlushAsyncLoading packages actually
poll. SlateLoadingThr + CriManaDecodeTh exist (loading-screen + movie machinery live); DOLL is
UE4's LEGACY EDL loader (FAsyncLoadingThread strings), not zenloader.

### REFRAMED (2026-07-09 session 4, issue #232): the GameThread is NOT stuck in FlushAsyncLoading — it TICKS; the wall is genuine scene non-activation (0 DrawIndex, 1.66M DispatchDirect)

Live-attach sampling at the steady-state stall (probes `boot232.sh`, `probe232w/x/y/z.py`,
`cpu232.sh`; target left running via `setsid`, PID in /root/pid232.txt) **overturns the session-3
"busy FlushAsyncLoading spin" characterization**:

- **The GameThread reaches Tick and runs the engine frame loop indefinitely.** 33/40 RIP samples of
  the guest main thread land in a **game-side frame-throttle**: a `do { pump(); } while (clock() <
  start+target)` loop at **eboot+0x5044c01** (containing fn starts eboot+0x5044740, a per-frame state
  tick taking a `this` object + calling vtable `*0x288`), whose pump path bottoms out in a
  `scePthreadCondTimedwait` return at **eboot+0x248e556** (an `FEvent::Wait(1ms)`-style sleep). It is
  NOT the 0x25b2980 ProcessAsyncLoading busy-spin — that function's entry never fires under a
  163 s breakpoint, because the loader long since drained. The frame loop advances **58k+ GpuFlips**,
  stable and unbounded.
- **Async loading has fully DRAINED, not deadlocked.** The `FAsyncLoadingThread` (ALT, name
  "FAsyncLoadingTh") is **parked on its zenaphore** (`k_cond_wait` via eboot+0x22ea954 ->
  0x2316b00). Its loader object (recovered from the parked thread's regs/stack, plausibility-gated
  on the counter layout): **QueuedPackagesCounter[+0x1b0]=0, ExistingAsyncPackagesCounter[+0x1b4]=0,
  QueuedPackages.Num[+0x58]=0, pendingIoBytes[+0x5d8]=0, ExternalReadQueue[+0x138]->next=0**. Nothing
  is queued or in flight. (The +0xd8 AsyncPackageLookup histogram is noise — the slot layout guess is
  wrong for this build; the counters are the reliable signal and they are all zero.) IoDispatcher /
  IoService / 64 PoolThreads all idle in k_eq_wait/k_cond_wait.
- **0 draws is GENUINE, not a decoder miss.** Packet-kind histogram over a full boot log:
  **1,662,586 DispatchDirect and ZERO DrawIndex / DrawIndexAuto** (plus SetRegsIndirect 23.6M,
  SetShRegDirect 11.6M, EventWrite/AcquireMem/ReleaseMem/WaitRegMem/WriteData, 50k Flip). The game
  submits only per-frame **compute**, never geometry. The recompiler/PM4 decoder is not dropping
  draws — the guest never emits a draw packet.
- **Movie / cutscene is NOT the gate (clean A/B).** New knob `PROSPER_DENY_SUBSTR=.usm` (hle_file
  `translate()`) makes every `.usm` open/stat fail ENOENT. With all movies denied, DOLL runs the
  identical frame loop — flips climb 2669 -> 6510, still 0 draws. Confirms (again, cf. session 1) the
  opening cutscene is coincidental, not the blocker.
- **Safe-area ratio was a real bug but NOT the draw gate.** `sceSystemServiceGetDisplaySafeAreaInfo`
  (1n37q1Bvc5Y) was unimplemented -> returned SUCCESS with the out-struct `{float ratio; u8[128]}`
  unfilled, so ratio read 0.0 (degenerate title-safe rect -> collapsed viewport was a plausible
  cull-everything cause). Fixed to fill ratio=1.0 (shadPS4 contract). Verified: **still 0 draws**
  (flips 6463) — so a zero safe-area was not what gated geometry. Kept as a correctness fix (same
  success+unfilled-out-struct class as GetStatus/ParamGetString/TmpMount2).

**So the #232 wall is: DOLL reaches a stable engine tick loop with async loading complete, but its
game/world flow never activates a scene that submits geometry.** The per-frame state machine at
eboot+0x5044740 (reads .data bool gates at 0x9803460/0x9803470, doubles at 0x9603030..48) is the
game's own boot/flow tick; something it polls has not flipped to "enter gameplay/level visible".
Threads named **DollLevelPreloa**(der), SaveLoadUpdate, DLCDataUpdate, ShareUpdate all sit in the
same free-list wait (eboot+0x2316b41/0x2316d50) — game-side loader/update workers idle. Several NIDs
the boot flow calls are still bare unimpl->0 stubs, now name-resolved (tools/dbg/nidguess.py):
**scePlayGoInitialize (ts6GlZOKRrE), scePlayGoOpen (M1Gma1ocrGE), sceSaveDataInitialize3
(TywrFKCoLGY), sceNpTrophy2CreateHandle/RegisterContext, sceShareInitialize** — PlayGo/SaveData are
the most likely to gate a "content ready -> start game" transition (a game waiting on PlayGo locus =
"chunk installed" or a SaveData mount result before it loads the first level). **Next: implement
scePlayGo (report everything installed/locus-local) and sceSaveData (mount succeeds, empty) with real
out-struct contracts, and trace the eboot+0x5044740 state machine's .data gate to the subsystem it
polls.**

### SESSION 5 (2026-07-10, issue #232): the PlayGo/SaveData/Trophy2/Share service contracts are implemented and the save flow now completes end-to-end — but still 0 DrawIndex; the "0x5044740 gate" hypothesis is DISPROVEN

Implemented the real contracts for the services DOLL's boot flow calls (all NID<->name pairs
verified against the PS5 3.20 library stub tables in `../PS5-3.20_Libs`; cross-checked shadPS4 +
Kyty where the API is PS4-inherited):

- **scePlayGo** — `Initialize`/`Open`/`GetLocus`/`GetProgress`/`GetToDoList`/`GetChunkId`/`GetEta`/
  `GetInstallSpeed`/`GetLanguageMask`/`Close`/`Terminate`. Reports everything installed &
  locus-local: `Open` writes handle=1 (was success+unfilled → the game queried loci with garbage);
  `GetLocus` fills every entry LOCAL_FAST(3); `GetProgress` done==total; `GetToDoList` empty.
  Live: **scePlayGoGetLocus is polled 1004×/run** — a per-chunk sweep (chunkIds 0x0001..0x03e7,
  ~1000 chunks) that now answers "local" for all. CONFIDENCE HIGH (two PS4 references agree).
- **sceSaveData (PS5 native surface)** — `Initialize3`/`CreateTransactionResource`/`Mount3`/
  `Prepare`/`Commit`/`Umount2`/`DirNameSearch`/`Terminate`. The **Mount3 desc layout was RE'd from
  DOLL's own wrapper** (eboot+0x2251610 disasm + live `PROSPER_SVCLOG` capture): `{u32 userId@0;
  char* dirName@8; u64 blocks@0x10; u32 mode@0x20; u32 txResourceId@0x28}`. Real fresh-console
  semantics: open-mode (mode&4==0) of a missing save → NOT_FOUND(0x809F0008); CREATE-mode
  (mode 4 or 0x20) makes the host save dir and mounts guest **/savedata0** onto it (new hle_file
  translation, PROSPER_SAVE0, default `/tmp/prosper-savedata0/<dirName>`). MountResult filled with
  the PS4 shape (mountPoint "/savedata0"@0, requiredBlocks@0x10=0, mountStatus@0x1c). **Live: the
  full flow now runs clean** — `Book` open→NOT_FOUND ×3, then CREATE→OK, remount(open)→OK,
  Prepare, Commit, Umount2; the game writes `SystemSaveData999.dat`/`LanguageSaveData998.dat` into
  the mounted dir. DirNameSearch (PS4 contract pinned by callsite eboot+0x224e920) → 0 hits (fresh).
  CONFIDENCE: HIGH on mount-desc + mode semantics; MED on the MountResult field offsets (PS5
  placement unproven; PROSPER_SVCLOG hexdumps the struct the guest reads for future verification).
- **sceNpTrophy2 lifecycle** — `CreateContext`/`CreateHandle`/`RegisterContext` succeed with valid
  ids (info queries stay "unavailable", the clean offline path). **sceShare** + **NpUniversalData
  System** succeed inert. All confirmed firing once each and returning cleanly.

**The "0x5044740 reads .data gates 0x9803460/0x9803470" hypothesis (sessions 4) is WRONG.** Static
xref scan: every reference to 0x9803460/0x9803470 is *inside* fn 0x5044740 itself, wrapped in
`__cxa_guard_acquire/release` (libc NIDs 3GPpjQdAMTw/9rAeANT2tyE, resolved via gotwho) around
function-local **static double** init (frame-timing constants at 0x9603038/40/48, computed from
`sceKernelReadTsc`). Live: both "gates" read **0x01 = the cxa-guard "initialized" flag**, not a
game-state bool. So 0x5044740 is a per-frame **timing throttle** with lazy static init — NOT the
level-activation gate. The services answering did not (and could not) "flip" it; it was always
just a once-init guard.

**The real remaining wall (precisely):** at steady state (13k+ flips, save flow complete), the
game-side workers **DollLevelPreloader / SaveLoadUpdate / DLCDataUpdate / ShareUpdate are idle UE
task-graph pool threads**, all parked in the same generic `scePthreadCondTimedwait` task-wait
(eboot+0x2316b41→0x22a5fe0, the pool worker loop) — they are named for their typical task but are
waiting for work to be **enqueued**, which never happens. FAsyncLoadingThread is drained (parked on
its zenaphore 0x22ea954); IoDispatcher/IoService idle; RenderThread/RHIThread park in the AGC submit
wait. Packet histogram unchanged: **0 DrawIndex, all DispatchDirect** (compute-only). So: the
service layer the level-preload path depends on now answers correctly and the save/PlayGo/trophy
bring-up completes, but **the game never enqueues the level-preload task** that would submit scene
geometry. The gate is upstream of the services — in whatever game-flow condition decides "begin
level load" (a menu/flow state machine advancing through `GameThread` vtable `*0x288` at
eboot+0x5044775, or an online/NP-state or first-run condition). Next: trace the GameThread's
`*0x288` state-advance object across frames to find which state it is stuck in and what predicate
keeps it from transitioning to "load first level"; and audit the one-shot online init calls the
flow makes right before settling (libSceNet/NetCtl/Http/Ssl/NpCppWebApi/NpWebApi2 all still
unimpl→0 — a first-boot "check for update / entitlement" flow may spin on one of them, or on an
sceNpState it reads as a wrong value).

Verification: ctest **63/63**; services RE'd from PS5 stub tables + guest disasm (no fabrication);
save flow verified end-to-end via PROSPER_SVCLOG + on-disk save files. Probes added:
run232svc.sh, mount232.{gdb,sh}, soak232.sh, state232.py, runstate232.sh, gates232.py. New knob
PROSPER_SVCLOG=1 (arg + page-guarded hexdump of the service family). Messenger smoke: see PR.

### SOLVED (2026-07-10, issue #232): DOLL's scene geometry was dropped at THREE unimplemented Gen5 AGC draw builders — implementing them makes DOLL render its first scene geometry (up to 94 draws/submit; frames present)

**The 6-session "the game never emits geometry / 0 DrawIndex" wall was a MISDIAGNOSIS.** DOLL's UE4 RHI
submits ALL scene geometry through the Gen5 indexed-draw builder trio, every one of which was an
`unimplemented -> 0` stub that appended **no packet** — so every draw was silently dropped *inside the
HLE boundary*, before it ever reached the PM4 stream the histogram counts. That is why the histogram
showed 0 DrawIndex despite the game running deep: the draws were destroyed one call below where anyone
was looking. The trio (NIDs verified against `../PS5-3.20_Libs`; live-call counts via `dump_call_log`):

- **`l4fM9K-Lyks` = sceAgcDcbSetIndexBuffer** — 5.46M calls/run
- **`8N2tmT3jmC8` = sceAgcDcbSetIndexCount** — 5.46M calls/run
- **`B+aG9DUnTKA` = sceAgcDcbDrawIndexOffset** — 5.46M calls/run

**ABI RE'd from live capture (`PROSPER_GFXLOG` dumps a0..a3):**
`setIndexBuffer(dcb, indexAddr=0x201d7c0000)`; `setIndexCount(dcb, 0xc350=50000)` (the index-buffer
CAPACITY, not the per-draw count); `drawIndexOffset(dcb, startIndex, indexCount, modifier=0x40000000)`
— the per-draw index count is arg2 (observed 3, 6, 0x24, 0xfa8, 0x2490, …), the start offset arg1.

**Fix (correctness-first, CONFIDENCE MED on arg roles):** three new PM4 sub-ops (R_INDEX_BASE 0x1b /
R_INDEX_COUNT 0x1c / R_DRAW_INDEX_OFFSET 0x1d, hle_agc.cpp), decoded (pm4_decode) into new Kinds
`SetIndexBase`/`SetIndexCount`/`DrawIndexOffset`, and threaded through `GpuState` (new `index_base` /
`index_num`) so DrawIndexOffset emits an indexed `Draw` (index_addr = base + offset·elemsize,
index_count = arg2 or the bound count). Index element size comes from the existing SetIndexType snapshot.

**Result (DOLL, live): first scene geometry renders.** 977+ DrawIndexOffset packets/steady-state,
**up to 94 indexed draws per submit**, and with `PROSPER_RENDER=1` the executor resolves DOLL's real
dynamic vertex-fetch descriptors (base=0x200a070000, **stride=40, num_records=50000** — real UE4 scene
vertex streams) and **presents frames** before the synchronous-llvmpipe/GC stop-the-world tension ends
the run (the documented render-vs-runtime cost, not a new bug). Some DOLL shaders still fail recompile
(`skip draw: recompile failed`, unsupported RDNA2 ops) — that is the recompiler frontier, downstream of
this fix, and is the next wall. **Messenger UNREGRESSED** (mandatory — shared GPU code): reaches its
real scene (vcount=1044, 643×), 1015 frames presented, **0 faults**, 0 DrawIndexOffset (never uses the
new path — the change is purely additive for it). ctest **63/63**. Probes added: flow232{,b,c}.py,
thr232.py, frame232.sh, findstr232.sh, rostr2.sh, got232.sh, call232.sh, xref232.sh, smoke_msg232.sh.

### REFRAMED (2026-07-10, issue #282): DOLL's noise is NOT unresolved descriptors — the descriptors resolve; the wall is 64KB-tile-mode texture DETILING (SW_64KB_S / SW_64KB_R_X)

Issue #282 was scoped as "T#/S#/V# descriptors don't resolve (pointer-chained EUD/SRT) → draws bind
garbage → noise." A full live investigation on `fix/issue-282-eud-descriptors` (the branch that already
carries #232 draw builders + #273 recompiler + #276 EUD descriptors + #278 submit-race) shows that
**premise is now largely obsolete**: DOLL's descriptors ALREADY resolve to real resources. What remains
is a **texture-tiling** problem, one layer below descriptor resolution.

**Evidence — descriptors resolve (PROSPER_GFXLOG `[restab]`/`[t#]`/`[dynvb]`, full 300 s boot):**
- **Constant buffers** decode to real guest addresses + sizes (e.g. `cls=0 binding=34 addr=0x1280e74220
  size=720`, matrices readable).
- **Textures (T#)** decode to real dims + real Gen5 IMG_FMTs: `fmt=170` (BC1_SRGB), `fmt=175` (BC4),
  `fmt=182` (BC7_SRGB), `fmt=71` (Float16×4), `fmt=13` (Float16), at real sizes (512×256, 1024×1024,
  2048×2048, 1920×1080, …) with plausible bases in the `0x20xxxxxxxx` texture heap.
- **Vertex buffers** resolve via the existing dynamic-fetch const-fold (`resolve_dynamic_fetch`) across a
  healthy range: `num_records` = 6, 8, 96, 200, 887, … up to **49992 / 50000** (real UE4 scene streams),
  stride 4/8/12/32/40. The fetch chain (`s_load_dwordx4` from a user-data-SGPR table pointer) is followed
  correctly. So the "pointer-chained V# unresolved" hypothesis does not hold for the current branch.

**Evidence — the noise is tiled textures read as linear (frame BMP, `PROSPER_DUMP_CONTENT`):** a captured
content frame (3840×2160) is a full-screen **repeating RGB block-weave** with a few black rectangles — the
textbook signature of a GPU-tiled surface sampled as if linear (geometry lands in the right screen regions;
the *texel content* is scrambled). It is NOT scattered geometry (positions are fine) and NOT a black frame
(draws execute + sample).

**Root cause — the detiler only knows SW_4KB_S (tile_mode 5); DOLL's textures are 64KB-tiled.**
`prosper::gpu::tile_mode_is_tiled` (src/gpu/tile.cpp) returns true ONLY for `tile_mode == 5`
(`TileMode::Sw4KbS`, the one mode The Messenger uses). Tile-mode histogram over a full DOLL boot
(`[t#] tile_mode=` counts):

| tile_mode | GFX10 swizzle          | count  | detiled? |
|-----------|------------------------|--------|----------|
| 0         | LINEAR                 | 11861  | n/a (linear) |
| 27        | SW_64KB_R_X (RT)       | 10665  | **NO → noise** |
| 9         | SW_64KB_S (standard)   |  9118  | **NO → noise** |
| 1         | SW_256B_S              |   780  | NO |
| 5         | SW_4KB_S               |   767  | yes |
| 24        | SW_64KB_Z_X (depth)    |   321  | NO |

So ~58% of DOLL's sampled textures use the two unhandled 64KB modes (9 and 27), and every one of them is
memcpy'd as linear → scrambled → the composite is noise. Mode 27 (SW_64KB_R_X) is the render-target
swizzle (the fullscreen post/deferred composites); mode 9 (SW_64KB_S) is regular material textures
(BC1/BC4/BC7).

**Derivation attempt (offline, coherence scoring) — 64KB_S is NOT a flat x/y bit-interleave.** Added a
gated diagnostic `PROSPER_DUMP_TILERAW` (src/gpu/agc_shader_layout.cpp) that dumps a tiled texture's raw
guest texel bytes at T#-decode time (fires every draw's stage build, so it captures textures even on the
intermittent boots whose content never reaches the render backend — 75 textures captured in one 120 s
run). For a 512×256 BC1 (exactly one 64KB tile = 128×64 blocks × 8 B) and a content-rich 512×512 BC1
(2 stacked tiles), a brute-force over all 1716 order-preserving interleavings of the 7 X-bits + 6 Y-bits,
scored by decoded-image local coherence (Σ|Δ neighbour|, low = smooth), found **no decisive winner**: the
best candidate improves coherence only ~4% over row-major and, rendered, is **still noise** (only the
top-left micro-tile is roughly coherent). Conclusion: unlike SW_4KB_S — which this codebase models well
enough with a flat Morton (`sw4kb_morton`) — **SW_64KB_S/_R_X need the real GFX10 addrlib swizzle equation**
(256 B micro-tile hierarchy + the "S"/"R_X" macro pattern; "_X" adds a pipe/bank XOR), not a bit-interleave.
Kyty is **not** a reference here (it is GCN/PS4 tiling, per CLAUDE.md). This is a bounded but genuine
graphics-RE task, tracked as **#288**; shipping a guessed detiler would violate correctness-first
(it would produce different-but-still-wrong pixels).

**Also observed (separate wall, downstream): ~6120 draws/run `recompile failed`, ALL vertex-shader**, with
coverage `first_bad fmt=5(SMEM) op=0x2(s_load_dwordx4)` — but that coverage is computed with `rt=nullptr`
(a static probe), so it flags the register-offset descriptor load that the *real* path (`rt != null`)
no-ops; the true drops are the remaining unsupported RDNA2 ops (`unsupported=2..14` in the same coverage).
That is the **recompiler frontier (#273 continuation)**, not descriptor resolution — those draws are
*dropped*, so they subtract geometry but do not create the noise.

**Net for #282:** the descriptor-resolution work it asked for is effectively already done; the real work is
split to (a) **#288 — 64KB tile-mode detiling** (the noise) and (b) **#273 — recompiler frontier** (dropped
draws). New diagnostic landed: `PROSPER_DUMP_TILERAW` (reusable for the #288 derivation). Messenger
unaffected (mode 5 only; the diagnostic is gated + additive). CONFIDENCE: HIGH on the reframing (live
frame + tile-mode histogram + descriptor dumps); the detiler itself is deliberately NOT shipped.
