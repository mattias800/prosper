# The render-loop frontier — post-boot-wall (2026-07-06)

**Status:** open (the next big frontier). The GfxDevice boot wall is RESOLVED (see
`GFXDEVICE_BRINGUP_PROBLEM.md`); the game now boots into asset loading + its frame loop but does not yet
render. This documents exactly where it is and what's needed, so the next session picks up cleanly.

## Where the game is now
- Boots through IL2CPP → GfxDevice bring-up → **loads `level0` + scene assets** (`sharedassets0.assets`,
  `unity_builtin_extra`, `globalgamemanagers`) → initializes audio (Ajm/AudioOut) → enters its frame loop.
- **Live and busy but not rendering:** ~48 threads idle in `k_wait_on_address` (job pool), a scheduler
  thread ping-ponging semaphores, an audio-mix thread (float accumulate at eboot+0x195d8e0). No draws.
- The **TRC R5089 "Forcing sce::Agc::suspendPoint" watchdog** (eboot+0x14eb200) fires periodically because
  no frame is presented (`(frame_counter − last_suspend) ≥ 3` forces a suspendPoint).

## What was fixed this session (render-loop heartbeat)
- The game drives frames via the **kqueue/kevent** model: `sceKernelCreateEqueue` ×4 ("Flip Event Queue
  GfxDeviceAgc", "EOP QUEUE", "UnityFTMFlipQueue", "eq to wait flip"), `sceVideoOutAddFlipEvent`, then
  `sceKernelWaitEqueue` for flip-completion events. Our equeue was fully stubbed → those queues never got
  events → two threads spun 18000× in WaitEqueue → no frame advanced.
- **FIXED (committed):** real equeue backend (`hle_kernel_time.cpp`) + `sceVideoOutAddFlipEvent/AddVblankEvent`
  registration + a ~60 Hz pump posting flip/vblank `SceKernelEvent`s + a blocking `WaitEqueue`. The
  flip-timing thread (eboot+0x14bd450) now cycles at ~60 Hz; the watchdog fires ~half as often. Gated
  `PROSPER_EVLOG` traces the handshake.

## The actual blocker (precisely located)
The **main thread is parked in Unity's PreloadManager** (the singleton at `eboot+0x1ff5228`, string
"Loading.PreloadManager"), in a **semaphore-gated work-queue consumer** entered via GfxDevice
`0xaee233 → 0xb06940`: `lock xadd -1,[singleton+0x130]` (count); if empty, block on the sem object
`[singleton+0x128]` via the generic wait helper `0x18a82a0` (a `sync_on_address` semaphore-with-timeout).
- **Verified with a HW write-watchpoint:** the count word is only ever decremented by the consumer itself
  — **no producer ever enqueues.** The main thread times out and re-waits forever.
- **Chain:** main thread → PreloadManager finishing `level0` → GfxDevice **GPU-resource upload** (create
  textures/buffers/pipelines for the loaded assets) → waits for the upload to complete on the GPU/render
  thread → never signaled (we don't execute GPU commands or post completion). No `SubmitDcb` fires, so the
  game is stalled *before* rendering, in the asset-upload step.

## What's needed next (the M4/M5 GPU-execution frontier)
Getting past `level0` load — and then to actual frames — requires **real AGC→Vulkan GPU command execution
+ completion signaling**, not another small fix:
1. **Execute submitted command buffers for real** (the back-half `run_command_buffer → GpuState` exists and
   renders offscreen in tests; wire it to a live Vulkan device + upload/execute the game's actual submits).
2. **Post GPU end-of-pipe (EOP) completion events** to the "EOP QUEUE" when a submit completes, so the
   GfxDevice upload wait returns and the PreloadManager producer enqueues the main-thread job.
3. Then the render loop should cycle: submit draws → EOP → flip → present → next frame.
- **Note:** the exact EOP registration mechanism isn't observable yet (the game submits nothing pre-upload),
  so implementing EOP requires either reaching the first submit or Gnm/AGC reference for the EOP-label +
  equeue-event association.

## Tooling added this session (all in `exec_image_linux.cpp`, gated, non-destructive)
- `PROSPER_HWBP=0xOFF` — race-free x86 HW execute breakpoint (perf_event_open) that logs registers +
  caller; safe on hot/multi-threaded code where the int3 logger races.
- `PROSPER_HWWATCH[=delta]` + `PROSPER_HWWATCH_REG=<reg>` — chained HW data write-watchpoint on
  `[reg+delta]`, armed on the first exec-bp hit; logs each writer's RIP + a `/proc/self/maps` classifier.
- `PROSPER_EVLOG` — traces the equeue + `sceVideoOut*` flip/vblank calls (how the handshake was mapped).

## Recommendation
The render milestone is the GPU-execution frontier — a large, multi-session build best done deliberately
(and it can't be fully validated until the game issues its first real submit). Meanwhile, reliable
committable progress toward full-game rendering = expanding RDNA2→SPIR-V recompiler coverage on the game's
real shaders (each spirv-val + execution verified). When GPU execution lands, more covered shaders = more
of the game renders.

## 2026-07-06 (overnight) — hard-stall confirmed; no easy crack; needs GPU execution
Exhaustively re-probed the stall. Findings:
- The last new AGC call before the stall, `libSceAgc::h9z6+0hEydk`, is called from `eboot+0x14eb1f2` —
  inside the TRC watchdog: it **IS `sce::Agc::suspendPoint`** (a per-frame checkpoint), not a blocker.
- "path … not considered suitable for apr reads" is a **benign** Unity async-page-read decision — it
  prints for files that loaded fine (globalgamemanagers etc.), not a stall indicator.
- gdb thread dumps during the stall show **every thread idle-waiting** (futex / nanosleep / audio-mix /
  scheduler) with **no thread in a read syscall** — the load is not actively progressing; it's a quiescent
  stall, and the PreloadManager work-queue producer never runs (count only ever decremented by the
  consumer). So it is NOT a lost-signal a punch could safely fix — the item is never produced (a punch
  would dequeue a phantom → crash).
**Conclusion:** the render-loop unblock is not crackable by AGC-call/file/semaphore shims in autonomous
iterations. It genuinely requires the **GPU-execution build** (execute submitted command buffers on a live
Vulkan device + write back GPU memory + post EOP/flip completion so Unity's asset-integration/upload
choreography advances) — a large, focused, likely reference/interactive effort — OR a deliberate
punch-through experiment done with a human watching each cascade. This is the clear next milestone for the
game's actual pixels; the recompiler and the offscreen `GpuState→frame` spine are both ready to receive
real draws the moment it lands.

## 2026-07-06 (later) — CORRECTION: the game gets much further than "no submit"
Re-ran under `PROSPER_EVLOG=1 PROSPER_GFXLOG=1` (WSL Linux build) with new event-delivery tracing. The
earlier "no SubmitDcb fires" claim was wrong — the game reaches deep into GfxDevice bring-up:
- **Creates 35 shaders** (`sceAgcCreateShader`, types ps/vs/cs — so the recompiler WILL be exercised).
- **`SubmitDcb #1`**: 71 dwords → 12 packets, **0 draws** (a state/setup submit, no geometry yet).
- **`SubmitFlip bufidx=-1 flipmode=1 arg=0`**: one initial blank flip (no buffer presented).
- Four equeues: `eq to wait flip`, `UnityFTMFlipQueue`, `EOP QUEUE`, `Flip Event Queue GfxDeviceAgc`.

Two steady-state waiters (the frame loop), both spinning:
- **Flip thread** `eboot+0x14bd47f` waits on *Flip Event Queue GfxDeviceAgc*. Our ~60 Hz pump posts flip
  events here and they **are delivered** (traced 1167× with ident=0x1001, filter=−10). The thread's loop
  (`0x14bd450`) processes each event and continues — flip handling is **working**.
- **FTM thread** `eboot+0x14dfb43` waits on *UnityFTMFlipQueue* for **user event id=999**
  (`sceKernelAddUserEvent(eq,999)` is called once). That user event is **never triggered**
  (`sceKernelTriggerUserEvent` is called 0×), so this thread never advances.

Root deadlock (unchanged in nature, now precisely located): the FTM user-event-999 producer and the
PreloadManager work-queue producer both live **downstream of GPU-resource-upload completion**, which we
never signal (no live GPU execution + no EOP writeback). Implementing the user/timer event sources for
real (done this session — `sceKernelAddUserEvent/TriggerUserEvent/AddHRTimerEvent/AddTimerEvent`, was
no-op) is correct HLE and verified by `test_equeue_events`, but does **not** unblock: the trigger call is
itself gated by the upstream stall. Confirmed empirically — boot reaches the identical suspendPoint loop
after the fix. **The GPU-execution build remains the real unblock.** New permanent `PROSPER_EVLOG` traces
(event delivery `-> delivered N ev(s)`, and the user/timer registration+trigger lines) make this loop
re-diagnosable in one run.

## 2026-07-06 (breakthrough) — the exact GPU→CPU fence handshake is the unblock
Dumped `SubmitDcb #1`'s 12 packets (`PROSPER_GFXLOG` now prints each packet's kind + payload, and the
ReleaseMem/WaitRegMem/WriteData builders now log their args). The setup submit is a **GPU-fence
handshake**, not draws:
```
  DrawReset | WaitFlipDone(h=0x1001) | ReleaseMem | WaitRegMem | ReleaseMem×2 |
  Flip(h=0x1001, bufidx=2) | AcquireMem | EventWrite(0x2e) EventWrite(0x2c) | ReleaseMem×2
```
The **ReleaseMem** (EOP) and **WaitRegMem** builder args expose the real pointers we were discarding
(captured incl. the guest-stack args a6..a8 via `__builtin_frame_address` — the stub tail-jumps, so
[rbp+16..]=a6..):
```
ReleaseMem  a1     a2     a3   a4    a5=dstGpuAddr     a6    a7(value?)   a8   (ret=eboot+0x3ae3dc)
            0x28   0x0    0x1  0x0   0x..715c4aa8      0x2   0x1          0x0   <- WaitRegMem polls THIS addr
            0x28   0x0    0x1  0x0   0x..715d3810      0x3   0x0          0x0
            0x14   0x200  0x1  0x3   0x..55464f0       0x2   0x4          0x0
            0x4    0x200  0x1  0x3   0x..55c0550       0x2   0x1bfba062   0x0   <- a7 looks like a real fence value
WaitRegMem  a1=0   a2=0x3 a3=0 a4=0x2(cmpFunc) a5=0x..715c4aa8  a6=0x1  a7=0xffffffff(mask?)
```
So a5=dstGpuAddr (confirmed: WaitRegMem waits on the SAME address a ReleaseMem writes), a3=0x1 is the
immediate-write selector, and the EOP value is in the stack args (a6/a7 — a7=0x1bfba062 is a plausible
fence value, but the exact value/selector/width mapping and WaitRegMem's mask/ref/cmpFunc encoding are
NOT yet disambiguated). `PROSPER_GFXLOG` now dumps all of this every run.

## 2026-07-06 — EOP fence writeback tried; NOT the CPU gate + full thread deadlock map
Implemented the fence writeback (`command_processor.cpp` `maybe_fence_write`, `PROSPER_AGC_FENCE`-gated):
on submit, write the completion value to `[dstGpuAddr]`. Tested all 5 value variants (a7 / a6 /
0xFFFFFFFF / 64-bit a7:a6 / 64-bit all-ones). **Result: none unblocks the game** — identical suspendPoint
loop, no draws, user-event 999 never triggered. So the EOP fence is a GPU-internal handshake; **no
stalled CPU thread is gated on the fence label.** (The writeback is kept — gated + `CONFIDENCE`-tagged —
as correct GPU-execution infra needed later.) Also verified our sync_on_address futex is correct
(`FUTEX_WAIT` re-checks `*addr==expected` atomically → no lost-wake race with the check-then-wait loops),
so producers genuinely never post.

**gdb 55-thread snapshot during the stall (the real topology):**
| Thread | Wait site (eboot+) | Role | Blocked on |
|---|---|---|---|
| 1 (main) | 0x18a83b5 | Unity PreloadManager | sync_on_address semaphore `[r14]`, count≤0 forever |
| 2 | 0xb0672a | GfxDevice work-queue (region 0xb06940) | sync_on_address semaphore `[r12]`, count≤0 |
| 11 | 0x9385d7 | (unidentified worker) | sync_on_address semaphore `[rbx]`, count≤0 |
| 7 | 0x14bd47f | flip thread | WaitEqueue (alive — processes ~1167 flips) |
| 8 | 0x14dfb43 | Unity FTM | WaitEqueue for user-event 999 (never triggered) |
| 9 | 0x14eb148 | TRC watchdog | nanosleep (forces suspendPoint) |
| 3,4 | 0x194c441/0x194c403 | scheduler | nanosleep / mutex on 0x194c2c0 |
| 5 | 0x195d8f0 | audio mix | — |
| 42 | Il2cpp+0x1e23b8 | C# thread | syscall |
| ~44 | 0x18ab088 / 0xae1af9 | job pool (idle) | sync_on_address, no work |

**The deadlock:** three threads (main / GfxDevice / worker-11) each block on a sync_on_address semaphore
whose count stays ≤0 — a producer/consumer cycle where the root kick never fires. All three wait loops are
the same shape: `mov (%reg),%rax; test; jle →WaitOnAddress(0x19b4050); else lock cmpxchg (decrement);
proceed`. The post side is `lock add [sem]; WakeByAddress(0x19b4060)`. The break-in event is NOT the EOP
fence (ruled out above) and NOT a futex bug (ruled out). **Next leads:** (a) identify thread 11's semaphore
(0x9385d7) and thread 2's (0xb0672a) — which produces which — to find the cycle's root; (b) check whether
the root expects a GPU-submit *with draws* + its EOP-QUEUE *event* (RELEASE_MEM interrupt → equeue event,
distinct from the label write we tried) — the "EOP QUEUE" equeue exists but nothing posts to it; (c)
whether Unity's async-load-completion callback (should post main's semaphore after level0 integrates) runs
on an idle pool thread that returns without posting. Tooling: `PROSPER_AGC_FENCE`, and the gdb snapshot
recipe (attach to the boot_trace child, `handle SIG34-40 nostop noprint pass`, `thread apply all bt`).

## 2026-07-06 — sceKernelWaitOnAddress ignored its timeout → BLOCKS FOREVER (bug found)
The three stuck threads call `sceKernelWaitOnAddress(addr, expected, &timeout, 0)` (the thunk is
eboot+0x19b4050; wake side eboot+0x19b4060 = `sceKernelWakeByAddress`). **Our impl passed `nullptr` for
the futex timeout, so every bounded wait blocked forever** and could never reach the guest's
timeout-exhausted branch. Fixed to honor `a2` (a `uint32` microseconds pointer; most waits pass a2=0 =
infinite, but ~25 pass 1000 µs) AND to return `SCE_KERNEL_ERROR_ETIMEDOUT` (0x80020060) on a real timeout
instead of always 0 — returning 0 told the guest its semaphore was *signaled* when it wasn't (phantom).

**Effect (gated behind `PROSPER_WAIT_TIMEOUT` for now):** with the timeout honored, the game leaves the
quiescent suspendPoint loop and takes a **new path** — it reaches audio init and then **throws a C++
exception**, whose stack unwind crashes: `sceKernelGetModuleInfoForUnwind` (libkernel `RpQJJVKTiFM`) is
unimplemented → libunwind reads garbage → `decodeEHHdr … Unsupported .eh_frame_hdr version` → stack-smash
abort. So **the game uses C++ exceptions for control flow (incl. wait timeouts), and our broken unwinder
makes any throw fatal.** This is likely a bigger unlock than the fence: with a working unwinder the throw
would be caught and the game could proceed.

**NEXT (concrete):** implement `sceKernelGetModuleInfoForUnwind(addr, flags, info*)` — given a code addr,
find the containing module and fill the `ModuleInfoForUnwind` struct (0x130 bytes: st_size@0x00,
name[256]@0x08, eh_frame_hdr_addr@0x108, eh_frame_addr@0x110, eh_frame_size@0x118, seg0_addr@0x120,
seg0_size@0x128). We have the data: each `Module` carries all program headers, and the `PT_GNU_EH_FRAME`
(0x6474e550) segment IS `.eh_frame_hdr`. Plumb per-module {base, eh_frame_hdr va/sz, seg0 va/sz, name} to
the HLE (mirror `set_tls_modules`), then fill the struct. Once the unwinder works, un-gate
`PROSPER_WAIT_TIMEOUT` and see how far the exception-driven path gets. Default stays infinite-wait (stable,
no crash) until then.

## 2026-07-06 — the exception path, followed to its end (unwind + TLS fixed; it's a SYMPTOM)
Implemented `sceKernelGetModuleInfoForUnwind` (committed) — libunwind now walks the full stack cleanly
(libc/Il2cpp/eboot, no eh_frame error). That exposed a SIGFPE in our own `k_tls_get_addr`: it kept the
per-thread DTV in a host `thread_local`, which under the guest %fs aliases guest memory → the map's
bucket_count read 0 → `hash % 0` → SIGFPE. Fixed by keying the DTV on `gettid` in a mutex-guarded global
map (committed; same %fs landmine class as the boot wall). With both fixed, under `PROSPER_WAIT_TIMEOUT=1`
the throw now unwinds + resolves TLS and runs a **guest handler at eboot+0xa9c0bb**, which derefs `[rdi+8]`
with **rdi=null** → SIGSEGV. The unwind reached the top of the stack (eboot+0xae ≈ entry) — i.e. the
exception is **UNCAUGHT**, reaching a terminate/error path.

**Conclusion — the timeout/exception cascade is a symptom, not the route to frames.** The stuck waits have
a *finite total budget* (the main/GfxDevice wait loops compute one and, when exhausted, branch to a
give-up path that throws). On real HW the awaited resource arrives within budget because the producer runs;
our **missing semaphore producer** means the budget expires → the game throws its own "timed out" error →
(uncaught) crash. So honoring the timeout just walks the game into its timeout-error handling; it does NOT
render. The productive root remains **the 3-thread sync_on_address producer that never posts** (see the
thread map above). The unwind + TLS fixes are correct and valuable regardless (C++ exceptions will occur
legitimately later). `PROSPER_WAIT_TIMEOUT` stays gated (off) so master is stable. NEXT: go back to the
producer — identify which of {main 0x18a83b5, GfxDevice 0xb0672a, worker 0x9385d7} semaphores is the cycle
root and what game event (GPU submit-with-draws + EOP-QUEUE event? async-load-completion callback?) is
supposed to post it within the budget.

## 2026-07-06 — punch-through experiment: clearing consumer waits does NOT create work (dead end)
Added a gated diagnostic (`PROSPER_PUNCH=<secs>`, off by default, `hle_kernel_mem.cpp`): an INFINITE
sync_on_address wait from one of the three stuck sites that stays blocked past `secs` fabricates its awaited
signal (bump `*addr`, wake) so the thread proceeds — purely to observe what's *downstream* of the deadlock.
Findings:
- The **GfxDevice (0xb0672a)** and **worker (0x9385d7)** waits are **infinite** (a2==0); the **main
  PreloadManager (0x18a83b5)** wait is **finite/budgeted** (a2=&timeout) — so on real HW main gets its
  signal within budget while the two workers wait indefinitely for work.
- Punching GfxDevice+worker every 3 s: they wake 14× each, **do nothing useful, and re-block** — no crash,
  **no SubmitDcb-with-draws, no advance** (still only the setup submit, 0 draws; suspendPoint loop
  continues). Main is never reached (it's not an infinite wait, so unpunched).
- **Conclusion: the deadlock is producer-side, not a lost signal.** Waking the consumers creates nothing
  because there is no real work item to process — the producer that would ENQUEUE work + post the
  semaphores is itself gated on GPU/asset-integration completion we don't provide. No consumer-side shortcut
  exists. (Consistent with the earlier HW-watchpoint finding that the PreloadManager count is only ever
  decremented.) **The GPU-execution + completion build is the required path** — execute the submitted Dcb on
  a real device and post the completion the producer waits on, so it enqueues real work → main proceeds →
  draws. The recompiler (34 shaders) and the offscreen GpuState→frame spine are ready to receive those draws.
The punch tool stays (gated off, `CONFIDENCE: LOW`, fabricates a fake for observation only).

## 2026-07-06 — EOP QUEUE is never waited on (last completion mechanism ruled out)
Confirmed over a full 30 s run: of the four equeues the game creates, only **UnityFTMFlipQueue** and
**Flip Event Queue GfxDeviceAgc** are ever `WaitEqueue`d (295 and 1762 times). The **"EOP QUEUE" equeue is
waited on 0 times** — so posting GPU end-of-pipe *events* there (the RELEASE_MEM-interrupt→equeue-event
form of completion, distinct from the fence-label write already ruled out) would have no consumer and
cannot unblock anything.

### Ruled-out unblock hypotheses (all tested this session)
| Hypothesis | Result |
|---|---|
| EOP fence **label** write on submit (5 value variants) | no effect — no CPU thread polls it |
| Honor WaitOnAddress **timeout** → guest handles it | guest throws an UNCAUGHT "timed out" exception (symptom) |
| **Punch** the stuck consumer semaphores | they wake, find no work, re-block — no draws, no advance |
| Post **EOP-QUEUE events** | no consumer — EOP QUEUE is never waited on |
| sync_on_address **lost-wake** bug | ruled out — FUTEX_WAIT re-checks atomically |

**Net:** there is no consumer-side or shim-level shortcut. The producer that would enqueue the
PreloadManager/GfxDevice work and post the semaphores is gated on **real GPU/asset-integration work
completing**, which requires executing the submitted command buffer on a real device and posting the
completion via whatever mechanism that producer actually polls (still unidentified — none of the obvious
ones above). This is the GPU-execution build; it is the definitive next milestone for game pixels and is a
focused, likely-interactive effort. Everything upstream (boot, IL2CPP, asset load, shader creation ×35,
recompiler ×34, flip loop, offscreen GpuState→frame spine) is in place.

## 2026-07-06 — guest-fs hardened (worker free-order + signal-handler %fs); new frontier = allocator returns null
Two follow-up fixes made the guest-fs path robust (both committed, gated):
- **Worker free-order:** `thread_trampoline` switched `%fs` to guest BEFORE `free(ts)`; `free` (host glibc,
  tcache in host TLS) then corrupted the host heap → intermittent null-alloc crash. Reordered so all host
  libc runs on host `%fs` and `guest_tls_activate_thread()` is last.
- **Crash-handler %fs:** a fault on a guest-`%fs` thread entered `fault_handler` on the guest `%fs`, so its
  snprintf/write + siglongjmp-return double-faulted → uncaught core dump. `guest_fs_enter_host_for_signal()`
  (magic-guarded) restores host `%fs` on the FATAL path only (SIGTRAP diagnostic paths + the GC RT-signal
  handler keep the guest `%fs`). Faults are now caught + reported cleanly.

**NEW FRONTIER (post-guest-fs, caught deterministically): `eboot+0x46beb4`.** A string build:
`alloc(0x809910, size=r14+1)` → `memcpy(0x4019b41d0, len=r15)` → `movb $0,[r13+r15]` (null-terminate).
The allocator returns **null** (`r13=0`), and `r15=0`, so it writes to addr 0 → SIGSEGV. Ground truth
(gdb, main thread, guest-fs ACTIVE — `fs_base` has the `PROS` magic): the allocator global
`[0x40204b838]=0x401f5a230` IS set, so `0x809910` calls the real allocator `0x808e10` which returns null.
**ROOT (captured): the allocator is FINE — `r14` (the string length) is garbage `0xffffffffcf3e1608`** = a
negative pointer-difference (`end - begin` that underflowed at `0x46be80 sub %rax,%r15`). `alloc(r14+1)` =
~16 EiB → correctly returns null → the `movb $0,[0+0]` crash. `0x808e10` is a real bump/pool allocator
(dec/or-15 alignment, `lea 0x10001(%rbx)` overflow check → returns null on absurd size). So the bug is
**upstream: a string object's begin/end pointers are bad**, during IME init (`sceIme*` calls, which we stub
to return 0 — a likely source: the game builds a std::string from an IME struct/result we left zeroed, so
`end-begin` underflows). Sibling worker fault at a relocated `…d7b` address is likely the same string path.
**NEXT:** trace back where the begin/end (`rax`/`r15` at `0x46be80`) come from — the object at `rbx`/`r12`
(a stack std::string, `[rbx+0x24]=0x4b`) — and which `sceIme*` (or other stubbed) call should have filled
it. Implementing that call to return a sane empty-string/struct likely clears this fault. Candidate: give
the `libSceIme` stubs real zeroed-but-valid output (empty strings with begin==end, not garbage ranges).
NOTE: this fault is only reached on the `-force-gfx-direct` + `PROSPER_GUEST_FS` path (both gated);
master's default boot is unaffected.

**ROOT PINNED (deterministic — 3/3 crashes identical rip + identical len `0xffffffffcf3e1608`): Unity
binary-asset DESERIALIZATION is misaligned.** Real caller chain (raw stack walk, rbp chain broken):
`0x46beb4` (std::string build) ← `eboot+0x7e4115` (a `std::string::assign(ptr,len)`-shape fn; the length is
`movslq (%rax)` — a 32-bit length read straight from the stream at `[rax]`) ← `eboot+0x7fcafb` ←
`eboot+0x1612c92`. `0x1612c70..0x1612c8d` is a classic **aligned binary reader**: `eax=[rbx+0x38]-[rbx+0x40]`
(cursor − base), `add $3 / and ~3` (round to 4), `[rbx+0x38]+=…` (advance cursor), `call 0x4007fc9f0`. So
`rbx` is a stream reader (`+0x40`=base, `+0x38`=cursor) — this is Unity's TypeTree/SafeBinaryRead reader
(cf. `GFXDEVICE_BRINGUP_PROBLEM.md` §2026-07-05, `SafeBinaryRead::Transfer` @ `eboot+0xd58710`). It reads a
dword as a string length and gets `0xcf3e1608` → the reader is **misaligned** (some earlier field consumed
the wrong byte count, so the cursor now points at non-length bytes). Because it's deterministic, it's a
real parse divergence, not a race/uninit. **Candidate causes:** a TypeTree field whose size/alignment we
diverge on, or wrong bytes fed in (file I/O returning wrong data for level0/sharedassets). **NEXT (deep,
the next milestone):** trace the reader from a known-good field back to the first divergent read — e.g. bp
`0x1612c70` and watch `[rbx+0x38]` (cursor) vs the expected TypeTree layout, or verify our `pread`/mmap of
`level0` returns byte-correct data. This is Unity-asset-deserialization RE — a large surface, the clear
next frontier now that the MT-deadlock + guest-TLS walls are down and the boot reaches scene load.
- **Refinement:** the length `0xcf3e1608` is NOT present as bytes in `level0`/`sharedassets0.assets`/
  `globalgamemanagers`(.assets) (searched LE `08 16 3e cf` + BE) — and it's identical across runs (so not
  an ASLR pointer). So it's a **computed/transformed value in memory**, not raw file bytes read at a wrong
  offset ⟹ NOT a simple file-I/O byte bug; it's deeper in the deserializer's logic (a field decoded/
  transformed wrong, or a wrong reader object). Note `0x7e4115`/`0x7fcafb`/`0x7e40d9` are libc++ std::string
  internals (low eboot addrs = statically-linked libc++); `0x1612c92` is the Unity/game caller holding the
  reader object `rbx`. NEXT: capture `rax` (the length source ptr) + `rbx` (reader) live and dump what
  region `rax` points into (decompressed buffer? heap object?) — the gdb conditional needs the value
  compared as UNSIGNED (`0xffffffffcf3e1608` is negative signed, so `> 0x10000000` is false) or matched
  exactly (`== 0xffffffffcf3e1608`).

## 2026-07-06 — PINPOINTED: misaligned deserialization is reading SHADER BYTECODE as a string length
Cracked it with a **fault-time stack walk** (reliable; the HWBP single-step approach SIGABRTs under
guest-fs — stack-smashing — so it's out here, as the docs long warned). Ground truth:
- The reader fn is `eboot+0x7e4090`: a length-prefixed binary reader. `rdi`=reader object; `[rdi+0x38]`=
  cursor ptr, `[rdi+0x48]`=end ptr, `[rdi+0x40]`=base ptr. It reads a u32 at `[cursor]`, advances cursor+4,
  then builds a `std::string` of that length. Recover the reader at the fault via frame A's saved r14:
  `r14 = *(long*)($rbp-0x10)` (frame B's live r14 = `&reader+0x38`, saved by frame A's prologue), then
  `reader = r14 - 0x38`.
- Captured: `reader`=a stack object; `base=[reader+0x40]`, `end=[reader+0x48]` bound a **37KB** window inside
  a **16 MB ANONYMOUS mapping** (`0x7ffefc7c0000..0x7ffefd7c0000`, rw-p, no file backing = a guest
  flexible/direct-mem pool). The cursor sits ~`0x7f74` into that window.
- **The bytes at the cursor are RDNA2 SHADER INSTRUCTIONS** (`0xbf8c…`=s_waitcnt, `0xf800…08cf`=EXP; the
  read "length" `0xcf3e1608` is a shader-code dword). Buffer start also looks like shader code.
⟹ Unity is **deserializing a shader** (SerializedShader/subprogram bytecode) from an in-memory buffer, and
the reader is **misaligned inside the bytecode blob** — it reads a code dword as the next field's length.
The parse diverged upstream (an earlier field consumed the wrong byte count so the cursor landed inside the
blob instead of at the next length/count). Because it's deterministic + the buffer holds plausible shader
code (not garbage), the buffer DATA is likely fine — the divergence is in how the parse consumes it (a
field type/size/alignment we make the game mis-handle, or a shader-blob sub-count we feed wrong; NOT file
I/O). **NEXT:** set a fault-time bp at `0x7e40b5` (the cursor read) OR walk the reader's prior reads to find
where the cursor first diverges from the expected shader-blob layout — i.e. what field just before offset
`0x7f74` should have advanced the cursor to a real length. Cross-ref Unity SerializedShader/ShaderData
TypeTree. Tooling added this session: `PROSPER_STUBDUMP` (stub idx→NID; ruled out the stub-as-source
hypothesis — stub #319 = `__stack_chk_guard`), `PROSPER_HWBP_R15` (conditional HWBP log — works OFF the
guest-fs path only), and `guest_fs_to_host_scoped`/`_restore_scoped` (fs swap for diagnostic handlers).

## 2026-07-06 — deserializer MECHANISM fully mapped (count-prefixed string arrays); + boot made deterministic
Two big things this cycle:
1. **Fixed the consistent worker crash: a data race in `prosper_on_unimpl`.** It mutated `g_count[idx]`
   + `g_order.push_back` with NO lock; once `-force-gfx-direct`/guest-fs bring worker threads online, they
   call unimplemented imports concurrently → the unordered_map/vector corrupt → SIGSEGV *inside
   prosper_on_unimpl on a worker* (traced via the new worker-fault region classifier → it was OUR code,
   not guest). Added a mutex. Effect: boot is now DETERMINISTIC — 5/5 runs exit 0 (was 90) and reach RUN
   ENDED at the main deser fault, instead of a worker crash killing the process first. Also fixed
   single-step-under-guest-fs (guest TCB now replicates the host `%fs:0x28` stack-guard / `:0x30` pointer-
   guard, so `-fstack-protector` signal handlers running transiently under guest `%fs` don't false-abort).
2. **The deserializer is a count-prefixed string-array reader** (static RE, no perturbation):
   - `0x7fc9f0` (frame C): reads a 4-byte COUNT at `[cursor]` (`movslq`), then loops `count` times calling
     the string reader `0x7e4090`, storing into an array of `0x28`-byte items (frame C loop `0x7fcaf0`).
   - So the parse is: {read count N} {read N length-prefixed strings} … repeated for many arrays.
   - The 6000+ traced reads are MANY small arrays read CORRECTLY (sane small lengths), then one field
     diverges and the cursor walks into the shader-bytecode blob → a code dword read as a string length →
     `alloc(~16 EiB)` → null → crash.
**Why this is the autonomous wall:** the divergence is ONE field among thousands of correct reads, deep in
the boot. Single-step tracing to it is impractical (0x7e40d9 is hit for every string in the whole boot;
7 min of stepping reached 0 shader-pool reads). Fault-time capture gives only the endpoint (reader+buffer).
The exact divergent field needs **Unity SerializedShader/TypeTree format reference** (to know the expected
field layout and spot the one we consume with the wrong size/count) or an interactive reference-backed
session — not more autonomous trace/static-RE cycles. Everything upstream is now solid + deterministic:
the game reliably boots (via the two gated switches) to this single, well-characterized deserialization
divergence.

## 2026-07-06 — ⭐⭐ guest initial-exec TLS landmine FIXED; boot now reaches INPUT/IME init
The `%fs` TLS fault below is **fixed** (gated `PROSPER_GUEST_FS`, validated). Implementation
(`src/host/guest_tls.cpp` + `exec_image_linux.cpp` swap stubs, all default-off):
- Give each guest thread its own **guest TCB + Variant-II static TLS** laid out below the thread pointer
  (main exe/eboot closest to TP), tdata copied + tbss zeroed, `[TP]`=self, and run guest code with
  `%fs = guest TP` (`guest_tls_activate_thread()` at `run_entry`). Total static TLS below TP = 0x6c0 bytes
  for our 3 TLS modules (eboot 0xd0 / Il2cpp 0x150 / libc 0x468).
- HLE handlers need the HOST `%fs` (host libc TLS), so the emitted import stubs (`emit_swap_stub`,
  FSGSBASE) check a magic at `[fs+0x108]` marking OUR guest TCB; if present they `wrfsbase` the stashed
  host TCB (`[fs+0x100]`) for the handler call and restore the guest `%fs` after — and if ABSENT (host
  thread / pre-entry) they tail-call exactly as before, so it's inert off the guest path. `stub_size`→96.
- **Result (`PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS="-force-gfx-direct"`): the `eboot+0xa9c0bb` fault and the
  worker fault are GONE. The boot advances into NEW subsystems — new `libSceAgc` direct-mode calls and
  `libSceIme` (keyboard/IME input init) — past audio init, no crash.** So the two fixes stack:
  `-force-gfx-direct` (bypass the MT gfx-jobs deadlock) + guest `%fs` TLS (back initial-exec thread-locals)
  → the game boots dramatically further. Linux 45/45 + Windows 20/20 green with the gate OFF (zero
  regression; guest_tls.cpp is `#ifdef __linux__`). CONFIDENCE: HIGH.
- **Open (next):** it settles back into a `suspendPoint` wait (further along now) — re-map where. Worker
  threads aren't yet `guest_tls_activate_thread()`'d (their swap stubs tail-call safely; wire them when a
  worker needs initial-exec TLS). Signal handlers run with whatever `%fs` is live — revisit if a fault
  handler misbehaves under the gate. Consider making `-force-gfx-direct` + guest-fs the default once the
  downstream path is stable.

## 2026-07-06 — ⭐ RENDER-LOOP DEADLOCK BROKEN via `-force-gfx-direct` (single-threaded rendering)
**The multi-session render-loop deadlock is broken.** Producer-RE (gdb thread dump + `PROSPER_SYNCLOG`
with caller offsets, done interactively with the user) pinned the exact topology, then a one-switch
experiment cracked it.

**Ground truth of the deadlock (all 3 stuck threads sit in `k_wait_on_address`):**
- The 3 semaphores live in ONE object `rbx` (Unity's MT GfxDevice job-scheduler): `[rbx+0x00]`=GfxDevice-
  worker sem (addr `…e458`, site `0xb0672a`), `[rbx+0xc0]`=main sem (`…e518`, site `0x18a83b5`),
  `[rbx+0x130]`=PreloadManager job count, `[rbx+0x178/0x180]`=another sem, `[rbx+0x1c8]` a flag.
- Timeline: GfxDevice posts main's sem ONCE (`0xb06717`, the release path `lock add [sem]; WakeByAddress`),
  main wakes, consumes one job, **re-waits** on `e518`; GfxDevice then acquires `e458` and blocks FOREVER —
  **`e458` is never posted by anyone (0 wakes)**. The worker (`0x9385d7`) is the same shape (woken once by
  `0x95d58d`, re-waits). So the MT client/worker rendezvous wedges after exactly one handoff.
- main's wait (`0x18a82a0`→`0x18a83b5`) is a *timed* semaphore acquire (deadline from clock `0x401d94798`),
  but we pass nullptr for the timeout (PROSPER_WAIT_TIMEOUT off) so it's infinite — confirming again the
  timeout path is a give-up, not the fix.

**The unblock:** the eboot contains Unity's own switches `force-gfx-direct` / `force-gfx-st` /
`force-gfx-mt` / `gfx-enable-gfx-jobs` (boot.config has NO gfx-jobs line → Unity defaults to MT + jobs).
Added a gated guest-argv injector (`PROSPER_GUEST_ARGS`, `exec_image_linux.cpp run_entry`; a legitimate
compat-layer config, off by default → normal boot still argc=1). Running with
`PROSPER_GUEST_ARGS="-force-gfx-direct"`: the game parses it (`Argument Count = 2, Arg 1 =
-force-gfx-direct`), the **`suspendPoint` watchdog loop is GONE**, and the boot advances into NEW code —
GfxDevice init now runs INLINE on the main thread (new AGC calls `H7uZqCoNuWk`, `-KRzWekV120`, `Zw7uUVPulbw`)
and it reaches **audio init** (Ajm/AudioOut) — further than ever before. This validates the docs' long-
standing MT-rendering hypothesis: our env can't complete the cross-thread gfx-jobs handshake, so forcing
single-threaded rendering (the game's own supported mode) sidesteps it. CONFIDENCE: HIGH.

**New frontier (post-deadlock) = the `%fs` initial-exec TLS landmine, CONFIRMED.** Two fresh faults appear
(racing threads), both rooted in guest initial-exec (`%fs`-relative) TLS aliasing the HOST glibc TCB:
- **Main thread** faults at `eboot+0xa9c0bb`. Chain: `eboot+0x10fa24c` calls `0xa99d50` (a lazy
  thread-local get-or-create: `mov %fs:0x0,%rax; mov -0xa8(%rax),%r14; if r14 nonzero return it else
  allocate+store`), which returns `[TP-0xa8]`; the result feeds `0xa9c0a0` which does `mov 0x8(%rdi)`.
  **Ground truth (gdb, main thread):** `fs_base=0x7ffff7e99740` is the HOST pthread TCB (we run guest code
  on the host `%fs`, deliberately, so real libc.prx works), so the guest's initial-exec var at `[TP-0xa8]`
  reads **host glibc garbage `0x2`** instead of a zero-init guest slot. On HW `[TP-0xa8]` starts 0 → the
  lazy-alloc path runs → valid object; here it returns `2` → `0xa9c0a0` derefs `[0xa]` → SIGSEGV.
- **Worker-thread SIGSEGV** at a high mapped region (`rip` ASLR-varying, e.g. `0x…82e4772d`): same class —
  a thread-local callback/vtable read via `%fs` returns host garbage → call through a bad pointer.
Both are the recurring landmine the project beat case-by-case before (`k_tls_get_addr`→gettid, exc-handler
stack-local). `-force-gfx-direct` newly runs Unity's GfxDevice/telemetry code that uses **initial-exec**
guest TLS (direct `%fs`-relative, NOT `__tls_get_addr`), which our host-`%fs` model doesn't back.

**FIXED (gated `PROSPER_GUEST_FS`, `src/host/guest_tls.cpp`).** Give each guest thread its own guest TCB
with the modules' static TLS laid out below the thread pointer (Variant II) and run guest code with `%fs` =
guest TP. `guest_tls_activate_thread()` is called on the main thread (as the last step before entering the
guest, in `run_entry`) and on each worker (`thread_trampoline`). HLE import stubs (`exec_image_linux.cpp`)
became `%fs`-swap stubs: they check a magic at `[fs+0x108]` marking OUR guest TCB, and if present swap `%fs`
to the stashed host TCB (`[fs+0x100]`) for the handler call, then restore the guest `%fs` — using FSGSBASE
(`rd/wrfsbase`). If the magic is absent (host-context thread, or main before entry), they tail-call the
handler on the current `%fs` exactly as before, so the mechanism is inert/safe when a thread isn't guest-
activated. stub_size bumped 32→96 for the larger stub. **Result:** with `PROSPER_GUEST_FS=1
PROSPER_GUEST_ARGS="-force-gfx-direct"` the `eboot+0xa9c0bb` crash is GONE (exit 90→124) — `[TP-0xa8]` now
reads its zero-init guest slot, `0xa99d50` lazy-allocs correctly, and the game runs stably past gfx/audio
init. Gated OFF by default (Linux 45/45 + Windows 20/20 unchanged; `PROSPER_GUEST_FS=1` alone reaches the
identical baseline stall = mechanism sound). With both flags the boot now progresses THROUGH gfx init →
audio init → **input/IME init** (`sceImeKeyboardGetInfo`/`sceImeUpdate`, further than any prior state) and
settles into a slow `suspendPoint`/no-draws frame loop (still just the 1 setup submit, 0 draws). NEXT:
characterize why the direct-mode frame loop doesn't reach a draw submit (each fix so far has peeled another
init layer — gfx→audio→IME). CONFIDENCE: HIGH.

## 2026-07-06 — GPU-executor Stages A/B/C landed; correct EOP writes CONFIRMED live (still not the unblock)
Built the GPU executor per `docs/GPU_EXECUTOR_DESIGN.md` (all committed, Linux 45/45 + Windows 20/20):
- **Stage A** — `execute_gpustate()` (Vulkan-agnostic core, `gpu_execute.hpp`) + the live-submit registry
  `set_submit_renderer`/`execute_and_present` (`gpu_executor.cpp`); `agc_driver_submit_dcb` calls it after
  folding, gated on `have_submit_renderer() && draws>0`. Inert until a device is registered. `test_gpu_execute`.
- **Stage B** — RESOLVES the "blocker for doing it correctly" below (capturing the fence value beyond a0–a5).
  Pinned the AGC ABI to Kyty `GraphicsCbReleaseMem(buf, action, gcr_cntl, dst, cache_policy, address,
  data_sel, data, …)`: a5=label address, stack arg 7=`data_sel`, stack arg 8=the **full 64-bit value**.
  `honor_eop_write` now writes the correct value honoring `data_sel` (1=32b, 2=64b, 3=GPU clock), on by
  default (`PROSPER_NO_EOP_WRITE=1` disables). `honor_write_data` copies WRITE_DATA's real dwords. `test_eop_write`.
- **Stage C** — `sceGnmAddEqEvent` (`b0xyllnVY-I`) EOP-event registration + `prosper_eq_trigger_eop()` on
  submit completion (shadPS4-pinned: ident=id, filter=GraphicsCore=-14). `test_equeue_events` extended.

**Live boot confirmation (boot_trace + `PROSPER_GFXLOG`):** the correct EOP writes now fire on the real
setup submit — e.g. `ReleaseMem addr=…4aa8 data_sel=2 data=0x1` is immediately followed by
`WaitRegMem a5=…4aa8` on the SAME address, and our `EOP write [..4aa8] data_sel=2 value=0x1` satisfies it.
So the earlier fence experiments' "guessed value" caveat is gone — we now write the **correct** values by
default. **Result: still the identical suspendPoint stall** (1 setup submit, 0 draws, 36 shaders created,
no advance). This is the decisive confirmation of the earlier finding: **no stalled CPU thread is gated on
the EOP fence label** — the correct value doesn't change that. The remaining gap is unchanged and precise:
the 3-thread sync_on_address producer (main `0x18a83b5` / GfxDevice `0xb0672a` / worker `0x9385d7`) that is
gated on GPU-resource-**upload** completion via a mechanism still not identified (all obvious ones ruled
out above). Since the game never submits draws pre-upload, Stage A's `execute_and_present` never fires on
the game yet — it is chicken/egg: identifying + posting the upload-completion the producer polls is the
one remaining boot-wall-caliber piece (needs interactive HW-watchpoint RE on the producer, per §"gdb
snapshot"). Everything else in the executor is built and verified.

So the game plants an EOP fence: RELEASE_MEM writes a completion value to label `A` when the GPU pipe
drains; WAIT_REG_MEM (and, cross-thread, the PreloadManager/FTM producers) block until `[A]` satisfies the
compare. **We never write `A`** — our AGC Dcb builders (`agc_cb_release_mem`, `agc_dcb_wait_reg_mem`,
`agc_dcb_write_data`) zero their payloads and the CommandProcessor's `apply()` treats events/fences as
no-ops (`command_processor.cpp:27`). Hence the eternal stall.

**The concrete unblock (next milestone):** on submit, honor the fence — since our CommandProcessor folds
each Dcb synchronously (the "GPU" is done the instant SubmitDcb returns), perform the RELEASE_MEM label
write to `[dstGpuAddr]` (1:1-mapped host write) and satisfy the matching WAIT_REG_MEM. That is correct EOP
semantics, not a hack. **Blocker for doing it correctly:** the fence *value* RELEASE_MEM writes is a 7th
argument passed on the guest stack (beyond a0–a5), which the current HLE ABI shim doesn't capture, and the
`cmpFunc`/`ref` encoding of WAIT_REG_MEM must be honored exactly (writing a guessed value would be a fake —
correctness-first). Doing this right needs either (a) capturing the stack args in the HLE trampoline, or
(b) the Kyty Gen5 RELEASE_MEM/WAIT_REG_MEM field reference. This is the precise, bounded task that should
crack the render loop — best done with the user/reference in the loop (boot-wall-caliber), but it is now a
*specific* mechanism, not a vague "GPU-execution build."

## 2026-07-06 (cont.) — ⭐ LIVE cursor-trace of the deser fault (ring-buffer tool)

Added `PROSPER_HWBP_ANOM=<hex>` to `exec_image_linux.cpp`: a 256-entry ring of every gated reader hit
`{rip, rax(cursor), [cursor]}` that dumps the ring the instant a read yields a value ≥ threshold (the bp
stays armed independent of `PROSPER_HWBP_MAX`, so set `MAX=0` to suppress the per-hit log). Run:
`PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_HWBP=0x7e40d9 PROSPER_HWBP_ANOM=0x400 PROSPER_HWBP_MAX=0`.

**Trace results (deterministic):**
- alignedString reader `eboot+0x7e40d9`: delta between reads = `4 + align4(len)` exactly, so `rax` IS the
  per-read source cursor. Reads `[…196–248]` are a valid name/keyword/property table (lengths 4–0x1a).
- Then a **+12564-byte hop** into a far region, where the reader consumes `0xb0` (176) and `0x400` (1024)
  **as alignedString lengths** — i.e. it applies the `{count}{alignedString[]}` model to NUMERIC cbuffer
  reflection (`usedSize` values). `0xb0` = UnityPerDraw usedSize (Fable). Reading `0x400` as a string count →
  loops reading shader bytecode as strings → huge length → `alloc(~16 EiB)` → null write → crash.
- Deser driver `eboot+0x1612c70` fires 6×; hits #4/#5/#6 share buffer base `…601920`: #5 cur=base+0x4840,
  #6 cur=base+**0x7a20** (Δ=0x31e0=12768), end=base+0xb304. Hit #6 begins the crash record with the cursor
  at 0x7a20 and immediately reads count=`0x400`. Same caller for all 6: `caller_rbp=eboot+0xd4cf72`.

**Pinned mechanism:** the parser at/above `eboot+0xd4cf72` calls the alignedString-array driver
(`0x1612c70`→`0x7fc9f0`→`0x7e40d9`) on a region that in Unity 2022.3 holds numeric `ConstantBuffer`
records — a **field-dispatch/version desync** (2021.2-shaped read model over 2022.3 data), matching Fable.
prosper has NO shader parser, so this is 100% game-side code; the root is EITHER a dispatch value our env
feeds the parser wrong OR data our env produced/left-stale (note the ~0xd4-byte zero region near the crash).
**Disassembly of the parser (`eboot+0xd4ce00..0xd4d000`, code seg file-off = 0xd830+vaddr):** it is
Unity's GENERIC typetree-driven `Transfer` machinery (vtable dispatch, not shader-specific):
- `d4cec0 add 0x8(%r8),%r14` / `d4cec4 mov 0x10(%r8),%ebx` — cursor positioned from **typetree-node fields**
  `[r8+8]` (byteOffset) and `[r8+0x10]` (byteSize).
- `d4ced2 call *0x38(%rax)` → element size; `d4cef7 div %rcx` → **count = byteSize / elementSize**.
- `d4cf6c call *0x88(%rax)` → the array driver `0x1612c70` (return `0xd4cf72` ✓), which reads the array.

So counts + cursor come from TYPETREE METADATA (`[r8+8]`/`[r8+0x10]`/the element-size vtable), not a
hardcoded layout. A typetree whose node offsets/sizes are 2021.2-shaped over 2022.3 data yields wrong
offsets → the cursor lands on numeric `ConstantBuffer` bytes → read as strings. Confirms Fable's wrong-model
at the instruction level.

**Node-ring probe (`PROSPER_HWBP_NODE=1`, bp at `0xd4cec0`, dump on worker fault):** typetree nodes are
SANE — `r8` walks a contiguous 24-byte node array (`…3f98`+0x18…), `[r8+8]` byteOffsets grow monotonically
(`0x54a88`→`0x55648`), `[r8+0x10]` byteSizes are sane (0x44–0x90). ⇒ evidence AGAINST worker guest-`%fs`
TLS corrupting the typetree context. CAVEAT: `0xd4cec0` is a very hot generic function; single-stepping it
perturbs the boot into an early UNRELATED fault (~40 hits, `mov esi,[rcx+8]`, non-null addr) before reaching
the deser crash — so this probe can't reach the crash node. The `0x7e40d9` anom-ring CAN reach the crash
(fewer hits). **No-guest-`%fs` control:** without guest-`%fs` the boot faults earlier INSIDE host libc
(`rip∈libc.so.6`, `addr=nil`) — the initial-exec TLS landmine — so the deser crash is only reachable WITH
guest-`%fs`; TLS can't be isolated by toggling it.

**Deduction:** the desync is SPECIFIC to CubeBlur's stereo variants (Fable), not systematic — a TLS/file-I/O
bug would break the thousands of shaders that parse fine. **NEXT (leads, ranked):** (1) reference-parse this
game's `Hidden/CubeBlur` with UnityPy/AssetRipper — if it parses clean, the shipped data is well-formed
2022.3 and the runtime desync is env-induced (dispatch/timing); if it also chokes, the file is unusual.
(2) gate the node capture to the crash buffer only (arm `0xd4cec0` after the 5th `0x1612c70` driver hit) to
read the crash node's offset/size + the element-size vtable result without perturbing the whole boot.
(3) understand why `-force-gfx-direct` routes this deser onto a worker and whether the worker path differs.

## 2026-07-06 (cont.) — ⭐ REFERENCE PARSE: the data is well-formed; the desync is ENV-INDUCED

Reference-parsed the game's assets with UnityPy 1.25.0 (installed via apt python3-pip):
- Game assets (`resources.assets`, `globalgamemanagers.assets`): all **58 shaders parse clean**; none is
  CubeBlur — it is a BUILT-IN shader.
- `Media/Resources/unity_builtin_extra` = a clean **2022.3.32f1** SerializedFile; **`Hidden/CubeBlur`
  (pathid 15104) parses perfectly** (2 subshaders, 4 props, 4 keywords). All 43 built-ins parse.
- (Aside: `unity default resources` is **2022.3.26f1** — a different sub-version — but CubeBlur is not in it.)

⇒ **The shipped CubeBlur data is well-formed 2022.3.32f1**; a reference parser reads it without error. So the
game's own (correct) runtime parser would too — **the deser desync is INDUCED BY OUR ENV**, not the data or a
version mismatch. Combined with: nodes are sane, thousands of shaders parse fine, and the crash is
worker-side. NOTE: our file reads (`hle_file.cpp`) are thin passthroughs over host `::pread`/`::read`, and
the "not suitable for apr reads" line is Unity's OWN benign log (see 2026-07-06 overnight entry), not a
fallback path of ours — so raw file I/O is likely faithful, weakening the file-corruption theory. Remaining
env-induced candidates, ranked: (a) **guest→host pointer translation `P()`** handing the parser a
wrong-but-valid buffer/cursor mid-parse; (b) **our allocator (malloc/mmap HLE) overlapping two regions** so a
concurrent worker corrupts the load buffer (cf. the ~0xd4-byte zero hole seen mid-buffer); (c) the blob
**decompression** path if our env mediates it; (d) a wrong return value from a Sony size/alloc query the
transfer uses. **NEXT (decisive):** gate the node capture to the crash buffer only — chain-arm the `0xd4cec0`
node bp on the 5th `0x1612c70` driver hit (reuse the existing `g_hwwatch_req` chaining) so only the crash
record is single-stepped — then read the crash node's `[r8+8]`/`[r8+0x10]` + the element-size vtable result
and compare to a known-good shader's. That pins whether the node offset/size or the element size is wrong,
which distinguishes (a)/(b) from a data/decompression issue.

## 2026-07-06 (cont.) — ⭐⭐ BUFFER DUMP: data byte-faithful; desync is DISPATCH-SIDE in m_CommonParameters

Added `PROSPER_HWBP_BUFDUMP=1`: at the driver bp (`0x1612c70`, rbx=reader) dumps the reader window
`[rbx+0x40..rbx+0x48]` per hit to `/tmp/prosper_buf_<n>.bin` (6 hits, no single-step perturbation). Diffed
hit #6 (37636 B, the crash record) against the UnityPy reference CubeBlur object (`obj.get_raw_data()`,
17168 B):
- Aligned on `Hidden/CubeBlur`, the 4 KB window before the name (which contains the crash cursor at buffer
  offset `0x7a20`) is **BYTE-IDENTICAL** to the reference. At the crash cursor both hold
  `00 04 00 00 | 00·0 | 00·0 | b0 00 00 00 | 01 01 01 | 03 | 10 01 00 00 | … | 00 04 00 00` — well-formed data.
- Mapping that offset through UnityPy's parsed typetree: it is
  `m_ParsedForm.m_SubShaders[0].m_Passes[0].progVertex.m_CommonParameters.m_ConstantBuffers` — the `0xb0`
  is `m_ConstantBuffers[0].m_Size` (176 = UnityPerDraw). UnityPy parses this region as constant buffers
  without error.

⇒ **The load is byte-perfect; the deser desync is 100% in the runtime parser's control flow under our env**,
and it is localized to the **`m_CommonParameters`** (shared/deduped program-parameter) path — a 2022-era
feature exercised by CubeBlur's stereo-variant fan-out. The parser reads a cbuffer numeric field (`0x400`)
as a `{count}` for an alignedString array, i.e. it is in the WRONG parse STATE on CORRECT data.

**Leading hypothesis (prosper-side, fixable):** a **vtable / function-pointer relocation** in the engine's
serialization machinery is wrong under our loader, so a virtual call during the typetree Transfer
(`call *0x38(%rax)`=element size, `call *0x88(%rax)`=array read) dispatches to the WRONG method → wrong
element size/count → numeric read as strings. This would be specific to whichever transfer/type path
`m_CommonParameters` uses, explaining why only this desyncs. **NEXT:** capture the RESOLVED targets of
`call *0x38`/`call *0x88` (and the transfer object's vtable ptr) at the crash and check them against the
engine's real method addresses; if a vtable slot is mis-relocated, fix the loader's relocation of that
region. Cross-check prosper's RELA/relative-reloc handling for the engine's `.data.rel.ro`/vtable sections.

## 2026-07-06 (cont.) — ⭐⭐⭐ EXACT MECHANISM: MatrixParameter read 12 bytes not 16 (SInt8 tail dropped)

Ground-truthed via UnityPy (forced Python path, hooked `read_value` to map ref offset -> field):
the crash cursor (buffer 0x7a20 = CubeBlur-object ref offset 5064) sits at a `MatrixParameter`'s
`m_Type`(SInt8)/`m_RowCount`(SInt8) fields:

```
off 5052  MatrixParameter { m_NameIndex(int)=2, m_Index(int)=0, m_ArraySize(int)=0,
off 5064    m_Type   SInt8 = 0,
off 5065    m_RowCount SInt8 = 4,  (+2 pad) }   -> 16-byte struct, ends at 5068
off 5068  m_VectorParams (vector)  <- correct next read position
```

`{m_Type=0, m_RowCount=4, pad, pad}` little-endian == **0x00000400**. Our runtime reads a u32 COUNT at
5064 (via driver 0x1612c70) = 0x400 = 1024, reserves 1024, reads garbage strings -> ~16 EiB alloc -> crash.
So our runtime consumed the `MatrixParameter` as **12 bytes (3 ints only), skipping the trailing two
`SInt8` fields + pad**, drifting exactly 4 bytes so the next vector-count read lands on `m_Type/m_RowCount`.

CubeBlur's UnityPerDraw cbuffer (in `m_CommonParameters`) has 1 MatrixParameter -> 4-byte drift -> the
count read lands on 0x400. This is why only CubeBlur (deduped stereo params with a matrix) trips it and
`m_Type`/`m_RowCount`-free shaders (#4/#5) don't. Data is byte-perfect; the bug is that the generated
typetree / read for `MatrixParameter` (and `VectorParameter`, same {…, SInt8 m_Type, SInt8 m_Dim/RowCount}
layout) uses a 12-byte stride instead of 16 -- i.e. its byteSize/children omit the SInt8 tail (+align4).
**NEXT:** find why the generated typetree gives MatrixParameter byteSize=12 (dropped SInt8 children or a
missing kAlignBytesFlag / SInt8-node), which is the fixable root.

### Crash backtrace + read-path (from the SIGSEGV dump, deterministic)
```
eboot+0x7e4115   alignedString length read (crash: reads garbage len -> ~16 EiB)
eboot+0x7fcafb   {count}{alignedString[]} array reader (0x7fc9f0)
eboot+0x1612c92  the array driver 0x1612c70 (reads u32 count from cursor, reserves it)
eboot+0xd4cf72   generic vector reader 0xd4ce00 (count = byteSize/elemSize; call *0x88 per elem)
eboot+0xd3d34b   struct reader (reads SerializedProgramParameters / ConstantBuffer; calls the vector reader)
eboot+0xd3db38 / 0xd3d8b1   outer struct readers
eboot+0xb500a6 / 0xb4fcfd / 0xaf3e43 / 0xae47e1 / 0xae4836   shader/subprogram load
eboot+0x147b483 / 0x1485851   asset load entry
```
- `eboot+0x15fe650` = MatrixParameter's `GenerateTypeTree` transfer (registers m_NameIndex/m_Index/m_ArraySize
  =4B, m_Type/m_RowCount=1B, + align via 0xd572e0). **It is NEVER called at runtime (0 hits)** — it's the
  editor-only GenerateTypeTree instantiation. So the release read is NOT via this; it's the generic
  node/vtable-driven read above (`0xd4ce00` computes `count = byteSize / elemSize` and dispatches the element
  read via `*0x88`; element size via `*0x38`).
- **So the fixable cause is: the element size / node byteSize used for the `MatrixParameter` (and
  `VectorParameter`) array is 12, not 16** — the generic reader strides 12 bytes/element and the next
  vector-count read lands 4 bytes early on `{m_Type,m_RowCount,pad}`=0x400. NEXT: capture the elemSize
  (`*0x38` result) for the `m_MatrixParams` vector read, and/or find where MatrixParameter's serialized size
  is computed as 12 (a compiled size fn or a typetree node byteSize). Undocumented Unity runtime typetree
  layout — needs disassembly of the `*0x38`/`*0x88` element-size/read methods for the param types.
