# The First Berserker: Khazan — `PPSA20447` status

**Rung 0.** The title boots, links every module, mounts and enumerates its save slots and reaches the
AGC path, but **no real frame is ever composited** — the only picture prosper produces is a flat
white 3840x2160 clear, which is not content. It has not reached rung 1.

First brought up 2026-08-22 from nothing: no tracker, no `COMPATIBILITY.md` row, no route, no prior
work of any kind. Everything below is from that session unless a link says otherwise.

## What it is

Unreal Engine 4, project directory `bbq/`, IoStore packaging (`global.utoc`/`global.ucas` plus
`pakchunk0-ps5` / `pakchunk0optional-ps5` / `pakchunk1-ps5` in all three container forms). Movies are
**Bink 2** (`.bk2`), not H.264 — so this title does not touch AvPlayer or videodec2 at all. Audio is
Wwise: 29 `ak*.prx` plug-ins plus `masteringsuite.prx` and `sceaudio3d.prx`, driven through
`libSceNgs2` and `libSceAjm`.

`self_dump` reports 53 needed modules and 56 imported libraries. Two things worth knowing before
starting work here:

- **AGC SDK version 13.** `[agc] register defaults requested for SDK version 13`, so this title is
  *not* in the pre-13 post-submit-visibility family (`ARCRUNNER_STATUS.md` §5, #2219/#2220) and
  `PROSPER_POST_SUBMIT_VISIBILITY` is irrelevant here.
- **It imports `libScePsml`, which is absent from the PS5 3.20 reference set.** The dump requires
  system software 12.70. Two of its NIDs are reached during boot — `3WVD91e12ZQ` and `+2KpvixvL6E` —
  and neither resolves against `../PS5-3.20_Libs/` or anywhere else available, so their names,
  signatures and error spaces are currently underivable. They are left unregistered on purpose.
- It does **not** import `libSceAudiodec`, so #2902 cannot affect it.

## Reproducing

```bash
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_SAVE0=<RUN>/save0 PROSPER_SAVEDATA_DIR=<RUN>/savedata \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA20447-app0 \
      --seconds 10 --count 24 --out <RUN>/shots --timeout 300
```

No pad route is needed or useful yet — nothing the title draws responds to input.

## Blocker 1 — `sceSaveDataGetEventResult` answered a drained queue with a code the guest never matches (FIXED)

Before this was fixed the title's **game thread parked forever** roughly four seconds into the boot.
The shape was unusually clean: `tools/screenshot` reported `1 distinct frame(s) in 33 publications`
over a **360 s** run with `guest=running`, and a `gdb` dump of the live process showed **all 79
threads blocked in a syscall** — the guest main thread inside `prosper::k_usleep`, every engine
thread (`RenderThread 0`, `RHIThread`, 42 × `TaskGraphThread`, 19 × `PoolThread`, `FAPREventQueueL`)
in a condition-variable or event-flag wait. Nothing was spinning and nothing was progressing.

With `PROSPER_SVCLOG=1` the only Sony call that repeated during the stall was
`sceSaveDataGetEventResult` — 5,020 times in 90 s, about 56 per second, with no other guest thread
making any call at all. Scanning the poll's own stack for guest return addresses gave a stable chain
(`eboot+0x796eb74` → `eboot+0x4036bc0` → `eboot+0x796bdaf` → `eboot+0x7969d82` → `eboot+0x1590380`),
and disassembling the first frame's function produced the loop verbatim:

```text
eboot+0x796eb30   vmovss xmm0,[rip+0x3aea72c]     ; the sleep interval
eboot+0x796eb38   call   0x1565790                ; FPlatformProcess::Sleep(float)
eboot+0x796eb41.. vmovups ...                     ; zero a 104-byte SceSaveDataEvent
eboot+0x796eb67   xor    edi,edi                  ; eventParam = NULL
eboot+0x796eb6f   call   0x8eaf3d0                ; sceSaveDataGetEventResult(NULL, &event)
eboot+0x796eb88   cmp    r12d,0x809f0008          ; <-- the ONLY value that ends the wait
eboot+0x796eb8f   jne    0x796eb30                ; anything else: sleep and poll again
```

prosper answered a drained queue with `0x809F0018`
(`hle_service.cpp`'s `SAVE_DATA_ERR_NO_EVENT`), which that `jne` never matches.

**`0x809F0018` is not a fiction — it is the better-established of the two codes, and it means
something else.** Four titles in the local dump set const-compare it against this exact call's result
and every one of them **sleeps and re-polls** on it:

| title | site | shape |
| --- | --- | --- |
| `PPSA15552` Dead Cells | `+0x173c9b0` | `cmp eax,0x809f0018; jne <exit>; mov edi,0x1f40; call sleep; jmp back` |
| `PPSA15552` Dead Cells | `+0x173cda0` | identical idiom, second site |
| `PPSA28061` Earthion | `+0x12f6d` | `cmp [rbp-0x7c],0x809f0018; jne; sleep(50 ms); jmp <repoll>` |
| `PPSA03831` Sonic Frontiers | `+0x18a2285` | `cmp eax,0x809f0018; jne; sleep(1); jmp <repoll>` |
| `PPSA05325` Sonic Origins | `+0x940385` | byte-identical (same SEGA save library) |

So `0x809F0018` is the **"the operation is still in flight, keep waiting"** code. prosper has nothing
in flight — every file operation here completes synchronously — so it was returning a
well-corroborated *"still busy"* in a state where nothing is busy. That is worse than an
unestablished value: it is a permanent lie, and a guaranteed infinite hang for any of those four
titles that reaches its wait loop.

`0x809F0008` is the value prosper already calls `SAVE_DATA_ERR_NOT_FOUND`, and PPSA20447's `cmp`
pins it as the drained-queue answer. Both codes are now spelled as separate literals in
`hle_service.cpp` — `SAVE_DATA_ERR_IN_FLIGHT` deliberately kept, unaliased, so a future asynchronous
`Mount3` has the right constant to hand.

**A second title corroborates the codes, and it corroborates the opposite of what this document first
claimed.** *Earthion* (`PPSA28061`, rung 2) was A/B'd for regressions and produced evidence instead.
Its wait loop at `eboot+0x12f30` const-compares **both** values, in consecutive instructions:

```text
12f65   call   sceSaveDataGetEventResult(NULL, &event)
12f6a   mov    [rbp-0x7c],eax
12f6d   cmp    [rbp-0x7c],0x809f0018   ; still in flight ->
12f74   jne    0x12f82
12f76   mov    edi,0xc350              ;   sleep 50 ms
12f7b   call   <sleep>
12f80   jmp    0x12f4d                 ;   and poll again
12f82   cmp    [rbp-0x7c],0x809f0008   ; NOT_FOUND ->
12f89   jne    0x12f8d
12f8b   jmp    0x12fb8                 ;   GIVE UP and return
...
12f99   lea    rdi,[rbp-0x70]; add rdi,0x20
12fa5   mov    edx,0x20
12faa   call   <memcmp>                ; event.dirName vs the dir it is waiting for
12fb4   jmp    0x12fb8                 ; match -> done
12fb6   jmp    0x12f4d                 ; not mine -> poll again
```

Over identical 120 s default launches the observable change is large:

| | `origin/master` | with the fix |
| --- | --- | --- |
| `sceSaveDataGetEventResult` | **2,400** calls, spanning the whole run (log line 45 to 63,046 of 63,064) | **5** |
| `sceSaveDataMount3` | **0** | 2 |
| `Prepare` / `GetMountInfo` / `Commit` / `Umount2` | **0 / 0 / 0 / 0** | 1 / 2 / 1 / 3 |
| `[shot] done:` counters | `source-distinct=24 pixel-distinct=16 max-pixel-stale=40.0s guest=running status=ok` | identical counters |

**But the mount/prepare/commit/unmount lifecycle appears because the title GAVE UP, not because
prosper delivered anything.** Earthion's success branch at `+0x12fb4` requires
`memcmp(event + 0x20, <dirName>, 0x20) == 0`, and the only event prosper ever queues comes from
`Umount2` and is zero-filled — no `dirName`. That branch is **structurally unreachable**, before and
after this change alike. What the fix does is let Earthion reach its *give-up* branch instead of
spinning on a false "still busy" forever, which is the truthful outcome when there genuinely is no
event. It is strictly better; it is **not** "its save subsystem now works", and this document
previously said so.

Two things fall out of that loop and both are recorded rather than left in a PR body:

- **It confirms the `SceSaveDataEvent` layout from a second title's bytes.** Earthion zeroes `0x68`
  (104) bytes and compares 32 bytes at `+0x20`, which pins `dirName[32] @ +0x20` in a 104-byte
  struct. `s_savedata_get_event`'s confidence on the tail field names is raised from MED to HIGH.
- **It names a gap prosper did not know it had** — see *Known gap* below.

## Known gap — no per-operation completion event carries a `dirName`

The only event `sceSaveDataGetEventResult` can ever return is the one `Umount2` queues, and it is
zero-filled: no `titleId`, no `dirName`. A title that waits for the completion of a *specific*
operation — as Earthion does, by `dirName` — can never be satisfied. Nothing is currently blocked on
this, because Earthion has a give-up branch and PPSA20447's loop reads only the return code, but any
asynchronous `Mount3`/`Prepare`/`Commit` implementation has to supply both halves:

1. return `SAVE_DATA_ERR_IN_FLIGHT` (`0x809F0018`) while the operation is actually running, and
2. queue a completion event carrying that operation's `dirName`, rather than bumping a counter.

Tracked on the tracker and on #1880.

Pinned by `tests/hle/service/test_savedata_event_drain.cpp`, which drives the real registered entry
points through the dispatcher: an empty queue must answer `0x809F0008`, the transcribed guest loop
must terminate on it, queued umount completions must still be delivered (type 1, initial user,
exactly 104 bytes written) *before* the drain answer, and a null event pointer must remain a distinct
parameter error.

**With the fix the boot advances a long way**: the title mounts `ProfileSettings`, runs
`Prepare`/`GetMountInfo`/`Commit`/`Umount2`, does a `DirNameSearch`, then walks all thirty save slots
(`SaveGame0Slot` … `SaveGame29Slot`, 113 `sceSaveDataMount3` calls in total). It then hits blocker 2.

## Blocker 2 — the guest calls its own OOM handler at ~6 s (OPEN; the direct-memory reading below is superseded)

At ~6 s the guest prints its own out-of-memory report and asserts:

```text
[memhle] ampr push-map va=0x206109eb40 len=0x40 -> FAILED      (x3,873 in one run)
[prosper] unimplemented: libScePsml::3WVD91e12ZQ  -> returning 0
[prosper] unimplemented: libScePsml::+2KpvixvL6E  -> returning 0

PS5 Out of Memory:
    AvailablePhysical: 14360772608
    UsedPhysical:      2504458240

[...]Allocator Stats for Binned3 are not in this build set BINNED3_ALLOCATOR_STATS 1 in MallocBinned3.cpp
[...]Fatal error!
[shot] guest thread ended: kind=2 detail=SIGSEGV at addr=(nil) rip=0x5c0004a20
[shot] guest backtrace: 0x411565b55 (eboot+0x1565b55)  0x4117abbae (eboot+0x17abbae)
```

The SIGSEGV is UE's crash handler running after the assert, not the first failure. The first failure
is visible under `PROSPER_MEMLOG=1`: **`alloc_main_dmem … -> ENOMEM (pool exhausted)`**.

What the guest does is a halving probe — it asks for the whole advertised budget and halves until a
request fits, then repeats — so it consumes essentially the entire pool by design:

```text
0xc000, 0x12c00000 (300 MiB), 0x200000000 (8 GiB), 0x100000000 (4 GiB), 0x80000000 (2 GiB),
0x40000000 (1 GiB), 0x20000000 (512 MiB), 0x8000000, 0x4000000, 0x1000000, 0x200000, 0x100000,
0x80000, 0x40000, 0x20000, 0x10000, …
```

Reading the physical addresses rather than only the sizes is what makes the picture legible, and it
is worth doing before forming any hypothesis here. Allocations 1-16 are **contiguous** and end at
`0x410000000`, which is `0x10000000 + 0x400000000` — they fill the 16 GiB pool exactly. The guest
then releases its 300 MiB scratch block (`release_dmem [0x10010000,0x22c10000)`), and the six later
allocations are served **out of that hole**: `0x1000c000`, `0x10218000`, `0x115a0000`, `0x13580000`,
`0x15be0000`, `0x15e00000`. When the hole runs out, `ENOMEM`.

**Read against the 2026-08-23 re-measurement below, this section describes the pool arithmetic
correctly and the *diagnosis* wrongly.** The allocation ledger above is accurate. What does not
follow from it — and what the rest of this document used to assert — is that the pool running out is
what makes the title assert. It is not. See *Blocker 2, re-measured*.

- **prosper's allocator is behaving correctly.** The reuse is legitimate, not overlapping — see
  `## Ruled out`.
- **The six later allocations are graphics allocations, not engine-heap ones.** Their `map_dmem`
  targets are `0x9fc0000000` and `0xa000000000`, which the guest's own boot banner names
  `Frame Buffer va range 9fc0000000 - a000000000`.

The guest's own report — **13.4 GiB available, 2.3 GiB used** — is its *own* accounting of the heap
it reserved, and is not in contradiction with prosper's pool being empty. That earlier reading was
wrong and is recorded in `## Ruled out` so it is not re-derived.

## Blocker 2, re-measured — the pool was never the wall (2026-08-23)

One real defect was found underneath this section and fixed (#2908); it is **not** what holds the
title, and saying so plainly is the point of this rewrite.

**The defect that was real: a refused Ampr map retired direct memory it never used.**
`sceAmprCommandBufferSetBuffer`'s map flavor claimed a physical range from the pool and then asked
the host to place it at the guest's VA. When that placement was refused the pool offset was simply
dropped — nothing referenced it, nothing could release it, and because the claim is made at 64 KiB
alignment every carcass retired a full 64 KiB stride however small the request. Khazan produces
**4,646 refusals in a 6 s boot** (4,041 fully page-aligned, 591 unaligned, 14 with an unaligned
length), so **290 MiB** of pool
went with them, against the 300 MiB scratch block that is the only headroom left after the halving
probe. *Sifu* (`PPSA03001`) produces **31,716** of them, i.e. **1.94 GiB**. Fixed in
`src/hle/memory/hle_kernel_mem.cpp`, pinned by `tests/hle/memory/test_ampr_map_refusal_releases_dmem.cpp`.

**The fix works and does not rescue the title.** With it, the post-release allocations stop being
scattered across the leak's fragments and become contiguous — `0x1000c000`, `0x10218000`,
`0x10220000`, `0x12200000`, `0x141e0000` against the leaking run's `…`, `0x11a70000`, `0x13a50000`,
`0x16340000` — and the guest gets **one more allocation than it used to** (`0x200000` at
`0x14400000`, which never happened before). It then prints the same `PS5 Out of Memory` at the same
point.

**And that is the finding: when the guest asserts, prosper's pool is not exhausted and no prosper
call has failed.** Measured on the fixed build, `PROSPER_MEMLOG=1`, `tools/screenshot`, default
route, Linux/RADV:

| | Khazan `PPSA20447` | Sifu `PPSA03001` |
| --- | --- | --- |
| largest free direct-memory block at the assert | **230.1 MiB** | **234.2 MiB** |
| `alloc_main_dmem` failures after the probe settles | **0** | **0** |
| any other failing prosper call in the whole run | **none** | **none** |
| `sceKernelBatchMap` page commits, all successful | 15,509 | 30,000+ |

Every `ENOMEM (pool exhausted)` in either run — 19 of them on Khazan — belongs to the halving probe
itself and is by design: the guest asks for the whole advertised budget and halves until a request
fits. *Unbound: Worlds Apart* takes 19 of the same answers and renders its title screen. **So the
issue body's "the first failure is `alloc_main_dmem … -> ENOMEM (pool exhausted)`" names a
by-design event, not the defect.**

**What the guest is actually doing is failing inside its own allocator with memory available on both
sides of the boundary.** Sifu says so explicitly — `Ran out of memory allocating 16385 bytes with
alignment 0`, `LowLevelFatalError [Line: 197]` — while its own report says 13.6 GB free and prosper
has 234 MiB free and is refusing nothing. Khazan reaches the same
`FGenericPlatformMemory::OnOutOfMemory` path (the `Allocator Stats for Binned3 are not in this
build` line is `FMallocBinned3::DumpAllocatorStats`) without flushing the size line before the crash
handler takes over. The next step is the guest's own disassembly at that call site, not another
memory-layer experiment: every memory-layer question this document could ask has now been answered
in the negative. Tracked separately as the reframed blocker.

## Ruled out

Each row is an executed experiment, not an opinion.

| Hypothesis | What killed it |
| --- | --- |
| The boot stalls on the startup **movie** (`bbq/content/movies/loadingmovie/*.bk2`), the #1949 video-splash family | `PROSPER_FILELOG=1` over a full run: **not one movie file is ever opened**, and there is zero AvPlayer or videodec traffic. The stall precedes any movie. |
| The stall is `LogDataTable … Fatal` — a UE fatal assert | It is not a death. The line lands at log line 50,965 of 402,849 and the guest issues **6,980 further APR reads** after it, ending the run with `guest=running`. |
| The stall is a **case-sensitivity** failure on `Project file not found: ../../../BBQ/BBQ.uproject` | That message is UE's own, normal for a packaged build with no `.uproject`; prosper resolves guest paths case-insensitively (#1006/#1226/#1233) and the very next guest line opens the lowercase path successfully. |
| The stall is `sceSaveDataMount3` returning **NOT_FOUND** for a save that does not exist yet | Pre-creating the save directory made `Mount3` answer `OK (created=0)`; the title then mounted, unmounted twice and searched — **and polled forever exactly as before**. The synchronous return value is not what the wait reads. |
| The wait needs the **event's contents** — a matching `titleId`/`dirName` in the 104-byte `SceSaveDataEvent` | Filling both from the live mount changed nothing (4,432 polls, still one distinct frame). |
| The wait needs a **queued event of the right type** — i.e. `Mount3` should post a completion | A probe serving a synthetic event on *every* poll, cycling `type` 0…7 with ~400 polls each over a 140 s run, never released the wait: `1 distinct frame(s)`, 7,985 polls. **This title's** loop reads the return code, not the event. (It does not generalise: *Earthion*'s loop at `eboot+0x12f99` **does** read the event, comparing `dirName` at `+0x20` — see *Known gap*.) |
| **No title distinguishes "no event" from "not found"**, so collapsing them is safe | False, and falsified by this change's own corroborating title. *Earthion* const-compares `0x809F0018` then `0x809F0008` in consecutive instructions at `eboot+0x12f6d`/`+0x12f82`, with different handling for each. The change is safe for a different and stronger reason — see the census in the tracker and #2910. |
| Blocker 2 is the **advertised direct-memory budget** (`PROSPER_DMEM_BUDGET_MB`, the #1213 suspicion) | `12288` and `8192` both reach the identical `PS5 Out of Memory` at **6.2 s**, matching the 16 GiB default exactly. The guest sizes its probe from whatever is advertised and consumes all of it either way. |
| The allocator hands out **overlapping physical ranges** once the pool is full — allocations 17-22 start at `0x1000c000`, back at the bottom, inside the 300 MiB block allocation 2 still appeared to hold | `release_dmem [0x10010000,0x22c10000)` is logged at run-log line 49, *before* the first of those at line 52, and its bounds are exactly allocation 2's. The reuse is correct. Recorded because the size list alone reads as a serious corruption bug and the disproof is one line that is easy not to look for — read the physical addresses **and** the releases, never the sizes alone. |
| The guest's `13.4 GiB available / 2.3 GiB used` report **contradicts** prosper's exhausted pool, so prosper is losing memory it handed out | It is not a contradiction. Those are UE's figures for the heap it reserved from prosper, sub-allocated 2.3 GiB of, and still considers 13.4 GiB free inside. Both statements are true at once. |
| This title is in the pre-SDK-13 post-submit-visibility family (#2219) | It reports **SDK version 13**, so that contract is already armed for it. |
| Blocker 2 is prosper's **direct-memory pool running out** | Falsified 2026-08-23 on the #2908 build. At the moment the guest asserts, the largest free block in the pool is **230.1 MiB** (Khazan) / **234.2 MiB** (Sifu), there are **zero** post-probe `alloc_main_dmem` failures, and **no prosper call of any kind fails in the entire run**. The 19 `ENOMEM (pool exhausted)` answers are the halving probe's own, by design, and *Unbound: Worlds Apart* takes the same 19 and renders its title screen. #2908. |
| The **`ampr push-map … -> FAILED`** lines are lost page commits — the guest's allocator cannot commit, so it declares OOM | Falsified 2026-08-23 by case analysis over the refusal's own code path; no probe is involved and none is needed. `map_phys_at(fixed=true)` returns null only when `range_is_free_reservation` declined **and** `prosper_mmap_noreplace` then failed (`src/host/platform/posix_shim.hpp:318` — `MAP_FIXED_NOREPLACE`, so `EEXIST` means a host VMA already covers part of the range and `EINVAL` means the address or length is not page-aligned, which no page commit ever is); `map_at` adds two further nulls that cannot arise here — an unavailable memfd, and a failed `MAP_FIXED` over one of prosper's own free reservations. So for every page of a refused range, `prosper_reserved_range_state` (`src/hle/memory/hle_kernel_mem.cpp:2921`, and note it answers about **one address**, not a range) returns one of **0 / 1 / 2 / 4** on POSIX — the full contract is 0/1/2/3/4 with **3** Windows-only and **4** POSIX-only, enumerated at `:2906-2910`; the earlier "exactly three states" here was wrong and the comment two lines above the line it cited says so. **None of the four loses the guest memory.** **2 = committed**: the guest already holds it. **1 = reserved, uncommitted**: the lazy-commit fault arm backs it on first touch (`src/host/image/exec_image_linux.cpp:2150`, gated on exactly `== 1` and on `addr >= 0x1000000000`), and the arm is live in this address region rather than merely gated for it: a Sifu census run logs `[lazy-commit] #1 mapped page=0x20e1520000` — 131.5 GiB. That page is **not** one of the refused VAs, so it evidences the mechanism working up here, not the disposal of any particular refusal. **4 = reserved but declining lazy commit** disposes exactly as 0 does, and cannot arise here anyway: the AMM window is searched upward from `kAmmWindowSearch` = 1 TiB (`:2537`), far above these VAs. **0 = untracked**: nothing rescues it — the unified-memory fallback spans only `GPU_VA_LO`..`GPU_VA_HI` = 4-64 GiB (`exec_image_linux.cpp:1115-1116`) while these VAs sit at **129.5-156.5 GiB** — so a genuinely lost commit would be a **fatal** SIGSEGV at that address. **Neither title takes one, and the fault each does take is the discriminator — by its ADDRESS, which is the part that carries it.** A lost commit at one of these VAs would fault *at* that VA, somewhere in 129.5-156.5 GiB. Both titles instead die at `SIGSEGV addr=(nil)`, on UE's own `int $0x45 ; nop ; ud2` trap, and a grep of both census runs finds no fault at any address in the Ampr range at all. (Both rescue arms also being floored far above zero is consistent with that but is not what settles it.) One thing the census below does **not** license, and an earlier version of this row claimed it did: it cannot say which of the four states the refused pages are in. `mincore` reports whether a **VMA exists**; `prosper_reserved_range_state` reports whether **prosper tracks the range**, and states 1 and 2 both require tracking (`hle_kernel_mem.cpp:2928`, `:2931` return 0 for anything absent from `g_maps`). Untracked-but-mapped is *part* of the population that produces an `EEXIST` refusal — the rest is tracked-and-committed, which `range_is_free_reservation` also declines — and a VMA census cannot separate them, so "the argument runs through states 1 and 2" was an inference from the wrong instrument — the same instrument-measures-X-claim-is-about-Y error this row already records once. The case analysis is complete over all four states, which is why it does not need to know. Refusing is protective: it is the #88 / #107 clobber the flavor discriminator exists to prevent. **Correction, and it matters more than the conclusion:** this row first rested on a measured "100% of refusals target memory the guest ALREADY HAS MAPPED", from a `mincore` probe of **one page** against ranges that are mostly `0x4000` — four host pages — while `MAP_FIXED_NOREPLACE` fails if *any* page is mapped. A quarter of each range was examined and the result asserted for all of it, in the direction the author wanted. Its replacement then claimed a structural argument that skipped the `prosper_mmap_noreplace` step and was not exhaustive over the null paths. Both were caught in review of #2947; the case analysis above is the third attempt and the first that closes. **The widened probe corroborates it, and this time the instrument measured what the claim says.** Every page of every refused range, whole-range `mincore`: Khazan **4,646 / 4,646**, Sifu **31,716 / 31,716** — 36,362 refusals, none containing an unmapped page. Splitting the alignment report also showed something the old line could not: 14 Khazan and 5 Sifu refusals have a page-aligned VA with an UNALIGNED length, a shape the single "page-aligned" flag hid entirely. #2908 / #2947. |
| Fixing the Ampr pool leak gets the title past the assert | It does not, and this is the difference between *a* cause and *the* cause. The fix demonstrably works — allocations go from fragmented to contiguous and the guest gets one more allocation than before — and the assert lands at the same point. #2908, and #2747 is the standing reminder. |
| The `LogDataTable … Fatal` line just before the memory report is the assert | It is UE's log-category **verbosity listing** (`%-40s %-12s`, empty message body), and it lands ~27,000 log lines before the crash in both runs. Restated here because it looks exactly like a fatal assert when grepped for. |
| `libScePsml::3WVD91e12ZQ` / `+2KpvixvL6E`, reached immediately before the OOM, are implicated | Not shared with the other carrier: **Sifu calls `libScePsml` zero times** and asserts identically. Their position in Khazan's log is proximity, not evidence. |

## Evidence

Captures, logs and the flattened eboot image live outside the repo (they contain dump-derived bytes).
No screenshot is checked in: a flat clear is not progression evidence.
