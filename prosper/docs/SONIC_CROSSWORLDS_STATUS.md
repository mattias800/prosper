# Sonic Racing: CrossWorlds (`PPSA08804`) — status

Tracker: [#1895](https://github.com/mattias800/prosper/issues/1895).
Engine: **Unreal Engine 5** (IoStore/APR), with **CRIWARE** (CRI Mana / Sofdec2 movie, CRI Audio,
CRI FS) and the **EOS SDK** (`prx/eossdk-ps5-shipping.prx`) alongside it. 64 guest threads at the
frame loop.

## Current rung: 1 — the SEGA logo renders

**Requires the #2012 fix** (PR #2015, a shared libkernel change reviewed separately). On master
without it the title still produces no frame at all — see *What unblocked it* below, which is the
measured before/after.

![Sonic Racing: CrossWorlds — SEGA logo](../../assets/screenshots/sonic-crossworlds-sega-logo.png)

Direct, unmodified `tools/screenshot` capture, Linux/RADV (`AMD Radeon 8060S (RADV STRIX_HALO)`),
native 3840x2160, default switches, no route:

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA08804-app0 --seconds 20 --count 20 --out ~/shots
```

The logo appears at t≈60 s and holds to t≈80 s (1,373 distinct colours, 430,916 non-black pixels,
decoded from the PNG rather than scored). A second arm with no diagnostics at all reproduced it at
t=45 s with the **identical** `pixel_crc32` (`0d70a70a`). From t≈100 s the composite becomes a
**uniform `RGBA(1,0,1,255)`** while the engine keeps producing frames — see *The frontier* below.

## What unblocked it: #2012

Before [#2012](https://github.com/mattias800/prosper/issues/2012) the title produced **no frame at
all**: 20 samples over 400 s at `present_count=1`, `frame_seq=0`, `rendered_samples=0`, one black raw
scanout. The guest's main thread was parked forever in `prosper::k_rwlock_wrlock` while the other 63
threads waited behind it — nothing was spinning, so it did not look like a hang from the outside.

The mechanism, measured rather than inferred:

1. `libScePosix`'s `pthread_rwlock_trywrlock` (`XhWHn6P5R7U`) was **not registered**, so the
   dispatcher answered its default `0` — "the write lock is yours" — for a lock nobody took.
2. The guest ran the critical section and unlocked a lock it never held.
3. `k_rwlock_unlock` forwarded that unmatched unlock to glibc, whose behaviour there is *undefined*:
   it took the reader-release path and drove `__readers` **negative**. Read out of the live process,
   the deadlocked lock held `__readers = 0xfffffff2` — `WRLOCKED` set and a reader count of exactly
   **-2**.
4. A negative reader count never drains, so every later `wrlock` on that lock blocks forever.

FreeBSD, the platform the guest was built against, cannot reach that state. Its rule is **not** "the
caller must hold it": `_pthread_rwlock_unlock` checks the *write* owner in userland and returns
`EPERM` for a non-owner write release, and everything else defers to `_umtx_op(UMTX_OP_RW_UNLOCK)`,
which returns `EPERM` only when there is no write owner **and** the reader count is zero. With
readers present it decrements and succeeds, with no thread-identity check at all — so releasing a
read hold from a different thread is legal there, and prosper must not refuse it (`sceFiberRun`
resumes a fiber on whichever host thread calls it, so a fiber that read-locks, yields and resumes
elsewhere releases from another thread).

The fix therefore registers the missing POSIX spellings and keeps the outstanding-hold accounting
**with the lock**, reproducing FreeBSD's rule directly. `PROSPER_RWLOCKLOG=1` reports refusals and
`=2` every rwlock operation (rate-limited per report kind);
`PROSPER_RWLOCK_UNSAFE_UNLOCK` — set to any value — restores the old forwarding, which is the A/B
lever that established all of this and must stay off in normal runs.

**The chain is closed by measurement, not by argument.** With only the unlock guard (step 3) in
place, the title reports **14** unmatched unlocks per boot, from one guest thread across 12 distinct
rwlock slots — one of which, `0x23c00d2538`, appears in the deadlocked thread's own backtrace. With
the missing registrations added as well (step 1), the same instrument reports **0**, and
`XhWHn6P5R7U` disappears from the unimplemented-NID list. So the fake "you acquired it" was the
*source* of the unmatched unlocks, and the guard is the backstop for any other source.

## The frontier: the composite after the logo

From t≈100 s to the end of a 400 s arm the presented frame is a single colour, `RGBA(1,0,1,255)`,
while `frame_seq` climbs 990 → 3,252 and `present_count` 305 → 1,059. **This is not a hang, and
`frame_seq` climbing is not evidence that it is alive** (instrument trap 90): the image is
byte-identical for 300 s. The engine itself is healthy — the AGC submission threads, `RHIThread`,
`RenderThread 0` and `CriManaDecodeTh` all accumulate CPU time throughout.

Two measured leads, both recorded on [#2013](https://github.com/mattias800/prosper/issues/2013):

- **Draws are dropped because their shaders do not recompile.** `[exec-recompile-reject]` names a
  graphics pair (`es=…`/`ps=…`) whose draws are skipped, and `PROSPER_DBG=1` gives the rejecting
  opcodes: VOP3 `0x307` (15 sites), VOP3 `0x156` (4), VOP1 `0x54` (3), SOPP `0x6` (`s_cbranch_*`),
  and two MIMG ops (`0x20`, `0x27`) that reject as `[mimg-unresolved]`.
- **Nine distinct compute programs are skipped** (`[compute] skip unsupported program`), plus one
  3D-tile volume dispatch skipped under #590.

## Unimplemented-call census (default launch, current master + #2012)

**21** NIDs reach the dispatcher's `-> returning 0` path, resolved against the PS5 3.20 firmware
export tables. `pthread_rwlock_trywrlock` was the twenty-first and is now implemented; the rest are
listed on the tracker. Nothing in the list is an entitlement or add-content query — the title's
`scePlayGo`/`sceAppContent` traffic is answered from local inventory, and the two unimplemented
`libScePlayGo` NIDs (`scePlayGoGetSupportedOptionalChunk`, `scePlayGoGetInstallChunkId`) are the
same two the 2026-08-05 baseline recorded.

## Ruled out

One line per falsified hypothesis, with the evidence that killed it.

- **APR same-size container collisions are not the startup blocker.** 274/274 reads resolved
  `OK method=id`, 0 size-fallbacks, 0 refusals — the 49 registration warnings are noise here.
  (Tracker #1895, 2026-08-04 CPU-only baseline; re-confirmed on `c3614f51`.)
- **The `IoService` AMPR spin is gone and was not the whole story.** #1965 fixed it; on `c3614f51`
  `IoService` accumulates 7 jiffies over a whole boot, and the title still produced no frame until
  #2012. (This lane, 2026-08-05.)
- **The stall was not a spin at all.** All 64 threads sampled over an 8 s window: the busiest is
  `RHIFrameFlipThr` at 24 jiffies (3%), the rest at 0–2. A CPU census alone would have called this
  process idle. (This lane.)
- **#1990's publish path is not the cause of the black frame.** `[rtt] PRESENT SOURCE EXTENT
  MISMATCH` fires exactly 7 times, all at submits #1–#7 with `fresh=0 retained=0` — the boot prefix
  before any pass exists — and never again; no `[agc] PUBLISH DROPPED` line appears in any arm.
  (This lane, `PROSPER_DBG=1` arm.)
- **Not trap 87's "silent failure prefix".** Sonic Frontiers publishes nothing for its first 946
  present callbacks and then recovers; here a pre-fix arm ran 400 s with `rendered_samples=0` and
  never recovered, and the post-fix arm composites from the first sample at t=20 s.
- **Entitlement answers cannot be the boot blocker.** No `sceAppContent*`/`sceNpEntitlementAccess*`
  ownership query is reached before the wall, and none of the 21 unimplemented NIDs is one.
  (Tracker #1895, 2026-08-05 breakpoint census; unchanged by this lane's arms.)

## Not verified

Rung 2. Whether the post-logo uniform frame is a movie surface, a dropped-draw composite, or a
legitimate loading screen. Any input route — nothing has been shown to respond to a pad yet. Windows
or macOS. Performance figures: every arm shared the GPU with other lanes.
