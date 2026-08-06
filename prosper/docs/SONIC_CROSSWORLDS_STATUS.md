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
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
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

From t≈80 s the presented frame is a single colour, `RGBA(1,0,1,255)`, while `frame_seq` and
`present_count` keep climbing. **This is not a hang, and `frame_seq` climbing is not evidence that
it is alive** (instrument trap 90). The engine itself is healthy — the AGC submission threads,
`RHIThread`, `RenderThread 0` and `CriManaDecodeTh` all accumulate CPU time throughout.

**How long it lasts, measured to 900 s** (2026-08-06). A clean 900 s arm sampling every 30 s stays in
the uniform state for **810 s without a single pixel changing**: `pixel_crc32` `8bf1b518` at every
sample from t=90 s to t=900 s, while `frame_seq` climbs 1,041 → 9,183 and `present_count` 322 → 3,035.
`rendered_samples` is 30 of 30 and `pixel_distinct_frames` is 3 (black prefix, logo, uniform). Nothing
in fifteen minutes of default launch leaves this state.

The composite is thin rather than absent. With `PROSPER_SUBMITLOG_DIM=3840x2160`, the 10 s window
before a uniform sample carries **53 submits targeting the 4K present extent, 89 draw items in
total** — one to two draws per submit, the same order as during the SEGA logo (44 submits / 73 draw
items at t=50 s). (GPU shared with two other lanes; read those as counts per interval, not as a rate.)

**And the guest is running an ordinary engine loop, not waiting on one thing.** `tools/hle_calls` over
a 400-tick window in each phase (positive control: `g_vo_flipstatus` fires in both) is the *same shape*
during the uniform frame as during the SEGA logo — mutex/TLS/clock traffic dominates, VideoOut flip and
vblank polling continues at the same rate, AGC register patching continues:

| handler | logo phase (t≈45 s) | uniform phase (t≈75 s+) |
|---|---|---|
| `k_getspecific` | 4,336 | 6,837 |
| `k_mutex_lock` / `k_mutex_unlock` | 3,938 / 3,934 | 4,784 / 4,776 |
| `g_vo_flipstatus` / `g_vo_vblankstatus` | 236 / 236 | 222 / 222 |
| `k_rwlock_init` / `k_rwlock_destroy` | — | 143 / 143 |
| `agc_patch_add_registers` | — | 128 |

**No `libSceVideodec2` or `libSceAvPlayer` handler appears in either window**, so the "the next thing
after the logo is a CRI Mana movie whose video never reaches the screen" hypothesis has no support
from the call census. The title is compositing almost nothing while doing normal work.

Two measured leads, both recorded on [#2013](https://github.com/mattias800/prosper/issues/2013):

- **Draws are dropped because their shaders do not recompile.** `[exec-recompile-reject]` names a
  graphics pair (`es=…`/`ps=…`) whose draws are skipped, and `PROSPER_DBG=1` gives the rejecting
  opcodes. Two `(es, ps)` pairs are affected on every arm — `vs=4445 ena=0x20` and
  `vs=1346 ena=0x302` — and the occurrence counter (logged on powers of two) reaches **256+** for one
  of them, so this is hundreds of dropped draws per boot, not the 32 the first census could see.
- **Nine distinct compute programs are skipped** (`[compute] skip unsupported program`), plus one
  3D-tile volume dispatch skipped under #590.

### The reject chain, and how deep it goes

`PROSPER_DBG=1` names the *first* unhandled instruction in each program, so the census is a peeled
onion: implementing one tier reveals the next inside the same shaders. Measured on current master
(`beeff2ab`) and after each tier landed, over identical 200 s `boot_trace` arms:

| tier | rejecting ops (sites/boot) | disassembly (`llvm-mc -mcpu=gfx1030` of the exact logged words) |
|---|---|---|
| master | VOP3 `0x307` (30), VOP3 `0x156` (8), VOP1 `0x54` (3) | `v_lshrrev_b16` (one site with `op_sel:[0,0,1]`), `v_max3_u32`, `v_rcp_f16_e32` |
| after those | VOP3 `0x303` (24), VOP1 `0x52` (10) | `v_add_nc_u16`, `v_cvt_u16_f16_e32` |
| after those | VOP3 `0x358` (14), VOP3 `0x365` (5), VOP3P `0xa` (5) | `v_med3_i16`, `v_mbcnt_lo_u32_b32`, `v_pk_add_u16` |

Everything in the first three tiers is the **16-bit VALU family**, which is why it peels this way:
one UE5 post-processing shader does its whole chain in packed 16-bit arithmetic. What survives all
three tiers, and is *not* that family, is the real remaining list: `v_and_b32_sdwa` /
`v_mov_b32_sdwa` in WORD-select form (VOP2 `0x1b`, VOP1 `0x1`), `v_pk_add_u16` (VOP3P `0xa`),
`v_mbcnt_lo_u32_b32` (VOP3 `0x365`), `buffer_load_format_xyz` (MUBUF `0x2`), `s_cbranch_vccz/vccnz`
(SOPP `0x6`/`0x7`, the #590 family), `s_flbit_i32_b64` (SOP1 `0x16`), and the two `[mimg-unresolved]`
descriptor failures. **The two MIMG rejects are not missing opcodes** — `0x20`/`0x27` disassemble to
`image_sample` / `image_sample_lz`, which the recompiler supports; their SRSRC does not resolve.

## Unimplemented-call census (default launch, current master + #2012)

**21** NIDs reach the dispatcher's `-> returning 0` path, resolved against the PS5 3.20 firmware
export tables. `pthread_rwlock_trywrlock` was the twenty-first and is now implemented; the rest are
listed on the tracker. Nothing in the list is an entitlement or add-content query — the title's
`scePlayGo`/`sceAppContent` traffic is answered from local inventory, and the two unimplemented
`libScePlayGo` NIDs (`scePlayGoGetSupportedOptionalChunk`, `scePlayGoGetInstallChunkId`) are the
same two the 2026-08-05 baseline recorded.

Re-run on `beeff2ab` (2026-08-06) the list is unchanged in kind. `sceRandomGetRandomNumber`
(`PI7jIZj4pcE`) is worth calling out separately: the dispatcher's default `0` is a **success** return
that leaves the caller's output buffer untouched, so the guest reads whatever was already there as
"random" data. Filed as its own issue rather than fixed here.

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
- **#2021 and #2003 change nothing here.** `sceSysmoduleIsLoaded` answering from prosper's own load
  history, and app-param/SKU answering from `param.json`, were both absent from the `c3614f51` base
  the rung-1 evidence was taken on. On `beeff2ab`, which has both, the boot is identical
  frame-for-frame: black to t≈20 s, SEGA logo `pixel_crc32=0d70a70a` (1,373 colours, 430,916
  non-black px) at t≈40–60 s, uniform `8bf1b518` from t≈80 s. The only `[sysmodule]` traffic is two
  first-query `IsLoaded` misses (`0xe2`, `0xba`). (This lane, 2026-08-06, two independent arms.)
- **RETRACTED within this lane: "the uniform frame is a phase, not a terminal state".** One 270 s arm
  sampled every 30 s showed pure black (`666f7b3f`, 0 non-black px) at its final t=270 s sample, and
  this doc briefly recorded that as the title leaving the uniform state. It does not reproduce: a
  clean 900 s arm is uniform at every sample from t=90 s to t=900 s, and a second 270 s arm at the
  same cadence is uniform at t=270 s. Two of the three arms are therefore against it, and the one
  that showed it had `gdb` attached to the process twice earlier in the run for `hle_calls`. Treat
  that black sample as unexplained and confounded, not as evidence of progression — and note the
  general shape: **a single sample from an arm that carried an invasive instrument is not a
  measurement of the subject.** (This lane, 2026-08-06.)
- **`0x156 = v_max3_u32` is confirmed, and `0x307` is NOT `v_lshlrev_b16`.** #2013 flagged the first
  as positional inference; a round-trip through `llvm-mc -mcpu=gfx1030` of the exact logged words
  confirms it and names the other two (`0x307 = v_lshrrev_b16`, `0x54 = v_rcp_f16`). The trap is one
  opcode over: **`0x305` is `v_mul_lo_u16`, and `v_lshlrev_b16` is `0x314`** — implementing 0x305 as
  a left shift compiles, validates, and silently computes the wrong value. Caught by a bit-exact
  execution test before it shipped. (This lane, PR for #2013.)
- **The post-logo wall is not a stalled CRI Mana movie.** `hle_calls` over 400-tick windows in both
  the logo and the uniform phase (positive control: `g_vo_flipstatus` present in both) shows no
  `libSceVideodec2` and no `libSceAvPlayer` handler at all, and the same mutex/TLS/flip-poll shape in
  both. The engine is doing ordinary work while compositing almost nothing. (This lane, 2026-08-06.)
- **`pidof <name>` selects the WRONG lane's process, and the result looks completely valid.** An
  `hle_calls` arm here attached by `pidof screenshot` and produced a full 47-handler histogram — for
  another worktree's title, because three lanes were running `screenshot` concurrently. The only tell
  is the tool's own header line naming the binary path. Select by
  `readlink /proc/<pid>/exe` and compare it against your own build directory. (This lane, 2026-08-06.)
- **A one-frame F9 grab is not a reliable instrument on this title.** The headless
  `PROSPER_CAPTURE_BUNDLE_TRIGGER_FILE` grab armed at present 470 and reported `[grab] frame-bundle:
  window had no submits` — a real measurement (that present interval carried no GPU submit at all),
  but not evidence that the title has stopped submitting: the same phase carries ~50 4K-target
  submits per 10 s. Use `PROSPER_CAPTURE_FRAMES>1` here. (This lane, 2026-08-06.)

## Not verified

Rung 2. Whether the post-logo uniform frame is a dropped-draw composite or a legitimate loading
screen — a movie surface is now ruled out above, but the other two are not separated. Whether the two
dropped `(es, ps)` pipelines are the UI layer, the post chain, or both, and therefore whether
finishing the reject list is enough to reach a title screen. Any input route — nothing has been shown
to respond to a pad yet. Windows or macOS. Performance figures: every arm shared the GPU with other
lanes.

## The next step, in order

1. **Finish the reject list** (#2013). The non-16-bit remainder is small and each piece is separable:
   the SDWA WORD-select forms of `v_and_b32`/`v_mov_b32`, `v_pk_add_u16`, `v_mbcnt_lo_u32_b32`,
   `buffer_load_format_xyz`, the #590 `s_cbranch_vcc*` family, `s_flbit_i32_b64`. The two
   `[mimg-unresolved]` failures belong with the bindless-descriptor work instead.
2. **Then re-measure the composite**, not before — the two pipelines reject at their *first*
   unhandled instruction, so no partial tier can be expected to change a pixel.
3. If the composite is still uniform with zero `[exec-recompile-reject]` lines, the cause is
   elsewhere and the nine skipped compute programs plus the skipped 16x16x16 volume dispatch are the
   next inventory.
