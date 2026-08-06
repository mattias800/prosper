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

- **Draws are dropped because one of their two shader stages does not recompile.**
  `[exec-recompile-reject]` names the pair (`es=…`/`ps=…`) whose draws are skipped and reports each
  stage's emitted SPIR-V size, so it also says *which* stage failed. **Four** `(es, ps)` pairs are
  affected on every arm, not the two the first census recorded: `vs=4445 fs=0` and `vs=1346 fs=0`
  fail in the **pixel** shader, and two pairs sharing one `es` at `vs=0 fs=88449` / `vs=0 fs=139235`
  fail in the **vertex** shader. The occurrence counter (logged on powers of two) reaches **256+**
  for one of them, so this is hundreds of dropped draws per boot.
- **Eight distinct compute programs are skipped** (`[compute] skip unsupported program`), plus one
  3D-tile volume dispatch skipped under #590.

### The reject chain — and the partition that matters more than the list

`PROSPER_DBG=1` names the *first* unhandled instruction in each program, so the census is a peeled
onion: implementing one tier reveals the next inside the same shaders. Every encoding named below
was round-tripped through `llvm-mc -mcpu=gfx1030` from the exact logged words
(`tools/re/disasm_words.py`), never read off an opcode table — see the `0x305` trap in *Ruled out*.

**Attributing each reject to the program it came from splits the census into two populations that
the earlier flat list had merged.** The attribution is mechanical: `PROSPER_SHADER_DUMP=<dir>` writes
the raw bytes of every failing stage, `shader_inspect --stage …` re-runs the recompiler over each
dump offline, and the exact logged words can be located by index inside those dumps.

| population | how many | what actually fails |
|---|---|---|
| the dropped **graphics** pipelines | 4 → **2** | a **descriptor resolution**, never a missing opcode |
| the skipped **compute** programs | 8 | missing instructions, and unstructured control flow |

Every one of the four dropped pipelines fails on an unresolved descriptor:

```text
[mimg-unresolved]  pc=80   srsrc=s0  srt_tag=NONE  (ps 4445, image_sample)     -- FIXED
[mimg-unresolved]  pc=654  srsrc=s8  srt_tag=0x60  (ps 1346, image_sample_lz)  -- FIXED
[mubuf-unresolved] pc=211  srsrc=s4  srt_tag=NONE  (the shared vs, buffer_load_format_xyz)
```

So **`buffer_load_format_xyz` was never a missing opcode.** `e0082000,6a010000` disassembles to
`buffer_load_format_xyz v[0:2], v0, s[4:7], vcc_lo idxen`, which the recompiler has always
supported; its V# does not resolve. It belongs with the two `[mimg-unresolved]` failures and the
bindless-descriptor work, exactly where the two MIMG rejects were already assigned.

The consequence is load-bearing: **finishing the opcode list cannot change what the four dropped
draws do**, because not one of them is blocked on an opcode. Opcode work can only restore compute
programs — which is still worth doing (a skipped LUT or exposure dispatch is a documented way a
whole composite collapses), but it is not the lever for these draws.

#### The two PIXEL stages: one 4K 2D_ARRAY target sampled at BASE_ARRAY=1

**Both `[mimg-unresolved]` failures were the same texture, and neither was a provenance failure.**
`PROSPER_DYNTRACE_FAIL=1` replays a failed stage's resource build with the scalar const-fold traced;
on both pixel shaders the fold **recovers the T# exactly**, from opposite provenance routes:

```text
PS 0x2600066500  MIMG pc=80  srsrc=s0  seed_t8=1 key=0xffffffff   (T# straight from the entry user data)
PS 0x284015a000  MIMG pc=654 srsrc=s8  have_t8=1  key=0x60        (T# loaded through a descriptor table)
both -> t8=41246e00 c2400000 021bc3bf d1b003ac 00010001 …
        base=0x41246e0000 3840x2160x1 type=13 fmt=36 tile=27
```

`type=13` is `SQ_RSRC_IMG_2D_ARRAY` and word 4 `0x00010001` is `BASE_ARRAY=1, LAST_ARRAY=1` — the
title samples **layer 1** of a 4K array target. The descriptor was resolved and then **discarded
during materialization**: all three paths that turn a decoded T# into a `ShaderResource` carried
`d.base_array != 0 → continue`. The recompiler then had no resource at that pc, reported an
unresolved descriptor, and the draw was skipped — which reads exactly like a provenance failure from
the reject line alone.

That guard was added (d43d82a6) as a deliberate fail-closed measure *while the per-slice byte stride
was unmodelled*. #5237659e then modelled it: `image_base_level_view` advances the view base by
`BASE_ARRAY * tiled_mip_chain_bytes(...)` for cube faces and 2D-array slices, and
`test_build_shader_resources` has pinned that 720896-byte stride since. **The three call-site guards
were never lifted, so the descriptor never reached the code that could resolve it.** The fix gates
the refusal on the image TYPE instead — only where the slice origin is genuinely unmodelled (1D_ARRAY,
2D_MSAA_ARRAY, a 3D UAV view) — and routes every path through one named predicate,
`image_descriptor_reject_reason`, which reports *why* a T# was refused.

Measured, two 260 s `tools/screenshot` arms on `61cc4877`, identical switches, same session:

| | before | after |
|---|---|---|
| `[mimg-unresolved]` lines | 10 | **0** |
| distinct dropped `(es, ps)` pipelines | 4 | **2** (both the vertex-stage pair) |
| `[mubuf-unresolved]` lines | 1 | 1 (unchanged) |
| skipped compute programs | 8 | 8 (unchanged) |
| composite | logo `0d70a70a`, then uniform `8bf1b518` | **identical** |

**The composite did not move.** Both arms reproduce the tracker's recorded hashes byte-for-byte, so
each carries its own positive control. Restoring these two pipelines is a necessary step that is not
by itself sufficient; the title is still rung 1.

#### The two VERTEX stages: the descriptor chain starts at a null pointer in guest memory

The same instrument settles the vertex stage, and **the answer is not the one the static recon
predicted**. The fold models this shader's entire "synthesized V#" idiom already — `s_bfe_u64`,
`s_or_b32`, `s_cmp_eq_u32` and `s_cselect_b32` are all implemented, the last one explicitly as *"the
vertex-fetch format patch's tail"*. It never gets to use them, because everything upstream is
unknown:

```text
[dyntrace] SMEM op=0xc bufload sdst=s16 sbase=s8 base=0x4148dca410 … imm=0x60 n=16
                                    ; s_buffer_load_dwordx16 s[16:31] <- a 160-byte cbuf, in range
[dyntrace] SMEM op=0x0  load sdst=s20 sbase=s18 base=0x0 …
[dyntrace]   addr 0x0 unreadable                       ; s[18:19] read back as NULL -> s20 unknown
[dyntrace] SMEM op=0x2  load sdst=s4  sbase=s16 base=0x3f800000 soff_field=106 soff_ok=0
                                    ; the V# load: SBASE is 0x3f800000 (the float 1.0), SOFFSET
                                    ; is vcc_lo, itself derived from the unknown s20
[dyntrace]   SOP2 pc=210 op=0xa dst=s7 src0=7(k0) src1=107(k0) ok=0   ; s_cselect: both arms unknown
[dyntrace] MUBUF fetch pc=211 op=0x2 SRSRC=s4 patched=0 (k=0000) have_descr=0 soff_known=0
[dyntrace]   MUBUF pc=211 SOFFSET untracked -> fetch left unresolved (not folded to 0)
```

Read the chain end to end: the shader loads 64 bytes of a constant buffer into `s[16:31]`, and the
bytes that land in `s[18:19]` are **zero** while `s16` reads `0x3f800000`. It then dereferences that
null pointer, extracts a bitfield from the result, and builds both the V#'s SOFFSET and its word-3
select out of it. The fold reads that constant buffer successfully (`base_ok=1 soff_ok=1`, in range
of its own `num_records=10 × stride=16`) — so this is **real guest data, correctly read, containing a
null bindless-table pointer**, not a fold that ran out of modelled opcodes.

So the question for the vertex stage is not "how do we resolve a runtime-selected descriptor". It is
**why that constant buffer's pointer field is zero when prosper realizes the draw** — the guest
either has not populated it through a path prosper replays, or prosper is reading the wrong buffer.
`CONFIDENCE: HIGH` on the trace; `CONFIDENCE: LOW` on which of those two it is — nothing here
separates them yet. Filed as [#2132](https://github.com/mattias800/prosper/issues/2132).

### The opcode tiers, measured

Identical 220 s `boot_trace` arms, `PROSPER_DBG=1`, one arm per tier. Absence is asserted from two
independent instruments, because a single one would not be enough: the live `[recompile-reject]`
line, **and** an offline `shader_inspect` pass over the dumped programs, which has no rate limit and
no cap and is therefore complete by construction over the shaders it is given.

| tier | rejecting ops (sites/arm) | disassembly of the exact logged words |
|---|---|---|
| `beeff2ab` | VOP3 `0x307` (30), VOP3 `0x156` (8), VOP1 `0x54` (3) | `v_lshrrev_b16` (one site with `op_sel:[0,0,1]`), `v_max3_u32`, `v_rcp_f16_e32` |
| + those (#2067) | VOP3 `0x303` (24), VOP1 `0x52` (10) | `v_add_nc_u16`, `v_cvt_u16_f16_e32` |
| + those (#2067) | VOP3 `0x358` (14), VOP3 `0x365` (5), VOP3P `0xa` (5) | `v_med3_i16`, `v_mbcnt_lo_u32_b32`, `v_pk_add_u16` |
| `f080fc23` (this lane's base) | VOP2 `0x1b` (4), VOP1 `0x1` (11), VOP3 `0x365` (5) | `v_and_b32_sdwa` with a WORD **destination**, `v_mov_b32_sdwa` with a WORD **source**, `v_mbcnt_lo_u32_b32` |
| + those | VOP3P `0xa`, VOP3 `0x357` — behind the three above *inside the same programs*, so they surface offline rather than as their own arm | `v_pk_add_u16` (including the `NEG` form below), `v_med3_f16` |
| + those | VOP1 `0x53` (5), VOP1 `0x51` (5), VOPC `0xab` (4) | the SDWA form of `v_cvt_i16_f16` / `v_cvt_f16_i16`, `v_cmp_le_u16_sdwa` |
| + those | VOP3 `0x311` (4), VOP1 `0x51` (5, now `src0_sel:WORD_1`) | `v_pack_b32_f16`, the WORD-source form of the same convert |
| + those (**where this lane stopped**) | VOP1 `0x1` (5) | `v_mov_b32_sdwa v14, sext(v8) src0_sel:WORD_0` — the **sign-extending** WORD move, one bit away from the zero-extending form above and a different operation |

Everything in every tier so far is the **16-bit VALU family** in some encoding — scalar, packed,
SDWA, compare, convert and pack. One UE5 post chain runs end to end in packed 16-bit arithmetic,
which is why it peels rather than showing its whole requirement at once.

**What is left after all of the above, and is not that family:**

- `s_cbranch_vccz` / `s_cbranch_vccnz` (SOPP `0x6`/`0x7`) and a forward `s_cbranch_execz`
  (SOPP `0x8`) — the #590 control-flow frontier. Two compute programs stop here, one on a
  `[compute-struct-reject] backward else pc=36 branch=117 target=118` shape.
- `s_flbit_i32_b64` (SOP1 `0x16`) — **implemented** by this lane, but it sits *behind* the
  `s_cbranch_vccz` in the same program, so it changes nothing until the control flow does.
- `v_mov_b32_sdwa` with **SEXT** on its source (`7e1c02f9,000c0608`): the zero-extending form is
  implemented, the sign-extending one is a different operation and still rejects. This is the very
  next tier and is small; it was left because the census had not converged and the peel had already
  moved twice past the point where the answer to *this title's* question was settled.
- `image_sample_lz` with `dim=SQ_RSRC_IMG_2D_ARRAY d16` (`f09c0f28,80ea003d`) and `image_atomic_add`
  with the same dim (MIMG `0x11`) — arrayed-image sampling and atomics, the deferred MIMG features
  already inventoried in [`RECOMPILER_REMAINING.md`](RECOMPILER_REMAINING.md).
- The three `[mimg-unresolved]` / `[mubuf-unresolved]` descriptor failures above.
- One "program" at `0x2d800d0000` that is **12,916 dwords of zero**. It is not a shader; treat a
  reject at `pc=0 words=00000000,00000000` as a bad `code_addr`, not as an opcode gap.

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

- **`buffer_load_format_xyz` is NOT a missing opcode, and the dropped draws are not an opcode
  problem at all.** #2013 listed MUBUF `0x2` alongside the missing instructions. Attributing every
  reject to its program shows all **four** dropped `(es, ps)` pipelines fail on an unresolved
  descriptor instead — two `[mimg-unresolved]` in the pixel shader and one `[mubuf-unresolved]`
  (`pc=211 srsrc=s4 srt_tag=NONE`) in the vertex shader that two of them share.
  `e0082000,6a010000` round-trips to `buffer_load_format_xyz v[0:2], v0, s[4:7], vcc_lo idxen`,
  which the recompiler has always supported. **So finishing the opcode list cannot change what those
  draws do** — it can only restore compute programs. (This lane, 2026-08-06.)
- **The earlier "two dropped `(es, ps)` pairs" count was low.** A 220 s arm reports **four**, and the
  two new ones fail in the **vertex** stage (`vs=0 fs=88449` / `vs=0 fs=139235`), which the
  `[exec-recompile-reject]` line already said in its per-stage SPIR-V sizes. Read `vs=`/`fs=` as
  "which stage produced nothing", not as a shader identifier. (This lane, 2026-08-06.)
- **`shader_inspect --stage` cannot see past the FIRST table-dependent instruction, so it under-reports
  a graphics stage's real reject.** On every one of this title's failing pixel shaders it stops at the
  `SMEM` at `pc=2`/`pc=3` with `reason=graphics-disabled` — the documented #1571 limitation — and one
  of them reports `unsupported=0` from generic coverage, which reads like a clean bill of health.
  The live `[recompile-reject]` from a real boot is the only instrument that names the actual failing
  instruction in a graphics stage. It is still exact for **compute**, which passes `allow_smem=true`.
  (This lane, 2026-08-06.)
- **The dropped draws were NOT four instances of one descriptor problem — they are two populations
  with unrelated causes, and neither is "the descriptor cannot be resolved".** A
  `PROSPER_DYNTRACE_FAIL=1` arm replays each failed stage's resource build with the scalar const-fold
  traced. The two **pixel** stages resolve their T# perfectly and lose it during *materialization*
  (`base_array != 0`, fixed here); the two **vertex** stages never resolve anything because the
  constant-buffer dword holding their bindless-table pointer reads **zero**. Do not carry "the four
  dropped pipelines fail on descriptor resolution" forward as one frontier — fixing the pixel half
  changed nothing about the vertex half. (This lane, 2026-08-06.)
- **The failing vertex `V#` is NOT blocked on being "synthesized in-shader".** The
  [#1895 static recon](https://github.com/mattias800/prosper/issues/1895) read
  `s_cselect_b32 s7, s7, vcc_hi` at `pc=210` as a runtime-selected word 3 that no table could
  contain, and proposed modelling a runtime-selected V#. The const-fold **already models that
  idiom** — `s_cselect_b32` is implemented in `gpu_executor.cpp` and commented *"the vertex-fetch
  format patch's tail"*, alongside `s_bfe_u64`, `s_or_b32` and `s_cmp_eq_u32`. The live trace shows
  `pc=210 ok=0` with **both** arms unknown and `scc` unknown, because a preceding
  `s_load_dword s20, s[18:19]` hit `addr 0x0 unreadable`. The select is downstream of the real
  failure, not the failure. Which arm is taken is therefore not a question that can be answered from
  this shader's own state — nothing is known. (This lane, 2026-08-06.)
- **`shader_inspect` is not the instrument for a failing graphics stage; `PROSPER_DYNTRACE_FAIL=1`
  is.** The previous entry below records that `shader_inspect --stage` stops at the first
  table-dependent SMEM. The replay path (`gpu_execute.hpp`, gated on `PROSPER_DYNTRACE_FAIL`) rebuilds
  the *failed* stage's table with `[dyntrace]`/`[resdump]`/`[udmap]` forced on, once per distinct
  shader address, and needs no address known up front. It answered both of this title's descriptor
  questions in a single 260 s arm after two lanes had reasoned about them statically. (This lane.)
- **A `[recompile-reject] pc=0 words=00000000,00000000` is a bad `code_addr`, not an opcode gap.**
  The skipped program at `0x2d800d0000` is 12,916 dwords of zero from its first word. It has been in
  every census since the first and is not an instruction the recompiler is missing. (This lane.)

## Not verified

Rung 2. Whether the post-logo uniform frame is a dropped-draw composite or a legitimate loading
screen — a movie surface is now ruled out above, but the other two are not separated. Whether the
remaining two dropped `(es, ps)` pipelines are the UI layer, the post chain, or both. Whether
resolving their descriptors is enough to reach a title screen: it is now known that finishing the
*opcode* list is not, because none of the four was blocked on one, **and that restoring the two
pixel-stage pipelines is not either — the composite is byte-identical with them recompiling.** What
the two now-restored pipelines actually draw has not been examined. Any input route — nothing has
been shown to respond to a pad yet. Windows or macOS. Performance figures: every arm shared the GPU
with other lanes.

## The next step, in order

1. **Find out why `s[18:19]` is null** — [#2132](https://github.com/mattias800/prosper/issues/2132),
   which carries the full trace and the two candidate explanations.
   The remaining two dropped draws share one vertex stage whose
   descriptor chain begins by dereferencing a constant-buffer dword that reads zero (traced above).
   The two candidate explanations — the guest fills it through a path prosper does not replay, or
   prosper realizes the draw against the wrong/stale buffer — are separated by watching that address:
   the cbuf is at a V# the user-data block supplies directly (`s[8:11]`), so `PROSPER_HWWATCH` on
   `V#.base + 0x60` during the uniform phase says whether anything ever writes it. That is the
   frontier; it is **not** a runtime-selected-V# modelling problem, which is what the earlier reading
   implied (see *Ruled out*).
2. **Look at what the two restored pixel pipelines now draw** before assuming the composite needs
   more pipelines. They recompile and submit; whether their output reaches the present source is a
   separate question, and `[rtt] PRESENT SOURCE EXTENT MISMATCH` still fires early in every arm.
3. **The remaining opcode work is control flow**, not arithmetic: `s_cbranch_vccz`/`vccnz` and a
   forward `s_cbranch_execz`, in two compute programs, one of them on a `backward else` shape. The
   16-bit VALU family is otherwise complete through five tiers.
4. **Then re-measure the composite.** It has not moved: black to t≈20 s, SEGA logo, then the uniform
   frame, on every arm of every tier so far — including the arm where the two pixel pipelines were
   restored.
