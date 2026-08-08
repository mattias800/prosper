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

**No `libSceVideodec2` or `libSceAvPlayer` handler appears in either window** — but **do not read
that as ruling the movie out; it was, and the ruling was wrong.** The census is a 400-tick sampling
histogram and the Videodec2 traffic is nine one-shot calls, so their absence here is a limit of the
instrument rather than a fact about the title. The "next thing after the logo is a movie whose video
never reaches the screen" hypothesis turned out to be **correct** (#2281). See the corrected row in
`## Ruled out` and instrument trap 137.

Two measured leads, both recorded on [#2013](https://github.com/mattias800/prosper/issues/2013):

- **Draws are dropped because one of their two shader stages does not recompile.**
  `[exec-recompile-reject]` names the pair (`es=…`/`ps=…`) whose draws are skipped and reports each
  stage's emitted SPIR-V size, so it also says *which* stage failed. **Four** `(es, ps)` pairs are
  affected on every arm, not the two the first census recorded: two report `fs=0` and so fail in the
  **pixel** shader, and two more — sharing one `es` — report `vs=0` and fail in the **vertex**
  shader. Those numbers are per-stage SPIR-V word counts, so a zero names the stage that produced
  nothing; they are not shader identifiers, and the non-zero one is not a name for the pipeline. The occurrence counter (logged on powers of two) reaches **256+**
  for one of them, so this is hundreds of dropped draws per boot.
- **Eight distinct compute programs were skipped** (`[compute] skip unsupported program`), plus one
  3D-tile volume dispatch skipped under #590. **Four of the eight now recompile and dispatch** — see
  *The eight skipped compute programs, one by one* below, which names every one of them and its exact
  blocking instruction. The composite did not move.

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
| the dropped **graphics** pipelines | 4 → 2 → **0** | a **descriptor resolution**, never a missing opcode |
| the skipped **compute** programs | 8 | missing instructions, and unstructured control flow |

At the census above, every one of the four dropped pipelines failed on an unresolved descriptor.
The lines are reproduced **unedited** — no `-- FIXED` marker or other annotation inside the block —
because their field values are the evidence for the reading that follows, and an annotated copy
cannot carry it. **All three are now fixed** — the two `[mimg-unresolved]` by #2131 and the
`[mubuf-unresolved]` by #2132; that is stated after the block rather than inside it, so the excerpt
stays verbatim and the current status stays unmissable:

```text
[mimg-unresolved] pc=80 srsrc=s0 srt_tag=NONE 0x0 key_res=null pc_res=null (7 res)
[mimg-unresolved] pc=654 srsrc=s8 srt_tag=0x60 key_res=null pc_res=null (12 res)
[mubuf-unresolved] pc=211 srsrc=s4 srt_tag=NONE 0x0 key_res=null (5 res)
```

**Status now: all three are FIXED — the two `[mimg-unresolved]` by #2131, the `[mubuf-unresolved]`
by #2132.** Dropped pipelines went 4 → 2 → **0**, `[mimg-unresolved]` 10 → 0 and
`[mubuf-unresolved]` 1 → 0. **The composite is byte-identical across all three states**, so this
whole frontier is closed without moving a pixel — see the two subsections below for each half.

Read those lines the way the reject path does, not by pattern-matching the tag. **`[mimg-unresolved]`
prints on a wider condition than the one that rejects**: it fires whenever `res` is null *or* carries
a class that is neither Texture nor StorageImage, while the reject immediately below is `if (!res)`.
For MIMG `0x00`/`0x0e` the class filter above it deliberately does not null a wrong-class `res`, so
the line can print on a resolved descriptor and **not** reject.

**And the two fields it prints are not the whole story.** `res` has *three* provenances —
`by_fetch_pc`, `by_srt_offset`, and `by_sgpr_base` — but the printf reports only the first two, as
`pc_res` and `key_res`. `by_sgpr_base` is class-blind and is never named in the output. So
`key_res=null pc_res=null` is **necessary but not sufficient**: it does not by itself establish that
`res` is null.

**#2131 is the confirming instance, and it is stronger than the argument.** Both of those pixel-stage
lines printed with `key_res=null pc_res=null`, and the const-fold nevertheless **recovered the T#
exactly** from both provenance routes. The descriptor was resolved and then discarded during
*materialization* by a `base_array != 0 → continue` guard — so the reject line was reporting a
downstream refusal while reading, to anyone who trusted the two printed fields, as a provenance
failure. Two lines that look like "the descriptor could not be found" meant "the descriptor was found
and thrown away." That is why the excerpt above is kept raw.

Note also that the live vertex stage reports `(5 res)`, not `no-table`: it *has* a resource table and
the key lookup fails inside it. (A table-less `shader_inspect` run of the same shader prints
`key_res=no-table` instead, which is the tool's limitation, not the guest's behaviour — see
*Ruled out*.)

So **`buffer_load_format_xyz` was never a missing opcode.** `e0082000,6a010000` disassembles to
`buffer_load_format_xyz v[0:2], v0, s[4:7], vcc_lo idxen`, which the recompiler has always
supported; its V# does not resolve. It belongs with the two `[mimg-unresolved]` failures and the
bindless-descriptor work, exactly where the two MIMG rejects were already assigned.

The consequence is load-bearing: **finishing the opcode list cannot change what those dropped
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

#### The two VERTEX stages: RESOLVED — the "null pointer" was prosper's own straight-line walk

**There was never a null pointer in guest memory, and no writer was ever missing.** The dword that
"reads zero" is not a pointer field: it is `float4(1.0, 0, 0, 0)` in an ordinary uniform buffer, and
prosper's scalar const-fold put it into the register the shader dereferences by executing a block
the wave cannot be on. Fixed in [#2132](https://github.com/mattias800/prosper/issues/2132).

The fold walks the decoded stream **straight-line and models no control flow**. That is harmless
while a shader is one basic block. This stage is not. Walked from the retained
`PROSPER_SHADER_DUMP` binary with instruction lengths validated against the emulator's own pcs — the
MUBUF the reject names lands on dword **211**, exactly where `[dyntrace]` reports it, and the
`s_cselect_b32` on dword **210**:

```text
pc=16   s_buffer_load_dwordx4  s[0:3], s[8:11], 0x50
pc=19   v_cmp_ge_f32_sdwa      vcc_lo, s0, 1.0        ; the cbuf holds FLOATS, compared against 1.0
pc=21   s_cbranch_vccz 151                            ; -> pc=173
pc=45   s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60
pc=113  s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60
pc=172  s_branch 47                                   ; -> pc=220, jumping OVER pc=173
pc=173  s_load_dword           s20, s[18:19], null
pc=188  s_load_dwordx4         s[4:7], s[16:17], vcc_lo
pc=210  s_cselect_b32          s7, s7, vcc_hi
pc=211  buffer_load_format_xyz v[0:2], v0, s[4:7], vcc_lo idxen
```

`pc=173` has **no fall-through predecessor** — `pc=172` is an unconditional `s_branch` past it — and
exactly one branch targets it. So whenever the wave executes `pc=173`, the two
`s_buffer_load_dwordx16` at `pc=45`/`pc=113` **did not run**: they are inside the block the branch
skips. The straight-line walk ran them anyway, and they write `s16..s31`.

**What they overwrite is the answer.** A vertex stage seeds user data at **s8**
(`user_sgpr_base = is_ps ? 0u : 8u`), so user-data dword *k* lands in s(8+*k*) — and this stage's
AGC header declares direct pointers at dw8 and dw10:

```text
[resdump]   direct offset_dw: … [8]=8 … [10]=10
[resdump] sgprs@0x8c: 37b4a410 00100041 0000000a 0004dfac   ; dw0..3  -> s8..s11  cbuf V# (160 B)
                      40079ca0 00100029 00000166 0004dfac   ; dw4..7  -> s12..s15 second V#
                      37b4a210 00000041                     ; dw8..9  -> s16:s17  table pointer
                      40208268 00000025                     ; dw10..11-> s18:s19  fetch-descriptor ptr
[udmap] declared direct: [8]@dw8=0x4137b4a210(readable) [10]@dw10=0x2540208268(readable)
```

Both pointers are seeded and **both are readable**. The walk replaces them with the constant
buffer's floats, so by `pc=173` `s[18:19]` is `0` and `s[16:17]` is `0x3f800000` — the float `1.0`
read as an address. That `[udmap]` line was printed in **every arm of this investigation** and read
as "the block is coherent, look elsewhere"; it was in fact naming the two pointers the fold then
destroyed.

The fix is a narrow CFG rule in the fold. A target qualifies only when **all** of these hold, and the
list is worth reading exactly as written because a weaker version of it fires on shapes it must not
(#2202 review):

- the instruction **physically** preceding it is an unconditional `s_branch` — physically, because
  the fold walks a *compacted* stream that drops ordinary VALU/EXP/DS/FLAT while preserving PCs, so
  "the previous element" is the previous *retained* instruction and a dropped block between them
  would be invisible;
- **exactly one** branch in the whole program targets it, counted over **both** directions, since a
  backward edge is a second predecessor;
- the program contains no indirect control transfer (`s_setpc_b64` / `s_swappc_b64` / `s_rfe_b64` /
  `s_call_b64`), which would make the CFG unrepresentable by any scan over branch displacements.

Its only predecessor is then that branch, and since a branch writes no register, **the state at the
target is exactly the state at the branch** — so the rule saves the whole interpreter state at the
branch and restores it at the target, rather than rolling back registers one at a time. That also
covers `scc`, the scalar spill slots and the vector index-mode table, which a register-only rollback
left carrying the skipped path forward. Published descriptor *uses* are deliberately not rolled
back: an instruction inside the skipped region is genuinely reachable on its own path.

This does not claim the state at the branch is itself right — if branches precede it the walk may
already be a chimera. It claims only that the target now agrees with one real path instead of two
mutually exclusive ones. `PROSPER_NO_BRANCH_EXCLUSIVE` restores the old walk and **must stay off**;
it exists only so the A/B below stays reproducible.

Measured, two 270 s `tools/screenshot` arms, same binary, same session, identical switches:

| | `PROSPER_NO_BRANCH_EXCLUSIVE=1` | default |
| --- | --- | --- |
| `[exec-recompile-reject]` lines | 8 | **0** |
| distinct dropped `(es, ps)` pipelines | 2 | **0** |
| `[mubuf-unresolved]` lines | 1 | **0** |
| `[dynfail] replaying VS` | 1 | **0** |
| skipped compute programs | 8 | 8 (unchanged) |
| composite | `666f7b3f` → `0d70a70a` → `8bf1b518` | **identical** |

**The composite did not move.** The off arm reproduces the recorded signature exactly — the same two
pipelines `vs=0 fs=88449` / `vs=0 fs=139235` sharing one `es`, and
`[mubuf-unresolved] pc=211 srsrc=s4 srt_tag=NONE 0x0 key_res=null (5 res)` — so the lever is
demonstrably connected and the identical composite is a real null rather than an undelivered change.
**Every dropped graphics pipeline and every unresolved descriptor in this title is now gone (4 → 0),
and the title is still rung 1.**

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
| + those | VOP1 `0x1` (5) | `v_mov_b32_sdwa v14, sext(v8) src0_sel:WORD_0` — the **sign-extending** WORD move, one bit away from the zero-extending form above and a different operation |
| `cdb40942` + those (this lane) | VOP1 `0x1` (9), VOP1 `0x5e` (4), VOP1 `0x5c` (3) | `v_mov_b32_sdwa … sext(…)` in **both** the WORD and BYTE source forms, `v_rndne_f16_sdwa`, `v_ceil_f16_sdwa` |
| + those (**where this lane stopped**) | VOP2 `0x25` (4), VOP1 `0x1` DPP (3) | `v_add_nc_u32_sdwa v4, 8, sext(v5) src1_sel:WORD_0` — SEXT one operand position over, on a VOP2; and `v_mov_b32_dpp v10, v4 row_xmask:4`, which is a cross-lane frontier, not an SDWA one |

Everything in every tier so far is the **16-bit VALU family** in some encoding — scalar, packed,
SDWA, compare, convert and pack. One UE5 post chain runs end to end in packed 16-bit arithmetic,
which is why it peels rather than showing its whole requirement at once.

**What is left after all of the above, and is not that family:**

- `image_sample_lz` with `dim=SQ_RSRC_IMG_2D_ARRAY d16` (`f09c0f28,80ea003d`) and `image_atomic_add`
  with the same dim (MIMG `0x11`) — arrayed-image sampling and atomics, the deferred MIMG features
  already inventoried in [`RECOMPILER_REMAINING.md`](RECOMPILER_REMAINING.md).
- `v_mov_b32_dpp v10, v4 row_xmask:4 bound_ctrl:1` (`7e1402fa,ff096404`) — a cross-lane DPP row
  control, not an SDWA one. It is the first instruction outside both the 16-bit VALU family and the
  descriptor frontier that this title has produced.
- `s_flbit_i32_b64 vcc_lo, s[14:15]` — **the opcode is implemented**; what fails is that the emitter
  has no value for `s14`/`s15` at that pc, so the lowering takes its `ok = false` arm. This is a
  scalar-value-tracking gap, not a missing instruction, and it is easy to mis-file as one.
- One "program" (address is run-local) that is **12,916 dwords of zero**. It is not a shader; treat a
  reject at `pc=0 words=00000000,00000000` as a bad `code_addr`, not as an opcode gap.

**Closed, and listed here only so the historical record below is readable:** the three
`[mimg-unresolved]` / `[mubuf-unresolved]` descriptor failures, and the fourth `[mimg-unresolved]`
described next. Every arm on current master reports **0** of each; the paragraph that follows is the
record of how the fourth was found and attributed, not an open item.

- **A fourth `[mimg-unresolved]`, exposed by the opcode work and since fixed.** `pc=686 srsrc=s40 srt_tag=0x60 key_res=null
  pc_res=null (47 res)` appears **only** on the arm taken after every tier above landed: `pc=686`
  occurs 0 times in the base, first-tier and second-tier census arms and 20 times in the last one.
  It is in a **compute** program that previously stopped at an arithmetic op long before reaching
  it — in the first-tier arm that same shader was still stopping at `pc=34`/`101`/`113`/`115`/`130`
  (the VOPC `0xab`, `s_flbit`, and the two 16-bit converts). So the opcode work did move something
  measurable — just not a pixel: it advanced a compute program far enough to hit the same
  descriptor frontier the graphics pipelines are stuck on.

  **The stage is established from the shader dump, not from the adjacent diagnostic.** It would be
  natural to read "compute" off the `[cfg-recompile-reject]` line printed at the same pc, and that
  inference is wrong: `emit_cfg_state_machine` is reached from graphics too (`rdna2_to_spirv.cpp`,
  the `complex_graphics_cfg` caller, whose own comment says the dispatcher selects from "this
  pixel/vertex's SCC, VCC, or EXEC bit"). What settles it is locating the exact word at the exact pc
  inside the retained `PROSPER_SHADER_DUMP` binaries: `word[686] == 0xf09c0f28` lands in an
  `exec_cs_*.bin` — the same 2,348-dword compute shader in every arm that contains it (the addresses
  in those filenames are run-local).

### The eight skipped compute programs, one by one

The flat reject census above answers "which instructions reject"; it does **not** answer "what stops
each program", and those are different questions whenever one program can reject at several pcs. The
attribution is mechanical and worth repeating rather than re-deriving:

```bash
# 1. one 240 s arm writes every failing compute program's raw bytes and the live reject lines
PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_DBG=1 \
PROSPER_SHADER_DUMP=~/shaders PROSPER_NO_FRAME_DUMPS=1 \
  timeout -s TERM -k 5s 240s ./build-linux/boot_trace <DUMP_ROOT>/PPSA08804-app0 > ~/census.log 2>&1
grep -E 'compute\] skip|cfg-recompile-reject|compute-struct-reject' ~/census.log

# 2. shader_inspect's per-instruction listing gives pc -> (fmt, op, words) for each dump, so every
#    live `[cfg-recompile-reject] pc=N fmt=F op=0xX` matches exactly one program
./build-linux/shader_inspect ~/shaders/exec_cs_<addr>.bin

# 3. name the instruction from its bytes, never from an opcode table
python3 prosper/tools/re/disasm_words.py <word0>,<word1>
```

On `cdb40942` that gives a **1:1** attribution — every live reject pc occurs in exactly one of the
eight dumps, with matching format and opcode. Addresses are run-local; the dword counts are not.

| dwords | blocking instruction (disassembled from the exact logged words) | state |
|---|---|---|
| 224 | `s_flbit_i32_b64 vcc_lo, s[14:15]` (`beea160e`) — opcode implemented, `s14`/`s15` untracked | still skipped |
| 636 | `v_mov_b32_sdwa v25, sext(v25) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1` | **restored** |
| 1128 | `image_atomic_add v19, v[28:30], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY` | still skipped |
| 1296 | `v_rndne_f16_sdwa v6, v6 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1`, then `v_add_nc_u32_sdwa v4, 8, sext(v5) src1_sel:WORD_0` behind it | **restored** |
| 1368 | `v_mov_b32_sdwa v14, sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0` | **restored** |
| 1448 | `v_mov_b32_sdwa v16, sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0` | **restored** |
| 2348 | `v_ceil_f16_sdwa v10, v10 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1`, then `v_mov_b32_dpp v10, v4 row_xmask:4 bound_ctrl:1` behind it | still skipped |
| 12916 | all zero from `pc=0` — a bad `code_addr`, not a shader | not a program |

**Every one of the seven real programs is stopped by a single instruction, and none of those
instructions is a branch.** That contradicts what this document previously recorded — see the
*Ruled out* row below — and it is the reason the remaining work is an instruction inventory rather
than the #590 structurizer.

Measured across three identical arms on one binary lineage, `PROSPER_DBG=1`, 240 s each:

| | `cdb40942` | + the SDWA/f16 tier | + VOP2 source SEXT |
| --- | --- | --- | --- |
| `[compute] skip unsupported program` | 8 | 5 | **4** |
| distinct `[cfg-recompile-reject]` sites | 7 | 4 | 3 |
| `[exec-recompile-reject]` (graphics) | 0 | 0 | 0 |
| `[mimg-unresolved]` / `[mubuf-unresolved]` | 0 / 0 | 0 / 0 | 0 / 0 |
| composite | `666f7b3f` → `0d70a70a` → `8bf1b518` | **identical** | **identical** |

**That table counts one log line, and a fifth failure prints a different one.** `[compute] skip
unsupported program` is the recompiler's message. A program that recompiles *successfully* and then
fails descriptor validation prints `[compute] skip invalid descriptor contract` instead, and has
never been in the total above — so the population is **one larger than every census here states**.
On master at `7a493df2` the split is **3** `skip unsupported program` (#2218 took one more of the
four) plus **1** descriptor-contract, and the offline capture records exactly that: four declined
dispatches per frame, `reason=shader-recompile` x3 and `reason=descriptor-contract` x1, in submits
8022/8039 and 8023/8040 of the repeating frame. Ask a capture for
`--bundle-extract-submit N out.prgcap` then `--inspect-only out.prgcap`; do **not** use the replay's
`operations=A/B` field, which is a replay setting over a decode count and can never report a decline
(instrument trap 141).


The composite row is a `tools/screenshot` arm (`--seconds 30 --count 9`, default switches, no
diagnostics) rather than the `boot_trace` arm, so the pixels are measured on a run that carries no
instrument at all. It reproduces the recorded hash sequence byte for byte, which is the arm's own
positive control; a reference decode of the PNGs confirms 1,373 distinct colours and 430,916
non-black pixels in the logo frame and exactly **one** colour, `RGB(1,0,1)`, over all 8,294,400
pixels of the uniform one.

**The lever and the null were then re-measured in ONE run**, because a skip count taken in
`boot_trace` and a pixel hash taken in `screenshot` are two frontends, and "the change did not reach
the frontend that draws" is exactly the alternative a cross-frontend comparison cannot exclude. A
single `screenshot` arm with `PROSPER_DBG=1` reports **4** skipped compute programs *and* the same
`666f7b3f` → `0d70a70a` → `8bf1b518` at every sample. Same process, same frame loop: the four
restored programs are dispatching in the run whose pixels did not change.

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

- **RUNG 2 REACHED -- THE TITLE SCREEN RENDERS COMPLETELY, AND "THE COMPOSITE GOES UNIFORM" WAS NEVER
  THE RIGHT DESCRIPTION.** With denser pulses on the route above (`scripts/sonic-crossworlds/`), the
  full sequence is **black -> SEGA -> Unreal Engine -> CRIWARE -> licensor -> auto-save notice ->
  TITLE SCREEN -> player-profile menu -> white**. The title screen is complete at 3840x2160: the 3D
  scene, every character, the track, the logo, `PRESS`, `License Information`, `(C)SEGA` and version
  `1.1.2`. **Reproduced across two independent 560 s runs.** Note the composite CRC does **not**
  match between them -- the scene is animated -- so the comparison is by content signature:
  **113,524 / 113,676** distinct sampled colours at 99.6% coverage in the first run against
  **113,476 / 114,379** at 99.6% in the second, at the same sample positions. For scale this document
  records the SEGA logo at 1,373 colours.
  *Method worth reusing:* opening twenty 4K frames is expensive, so the unexamined ones were **scored
  before being opened** -- distinct sampled colours and non-black ratio -- and the two that stood two
  orders of magnitude above the rest were the only ones opened. A CRC tells you frames differ; it does
  not tell you which one is worth looking at.
  **The defect is now per-screen, not global.** The title screen's scene renders; the profile menu's
  UI composites correctly (rank badge, account name, BACK / VIEW PLAYER / NEXT) while its central
  content panel stays black. So "no scene-geometry pass" is a statement about *some* screens. The
  terminal white state and the letterboxed auto-save notice remain unexplained. Rung 3 (gameplay) is
  not reached. #2013.

- **THE TITLE WAS WAITING FOR A BUTTON, AND THE RENDERER WAS FINE ALL ALONG.** The one item this
  document listed as untested -- *"any input route -- nothing has been shown to respond to a pad
  yet"* -- is the one that moves. With `scripts/sonic-crossworlds/advance-boot-logos.pad` the title
  renders, in order and correctly at 3840x2160, the **Unreal Engine** logo (`42763e4a`), the
  **CRIWARE** logo (`5170ed80`) and the **legal / licensor text screen** (`b3b61854`). **None of the
  three had ever been observed on this title**; four independent no-pad observations -- two arms of
  mine and the two master arms above -- show only black / SEGA logo / black. Two pad arms produce an
  identical CRC set with sampling counts differing by one.
  **The edge is the mechanism.** An arm holding Cross continuously from frame 0, with delivery
  confirmed from the guest's own read (`[pad] pad_read_state call#512 connected=1 buttons=0x4000`),
  advanced nothing over 170 s. A held button is not a press: the guest needs a neutral->pressed
  transition. That distinction is why every earlier "input does nothing" impression was wrong.
  **Strong but not airtight:** nine samples over 270 s would very likely have caught one of these
  screens had they played regardless of input, and four no-pad observations caught none -- but a
  same-session alternating A/B has not been run. **It does not reach a title screen** -- it reaches
  the game's own **auto-save notice** (`824976b1`) -- and past that, with denser pulses, **the title
  screen itself** and then the player-profile menu. *The route's own shape was the limiter at first:* an earlier
  version ended in a long hold and stopped at the legal screen, which by the edge mechanism above is
  a limit the route created rather than one the title has -- continued pulses reached the notice.
  Flagged and not claimed: that notice renders **letterboxed** into a horizontal band rather than
  filling the target, which may be its own presentation or a viewport defect; unchecked against
  hardware. #2013.

- **NO UNIMPLEMENTED NID IS BEING POLLED -- THIS IS NOT SONIC FRONTIERS' WALL.** That title's
  four-session black screen was one unregistered NID answering `SCE_OK` and being called **1,319
  times** (#2023), so the same census was the obvious first move here. It comes back clean:
  `PROSPER_PROGRESS=5 PROSPER_PROGRESS_UNIMPL=1`, `boot_trace`, 120 s, **12 distinct unimplemented
  functions** and the only one with any volume at all is `libScePosix::Xs9hdiD7sAA` at **127** calls
  -- `pthread_setschedparam` per `../PS5-3.20_Libs`, i.e. the title setting thread priorities and
  getting a benign success. Everything else is called once or twice, `libScePlayGo` included (2 and
  1), so a content-availability wait is not what is happening either.
  **And the frame loop is healthy, not stalled:** over the same run, presents climb steadily to
  1,287 in 119 s (~10.8/s), flips to 403, `draws_cum` to 11,749 and dispatches to 17,500. *Read
  `draws_last` carefully* -- it is the draws in the **last submit**, not in the heartbeat interval,
  so its frequent `0` is not "the title stopped drawing": `draws_cum` rises by ~425 every 5 s
  throughout. The title runs, draws ~85 times a second, dispatches ~140 times a second, presents,
  and shows black. #2013.

- **RESTORING THE LAYERED ATOMIC DISPATCH DOES NOT MOVE THE COMPOSITE.** #2265's full-screen
  `IMAGE_ATOMIC_ADD` dispatch was the strongest remaining candidate -- it writes the image the
  presented frame is composed from, and unlike the four programs restored before it, it covers the
  whole screen. With the fix (#2356) the gap is provably closed on a live boot: the
  `skip invalid descriptor contract` line goes 1 -> **0**, the 20 `[compute-descriptor]` issues go to
  **0**, and the staging trace reports `atomic-image buffer binding=37 ... bytes=66355200` **164
  times** -- `3840*2160*2*4`, i.e. both layers, where a single-layer staging would read 33,177,600.
  **The presented frame is byte-identical to master across the whole documented route.**
  `tools/screenshot`, default launch, `--seconds 30 --count 9` (270 s -- the same arm as the composite
  row above): all **three** master states reproduce exactly, `666f7b3f` (x1), `0d70a70a` (x1) and
  `8bf1b518` (x7). Frames were opened rather than judged by hash: `666f7b3f` and `8bf1b518` are both
  black (different CRCs, visually identical) and `0d70a70a` is the SEGA logo on black, so the route
  is black -> logo -> black and stays there. **Match the window to the claim:** a first pass at
  `--seconds 8 --count 9` (72 s) ends on the logo and never reaches `8bf1b518` at all, so it cannot
  speak to the post-logo state that #2013 is about -- the arm was re-run at 270 s for that reason.
  Five restored programs across two sessions have now left the composite byte-identical. Look elsewhere before spending on the three remaining `skip unsupported program`
  entries -- their restoration is worth doing on the charter's own grounds, but it should not be
  expected to change the picture. #2356, Refs #2265 / #2013.

- **THE DESCRIPTOR-CONTRACT SKIP IS NOT A STALE CACHED SPIR-V.** The hypothesis was that the program
  compiled once against a single-layer resolution, was cached by `recompile_compute_shader_cached`,
  and is validated on later dispatches against the two-layer resource -- which would make it a
  general recompile-time vs bind-time divergence rather than a title-local gap. A/B on one
  `boot_trace` binary, 75 s per arm, `PROSPER_DBG=1 PROSPER_RENDER_TIMING=1`, arm B adding
  `PROSPER_NO_SHADER_CACHE=1`:

  | arm | shaders hit | miss | bypass | `skip invalid descriptor contract` | `skip unsupported program` |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | cache on | 21.1 | 1.1 | 0.0 | 1 | 3 |
  | cache off | 0.0 | 0.0 | **22.2** | 1 | 3 |

  The lever provably moved -- every compile in arm B is fresh against the live table -- and the
  failure population is unchanged in both class and count. Note the counts are by **class**, not by
  program address: guest code addresses are run-local and every one of them differs between the two
  arms, so an address-keyed comparison across runs is void (this cost the first pass of this very
  A/B). #2265.

- **THE REMAINING SKIPPED COMPUTE PROGRAMS ARE NOT ALL RECOMPILER OPCODE GAPS, AND THE ATOMIC ONE
  IS NOT A MISSING FEATURE -- IT IS THREE COPIES OF ONE PREDICATE THAT DISAGREE.** Program
  `0x288012e000` recompiles, then fails descriptor validation on **20** bindings (37-56), every one
  `image_atomic_add` (MIMG `op=0x11`) against a **two-layer** R32_UINT 2D-array image --
  `[compute-resource] binding=37 class=4 fmt=2 comps=1 dims=3840x2160x2 pc=751`. The dispatch is
  full-screen: `240x135` groups of 64 threads is exactly 1920x1080 in 8x8 tiles. **Four** sites
  independently encode which image-atomic image shapes are supported, and **#2272 generalised exactly
  one of them** to 2D_ARRAY (line numbers as of `7a493df2`):

  | site | where | 2D_ARRAY accepted? |
  | --- | --- | --- |
  | coverage predicate | `src/gpu/rdna2_to_spirv.cpp:16031` (`i.mimg_dim == 1u`) | no |
  | lowering gate | `src/gpu/rdna2_to_spirv.cpp:10209` (`atomic_2d_array`) | **yes, #2272** |
  | validator carve-out | `src/gpu/shader_resources.cpp:1031` (`r.img_dim == 1 && r.depth == 1`) | no |
  | backend materialization | `frontends/shared/live_compute.cpp:4304` (same clause) | no |

  So the lowering emits the buffer-backed binding for a 2D_ARRAY atomic and the validator then
  rejects that exact binding as `WrongType`. That is why the capture shows `recompiled=yes` with
  `descriptors=20` **and** a descriptor-contract failure -- a pair that reads as contradictory until
  the three sites are read together. Buffer-flattening is the **designed** path for
  `image_atomic_add`, not a fallback (`tests/test_rdna2_spirv_struct.cpp:3483` requires
  `kind == StorageBuffer` with `atomic_access` and `report.ok()`); it exists as the RADV
  image-atomic workaround. **This is #2293's defect one iteration later** -- that PR is titled *"the
  image-atomic opcode list existed in THREE places and all three had to agree"*, and the same triple
  now disagrees on the *dimension* predicate instead of the opcode list. **Do NOT simply widen sites
  1 and 3: the `compute_atomic_buffer` index is still `coords[0] + coords[1]*width` with the layer in
  neither the index nor the bound**, so relaxing the validator alone makes this dispatch *run* and
  every layer atomically accumulates into layer 0's texels -- a silent wrong result on a device-scope
  atomic, strictly worse than the current skip, and it would read as progress in a screenshot. Correct
  order: teach the backend's detile/retile to cover the layers, fold the layer into the shader index
  and bound to match the layout it produces, **then** widen the predicates. Note the backing
  allocation is **already** large enough -- the resource reports `available=66355200` and
  `3840*2160*2*4 = 66,355,200` exactly -- so sizing is not the obstacle; the **layout** is.
  `tiled_surface_bytes(width, height, tile_mode, pitch, bytes_per_texel)` (`src/gpu/tile.hpp:81`) takes
  no depth or layer argument, so the detile behind the atomic buffer view is inherently 2D and the
  tiled **array slice pitch** is the open unknown that needs evidence rather than an assumed
  `width*height`. Tracked on **#2265**,
  which owns this chain; Refs #2272 / #2293.
- **THE UNIFORM COMPOSITE IS NOT THE UNAUTHORED 16³ GRADING LUT.** The title binds a 16x16x16 2-byte
  storage image whose authoring dispatch was skipped (`3D tile mode has no volume address pattern` --
  `Sw4KbS`, tile mode 5), and the frozen composite is a single colour, so the LUT looked like the
  cause. #2229 implements `SW_4KB_S3` volume addressing and the dispatch now runs -- **verified by the
  lever**: the skip count goes 1 -> 0 between arms. The rendered output does not move. Same route
  (`screenshot --count 3 --every 60`), master vs fix, both arms give `(25,25,25)` then `(0,0,0)` twice,
  all 8,294,400 pixels, one distinct colour each; on a shorter route the two builds' PNGs are
  byte-identical (same SHA-256). Authoring that image changes nothing observable, so the uniform
  composite has another cause. #2229 is still a real gap and is fixed on its own merits.

- **THE SKIPPED COMPUTE PROGRAMS DO NOT FAIL ON CONTROL FLOW.** This document and #2013 both recorded
  "they fail on **control flow**, not arithmetic: 21 `[cfg-recompile-reject]` and 26
  `[compute-struct-reject]` lines per boot, on `s_cbranch_vccz`/`vccnz` and a forward
  `s_cbranch_execz`". Attributing every live reject to the program it came from shows the opposite:
  **all seven real programs stop on one instruction each, and not one of them is a branch** —
  `v_mov_b32_sdwa … sext(…)` ×3, `v_rndne_f16_sdwa`, `v_ceil_f16_sdwa`, `image_atomic_add
  dim:2D_ARRAY`, and an `s_flbit_i32_b64` whose *opcode is implemented* and whose scalar source is
  untracked. **The proof is the A/B, not the reading of a diagnostic**: implementing two SDWA
  families and nothing else restored **four of the seven**, and the branch shapes those programs
  contain are untouched by that change.
  **Why the wrong reading was available.** `[compute-struct-reject]` is **one tag over two roles** —
  14 emission sites inside `detect_forward_ifs` and 7 inside the structured emitter — so counting the
  lines merges a routing decision with a terminal failure. On these programs the
  `detect_forward_ifs` lines are the routing decision: the recompiler then runs the CFG dispatcher,
  which accepted those same shapes and carried on to the arithmetic op that `[cfg-recompile-reject]`
  names. And `structured emission stopped next-pc=N next-if=…` — 12 of the 26 lines — is not a
  control-flow complaint at all: `next-pc` is the **blocking instruction**, matching the
  `[cfg-recompile-reject]` pc exactly, and `next-if=4294967295` says in the diagnostic's own fields
  that no `if` is pending. **The caveat matters**: a `detect_forward_ifs` decline is only a routing
  decision when the dispatcher is actually selected (`cfg_dispatch_safe` and either
  `exact_compute_wave_cfg` or `cfg_branches > 2`); otherwise `cf_rejected` clears the loop list and
  the branch genuinely does block. So ask **which emitter ran last**, and attribute a reject to a
  *program* before attributing it to a *cause*. (This lane, 2026-08-06, on `cdb40942`.)
- **Restoring HALF the skipped compute programs does not move the composite.** 8 → 5 → **4** skipped
  programs across three identical arms on one binary lineage, with 0 dropped graphics pipelines and 0
  unresolved descriptors throughout, and the composite is byte-identical at every step:
  `666f7b3f` → `0d70a70a` → `8bf1b518`, the uniform frame exactly one colour `RGB(1,0,1)` over all
  8,294,400 pixels by reference decode. **The lever and the null are measured in the same run** — a
  `screenshot` arm with `PROSPER_DBG=1` reports 4 skipped programs *and* the unchanged hashes — so
  this is a real null rather than a change that never reached the frontend being photographed.
  **Read the scope precisely**: what
  is falsified is *"restoring these four moves the composite"*. Four programs are still skipped —
  `image_atomic_add dim:2D_ARRAY`, `v_mov_b32_dpp row_xmask`, the untracked-scalar `s_flbit`, and the
  all-zero non-program — so "a skipped compute dispatch is the cause" is **not** ruled out.
  (This lane, 2026-08-06.)

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
- **~~The post-logo wall is not a stalled CRI Mana movie.~~ THIS ROW WAS WRONG — the wall IS a movie.**
  The original entry read: *"`hle_calls` over 400-tick windows in both the logo and the uniform phase
  (positive control: `g_vo_flipstatus` present in both) shows no `libSceVideodec2` and no
  `libSceAvPlayer` handler at all."* The measurement was real; the conclusion did not follow.
  `PROSPER_SVCLOG=1` over the same phase shows `sceVideodec2CreateDecoder -> 0x10001 codec=2382845`
  and eight `sceVideodec2Decode` calls, and the A/B settles it: implementing VP9 decode (#2281)
  carries the title **through** the wall — intro video, Unreal Engine splash, auto-save notice, and
  the garage scene at 64,461 colours.
  **Why the census produced a false negative, which is the reusable part.** The control was
  `g_vo_flipstatus` at **236 calls** in the window; the Videodec2 traffic is **nine** calls, fired
  once when the movie starts. A 400-tick *sampling* histogram exists to show steady-state work, and
  nine one-shot events are exactly what it does not retain. So the control proved the instrument
  fires on a HIGH-RATE handler and could not, even in principle, establish that it would catch a
  nine-call event — which is the quantity the null was about. **Match a control's rate to the rate
  of the thing you are looking for**; a control at 236 validates nothing about detecting 9. See
  instrument trap 137. (Corrected by this lane, 2026-08-08, #2281.)
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
- **THE WHOLE "null bindless-table pointer" SEARCH WAS FOR A PHANTOM — the dword is `1.0f` uniform
  data, and prosper's own const-fold put it in that register. Do not resume any hunt for the writer,
  the stale buffer, or the legitimate-null arm.** The fold walks the decoded stream straight-line and
  models no control flow, so it executed `s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60` at `pc=45`
  and `pc=113` — both inside the block that `s_cbranch_vccz` at `pc=21` skips whenever `pc=173`
  executes — and overwrote the stage's two **declared direct pointers** (dw8/dw10, seeded at
  `s16..s19` because a vertex stage seeds at s8, and both reported `readable` by `[udmap]` in every
  arm) with the constant buffer's floats. Same-binary, same-session A/B on
  `PROSPER_NO_BRANCH_EXCLUSIVE`: off → 8 `[exec-recompile-reject]`, 1 `[mubuf-unresolved]`, 1 VS
  replay; on → 0/0/0. **All four candidates on #2132 are dead at once**, because all four presumed
  the value was something the guest produced: a missing GPU writer, a guest CPU write, a stale/wrong
  buffer, and a legitimate null the shader guards. It was none of them. (This lane, 2026-08-06, #2132.)
- **Restoring the last dropped pipelines does NOT change the post-logo composite.** With 0 dropped
  pipelines, 0 `[mimg-unresolved]` and 0 `[mubuf-unresolved]`, both A/B arms produce the identical
  hash sequence `666f7b3f` → `0d70a70a` → `8bf1b518`, and the post-logo frame is exactly **one**
  distinct colour, `RGB(1,0,1)`, over all 8,294,400 pixels. Since the lever demonstrably changes the
  reject counts in the same session, this is a real null and not an undelivered change.
  **Read the scope precisely**: what is falsified is *"restoring these pipelines moves the
  composite"*. It is **not** established that "the uniform frame is not a dropped-draw composite" —
  the counters prove these stages now *recompile and submit*, not that their draws executed, wrote
  anything, or that what they wrote reaches the present source, and `[rtt] PRESENT SOURCE EXTENT
  MISMATCH` still fires early in every arm. Recompiling is not contributing. The delivery path and
  the 8 skipped **compute** programs are both still open. (This lane, 2026-08-06, #2132; scope
  narrowed after the #2202 review, which caught this row overreaching its own commit's
  *Not verified* section.)
- **No GPU-side writer that prosper executes ever fills the vertex stage's null constant-buffer
  pointer.** *(Superseded by the two rows above — the negative was correct and the question was
  wrong; nothing was supposed to write it.)* The remaining two dropped pipelines fail because a
  constant-buffer dword holding a
  bindless-table pointer reads zero (#2132). Queried against the shared guest-write history at the
  failing stage's own replay: `no recorded GPU writer overlaps [0x…470,+0x40) (history=8589 recorded:
  color=39 compute-buffer=4744 dma-data=8127 write-data=1111)` — 8,589 retained writes, all four
  recorders demonstrably firing, none overlapping that window. Measured on a
  `PROSPER_WRITER_PROVENANCE=1` arm, which matters: that switch selects **unfiltered** retention, and
  without it DmaData/WriteData discard writes below 256 bytes / 64 dwords — exactly the size that
  could hide a write to a 64-byte window. The negative carries its own positive
  control (#2143 made it do so; before that the same query printed 183 bare negatives while the
  colour recorder had never been armed, and the result was VOID rather than negative). **Two
  populations remain unexamined and are NOT ruled out**: a *skipped* compute dispatch records nothing
  by construction — this title skips 8 per boot, and that is the leading hypothesis — and guest CPU
  writes are recorded by no writer kind at all. (This lane, 2026-08-06.)
- **The zero is not run-to-run nondeterminism.** *(Correct, and now explained: it is deterministic
  because it is a compile-time constant in a uniform buffer, reached by a fold that takes the same
  wrong path every boot.)* The pointer field reads zero in **five** independent
  boots while its containing allocation base moves every time (`0x4148dca410` / `0x4137f4a410` /
  `0x400e6ba410` / `0x413080a410` / `0x413111a410`), with the same offsets and the same neighbouring
  `0x3f800000` each time. **This does not rule out an ordering defect** — only a nondeterministic
  one. A *deterministic* mis-ordering (realizing the stage table before a producer that precedes it
  in `command_order`) reads zero on every boot with a moving base too, and is entirely consistent
  with all five. Read this row as "stop looking for a flaky race", not as "ordering is cleared".
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

Rung 2. Whether the post-logo uniform frame is a legitimate loading screen — the movie surface and
the dropped-draw composite are both now ruled out above, which leaves that reading and "content the
skipped compute programs were supposed to produce" unseparated. What the four now-restored **graphics
pipelines** actually draw has not been examined; that they submit and recompile is established, that
their output reaches the present source is not. Whether restoring the **remaining four** skipped
compute programs moves the composite — the first four did not. Any input route — nothing has
been shown to respond to a pad yet. Windows or macOS. Performance figures: every arm shared the GPU
with other lanes.

## The next step, in order

**The graphics-descriptor frontier is closed — do not reopen it.** 0 dropped pipelines, 0
`[mimg-unresolved]`, 0 `[mubuf-unresolved]` (#2131 + #2132), and the composite is byte-identical
before and after. **The #590 control-flow frontier is not this title's blocker either** — see the
first *Ruled out* row. Everything below is what remains.

1. **Establish whether the four now-restored graphics pipelines reach the present source at all.**
   Ordered first because it is the cheapest way to tell a *delivery* defect from a *content* one, and
   because the counters above cannot: they prove these stages recompile and submit, not that their
   draws executed, wrote anything, or contributed a pixel. A byte-identical composite while four
   more pipelines submit is itself weak evidence for a delivery problem, and
   `[rtt] PRESENT SOURCE EXTENT MISMATCH` still fires once early in every arm. The same argument now
   applies to the four restored **compute** programs, and with more force: restoring them changed the
   skip count and not one pixel.
2. **The three compute programs still skipped** — and **lower this in priority than its position
   suggests.** The argument for it was that a skipped LUT/exposure dispatch is a documented way a
   whole composite collapses. That argument is now spent on this title: **five** restored programs
   across two sessions have left the composite byte-identical, the most recent (#2356) being a
   *full-screen* dispatch writing the image the presented frame is composed from. Restore them
   because the charter requires unsupported ops to be implemented, not because they are expected to
   change the picture. In ascending cost:
   - `s_flbit_i32_b64 vcc_lo, s[14:15]` — a **value-tracking** gap, not a missing opcode. The
     lowering exists and takes its `ok = false` arm because `s14`/`s15` are unknown at that pc.
   - `v_mov_b32_dpp v10, v4 row_xmask:4 bound_ctrl:1` — a cross-lane DPP row control, in the
     **2,348-dword** program (the one that stopped on `v_ceil_f16_sdwa`), not in either of the
     programs this lane restored.
   - the 12,916-dword all-zero `code_addr`, which is not a program and should not be counted as one.
3. **Re-measuring the composite is no longer a pending step -- it has been done and it is a
   `## Ruled out` row.** The route is black to t~20 s, the SEGA logo (`0d70a70a`, 1,373 colours),
   then a single-colour `RGB(1,0,1)` frame (`8bf1b518`) -- note that is (1,0,1) out of 255, i.e.
   **near-black, not magenta**, which is easy to misread from the numbers alone. Byte-identical
   across both arms of the #2132 A/B, all three arms of the earlier compute work, and #2356, where
   the reject and skip counts differ and the pixels do not. **Match the window to the claim**: a
   short arm ends on the logo and never reaches `8bf1b518` at all, so use the 270 s
   `--seconds 30 --count 9` arm for any composite comparison.

4. **What has NOT been examined, and is where the evidence now points.** A steady-state frame carries
   **25 draws and 39 computes with no scene-geometry pass at all** (full inventory on #2303). The
   graphics-descriptor and control-flow frontiers are closed, every pipeline recompiles, and the
   compute inventory is now spent -- so the question is no longer "why is submitted work dropped"
   but **"why does the guest submit so little"**, which is a CPU/logic-side question rather than a
   renderer one. Nothing has been shown to respond to a pad yet either. Start there.
