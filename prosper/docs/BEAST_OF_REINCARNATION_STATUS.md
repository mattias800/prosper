# Beast of Reincarnation (`PPSA29343`) — status

Tracker: [#2916](https://github.com/mattias800/prosper/issues/2916).
Developer/publisher: Game Freak / SEGA. Engine: **Unreal Engine 5** (IoStore + AgcRHI), with
**Wwise** audio (25 `ak*.prx` plugin modules under `prx/`).

## Engine generation — how it was established

The dump's command-line file is `uecommandline.txt`, without the `4` that every UE4 title in this
corpus carries. That naming is **not** by itself the discriminator, and reading it as one is a
trap: `PPSA08804` (*Sonic Racing: CrossWorlds*) ships the same spelling and is already recorded as
Unreal Engine 5. Three independent checks say UE5 here:

1. **UE5-only subsystem strings in `eboot.bin`** (`strings`, case-insensitive counts):
   `Nanite` 51, `Lumen` 77, `VirtualShadowMap` 7, `WorldPartition` 56, `Substrate` 7,
   `HeterogeneousVolumes` 7, `SparseVolumeTexture` 10. None of these exist in UE4.
2. **The IoStore container version.** `global.utoc` byte 16 is the `EIoStoreTocVersion`, and this
   title's is **6** against **2/3** for every UE4 title in the corpus
   (`PPSA03026`=2; `PPSA07809`, `PPSA17168`, `PPSA17942`, `PPSA20447`=3). The only other **6** is
   `PPSA08804`, the known UE5 title.
3. **`uecommandline.txt` content** names `BeastOfReincarnation.uproject`, and the eboot's source
   paths include `/Runtime/AgcRHI/`, UE's PS5 RHI.

Narrowing the **minor** version is weaker and is recorded as a hypothesis, not a finding. Two
independent directions agree, which is why it is worth stating at all:

- **String presence/absence.** `Substrate`, `HeterogeneousVolumes` and `SparseVolumeTexture` are
  present (5.2/5.3-era) while `MegaLights`, `AnimNext`, `Chooser` and any Nanite-tessellation
  string are absent (5.4/5.5-era).
- **The container version itself.** TOC version **6** is `EIoStoreTocVersion::OnDemandMetaData`,
  which is UE 5.3-era — so the same byte that establishes "UE5, not UE4" also brackets the minor
  version, from an axis that has nothing to do with which subsystems the game happens to link.
  (Contributed by the reviewer of PR #2917.)

That brackets it around **UE 5.3**. `CONFIDENCE: HIGH` for "UE5"; `CONFIDENCE: LOW` for "5.3" —
two agreeing indirect signals are still two indirect signals, and no engine version string appears
anywhere in the dump.

`sce_sys/param.json` reports `sdkVersion 0x1100000000000000` (SDK **11.00**) and
`requiredSystemSoftwareVersion 0x1260000000000000` (**12.60**). SDK 11 is below the SDK-13 gate on
prosper's post-submit completion-visibility contract, so `PROSPER_POST_SUBMIT_VISIBILITY` is *not*
armed for this title by default — relevant if a submit race ever shows up here, and not something
observed yet.

## Current rung: 1 — real content renders, under one default-off lever

**Rung 0 on a default launch, rung 1 with `PROSPER_CB_EFC_NO_COLOR=1`, and the difference is a
prosper defect rather than anything the title does.** Say it that way round every time: on the
default route this title presents a flat 4K clear and nothing else, and the reason is that prosper
runs the guest's `CB_COLOR_CONTROL.MODE = ELIMINATE_FAST_CLEAR` passes — over **8,192** of them per
boot here — as ordinary colour draws that paint over the composite ([#1588](https://github.com/mattias800/prosper/issues/1588)).
Suppress the colour writes on those draws and the title's own screens appear underneath.

![Beast of Reincarnation — the GAME FREAK developer logo at 3840x2160](../../assets/screenshots/beast-of-reincarnation-game-freak-logo.png)

The **GAME FREAK** developer logo, t≈45 s, direct unmodified `tools/screenshot` capture at
3840x2160 on Linux/RADV.

![Beast of Reincarnation — the Digital Deluxe bonus dialog with its item list, scrollbar and OK button, at 3840x2160](../../assets/screenshots/beast-of-reincarnation-deluxe-bonus-dialog.png)

The title's own **"Digital Deluxe Version bonus received."** dialog, t≈90 s and held to the end of
the run — the item list (`Big Dipper`, `Black Shiba Skin`, `Special Hat`, `Amber`,
`Crop Seedling: Tomato`, `Crop Seedling: Wheat`, …), a scrollbar, the orange
`*You can view these items through the Bed screen.` note and a styled **OK** button, all
composited correctly. 6,452 distinct colours over 1,707,086 non-black pixels.

That is real content from the live renderer, so it is rung 1. It is **not** rung 2: no title screen
has been observed, and the dialog is an interactive surface the title reached, not a title screen.

Note what the dialog also demonstrates: the add-content path answered from **local inventory** and
the title granted the SKU's own bonus items. Nothing was blanket-approved to get here.

### The A/B, stated as an A/B

Same frontend, same route, same 5 s sampling interval, same build, one variable:

| arm | t=45 s | t=90 s onward | `pixel-distinct` | producing rate |
| --- | --- | --- | --- | --- |
| default (lever off) | flat white `96a57e22`, 1 colour | flat white, 1 colour | 3 of 48 | 4.7 fps, 1% active |
| `PROSPER_CB_EFC_NO_COLOR=1` | **GAME FREAK logo**, 256 colours | **Deluxe dialog**, ~6,300 colours | 20 of 30 | 9.2 fps, 56% active |

Replicated: **2 of 2** unrouted lever runs reach the logo, and their t=35 s frames are
**byte-identical** (`pixel_crc32 = d951f1d2` in both).

**The lever also correlates with a driver crash, and that is recorded here rather than buried in the
issue.** Across five runs: 0 faults in 2 default runs (240 s, 275 s), **2 faults in 3 lever runs** —
a SIGSEGV inside `libvulkan_radeon.so` on `AgcSubmissionTh` with `rax=0`, identical instruction
bytes and identical prosper frames both times ([#2915](https://github.com/mattias800/prosper/issues/2915)).
**Neither the cause nor even the association is established, and the first draft of this paragraph
overstated both.** Two things cut against the obvious reading:

- **The arms are not one variable at the GPU.** The lever arm sits at 9.2 fps / 56% active against
  4.7 fps / 1% — roughly **12x** more real GPU work. "Correlates with the lever" and "correlates
  with actually rendering" are not separated by five runs, and the second is at least as good an
  explanation of a driver fault.
- **Zeroed write masks are not a new state for this code.** `PROSPER_LEGACY_CB_DISABLE_MASK`
  (#1724) performs the *identical* three assignments ten lines above, on a different mode
  population. If all-zero masks alone produced a null at pipeline creation, the hazard would not be
  specific to this lever — which is an argument from the code path being identical, not from a
  clean run against that lever.

A cheap discriminator, if it is worth one run: `PROSPER_LEGACY_CB_DISABLE_MASK=1` on this title,
which zeroes masks on a different population and should crash too if the zeroing is the mechanism.

So: treat the lever as a measuring instrument with a **suspected** defect, not as a candidate fix.
None of this weakens the A/B — the run that produced both screenshots completed 30 of 30 samples
`guest=running status=ok`, and a crash cannot synthesise a correctly composited 4K UI.

The lever is declared in `prosper/src/gpu/state/render_state.cpp` and is **default off and not a
shipping mode**. Read [#1706](https://github.com/mattias800/prosper/issues/1706) before reading the
positive arm as "EFC is the operation at fault": prosper's decoded `MODE` is not per-draw-trustworthy,
because a utility sequence's operation bits stay latched onto later ordinary draws. So the lever
suppresses colour on every draw that **decodes** as EFC, which is a superset of the draws that
**are** one. What the arm establishes is that **the population of draws prosper decodes as EFC is
what flattens this composite** — a localisation, not an identification.

**That asymmetry is the argument that should stop anyone flipping the default, and it is stronger
than the three practical reasons above.** A positive arm here can only ever localise a *population*
of draws; identifying the *operation* is a different claim, and this lever is structurally unable to
make it. So no amount of good-looking output from the positive arm justifies making it the default —
the evidence it produces is the wrong shape for that decision, not merely insufficient in quantity.

### First contact: rung 0, a flat white 4K clear

First contact, 2026-08-22, `tools/screenshot`, Linux/RADV, **default launch**
(`PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct`, no pad, no throttle), 240 s / 48 samples:

- every sample `3840x2160`, `distinct_rgb_colors = 1`, every pixel `RGBA(255,255,255,255)`,
  `pixel_crc32 = 96a57e22` from t=5 s to t=240 s — `pixel-distinct=1` over the whole run;
- the guest was **healthy** throughout: `guest=running status=ok`, `frame_seq` 177 → 12,110,
  `present_count` 178 → 6,000-odd, 12,120 published frames;
- and it was **drawing**: 301 submits of 22 draw items each, real `[t#]` texture descriptors,
  1920x1080 and 2048x2048 surfaces, `[restab]` constant buffers with sensible values.

So this was never a "nothing happens" boot. It was a boot whose output never reached the screen.

### What was actually wrong

`PROSPER_DROPPED_DRAW_CENSUS=1` keyed the loss by target:

```
[dropped-draw] 1024 draws discarded before the renderer
[dropped-draw]   target=0x406ba40000 reason=shader-recompile x309
[dropped-draw]   target=0x4069ca0000 reason=shader-recompile x308
[dropped-draw]   target=0x9fc0000000 reason=shader-recompile x202
[dropped-draw]   target=0x9fc2000000 reason=shader-recompile x202
```

`0x9fc0000000` and `0x9fc2000000` are this run's two guest **display buffers** — the boot log's
`Frame Buffer va range 9fc0000000 - a000000000`. **404 draws into the scanout, every one dropped,
all for the same reason**, which is exactly why the presented frame stayed at its clear colour.

Every dropped scanout draw named the same pixel shader. Under `PROSPER_DBG=1` the recompiler said
why:

```
[recompile-reject] sh=bfa00003/124 mode=unresolved-operand pc=N words=7e040ef9,00061402
  fmt=7 op=0x7 ... stage=fragment program=none role=terminal
```

`llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble` on those dwords:

```
v_cvt_u32_f32_sdwa v2, v2 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
v_cvt_u32_f32_sdwa v0, v3 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
```

— the ordinary "pack two converted integers into one VGPR" idiom, in the first four instructions of
a 124-dword shader. prosper already admitted **exactly this shape** for the signed sibling
`v_cvt_i32_f32` (VOP1 `0x08`) and not for the unsigned one (`0x07`); the two differ only in the
signedness of the 32-bit conversion, and the inserted half is its low 16 bits either way. Fixed by
teaching the decoder and the emitter the unsigned opcode, with an execution-differential test that
asserts the **preserved** half — a lowering that ignored the destination select would overwrite all
32 bits and still "recompile".

### After the recompiler fix: the scanout draws land, the default frame is still a clear

Same route, same duration, `PROSPER_DROPPED_DRAW_CENSUS=1`, current branch:

```
[dropped-draw] 256 draws discarded before the renderer
[dropped-draw]   target=0x406b560000 reason=shader-recompile x127
[dropped-draw]   target=0x40697c0000 reason=shader-recompile x126
[dropped-draw]   target=0x4060420000 reason=shader-recompile x1
[dropped-draw]   target=0x4067b80000 reason=shader-recompile x1
[dropped-draw]   target=0x4069920000 reason=shader-recompile x1
```

**No `0x9fc*` row survives.** Every remaining dropped draw goes to an offscreen surface, and they
all come from the one vertex shader below. The presented frame also stopped being invariant — it
goes white, **black** at t=10 s and t=15 s, then back to white and holds — which is what a scanout
draw that now executes looks like when the surface it reads is empty. Frame rate roughly doubled
(t=5 s at `frame_seq` 380 against 177 before), consistent with 400-odd fewer per-draw rejections
per second.

On the **default** route it is still a flat clear, so the default route is still rung 0: this fix
removed one blocker and exposed the next, and it did not produce a frame on its own. The frame comes
from the second half, below.

### Still open

- **`v_mbcnt_lo_u32_b32 v3, -1, 0` in a 401-dword vertex shader** (VOP3 `0x365`, words
  `d7650003,000100c1`) — **the whole remaining dropped-draw census.** The shader also carries
  `s_ff1_i32_b32` on a 64-bit mask and a `ds_*` LDS access, so it is a per-wave compaction
  vertex stage (NGG-shaped), not a stray instruction. The existing LDS wave model is
  compute-shaped — it needs a workgroup and LDS, which a vertex stage has neither of — so it does
  not carry over as written. See the linked issue.
  What its draws target on a live run: the two offscreen surfaces above; on the pre-fix run its
  pairs wrote `64x64` and `32x32` targets (`color0=0x406a680000/64x64`,
  `0x40688e0000/64x64`, `0x40599c0000/32x32`), i.e. atlas or tile-sized surfaces rather than the
  main scene colour. So **whether it is what stands between this title and a first frame is
  NOT established** — it is the only thing still being dropped, which is a different claim.
- **Ten distinct compute programs skipped** on a 240 s boot. All ten terminal reasons, with the
  rejected instruction as `llvm-mc -arch=amdgcn -mcpu=gfx1030 -disassemble` reads it:

  | reject mode | instruction | programs |
  | --- | --- | --- |
  | `cfg-recompile-reject` unresolved-operand | `s_and_b64 s[0:1], vcc, lit(0x0)` / `s_and_b64 vcc, vcc, lit(0x0)` | 2 |
  | `cfg-recompile-reject` unresolved-operand | `s_flbit_i32_b64 vcc_lo, s[14:15]` | 1 |
  | `cfg-recompile-reject` unresolved-operand | `v_max3_i32 v4, v2, v3, v1` / `v_max3_i32 v9, v11, v7, v15` | 2 |
  | `cfg-recompile-reject` unresolved-operand | `image_atomic_add v12, v[14:15], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D glc` | 1 |
  | `recompile-reject` unresolved-operand | `s_cbranch_execz 10` | 1 |
  | `recompile-reject` unresolved-operand | `s_cmp_eq_u32 1, vcc_lo` | 1 |
  | `recompile-reject` | `counted-loop enters with narrowed EXEC` | 2 |

  Note the mode: every one of these is **`unresolved-operand`**, not `unknown-encoding` — the
  reject site distinguishes the two, and `unresolved-operand` means the opcode itself was handled
  and a *source* could not be resolved. Four of the six instruction-shaped rows read or write a
  64-bit wave mask (`vcc`, `s[14:15]`), which is the wave-model family
  `docs/RECOMPILER_REMAINING.md` already tracks. Do **not** read this table as "seven missing
  opcodes". Ten is a small number for a Lumen/Nanite title and none of them is on the scanout path.
- **One 1920x1080 compute image with `fmt=21` is undecodable**, so its dispatch is skipped (#590).

## Boot health

`BOOT_COMPLETE` is not printed by this title's path, but the guest reaches UE's engine loop within
seconds and stays up for the whole run. 15 distinct unimplemented Sony functions on a 240 s boot,
all resolved against the PS5 3.20 library set:

| calls | library | NID | name |
| --- | --- | --- | --- |
| 6,143 | `libScePosix` | `Xs9hdiD7sAA` | `pthread_setschedparam` |
| 921 | `libSceNpWebApi2` | `3Tt9zL3tkoc` | `sceNpWebApi2CheckTimeout` |
| 177 | `libSceLibcInternalExt` | `qBS714-Jr3g` | *(not in the 3.20 set)* |
| 3 | `libScePlayGo` | `8-e7E989rCU` | `scePlayGoGetInstallChunkId` |
| 2 | `libScePlayGo` | `IfiN+-oeVWI` | `scePlayGoGetSupportedOptionalChunk` |
| 1 each | — | — | `sceCoredumpRegisterCoredumpHandler`, `sceHttp2CreateTemplate`, `sceNpWebApi2PushEventCreateHandle`, `sceVoiceQoSInit`, `sceNetResolverCreate`, `sceAcmContextCreate`, `sceSystemServiceDisableNoticeScreenSkipFlagAutoSet`, `sceSystemServiceSetNoticeScreenSkipFlag`, and `libSceAudioOut2::TViD1EZXkNI` / `libkernel::tU5e3f9gSiU` (neither in the 3.20 set) |

None of these is a wall: the two high-count ones are a thread-priority setter and a poll, and the
dispatcher's default `0` is a plausible answer for both. They are listed so a later lane does not
have to re-measure them.

## Ruled out

- **"The title is not rendering / the guest is stuck" — false.** The first 240 s run showed a flat
  white 4K frame and nothing else, which reads as a dead boot. It is not: `guest=running status=ok`,
  `frame_seq` 177 → 12,110, 301 submits of 22 draw items each, and real texture/constant-buffer
  descriptors throughout. The output was being *discarded* at the executor, not never produced.
  (This is the general shape of instrument trap: a uniform composite is evidence about the
  presented surface, never about the guest.)
- **"The `uecommandline.txt` spelling identifies the first UE5 title in the corpus" — false.**
  `PPSA08804` ships the same file name and the same IoStore TOC version 6, and
  `COMPATIBILITY.md` already records it as Unreal Engine 5. The engine generation had to be
  established from engine-subsystem strings and the container version, not from the file name.
- **"`Project file not found: ../../../beastofreincarnation/beastofreincarnation.uproject` in the
  boot log is the blocker" — false.** UE prints this from a staged build whose `.uproject` is not
  packaged and then proceeds; the title goes on to load its IoStore containers, create AGC command
  buffers and issue thousands of draws after it. It is noise, and it is the first alarming-looking
  line in the log, so it is recorded here to stop it being chased.
- **"`sdwa=6/0/6/6` on the fragment reject means the SDWA wrapper is inert and the opcode is
  missing" — false, and it cost time.** Those are the *struct defaults*: the decoder only fills the
  select fields on the branches that accept a shape, so a refused instruction reports selects it
  does not have. The real control word `0x00061402` is `WORD_0 / UNUSED_PRESERVE / DWORD`, and the
  opcode was implemented all along. See the linked diagnostic issue.
- **A shader-recompile hypothesis derived from `shader_inspect`'s offline `generic-unsupported`
  list is not the live reject.** On this title the offline pass flagged `SOP1 0x10`, `SOP2 0x32`,
  `VOP1 0x1` and `VOP2 0x25` in the failing vertex shader; the live terminal reason was VOP3
  `0x365`. Take the instruction from the live `[recompile-reject]` line, as
  `tools/shader_inspect/README.md` § "`--stage` cannot prove a shader is unsupported" says.
