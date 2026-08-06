# GRIS, Sonic Origins, and Space Adventure Cobra bring-up

Validated on Linux on 2026-07-25, with GRIS opening gameplay and Cobra tutorial combat revalidated
on 2026-07-31. This note records the reproducible evidence for issue #1356. Raw PCM, verbose logs,
and GPU captures are local diagnostics and are intentionally not committed.

## Result matrix

| Title | Revision | Visual milestone | Audio evidence |
| --- | --- | --- | --- |
| GRIS (`PPSA09804`) | 01.001.000 | Native 1920×1080 opening gameplay with scripted movement | CLEAN on current master over the first 35 seconds: `rms=0.0082`, `peak=0.1173`, duplicated grains 0.0% |
| Sonic Origins (`PPSA05325`) | Complete Sonic Origins Plus 02.002.000 base+update, four DLC payloads with mount records | Black startup loop; root cause remains open (#1905) | AudioOut2 port 17 runs, but guest PCM remains zero in the black state |
| Space Adventure Cobra — The Awakening (`PPSA17337`) | 01.004.000 | Native 1920×1080 tutorial combat with scripted progression | CLEAN, `rms=0.0436`, `peak=0.1880`, duplicated grains 0.0% |

## Visual evidence

### GRIS

![GRIS — New Game title](screenshots/issue-1356-gris-title.png)

Route: `scripts/gris/reach-title-screen.pad`. The title appears without input; the comments-only
route keeps a neutral scripted controller connected and prevents an evidence run from selecting
**NEW GAME**.

![GRIS — opening gameplay](../../assets/screenshots/gris.png)

Route: `scripts/gris/reach-first-gameplay.pad`. Poll-safe Cross edges select the default New Game
entry and cover the opening sequence's timing variation. Right is held from 78 through 150 seconds.
On exact master `2562269711f89b59f7f3038eab1bb4dcf8468b52`, the direct native `screenshot` route
reached the controllable ink-ground scene around 130 seconds: the character moved and animated under
the scripted input, then settled after the route returned to neutral. The retained 170-second run
contained 85 source-distinct and 85 pixel-distinct unmodified 1920×1080 frontend frames.

### Space Adventure Cobra — The Awakening

![Space Adventure Cobra — The Awakening title](screenshots/issue-1356-space-adventure-cobra-title.png)

![Space Adventure Cobra — tutorial combat](../../assets/screenshots/space-adventure-cobra.png)

Route: `scripts/cobra/reach-title-or-gameplay.pad`.

The route combines early flip-anchored Cross pulses with wall-clock hold-Square actions for the
opening movies and later Cross pulses for dialogue. A fresh-save, default-configuration run observed
Square at 60, 70, 80, 90, and 100 seconds, reached the readable `Press R2 to shoot with the
Psychogun.` tutorial at 85 seconds, and continued through full-color combat for the complete
180-second bound. All 36 captures came from distinct composited source frames; 33 were pixel-distinct.

Write watches remained fully enabled in that run: 63 registrations covered 65,504 pages and handled
8,447 write faults without a worker crash. The fault path temporarily restores host `%fs` around the
host mutex and `mprotect`, then restores guest `%fs` before resuming the interrupted store. A
production-handler regression test locks that TLS boundary. The normal screenshot frontend produced
the committed images at 1920×1080 with no debug shader, resource override, render scaling, sparse
rendering, warm-up shortcut, or write-watch escape hatch.

## Reproduction

Run from the repository root and point each command at a legally obtained app directory:

```bash
PROSPER_PAD_SCRIPT=@prosper/scripts/gris/reach-title-screen.pad \
  prosper/build-linux/screenshot /path/PPSA09804-app0 \
  --seconds 1 --count 35 --timeout 90 --out "$HOME/prosper-artifacts/gris-shots"

PROSPER_PAD_SCRIPT=@prosper/scripts/gris/reach-first-gameplay.pad \
  prosper/build-linux/screenshot /path/PPSA09804-app0 \
  --seconds 2 --count 85 --timeout 210 --out "$HOME/prosper-artifacts/gris-gameplay-shots"

PROSPER_SAVEDATA_DIR="$HOME/prosper-artifacts/cobra-savedata" \
PROSPER_SAVE0="$HOME/prosper-artifacts/cobra-save0" \
PROSPER_PAD_SCRIPT=@prosper/scripts/cobra/reach-title-or-gameplay.pad \
  prosper/build-linux/screenshot /path/PPSA17337-app0 \
  --seconds 5 --count 36 --timeout 195 --out "$HOME/prosper-artifacts/cobra-shots"
```

Capture final mixed PCM with `PROSPER_AUDIO_DUMP="$HOME/prosper-artifacts/<title>"`. GRIS's title is silent, so use
`scripts/gris/reach-first-gameplay.pad` for its audio exercise; the title-screen route remains the
visual evidence route. Analyze the active port using its logged format; GRIS uses stereo float32 on
port 1 (its auxiliary port 3 remained silent):

```bash
python3 prosper/tools/audio_analyze.py "$HOME/prosper-artifacts/gris.port1.raw" \
  --fmt f32 --channels 2 --rate 48000 --tail-seconds 30
```

The 2026-07-25 validated GRIS verdict was:

```text
CLEAN: corr(block 1024f)=+0.095 neighbor-max=+0.047 spike=+0.048 (threshold 0.35) dup-grains=0.0% rms=0.1800 peak=1.2689
```

The 2026-07-31 exact-master gameplay rerun opened the same stereo float32 port 1 and auxiliary
stereo s16 port 3. Port 1's first 35 seconds were non-zero and passed the repetition check:

```text
CLEAN: corr(block 1024f)=+0.008 neighbor-max=+0.012 spike=-0.004 (threshold 0.35) dup-grains=0.0% rms=0.0082 peak=0.1173
```

The final 30 seconds of the 170-second route were silent and therefore produced no repetition
verdict. That quiet tail is not evidence that the complete capture or output path was silent.

The validated Cobra verdict was:

```text
CLEAN: corr(block 1024f)=-0.049 neighbor-max=+0.032 spike=-0.081 (threshold 0.35) dup-grains=0.0% rms=0.0436 peak=0.1880
```

## Snapshot guards (rung 6)

Both titles were taken to rung 6 on 2026-08-01 with reviewed content guards, using the routes
recorded above unchanged:

- `gris-gameplay` — window 155-235 s, after the route's last input, so the guarded span is the
  settled ink-ground idle scene rather than the 78-150 s movement phase.
- `cobra-gameplay` — window 170-198 s, after the route's last input at 124 s, so the camera is held.

Both put the whole discriminating load on SSIM. GRIS is blind on both numeric axes at once (its title
screen is 17x richer in colour than its gameplay, and non-black coverage is exactly 1.0000 for logos,
title, intro and gameplay alike, because it is a bright paper page with no true black). Cobra inverts
the usual coverage argument: its gameplay coverage (0.8152-0.9037) is *lower* than its menus
(0.9908-1.0000), because the bottom of the combat frame is the dark underside of the walkway, so its
coverage floor is deliberately left low. Details and thresholds are in each entry's `_note` in
`tools/snapshot/snapshots.json`.

One throughput observation worth recording, since no snapshot guard measures frame time: on the same
machine and at the same scale, GRIS runs at roughly 300 presents/s while **Cobra manages only 6-12
(median 9)**. Cobra renders correctly and passes its guard; it is simply very slow, and nothing in
the automated suite would say so.

## Ruled out

One line per dead hypothesis, the evidence that killed it, and where that evidence lives.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| `targetContentVersion: 02.001.000` proves the supplied directory is update-only or incomplete | **Falsified.** The assembled 02.002.000 directory identifies itself as Sonic Origins Plus, contains the base executable/content and all four classic RSDK games, and includes four installed DLC payloads with mount records. The target version is update lineage, not a directory-completeness flag. The two loose UI misses remain real, but they are not grounds for rejecting the game dump. | #1905, tracker #1871, this doc |
| The two early `/app0/raw/ui/...` ENOENT results are a terminal archive, update-overlay, mount, or path-resolution blocker | **Falsified as the asserted startup blocker.** On current master, both failures are handled before entitlement enumeration; the guest then makes 142 successful APR resolve calls across 38 other unique UI, font, language, and audio paths without a fault. A later DLC mount cannot explain the earlier absolute base-app misses. The absent resources may still affect an individual visual if used conditionally, but the black-frame root cause remains open. | #1905, tracker #1871 |
| A PS5 `launchActivity` Game Intent routes around the current black startup state | **Falsified.** The update declares `launchActivity` support and ships the classic RSDK files, and the guest genuinely receives and consumes an exact `TITLE_SONIC_1_CLASSIC` intent — its `activityId` property is read and recognized. It still remains black and does not open `raw/retro/Sonic1u.rsdk`. This proves the activity route is insufficient, not that the handled UI misses cause the black frame. Truthful default no-intent behaviour is preserved. | this doc, #1905 |
| Consuming the four installed add-content records naturally advances Sonic into an entitlement-key or mount path and changes the startup state | **Falsified at the `f72d8f0` black-loop boot depth.** After #1916, a valid routed 60 s CPU-only arm makes the count query and a real four-entry list call, then no individual info, key, AppContent, or mount call. A native/full-cadence 90 s renderer arm remains byte-identical black across 18 direct samples and port 17 remains silent. The result is scoped to this boot state; re-run it after any fix that advances the guest. | #1905, #1916 |
| Sonic is black because it submits no GPU work or a shader/resource stage fails realization | **Falsified for the captured present-20 frame.** A deterministic whole-frame bundle contains 36 realized operations across submits 447-468; every extracted capsule reports `failed=0` and no failure diagnostics, its temporal closure is complete, and offline replay succeeds. The complete 3840x2160 result is nevertheless uniformly black. The first live `Vulkan render FAILED` line occurs earlier on empty submit 448, so it is a missing presentable scanout result rather than a failed Vulkan operation. | #1905 |
| Submit 463's operation-5 black target is caused solely by operation 2 sampling stale zero guest bytes for its Uint32 view of the persistent depth plane | **Falsified, while exposing and fixing a real generic interop defect.** The authoritative D32S8 depth plane at `0x2064ae0000` is uniformly `0.5`, but the old Uint32 compute path read its stale zero guest backing. A GPU-only raw-bit bridge changes operation 2's 3840x2160 binding-7 output from all zero (`ccc433ff6d980383`) to the exact depth-plane bits (`1d0ffd6fc0338383`). Operation 5 still produces the identical black `26ed8b6191338383` target with a complete six-operation closure, so the stale depth alias was real but not sufficient to explain this frame's black composite. | #1905, this doc |
| Operation 5 turns black because one of its large shader/resource paths fails or reads the wrong downstream input | **Falsified for the captured frame.** The exact dependency graph proves operation 1 writes operation 5's binding 119 at `0x204ec40000`. Operation 1's complete four-dword fragment program unconditionally exports zero to every RGBA channel. Operation 5 samples that texture first, computes `(uint(alpha * 255) & 64)`, and bypasses its large body when the bit is clear; its fallback exports binding 32 words 84-87, which are `(0,0,0,1)`. The black result is therefore selected by the guest shader before the other scene resources can affect it, not produced by a failed operation-5 translation. The remaining blocker is upstream of this submit's GPU program. | #1905, this doc |
| The frontend is stalled on a Sony call prosper has not implemented (the classic "guest polls a missing stub" wall) | **Falsified.** A 185 s routed CPU-only arm on `9dcb6c4b` with `PROSPER_PROGRESS=5 PROSPER_PROGRESS_UNIMPL=1` dumps the unimplemented-NID call-count table 38 times and every dump reads `(0 distinct unimplemented functions)`. The table is the per-call-count form specifically built to expose a *polled* missing stub, so a deduped first-seen log cannot be hiding one. | #1905, this doc |
| The frontend is waiting on a Sony service, an incomplete asset load, or an APR read that never completes | **Falsified for the post-resident state.** With `PROSPER_FILELOG=1` the whole resident set (`param_tech.rfl`, `NeedleShader.pac`, `ui_resident`/`ui_text_texture`/`scalablefont`, the CRI ACB/AWB banks, `rfl_resident.pac`, all twelve `text_common_*`, `segafont.pac`) resolves and reads OK and the **last file operation completes before the t=5 s heartbeat**; the following 180 s contain zero file activity. Under `PROSPER_SVCLOG=1` the only per-frame service traffic is `sceUserServiceGetEvent` (4,264 calls / 60 s) plus `scePadReadState`; `sceSaveDataInitialize3` and the entitlement list call happen once each and no mount, key or trophy call ever blocks. | #1905, this doc |
| The black startup state is input-gated and previous arms simply never pressed the button that dismisses it | **Falsified — and the suspicion was well founded, because the committed route really is short.** `scripts/sonic/reach-title-or-gameplay.pad` ends at **pad-read 404**, and the route clock is pad reads, not seconds, so at the ~61 reads/s of a CPU-only arm it delivers its last edge at **t≈6.6 s** and every previous CPU-only run then ran input-free. A 12,000-read probe route (`scripts/sonic/long-input-probe.pad`, ~400 Cross pulses with Options every tenth) changes nothing: the 200 s CPU-only arm and a 160 s live-renderer arm both plateau on a **byte-identical resolved-path set**, and no run in the series ever requests `ui_mainmenu*.pac`, `stage_title.pac`, `gedit_stage_title.pac`, a `.rsdk` or a `.usm`. | #1905, this doc |
| The injected Unity guest argv (`PROSPER_GUEST_ARGS=-force-gfx-direct`, which `screenshot` defaults on) puts this non-Unity engine on a different path | **Falsified.** Two 30 s CPU-only arms differing only in that variable resolve a **byte-identical 40-path set** and flip at the same rate (1,534 vs 1,533 flips), with the same six distinct APR misses. The A/B is self-validating: it compares the resolved-path sets directly, so "no difference" is a measurement rather than an absence. | #1905, this doc |
| The absent `raw/ui/ui_startup.pac` and `raw/ui/rpl_texture/` group prove the supplied `raw/` tree is a truncated repack | **Falsified as an integrity claim.** The tree is internally consistent: `ui_museum_item_texture_l_art1..214` is contiguous with no gaps, all seven multi-language `ui_*` families are complete at 12/12 languages, and `raw/ui` contains no subdirectories at all. The eboot's name pool contains exactly one base-game `ui/` entry with no file (`ui/ui_startup`) plus the six region rating textures (`ui_title_cero`/`esrb_e10`/`esrb_rp`/`nocopy`/`pegi`/`titile_healthy`) — one functional group, requested back to back, consistent with an optional region/legal asset set this build does not ship. Their absence is therefore not evidence of a damaged dump, and remains insufficient to explain the black frame. **One leg of the original argument is withdrawn — see the `AMPRIDX3` row below. A second leg is now contradicted by measurement: `ui_startup.pac` is the sole absent member of a 34-entry resident resource list whose other 33 members all ship (`tested=34 found=33 missing=1`), which is not the shape of an optional group — see the boot-call census section. The tree-consistency and 12/12-language legs still stand, and the file's absence is still not shown to cause the black frame.** | #1905, this doc |
| `ampr_emu.index` is the application's own `AMPRIDX3` file index, so a name absent from it is a name this build does not ship | **Falsified — misattribution.** The index is *repack scaffolding*, not the game's: the eboot contains no `AMPRIDX` string anywhere, prosper never opens the file (no reference in `src/` or `tools/`), and it sits beside `dlc_emu.ini`, four `fakelib/*.sprx` stubs and a `_DUPLEX_/duplex.nfo` — and its entries include `/app0/dlcN/dlc/dlcN/...` mount paths a shipped app0 index could not contain. It therefore records what the repacker's tool found on disk, which cannot establish what the application expects. This does not resurrect the truncated-repack claim (the contiguity and language-completeness legs stand on their own); it removes one leg that was never evidence. | #1905, this doc |
| Sonic is black because prosper renders content and then loses it in the present path (the #1990 publish wall) | **The wall is real and it is not the cause.** On `c77c66b4` Sonic reproduces the Frontiers signature exactly: `frame_seq` freezes at **1** while `present_count` climbs (21 → 49 → 76 → … across eight 6 s samples), i.e. exactly one frame is ever published. With #1990 applied, `frame_seq` advances again (110 / 300 / 490 / 669 / 844 across 8 s samples) and every pixel is unchanged black. #1990's own report names why: `[rtt] PRESENT SOURCE EXTENT MISMATCH: no pass produced a 3840x2160 (33177600-byte) present source — px_front=none px_vo=none px_last=none` on submit after submit, with one wrong-extent `px_last=1920x1080` candidate — the frame that poisoned the retained slot and made the freeze permanent. So Sonic needs #1990, and #1990 does not make it render. | #1905, #1990, #1986 |
| The 3840x2160 `R16G16B16A16_SFLOAT` scene target holds full-frame content that the publish path discards (`px_nonblack=8294400`) | **Falsified — that number was the instrument, and it is fixed.** `PROSPER_PASS_LOG`'s counter stepped 4 bytes and tested `p`, `p+1`, `p+2` — never `p+3` — as if every target were 8-bit RGBA, so over an 8-byte FP16 texel it read `{R_lo, R_hi, G_lo}` then `{B_lo, B_hi, A_lo}`. The target's texels are **RGB bit-zero with alpha `0x3c05` (1.00488)**, measured byte-exactly, and that alpha's low mantissa byte `0x05` sits at texel byte 6 — the second group's `p+2` — so exactly one of the two groups per texel counted and the line printed precisely `w*h`. An alpha of exactly 1.0 would have counted **zero** (`0x3c00`'s only non-zero byte is the skipped `p+3`), so the number was never a property of the image, only of its bit layout. Counting through `inspection_rgba8` reports **0** for the same pass. | #1905, this PR |
| Every GPU target is black only in the retained offline replay, so the live render may still hold content | **Falsified on the live path, at bit level.** A `PROSPER_DUMP_PERSISTENT` census (deliberately *not* in the `live_gpu_targets` disable list, so it observes the normal persistent-render path) dumps all **10** persistent color targets across three consecutive live submits and now reports the *pre*-conversion bytes beside the converted count, because "black" and "empty" are different findings. Every target has `rgb_nonblack=0` except a 256x256 grey utility texture (`0x2076f60000`) and **two** texels of a 960x540 bloom target. The raw bytes name what the colour count cannot: the final 4K RGBA8 composite `0x2010870000` is `00 00 00 ff` per texel (opaque black, 8,294,400 of 33,177,600 bytes non-zero — the alpha byte only); the 4K FP16 scene target `0x20168f0000` is `00 00 00 00 00 00 05 3c` (RGB bit-zero, alpha 1.00488); the mask target `0x204ec40000` and the 1080p `B10G11R11` target are **entirely zero**. So the frame is black *by content*, not merely by conversion — the guest-programmed-black finding is a property of the title's live frame, not of the replay or of a clamp. | #1905, this doc |
| The guest's own scanout memory holds a frame prosper simply fails to publish | **Falsified.** `PROSPER_DUMP_SCANOUT` dumps ten raw flipped guest buffers (both registered VideoOut buffers, 33,177,600 bytes each); every sampled pixel of every dump is `(0,0,0)`. The guest's last step into the scanout is a *compute dispatch* (`0x2010870000 -> 0x2012850000`), and neither its output nor any draw reaches those addresses with content. | #1905, this doc |
| The `dlc3`-prefixed CRI wave-bank APR misses leave the Atom banks without waveform data (the silent-audio candidate, #1993) | **Falsified as the silent-audio cause — the bank loads on the base-content-root retry.** On `c3614f51` the pattern is exact and repeats for every bank: `[apr] resolve MISS /app0/dlc3/sound/X.awb`, then immediately `[apr] content-root fallback /app0/sound/X.awb -> /app0/raw/sound/X.awb` and `[apr] resolve /app0/sound/X.awb -> id=12 size=28352512`. **Seven** banks do this, not the four #1993 recorded (`STH1_music`, `STH2_music`, `SCD_music`, `HITE_music`, `HITE_missions`, `Music03_S3K`, `Music09_Museum`). **Why the guest asks the DLC prefix first is NOT settled here**, and the difference matters for where a fix would belong: an add-content override search and #1993's own competing reading — a single engine-global content root that the DLC mount overwrote — both produce this log. The one detail that discriminates leans toward the latter: all three repeats target the same `dlc3` prefix while **four** DLC mount, whereas a genuine override search would probe each mount once. It does not matter for the audio path, because the bank resolves either way, so do not add a `/app0/dlc*/` -> `/app0/raw/` alias on this evidence. The silent AudioOut2 port has another cause. | #1905, #1993, this doc |
| The absent `raw/ui/rpl_texture/ui_title_nocopy.dds` request says something about the dump | **Falsified — it was prosper's own wrong answer, and it is gone.** The title derives its **region** from the app's own declaration at `eboot+0x51230c`: it calls `sceAppContentAppParamGetInt(USER_DEFINED_PARAM_1)` into a pre-zeroed slot and branches `2 -> "EU"`, `1 -> "US"`, anything else -> `"JP"`, with JP additionally setting a flag the other two clear. That branch mapping is **read off the disassembly** at `eboot+0x51231c..0x512357` (the two-character codes are the immediates `0x5545`/`0x5355`/`0x504a`), not measured. This application declares `userDefinedParam1: 2` and `contentId EP0177-PPSA05325_00-…`, so the correct answer is **EU**, and that is what the handler now writes — measured directly rather than inferred, by breaking on `prosper::s_appcontent_int` and reading its out-slot after the return: `paramId=1 ret=0 value=2` (the neighbouring `sceSystemServiceParamGetInt(1)` reads `value=1`, en-US). Before #2003 `s_appcontent_int` was `*(int32_t*)PW(a1) = (a0 == 0) ? 3 : 0`, so this same call wrote `0`, which the branch above maps to **JP** — i.e. every earlier Sonic arm in this document is best read as having run the title as Japanese. That last step is an inference from the disassembly plus the disappearance of the JP-only request, not an A/B against the old binary. `ui_title_nocopy` / `ui_title_cero` / `ui_titile_healthy` are the JP legal-and-rating set (`ui_title_pegi` is the EU one). On current master the request is gone and `ui_startup.pac` is the only remaining UI miss. Nothing about the dump changed; prosper's answer did. | #1905, #2003, this doc |
| The absent `raw/ui/ui_startup.pac` parks the resource loader (the "handled ENOENT is still a stalled state machine" reading) | **Falsified at the consumer, not just at the resolver.** The widened `apr_miss_callsite` annotation names the frame: `ra4=eboot+0x990235`. `eboot+0x98fa80` is a thin `exists()` predicate over the APR resolve (`call 0xf0600; test eax,eax; sete al`), reached as a virtual call `call QWORD PTR [rax+0x88]` at `eboot+0x99022f`. The **not-found** branch at `eboot+0x990239` releases its temporaries and jumps to `eboot+0x990120`, which is a loop head: `add r14,0x18; cmp r12,r14; je <exit>`. So a missing entry advances the cursor and the loop continues — the miss is skipped by construction, with no error path and no wait. Whether the *startup scene* later needs what that package would have contained is a separate question this does not answer. | #1905, this doc |
| Sonic skipped initialization it still needed because `sceSysmoduleIsLoaded` used to claim every module was loaded (#2021/#2002), and the CRI Mana group is parked for that reason | **Falsified.** Sonic does import `fMP5NHUOaMk` and does query it — a `beeff2ab` boot logs `[sysmodule] IsLoaded -> UNLOADED` for ids `0x115`, `0x105` and `0x110` — but all **five** static call sites are the benign `if (not loaded) { LoadModule(id); mark-that-we-own-it; }` idiom, verified by disassembly (`eboot+0x8c4f0e`, `+0x9273e5`, `+0x92740f`, and the two finalize-side sites at `+0x927699`/`+0x9276c5`). The load then succeeds, so the only state #2021 changes is the ownership bit the object clears at finalize (`or BYTE [rbx+0x150],1` / `,2`; `or BYTE [r14+0x80],0x80`). The whole-boot result is unchanged: same 40-path resolved set, no `.usm`, no `.rsdk`, 10/10 black samples. | #1905, #2021, this doc |
| An unsupported shader op, storage format or dispatch is silently skipped, and that is what collapses the composite | **Falsified on the live path, with a positive control.** Two 18 s live-renderer arms differing only in `PROSPER_DBG` match **zero** lines for `recompile-reject` (covering `[recompile-reject]`, `[cfg-…]` and `[exec-…]`) and zero for the literal pattern **`\[compute\] skip`** — the emitter writes `skip`, not `skipped` (`src/gpu/gpu_executor.cpp`: `skip unregistered/unreadable`, `skip unsupported`, `skip invalid descriptor contract`), so a grep for `skipped` would have returned zero by construction. The recompiler zero is a measurement rather than a void window because the `DBG=1` arm gains two prefixes the `DBG=0` arm does not have at all — `[compute-cfg]` ×6 and `[graphics-cfg]` ×6, emitted from the same translation unit under the same gate — and no line in either arm reports a non-zero `cf_rejected`. The three `[compute] skip` sites are **not** `DBG`-gated, so their zero needs no control. Note the twelve CFG lines are the count of *branching* shaders (they are also conditioned on `cfg_branches != 0`), so they are the channel control; the zero rejects carry the conclusion. | #1905, this doc |
| The 22 draws per flip are what survives prosper dropping the rest of a larger frame at *translation* | **Falsified.** `state.draws` is filled by the PM4 decoder (`src/gpu/command_processor.cpp`) and printed by `progress_heartbeat` (`src/hle/hle_agc.cpp`), so the counter is pre-translation by construction; a `PROSPER_NO_COMPUTE=1` arm — where nothing is translated and therefore nothing can be rejected — reports the identical per-flip figures (66,988 draws and 42,734 dispatches over 3,044 flips at t=50 s: 22.01 and 14.04), which is the consistency check on that reading rather than its proof. **Scope it exactly:** this closes *translation* loss, not *decode* loss. A draw inside a PM4 packet prosper never decodes is invisible to this counter too, and the `walk=` field that would bound it is on the `[agc] <who> #N` form, not the `[agc] SubmitDcb #N: %u dwords -> %zu packets applied` form this title takes (266 of 266 lines in a 10 s `PROSPER_GFXLOG` arm). That bound is still open. | #1905, this doc |

## Sonic Origins dump audit

Sonic reaches a stable frontend frame loop with all decoded GPU operations realized, a connected
scripted controller, CRI Atom banks loaded, and AudioOut2 advancing. A 100-second `boot_trace` run
observed more than 300 flips and exercised the full route without a guest fault. Prosper does not yet
reach an authentic title screen from this complete application.

The application metadata records:

```text
contentVersion:       02.002.000
targetContentVersion: 02.001.000
originContentVersion: 01.000.000
```

`targetContentVersion` describes the installed update's lineage. It must not be read as proof that
the merged application directory contains no base title. The inventory contains Sonic 1, Sonic 2,
Sonic 3 & Knuckles, and Sonic CD payloads plus four installed DLC payloads with mount records.

With `PROSPER_FILELOG=1`, the complete unresolved-path set in a bounded current-master CPU run is:

```text
/app0/raw/ui/ui_startup.pac
/app0/raw/ui/rpl_texture/ui_title_nocopy.dds
```

Neither path exists as a loose file under `PPSA05325-app0`, but ENOENT is handled. Immediately after
the second failure the guest successfully resolves `ui_resident.pac`, `ui_text_texture.pac`,
`scalablefont.pac`, its CRI banks, every common-language PAC, and the bitmap font. The 35-second run
records 148 successful resolve calls across 40 unique paths; 142 calls across 38 unique paths occur
after the second miss, with no guest fault. Entitlement enumeration happens later, so the later
entitlement/DLC-mount path cannot explain these earlier absolute `/app0/raw/...` requests; the trace
does not establish that an archive or update overlay should have supplied them. Installed-DLC
enumeration landed separately in #1916, but it is not this temporal cause.

The game still publishes black scanouts and its correctly initialized AudioOut2 buffers remain zero.
That root cause is open: the absent resources may still affect a visual if the guest uses them
conditionally, but the trace does not support treating their handled absence as the startup blocker.
Do not alias another PAC/DDS or use a black frame as a success screenshot.

### Post-add-content frame localization

On exact master `f72d8f0` after #1916, a valid 60-second CPU-only route consumed all four installed
records: the guest made the zero-capacity count query and followed it with a real `listNum=4` call.
It made no subsequent individual-info, entitlement-key, AppContent, or mount call in that bounded
boot. A separate direct native 3840x2160/full-cadence renderer run remained black for all 18 samples
through guest present 456, while stereo float32 port 17 remained mathematically silent. The four-entry
enumeration is therefore no longer hidden, but it does not by itself change the current startup state.

The renderer path is active rather than empty. A whole-frame bundle scheduled at guest present 20
retains 22 submits (`447..468`) and 36 fully realized draw/compute operations. Offline replay resolves
both temporal edges, uses one captured boundary seed, leaves no bounded or unresolved frontier, and
still produces a one-colour black 3840x2160 image. Every extracted operation capsule has zero failed
stages. The first live `Vulkan render FAILED` line belongs to empty submit 448, before the later
draw-carrying submissions; here it means that no cached scanout pixels were selected, not that a Vulkan
operation failed. The capture names the current front buffer as absent from the live RTT cache, while the
complete replay endpoint is also black. Runtime intermediate-output evidence is still required to decide
whether repairing that presentation miss alone could reveal content.

Manifest-only operation and dependency inspection identifies candidates but does not yet localize the
first black-producing operation. Submit 463's final draw (operation 5 / `draw[3]`) is a full-resolution
composite into `0x20168f0000`. The pre-submit capture payloads for its five 3840x2160 texture bindings
(`b118..b122`) are zero. The graph records `b118` and `b119` as outputs of operations 2 and 1 in that same
submit, so their values when operation 5 consumes them are unknown; `b120..b122` are external leaves and
remain candidate inputs. Two smaller utility/static textures (`b115` and `b123`) are genuinely nonzero,
so this is not a blanket-zero capture. The pre-submit payloads for the two large scene-data constant
buffers (`b34` and `b35`) are also zero.

Runtime inspection completed that first discriminator. Operation 1's output is transparent black.
Operation 2 reads binding 6 at `0x2064ae0000` through a one-component Uint32 T#, while the
authoritative renderer-owned D32S8 plane at that address contains `0.5` in every texel. The old path
did not offer the persistent depth import for an integer descriptor and detiled the stale all-zero
guest allocation instead. The generic fix preserves the raw D32 bit payload entirely on the shared
Vulkan device: depth image to transfer buffer to an owned `R32_UINT` sampled image, with the renderer
layout restored afterward. The production regression contrasts the same stale backing at a non-DS
address against the exact DS address, requires `0x3f000000` rather than zero, then re-snapshots both
D32 and stencil planes byte-exactly.

On the retained submit-463 capsule, the fix moves operation 2's binding-7 linear result from all-zero
hash `ccc433ff6d980383` to `1d0ffd6fc0338383`, exactly the captured depth-plane hash. That positive
control proves the source lever moved. Operation 5 nevertheless remains the identical uniformly black
target (`26ed8b6191338383`; closure `6/6`, unresolved `0`). The depth alias was therefore a real
translation defect, but it is not the sole cause of the black composite; the remaining cause is
downstream or independent.

Static output slicing and the exact operation graph narrow that result further. Operation 1 is the
producer of operation 5's binding 119 (`0x204ec40000`), and its fragment shader consists only of
`v_mov_b32 v0, 0`, an RGBA export of `v0`, and `s_endpgm`. Operation 5's first sample reads binding 119,
multiplies its alpha by 255, converts it to an integer, and tests bit 6. Zero alpha makes the test false,
so the shader skips its large scene body and exports the constant-buffer fallback `(0,0,0,1)`. This is
not a stale captured seed: the graph names operation 1 as the in-submit producer, and replay executes it
before operation 5. The captured frame's black output is therefore guest-programmed at this point. A
resource override could force the unused branch for localization, but it would not model a producer the
guest actually submitted and is not evidence of a compatibility fix.

The dependency graph continues through submit 465: operation 16 reads `0x20168f0000` and writes
`0x203a7d0000`, operation 17 reads that target, and operation 18 reads it and writes `0x2010870000`;
submit 467 then reads `0x2010870000`. All operations are realized and failure-free, while the complete
replay endpoint is black. Those edges establish ordering and dependency, not the values written by each
producer. Further GPU localization inside operation 5 is no longer useful for this frame: the guest's
operation-1 zero mask prevents that body from running. The next discriminator must move upstream to the
guest progression/state that submits only the zero-mask path, then revisit the submit-465 chain after a
capture whose operation-1 mask is nonzero. Exact hashes and command shapes are retained in
[#1905](https://github.com/mattias800/prosper/issues/1905#issuecomment-5172641024) and its follow-up.

### Game Intent activity audit

The complete app contains the four classic RSDK data files, so an authentic PS5 activity launch was
also tested rather than assuming the normal menu was the only route. `sce_sys/param.json` permits the
standard `launchActivity` intent. Guest disassembly independently shows that Sonic reads its
`activityId` property and recognizes `TITLE_SONIC_1_CLASSIC` as its Sonic 1 Classic boot selection.

Prosper now models that shell action with `PROSPER_GAME_INTENT_ACTIVITY_ID`: it advertises one pending
System Service event, delivers event `0x10000017` once, and implements the Game Intent receive,
property, and terminate contracts. With the exact activity selected, the guest trace proves that
Sonic consumes all three relevant calls:

```text
sceSystemServiceReceiveEvent
sceNpGameIntentReceiveIntent
sceNpGameIntentGetPropertyValueString(..., "activityId", ..., 0x21)
```

That route still requests `ui_startup.pac` and `ui_title_nocopy.dds`, but those failures are handled
and do not by themselves explain why control does not reach the classic runtime. A 30-second filtered
trace never opens `raw/retro/Sonic1u.rsdk`; a 44-second
native capture remains black after frame 1, and the active stereo float32 48 kHz port 17 capture is
silent (`rms=0`). The activity experiment therefore proves that the requested activity is not
sufficient to escape the black state; it does not change the compatibility result or identify the
black-frame cause. Reproduce it with:

```bash
PROSPER_GAME_INTENT_ACTIVITY_ID=TITLE_SONIC_1_CLASSIC \
PROSPER_PAD_SCRIPT=@prosper/scripts/sonic/reach-title-or-gameplay.pad \
  prosper/build-linux/screenshot /path/PPSA05325-app0 \
  --seconds 1 --count 60 --timeout 120 --out "$HOME/prosper-artifacts/sonic-activity-shots"
```

The control is off by default; an ordinary launch continues to report no pending Game Intent.

### Post-resident steady state and thread census (2026-08-04, master `9dcb6c4b`)

The frontier moved upstream of the GPU once the captured frame was classified as guest-programmed
black. This section records what the guest actually does *after* its resident load, so the next lane
does not have to re-measure it.

**The load finishes almost immediately and nothing follows it.** In a routed CPU-only arm
(`PROSPER_NO_COMPUTE=1`), every resolve/read in the resident set completes before the t=5 s
`PROSPER_PROGRESS` heartbeat. The remaining 180 s contain no file operation at all, and the resolved
path set is a fixed 46 entries — the same set under the live renderer with continuous input.

**The frame work is constant, not merely repetitive.** Across the whole arm the guest submits exactly
**22 submits, 22 draws and 14 dispatches per flip** (185 s: 248,048 draws / 157,954 dispatches /
11,274 flips). That is the same 22 submits and 36 operations the retained present-20 bundle contains,
so the frame the earlier lanes dissected is the *only* frame Sonic ever builds: the engine reaches its
steady render state before present 20 and never changes it. There is no fade, no animation and no
progressive load behind the black output.

**Thread census.** Guest-named threads recovered by attaching to the settled process and classifying
stack contents against the eboot load base (57 threads total):

| Guest thread name | Count | Sampled behaviour |
| --- | --- | --- |
| `RsdxThread` | 13 | active — stacks advance between samples |
| `CRI MPV Worker` | 12 | **permanently parked**, identical stacks across all samples |
| `JobTPool1..12:0` | 12 | thread pool, mixed idle/active |
| `CRI FS *` / `CRI Server Mana` / `CRI Audio Outpu` | 6 | active |
| `module-rt` | 3 | prosper-side |
| `CriManaDecodeTh` | 1 | **permanently parked** |
| `Rsdx11CoreCommo`, `JobQueue`, `WorkerIO`, `WorkerResource`, `WorkerNVMe` | 5 | one core thread active; NVMe/queue idle |

The CRI Mana / MPV movie-player group (13 threads) is fully constructed and then never used: no `.usm`
is opened in any arm of this series. Whether Sonic's boot is supposed to play a movie here is the open
question — it is the one large subsystem that is initialised and idle.

**No thread is blocked on a primitive prosper never signals.** `tools/guest_bt` (the right instrument
here — it bridges the HLE stub boundary that defeats a plain gdb stack walk) resolves guest thread 1's
chain through the Sony sleep stub to `eboot+0x933088`, reached from `eboot+0x930360`, which the entry
region calls at `eboot+0x50ef27`. That call site is followed by `mov r14d, eax` and a virtual teardown
of the global at `eboot+0x3522950`, so `eboot+0x930360` is the RSDX **application run loop** and the
sleep is its frame pacing. Main is therefore healthy and inside the normal run loop, not parked on a
wait. Every other thread resolves to an ordinary `pthread_cond_wait` / `cond_timedwait` / `nanosleep`
pool idle. The whole "some guest thread waits forever for something prosper never posts" family is
closed for this state.

**What is therefore left.** The guest is alive, not deadlocked, and asks prosper for nothing it does
not get: no unimplemented call, no pending file, no blocking service, no missing input. It simply
re-renders an empty scene. The next discriminator has to identify the internal condition the RSDX
frontend is testing — the candidates now worth separating are (a) a wrong value prosper already
returned earlier in boot (a registered-but-mismodelled call, which the unimplemented-NID table cannot
see by construction) and (b) a CRI Mana / Sofdec2 boot-movie step that never starts. Neither is
addressable by another whole-frame GPU capture of this state.

**The steady state's complete HLE surface (2026-08-05, `tools/hle_calls`).** The `PROSPER_SVCLOG` census
above bounds the *instrumented* surface, not the guest's traffic — roughly 1,035 handler registrations
are served by 94 `svc_log` call sites. `tools/hle_calls` (#1980) counts every handler instead, and two
400-tick windows on `c77c66b4` agree: the settled loop enters exactly **47** distinct handlers
(`total≈21,000` calls per window, positive control `s_user_getevent`=4). Everything but three of them is
`agc_*`, `audio2_*` or `k_*`; the entire external surface is **`s_user_getevent`, `s_syss_getstatus` and
`pad_read_state`, one call each per frame**. That is the complete list of places a wrong value could
still be hiding *at this depth*, and it is short.

It is not the list for candidate (a), which is about init. `hle_calls` cannot see init: it attaches to a
live pid, and an arm that launched `boot_trace` and attached at t=1 s with `--filter '^s_'` still opened
its window inside the frame loop — the "boot" and "settled" passes returned the identical two-handler
histogram at `entries=4000`. Counts would not be enough either, since a mismodelled call returns a wrong
*value*. Both gaps are filed as **#1997** (launch mode + return-value recording), which is the concrete
enabler for candidate (a).

**Route caveat that cost this series time.** `.pad` route positions are **pad reads, not seconds**, so
a route's wall-clock length scales with the guest's frame rate: the 404-read
`reach-title-or-gameplay.pad` lasts ~6.6 s in a 61 reads/s CPU-only arm but ~80 s in a ~5 reads/s live
renderer arm. `scripts/sonic/long-input-probe.pad` exists so an input-gated hypothesis can be excluded
without that ambiguity; it is an investigation aid, not an evidence route.

### Present path: nothing Sonic renders can be published (2026-08-05, master `c77c66b4`)

The publish path was measured directly rather than inferred from the screenshots, because Sonic
carries the Frontiers publish-wall signature: `frame_seq` frozen at 1 while `present_count` climbs.
It does have that wall, and it needs #1990 — but the wall is downstream of the real gap. Both
findings are in the `## Ruled out` table above; the structural facts behind them are here, since they
are what the next lane has to work with.

A `PROSPER_PASS_LOG` census over 3,859 pass records finds **six distinct render targets and not one
of them is a registered VideoOut buffer** (`vo=1` count: **0**). The registered scanout pair is
`0x2012850000` / `0x2014830000` (the front alternates between them), and neither address appears
anywhere in `PROSPER_DUMP_PERSISTENT`'s enumeration of the RTT cache — that census walks every entry
in `g_rtt`, and it lists ten targets, none of them a scanout. (The accompanying `scanout=MISS` on the
same line is consistent but weaker on its own: `cached_scanout` returns null on four distinct
conditions, so a MISS alone would not prove the entry is absent. The absence is carried by the
enumeration and by the `vo=1` count of zero.) Per flip the guest renders:

| target | extent | format | readback |
| --- | --- | --- | --- |
| `0x204ec40000` | 3840x2160 | `R8G8B8A8_UNORM` (37) | deferred |
| `0x20168f0000` | 3840x2160 | `R16G16B16A16_SFLOAT` (97) | once deferred, once read back |
| `0x20274e0000` | 1920x1080 | `B10G11R11_UFLOAT_PACK32` (122) | deferred |
| `0x206fa20000`, `0x206fea0000`, `0x2070320000` | 960x540 | `R16G16B16A16_SFLOAT` (97) | deferred |
| `0x203a7d0000` | 3840x2160 | `R8G8B8A8_UNORM` (37) | deferred |
| `0x2010870000` | 3840x2160 | `R8G8B8A8_UNORM` (37) | **always deferred** |

`0x2010870000` is the last *draw* target in the frame, and the guest's final step into the scanout is
a **compute dispatch** (`0x2010870000 -> 0x2012850000`). Two independent consequences follow, and they
are worth separating:

1. **Present-source selection cannot see this frame at all.** A pass becomes a candidate only when it
   has CPU pixels *and* its format is exactly `R8G8B8A8_UNORM`. `0x2010870000` satisfies the format
   and never the pixels: it is neither `is_vo` nor `front_va`, so it takes the intermediate-RTT
   deferral every time (`live_renderer.cpp`'s `defer_readback`), and a deferred pass has no
   `rendered_pixels` to offer. Two recovery paths exist and both miss for a precise reason worth
   keeping straight: the deferred-scanout materialization is gated on `is_vo` (never true here), and
   the final span's `cached_scanout` recovery is keyed on the **flipped address** rather than on any
   pass — it would materialize a persistent target found at a VideoOut address — but the RTT-cache
   enumeration above contains no entry for either scanout VA. So the gap is not "the wrong pass was
   chosen"; it is that **nothing prosper owns exists at the address the guest flips**. This is the same
   shape as #1968 §5 ("why no post-intro pass targets the flipped VideoOut buffer") on a second,
   unrelated title, so it is a **cross-title present-path gap, not a Frontiers quirk** — and on Sonic
   it is total rather than partial.
2. **It is nevertheless not why the screen is black.** Every one of those targets is black at bit
   level, not merely after conversion — the final 4K composite is `00 00 00 ff` per texel and the 4K
   FP16 scene target is RGB bit-zero — and so are the raw guest scanout buffers. There is no content
   anywhere for the present path to lose. Repairing (1) on Sonic today would publish black.

So the frontier is unchanged in substance and much better bounded: the guest builds a frame with **no
colour in it** — the 4K composite is opaque black per texel and the scene target's RGB bits are zero,
so this is measured content rather than a conversion artifact — and the reason is upstream of every GPU
and present mechanism prosper has now inspected.

### APR miss census, with guest call sites (2026-08-05)

`[apr] resolve MISS` now names the guest code that asked (a best-effort stack scan; see
`apr_miss_callsite` in `src/hle/hle_file.cpp` for how to read it). A 30 s CPU-only arm on
`c77c66b4` records six distinct misses, in two groups:

```text
[apr] resolve MISS /app0/raw/ui/ui_startup.pac                 ra2=eboot+0xf0609
[apr] resolve MISS /app0/raw/ui/rpl_texture/ui_title_nocopy.dds ra2=eboot+0xf0609
[apr] resolve MISS /app0/dlc3/sound/HITE_music.awb      ra=eboot+0xef4a8 ra2=eboot+0xf0609
[apr] resolve MISS /app0/dlc3/sound/HITE_missions.awb   ra=eboot+0xef4a8 ra2=eboot+0xf0609
[apr] resolve MISS /app0/dlc3/sound/Music03_S3K.awb     ra=eboot+0xef4a8 ra2=eboot+0xf0609
[apr] resolve MISS /app0/dlc3/sound/Music09_Museum.awb  ra=eboot+0xef4a8 ra2=eboot+0xf0609
```

Both groups arrive through the same guest wrapper. `eboot+0xef4a3: call 0xf0600` returns to
`0xef4a8`; inside that wrapper `0xf0604: call 0xc47370` (the APR import thunk) returns to `0xf0609`.
On a non-zero result the wrapper formats the guest's own message —
`E2020010981FS:resolveFilepathsToIds() failed. : 0x%08x, path:%s` — clears its boolean out-parameter
and returns `-1`. It is a graceful existence/resolve helper, so *this* frame handles the miss; what
its callers do with a false answer is still open.

**The four `dlc3` wave banks are new, and they are not a missing dump.** All four exist as
`raw/sound/<name>.awb`; the guest asks for them under the DLC3 mount point instead. They appear only
now because installed-DLC enumeration landed in #1916, after the earlier censuses. This is the same
class as the `/app0/X -> /app0/raw/X` content-root fallback already in `apr_resolve_impl` (an external
CRI ACB naming its companion AWB relative to a content root), one mount level further out, and it is
the first concrete candidate for the silent AudioOut2 port. Tracked separately as **#1993** — it is an
audio path, not the black frame.

**That last paragraph is superseded: the DLC-prefixed miss is not a gap at all.** The census above
stopped at the miss. Read one line further and the guest retries the same bank against the base
content root, where it resolves — see the `## Ruled out` row, and #1993. The census is repeated below
on current master because both of its groups changed.

### Boot-call census with return values (2026-08-05, master `c3614f51`)

`tools/hle_calls --launch --values` (#1997) opens its window before the guest's first instruction, so
this is the **whole boot**, not the settled loop, and each row carries what prosper answered. Every
arm here is CPU-only (`PROSPER_NO_COMPUTE=1`) and self-checked: `first-calls` leads with
`s_ok`/`s_videodec2_query_compute_memory`/`s_user_initial` rather than the frame-loop pollers (so the
window really covered init), and `s_user_getevent` reports `0x0 x1, 0x80960007 x529` — its known
deliver-LOGIN-once contract — as the value-capture positive control.

**The complete non-hot boot surface is 445 calls.** Excluding `agc_*`, `k_*`, `audio2_*`, `pad_*` and
the two per-frame service pollers, the guest makes 445 calls across 49 distinct handlers and then
**stops**: after ordinal 445 nothing but `s_user_getevent`, `s_syss_getstatus` and `pad_read_state`,
one each per frame, forever. The shape of the boot is:

| ordinals | what runs |
| --- | --- |
| 1-14 | guest `f_write` banner (`Rsdx-Default Allocator onion/garlic/main`, `Poolmemory CPUGPU`) |
| 15-19 | AGC register defs, `sceVideoOutOpen` (`ret 0x1001`), `RegisterBuffers2` |
| 20-37 | first APR measures; Videodec2 compute-memory query x4 + `allocate_compute_queue` |
| 39-46 | UserService initial-user/name/id, **`sceAppContentAppParamGetInt`**, `sceSystemServiceParamGetInt`, GameIntent init |
| 47-65 | resident APR resolves (this is where `ui_startup.pac` misses) |
| 67-81 | NP state callback, NetCtl callback + state, SaveData init3, NpUds/Trophy2/NpEntitlement init, Share |
| 82-167 | JSON object/string churn (Share content parameters) |
| 168-214 | entitlement list x2, NpUds events, Trophy2 create/register, **`sceAppContentAddcontMount` x4 (all `ret 0`)** |
| 215-439 | the bulk of the asset load — `f_apr_resolve` / `resolve_ids` / `measure_read_file` |
| 440-445 | `sceNetPoolCreate`, `sceSslInit`, `sceHttp2Init`, `sceNpWebApi2Initialize`, `sceNpWebApi2CreateUserContext`, **`sceNpHasSignedUp` (last call of the boot)** |

Two things in that table are new relative to everything above it in this document. **All four DLC
now mount** (`s_appcontent_addcont_mount` x4, each returning 0) — the earlier record that Sonic
"consumes the four installed records but does not continue into a key/mount path" is out of date.
And the **asset load terminates at ordinal 439** having loaded only the resident set: no scene, no
`ui_mainmenu*`, no `stage_title`, no `.rsdk`, no `.usm`.

**No return value in the census is implausible.** The full `^s_` value histogram over 3,000 ticks:
`s_user_getevent` `0x80960007`/`0x0`, `s_nptrophy2_unavailable` `0x80551500` (deliberate — see its
comment in `hle_service.cpp`), `s_npweb_create_user_context` `0x3e9` (a context id),
`s_ssl_init`/`s_http2_init`/`s_net_pool_create`/`s_np_register_state_cbA` `0x1` (ids), and `0x0`
everywhere else. The file/graphics surface is equally clean: `f_apr_resolve_ids` returns `0x0` x87
and `0x80020002` (ENOENT) x22 — which decomposes as the seven wave banks retried three times each
(the three-repeat factor is #1993's measurement, not this run's) plus the single `ui_startup.pac`
miss — `f_apr_resolve`
`0x0` x61, `g_vo_open` `0x1001`.

So the "a registered-but-mismodelled value returned during init" candidate is now **bounded rather
than confirmed** — though read the bound precisely: it is over 49 **prosper handler symbols**, not over
49 Sony functions. Alias stubs cover several Sony entry points each (`s_ok` alone is registered for
`sceLoginDialogInitialize`, `sceUserServiceInitialize`/`Terminate`, `sceNpRegisterStateCallback`,
`sceNpCheckCallback` and more), and every one of them collapses into a single census row reading `0x0`
— which is exactly what a mismodelled call looks like. The search space is a specific list whose every
answer is recorded,
and the one value that *was* wrong (`USER_DEFINED_PARAM_1`, fixed by #2003) is the region, above.
`--values` cannot see a wrong **out-struct** — only the return register — so a handler that returns 0
while writing the wrong bytes through a pointer is still reachable, and that is the residue of this
candidate.

**The resident resource list has 34 entries and exactly one of them is absent.** Breaking on the
`test al,al` at `eboot+0x990235` — the instruction immediately after the virtual `exists()` call that
produces every one of these APR requests — counts the whole pass:

```text
RESRESULT armed=3 ticks=1500 control=239 loop_iters=1 tested=34 found=33 missing=1
          seq=1101111111111111111111111111111111
```

Thirty-four entries are tested, **thirty-three exist**, and the single miss is the **third** entry
(`seq[2]`), which the file log independently names as `/app0/raw/ui/ui_startup.pac`. `loop_iters=1`
(the not-found continue at `eboot+0x990120`) equals `missing=1`, so the two counters agree.
`control=239` is the same per-frame guest control as above, so this is a measurement rather than a
void window.

This does **not** show that the absence causes the stall — the loop skips the entry cleanly, as the
`## Ruled out` row above establishes. What it does settle is the *character* of the absence: this file
is not part of "an optional region/legal asset set this build does not ship", it is the sole missing
member of the title's own 34-entry resident list, every other member of which is present and loads.
Any future reading of the earlier dump-integrity row has to account for that.

**The guest thinks it has finished booting — it hides the system splash screen at tick 22.** Sonic
imports `sceSystemServiceHideSplashScreen` (`Vo5V8KAwCmk`, PLT `eboot+0xc487c0`), which is how a PS5
title tells the shell to stop compositing the store splash over it and start showing the title's own
output. A CPU-only arm breaking on the guest PLT reports:

```text
SPLASH first-hit sceSystemServiceHideSplashScreen at tick=22
SPLASHRESULT armed=4 ticks=1500 sceSystemServiceHideSplashScreen=1 \
    sceSystemServiceReceiveEvent=0 sceCommonDialogIsUsed=0 CONTROL_run_loop_frame_pacing=239
```

Same self-check as the movie arm: `armed=4`, and `CONTROL_…=239` is a guest address the run loop
reaches every frame, so the two zeros are measurements. The call happens **once, very early**, and is
never repeated. So the black frame is not a title still sitting in a pre-display boot phase waiting to
be released — the guest has already declared itself ready to display, and what it then displays is an
empty scene, forever. That also rules out `sceCommonDialogIsUsed` / `sceSystemServiceReceiveEvent`
gating: neither is ever called.

**Candidate (b) has a named artifact.** `raw/movie/` holds 83 `.usm`, and all but a few are per-game
intros/outros or `tutorial_<n>_<m>` — but it also holds **`sonicteam_logo_4k.usm` (6,476,896 bytes)**,
`PencilTest.usm`, `soniccd_op_4k.usm` and three `sonic_30th_as_*.usm`. A boot logo movie therefore
exists in this application, the Videodec2 compute queue is allocated at ordinal 33-37, the CRI Mana /
MPV worker group is constructed — and **no `.usm` is ever opened in any arm**. That is the sharpest
remaining frontier, and it is a rung-1 target in its own right: the SONIC TEAM logo is exactly the
kind of splash the ladder counts. The movie names are not in the eboot's string pool (nor in
`rfl_resident.pac`, `bindata/*.bin`, `param_tech.rfl` or `ui_resident.pac` as plain text), so the boot
sequence that would name one is data-driven and has still to be located.

The guest side of that path is already mapped, so the next lane does not have to re-find it.
`eboot+0x6d76c0` is the **movie-path builder / player entry**: it takes a movie **index** in `esi`,
looks the entry up in a 0x68-byte-stride table, and selects the `"_4k.usm"` (`eboot+0xdb231f`) or
plain `".usm"` (`eboot+0xdba47f`) suffix from a per-entry flag at `+0x10932`
(`cmp BYTE PTR [rcx+rdx*1+0x10932],0x0` … `cmove rdx,rcx` at `eboot+0x6d7719..0x6d772f`). It has
exactly two callers — `eboot+0x703a67` (in `eboot+0x701280`) and `eboot+0x796016` (in
`eboot+0x795870`).

**None of the three is ever entered, and that is a measurement, not an inference.** A CPU-only arm
breaks on all three guest addresses plus a fourth as a control, and reports:

```text
MOVIERESULT armed=4 ticks=1500 movie_hits=0 control_hits=243
```

Two things make that a real negative rather than a void run. `armed=4` says all four breakpoints were
actually inserted — guest breakpoints **cannot** be armed before `run`, because prosper maps the guest
image itself and the addresses do not exist when gdb inserts breakpoints, so they are armed on the
first `prosper::k_usleep` tick instead. And `control_hits=243` is a **guest** address the run loop
reaches every frame (`eboot+0x933063`, the frame-pacing compare ahead of the sleep at
`eboot+0x933083`), so a guest breakpoint of exactly this kind demonstrably fires in this window while
the movie ones do not.

**Read that negative with its arming caveat, because the caveat lands exactly where the hypothesis
does.** The guest breakpoints go in on the *first* `k_usleep` tick, so `movie_hits=0` strictly means
"never entered after the first tick" — and a boot logo is precisely the thing you would expect
*before* it. `control_hits=243` proves that breakpoints fire inside the window, not that the window
opens early enough. The genuinely init-covering evidence that no movie is ever requested is the
`--launch --values` census above, which starts at the first instruction and contains no file open of
any `.usm` and no Videodec2 call beyond the boot-time queue allocation. The same caveat applies to
reading `SPLASH first-hit … at tick=22` as a *first* hit.

Finding which state should call `eboot+0x6d76c0`, and with which index, is the open work.

Both callers are reached **indirectly**, which is why a static walk stops here. `eboot+0x701280` is
never called: it is taken by address at `eboot+0x70111c` and installed as a task alongside a companion
at `eboot+0x700c80`, and the install is gated at `eboot+0x7010ff` on
`cmp BYTE PTR [rax+0x13e],0x0` falling through to a predicate `eboot+0x7f3560` — which compares two
**floats** (`[rsi+0x124]` against `[rdx+0x11c]`), i.e. a timeline position, and the surrounding code
names its state object `GOCTinyFsm2`. That reads as an in-stage cutscene trigger, not a boot logo.
`eboot+0x795870` is a **vtable slot** (`0x1e191a8`) with no direct reference at all, so it is the more
likely boot path and the better place to start. Neither conclusion is measured — both are read off the
disassembly — and the useful next step is dynamic: break on `eboot+0x6d76c0`'s two callers *and* on
whatever writes the movie index, rather than another static walk.

### The one hypothesis that joins all of it (2026-08-05) — labelled as a hypothesis

Everything measured above is compatible with a single reading, and the next lane should test it before
anything more expensive. It is **not** established, and it is written here as a candidate:

> `raw/ui/ui_startup.pac` **is** the boot sequence. Its absence is why the frontend's first scene is
> empty, why no logo movie is requested, and why the state machine never advances to `ui_mainmenu`.

What supports it: the package is named `ui_startup`; it is the **third** entry of the title's 34-entry
resident list and the only absent one; the guest hides the system splash at tick 22 and then displays
an empty scene forever, i.e. it believes it is past boot; no scene, menu, `.rsdk` or `.usm` is ever
requested afterwards; and the boot logo movie (`sonicteam_logo_4k.usm`) exists, its decoder is
initialised, and its player entry is measurably never entered. A UI scene that drives the logo, the
legal screens and the hand-off to the title menu would account for every one of those at once.

What argues against it, and must not be waved away: the loop that requests the package **skips a
missing entry cleanly** (`eboot+0x990239` -> `eboot+0x990120`), with no error path, no retry and no
wait — so nothing yet shows the *state machine* depends on it, only that the *loader* does not.

How to settle it, in increasing cost: (1) check whether `raw/ui/ui_startup.pac` exists in another copy
of this title's app0 — if it does, this is a dump gap and the honest result is to record it as the
blocker, not to alias anything; (2) find the guest state that consumes the loaded package handle and
see whether it has a no-package path; (3) break on the writer of the movie index and see whether any
state ever selects one. **Do not fabricate the package, alias another `.pac` to it, or force the
`exists()` predicate true** — that would model a producer the guest never had and could not be
progression evidence even if it changed the screen.
### Current-master re-run and the boot-gate bound (2026-08-06, master `beeff2ab`)

**#2021** (`sceSysmoduleIsLoaded` answers from prosper's own load history) and **#2031** (the
render-target-0 blend key on every SDK version) landed after the section above, and both touch
surfaces this title uses, so the baseline was re-measured rather than assumed. **Neither moved
anything.** A direct headless `screenshot` arm at native 3840x2160 with `PROSPER_RENDER_SCALE=1
PROSPER_RENDER_EVERY=1` and `scripts/sonic/long-input-probe.pad`, 10 samples over 120 s, reports
`source-distinct=10 pixel-distinct=1`, every sample `crc=666f7b3f`. A `PROSPER_NO_COMPUTE=1` arm over
the same route resolves the same 40 unique paths, opens no `.usm` and no `.rsdk`, and settles into
the same 22 draws / 14 dispatches per flip. Sonic Origins remains **rung 0**; nothing here is a
visual claim.

Scope that precisely, because the neighbouring section is easy to contradict: **#2003 is not part of
this arm's claim.** `c3614f51` — the master the census above ran on — already contained it, and that
section records what it *did* change here (the region flipped JP→EU and the
`ui_title_nocopy.dds` request disappeared). "Neither moved anything" is about #2021 and #2031 only.

`#1990`'s wall still fires on this title on current master — `[rtt] PRESENT SOURCE EXTENT MISMATCH …
px_front=none px_vo=none px_last=none, offered 0 bytes; serving the retained frame instead (published
so far: fresh=1 retained=465)` — which is the same reading as before: selection has nothing to offer,
and every target it could offer is black anyway.

**Bounding the "registered-but-mismodelled value" candidate statically.** That candidate is invisible
to every absence check by construction (the call happens; the answer is wrong), and a runtime
return-value histogram says what prosper returned, not whether the guest looked. `nid_gate_scan.py`
gained an `--all-nids` mode for exactly this question — it classifies **every** import of a module in
one pass, so the call sites that cannot be affected by any answer are struck off before a boot:

```bash
python3 tools/re/nid_gate_scan.py <DUMP_ROOT>/PPSA05325-app0/eboot.bin \
    --all-nids --names ../PS5-3.20_Libs
```

Of 813 NID-shaped dynsym entries, 536 imports are actually called. The tool's own summary states the
split, and it is deliberately a **three**-way one, not two:

```text
# eboot.bin: 536 imported NIDs are called; 247 shown at --min-gated=1
#   247 gated, 157 ignored-only (cannot matter), 132 unresolved (>=1 forward/undecodable window
#   — NOT cleared, read by hand)
#   site buckets: alu-gate=125 const=1 forward=3445 ignored=14432 nonzero=3440 other-cmp=166
#                 undecodable=2699
```

**Not-gated is not the same as cleared.** Only the **157** `ignored`-only rows are struck off; the
**132** unresolved rows carry at least one `forward` window (the result left the window still live —
returned, spilled or tail-jumped, so the gate is in a caller) or at least one `undecodable` one, and
**2,699 of the 24,308 windows — 11.1% — are undecodable**, which is a void sample rather than a
negative. Anyone narrowing the candidate list from this table must work through those 132 by hand
rather than treat the 247 as the whole search space. Two further properties matter more than the
numbers:

- It is a **static upper bound on what can matter**, not a list of what runs. Cross it with the boot
  census (`hle_calls --launch --values`) before treating any row as live: the
  `PROSPER_PROGRESS_UNIMPL` table reads `(0 distinct unimplemented functions)` on this title, so every
  row whose prosper handler is *unregistered* is called by code Sonic's boot never reaches.
- The gated rows that prosper answers with a **shared** stub are the residue worth reading by hand,
  because one prosper symbol can cover several Sony entry points and collapses into a single `0x0`
  row in any runtime histogram. **This residue is open, and it is larger than the two rows checked
  below.** Crossing the 247 gated rows against prosper's registration tables, **41** are answered by
  a handler registered for more than one Sony name — `s_ok` covers 15 entry points and `k_attr_noop`
  covers 20. Those 41 split **24 `libkernel` / 4 `libSceLibcInternal` / 13 across the eight service
  libraries** (`libSceNpUniversalDataSystem` 3, `libSceErrorDialog` 3, `libSceNpManager` 2, and one
  each from `libSceMsgDialog`, `libSceCommonDialog`, `libSceLoginDialog`, `libScePad`, `libSceRtc`).
  Quote the split rather than a "non-libc" subtotal: whether `libkernel` counts as libc is a
  definition, not a measurement, and it is the whole difference between the 37 and 13 that two
  readings of the same 41 produce. Only the **`s_ok`** intersection was worked through
  here, and it is exactly two: `sceCommonDialogIsUsed`, which the census above measured at zero
  calls, and `sceLoginDialogInitialize` at `eboot+0x9b5d0e`, which is the same init-and-record idiom
  as the sysmodule sites (`test eax,eax` → `or BYTE PTR [rbx+0x6a8],0x2`) and so cannot change what
  the frontend renders whichever way it is answered. **The other 39 have not been read.** A
  dispatcher-authoritative NID→handler→arity dump would make that cross cheap and drift-proof for
  every future title; tracked as #2070.

Verified against a known answer before use: in single-NID mode the sweep reproduces the
`sceSysmoduleIsLoaded` result byte-for-byte after the refactor (`const=1 nonzero=4`, the same five
sites), and directory-mode output over two other dumps `diff`s clean against the pre-refactor tool.
The `--all-nids` figures above were first produced by a throwaway driver written against the same
classifier and then reproduced by the committed mode; that is a re-derivation of the *enumeration*,
not an independent implementation of the classifier, so it constrains the symbol walk and not the
bucket rules — those are covered by `test_nid_gate_scan.py`.
