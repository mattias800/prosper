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
| The absent `raw/ui/ui_startup.pac` and `raw/ui/rpl_texture/` group prove the supplied `raw/` tree is a truncated repack | **Falsified as an integrity claim.** The tree is internally consistent: `ui_museum_item_texture_l_art1..214` is contiguous with no gaps, all seven multi-language `ui_*` families are complete at 12/12 languages, `raw/ui` contains no subdirectories at all, and the application's own `AMPRIDX3` index enumerates 631 of the 633 files actually present. The eboot's name pool contains exactly one base-game `ui/` entry with no file (`ui/ui_startup`) plus the six region rating textures (`ui_title_cero`/`esrb_e10`/`esrb_rp`/`nocopy`/`pegi`/`titile_healthy`) — one functional group, requested back to back, consistent with an optional region/legal asset set this build does not ship. Their absence is therefore not evidence of a damaged dump, and remains insufficient to explain the black frame. | #1905, this doc |

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

**Route caveat that cost this series time.** `.pad` route positions are **pad reads, not seconds**, so
a route's wall-clock length scales with the guest's frame rate: the 404-read
`reach-title-or-gameplay.pad` lasts ~6.6 s in a 61 reads/s CPU-only arm but ~80 s in a ~5 reads/s live
renderer arm. `scripts/sonic/long-input-probe.pad` exists so an input-gated hypothesis can be excluded
without that ambiguity; it is an investigation aid, not an evidence route.
