# prosper — progress blog

**Newest first.** Every screenshot checked into this repository, and the story around the ones worth
a story. Read down from the top and stop when you reach something you have already seen.

**This file is written by hand.** It used to be generated from git history, which meant its captions
were commit subject lines — they said what the *change* was, never what the *picture* is, and there
was nowhere to put "we finally reached gameplay in this one, look." That is the whole point of a
blog, so the generator and its CI gate are gone.

[`COMPATIBILITY.md`](COMPATIBILITY.md) remains the per-title overview and
[`PROGRESS_TRACKER.md`](PROGRESS_TRACKER.md) the per-title rung table — that one *is* still generated
from the tracker issues, and still gated, because it is a projection of state rather than a story.

## How to add an entry

Put it at the **top**, under a `## YYYY-MM-DD` heading. An entry needs an image and a sentence about
what you are looking at. Anything else is optional — write a paragraph when a title finally does
something, write one line when it is just another capture.

```markdown
## 2026-08-22

### Beast of Reincarnation: the whole game was under a coat of white paint

<p align="center"><img src="assets/screenshots/beast-of-reincarnation-deluxe-bonus-dialog.png" alt="Beast of Reincarnation — the game's own Digital Deluxe bonus dialog: an item list with Big Dipper, Black Shiba Skin, Special Hat, Amber and crop seedlings, a scrollbar, an orange note and an OK button, rendered at 3840x2160"></p>

Game Freak's first PS5 title, added to the library tonight. On a default launch it presents a
**flat white 4K frame** and holds it for four minutes — and every renderer diagnostic says the
emulator is fine, because it is: the guest runs, the frame counter climbs past twelve thousand,
and prosper renders hundreds of real draw batches a second. Nothing was wrong with the picture.
There simply was no picture, and then prosper painted over the place where it should have been.

Two things were doing that. The first was one missing instruction form. The pixel shader that
writes both of the title's 3840x2160 scanout buffers opens with
`v_cvt_u32_f32_sdwa v2, v2 dst_sel:WORD_0` — convert a float to an integer and drop the result into
the *low half* of a register, leaving the high half alone, which is how a shader packs two numbers
into one. prosper already understood exactly that instruction in its **signed** form and had never
been shown the unsigned one, so the shader was rejected and **404 draws per run into the visible
framebuffer were discarded**. Four lines fixed it. The census of dropped draws into the scanout
went from 404 to zero.

That was not enough on its own, because of the second thing, which is more interesting. The frame
went from white to... a different flat colour. This title issues over **eight thousand**
`ELIMINATE_FAST_CLEAR` passes per boot — a hardware operation that expands a compressed
"this whole surface is one colour" record into real pixels, *instead of* running a pixel shader.
prosper does not model compressed colour surfaces at all, so it has nothing to expand; it runs
those passes as ordinary draws, which paint the surface. That gap has been on file as
[#1588](https://github.com/mattias800/prosper/issues/1588) for a while with no title that made it
matter. Here it was the entire visible output.

<p align="center"><img src="assets/screenshots/beast-of-reincarnation-game-freak-logo.png" alt="Beast of Reincarnation — the GAME FREAK developer logo in white on black, rendered at 3840x2160"></p>

Add a default-off switch that stops those passes writing colour, and the game is just *there*
underneath: the GAME FREAK logo, then the Digital Deluxe bonus dialog above — item list, scrollbar,
that orange footnote, a properly styled OK button, 6,452 distinct colours. It had been rendering
all along.

So: **rung 1, honestly** — that switch is not a fix and is not on by default, the same passes
happen in every other title (one *Plucky Squire* trace has 36,613 of them), and the switch itself
now looks like it may crash the driver
([#2915](https://github.com/mattias800/prosper/issues/2915)). No title screen yet either. But the
measurement is now on #1588 instead of the argument, and two other titles that also end on a pure
white frame — *Astro Bot* and *Sonic Racing: CrossWorlds* — are suddenly one run each away from
knowing whether they have the same problem.

One thing worth writing down for whoever gets this title next. Its command-line file is
`uecommandline.txt`, without the `4` that every UE4 title in the library carries, and that looked
like the one real clue going in — the first UE5 title in the corpus. It is not: *Sonic Racing:
CrossWorlds* ships the same file name, and `COMPATIBILITY.md` already calls it Unreal Engine 5.
The actual evidence is duller and better — `Nanite`, `Lumen`, `VirtualShadowMap` and
`WorldPartition` in the executable's strings, and an IoStore container version of 6 where every UE4
title in the library has 2 or 3.


### Hi-Fi RUSH reaches its title screen on the first try

<p align="center"><img src="assets/screenshots/hifi-rush-title.png" alt="Hi-Fi RUSH title screen — the yellow branding, shattered logo and Press Any Button prompt, rendered at 3840x2160"></p>

A title added to the library this evening, booted for the first time a few hours later. `BOOT_COMPLETE`
at **281 ms**, and the complete 4K title screen holds from t≈224 s to the end of the run — on a
**default launch**, no throttle, no non-default switches, no pad input. `tools/screenshot`,
45/45 samples, `guest=running status=ok`.

With a pad route it goes further: through the first-boot language wizard and its settings pages,
into a loading hold. Not gameplay, so this is rung 2.

<p align="center"><img src="assets/screenshots/hifi-rush-rooftop-black-materials.png" alt="Hi-Fi RUSH Vandelay rooftop — correct geometry, depth and sky gradient, with every opaque surface shaded flat black"></p>

The second picture is the interesting one, and it is a defect rather than a success. This is the
Vandelay rooftop, and almost everything about it is right: the geometry is there, the depth sorting is
right, the sky gradient is smooth, and the "WELCOME TO VANDELAY" billboard renders in full colour
because it is emissive. Every *opaque* surface is a silhouette. So vertex processing and the
transparent path both work and only opaque material shading fails — which is a much smaller problem
than the picture first suggests. The prime suspect is 202 distinct `[dynvb] unknown V# format`
signatures, mostly `0x0`.

One more thing this title taught us, which cost the lane a 1168-second run: pressing Cross on the
language page raises a *"Continue with these settings?"* dialog whose cursor starts on the **right-hand**
option, so every Cross answers it negatively and returns to the same page. The run was completely
healthy the whole time — 70 of 90 samples pixel-distinct — and looked exactly like a renderer wall.
That is now the fourth title where input mapping has impersonated one. The committed route is
`cross` → `left` → `cross`.

Tracker [#2891](https://github.com/mattias800/prosper/issues/2891).

## 2026-08-21

### Beneath reaches gameplay

<p align="center"><img src="assets/screenshots/beneath-gameplay.png" alt="..."></p>

The opening dive. The waypoint counts down as the route moves and the dialogue plays over it —
this is the real scene, not a cutscene. Tracker [#1898](...).
```

> An entry is evidence of what rendered **on the day it was written**. It is not a claim about the
> title's current state — for that, read the tracker. Nothing is ever removed when a title moves on,
> because the point of a blog is that it records *when* things happened.

## 2026-08-22

### PGA TOUR 2K25 told us exactly what was wrong, in English, and it was not what it said

No picture with this one — the title still does not render. The finding is what moved.

A new Unity 6 / IL2CPP title, 35 GB, and it died 1.2 seconds into every boot. What made it unusual
is that it *announced* the crash first:

```
ERROR...
 PSN is an old version that cannot be used by the current player runtime.
 Please update the PSN native module and any associated managed assemblies to the latest versions
```

That message is a lie in the most useful possible way. Nothing is old, and no module is missing.
The string is not in the dump anywhere as plain text — it lives compressed inside the Unity data —
so it had to be chased through the disassembly of `Il2cppUserAssemblies.prx` instead, which is where
it turned into an exact mechanism. The game holds two globals: an argument count and a pointer to a
"plugin args" struct. It checks the count has a bit above the low nibble, then checks the struct
says `size == 0x10` and `version == 0x200`. Both globals were zero, so it took the mismatch branch —
and the mismatch branch's own third `printf`, the one that would have politely told us which version
it found, reads that version straight off the NULL pointer. The error handler is the crash.

`tools/re/xref.py` found exactly one writer of the pointer, and it was a two-instruction function
that stores `rdi` and `rsi` and returns: the module's `module_start(size_t argc, const void *argp)`.
prosper was starting the module with `(0, NULL)`.

The good part is what was already in the tree. prosper has had that exact descriptor —
`{size = 0x10, version = 0x200, callback = 0}` — since the PSN.prx work, sitting in
`exec_image_linux.cpp` under a comment reading *"the descriptor Sony's PSN/SaveData plugin
module_start validates"*. Same two constants. The Unity PSN package simply does not always ship as
its own `PSN.prx`: with IL2CPP its native half can be linked straight into the user-assemblies
module, and then the handshake belongs to *that* module's `module_start`. It only ever needed to be
told about one more module.

With that fixed the title runs six times longer, streams its assets through Ampr, brings up the
compute backend and submits over a thousand draws — and then dies again, on a thread called
`Background Job.`, in a loop scanning for `:` and CRLF. An HTTP header parser, walking a NULL
buffer, because `sceHttp2SendRequest` returned `0` for a request that was never sent, `0` is
`SCE_OK`, and `sceHttp2GetAllResponseHeaders` then reported success without writing anything to its
out-parameters. Rung 0 still, but every metre of that is now named:
[#2894](https://github.com/mattias800/prosper/issues/2894), tracker
[#2895](https://github.com/mattias800/prosper/issues/2895).

## 2026-08-21

### Yakuza Kiwami allocates its entire game heap through a Sony API nobody had implemented

No picture with this one — the title still does not render. The finding is what moved.

*Yakuza Kiwami* (`PPSA31334`) died **0.0 seconds** into every boot, writing to address `0x1d0000`.
That address is the tell: it is far too low to be a real guest pointer, and it is what you get when
an allocator is handed a base address of roughly nothing and starts walking.

The base came from `sceAmprAmmGetVirtualAddressRanges`. AMM is the memory-mapping half of
libSceAmpr — the same command-buffer construct prosper already used for asynchronous **file reads**,
pointed at pages instead of bytes — and this title runs its *whole* game heap through it: it asks
AMM for up to 512 GiB of address space, hands it 10 GiB of physical memory, and then maps 2 MiB
chunks in on demand for the rest of the run. All seven of the AMM entry points it needs fell to
prosper's unimplemented default, which returns 0 and writes nothing.

Returning 0 sounds harmless. It is not, when the guest is reading your *out-parameters*: the
initialiser reaches that call on a path that never zeroes its own struct, so "wrote nothing" meant
the allocator took its virtual-address window from whatever was left on the stack. Everything after
that was the allocator faithfully doing what it was told.

With AMM implemented the boot now reserves a real 68 GiB window, takes its 10 GiB pool, and services
23 map commands before it gets somewhere new — far enough to initialise save data and start loading
assets, where it stops with the game's own message:

```
Failed!! Load Devil2 Shader Archive
Failed!! Load Ptc Shader Archive
```

That is the next wall, and it is a different one: `sceAmprAprCommandBufferReadFileGatherScatter`
([#2872](https://github.com/mattias800/prosper/issues/2872)), the scatter/gather sibling of a file
read prosper already implements. The archives never arrive, so the object is null, so the next
method call dereferences it. Still rung 0 — but the fault moved from the memory allocator to the
asset loader, which is the direction that counts. Tracker
[#2864](https://github.com/mattias800/prosper/issues/2864).

One footnote worth having: this turned out not to be a one-title fix. *Judgment* (`PPSA02739`),
onboarded the same day, imports **all seven** of the same AMM entry points — and both of the follow-up
gaps too, the scatter/gather read and AMM's `Unmap`. Checked by NID against its own import table,
not inferred from the shared publisher.

### Our first CryEngine title deadlocks 81 ms in, on a library it never asked for

No picture this time — the interesting thing about *Sniper Ghost Warrior Contracts 2* (`PPSA03130`)
is that there was nothing to photograph, and *why* there was nothing.

The first boot attempt produced no screenshots, no log past the renderer line, and an empty
manifest. That is a shape worth recognising, because it reads like a defect and is not one: the run
had been killed from outside. `tools/screenshot --timeout` cannot fire during boot — the deadline is
checked inside the sampling loop, and that loop is only reached after `boot_program()` returns.
`boot_program()` ends by running the guest's own module initialisers, so a title can sit inside it
forever with the tool's own limit inoperative.

prosper has recorded seven boot phases for a long time. It turned out **no build the project ships
could print any of them** — the feature was compile-time optional, the default build excluded the
whole folder from `prosper_core`, `enable()` was never called anywhere in the tree, and nothing
subscribed to the event bus. Four independent reasons, any one sufficient. `PROSPER_BOOTPHASE=1` now
prints them, and the answer arrived in one run:

```text
[bootphase] +80.6ms MODULES_MAPPED
[bootphase] +81.0ms STUBS_INSTALLED
[bootphase] +81.1ms GUEST_INITS_RUNNING     <- and BOOT_COMPLETE never comes
```

Not slow, not starved — **stuck**. Over 221 s the process used 0.00 s of CPU and read 0 bytes, with
every thread parked in a futex. `guest_bt` named the frame: the `module_start` of
`sce_module/libSceNpCppWebApi.prx`, a library this title **does not import** (its own NP library is
the unrelated `libSceNpWebApi2`). prosper preloads it because the file exists, under a rule added for
*Sonic Origins*, which really does import it. Remove that one file from the tree and the same binary
reaches `BOOT_COMPLETE` in 70 ms and runs.

So a module preloaded for one title had been silently wedging another, and the fix is to preload it
only when something actually imports it.

Which raised the obvious question a reviewer asked and I had not: *how many titles does that change?*
I had checked two. The answer is a census — across the tracked titles, 42 ship that PRX, 40 keep it, and two
lose it: this title, and **Sonic Frontiers**, which nobody had looked at and which has no snapshot
guard to notice. It appears to be harmless (import resolution is by NID, and not one of the 41,638
NIDs that module exports is imported by anything in Frontiers' link graph) but "appears to be" is the
honest phrasing, and a confirming boot of Frontiers belongs to the lane that owns it. A flag on a
shared list is never a two-title question.

Behind that wall the title is still at rung 0, and honestly so. Unmodified, with the fix in, it
boots in 91 ms, streams its assets, and drives a 4K present loop at ~21 flips a second — while prosper
composites exactly nothing. Every sample is a raw guest scanout: one distinct colour, zero non-black
pixels, `published_frames=0`. Two runs on two different trees agree, so it is not an artifact. The
next wall is that no pass produces a present source at all — an ordinary graphics problem, and a much
better place to be than a deadlock.

One footnote worth keeping, because it nearly became a finding. Mid-run the thing looked *parked*: 1 %
CPU, no disk reads, eighteen threads asleep, and exactly one of them — Wwise's `AK::BankManager` —
blocked on a mutex while everything else waited on conditions. That asymmetry reads like a deadlock
with a culprit's name attached. It wasn't; the run resumed thirty seconds later. The box was 70-90 %
I/O-stalled by an unrelated archive extraction the whole time, and a warm page cache meant the "no
disk reads" number was measuring the wrong thing entirely — the read *syscalls* were climbing fine.
A mutex wait is not proof of a deadlock. The holder may just be slow.

### Sonic Frontiers' black world had two locks on the door, not one

No picture in this one — the world is still black. What changed is that we now know how far away it
is, and the number is smaller than it looked.

The Cyber Space stage reaches gameplay with a running clock, and the world behind the HUD is black
because sixteen of the stage's compute programs never execute. Three of them are the ones that
matter: screen-width passes over the frame the player is supposed to see. Their reject line has
named `image_load_mip` for a while, and the issue tracking it said so: *the* single remaining
blocker.

A reject line cannot say that. The recompiler stops at the **first** instruction it cannot lower and
reports that one; it has no way to tell you what is behind it. So we built a throwaway measurement
build — accept `image_load_mip` at LOD 0, knowingly wrong, never merged, run it where nothing is
submitted — purely to ask "and then what?". And then the same three programs stopped again, all of
them, on `s_getpc_b64`.

That second lock turned out to be a small one. The compiler had put a little constant table straight
into the shader blob and addressed it PC-relatively, which prosper already folds — but only when the
table is read back with an *untyped* load. Frontiers reads it with a typed one,
`tbuffer_load_format_x` at `32_FLOAT`, which is a 32-bit format that converts nothing, so the bytes
it fetches are the bytes that are there. The fold was already correct for it; a guard spelled
"untyped only" was the whole obstacle. That is what this PR fixes.

Two things nearly went wrong on the way, and both are the same shape — a check that looked like a
check. A typed fetch also honours the descriptor's channel routing, which the untyped one ignores,
and this game's table descriptor routes three of its four channels to a constant zero; the first
version of the fix only looked at the format, which would have been right here and silently wrong one
instruction over. And the test written to pin the *second* correction passed under a mutation that
deliberately broke the thing it was pinning — it was asserting "did the program compile", and that
program does not compile for an unrelated reason, so it could never have failed. It asks the detector
directly now.

With both cleared, all three programs vanish from the skipped list — they recompile and they run.
(Only one of the two is fixed here. The other is still open, so this measurement was taken with a
throwaway build that waves the mip instruction through at level zero — deliberately wrong output,
never merged, run where nothing is submitted. It answers "is anything else in the way", and the
answer is no.)

```text
before   [compute-census] 65536 dispatch decisions over 30 program(s)   13 programs listed, all executed=0
after    [compute-census] 65536 dispatch decisions over 30 program(s)   10 programs listed
         gone: 0x2005714000 (3840x135)  0x2005717e00 (3840x405)  0x200571bd00 (3840x270)
```

So the remaining work on those three really is one instruction now, which is what everyone thought
yesterday and was not true yesterday. And we know what it has to read: the guest builds itself a
2048x2048 two-channel float pyramid and binds **thirteen** descriptors to it — one per mip level to
write each one, plus one whole-chain view to read the finished thing back. The levels are sitting in
guest memory the whole time, at offsets prosper already computes correctly for each of those twelve
single-level descriptors. It just has never been able to look at them together.


### The Messenger's title screen runs at 206 fps and 0 fps at the same time

<p align="center"><img src="assets/screenshots/messenger-title-fps-overlay.png" alt="The Messenger's title screen with a burned-in overlay reading 2.9 FPS (206.3 PRESENTED) 1920X1080"></p>

That counter is burned in by the new `screenshot --fps-overlay`, and both numbers are true. prosper
handed this picture to the display **206 times a second**. Across one 120-second stretch of that run
it published **25,015 frames and exactly one of them differed** from the frame before it. It is a
still image being re-served at full speed.

Nothing is wrong here: it is a title screen, and a title screen is allowed to sit still. The point is
that **a framerate counted from presents cannot tell this apart from a hung game.** That is not
hypothetical — it is how R-Type Delta's regression (#2783) hid for nine days while the guest reached
stage 1 and every presented frame was the same retained one. Any counter we shipped that reported
"206 fps" here would have reported it there too.

So prosper now counts **distinct guest frames** — publications whose pixels actually changed — and
every place a framerate appears shows both, honest number first. `PROGRESS_TRACKER.md` has an FPS
column for it, sourced from a new `FPS record:` line in the game trackers.

### And the first real number: Blue Prince at 4.7 fps

<p align="center"><img src="assets/screenshots/blue-prince-cinematic-fps-overlay.png" alt="Blue Prince's opening cinematic — a blueprint of Mount Holly — with an overlay reading 4.7 FPS 94% ACTIVE 1920X1080"></p>

The Day One opening cinematic, at native 1080p with no snapshot acceleration, over fifteen minutes.
**4.7 frames per second, and 95% of the run was spent producing them** — that second number is what
says the measurement is honest rather than an average of a fast stretch and a frozen one. The rate
never left the 4.53–4.80 band across all 59 windows. (The overlay in the shot reads 94%, because it
is the running figure at that moment; 95% is where the full run finished.)

This is the "we have work to do" end of the scale, and it is now written down where you can find it
without re-running anything. For comparison the same instrument reads The Messenger's animated
stretches at 15–23 fps.

### Why The Plucky Squire's cutscene never ends

No picture for this one — it is a measurement, and it replaced a guess that had been sitting in the
notes as the frontier.

The record said the route "stops driving input at 525 s", implying the cutscene was waiting for a
button. It is not waiting at all. It advances roughly **300x too slowly to finish**, and two facts
multiply into that:

- the guest's tick rate **collapses 147x** when the 3D world streams in — about 25 polls per second
  in the menus, **0.19** once the level loads;
- in-game time advances **per flip, not per second**. That is deliberate and correct — each flip is
  budgeted one refresh interval — but it means the game clock moves ~16.7 ms per flip however long
  that flip actually took.

At 0.19 flips per second, a 60-second intro needs hours. The 1,200-second run that "never reached
gameplay" had bought a few seconds of it.

It is demonstrated rather than argued: raising only the flip rate walks the guest straight past the
intro to the storybook camera. So the wall is throughput, and the title is CPU-bound in texture
realization — the GPU sits at 5-16% throughout.

### Two more titles reach gameplay — and one reaches it in the dark

**Beneath** (`PPSA27640`) plays. This is the opening dive aboard the science ship: the waypoint
marker counts down as you move, and the characters talk over it. A cutscene would not have a live
distance readout, which is how we know it is the real thing.

<p align="center"><img src="assets/screenshots/beneath-gameplay.png" alt="Beneath — the opening dive, waypoint HUD reading 21m, dialogue subtitles over a dark seabed"></p>

It is very dark, and that is the game rather than us — but it is worth an eye-check when this one
comes up for manual verification. Getting here needed no renderer work at all; the title was one
input route away. What it *did* need was `PROSPER_NULL_PAGE=1`, and the reason is a nice one: the
game's stack unwinder walks one hop past the end of the frame-pointer chain **on purpose**, because
it stops on a null return address rather than a null frame pointer. We enter the guest with `rbp`
zeroed — which is correct — so that last read lands on address `0x8` and faults. The flag gives the
guest back a low page that reads as zero, which is what the console gave it.

### R-Type Delta, blank for nine days, draws stage 1 again

<p align="center"><img src="assets/screenshots/rtype-delta-stage1-restored.png" alt="R-Type Delta stage 1: the R-9 and its Force device over a sunset cityscape with enemy formations and the BEAM and score HUD"></p>

The R-9, its Force pod, enemy formations, and a city at sunset. This one is a good story. For nine
days the title rendered its logo and its whole opening movie and then went blank forever — the guest
was fine, reaching stage 1 and writing its save, while prosper published the same retained frame on
every flip.

One shader did that. `sprite_i_vv.ags` is the title's **sprite vertex shader**, so it draws the
menus, the HUD and the gameplay — everything except the logo and the movie, which is exactly the
symptom. The recompiler had been refusing it since a change in August that saved a wave mask in a
register pair and never ended that lifetime; when the shader later reused the same pair for an
ordinary table address, the stale mask made it look like ballot bits and the read was refused.

Reaching it also needed a route, and the title screen taught us something: its prompt is the PS5
**OPTIONS** glyph, not Cross. A Cross-only route sits there forever.

### Sonic Frontiers reaches Cyber Space — and the world is black

<p align="center"><img src="assets/screenshots/sonic-frontiers-cyberspace-hud.png" alt="Sonic Frontiers Cyber Space — stage clock at 00:55.89, ring counter, star medals and speedometer over an entirely black screen"></p>

Not a pretty picture, and it is here because it is honest. That is a real running stage — the clock
reads 55 seconds, the speedometer needle moves, the music plays — with a hundred streamed terrain
sectors behind a world that never draws. Sixteen of the stage's thirty-two compute programs never
execute.

Three separate recompiler fixes have now unblocked programs on this title and changed **zero
pixels**, so we have stopped assuming the next one will be different. It also prompted a rule
change: reaching gameplay is no longer enough for rung 3, which now asks that the scene actually
render. Frontiers and *Grand Theft Auto V* both sit at rung 2 because of it. Neither regressed —
we just stopped counting a black screen as a win.

## 2026-08-20

### dragon-quest-vii-opening-chapter.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-opening-chapter.png" alt="dragon quest vii opening chapter"></p>

feat(dq7): the route reaches the opening chapter in Estard, and Unreal titles get an IoStore package oracle (#2779)

`0ea7868c` · [`assets/screenshots/dragon-quest-vii-opening-chapter.png`](assets/screenshots/dragon-quest-vii-opening-chapter.png)

### bendy-dark-revival-gameplay.png

<p align="center"><img src="assets/screenshots/bendy-dark-revival-gameplay.png" alt="bendy dark revival gameplay"></p>

feat(bendy-dark-revival): rung 2 -> rung 3 — a route reaches the PPSA27624 Chapter 1 sections (#2769)

`5501dd45` · [`assets/screenshots/bendy-dark-revival-gameplay.png`](assets/screenshots/bendy-dark-revival-gameplay.png)

### tales-graces-f-gameplay-dialogue.png

<p align="center"><img src="assets/screenshots/tales-graces-f-gameplay-dialogue.png" alt="tales graces f gameplay dialogue"></p>

feat(route): Tales of Graces f Remastered reaches gameplay -- the wall was two OPTIONS-button gates, not the renderer (#2763)

`f249929d` · [`assets/screenshots/tales-graces-f-gameplay-dialogue.png`](assets/screenshots/tales-graces-f-gameplay-dialogue.png)

### tales-graces-f-gameplay.png

<p align="center"><img src="assets/screenshots/tales-graces-f-gameplay.png" alt="tales graces f gameplay"></p>

feat(route): Tales of Graces f Remastered reaches gameplay -- the wall was two OPTIONS-button gates, not the renderer (#2763)

`f249929d` · [`assets/screenshots/tales-graces-f-gameplay.png`](assets/screenshots/tales-graces-f-gameplay.png)

### sonic-origins-sonic-team-logo.png

<p align="center"><img src="assets/screenshots/sonic-origins-sonic-team-logo.png" alt="sonic origins sonic team logo"></p>

docs(compat): refresh the checked-in visual evidence, and repair two trackers that deny screenshots already on master (#2737)

`29f4db65` · [`assets/screenshots/sonic-origins-sonic-team-logo.png`](assets/screenshots/sonic-origins-sonic-team-logo.png)

### tales-graces-f-title-no-input.png

<p align="center"><img src="assets/screenshots/tales-graces-f-title-no-input.png" alt="tales graces f title no input"></p>

docs(compat): refresh the checked-in visual evidence, and repair two trackers that deny screenshots already on master (#2737)

`29f4db65` · [`assets/screenshots/tales-graces-f-title-no-input.png`](assets/screenshots/tales-graces-f-title-no-input.png)

### issue-2731-tales-graces-f-movie-chroma.png

<p align="center"><img src="prosper/docs/screenshots/issue-2731-tales-graces-f-movie-chroma.png" alt="issue 2731 tales graces f movie chroma"></p>

docs(compat): refresh the checked-in visual evidence, and repair two trackers that deny screenshots already on master (#2737)

`29f4db65` · [`prosper/docs/screenshots/issue-2731-tales-graces-f-movie-chroma.png`](prosper/docs/screenshots/issue-2731-tales-graces-f-movie-chroma.png)

### issue-2734-little-nightmares-3-corrupt-save-modal.png

<p align="center"><img src="prosper/docs/screenshots/issue-2734-little-nightmares-3-corrupt-save-modal.png" alt="issue 2734 little nightmares 3 corrupt save modal"></p>

docs(compat): refresh the checked-in visual evidence, and repair two trackers that deny screenshots already on master (#2737)

`29f4db65` · [`prosper/docs/screenshots/issue-2734-little-nightmares-3-corrupt-save-modal.png`](prosper/docs/screenshots/issue-2734-little-nightmares-3-corrupt-save-modal.png)

### asterix-babylon-gameplay.png

<p align="center"><img src="assets/screenshots/asterix-babylon-gameplay.png" alt="asterix babylon gameplay"></p>

bringup(asterix-babylon): a Triangle-aware input route reaches GAMEPLAY (rung 2 -> 3) (#2736)

`5404e173` · [`assets/screenshots/asterix-babylon-gameplay.png`](assets/screenshots/asterix-babylon-gameplay.png)

## 2026-08-19

### plucky-squire-chapter1-intro.png

<p align="center"><img src="assets/screenshots/plucky-squire-chapter1-intro.png" alt="plucky squire chapter1 intro"></p>

docs(plucky): status doc — the chapter-one world renders, plus five falsified hypotheses (#2742)

`675ff2f6` · [`assets/screenshots/plucky-squire-chapter1-intro.png`](assets/screenshots/plucky-squire-chapter1-intro.png)

## 2026-08-08

### sonic-crossworlds-title.png

<p align="center"><img src="assets/screenshots/sonic-crossworlds-title.png" alt="sonic crossworlds title"></p>

docs(crossworlds): rung 2 — the title screen renders completely, reproduced across two runs (#2360)

`a3a88356` · [`assets/screenshots/sonic-crossworlds-title.png`](assets/screenshots/sonic-crossworlds-title.png)

## 2026-08-07

### arcrunner-title-screen-default-route.png

<p align="center"><img src="assets/screenshots/arcrunner-title-screen-default-route.png" alt="arcrunner title screen default route"></p>

ArcRunner: the per-fold account — the guest's builder is released mid-fold, and the contract that forbids it is version-gated off (#2219)

`a22cf8de` · [`assets/screenshots/arcrunner-title-screen-default-route.png`](assets/screenshots/arcrunner-title-screen-default-route.png)

## 2026-08-06

### sonic-frontiers-autosave-notice.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-autosave-notice.png" alt="sonic frontiers autosave notice"></p>

fix(hle): sceSaveDataTransferringMountPs4 must report "no PS4 save", not SCE_OK — Sonic Frontiers reaches its title screen (#2208)

`7801a5bc` · [`assets/screenshots/sonic-frontiers-autosave-notice.png`](assets/screenshots/sonic-frontiers-autosave-notice.png)

### sonic-frontiers-main-menu.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-main-menu.png" alt="sonic frontiers main menu"></p>

fix(hle): sceSaveDataTransferringMountPs4 must report "no PS4 save", not SCE_OK — Sonic Frontiers reaches its title screen (#2208)

`7801a5bc` · [`assets/screenshots/sonic-frontiers-main-menu.png`](assets/screenshots/sonic-frontiers-main-menu.png)

### sonic-frontiers-title-screen.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-title-screen.png" alt="sonic frontiers title screen"></p>

fix(hle): sceSaveDataTransferringMountPs4 must report "no PS4 save", not SCE_OK — Sonic Frontiers reaches its title screen (#2208)

`7801a5bc` · [`assets/screenshots/sonic-frontiers-title-screen.png`](assets/screenshots/sonic-frontiers-title-screen.png)

### arcrunner-title-screen.png

<p align="center"><img src="assets/screenshots/arcrunner-title-screen.png" alt="arcrunner title screen"></p>

docs(arcrunner): the title screen is behind the movie, and the throttle rescues by DELAY not by lock hold (#2205)

`40129122` · [`assets/screenshots/arcrunner-title-screen.png`](assets/screenshots/arcrunner-title-screen.png)

### sonic-origins-sega-logo.png

<p align="center"><img src="assets/screenshots/sonic-origins-sega-logo.png" alt="sonic origins sega logo"></p>

fix(savedata): sceSaveDataCreateTransactionResource must return a positive resource id (#2121)

`e404841c` · [`assets/screenshots/sonic-origins-sega-logo.png`](assets/screenshots/sonic-origins-sega-logo.png)

### arcrunner-default-route-logos.png

<p align="center"><img src="assets/screenshots/arcrunner-default-route-logos.png" alt="arcrunner default route logos"></p>

ArcRunner: the init-side generation census — the #1756 question answered, four falsifications, and real graphics off the throttle (#1226) (#2122)

`8c8e74f5` · [`assets/screenshots/arcrunner-default-route-logos.png`](assets/screenshots/arcrunner-default-route-logos.png)

### arcrunner-intro-city.png

<p align="center"><img src="assets/screenshots/arcrunner-intro-city.png" alt="arcrunner intro city"></p>

ArcRunner renders its intro cinematic — the #1226 fault is a submit-timing race (follow-up to #2077) (#2086)

`0a6f82a8` · [`assets/screenshots/arcrunner-intro-city.png`](assets/screenshots/arcrunner-intro-city.png)

### arcrunner-intro-space-station.png

<p align="center"><img src="assets/screenshots/arcrunner-intro-space-station.png" alt="arcrunner intro space station"></p>

ArcRunner renders its intro cinematic — the #1226 fault is a submit-timing race (follow-up to #2077) (#2086)

`0a6f82a8` · [`assets/screenshots/arcrunner-intro-space-station.png`](assets/screenshots/arcrunner-intro-space-station.png)

### rtype-delta-force-select.png

<p align="center"><img src="assets/screenshots/rtype-delta-force-select.png" alt="rtype delta force select"></p>

fix(gpu): a saved-EXEC wave mask must survive the NGG fetch-index fold — R-Type Delta reaches its title screen (#2061)

`83e3383a` · [`assets/screenshots/rtype-delta-force-select.png`](assets/screenshots/rtype-delta-force-select.png)

### rtype-delta-title.png

<p align="center"><img src="assets/screenshots/rtype-delta-title.png" alt="rtype delta title"></p>

fix(gpu): a saved-EXEC wave mask must survive the NGG fetch-index fold — R-Type Delta reaches its title screen (#2061)

`83e3383a` · [`assets/screenshots/rtype-delta-title.png`](assets/screenshots/rtype-delta-title.png)

### crisis-core-main-menu.png

<p align="center"><img src="assets/screenshots/crisis-core-main-menu.png" alt="crisis core main menu"></p>

Crisis Core (PPSA07809) reaches rung 2 — and its #1945 fault is a race the guest wins, not a late write (#2060)

`e311e6cd` · [`assets/screenshots/crisis-core-main-menu.png`](assets/screenshots/crisis-core-main-menu.png)

### crisis-core-title.png

<p align="center"><img src="assets/screenshots/crisis-core-title.png" alt="crisis core title"></p>

Crisis Core (PPSA07809) reaches rung 2 — and its #1945 fault is a race the guest wins, not a late write (#2060)

`e311e6cd` · [`assets/screenshots/crisis-core-title.png`](assets/screenshots/crisis-core-title.png)

### crisis-core-voice-language.png

<p align="center"><img src="assets/screenshots/crisis-core-voice-language.png" alt="crisis core voice language"></p>

Crisis Core (PPSA07809) reaches rung 2 — and its #1945 fault is a race the guest wins, not a late write (#2060)

`e311e6cd` · [`assets/screenshots/crisis-core-voice-language.png`](assets/screenshots/crisis-core-voice-language.png)

### sonic-frontiers-middleware-credits.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-middleware-credits.png" alt="sonic frontiers middleware credits"></p>

fix(present): publish the flipped VideoOut buffer when no pass ever targets it — with a real guest-authorship test and the composited-frame gate intact (#2068)

`33dac763` · [`assets/screenshots/sonic-frontiers-middleware-credits.png`](assets/screenshots/sonic-frontiers-middleware-credits.png)

### sonic-frontiers-sega-logo.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-sega-logo.png" alt="sonic frontiers sega logo"></p>

fix(present): publish the flipped VideoOut buffer when no pass ever targets it — with a real guest-authorship test and the composited-frame gate intact (#2068)

`33dac763` · [`assets/screenshots/sonic-frontiers-sega-logo.png`](assets/screenshots/sonic-frontiers-sega-logo.png)

### rtype-delta-opening-movie-colour.png

<p align="center"><img src="assets/screenshots/rtype-delta-opening-movie-colour.png" alt="rtype delta opening movie colour"></p>

fix(gpu): recognise an AvPlayer chroma plane declared as a one-layer 2D array (#2037)

`08d75128` · [`assets/screenshots/rtype-delta-opening-movie-colour.png`](assets/screenshots/rtype-delta-opening-movie-colour.png)

### issue-1946-health-warning-before-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1946-health-warning-before-after.png" alt="issue 1946 health warning before after"></p>

fix(agc): offer the render-target-0 blend key on every SDK version — The Oregon Trail's whole UI layer was unblended (#1946) (#2031)

`beeff2ab` · [`prosper/docs/screenshots/issue-1946-health-warning-before-after.png`](prosper/docs/screenshots/issue-1946-health-warning-before-after.png)

### issue-1946-slate-blend-before-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1946-slate-blend-before-after.png" alt="issue 1946 slate blend before after"></p>

fix(agc): offer the render-target-0 blend key on every SDK version — The Oregon Trail's whole UI layer was unblended (#1946) (#2031)

`beeff2ab` · [`prosper/docs/screenshots/issue-1946-slate-blend-before-after.png`](prosper/docs/screenshots/issue-1946-slate-blend-before-after.png)

### sonic-crossworlds-sega-logo.png

<p align="center"><img src="assets/screenshots/sonic-crossworlds-sega-logo.png" alt="sonic crossworlds sega logo"></p>

docs(crossworlds): Sonic Racing: CrossWorlds reaches rung 1 — the SEGA logo renders (#2039)

`f8b0f040` · [`assets/screenshots/sonic-crossworlds-sega-logo.png`](assets/screenshots/sonic-crossworlds-sega-logo.png)

### little-nightmares-3-eula.png

<p align="center"><img src="assets/screenshots/little-nightmares-3-eula.png" alt="little nightmares 3 eula"></p>

docs(little-nightmares-3): #2014 is a wrong background clear, not a channel map — plus the first input route (#2041)

`f811615a` · [`assets/screenshots/little-nightmares-3-eula.png`](assets/screenshots/little-nightmares-3-eula.png)

## 2026-08-05

### little-nightmares-3-title-screen.png

<p align="center"><img src="assets/screenshots/little-nightmares-3-title-screen.png" alt="little nightmares 3 title screen"></p>

docs(little-nightmares-3): rung 2 — the title screen renders; land the ruled-out record (#2017)

`080263cf` · [`assets/screenshots/little-nightmares-3-title-screen.png`](assets/screenshots/little-nightmares-3-title-screen.png)

### rtype-delta-rung1-logo-and-opening-movie.png

<p align="center"><img src="assets/screenshots/rtype-delta-rung1-logo-and-opening-movie.png" alt="rtype delta rung1 logo and opening movie"></p>

docs(rtype): R-Type Delta reaches rung 1 — the #1746 startup race is host CPU speed, not a prosper defect (#2009)

`c3614f51` · [`assets/screenshots/rtype-delta-rung1-logo-and-opening-movie.png`](assets/screenshots/rtype-delta-rung1-logo-and-opening-movie.png)

### oregon-trail-title-screen.png

<p align="center"><img src="assets/screenshots/oregon-trail-title-screen.png" alt="oregon trail title screen"></p>

diag(fault): probe every guest-pointer register at a worker fault, not just rdx/rax (#1992)

`034f959a` · [`assets/screenshots/oregon-trail-title-screen.png`](assets/screenshots/oregon-trail-title-screen.png)

### oregon-trail-gameloft-splash.png

<p align="center"><img src="assets/screenshots/oregon-trail-gameloft-splash.png" alt="oregon trail gameloft splash"></p>

fix(hle): scePthread* must return libkernel-encoded errors — Oregon Trail advances three checkpoints (#1984)

`38619f29` · [`assets/screenshots/oregon-trail-gameloft-splash.png`](assets/screenshots/oregon-trail-gameloft-splash.png)

### oregon-trail-health-warning.png

<p align="center"><img src="assets/screenshots/oregon-trail-health-warning.png" alt="oregon trail health warning"></p>

fix(hle): scePthread* must return libkernel-encoded errors — Oregon Trail advances three checkpoints (#1984)

`38619f29` · [`assets/screenshots/oregon-trail-health-warning.png`](assets/screenshots/oregon-trail-health-warning.png)

### bendy-dark-revival-title.png

<p align="center"><img src="assets/screenshots/bendy-dark-revival-title.png" alt="bendy dark revival title"></p>

fix(avplayer): end a source nobody consumes on its own media clock (#1981)

`afe4a2b0` · [`assets/screenshots/bendy-dark-revival-title.png`](assets/screenshots/bendy-dark-revival-title.png)

### asterix-babylon-intro-cutscene.png

<p align="center"><img src="assets/screenshots/asterix-babylon-intro-cutscene.png" alt="asterix babylon intro cutscene"></p>

feat(avplayer): implement sceAvPlayerJumpToTime and honour the guest file-replacement reader (#1974)

`ff72e77c` · [`assets/screenshots/asterix-babylon-intro-cutscene.png`](assets/screenshots/asterix-babylon-intro-cutscene.png)

### asterix-babylon-title.png

<p align="center"><img src="assets/screenshots/asterix-babylon-title.png" alt="asterix babylon title"></p>

feat(avplayer): implement sceAvPlayerJumpToTime and honour the guest file-replacement reader (#1974)

`ff72e77c` · [`assets/screenshots/asterix-babylon-title.png`](assets/screenshots/asterix-babylon-title.png)

### little-nightmares-3-boot-splash.png

<p align="center"><img src="assets/screenshots/little-nightmares-3-boot-splash.png" alt="little nightmares 3 boot splash"></p>

fix(ajm): accept the config revision — Little Nightmares III reaches rung 1 (#1966)

`1fc8ece9` · [`assets/screenshots/little-nightmares-3-boot-splash.png`](assets/screenshots/little-nightmares-3-boot-splash.png)

### sonic-frontiers-opening-sequence.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-opening-sequence.png" alt="sonic frontiers opening sequence"></p>

docs(sonic-frontiers): rung 1 — it renders; the rung-0 reading was a failure-only diagnostic (#1969)

`440d84d2` · [`assets/screenshots/sonic-frontiers-opening-sequence.png`](assets/screenshots/sonic-frontiers-opening-sequence.png)

### sonic-frontiers-sonic-team-logo.png

<p align="center"><img src="assets/screenshots/sonic-frontiers-sonic-team-logo.png" alt="sonic frontiers sonic team logo"></p>

docs(sonic-frontiers): rung 1 — it renders; the rung-0 reading was a failure-only diagnostic (#1969)

`440d84d2` · [`assets/screenshots/sonic-frontiers-sonic-team-logo.png`](assets/screenshots/sonic-frontiers-sonic-team-logo.png)

## 2026-08-04

### oregon-trail-legal-popup.png

<p align="center"><img src="assets/screenshots/oregon-trail-legal-popup.png" alt="oregon trail legal popup"></p>

feat(oregon): reach rung 1 — the startup legal popup renders on a default launch (#1950)

`e92d2f7f` · [`assets/screenshots/oregon-trail-legal-popup.png`](assets/screenshots/oregon-trail-legal-popup.png)

### beneath-title.png

<p align="center"><img src="assets/screenshots/beneath-title.png" alt="beneath title"></p>

docs: record Beneath title screen

`0dafd22d` · [`assets/screenshots/beneath-title.png`](assets/screenshots/beneath-title.png)

### forgotten-city-title.png

<p align="center"><img src="assets/screenshots/forgotten-city-title.png" alt="forgotten city title"></p>

docs: record first-run compatibility baselines

`1640bb30` · [`assets/screenshots/forgotten-city-title.png`](assets/screenshots/forgotten-city-title.png)

### tactics-ogre-hevc-movie.png

<p align="center"><img src="assets/screenshots/tactics-ogre-hevc-movie.png" alt="tactics ogre hevc movie"></p>

docs(tactics-ogre): record restored HEVC presentation

`0d2f9a9f` · [`assets/screenshots/tactics-ogre-hevc-movie.png`](assets/screenshots/tactics-ogre-hevc-movie.png)

### tactics-ogre-reborn-gameplay.png

<p align="center"><img src="assets/screenshots/tactics-ogre-reborn-gameplay.png" alt="tactics ogre reborn gameplay"></p>

Document Tactics Ogre tutorial battle

`038c166d` · [`assets/screenshots/tactics-ogre-reborn-gameplay.png`](assets/screenshots/tactics-ogre-reborn-gameplay.png)

## 2026-08-03

### tactics-ogre-reborn-opening-scene.png

<p align="center"><img src="assets/screenshots/tactics-ogre-reborn-opening-scene.png" alt="tactics ogre reborn opening scene"></p>

Document Tactics Ogre opening scene

`4c8e3997` · [`assets/screenshots/tactics-ogre-reborn-opening-scene.png`](assets/screenshots/tactics-ogre-reborn-opening-scene.png)

### house-of-the-dead-2-remake-gameplay.png

<p align="center"><img src="assets/screenshots/house-of-the-dead-2-remake-gameplay.png" alt="house of the dead 2 remake gameplay"></p>

Document House of the Dead 2 gameplay

`b01e057c` · [`assets/screenshots/house-of-the-dead-2-remake-gameplay.png`](assets/screenshots/house-of-the-dead-2-remake-gameplay.png)

### tactics-ogre-title.png

<p align="center"><img src="assets/screenshots/tactics-ogre-title.png" alt="tactics ogre title"></p>

Add Tactics Ogre title milestone

`4d192dc1` · [`assets/screenshots/tactics-ogre-title.png`](assets/screenshots/tactics-ogre-title.png)

### house-of-the-dead-2-remake-title.png

<p align="center"><img src="assets/screenshots/house-of-the-dead-2-remake-title.png" alt="house of the dead 2 remake title"></p>

Document newly tested game compatibility

`fbde2b4c` · [`assets/screenshots/house-of-the-dead-2-remake-title.png`](assets/screenshots/house-of-the-dead-2-remake-title.png)

### rtype-delta-movie-coordinate-progress.png

<p align="center"><img src="assets/screenshots/rtype-delta-movie-coordinate-progress.png" alt="rtype delta movie coordinate progress"></p>

fix(gpu): honor unnormalized sample coordinates

`d4fa07a8` · [`assets/screenshots/rtype-delta-movie-coordinate-progress.png`](assets/screenshots/rtype-delta-movie-coordinate-progress.png)

### blue-prince-hall.png

<p align="center"><img src="assets/screenshots/blue-prince-hall.png" alt="blue prince hall"></p>

Fix Blue Prince hall snapshot guard (#1813)

`730d434e` · [`assets/screenshots/blue-prince-hall.png`](assets/screenshots/blue-prince-hall.png)

## 2026-08-02

### earthion-title-menu.png

<p align="center"><img src="assets/screenshots/earthion-title-menu.png" alt="earthion title menu"></p>

feat(earthion): route past the intro — the "missing picture" was a black text page (#1590) (#1775)

`3d1a7480` · [`assets/screenshots/earthion-title-menu.png`](assets/screenshots/earthion-title-menu.png)

### tales-graces-f-options.png

<p align="center"><img src="assets/screenshots/tales-graces-f-options.png" alt="tales graces f options"></p>

feat(talesgraces): routes to the title screen and the new-game Options screen (PPSA19991 rung 2) (#1759)

`30477a2d` · [`assets/screenshots/tales-graces-f-options.png`](assets/screenshots/tales-graces-f-options.png)

### tales-graces-f-title.png

<p align="center"><img src="assets/screenshots/tales-graces-f-title.png" alt="tales graces f title"></p>

feat(talesgraces): routes to the title screen and the new-game Options screen (PPSA19991 rung 2) (#1759)

`30477a2d` · [`assets/screenshots/tales-graces-f-title.png`](assets/screenshots/tales-graces-f-title.png)

### bendy-gameplay.png

<p align="center"><img src="assets/screenshots/bendy-gameplay.png" alt="bendy gameplay"></p>

docs(compat): boot sweep of fourteen titles on 3a473bca — corrected rungs, four new rows, and a frame-rate census (#1740)

`8fc79ca0` · [`assets/screenshots/bendy-gameplay.png`](assets/screenshots/bendy-gameplay.png)

### bendy-title.png

<p align="center"><img src="assets/screenshots/bendy-title.png" alt="bendy title"></p>

docs(compat): boot sweep of fourteen titles on 3a473bca — corrected rungs, four new rows, and a frame-rate census (#1740)

`8fc79ca0` · [`assets/screenshots/bendy-title.png`](assets/screenshots/bendy-title.png)

### pathless-title.png

<p align="center"><img src="assets/screenshots/pathless-title.png" alt="pathless title"></p>

docs(compat): boot sweep of fourteen titles on 3a473bca — corrected rungs, four new rows, and a frame-rate census (#1740)

`8fc79ca0` · [`assets/screenshots/pathless-title.png`](assets/screenshots/pathless-title.png)

### plucky-squire-title.png

<p align="center"><img src="assets/screenshots/plucky-squire-title.png" alt="plucky squire title"></p>

docs(compat): boot sweep of fourteen titles on 3a473bca — corrected rungs, four new rows, and a frame-rate census (#1740)

`8fc79ca0` · [`assets/screenshots/plucky-squire-title.png`](assets/screenshots/plucky-squire-title.png)

### astro-bot-opening-cinematic.png

<p align="center"><img src="assets/screenshots/astro-bot-opening-cinematic.png" alt="astro bot opening cinematic"></p>

docs(astro): Astro Bot enters COMPATIBILITY.md at rung 2 — the title screen renders (#1736)

`6d7b69e9` · [`assets/screenshots/astro-bot-opening-cinematic.png`](assets/screenshots/astro-bot-opening-cinematic.png)

### astro-bot-title.png

<p align="center"><img src="assets/screenshots/astro-bot-title.png" alt="astro bot title"></p>

docs(astro): Astro Bot enters COMPATIBILITY.md at rung 2 — the title screen renders (#1736)

`6d7b69e9` · [`assets/screenshots/astro-bot-title.png`](assets/screenshots/astro-bot-title.png)

### astro-bot-worldmap-background.png

<p align="center"><img src="assets/screenshots/astro-bot-worldmap-background.png" alt="astro bot worldmap background"></p>

fix(gpu): CB_COLOR_CONTROL.MODE must not override the guest's colour write mask (#1728)

`c8fe4afd` · [`assets/screenshots/astro-bot-worldmap-background.png`](assets/screenshots/astro-bot-worldmap-background.png)

## 2026-08-01

### tales-graces-f-criware.png

<p align="center"><img src="assets/screenshots/tales-graces-f-criware.png" alt="tales graces f criware"></p>

fix(hle): deliver the APR completion event for a zero-tag binding (#1666) (#1667)

`fa1f07b3` · [`assets/screenshots/tales-graces-f-criware.png`](assets/screenshots/tales-graces-f-criware.png)

### tales-graces-f-publisher.png

<p align="center"><img src="assets/screenshots/tales-graces-f-publisher.png" alt="tales graces f publisher"></p>

fix(hle): deliver the APR completion event for a zero-tag binding (#1666) (#1667)

`fa1f07b3` · [`assets/screenshots/tales-graces-f-publisher.png`](assets/screenshots/tales-graces-f-publisher.png)

### issue-1630-grid-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-grid-after.png" alt="issue 1630 grid after"></p>

feat(app): per-title background art and focus music in the library (#1647)

`7da42075` · [`prosper/docs/screenshots/issue-1630-grid-after.png`](prosper/docs/screenshots/issue-1630-grid-after.png)

### issue-1630-grid-before.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-grid-before.png" alt="issue 1630 grid before"></p>

feat(app): per-title background art and focus music in the library (#1647)

`7da42075` · [`prosper/docs/screenshots/issue-1630-grid-before.png`](prosper/docs/screenshots/issue-1630-grid-before.png)

### issue-1630-library-background-1.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-library-background-1.png" alt="issue 1630 library background 1"></p>

feat(app): per-title background art and focus music in the library (#1647)

`7da42075` · [`prosper/docs/screenshots/issue-1630-library-background-1.png`](prosper/docs/screenshots/issue-1630-library-background-1.png)

### issue-1630-library-background-2.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-library-background-2.png" alt="issue 1630 library background 2"></p>

feat(app): per-title background art and focus music in the library (#1647)

`7da42075` · [`prosper/docs/screenshots/issue-1630-library-background-2.png`](prosper/docs/screenshots/issue-1630-library-background-2.png)

### issue-1630-library-background-3.png

<p align="center"><img src="prosper/docs/screenshots/issue-1630-library-background-3.png" alt="issue 1630 library background 3"></p>

feat(app): per-title background art and focus music in the library (#1647)

`7da42075` · [`prosper/docs/screenshots/issue-1630-library-background-3.png`](prosper/docs/screenshots/issue-1630-library-background-3.png)

### syberia-gameplay.png

<p align="center"><img src="assets/screenshots/syberia-gameplay.png" alt="syberia gameplay"></p>

docs(syberia): validated route to gameplay, and localize the black scene to one format gap (#1622)

`5ee5e785` · [`assets/screenshots/syberia-gameplay.png`](assets/screenshots/syberia-gameplay.png)

### syberia-title.png

<p align="center"><img src="assets/screenshots/syberia-title.png" alt="syberia title"></p>

docs(syberia): validated route to gameplay, and localize the black scene to one format gap (#1622)

`5ee5e785` · [`assets/screenshots/syberia-title.png`](assets/screenshots/syberia-title.png)

### worms-armageddon-gameplay.png

<p align="center"><img src="assets/screenshots/worms-armageddon-gameplay.png" alt="worms armageddon gameplay"></p>

fix(pad): scePadGetHandle looks up an open handle instead of fabricating one (#1623)

`823c9670` · [`assets/screenshots/worms-armageddon-gameplay.png`](assets/screenshots/worms-armageddon-gameplay.png)

## 2026-07-31

### syberia-profile.png

<p align="center"><img src="assets/screenshots/syberia-profile.png" alt="syberia profile"></p>

fix(agc): register sceAgcAcbWriteData — Syberia goes from hard hang to its profile menu (#1610)

`961a6cdd` · [`assets/screenshots/syberia-profile.png`](assets/screenshots/syberia-profile.png)

### nikoderiko-title.png

<p align="center"><img src="assets/screenshots/nikoderiko-title.png" alt="nikoderiko title"></p>

docs(compat): add Nikoderiko at title screen and The Oregon Trail at research tier (#1608)

`a933091d` · [`assets/screenshots/nikoderiko-title.png`](assets/screenshots/nikoderiko-title.png)

### greak-title.png

<p align="center"><img src="assets/screenshots/greak-title.png" alt="greak title"></p>

feat(recompiler): lower s_ttracedata — Greak and Rugrats reach gameplay (#1600)

`5a0eb7b6` · [`assets/screenshots/greak-title.png`](assets/screenshots/greak-title.png)

### greak.png

<p align="center"><img src="assets/screenshots/greak.png" alt="greak"></p>

feat(recompiler): lower s_ttracedata — Greak and Rugrats reach gameplay (#1600)

`5a0eb7b6` · [`assets/screenshots/greak.png`](assets/screenshots/greak.png)

### rugrats-title.png

<p align="center"><img src="assets/screenshots/rugrats-title.png" alt="rugrats title"></p>

feat(recompiler): lower s_ttracedata — Greak and Rugrats reach gameplay (#1600)

`5a0eb7b6` · [`assets/screenshots/rugrats-title.png`](assets/screenshots/rugrats-title.png)

### rugrats.png

<p align="center"><img src="assets/screenshots/rugrats.png" alt="rugrats"></p>

feat(recompiler): lower s_ttracedata — Greak and Rugrats reach gameplay (#1600)

`5a0eb7b6` · [`assets/screenshots/rugrats.png`](assets/screenshots/rugrats.png)

### asterix-slap-them-all.png

<p align="center"><img src="assets/screenshots/asterix-slap-them-all.png" alt="asterix slap them all"></p>

docs(compat): add Asterix Slap Them All and Summer Sports Games at gameplay (#1604)

`2421d503` · [`assets/screenshots/asterix-slap-them-all.png`](assets/screenshots/asterix-slap-them-all.png)

### summer-sports-games.png

<p align="center"><img src="assets/screenshots/summer-sports-games.png" alt="summer sports games"></p>

docs(compat): add Asterix Slap Them All and Summer Sports Games at gameplay (#1604)

`2421d503` · [`assets/screenshots/summer-sports-games.png`](assets/screenshots/summer-sports-games.png)

### joe-mac-menu.png

<p align="center"><img src="assets/screenshots/joe-mac-menu.png" alt="joe mac menu"></p>

docs(compat): record five newly triaged titles, two of them rendering (#1596)

`bb5a11a2` · [`assets/screenshots/joe-mac-menu.png`](assets/screenshots/joe-mac-menu.png)

### joe-mac.png

<p align="center"><img src="assets/screenshots/joe-mac.png" alt="joe mac"></p>

docs(compat): record five newly triaged titles, two of them rendering (#1596)

`bb5a11a2` · [`assets/screenshots/joe-mac.png`](assets/screenshots/joe-mac.png)

### worms-armageddon-title.png

<p align="center"><img src="assets/screenshots/worms-armageddon-title.png" alt="worms armageddon title"></p>

docs(compat): record five newly triaged titles, two of them rendering (#1596)

`bb5a11a2` · [`assets/screenshots/worms-armageddon-title.png`](assets/screenshots/worms-armageddon-title.png)

### alex-kidd.png

<p align="center"><img src="assets/screenshots/alex-kidd.png" alt="alex kidd"></p>

test(snapshot): reviewed alexkidd-gameplay content guard — PPSA02664 reaches ladder rung 6 (#1582)

`a6d995fa` · [`assets/screenshots/alex-kidd.png`](assets/screenshots/alex-kidd.png)

### dragon-quest-vii-onboarding.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-onboarding.png" alt="dragon quest vii onboarding"></p>

docs: record Dragon Quest VII onboarding

`aeee4d48` · [`assets/screenshots/dragon-quest-vii-onboarding.png`](assets/screenshots/dragon-quest-vii-onboarding.png)

### dragon-quest-vii-name-confirmation.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-name-confirmation.png" alt="dragon quest vii name confirmation"></p>

Document DQ7 name confirmation milestone

`d2fe1c66` · [`assets/screenshots/dragon-quest-vii-name-confirmation.png`](assets/screenshots/dragon-quest-vii-name-confirmation.png)

### dragon-quest-vii-name-entry.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-name-entry.png" alt="dragon quest vii name entry"></p>

Document Dragon Quest name entry

`297ec493` · [`assets/screenshots/dragon-quest-vii-name-entry.png`](assets/screenshots/dragon-quest-vii-name-entry.png)

### space-adventure-cobra.png

<p align="center"><img src="assets/screenshots/space-adventure-cobra.png" alt="space adventure cobra"></p>

fix(runtime): preserve guest TLS across write-watch faults

`82eadf58` · [`assets/screenshots/space-adventure-cobra.png`](assets/screenshots/space-adventure-cobra.png)

### gris.png

<p align="center"><img src="assets/screenshots/gris.png" alt="gris"></p>

docs: record GRIS opening gameplay

`b08f0a94` · [`assets/screenshots/gris.png`](assets/screenshots/gris.png)

### issue-1459-astrobot-blue-fmv-gpu-present.png

<p align="center"><img src="prosper/docs/screenshots/issue-1459-astrobot-blue-fmv-gpu-present.png" alt="issue 1459 astrobot blue fmv gpu present"></p>

docs: capture Astro Bot blue intro

`3f72a8ce` · [`prosper/docs/screenshots/issue-1459-astrobot-blue-fmv-gpu-present.png`](prosper/docs/screenshots/issue-1459-astrobot-blue-fmv-gpu-present.png)

## 2026-07-30

### issue-1471-library-empty.png

<p align="center"><img src="prosper/docs/screenshots/issue-1471-library-empty.png" alt="issue 1471 library empty"></p>

fix(app): address review findings on the library view

`44d1689d` · [`prosper/docs/screenshots/issue-1471-library-empty.png`](prosper/docs/screenshots/issue-1471-library-empty.png)

### issue-1471-library-scrolled.png

<p align="center"><img src="prosper/docs/screenshots/issue-1471-library-scrolled.png" alt="issue 1471 library scrolled"></p>

fix(app): address review findings on the library view

`44d1689d` · [`prosper/docs/screenshots/issue-1471-library-scrolled.png`](prosper/docs/screenshots/issue-1471-library-scrolled.png)

### issue-1471-library-grid.png

<p align="center"><img src="prosper/docs/screenshots/issue-1471-library-grid.png" alt="issue 1471 library grid"></p>

feat(app): draw the game library with Dear ImGui

`7cf767fe` · [`prosper/docs/screenshots/issue-1471-library-grid.png`](prosper/docs/screenshots/issue-1471-library-grid.png)

### issue-1459-astrobot-linux-indirect-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-1459-astrobot-linux-indirect-title.png" alt="issue 1459 astrobot linux indirect title"></p>

gpu: execute AGC indirect work after producers

`e85c527c` · [`prosper/docs/screenshots/issue-1459-astrobot-linux-indirect-title.png`](prosper/docs/screenshots/issue-1459-astrobot-linux-indirect-title.png)

## 2026-07-29

### issue-1469-drop-messenger.png

<p align="center"><img src="prosper/docs/screenshots/issue-1469-drop-messenger.png" alt="issue 1469 drop messenger"></p>

docs(app): interactive-open evidence screenshots (#1469)

`3f7f9929` · [`prosper/docs/screenshots/issue-1469-drop-messenger.png`](prosper/docs/screenshots/issue-1469-drop-messenger.png)

### issue-1469-picker-messenger.png

<p align="center"><img src="prosper/docs/screenshots/issue-1469-picker-messenger.png" alt="issue 1469 picker messenger"></p>

docs(app): interactive-open evidence screenshots (#1469)

`3f7f9929` · [`prosper/docs/screenshots/issue-1469-picker-messenger.png`](prosper/docs/screenshots/issue-1469-picker-messenger.png)

### issue-1469-reject-not-a-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-1469-reject-not-a-title.png" alt="issue 1469 reject not a title"></p>

docs(app): interactive-open evidence screenshots (#1469)

`3f7f9929` · [`prosper/docs/screenshots/issue-1469-reject-not-a-title.png`](prosper/docs/screenshots/issue-1469-reject-not-a-title.png)

### issue-1469-relaunch-blasphemous2.png

<p align="center"><img src="prosper/docs/screenshots/issue-1469-relaunch-blasphemous2.png" alt="issue 1469 relaunch blasphemous2"></p>

docs(app): interactive-open evidence screenshots (#1469)

`3f7f9929` · [`prosper/docs/screenshots/issue-1469-relaunch-blasphemous2.png`](prosper/docs/screenshots/issue-1469-relaunch-blasphemous2.png)

### issue-1459-astrobot-worldmap-current.png

<p align="center"><img src="prosper/docs/screenshots/issue-1459-astrobot-worldmap-current.png" alt="issue 1459 astrobot worldmap current"></p>

docs: capture current Astro Bot world map

`2e83d1ea` · [`prosper/docs/screenshots/issue-1459-astrobot-worldmap-current.png`](prosper/docs/screenshots/issue-1459-astrobot-worldmap-current.png)

### issue-1466-astrobot-direct-tile.png

<p align="center"><img src="prosper/docs/screenshots/issue-1466-astrobot-direct-tile.png" alt="issue 1466 astrobot direct tile"></p>

perf(gpu): tile mapped storage images directly

`1b8eeed6` · [`prosper/docs/screenshots/issue-1466-astrobot-direct-tile.png`](prosper/docs/screenshots/issue-1466-astrobot-direct-tile.png)

## 2026-07-26

### issue-1287-hall-live-fixed.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-live-fixed.png" alt="issue 1287 hall live fixed"></p>

docs: live Blue Prince gameplay at oracle parity (#1287 rung-5 evidence) (#1438)

`ffbb7d74` · [`prosper/docs/screenshots/issue-1287-hall-live-fixed.png`](prosper/docs/screenshots/issue-1287-hall-live-fixed.png)

### issue-1287-hall-live-vs-oracle.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-live-vs-oracle.png" alt="issue 1287 hall live vs oracle"></p>

docs: live Blue Prince gameplay at oracle parity (#1287 rung-5 evidence) (#1438)

`ffbb7d74` · [`prosper/docs/screenshots/issue-1287-hall-live-vs-oracle.png`](prosper/docs/screenshots/issue-1287-hall-live-vs-oracle.png)

### issue-1287-manor-approach-live.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-manor-approach-live.png" alt="issue 1287 manor approach live"></p>

docs: live Blue Prince gameplay at oracle parity (#1287 rung-5 evidence) (#1438)

`ffbb7d74` · [`prosper/docs/screenshots/issue-1287-manor-approach-live.png`](prosper/docs/screenshots/issue-1287-manor-approach-live.png)

### issue-1427-hall-geometry-restored.png

<p align="center"><img src="prosper/docs/screenshots/issue-1427-hall-geometry-restored.png" alt="issue 1427 hall geometry restored"></p>

fix(render): upload a buffer binding's whole declared range, not the first 1 MiB (#1429)

`ad5a840a` · [`prosper/docs/screenshots/issue-1427-hall-geometry-restored.png`](prosper/docs/screenshots/issue-1427-hall-geometry-restored.png)

### issue-1427-oracle-before-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1427-oracle-before-after.png" alt="issue 1427 oracle before after"></p>

fix(render): upload a buffer binding's whole declared range, not the first 1 MiB (#1429)

`ad5a840a` · [`prosper/docs/screenshots/issue-1427-oracle-before-after.png`](prosper/docs/screenshots/issue-1427-oracle-before-after.png)

### issue-1287-hall-materials-fixed.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-materials-fixed.png" alt="issue 1287 hall materials fixed"></p>

docs: Blue Prince hall with correct materials (#1287 milestone frame) (#1418)

`0f7d9310` · [`prosper/docs/screenshots/issue-1287-hall-materials-fixed.png`](prosper/docs/screenshots/issue-1287-hall-materials-fixed.png)

## 2026-07-25

### issue-1334-hall-default-tonemapped.png

<p align="center"><img src="prosper/docs/screenshots/issue-1334-hall-default-tonemapped.png" alt="issue 1334 hall default tonemapped"></p>

fix(gpu): GPU-copy the MSAA resolve into the destination persistent image (#1382)

`6479cd5f` · [`prosper/docs/screenshots/issue-1334-hall-default-tonemapped.png`](prosper/docs/screenshots/issue-1334-hall-default-tonemapped.png)

### issue-1287-hall-bundle-tonemapped.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-bundle-tonemapped.png" alt="issue 1287 hall bundle tonemapped"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`a3613436` · [`prosper/docs/screenshots/issue-1287-hall-bundle-tonemapped.png`](prosper/docs/screenshots/issue-1287-hall-bundle-tonemapped.png)

### issue-1287-hall-magenta-prosper.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-magenta-prosper.png" alt="issue 1287 hall magenta prosper"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`a3613436` · [`prosper/docs/screenshots/issue-1287-hall-magenta-prosper.png`](prosper/docs/screenshots/issue-1287-hall-magenta-prosper.png)

### issue-1287-hall-night-prosper.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-night-prosper.png" alt="issue 1287 hall night prosper"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`a3613436` · [`prosper/docs/screenshots/issue-1287-hall-night-prosper.png`](prosper/docs/screenshots/issue-1287-hall-night-prosper.png)

### issue-1287-hall-nobatch-live.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-hall-nobatch-live.png" alt="issue 1287 hall nobatch live"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`a3613436` · [`prosper/docs/screenshots/issue-1287-hall-nobatch-live.png`](prosper/docs/screenshots/issue-1287-hall-nobatch-live.png)

### issue-1287-vestibule-prosper.png

<p align="center"><img src="prosper/docs/screenshots/issue-1287-vestibule-prosper.png" alt="issue 1287 vestibule prosper"></p>

docs: current Blue Prince hall frames for the #1287 oracle request (#1375)

`a3613436` · [`prosper/docs/screenshots/issue-1287-vestibule-prosper.png`](prosper/docs/screenshots/issue-1287-vestibule-prosper.png)

### issue-1356-gris-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-1356-gris-title.png" alt="issue 1356 gris title"></p>

feat: bring GRIS and Cobra to title with audio (#1368)

`8b37be95` · [`prosper/docs/screenshots/issue-1356-gris-title.png`](prosper/docs/screenshots/issue-1356-gris-title.png)

### issue-1356-space-adventure-cobra-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-1356-space-adventure-cobra-title.png" alt="issue 1356 space adventure cobra title"></p>

feat: bring GRIS and Cobra to title with audio (#1368)

`8b37be95` · [`prosper/docs/screenshots/issue-1356-space-adventure-cobra-title.png`](prosper/docs/screenshots/issue-1356-space-adventure-cobra-title.png)

### dragon-quest-vii-title.png

<p align="center"><img src="assets/screenshots/dragon-quest-vii-title.png" alt="dragon quest vii title"></p>

docs: publish Dragon Quest VII title capture

`18abdf28` · [`assets/screenshots/dragon-quest-vii-title.png`](assets/screenshots/dragon-quest-vii-title.png)

### issue-1352-wall-shading-after.png

<p align="center"><img src="prosper/docs/screenshots/issue-1352-wall-shading-after.png" alt="issue 1352 wall shading after"></p>

fix(gpu): DEPTH_CLEAR_ENABLE acts only through the enabled depth-write path (#1354)

`feb5822d` · [`prosper/docs/screenshots/issue-1352-wall-shading-after.png`](prosper/docs/screenshots/issue-1352-wall-shading-after.png)

### issue-1352-wall-shading-before.png

<p align="center"><img src="prosper/docs/screenshots/issue-1352-wall-shading-before.png" alt="issue 1352 wall shading before"></p>

fix(gpu): DEPTH_CLEAR_ENABLE acts only through the enabled depth-write path (#1354)

`feb5822d` · [`prosper/docs/screenshots/issue-1352-wall-shading-before.png`](prosper/docs/screenshots/issue-1352-wall-shading-before.png)

## 2026-07-24

### blue-prince-title.png

<p align="center"><img src="assets/screenshots/blue-prince-title.png" alt="blue prince title"></p>

Add Blue Prince and Terminator docs screenshots (#1342)

`5e11d900` · [`assets/screenshots/blue-prince-title.png`](assets/screenshots/blue-prince-title.png)

### terminator-title.png

<p align="center"><img src="assets/screenshots/terminator-title.png" alt="terminator title"></p>

Add Blue Prince and Terminator docs screenshots (#1342)

`5e11d900` · [`assets/screenshots/terminator-title.png`](assets/screenshots/terminator-title.png)

### terminator.png

<p align="center"><img src="assets/screenshots/terminator.png" alt="terminator"></p>

Add Blue Prince and Terminator docs screenshots (#1342)

`5e11d900` · [`assets/screenshots/terminator.png`](assets/screenshots/terminator.png)

### gta5-main-menu.png

<p align="center"><img src="assets/screenshots/gta5-main-menu.png" alt="gta5 main menu"></p>

docs: show GTA V current renderer state (#1339)

`e6b8fb06` · [`assets/screenshots/gta5-main-menu.png`](assets/screenshots/gta5-main-menu.png)

### gta5-title.png

<p align="center"><img src="assets/screenshots/gta5-title.png" alt="gta5 title"></p>

docs: show GTA V current renderer state (#1339)

`e6b8fb06` · [`assets/screenshots/gta5-title.png`](assets/screenshots/gta5-title.png)

### blasphemous2-title.png

<p align="center"><img src="assets/screenshots/blasphemous2-title.png" alt="blasphemous2 title"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`958979f6` · [`assets/screenshots/blasphemous2-title.png`](assets/screenshots/blasphemous2-title.png)

### blasphemous2.png

<p align="center"><img src="assets/screenshots/blasphemous2.png" alt="blasphemous2"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`958979f6` · [`assets/screenshots/blasphemous2.png`](assets/screenshots/blasphemous2.png)

### dead-cells-title.png

<p align="center"><img src="assets/screenshots/dead-cells-title.png" alt="dead cells title"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`958979f6` · [`assets/screenshots/dead-cells-title.png`](assets/screenshots/dead-cells-title.png)

### dead-cells.png

<p align="center"><img src="assets/screenshots/dead-cells.png" alt="dead cells"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`958979f6` · [`assets/screenshots/dead-cells.png`](assets/screenshots/dead-cells.png)

### evergate-title.png

<p align="center"><img src="assets/screenshots/evergate-title.png" alt="evergate title"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`958979f6` · [`assets/screenshots/evergate-title.png`](assets/screenshots/evergate-title.png)

### evergate.png

<p align="center"><img src="assets/screenshots/evergate.png" alt="evergate"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`958979f6` · [`assets/screenshots/evergate.png`](assets/screenshots/evergate.png)

### messenger-title.png

<p align="center"><img src="assets/screenshots/messenger-title.png" alt="messenger title"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`958979f6` · [`assets/screenshots/messenger-title.png`](assets/screenshots/messenger-title.png)

### messenger.png

<p align="center"><img src="assets/screenshots/messenger.png" alt="messenger"></p>

docs: refresh public README + COMPATIBILITY with screenshots and current status

`958979f6` · [`assets/screenshots/messenger.png`](assets/screenshots/messenger.png)

## 2026-07-19

### issue-897-astrobot-linux-natural-opening-midfade.png

<p align="center"><img src="prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-midfade.png" alt="issue 897 astrobot linux natural opening midfade"></p>

docs(astrobot): attach natural Linux graphics captures

`2a09b44d` · [`prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-midfade.png`](prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-midfade.png)

### issue-897-astrobot-linux-natural-opening-visible.png

<p align="center"><img src="prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-visible.png" alt="issue 897 astrobot linux natural opening visible"></p>

docs(astrobot): attach natural Linux graphics captures

`2a09b44d` · [`prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-visible.png`](prosper/docs/screenshots/issue-897-astrobot-linux-natural-opening-visible.png)

## 2026-07-18

### issue-825-astrobot-windows-sony-presents.png

<p align="center"><img src="prosper/docs/screenshots/issue-825-astrobot-windows-sony-presents.png" alt="issue 825 astrobot windows sony presents"></p>

docs(astrobot): attach Windows progress captures

`815a84b2` · [`prosper/docs/screenshots/issue-825-astrobot-windows-sony-presents.png`](prosper/docs/screenshots/issue-825-astrobot-windows-sony-presents.png)

### issue-825-astrobot-windows-title.png

<p align="center"><img src="prosper/docs/screenshots/issue-825-astrobot-windows-title.png" alt="issue 825 astrobot windows title"></p>

docs(astrobot): attach Windows progress captures

`815a84b2` · [`prosper/docs/screenshots/issue-825-astrobot-windows-title.png`](prosper/docs/screenshots/issue-825-astrobot-windows-title.png)

## 2026-07-17

### issue-825-astrobot-linux-sony-presents.png

<p align="center"><img src="prosper/docs/screenshots/issue-825-astrobot-linux-sony-presents.png" alt="issue 825 astrobot linux sony presents"></p>

docs(astrobot): attach Linux loading screenshot

`a1395e75` · [`prosper/docs/screenshots/issue-825-astrobot-linux-sony-presents.png`](prosper/docs/screenshots/issue-825-astrobot-linux-sony-presents.png)
