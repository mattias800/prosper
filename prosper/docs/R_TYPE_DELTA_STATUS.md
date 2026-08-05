# R-Type Delta: HD Boosted status

`PPSA26414` is **rung 1** — the publisher logo and the whole opening movie render as real,
recognizable frames from the live renderer, with a chroma defect. A default launch still dies first
on the title's logged-in-user startup race (#1746), which is now measured to be decided by host
single-thread CPU speed and is a product decision rather than an open investigation; the rung-1
frames were taken on an unmodified binary whose *host* was CPU-contended so the title wins that race.
Start from **`## The race is decided by host single-thread CPU speed`** and **`## Rung 1 reached`**
below; everything before them is the older movie-pipeline record (#1753, #1807, both closed).

## Movie resource failure

The rejected fragment shader is `shader/movie_yuv_p.ags`. Its AGC metadata declares two direct
textures at user SGPRs 0 and 8, two direct samplers at SGPRs 16 and 20, and one constant buffer at
SGPR 24. The shader samples texture slot 1 / sampler slot 1 first at pc 52 (`T# s8`, `S# s20`), then
texture slot 0 / sampler slot 0 at pc 57. The pc-52 instruction is a supported 2D `IMAGE_SAMPLE`;
the failure is that texture slot 1 was never constructed.

Guest disassembly at `eboot+0x19d10` explains why:

1. The title reads `AvPlayerFrameInfoEx::video.height` at frame offset `0x1c` and `pitch` at `0x3c`.
2. It constructs the luma R8 T# using `pitch` as the descriptor width and `height` as its height.
3. It asks the guest AGC library for that descriptor's allocation size and requires it to equal
   `pitch * height`.
4. Prosper previously published a tight 1920-byte pitch. The AGC sampled-linear footprint is
   256-byte aligned, so the descriptor occupies `2048 * 1080 = 2,211,840` bytes while the title
   expects `1920 * 1080 = 2,073,600`. The title prints `unexpected y_texture size` and returns
   before constructing the chroma T# in texture slot 1.
5. The one-time initialization flag has already been cleared. Later frames bind the valid luma T#
   and the never-constructed chroma T#, so pc 52 cannot resolve SGPR 8.

The PS5 `AvPlayerVideoEx` ABI keeps physical and visible extents separate: crop offsets are four
32-bit pixel counts at offsets 20, 24, 28 and 32, followed by the physical pitch at offset 36.
Prosper now stages 1920 visible bytes into a 2048-byte physical row and publishes
`crop_right_offset=128`. R-Type's own movie code subtracts left/right crop from pitch when it builds
the visible quad, so the padding is not displayed.

This is also the cross-title contract needed by GRIS. Its PS5 video wrapper computes visible width
as `pitch - crop_left_offset - crop_right_offset`. The earlier experiment in #1393 that padded the
pitch but left all crop fields zero correctly removed the stride corruption, but exposed the
128-pixel tail as a right-edge strip. Padding and crop are one contract; testing only the first half
made the correct physical layout look invalid.

## CPU guard and mutation evidence

`test_avplayer` supplies a 1920x1080 NV12 frame with 2048-byte decoder strides and checks all of the
following without launching Vulkan:

- the callback-owned frame ring has 2048-byte luma and chroma rows;
- only 1920 visible bytes per row are copied and padding remains cleared;
- stream and frame metadata publish pitch 2048, crop right 128 and zero for the other crop offsets;
- luma and chroma guest addresses preserve exact 2048-byte layout provenance;
- the R-Type descriptor footprint equals `pitch * height`;
- the GRIS visible-width expression evaluates back to 1920.

Two mutations prove the checks exercise both failure mechanisms. Restoring `pitch=width` makes the
named R-Type footprint check fail. Keeping pitch 2048 but forcing crop right to zero leaves the
R-Type footprint check green and makes the named GRIS right-edge-strip check fail.

## Live validation

Candidate `4114391c` was run for 60 seconds through the bounded #1746 startup diagnostic. The three
byte patches applied at the expected addresses, AvPlayer opened `CRG.mp4`, allocated two 3,317,760
byte guest texture buffers, returned those buffers alternately, and advanced timestamps. All #1753
failure signals were absent: no `unexpected y_texture size`, no pc-52 `mimg-unresolved` from SGPR 8,
no pc-52 `recompile-reject` for `f080030a`, and no execution reject for fragment program
`0x2011c0d700`. No other shader/resource rejection appeared. This proves the upstream descriptor
construction and shader-resolution defect is fixed.

It does **not** establish a visual milestone. All 49 retained 1920x1080 frames were inspected. Early
and late samples were solid clears; the only non-flat interval contained 2--75 colours per frame and
showed broad diagonal purple/grey gradients rather than recognizable movie imagery. There is no
screenshot because incorrect diagnostic output is not progression evidence. The remaining movie
sampling/content frontier is tracked separately in #1807.

The same candidate completed the GRIS cross-title route with 45 source-distinct and 42 pixel-distinct
frames, maximum pixel staleness 1.0 seconds, and status `ok`. All 45 frames were inspected. The
Nomada/Devolver opening logos and title screen have no 128-pixel right strip, horizontal stride
corruption, or U/V colour cast. This verifies that the padded-pitch/crop contract does not regress
the known consumer that exposed the earlier half-fix.

## Movie sampler-coordinate recovery

An F9 capture of present 1740 retained the next submit as a replayable two-draw frame. The first
draw uses fragment program `0x2011c0d700` and samples the staged movie planes at the exact AvPlayer
addresses: an R8 2048x1080 luma plane and an RG8 1024x540 chroma plane. Both resources are complete
and non-flat. Converting those captured bytes as padded NV12 on the CPU produces a sharp,
full-colour movie frame, while replaying the same submit on unmodified `dad8f518` produces the
purple diagonal gradient (`c87261d44cd4426b`). That separates decoder/staging correctness from the
downstream sampling failure without another title boot.

Both direct S# descriptors set `FORCE_UNNORMALIZED`, and the vertex shader supplies pixel/texel
coordinates rather than normalized UVs. The descriptor decoder already preserved that bit, but no
renderer consumer applied it. Treating coordinates such as 1920 as ordinary normalized values made
the sampler wrap or clamp almost the whole quad onto a few edge texels; the triangle-strip primitive
boundary made the resulting alias appear as a hard diagonal.

The generic lowering keeps the ordinary Vulkan sampler, preserving the guest's wrap modes and LOD
state, and scales only spatial sample coordinates and explicit gradients by the corresponding
resource extent in the shader recompiler. Array layers, DREF, LOD/bias, packed texel offsets and
integer image loads keep their established meanings. Because the reciprocal extents are embedded in
SPIR-V, the compiled-shader cache key includes the unnormalized contract and its dimensions.

Recompiling every retained raw shader on `2240551d` changes the exact replay hash to
`51d9511c303d9182` and recovers recognizable, detailed movie content. Disabling only the coordinate
lowering returns byte-for-byte to `c87261d44cd4426b`; the focused live-renderer mutation likewise
keeps its named pixel check executing and makes only that check fail. This is real detail recovery,
but the replay is not a live visual oracle. Its strong green/purple cast is an apparatus artifact:
the capture stores both plane payloads but records `linear_row_pitch_bytes=0`, and replay has no
process-local AvPlayer pitch registry. The renderer therefore cannot recognize the two-channel
chroma contract and deliberately takes its legacy narrow-texture coverage path, broadcasting U as
both U and V. An independent BT.709 model with `(U,U)` matches the replay at 42.17 dB / 0.9816 SSIM;
the correct `(U,V)` model does not. Live chroma correctness remains unverified, not known broken.

The replay also preserves the captured movie quad's 1920x608 placement with top/bottom bars. No
hardware or live-product oracle establishes whether that placement is intended, so it is recorded
without calling it a defect. The coordinate fix does not bypass the default-launch startup race
(#1746), and retained replay evidence does not change the compatibility rung.

<p align="center">
  <img src="../../assets/screenshots/rtype-delta-movie-coordinate-progress.png" alt="Intermediate R-Type Delta retained-frame replay after texel-coordinate recovery; recognizable detail is visible, while replay's missing row-pitch provenance causes a green-purple cast"><br>
  <sub>Intermediate retained-frame replay: detail recovery is real; the green/purple cast is a replay-provenance artifact, and live chroma/placement correctness remains unverified.</sub>
</p>

## The shell state machine, decoded to guest symbol names

Earlier notes described the pre-graphics shell dispatcher (`eboot+0x1080`, called once per iteration
from `eboot+0x17c0`) with placeholder stage names. The dispatcher is fully decoded now, and the names
are the title's own:

`eboot+0x1950(sceneId, phase)` is `sceKernelDlsym(moduleHandle[sceneId], phaseName[phase - 1])`.
The handle array is `eboot+0x4e4530`; the name table is `eboot+0x3e9860` and holds exactly six
entries — `DLLInit`, `DLLMain`, `DLLExit`, `DLLLoadStart`, `DLLLoadCheck`, `DLLLoadEnd`. The
`phase - 1` index comes from `shl rsi,0x20 / add / sar rax,0x1e` at `eboot+0x1980..0x199c`. The
state word is `eboot+0x4e0658`, initialized to 1 at `eboot+0x17a8`; the jump table is
`eboot+0x3e97b0`.

| State | Handler | Calls | Transition |
| --- | --- | --- | --- |
| 1 | `eboot+0x10f0` | `DLLLoadStart` | → 2 |
| 2 | `eboot+0x12dc` | `DLLLoadCheck` | stays while it returns 0; → 3 on nonzero |
| 3 | `eboot+0x11a8` | `DLLInit` | → 4 (written *before* the call) |
| 4 | `eboot+0x11d9` | `DLLMain` | stays while it returns nonzero |
| 5 | `eboot+0x118d` | — | counts `eboot+0x4e065c` (set to 8 at `eboot+0x1371`) down to 0, → 1 |

`eboot+0x1080` calls the common app update `eboot+0xa970` at `eboot+0x1586` on **every** iteration,
whatever the state. That update is where the fault happens, so the mapping is directly checkable
against a boot: `PROSPER_SYNCLOG=1` logs every resolved `sceKernelDlsym`, and a default boot shows
exactly `1 × DLLLoadStart`, `80–86 × DLLLoadCheck`, `1 × DLLInit`, and **zero** `DLLMain`,
`DLLExit` or `DLLLoadEnd` — which independently confirms both the state mapping and the existing
record that state 4 is never reached. Because the dispatcher re-resolves the symbol on every
iteration, that dlsym count is a free, non-invasive iteration counter for the pre-graphics loop.

The faulting branch is gated by `app+0x20c0` (`eboot+0x4e66b0`, read at `eboot+0xc275`) becoming
non-NULL. Its only writer is the one-instruction virtual setter `eboot+0xa100`
(`mov [rdi+0x20c0], rdx ; ret`, vtable slot `eboot+0x4108a8`). With that setter armed,
`PROSPER_HWBP=0xa100 PROSPER_HWBP_ALLTHREADS=1` records exactly one hit, with
`ret=runtime-prx+0x99c` — the title's own `title_Release.prx` queues the request from inside
`DLLInit`, on the main thread. The same dispatcher iteration then reaches `eboot+0xc2b3` →
`eboot+0x24030` → `users.front()` and faults at `eboot+0x24055`.

## The load-completion gate, decoded

`DLLLoadCheck` (`title_Release.prx+0x17b0`) is three instructions of substance:

```text
17b4:  rdi = [prx+0x3e318]         ; the shell object DLLLoadStart was handed
17bb:  esi = 0x7d0                 ; 2000
17c3:  call [rax+0x4f8]            ; shell->poll(2000)
17c9:  test eax,eax ; je 0x1836    ; zero -> "still loading", return 0
```

The shell object's vtable is **`eboot+0x4124e8`**, so slot `+0x4f8` is `eboot+0x3c760`. It lazily
constructs a process-global at `eboot+0x6f5df8` and calls `eboot+0x507a0(set, key)`, which is an
ordinary balanced-tree `lower_bound`: it walks nodes comparing `[node+0x20]` against the key, takes
the value at `[node+0x28]`, and returns `1` only when that job's outstanding counter `[value+0x3c]`
is zero (`-1` when the tree is empty, `-2` when the key is absent, `0` while work remains). So
`0x7d0` is a **job-group key, not a time budget**, and the load finishes when group 2000's counter
drains. Two `TRDid` worker threads do the reading (45 `pread`s of `title_tga.bin`, 19 of the fonts).

Recovering that vtable statically is ambiguous — this eboot is built without RTTI, so vtables have no
typeinfo boundary and a contiguous run of relative relocations spans several of them. The reliable
route is one HWBP on the **PRX side** of the virtual call, because `rax` there *is* the vtable
pointer, and `PROSPER_HWBP` prints `rax`. `PROSPER_HWBP` adds the eboot base, so a PRX address is
reached by passing `prx_addr - eboot_base`:

```bash
# title_Release.prx loads at 0x800000000; eboot base is 0x410000000
PROSPER_HWBP=0x3F00017C3 PROSPER_HWBP_MAX=3 PROSPER_HWBP_ALLTHREADS=1 ./build/boot_trace <DUMP_ROOT>/PPSA26414-app0
# [hwbp] #1 rip=runtime-prx+0x17c3 rax=0x4104124e8 ...   -> vtable = eboot+0x4124e8
```

## The race is decided by host single-thread CPU speed (proven, with an executed positive arm)

Measured on `bffa5e25` with a low-volume timeline (`PROSPER_SYNCLOG=1` plus **both**
`PROSPER_SYNCLOG_SEMA_ONLY=1` and `PROSPER_SYNCLOG_COND_ONLY=1`, which suppresses the sema/cond flood
while keeping `[thread] create` and `[dlsym]` — 153 lines total, so it does not itself move the race):

| Event | live renderer | no GPU backend | 1 logical CPU + 3 spinners |
| --- | --- | --- | --- |
| `[thread] create … name=INPUT` | 0.1154 s | 0.0644 s | 0.2622 s |
| its 400 ms sleep expires | 0.5154 s | 0.4644 s | 0.6622 s |
| `DLLLoadStart` | 0.1304 s | 0.0788 s | 0.3381 s |
| `DLLInit` (load complete) | 0.3709 s | 0.3324 s | **1.4711 s** |
| load duration | 244 ms | 254 ms | **1133 ms** |
| outcome | SIGSEGV `+0x24055` | SIGSEGV `+0x24055` | **no fault, 3,323 `DLLMain`** |

The deficit on an unthrottled host is **95–145 ms** across routes. Throttling *single-thread* CPU to
roughly a quarter of one core stretches the load 4.5× and the title **wins its own race and boots** —
the first time this title has passed `eboot+0x24055` without patching guest bytes or capping a guest
sleep. That is the positive arm the earlier analyses lacked: the race is decided by how fast the host
executes the title's own loader, and a PS5's Zen 2 core is slower than this host's, which is why the
title ships working.

Two apparatus corrections belong with that table, because both produced wrong answers first:

- **CPU affinity is not a CPU throttle.** `taskset -c 0` and `taskset -c 0,1` change the load
  duration by under 10 % (254 → 281 ms), because a single runnable thread still gets a whole logical
  CPU no matter how few are in the mask. Any "not CPU-bound" conclusion drawn from an affinity mask
  alone is **void**; the throttle has to be *contention* (competing spinners on the same CPU) or a
  cgroup quota.
- **A high-volume diagnostic changes the outcome.** `PROSPER_PREADLOG=1` piped through a
  line-flushing consumer added ~180 ms and the title *survived*; `PROSPER_FRAME_DUMPS=1` (24 MB BMP
  per frame) moved the fault from 0.386 s to 0.780 s. Any startup-timing number for this title taken
  with a verbose log is worthless — measure with the low-volume combination above.

## Ruled out

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| pc-52 `IMAGE_SAMPLE` is an unsupported shader opcode | **Falsified.** The opcode/dimension/dmask are already implemented. The exact shader metadata resolves pc 52 to direct texture slot 1 at SGPR 8 and sampler slot 1 at SGPR 20; the missing texture is upstream. | #1753, this doc |
| SGPR 8 is a bindless or dynamically indexed resource that needs a new resolver | **Falsified.** `movie_yuv_p.ags` declares direct texture offsets 0 and 8 and direct sampler offsets 16 and 20. The title explicitly binds texture slots 0 and 1. | this doc |
| The chroma bind packet is lost by Prosper | **Falsified.** The title returns before constructing the chroma descriptor, then later binds that uninitialized slot itself. There is no valid descriptor for a packet decoder to recover. | guest `eboot+0x19d10`, this doc |
| AvPlayer output must remain tight because padded output caused GRIS's right-edge strip | **Falsified.** GRIS subtracts the ABI crop offsets from pitch. The prior padded experiment left crop right zero; the missing crop, not the physical padding, exposed the strip. | #1393, GRIS `eboot+0xf6d5ab`, this doc |
| Reporting pitch 2048 while retaining tight rows is sufficient | **Falsified by construction.** The title advances plane addresses by the published physical extent. The CPU fixture verifies row-by-row storage and both plane bases, not metadata alone. | `test_avplayer` |
| The pc-52 resource rejection was the only movie-visual blocker | **Falsified live.** The size mismatch and all shader/resource rejects are gone, but retained frames contain only flat clears and low-colour gradients. The downstream frontier is #1807. | bounded `4114391c` run, #1807 |
| The decoded/staged movie planes contain the low-colour gradient | **Falsified.** The exact captured R8/RG8 bytes produce a sharp full-colour frame through an independent CPU NV12 conversion; only Prosper's replay of the same bytes collapses to the gradient. | present 1740 capture, #1807 |
| Vulkan's native unnormalized-coordinate sampler can apply `FORCE_UNNORMALIZED` directly | **Falsified against the live S#.** R-Type combines the bit with wrap addressing (`CLAMP_X/Y=0`) and `maxLod=15.9961`; Vulkan requires clamp addressing and zero LOD for an unnormalized sampler. The native candidate correctly became fail-visible but could not represent the guest contract. | retained submit 1741, #1807 |
| The diagonal proves the fullscreen strip's topology or interpolation is wrong | **Falsified by the coordinate-only A/B.** Recompiling the same raw VS/FS and changing only texel-to-normalized lowering recovers detailed content and removes the diagonal; disabling only that lowering reproduces the exact old hash. | hashes `51d9511c303d9182` / `c87261d44cd4426b`, #1807 |
| The corrected replay's green/purple cast proves live chroma upload or shader conversion is wrong | **Falsified as a replay claim.** Captured U/V bytes are distinct in 494,043/552,960 pairs, T# swizzle `(X,Y,0,1)` and the shader's BT.709 component flow are correct, and a forced `(U,U)` CPU model matches replay at 42.17 dB / 0.9816 SSIM. Capture reports row pitch zero, so replay misses the AvPlayer contract and intentionally broadcasts the first narrow channel. Live output has not passed the startup race and remains unverified. | submit 1741 capture metadata, #1807 |
| The first state-3 renderer completion poll or `sceVideoOutLatencyControlWaitBeforeInput` should delay startup until the LOGIN pump runs | **Falsified.** Both double-buffered completion qwords are initialized to 2, and an armed breakpoint recorded zero hits at the poll while the sole downstream VideoOut call proved execution crossed it before the ordinary fault. Available implementations allow the latency-control hint to return OK immediately, so adding a guessed delay has no oracle. The remaining #1746 frontier is an unidentified generic scheduling contract, if one exists. | `eboot+0x1acb0/+0x1acca`, CPU-only `cb7602b7` run, #1746 |
| The shell's fixed 512-update state-4 delay causes the default startup race | **Falsified by reachability.** State 3 calls `DLLMain(1)`, writes shell state 4, and calls the common app update in the same invocation; that update faults on the empty user vector before the dispatcher can enter state 4 and perform its first `DLLMain(2)` poll. The 512-update counter exists only downstream, after the input worker has already won. Post-#1837 HWBP correctly reports zero hits at the unreachable state-4 test; the merged guest-entry boundary is not missing this execution. | guest `eboot+0x11a8..0x11ef`, fault return `eboot+0x158b`, #1746 |
| A prosper Pad/UserService HLE side effect makes the pending pad-vibration request non-NULL earlier than hardware would, so the faulting branch is prosper-induced | **Falsified.** `app+0x20c0`'s only writer is the setter `eboot+0xa100`. Armed on every thread it records exactly one hit, from `ret=runtime-prx+0x99c`: the title's own runtime PRX queues the request from inside `DLLInit`. No prosper call sets it, and both consumer branches (`eboot+0x23ee0` and `eboot+0x24030`) dereference the user vector, so suppressing either branch would not help. | `PROSPER_HWBP=0xa100`, default route on `9dcb6c4b`, #1746 |
| Pacing the guest's pre-graphics loop at the display rate, as hardware does, delays `DLLInit` past the INPUT worker's 400 ms sleep | **Falsified by a lever-verified A/B.** A local diagnostic that pads only the main thread's per-iteration `sceKernelDlsym` by 16 ms moved the `DLLLoadCheck` count from 83–86 to **12** — the lever demonstrably worked — and the identical fault still occurred at `eboot+0x24055`. The live-render route independently gives 31 iterations and the same fault. The title's loader completes on wall clock in its own worker, so a slower dispatcher only reduces the poll count; it does not postpone `DLLInit`. Pacing made the load *finish sooner* (12 × 16 ms ≈ 0.19 s), because the free-running loop had been contending with the loader worker for CPU. | CPU-only and render routes on `9dcb6c4b`, #1746 |
| Faithful (slower) guest-visible I/O timing is the lever that lets the title win its own race — a PS5 loading the same assets from its own storage takes longer than 400 ms | **Falsified quantitatively; it points the wrong way.** A default boot reads **101.4 MB** before the fault: `data/sound/sound.bnk` 26.05 MB in 398 reads (all of them *before* the first `DLLLoadCheck`), `data/tex/title_tga.bin` 56.68 MB in 45 preads inside the poll window, and ~18 MB of `data/font/*.tga`. Prosper's effective pre-fault rate is roughly 0.3 GB/s; PS5 storage is an order of magnitude faster, so faithful storage timing would *shorten* the window the INPUT worker needs. No storage model makes 101 MB take 400 ms. | `PROSPER_FILELOG=1` byte census on `9dcb6c4b`, #1746 |
| The title's loader is frame-budgeted (a fixed streaming quota per `DLLLoadCheck`), so the iteration count is the governor | **Falsified.** The count is not invariant: 80, 83, 86, 86 across identical commands, 103 with `PROSPER_FILELOG=1`, 31 on the render route and 12 under a 16 ms dispatcher pad — it tracks loop speed, not work quanta. All 398 `sound.bnk` reads also complete before the first poll, so the bulk of the load is not streamed across polls at all. | dlsym census, five routes on `9dcb6c4b`, #1746 |
| prosper's `sceKernelUsleep` has a units or timebase error that manufactures the 400 ms window | **Falsified.** `eboot+0x19e0` is `imul edi,edi,0x3e8 ; jmp sceKernelUsleep`, so the guest's `Sleep(0x190)` really requests 400,000 µs, and `k_usleep` (`hle_kernel_time.cpp`) nanosleeps exactly that. The window is the title's own, at the requested length. | guest `eboot+0x19e0`/`+0x23210`, `hle_kernel_time.cpp:401`, #1746 |
| The INPUT worker is created late under prosper, so its 400 ms window starts too late | **Falsified.** `[thread] create … name=INPUT` appears at `PROSPER_SYNCLOG` line 66 of a default boot, roughly 2,150 lines before the `DLLLoadStart` dlsym that begins the load. The worker is created and asleep long before the load starts. | `PROSPER_SYNCLOG=1` default boot on `9dcb6c4b`, #1746 |
| A missing UserService event-delivery behavior can populate the title's user vector before the crash | **Falsified by the guest call graph and live call census.** The vector's only producer is the title's LOGIN handler, reached only when its input worker calls `sceUserServiceGetEvent` after a deliberate 400 ms sleep. Default launch makes zero `GetEvent` calls before `users.front()` faults. `Initialize` and `GetInitialUser` can report service state and a user id, but cannot invoke that title-owned handler; doing so would require a title-memory mutation or scheduling shim rather than an API implementation. Prosper's process-global delivery flag and no-op `Initialize`/`Terminate` leave a separate generic lifecycle question, but the repository has no PS5 hardware oracle for its correct transitions. That unproven lifecycle contract cannot cause or repair this fresh-launch ordering defect. | guest `eboot+0x23690`, `+0x23860`, `+0x49070`; service trace; #1746 |
| The load's completion is gated on GPU-side work that prosper retires early or never performs (the standing "next hypothesis to test") | **Falsified by a backend A/B.** The load takes 244 ms with the live Vulkan renderer and 254 ms with `PROSPER_NO_COMPUTE=1` and no renderer at all — the progression-only no-op backend is the positive control that the GPU really was absent. A duration that is *unchanged* when every GPU submit disappears cannot be paced by GPU completion. | `bffa5e25`, low-volume timeline, #1746 |
| Guest-visible file I/O timing is a material part of the load, in either direction | **Falsified by syscall time, not by byte count.** `strace -f -c` over a complete pre-fault boot (same run reproduced the fault, and its 68 `pread64` calls match the standalone trace) charges **9.1 ms to 68 `pread64` and 8.1 ms to 1,599 `read`** — ~17 ms of file I/O inside a 244 ms load. The earlier byte census argued about direction; this measures the time and shows the term is negligible either way. | `strace -f -c` on `bffa5e25`, #1746 |
| The load is bound by wall-clock waits the title requests, so it would take the same time on hardware | **Falsified.** Per-thread `clock_nanosleep` census during the load: the reading worker sleeps only `usleep(0)` × 78, the main thread `1 µs` × 173 and `1 ms` × 45; the 5 ms × 55 and 20 ms × 14 loops belong to the audio threads and run regardless. Guest-requested sleep on the threads that gate the load totals ~56 ms of a 244 ms window, and CPU contention stretches that window 4.5×, which a wall-clock wait could not do. | `strace -f -tt -T -e trace=pread64,clock_nanosleep`, spinner arm, #1746 |
| Prosper answers some call in the 13–24 ms between the INPUT worker's creation and `DLLLoadStart` (`sceKernelLoadStartModule`, the NP/Scream/audio initialisations) ~140 ms too fast, and that is where the fidelity gap lives | **Not supported, and no longer needed.** It was the leading candidate once I/O and GPU were excluded, but the CPU-contention arm accounts for the entire deficit inside the *load* itself, and it does so with an executed positive outcome. Recorded so the next reader does not chase a module-load latency model for which this repository has no oracle. | this doc, #1746 |
| The chroma cast in the corrected movie replay is only a capture-provenance artifact, so live output may be correct | **Falsified live.** With the race won by host CPU contention, the unmodified default route renders the publisher logos and the whole opening movie with the *same* green/magenta cast as the replay. Live chroma is now verified broken, not merely unverified. Tracked in #2005. | throttled `screenshot` route on `bffa5e25`, #2005 |

## Next frontier

**The #1746 investigation is finished as an investigation.** Its cause is measured and its causal
chain has an executed positive arm (see the CPU-contention section above), and the whole space of
mechanisms prosper could plausibly own — I/O timing in either direction, GPU-paced uploads,
display-rate pacing of the guest loop, a fixed shell poll budget, a UserService lifecycle or
event-delivery contract, guest-requested wall-clock waits, and a module-load latency model — is now
falsified in `## Ruled out`, each with the evidence that killed it. What remains is **not** an
engineering question with a hidden answer; it is a product decision, and it is the project owner's:

> R-Type Delta races its own 400 ms user-event delay against its own asset load. It wins on a PS5
> because a PS5's core is slower than a modern desktop's. prosper is *correct* here — every guest
> sleep is honoured to the nanosecond and every service answer is faithful — and the title still
> dies, because being correct and fast is enough to lose this race. Any fix is therefore a
> deliberate, titled compatibility decision (the class of thing Proton's per-game fixes and DXVK's
> app profiles are), not a defect repair. Do not implement one without that decision.

Do not re-open this as a search for a generic contract. If a future reader believes one exists, the
bar is a mechanism that survives the backend A/B, the syscall-time census, and the CPU-contention arm
in the two sections above — not a new guess.

## Rung 1 reached (2026-08-05, `bffa5e25`)

With the race won by host CPU contention — **unmodified binary, unmodified guest, no patched bytes,
no capped guest sleep, standard `tools/screenshot` frontend, default route** — the title boots and
runs its frame loop for minutes. A 75 s route sampled every 3 s returned **25 of 25 source-distinct
and 25 of 25 pixel-distinct** 1920×1080 frames, up to 38,964 distinct colours each, reaching guest
frame 2,162: the Clear River Games publisher logo, then the full opening movie (the R-9 fighter in
its hangar) as sharp, recognizable, letterboxed 1920×608 content. That clears rung 1 — real frames
from the live renderer, not black and not diagnostic-only. The route is reproducible with 12 busy
loops pinned to the same four logical CPUs as prosper:

```bash
for i in $(seq 1 12); do taskset -c 0-3 bash -c 'while :; do :; done' & done
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  taskset -c 0-3 ./build/screenshot <DUMP_ROOT>/PPSA26414-app0 --seconds 3 --count 25 --out <dir>
```

<p align="center">
  <img src="../../assets/screenshots/rtype-delta-rung1-logo-and-opening-movie.png" alt="R-Type Delta HD Boosted rung 1: the Clear River Games publisher logo and the opening movie's R-9 hangar shot, both rendered live"><br>
  <sub>Rung 1, <code>tools/screenshot</code>, default route on <code>bffa5e25</code>, unmodified binary and guest, host CPU contended so the title wins its own #1746 startup race. Left: Clear River Games logo (sample 03, guest frame ~250). Right: opening movie (sample 17, guest frame 1660). Both carry the live chroma cast — greys render green, the image is magenta-tinted.</sub>
</p>

Two things are visible in those frames and both are real, live defects rather than apparatus:

- **The movie's chroma is wrong** — greys render green and the whole image carries a magenta cast,
  the same signature the #1807 replay showed and attributed to missing capture provenance. It is now
  reproduced on the live path, so that attribution was incomplete. Tracked in #2005.
- Software H.264 fallback: `No support for codec h264 profile 77` / `Failed setup for format vaapi`.
  The movie decodes and plays anyway; worth checking whether VAAPI should have taken profile 77.

**Rung 2 (title screen) is not reached, and the reason is now bounded.** A 272 s continuation of the
same route (34 samples, 8 s apart) watched the movie end and the title enter its next phase. Samples
00–07 (to 64 s) carry movie content — 6,559 to 38,682 distinct colours each — and then **every one of
the remaining 26 samples, 72 s through 272 s, guest frames 2,451 → 5,088, has exactly one distinct
colour**: the dark blue-grey `(11,13,28)` the earlier `PROSPER_USLEEP_MAX_US` diagnostic also saw,
fading toward black. The frame loop keeps running the whole time (frames advance ~15/s) and the
presented extent stays 1920×1080, so this is a content defect in the post-movie screens, not a hang
and not a present failure. That is the rung-1 → rung-2 blocker and it is now separable from #1746:
the race no longer hides it. Tracked in #2006.

Two apparatus notes for whoever picks that up:

- **`tools/screenshot` prints `status=ok` after the guest thread has died** (#2007). On the
  unthrottled default route it wrote 25 PNGs and reported `status=ok` while the log also carried
  `[shot] guest thread ended: kind=2 detail=SIGSEGV … +0x24055`. The tells are
  `source-distinct=1 pixel-distinct=1` and `max-*-stale` equal to the whole route. Read the
  `guest thread ended` line before believing any `status=ok`, or pass `--min-pixel-distinct-frames`.
- The flat post-movie colour is *not* the same value as the pre-fault black (`crc=064567f8`); the two
  are distinguishable by CRC, which makes them separable in a route manifest.
