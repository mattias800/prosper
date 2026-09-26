# Outer Wilds (`PPSA08102`) — status

Tracker: [#3804](https://github.com/mattias800/prosper/issues/3804). Engine: Unity 2019.4.39f1 / IL2CPP.

**Rung 3 — a routed first-person scene renders on Linux.** On `main`
`ce84f7f7d305` (2026-09-23), the unmodified `tools/screenshot` frontend showed the Annapurna and
Mobius Digital opening logos, then a “PRESS X TO START” prompt by about 40 seconds. An animated
starfield and a dim campfire/tree scene render behind the prompt. The guest was still running when
the 90-second capture ended. The [T+60 screenshot](../../assets/screenshots/outer-wilds-title-start.webp)
is a 1920-wide WebP derived with `prosper/tools/screenshots/shrink.py` (SHA-256
`dc18471330cf77e204fcd8dc98877deb76ec58dd8cbec62c0aa79aa5bffd828b`). The unmodified
3840×2160 frontend PNG is retained privately under `<EVIDENCE_ROOT>/outer-wilds-20260923/`
(SHA-256 `40e847af6c3855341de10e3a0fcf479d1b422d037f1cad124aa9faf7433024be`).

The historical fresh-save route in [`prosper/scripts/outer-wilds-PPSA08102/`](../scripts/outer-wilds-PPSA08102/)
presses Cross at 50 and 70 seconds, then Square at 110 seconds. The first two presses reach and
accept NEW EXPEDITION; the Square press cleared the Wake Up prompt on `b77cffc831de`. The native
T+122 frame shows a wooded first-person scene and T+180 shows its Look Around / Move prompts. The
owner also observed the in-game world and described its rendering as almost correct, while reporting
roughly 2 FPS. This is a recognizable scene and supports rung 3; the short visual report is not a
whole-game compatibility verdict or a measured speed comparison with PS5 hardware.

On main after #3863, Square at 110 seconds arrived before the Wake Up prompt. A new
[`reach-first-person.pad`](../scripts/outer-wilds-PPSA08102/reach-first-person.pad) retries
Square at 135–137 seconds. On the #3864 source-equivalent binary with opt-in
`PROSPER_BACKEND_TEXTURE_PRESSURE_FLUSH=1`, its 200-second screenshot
reached the first-person wooded scene; a quiet in-world F8 window at 230 seconds recorded
47 known-new deliveries in roughly five seconds. These observations establish route progression
and a short-window completed-version count, not whole-game FPS.

The [first-person screenshot](../../assets/screenshots/outer-wilds-first-person-night.webp) is a
1920-wide WebP downscaled by `prosper/tools/screenshots/shrink.py` from an unmodified 3840×2160
`prosper-app` F9 readback. The source BMP SHA-256 is
`a9892511aa315ebb592152c1ffe2a8b526599fbac46908297f8731341873620f`; the committed WebP
SHA-256 is `8647cf104fb3b12548de5d4dab8db440f056c697406c19341abef523bd2fee8f`.
The readback was armed at guest present 1789 and written at present 1790. It shows the trees,
launch tower, planet and Look Around / Move prompts without the later white artifact.

**Known visual defects.** The game wordmark is absent on prosper's title frame and present in the
PS5 title-screen oracle attached to [#3804](https://github.com/mattias800/prosper/issues/3804).
A later interactive F9 screenshot shows a large nearly white blob over the right side of the wooded
scene. The first such bundle writer was interrupted; a fresh live F9 at guest present 1891/1892
completed both screenshot and replay bundle. The bundle's selected submit is 1889, so the screenshot
and replay are nearby but not identical frames ([#3824](https://github.com/mattias800/prosper/issues/3824)).
The replay retains the blob, and its pre-frame front-buffer seed already contains it. This proves
persistence across the selected submit, not its first producer. No matching PS5 gameplay oracle has
been captured.

## Reproduction and evidence

The baseline used the exact `ce84f7f7d305` screenshot binary (SHA-256
`892ed8abeb03fdbc3d0eb8d6b3bd40a382876828b764684ec239ca1f1c4a839d`), with no gamepad
route, guest flags, or diagnostic substitutions:

```bash
PROSPER_GAME_ROOT=<DUMP_ROOT> \
  <BUILD>/screenshot <DUMP_ROOT>/PPSA08102-app0 \
    --seconds 5 --count 18 --timeout 95 --out <EVIDENCE_ROOT>/PPSA08102/first-live/default-ce84
```

The tool completed 18/18 captures through T+90.1 seconds, exited 0, and reported
`guest=running`, `status=ok`. The T+5 image was black `raw_scanout`; T+10 through T+90 were 17
composited, pixel-distinct samples. The manifest SHA-256 is
`25c7fb62c266c556c429f8f352cf3c980ac92ec0410070634408d792b531ad1d`; the run log SHA-256 is
`43d2ede23fc714c62c4de3da00dd677eb5c82bca2983b2a81010ce61925f71ad`. Raw artifacts remain
private under `<EVIDENCE_ROOT>/PPSA08102/first-live/default-ce84/`.

`prerender_check.py` compared the T+60 PNG with four same-resolution ordinary image assets in the
dump and found no match (highest informative overlap 0.01%). That check cannot inspect images
inside asset containers or certify correct 3D rendering. The visible prompt and continuing
animation are the progression evidence; frame counts alone would not establish the scene.

For the fresh-save Wake Up run, the route's README records both input-read timestamps and artifact
hashes. The native capture completed 90/90 samples through T+180, with 87 composited and 74
pixel-distinct samples, and ended `guest=running`, `status=ok`. Those counts establish continuing
publication; the in-world image and controller prompts establish the gameplay milestone. The owner
manually inspected a later live run. Performance work uses the same route with isolated saves and
counts completed producer versions rather than treating repeated presentations as new frames.

## Next checks

Capture a complete F9 bundle while the white blob is visible, then isolate its first wrong producer
in replay. Compare a matched gameplay frame with PS5 hardware for lighting and visual fidelity.
Measure speed on a named stable scene in an uncontended GPU window, with completed producer cadence
separate from repeated presentation and from the PS5 comparison.

## Ruled out

One line per falsified hypothesis, the evidence that killed it, and the link. Read this before
forming a performance hypothesis on this title — every row here cost a session to establish.

- **An AVX2 kernel for 16-bit index expansion is a speedup worth defaulting on** — falsified
  2026-09-26 (#3866). Three independent reasons, any one of which is sufficient:
  1. **The portable loop was never scalar.** GCC 16.1.1 at the project's `-O2` compiles
     `copy_indices_u16_max` (then named `…_scalar`) into an SSE2 loop handling **8 indices per iteration**
     (`movdqu` → `punpcklwd`/`punpckhwd` → two `movups`) — the same granularity as the
     hand-written kernel. There was no one-index-per-iteration baseline to beat. An independent
     reviewer reproduced the same signature on GCC **16.2.1** outside the container, so the
     codegen claim is not a property of one toolchain version.
  2. **Given the same ISA the compiler beats the intrinsics.** With `-march=x86-64-v3`, or with
     nothing but `__attribute__((target("avx2")))` on the *identical* portable loop, GCC emits
     exactly the instructions the hand kernel uses (`vpmovzxwd`, `vpmaxud`, 32-byte stores) and
     goes **wider — 16 indices per iteration against the kernel's 8**. Measured interleaved in
     one binary and one process, minimum of 60 batched samples per arm, the hand kernel is
     **1.21×–1.78× slower than its own source loop given the same ISA** at every size ≥ 384,
     and **exactly equal to it at n = 64**, where eight iterations leave both dominated by
     prologue and epilogue. So even where the idea is right, the implementation is dominated by
     a compiler flag.
  3. **The scale cannot reach the deficit.** The SSE2→AVX2 gap is real and large — **3.8×–6.1×**,
     because SSE2 has no `pmaxud` and GCC emits a six-instruction sign-flip blend per four lanes
     — but a large multiple of a small number is a small number. Across three runs and five
     sizes the shipped path costs **0.141–0.282 ns/index** and the kernel **0.030–0.070**, so
     each index saves **0.103–0.213 ns**. Recovering **1 ms of a frame therefore needs
     4.7–9.7 million 16-bit indices expanded in that frame**, and covering this route's
     **39 ms/frame** deficit would need **183–378 million** — order 60–126 million triangles per
     frame. Put as a rate instead of per frame, which is the form that does not need a frame
     count: at the shipped cost, **every 1 M indices/s is 0.014–0.028% of wall clock**, and the
     kernel can return at most three quarters of that. The prior end-to-end A/B could not
     resolve it either: 10.932 vs 11.144 completed versions/s over six recurring-shape records
     per arm.

  **How the numbers were taken**, because the box is shared with two other lanes and a
  contended measurement here is worthless: arms **interleaved** (A,B,A,B) within one process,
  never all-of-A then all-of-B; one binary and one set of buffers for every arm; **load average
  recorded per run** — 39.30, 31.60 and 17.48. Absolute times moved 5–15% across those loads
  while the **ratios moved under 3%**, except at n = 24576 (1.36× vs 1.78×), which is the size
  whose working set straddles L2. Every per-index figure was derived **twice by independent
  routes**, a three-arm scratch benchmark and the committed `--bench` arm. The first draft of
  this row quoted "about 4×" and "1.35×–1.5×"; recomputing from the raw samples gave
  3.8×–6.1× and 1.21×–1.78×, so both were wrong in both directions — the second derivation
  is what caught it.

  **The kernel survives; the switch does not.** `copy_indices_u16_max_avx2` is still in
  `src/gpu/execute/index_expand.hpp` with **no production caller at all**, purely so that
  `test_index_expand --bench` can re-run the interleaved A/B above — the measurement reason 3
  rests on — in seconds rather than at a session's cost. There is no environment variable and no
  dispatcher branch: an earlier revision of this branch had `PROSPER_INDEX_EXPAND_SIMD=1`,
  default off, and it was removed on review (#3866). The `PROSPER_UD_TAIL_ALIGN` precedent does
  **not** transfer — that gates behaviour observable only end-to-end in a live title, so a
  runtime switch is the only way to A/B it, whereas index expansion is a pure function of
  `(dst, src, count)` that a unit test A/Bs directly. A switch here could only ever have selected
  the option reasons 1 and 2 rule out. Do not reintroduce one to "measure it properly"; run the
  bench arm.
  **What would reopen this:** `PROSPER_INDEX_EXPAND_STATS=1` prints the emulator's actual
  16-bit and 32-bit index volume per second (`src/gpu/execute/index_expand.hpp`), and
  `test_index_expand --bench` prints the per-index cost of each kernel with interleaved arms.
  Multiply the two. If a route really does expand millions of indices per second, there are two
  levers and neither is a hand-written kernel: raise the **build's ISA baseline** (a
  project-wide `-march` decision that drops pre-2013 hosts, so it needs measuring against every
  title, not just this one), or keep the runtime dispatch exactly as it is and replace the
  intrinsics body with the **portable loop under `__attribute__((target("avx2")))`**, which is
  what measured fastest here. The dispatch cost is identical either way; only the kernel body
  differs. `CONFIDENCE: HIGH` for reasons 1 and 2, which are properties of the emitted code and
  reproduce from a single compile. `CONFIDENCE: MED` for reason 3's threshold: the per-index
  costs are measured, but they are **cache-warm**, which overstates the kernel's advantage
  against guest memory the renderer has not just touched, and the index volume itself **has not
  been measured on this route** — the instrument to measure it is the one named above. Both of
  those biases point the same way, toward the kernel mattering less rather than more.

- **The `8.33 ms` backend buffer-copy figure on this route is superseded before it was ever
  used, and is not a current number.** It was taken with `CpuRttSnapshotPool` matching a
  free-list entry on **exact size**; changing that to match on **capacity** took allocate-and-copy
  from 56,543 MiB to 1,096 MiB per ~120 s on one title, and the pool hit rate from 38.5% to
  99.5%. **That change was on another lane's branch and not on `main`** as of 2026-09-26, so it
  is not yet a property of this title's baseline — check whether it has merged before quoting
  either number. Deliberately not re-measured here: the only available comparison would have
  been against an unmerged branch. Re-baseline before attributing anything on this route to
  buffer copy. `CONFIDENCE: MED` — the pool figures are another lane's measurement on another
  title, restated here so this route's own number is not read as current, and not independently
  reproduced.
