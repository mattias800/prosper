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
