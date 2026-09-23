# Outer Wilds (`PPSA08102`) — status

Tracker: [#3804](https://github.com/mattias800/prosper/issues/3804). Engine: Unity 2019.4.39f1 / IL2CPP.

**Rung 2 — a live title-start prompt renders on a default Linux launch.** On `main`
`ce84f7f7d305` (2026-09-23), the unmodified `tools/screenshot` frontend showed the Annapurna and
Mobius Digital opening logos, then a “PRESS X TO START” prompt by about 40 seconds. An animated
starfield and a dim campfire/tree scene render behind the prompt. The guest was still running when
the 90-second capture ended. The [T+60 screenshot](../../assets/screenshots/outer-wilds-title-start.png)
is an exact copy of the frontend PNG (SHA-256
`40e847af6c3855341de10e3a0fcf479d1b422d037f1cad124aa9faf7433024be`).

**Visual fidelity is unverified.** The game wordmark is absent in the sampled prompt frames; this
is a candidate difference, not an established PS5 defect without a hardware oracle. No input was
sent, so main-menu interaction and gameplay are untested. The 50% speed goal has not been measured
on a stable scene against PS5 hardware.

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

## Next checks

Obtain a PS5 reference for this title revision to assess the wordmark and scene lighting. A
separate bounded Cross route can then test whether the start prompt advances to a menu. Measure
speed only on a named stable scene with an uncontended GPU window and a PS5 comparison.
