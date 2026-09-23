# Outer Wilds: menu-to-Wake Up route

`probe-new-expedition.pad` is validated to the **Wake Up prompt**, not gameplay.
Tracker [#3804](https://github.com/mattias800/prosper/issues/3804) records that one Cross at
50–50.5 seconds was observed by the guest and exposed a menu with NEW EXPEDITION selected by
about 56 seconds. The second Cross at 70–70.5 seconds is the smallest input that can test that
selection without guessing about a later dialog or cinematic.

On `b77cffc831de` (2026-09-23), a fresh-save, native-cadence `screenshot` run observed both
Cross press/release pairs at 50.036/50.543 and 70.011/70.511 seconds after the first pad poll.
The menu showed NEW EXPEDITION selected at T+60, `Loading... 26%` at T+72, and a centered
`□ Wake Up` prompt over black by T+100. The T+100 and T+180 PNGs are byte-identical; the
guest remained running. This establishes that NEW EXPEDITION was accepted and reached the
Wake Up prompt. It does not establish gameplay rendering or a world-rendering defect: the
prompt explicitly awaits another input.

From the checkout root inside the `ps5ys` distrobox, after building `screenshot` from the commit
being tested, run during an allocated GPU window with a fresh, private save and evidence directory.
Use a new run ID in place of `new-expedition-01` on every attempt:

```bash
PROSPER_GAME_ROOT=<DUMP_ROOT> \
PROSPER_SAVE0=<EVIDENCE_ROOT>/PPSA08102/new-expedition-01/save0 \
PROSPER_SAVEDATA_DIR=<EVIDENCE_ROOT>/PPSA08102/new-expedition-01/savedata-mem \
XDG_CACHE_HOME=<EVIDENCE_ROOT>/PPSA08102/new-expedition-01/xdg-cache \
MESA_SHADER_CACHE_DIR=<EVIDENCE_ROOT>/PPSA08102/new-expedition-01/mesa-cache \
PROSPER_PAD_SCRIPT=@prosper/scripts/outer-wilds-PPSA08102/probe-new-expedition.pad \
PROSPER_PAD_SCRIPT_LOG=1 \
  <BUILD>/screenshot <DUMP_ROOT>/PPSA08102-app0 \
    --seconds 2 --count 90 --timeout 195 \
    --out <EVIDENCE_ROOT>/PPSA08102/new-expedition-01/shots
```

The measured run saved 90/90 samples through T+180.2 and ended `guest=running`, `status=ok`,
with 87 distinct source frames and 40 pixel-distinct frames. The exact `screenshot` binary
SHA-256 was `c46d3cd5356863c15768621452af9ef800674c847272bec0c58c9d915f494195`.
The private log SHA-256 is `645e15f1bb2e99238894cb29baca44e2518dcdb1eaea473edb76bb0d79202ddd`;
the manifest SHA-256 is `5bfc6ce5a010e9e4f99cb902449fbc91656812f1952d22d1ad4c1a79d213750f`;
the T+180 PNG SHA-256 is `16e0f3b5e588f77413c4af0b77926c80d0e922723bf1839fa9099851ebadf811`.
Original artifacts remain private under `<EVIDENCE_ROOT>/PPSA08102/new-expedition-01/`.

`probe-wake-up.pad` adds a one-second Square press at 110 seconds, after the observed prompt.
On a second fresh-save run of the same `b77cffc831de` binary, the guest observed Cross
press/release at 50.015/50.502 and 70.058/70.561 seconds, then Square at 110.008/111.292.
The T+100 image still shows `□ Wake Up`. The prompt clears after Square; by T+122 the
native frame shows trees, a structure, and a lit object in the sky from a first-person view.
The T+180 frame retains the wooded scene and displays `Look Around` and `Move` controller
prompts. This establishes a live, recognizable in-world scene under `tools/screenshot`.
The image is very dark; no matching PS5 hardware capture or direct draw census was taken.

That second run completed 90/90 samples through T+180, with 87 composited samples, 74
pixel-distinct samples, `guest=running`, and `status=ok`. The private log SHA-256 is
`0c8b5d788f439ed1e6e037560ba5df42e891c55fee544b7a2573f8762d34d897`;
the manifest SHA-256 is `c454f9e0aca4b576e0b033e774d9548d6391957ff4ea4a5469ad7c7ced54d8cc`;
the native T+180 PNG SHA-256 is `a7db3da34faa1a35c4662758fa38c9295b5c5926801caf45335ca02cdaca4630`.
Original artifacts remain private under `<EVIDENCE_ROOT>/PPSA08102/wake-up-01/`.

The absent OUTER WILDS wordmark is confirmed against the PS5 title-screen oracle attached to
[#3804](https://github.com/mattias800/prosper/issues/3804); its cause remains unknown. The
in-world scene has a positive live visual report, but no matching PS5 gameplay oracle. Measure
steady-state speed only
after identifying a stable scene, in an uncontended GPU window at native resolution, and compare
that scene's guest flip rate with PS5 hardware; the `--seconds 2` screenshot interval is only a
sampling cadence and is not a speed measurement.
