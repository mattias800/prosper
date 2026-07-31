# colorstate_report — "do draws reach the scanout, and with what colour state?"

When a title presents black, four explanations compete:

1. **No graphics draws at all** — the render path never submits.
2. **Draws exist but only target offscreen surfaces**, never composited to scanout.
3. **Draws write scanout with colour suppressed** — `CB_COLOR_CONTROL.MODE=DISABLE`,
   or a zero `CB_TARGET_MASK` / `CB_SHADER_MASK`.
4. **Content is produced and the present path drops it.**

`PROSPER_COLORSTATETRACE` emits the raw register evidence that separates 1–3, but on a
4K title it is millions of lines. This tool reduces it to a verdict.

## Recipe

Get the scanout VA from the guest's own memory map — the emulator logs
`Frame Buffer va range <lo> - <hi>` during boot. Everything outside it is offscreen.

```bash
# One routed run. WxH filters to scanout-sized colour targets and keeps the log bounded.
PROSPER_COLORSTATETRACE=3840x2160 <your route> > run.log 2>&1

python3 tools/colorstate/colorstate_report.py run.log --scanout-prefix 0x9fc
```

Output names the scanout surfaces actually written, breaks scanout draws down by
`mode` / effective mask / resolved write mask, and gives a per-guest-minute
suppressed-percentage series.

`--selftest` runs the parser and reporter against a fixture; it needs no capture.

## Read the per-minute series, not a single number

**A suppressed-draw count means nothing on its own.** Compare a phase whose output is
known good against the phase under investigation, using screenshots to label them.

The Plucky Squire is the cautionary case. When its screen goes black, scanout draws
flip to 83% `mode=0` — which looks exactly like explanation 3 and is the same shape
that explains Astro Bot's black world map. It is not the cause. While the world is
*visibly rendering*, the suppressed fraction is **higher still**:

```
16:08  total=  15072  suppressed=  7.3%   <- menu, renders correctly
16:09  total=  40336  suppressed=  3.3%   <- menu, renders correctly
16:10  total= 179621  suppressed= 83.2%   <- black
16:13  total=   6692  suppressed= 87.6%   <- world VISIBLE
16:14  total=   7799  suppressed= 95.3%   <- world VISIBLE
```

`CB_DISABLE` here is the engine's depth and shadow prepass, which scales with scene
geometry. Reading only the black phase would have produced a confident wrong answer.

## Caveats

- Records are attributed to the **most recent guest log timestamp**, so a minute in
  which the guest logs nothing absorbs neighbouring records. Treat bucket boundaries as
  approximate and lean on buckets you can independently label from screenshots.
- The trace runs *before* draw realization, so a record here is a draw the guest
  programmed — not proof it was realized. Confirm realization with
  `gpu_replay --inspect-only` on a capture.
- An **absent** mask means write-all; a **present zero** means write-nothing. The tool
  preserves that distinction, which is the reason the raw triple is retained at all.
- The scanout VA is title- and run-specific. The tool refuses to guess a default.
