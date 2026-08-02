# colorstate_report — "do draws reach the scanout, and with what colour state?"

When a title presents black, four explanations compete:

1. **No graphics draws at all** — the render path never submits.
2. **Draws exist but only target offscreen surfaces**, never composited to scanout.
3. **Draws write scanout with colour suppressed** — a zero `CB_TARGET_MASK` /
   `CB_SHADER_MASK`. (`CB_COLOR_CONTROL.MODE` does **not** suppress; see #1724.)
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
flip to 83% `mode=0`, which looks like a colour-suppression explanation. It is not the
cause. While the world is *visibly rendering*, the `mode=0` fraction is **higher still**:

```
16:08  total=  15072  suppressed=  7.3%   <- menu, renders correctly
16:09  total=  40336  suppressed=  3.3%   <- menu, renders correctly
16:10  total= 179621  suppressed= 83.2%   <- black
16:13  total=   6692  suppressed= 87.6%   <- world VISIBLE
16:14  total=   7799  suppressed= 95.3%   <- world VISIBLE
```

`CB_DISABLE` here is the engine's depth and shadow prepass, which scales with scene
geometry. Reading only the black phase would have produced a confident wrong answer.

**Since #1724, `mode` alone is not a suppression signal at all** — the renderer derives the
colour write mask from `CB_TARGET_MASK & CB_SHADER_MASK` and ignores `MODE`. The two
genuinely disagree in the field: of 268,899 traced Plucky Squire draws, 36,613 are `mode=0`,
and **8,326 of those carry a non-zero mask**.

The `suppressed=` column above is therefore mask-derived, not `mode`-derived; the `modeN=`
breakdown beside it is the raw mode census and is no longer a suppression count. The table
above predates that change, so its percentages are `mode=0` fractions — read them as such.

### What the world phase actually contains (#1724)

The table above is the phase that made `MODE=DISABLE` look load-bearing. Joining `mode`
against the guest's own `effective` mask (`CB_TARGET_MASK & CB_SHADER_MASK`) over the same
run, **all records**, shows the two signals almost always agree — so #1724's removal of the
`MODE`-keyed override changes very little of it:

```
minute   records   mode=0   mode=0 with effective!=0
16:07     ~8,000    1,652 aff   16:10   179,621   2,755 aff
16:08    ~15,000    2,293 aff   16:13     6,692      33 aff   <- world VISIBLE
16:09    ~40,000    2,817 aff   16:14     7,799      30 aff   <- world VISIBLE
                                16:15       564       0 aff
TOTAL   533,964   200,113 mode=0   9,580 affected (4.79%)
```

**The affected draws are front-loaded in the menu/loading minutes and stop before the world
phase**: 63 of them across 16:13-16:15, or 0.35% of that phase's `mode=0` draws. The engine's
depth and shadow prepass programs `CB_TARGET_MASK=0` **as well as** `MODE=DISABLE`.

Two cautions for anyone re-deriving this:

- **Judge it on all records, not scanout-only.** Restricting to the scanout prefix gives 854
  affected, which understates by 11.2x: 8,105 of the hidden draws are a single pixel shader
  (`ps=0x3015320000`) writing MRT0 of a 3840x2160 **offscreen** G-buffer that later composites.
- **`snap1/snap2/snap3/guest.log` in one run directory are not independent samples.** They are
  nested byte-prefixes of the same run (`cmp -n` confirms), so their differing `mode=0`
  fractions are the log growing, not four observations. An earlier revision of #1724 cited them
  as four independent logs showing a "flat" affected count; that reasoning was circular and has
  been withdrawn. The per-minute decomposition above is the load-bearing evidence.

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
