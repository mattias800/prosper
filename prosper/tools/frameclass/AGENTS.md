# `tools/frameclass/`

Answers **"is there anything in this picture?"** for a directory of captured frames, and
distinguishes the three cases that a bring-up lane keeps confusing.

`tools/screenshot` already answers *"is the picture changing"* — `pixel-distinct`, `source-distinct`,
`max-pixel-stale`. That is a different question, and a title can score perfectly on it while
publishing nothing a person would call a frame. This tool is the other half, and it exists because
both obvious hand-rolled versions of it are wrong in ways that read as a result:

- **Downsampling before counting colours destroys thin UI.** A 4K frame resized to 160x90 loses
  white menu text on black entirely. Measured on *Stray* (`PPSA02101`) on 2026-08-27: frames
  reported as "flat black, 1 colour" contained START GAME / SETTINGS / CREDITS and a legible build
  stamp, at full brightness. The title was one edit away from being written up as not rendering.
- **"Fraction of non-black pixels" scores a flat WHITE clear as a perfect frame** — 100% non-black,
  and every "is it black?" test passes it.

So it reports three numbers that disagree with each other, sampled at full resolution: the brightest
channel anywhere, the share of pixels above a noise floor, and the distinct-colour count taken
*before* any resize. `UI-ON-BLACK` is the class worth looking for — bright pixels covering under 2%
means the guest is drawing its HUD or menu and the **world** is what is missing, which is the
difference between "this title renders nothing" and "this title is at rung 2".

It is a reporting tool, not a gate: it always exits 0. Compare runs by reading the table.

```bash
python3 prosper/tools/frameclass/frameclass.py ~/work/shots
```

**Validated against known answers rather than asserted** — the GTA V bank interior and the Unbound
cinematic classify CONTENT, Stray's main menu UI-ON-BLACK, and a black clear and a white clear both
FLAT. Re-run that check if you change a threshold; a classifier nobody has seen fire correctly on a
known case is worth nothing.
