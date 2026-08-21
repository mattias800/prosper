# Hi-Fi RUSH (`PPSA17168`) routes

`reach-gameplay.pad` drives the title from a cold boot past the `Press Any Button` title screen
and into the first-boot language/settings wizard.

Anchors are wall-clock seconds from the **first pad poll**, not from process start, so the route
is robust to how long the IoStore asset load takes — which varies by more than a minute between a
cold and a warm page cache on the same machine.

Cross is the confirm button everywhere observed so far: the title screen takes any button, and
the language page's own on-screen prompt reads `(X) Next`.

## Known caveat: the `left` beat also edits settings

The `left` beat is unconditional, because a blind script cannot know whether a dialog is up. On a
page where no dialog is open, `left` lands on the focused settings row and steps its *value* — so
a long run walks the wizard's own options. Measured 2026-08-21: a run reached the accessibility
page with the UI in Arabic, because repeated `left` presses had cycled **Text Language** backwards
off English.

That is cosmetic — the title keeps progressing and the route still does its job — but it makes the
captures harder to read, and it means this route is **not** suitable as a snapshot guard as it
stands. A guard route wants per-screen windows derived from a known frame ordinal, not a blind
cycle.

```bash
SDL_VIDEODRIVER=offscreen PROSPER_VULKAN_LIB=libvulkan.so.1 \
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_PAD_SCRIPT=@scripts/hifi-rush/reach-gameplay.pad PROSPER_PAD_SCRIPT_LOG=1 \
timeout -s KILL 1250 ./screenshot <DUMP_ROOT>/PPSA17168-app0 \
    --seconds 13 --count 90 --timeout 1200 --out <OUT_DIR>
```

Two things about that command line are deliberate, not incidental:

- The **outer `timeout` is the one that can bound a boot.** `tools/screenshot --timeout` is
  checked inside the sampling loop, which is only reached after `boot_program()` returns, so it
  cannot fire on a title that stalls during boot (instrument trap 214). Keep the two limits at
  different values and quote which one fired.
- The **sampling interval is prime** (7 s / 11 s / 13 s have all been used here) so that the
  sample grid cannot divide a title-side cycle period and report a phase as a frozen state
  (instrument trap 211).

`PROSPER_PAD_SCRIPT_LOG=1` is not optional for evidence: without it the route cannot distinguish
"the guest ignored the input" from "the input was never applied", and a route parsed as inert
looks exactly like a title stuck on its first screen (#2439).
