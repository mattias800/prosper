# GFX10 `SW_4KB_S` texture tiling — the exact element-swizzle order

**Status: SOLVED (2026-07). Pixel-verified against The Messenger's (PPSA24651) title art.**
This documents the exact within-tile element order for the PS5/RDNA2 `SW_4KB_S` (standard 4 KB) swizzle
mode — the layout PS5 games store colour textures and render targets in. We could not find this specific
element order published in any open-source emulator/addrlib, so it is written up in full here.

## Why this matters (the symptom it fixes)

On a real PS5, the GPU's texture unit de-swizzles tiled textures in hardware when a shader samples them.
prosper runs on a desktop GPU (llvmpipe/Vulkan) with a *different* native layout, so we must de-swizzle
PS5-tiled bytes into linear ourselves before upload (`src/gpu/tile.cpp`). If that de-swizzle is even
slightly wrong, pixels land in *almost* the right place in a **regular repeating pattern**.

Our previous order was a plain "Y in the low bit" Morton (`[y0,x0,y1,x1,y2,x2,y3,x3,y4,x4]`). It got the
coarse structure right (images were recognizable) but the **lowest element bits were wrong**. A wrong
*low* bit only swaps pixels that are 1–2 apart — which is **invisible in flat/gradient regions** (the
swapped pixels have near-equal values) but **serrates every edge** (where neighbours differ). The result
was a cross-game, per-pixel "dither" on every tiled texture, and — on small low-res glyphs like caption
text and the on-screen button prompt — enough scrambling to render them *unreadable*. So the "missing
text" (#102) and the "dither" (#118) were the **same** bug: a wrong texture de-swizzle.

## The correct element order

For a `SW_4KB_S` tile at 32 bpp the tile is **32×32 texels** (4096 bytes / 4). The element index within
the tile (`0..1023`, 10 bits) is built from the texel's in-tile coords `(ix, iy)` (each 0..31) as:

```
element bit:  0    1    2    3    4    5    6    7    8    9
coord:        x0   x1   y0   y1   y2   x2   y3   x3   y4   x4
```

i.e. **the lowest four bits are a 4×4 pixel sub-block `[x0, x1, y0, y1]`** (both low X bits, then both
low Y bits), and **above that the usual interleaved `(y, x)` Morton pairs** (`y2 x2 y3 x3 y4 x4`). Golden
positions this implies:

| texel (x,y) | element |
|-------------|---------|
| (1,0)       | 1  (x0) |
| (2,0)       | 2  (x1) |
| (0,1)       | 4  (y0) |
| (0,2)       | 8  (y1) |
| (4,0)       | 32 (x2) |
| (0,4)       | 64 (y2) |

The distinguishing feature vs a naive Morton is the **`[x0, x1, y0, y1]` low nibble** — a 4×4 block, not
an interleaved 2×2 `[x0,y0,x1,y1]` or `[y0,x0,y1,x1]`. That single detail is what removes the edge
serration; getting the higher pairs wrong instead *scrambles* the image (easy to reject visually).

### The within-tile order is bytes-per-element-DEPENDENT

A 4 KB tile is a fixed 4096 bytes, so its texel dimensions depend on bpe (1 B → 64×64, 2 B → 64×32,
4 B → 32×32, 8 B → 32×16, 16 B → 16×16; see `sw4kb_dims`). Crucially, the *within-tile element order is
not the same at every bpe* — AMD's standard-swizzle `SW_PATTERN` genuinely differs per element size.
Two orders are now **pixel-verified against The Messenger's live surfaces**:

| bpe | tile | low block | full order (element bit → coord) |
|-----|------|-----------|----------------------------------|
| 4 B (RGBA8) | 32×32 | `[x0,x1, y0,y1]` (L=2) | `x0 x1 y0 y1 y2 x2 y3 x3 y4 x4` |
| 1 B (R8)    | 64×64 | `[x0,x1,x2,x3, y0,y1,y2,y3]` (L=4) | `x0 x1 x2 x3 y0 y1 y2 y3 y4 x4 y5 x5` |

Both share the shape **"a low block of L X-bits then L Y-bits, then interleaved `(y,x)` Morton pairs"**,
but with **L = 2 at 32 bpp and L = 4 at 8 bpp** (`sw4kb_morton` selects L by tile geometry). The earlier
version of this code applied the 32-bpp L=2 pattern at *every* bpe; that is correct at 32 bpp but at 8 bpp
it scrambles every 64×64 tile into an **unreadable weave** — which is exactly why the intro **caption text
was invisible**: the game's only R8 surface is the **2048×1024 caption/glyph font atlas** (a TextMeshPro-
style atlas of Latin + CJK glyphs), sampled by a ~260-quad batched text mesh. With L=2 the atlas decoded to
noise, so every glyph sampled garbage coverage and the text rendered as nothing. With **L=4 the atlas
resolves into a clean grid of readable glyphs** and the caption renders. (#102 was therefore *two* bugs: the
32-bpp low-nibble error #118, and this separate 8-bpp order.)

It is **round-trip-verified** at all bpe (tile∘detile = identity) and **pixel-verified** at 32 bpp and
8 bpp. The 16-bpp and BC-block geometries (64×32, 32×16, 16×16) fall back to the L=2 shape; they round-trip
but have no game-observed instance to pixel-verify yet — spot-check visually as those surfaces surface.

### How the 8-bpp order was found (the TV-minimisation trick)

Guessing 8-bpp bit orders by hand failed (a dozen hand-picked orders + several tile geometries all stayed a
weave). What cracked it: a **total-variation-minimising search** over within-tile bit permutations. A
correct de-swizzle of *structured* data (glyphs) is spatially smooth, so the permutation that minimises
Σ|Δpixel| over a content-rich crop is the right one. Hill-climbing (swap two bit-assignments, keep if TV
drops) from a few seeds converged straight onto `[x0 x1 x2 x3 y0 y1 y2 y3 y4 x4 y5 x5]`, which rendered the
atlas as readable Latin + CJK glyphs. The raw tiled R8 bytes were captured with `PROSPER_DUMP_RAWTILE`
(extended to the narrow single-channel path in `live_renderer.cpp`) and the search run offline in NumPy.
This trick works for any scrambled-but-structured surface where you lack the reference order — let
smoothness pick the permutation, then confirm on a region with known content (here, the Latin glyph block).

## How it was derived (method, for reproducibility)

Empirical bisection against a known-good reference image (the game's title art), because the exact order
was not available in reference code:

1. **Capture raw tiled bytes.** `PROSPER_DUMP_RAWTILE` (extended to dump small textures by address) writes
   the exact bytes read from guest memory for a sampled surface — e.g. the 1920×1080 title composite.
2. **Confirm it's a de-swizzle problem, not source data.** De-swizzling the raw bytes with our current +
   many alternative bit orders always left the pattern *in flat areas smooth but edges serrated* — proving
   the bytes are coherent and only our permutation is off (offline tool: `deswizzle_sweep.py`).
3. **Bisect the permutation visually.** Flipping the *high* bit-pairs scrambled the whole image (so they
   were already correct); the clean candidates differed only in the **low 4 bits**. Zooming a hard sprite
   edge (the ninja's sword/hand) at each of the 4 low-nibble arrangements, exactly one — `[x0,x1,y0,y1]` —
   produced crisp cel-shaded edges with **zero serration**; the others serrated.
4. **Cross-check.** Two independently-listed orders that a human confirmed "perfect" reduced to the same
   permutation, and flat gradients are perfectly smooth (a wrong low bit would still be invisible there,
   but a wrong *high* bit would not — and those were rejected in step 3).

Wrong references to avoid: shadPS4's `tiling.cpp` / `micro_32bpp.comp` implement **gfx9 (PS4) GNM**
bank/pipe macro-tiling (an inverse-Morton `rmort` LUT, X-low micro-tile, pipe/bank interleave). That is a
*different* scheme; PS5 `tile_mode == 5` is gfx10 `SW_4KB_S`, a pure bit-permutation with **no** pipe/bank
term. Mixing the two sent this investigation down a long detour.

## Where it lives

`src/gpu/tile.cpp` → `sw4kb_morton()`. Golden asserts in `tests/test_tile.cpp` (`gpu_surface_detile`).
Diagnostics: `PROSPER_DUMP_RAWTILE`, `PROSPER_DUMP_ATLAS`, and `deswizzle_sweep.py` (offline order sweep).

Refs: #118 (dither), #102 (missing text — same root).
