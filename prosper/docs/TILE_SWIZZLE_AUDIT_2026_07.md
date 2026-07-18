# GFX10 tiling / detile swizzle audit — prosper vs AMD addrlib (2026-07)

A bit-exact sweep of prosper's GFX10 tiling/detile swizzle math (`src/gpu/tile.cpp`) against the AMD
address library (`addrlib`) GFX10 swizzle-pattern tables, run against master `e00a9b0` on 2026-07-18.
Sixth GPU-audit sweep (recompiler #878–#883, PM4 #911, descriptors #912, vk_translate #913, BC decode
#914). Tracked by #924.

## Result: standard-swizzle + DCC/detile clean; one real R_X/Z_X pattern-family finding (tracked in #928)

The swizzle is a pure deterministic bit-permutation, so findings are bit-exact verifiable. The
standard-swizzle (`SW_4KB_S`, `SW_64KB_S`) tables and the DCC/detile geometry are **correct**. The
pipe/bank-interleaved `SW_64KB_R_X` / `SW_64KB_Z_X` modes carry a genuine defect — they were
transcribed from the **Navi1x (gfx10.1)** patterns, but PS5 is **gfx10.3 = RB+** — filed as **#928**
and deliberately **not** fixed in this sweep (see below).

## Reference

The AMD addrlib GFX10 swizzle-pattern header (Mesa-vendored `gfx10SwizzlePattern.h`), decoded via its
`PATINFO → NIBBLE01/2/3` equation tables. PS5's Oberon GPU is gfx10.3 (RDNA2) = RB+, so the
`GFX10_SW_*_RBPLUS_*` pattern variants are the authoritative ones. Two prosper derivation docs
(`GFX10_SW_4KB_S_TILING.md`, `GFX10_SW_64KB_TILING.md`) and the pixel-verified 32-bpp (#118) and 8-bpp
R8-font (#256) orders are additional ground truth.

## Clean (verified bit-exact)

- **SW_4KB_S** — all five per-bpp element orders (1/2/4/8/16 B) match the addrlib `SW_4K_S_RBPLUS`
  pattern *and* the two pixel-verified orders; the `#379` low-X/Y-pair regression is not present; the
  per-bpp tile geometry (64×64 … 16×16, wide-before-tall) is correct.
- **SW_64KB_S** — the 64KB standard-swizzle bit order and 4KB→64KB tile-bit extension match addrlib.
- **DCC / detile geometry** — the DCC fast-clear code interpretation (`0x00/0x40/0x80/0xC0` →
  RGBA `0000/0001/1110/1111`, with `alpha_is_on_msb` placement and 3-component A=255 forcing) is
  correct; the mip/tail addressing reproduces addrlib's tail-first-then-reversed chains bit-exactly
  against the tests; 2D-array slice stride and 2D/3D volume block-grid math are correct. (Caveat: the
  vendored addrlib header carries only the swizzle/DCC index tables, not the `GetMetaBlkSize` decoder,
  so the DCC meta-block dimensions were verified against the existing tests rather than re-derived
  numerically — no discrepancy found.)

## The finding (tracked in #928, not fixed here)

`kSw64kRX` / `kSw64kZX` reproduce the **Navi1x** `GFX10_SW_64K_{R,Z}_X_1xaa` 16-pipe rows exactly, but
the `_RBPLUS_` (gfx10.3) equations differ: the pipe-select **output bit 8** is `X3^Y3` (Navi1x) where
RB+ uses a `Y4^(high X^Y)` term; the R_X **1bpe micro-tile** uses the Navi1x order where RB+ matches
prosper's own pixel-verified S-mode 1B order; and R_X **4bpe bits 6/7** swap X2/Y2. These send some
texels to the wrong 64KB-block pipe / byte offset on the mode-27 and mode-24 detile paths.

**Why it is not fixed in this sweep** (all three are independently disqualifying for an unverified
change): (1) **cross-lane** — mode 27 is DOLL's render targets (`area:ue4`) and mode 24 is Astro Bot's
(#825), so it needs those lanes' coordination and pixel evidence; (2) **no pixel oracle** — no clean
authored R_X/Z_X surface exists in-tree to validate a fix (captures are speckle / compute-dest), and
the current tables were empirically tuned, so a blind change risks regressing live behavior; (3) the
exact RB+ pipe equation is **PKR-count-dependent** (4/8/16), unknown without a live capture (though the
bits-9/10/11 match strongly implies 8-PKR). #928 records the full bit-exact analysis and the unblock
checklist. This is the project's evidence-hierarchy rule in action: never change a live-affecting,
empirically-tuned PS5 path to match a secondary reference without direct title evidence.
