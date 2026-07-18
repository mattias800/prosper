# PM4 register-definition audit — prosper vs the RDNA2 (gfx10.3) register database (2026-07)

A systematic sweep of prosper's GPU register definitions and their consumers against the
authoritative RDNA2 register truth, run against master `53df2b7` on 2026-07-18. Companion to the
RDNA2 ISA recompiler audit (`RDNA2_ISA_AUDIT_2026_07.md`, issues #878–#883) and tracked by #911.

## Scope and reference

- **In scope:** the register OFFSET constants and per-field `*_SHIFT` / `*_MASK` constants in
  `src/gpu/pm4_registers.hpp` (887 constants across DB / CB / PA / SPI / COMPUTE / GE-VGT-TA groups),
  the standard `IT_*` PM4 type-3 opcode constants, and the register-field extraction in the
  consumers (`render_state.cpp`, `command_processor.cpp`, and `gpu_executor.cpp`'s dispatch resolve).
- **Deliberately out of scope:** the PM4 packet *framing* and payload layout in `pm4_decode.cpp`.
  That is prosper's own **AGC-internal** encoding produced by `hle_agc.cpp` (the `IT_NOP` `R_*`
  sub-opcodes are a Sony/Kyty AGC convention, not hardware PM4), a producer/consumer contract
  verified by the existing `test_pm4_decode` / `test_command_processor` / `test_agc_dcb` tests and by
  `hle_agc.cpp` consistency — not by an AMD hardware reference. Sweeping it against the AMD PM4 spec
  would produce spurious findings, so it was excluded.
- **Reference (authoritative):** the Mesa AMD register database (`gfx10.json` base + `gfx10.3`
  overlay = RDNA2), flattened to a grep-able `REG name offset= / FIELD name bits=[hi:lo] shift= mask=`
  form. Offsets were reconciled through the class base convention (prosper's constant is a
  class-relative dword index: CONTEXT base `0xA000`, SH base `0x2C00`, UCONFIG base `0xC000`, versus
  the reference's absolute byte address). Field shift/mask are convention-independent and directly
  comparable. `IT_*` opcodes were cross-checked against the stable consensus gfx10 PM4 opcode values.

## Method

Eight register-group reviewers compared their slice of `pm4_registers.hpp` + the consumers with the
reference; every candidate finding was re-derived by an independent adversarial verifier (the most
common false positive — an offset "mismatch" that dissolves once the class base is applied — was
explicitly guarded against). Deliberate PS5/AGC deviations backed by live evidence, and registers
prosper simply does not define/consume, were excluded by design.

## Result

The register layer is **overwhelmingly correct**: every offset reconciles, and every field actually
extracted by a consumer uses the right register, shift, and mask. Across all 887 constants and their
consumers the sweep produced **one confirmed defect** (low severity) and **one refuted candidate**.

### Confirmed (fixed in this PR)

**[LOW] `resolve_compute_launch` reads `COMPUTE_NUM_THREAD_X/Y/Z` as full 32-bit values instead of
the `NUM_THREAD_FULL` field [15:0]** — `src/gpu/gpu_executor.cpp:2184`.

- **Reference:** `COMPUTE_NUM_THREAD_X.NUM_THREAD_FULL` = bits [15:0] (mask `0xFFFF`);
  `NUM_THREAD_PARTIAL` = bits [31:16] (residual-thread count for partial dispatches, which prosper
  does not model).
- **Code behavior:** the consumer assigned `out.local_x/y/z = reg(COMPUTE_NUM_THREAD_*)` with no
  mask, folding any nonzero `NUM_THREAD_PARTIAL` into the workgroup local size.
- **Consequence / severity:** the exercised titles program these registers with small counts (high
  half zero), so the value is correct today — hence LOW. A guest that ever set `NUM_THREAD_PARTIAL`
  would get a wildly wrong local size (e.g. `0x3_0020` instead of `32`) and a corrupt dispatch.
- **Fix:** add the `COMPUTE_NUM_THREAD_FULL_SHIFT/MASK` field constants (they were missing) and mask
  the read to [15:0]. Regression test in `test_command_processor.cpp` programs a set `PARTIAL` field
  and asserts the resolved local size ignores it.

### Refuted (documented, not changed)

**Index-size consumer treats any nonzero index type as 32-bit** — `command_processor.cpp:1499`,
`uint32_t elem = index_type ? 4u : 2u`. In hardware terms an 8-bit index (`VGT_INDEX_TYPE_MODE`:
16-bit=0, 32-bit=1, 8-bit=2) would resolve to 4 bytes instead of 1. **Refuted as unreachable BY
CONSTRUCTION:** `index_type` here is NOT a decoded `VGT_INDEX_TYPE` hardware register field — it is
`c.index_size = pl[0]` of the AGC `IT_INDEX_TYPE` packet, i.e. the argument to `sceAgc*SetIndexSize`
(the Gnm/AGC `IndexSize` enum), which exposes only 16-bit (0) and 32-bit (1). There is no 8-bit
`IndexSize` in the AGC path at all, so no reached — indeed no *representable* — value hits the wrong
branch. The `elem = index_type ? 4u : 2u` form is therefore correct for every value the AGC API can
produce; a defensive remap would be dead code. (Recorded so that if prosper ever decodes the raw
hardware `VGT_INDEX_TYPE` register directly, this branch is revisited.)

## Per-group coverage (all verified clean)

- **DB (219):** all offsets reconcile (context base); every consumed field
  (`DB_DEPTH_CONTROL`, `DB_STENCIL_CONTROL`, `DB_STENCILREFMASK[_BF]`, `DB_RENDER_CONTROL`,
  `DB_SHADER_CONTROL`, clears) uses the correct shift/mask. Legacy GFX6-named fields
  (`DB_DEPTH_INFO` tile-mode) are defined but never field-extracted → no wrong output.
- **CB (153):** color-target dims (`CB_COLOR*_ATTRIB2` MIP0_WIDTH/HEIGHT +1), formats
  (`CB_COLOR*_INFO`), blend (`CB_BLEND*_CONTROL`), and the `CB_TARGET_MASK` nibble ordering all match.
- **PA (190):** cull/front-face/poly-mode (`PA_SU_SC_MODE_CNTL`), scissor, and clip/viewport fields
  all match (the field layout underlying the historical #534 front-face bug is correct here — that
  bug was in the Vulkan enum translation, not the register decode).
- **SPI (160):** context vs SH register-class split correct; `SPI_SHADER_PGM_*` RSRC1/RSRC2 fields,
  PS input mapping, and the 256-byte-aligned program-address encoding all match.
- **COMPUTE (48):** all offsets (SH base) and RSRC1/RSRC2/DISPATCH fields match; the one consumer
  masking defect above is the sole finding.
- **GE-VGT-TA (43):** primitive/index-type registers and fields match (the one consumer question was
  the refuted VGT_INDEX_TYPE note above).
- **IT opcodes (48):** the standard PM4 type-3 opcode constants match the consensus gfx10 values.
