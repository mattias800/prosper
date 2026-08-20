# `src/gpu` — the AGC → Vulkan translation stack

Everything that turns a PS5 guest's GPU work into Vulkan work lives here. The folders are ordered by
**the direction data flows through the stack**, and that ordering is the fastest way to guess where
something belongs:

| folder | what it owns |
| --- | --- |
| `pm4/` | decoding the guest's command buffer |
| `agc/` | Sony AGC / Gen5 descriptor formats — reading a shader's declared resources |
| `recompiler/` | RDNA2 machine code → SPIR-V |
| `resources/` | resolving what a shader binds to real memory and images |
| `state/` | fixed-function render state, and its translation to Vulkan |
| `texture/` | guest texture memory: tiling, block compression, layout |
| `execute/` | building and submitting the actual GPU work |
| `present/` | scanout |
| `capture/` | serializing a frame for offline replay |
| `timeline/` | frame timelines and bundles |
| `diagnostics/` | observation only — nothing here is on a rendering path |

A folder is a **claim about coupling**: things in one folder are expected to change together. Where
that turns out to be false, the seam is worth recording on the structure issue rather than smoothing
over by dropping a file into whichever folder is convenient.

**The renderer itself is not here.** `src/gpu` is the translation and decode layer; the live Vulkan
renderer and its per-frame orchestration live under `frontends/shared/live/`. If you are chasing "why
does this frame look wrong", the split matters: a wrong *descriptor* is usually this tree, a wrong
*draw order or barrier* is usually that one.

Build note: `CMakeLists.txt:71` globs `src/*.cpp` recursively with `CONFIGURE_DEPENDS`, so a new
subfolder here needs no source-list edit.

**These files cite SYMBOLS AND UNIQUE STRINGS, never `file:line`.** That is a deliberate policy,
not an oversight, and it was bought expensively during the review of the commit that added them:

- Three citations named line numbers in `rdna2_to_spirv.cpp` taken from before #2752 split it from
  25,448 lines to 4,070, so they pointed **past EOF within a day**. The author had verified them
  in a worktree 13 commits behind master, where they were true.
- A replacement citation into `gpu_executor.cpp` was **correct when written** and was displaced by
  an unrelated merge hours later. It then still *resolved* — into an unrelated block — which is
  worse, because it fails silently instead of visibly.

A folder map is exactly the document that outlives the line numbers in it. A symbol survives a
restructure, a `grep -rn` finds it in a second, and a wrong one is obvious rather than plausible.
The two `CMakeLists.txt` line numbers below are the deliberate exception, and even those should be
confirmed by grepping the construct rather than trusted.

**Do not confuse `src/gpu/diagnostics/` with `src/diagnostics/`.** They are different folders with
opposite build treatment: `CMakeLists.txt:75-77` excludes `src/diagnostics/` unless
`PROSPER_DIAGNOSTICS` is on (default OFF at `:21`), and that exclusion's pattern does **not** match
this one, which always builds.
