# `recompiler` — RDNA2 machine code → SPIR-V

Takes a guest shader's instruction bytes and emits a SPIR-V module.

- `rdna2_decode` — instruction decode: formats, opcodes, operands. Pure and side-effect free, which
  makes it the cheapest thing in the stack to unit-test.
- `rdna2_to_spirv` (+ `_internal`, `emit_alu`, `emit_cfg`, `alu_support`, `cfg_support`) — the
  translator: register state, control-flow structurization, and per-instruction lowering.
- `spirv_builder` — small hand-built SPIR-V modules. **One of them SHIPS**:
  `frontends/shared/live/live_compute.cpp`'s `prepare_compare_pipeline()` feeds
  `build_compute_compare_uvec4()` straight to `vkCreateShaderModule` on the live path. Treating
  them as test fixtures is what let an invalid `OpAccessChain` reach real devices (#1711), and
  `tools/spv_validate` exists to stop exactly that — so a change here is a change to shipped
  shader code, not to a fixture. It is not a general emitter either.
- `gta5/`, `indirect/` — see their own AGENTS.md.

**An unsupported op is a FATAL gap, not an acceptable skip.** Reject paths exist as a fail-visible
backstop for genuinely unknown encodings — mark `CONFIDENCE: LOW`, log loudly, file an issue with the
exact opcode — but every one hit on a live boot is the next thing to implement. A silently skipped
instruction drops real rendered content and reads as "handled".

## `PROSPER_CFG_TRIP_BOUND` — is this a non-terminating loop?

A guest program whose control flow neither structured emitter accepts is lowered by
`emit_cfg_state_machine` into ONE SPIR-V loop over a switch of dispatch ordinals. A recompiled loop
that never ends hangs the GPU into a driver reset, and from outside that is indistinguishable from a
slow shader or from a defect elsewhere in the submit. `PROSPER_CFG_TRIP_BOUND=N` caps that back edge
so the question becomes one run. It is a **diagnostic**: truncating guest control flow produces wrong
results by construction.

    PROSPER_CFG_TRIP_BOUND=4096            # required: the cap
    PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR  # one program, so other shaders stay byte-identical
    PROSPER_CFG_TRIP_BOUND_PHASE=0         # REQUIRED -- nothing is emitted without it
    PROSPER_CFG_TRIP_BOUND_ORDINAL=45      # optional: cap ONE loop inside a multi-loop program

Three things about it are load-bearing and each has cost someone a run:

- **It covers the CFG dispatcher only.** The two structured loop emitters never call it, so a null
  result from a structurizer-accepted program means *not measured*, never *does not run away*.
- **The witness is COMPUTE-only; the cap is not.** The device-side hit record lives in the internal
  GDS buffer, which only the compute executor binds, prepares and reads back. A graphics program is
  still capped — and says so, once, on stderr — but publishes nothing, so on a draw you must confirm
  the draw still ran (`PROSPER_DRAW_PROGRAM_CENSUS`) before reading a surviving device as a hit.
- **A plain cap says "some loop here", not "this loop".** `_ORDINAL=K` counts only the traversals
  about to re-enter ordinal K, so one arm per candidate header localizes the runaway; read the
  ordinal → guest-pc map the arm prints, never map one by hand. Astro Bot's world-map pixel program
  `0x5008f1400` has four guest loops behind one dispatcher and this is how the runaway among them was
  named (#3193).

Every SPIR-V emitter path is `spirv-val`-gated in CI (`tools/spv_validate`) with one representative
module per path, not one per game shader.

Primary reference: *"RDNA 2" Instruction Set Architecture Reference Guide* (AMD doc 70648). PS5
extensions and AGC behaviour still require live title evidence.
