# `recompiler` — RDNA2 machine code → SPIR-V

Takes a guest shader's instruction bytes and emits a SPIR-V module.

- `rdna2_decode` — instruction decode: formats, opcodes, operands. Pure and side-effect free, which
  makes it the cheapest thing in the stack to unit-test.
- `rdna2_to_spirv` (+ `_internal`, `emit_alu`, `emit_cfg`, `alu_support`, `cfg_support`) — the
  translator: register state, control-flow structurization, and per-instruction lowering.
- `spirv_builder` — small hand-built SPIR-V modules. **These SHIP**: `live_compute.cpp:2155` feeds
  `build_compute_compare_uvec4()` straight to `vkCreateShaderModule` on the live path. Treating
  them as test fixtures is what let an invalid `OpAccessChain` reach real devices (#1711), and
  `tools/spv_validate` exists to stop exactly that — so a change here is a change to shipped
  shader code, not to a fixture. It is not a general emitter either.
- `gta5/`, `indirect/` — see their own AGENTS.md.

**An unsupported op is a FATAL gap, not an acceptable skip.** Reject paths exist as a fail-visible
backstop for genuinely unknown encodings — mark `CONFIDENCE: LOW`, log loudly, file an issue with the
exact opcode — but every one hit on a live boot is the next thing to implement. A silently skipped
instruction drops real rendered content and reads as "handled".

Every SPIR-V emitter path is `spirv-val`-gated in CI (`tools/spv_validate`) with one representative
module per path, not one per game shader.

Primary reference: *"RDNA 2" Instruction Set Architecture Reference Guide* (AMD doc 70648). PS5
extensions and AGC behaviour still require live title evidence.
