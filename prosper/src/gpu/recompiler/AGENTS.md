# `recompiler` — RDNA2 machine code → SPIR-V

Takes a guest shader's instruction bytes and emits a SPIR-V module.

- `rdna2_decode` — instruction decode: formats, opcodes, operands. Pure and side-effect free, which
  makes it the cheapest thing in the stack to unit-test.
- `rdna2_to_spirv` (+ `_internal`, `emit_alu`, `emit_cfg`, `alu_support`, `cfg_support`) — the
  translator: register state, control-flow structurization, and per-instruction lowering.
- `spirv_builder` — small hand-built SPIR-V modules used as test fixtures, **not** a general emitter.
- `gta5/`, `indirect/` — see their own AGENTS.md.

**An unsupported op is a FATAL gap, not an acceptable skip.** Reject paths exist as a fail-visible
backstop for genuinely unknown encodings — mark `CONFIDENCE: LOW`, log loudly, file an issue with the
exact opcode — but every one hit on a live boot is the next thing to implement. A silently skipped
instruction drops real rendered content and reads as "handled".

Every SPIR-V emitter path is `spirv-val`-gated in CI (`tools/spv_validate`) with one representative
module per path, not one per game shader.

Primary reference: *"RDNA 2" Instruction Set Architecture Reference Guide* (AMD doc 70648). PS5
extensions and AGC behaviour still require live title evidence.
