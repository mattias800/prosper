# `gpu/recompiler/ir` — the recompiler's SSA IR (ADR 0012)

RDNA2 instructions are moving, one instruction family at a time, from direct SPIR-V emission in
`emit_alu` (`../rdna2_emit_alu.cpp`) to a lowering through this IR. The scalar ALU (SOP1, SOP2, SOPC,
SOPK) is the first family. Everything here is private to `gpu/recompiler`.

| path | role |
| --- | --- |
| `ir.hpp`, `ir.cpp` | the IR core: `Type`, `Value`, `Opcode`, `Inst`, `Block` and the rules a Block enforces |
| `builder.hpp` | typed helpers that append instructions to a Block |
| `eval.hpp`, `eval.cpp` | the evaluation half of every opcode: what the constant folder computes |
| `emit/` | the SPIR-V half of every opcode: a Block emitted through `SpirvCompute`, in Block order |
| `ops/` | ONE definition per guest operation (`scalar_alu.cpp`, `scalar_compare.cpp`), used by both the folder and the emitter |
| `lower/` | RDNA2 -> IR per format: `sop1.cpp`, `sop2.cpp`, `sopc.cpp`, `sopk.cpp`, and the relocated `*_routes.cpp` |
| `HOW_TO_ADD_A_FAMILY.md` | the recipe for migrating the next family |

**The boundary that matters: `ops/` holds semantics, `lower/` holds bookkeeping.** A definition in
`ops/` says what D and SCC are as functions of the operands, nothing else. Which register receives D,
the descriptor tag it retires, wave-mask domains and proof state belong to the lowering. That split is
what lets the constant folder (`../rdna2_recompile_shared.hpp`, `shader_constant_operand`) and the
emitter share one definition (#2514) — before this, SOP2 was written twice and nothing kept the two
copies equal.

**The `*_routes.cpp` files are not IR yet.** They are emit_alu's mask-domain and proof-carrying
shapes (Bool-domain wave masks, VCC/EXEC bridges, entry-M0 tokens, 64-bit pair ops), moved verbatim by
`tools/refactor/extract_case_routes.py` and still calling `SpirvCompute` directly. Each is a member
function of its format's lowering class (`lower/scalar_lowering_internal.hpp`) returning what emit_alu
returned, or `nullopt` for "not my shape". Their ORDER in each `lower()` is the old `if` chain's order
and is load-bearing. Migrating a route onto IR definitions is later work; moving one without the
corpus gate is not allowed.

**Emission order is the byte-identity contract.** A Block is emitted in the order the builder
appended, and a definition appends in the order the old emitter called `SpirvCompute`, with the same
expression nesting (the compiler's argument evaluation order then matches too). A whole-program pass
that reorders instructions does not belong here until a family PR explains and execution-tests the
resulting SPIR-V differences.

Size rules (ADR 0012 decision 5, enforced by `dir-file-cap` in `tools/ci/check_arch_ratchet.py`): a
file aims at 500 lines and must not pass 1,000; a function aims at 80. Seven relocated routes are
over 80 lines today because they were moved verbatim; splitting them is a behaviour-neutral follow-up.

Tests: `tests/gpu/recompiler/ir/`, one file per piece; `test_ir_scalar_alu_execution` runs every
definition's SPIR-V in the integer interpreter against its own evaluation.
