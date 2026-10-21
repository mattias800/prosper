# `ir/lower` — instruction lowering and register bookkeeping

This folder connects decoded RDNA2 instructions to the IR definitions in `../ops/` and to the
SPIR-V emitter. Operand reads, destination writes, execution masks and proof-state updates belong
here; operation values belong in `ops/`. The parent [folder map](../AGENTS.md) and
[family recipe](../HOW_TO_ADD_A_FAMILY.md) describe that boundary.

The `*_routes.cpp` and `*_stages.cpp` files also hold mechanically relocated direct-emission code.
Those routes still call `SpirvCompute`; their order and shared state preserve the original
`emit_alu` behavior. Moving a route does not migrate its semantics onto the IR.

`lowering_context.hpp` provides the shared emitter/register/instruction references. Scalar and
vector family files own their format's lowering. Memory address/value definitions retain separate
lowerings from resource resolution and access routes. In particular, `flat_case_routes.*` owns FLAT
access and proof bookkeeping, resolved windows and per-invocation scratch; `flat_window*` owns its
address arithmetic.

`vop3_case_routes.*` holds the remaining VOP3 dispatch chain, cube coordinates and MBCNT routes.
It borrows the parent operand and modifier callbacks; the parent retains their setup, CLAMP check
and destination predication. Its tail methods preserve the original else-chain order.

Keep extracted files at most 500 lines and functions around 80 lines. Use the refactor tools for
moves, preserve emission order, and qualify relocations with token read-back and the corpus gate.
