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

`vop1_case_routes.*` owns the remaining VOP1 orchestration and early, SDWA and residual-switch
routes. It borrows the original operand callback and register map; source setup, route order and
final destination predication remain together in this family. Its legacy lane-local READFIRST
route asks `rdna2_arbitrary_placeholder_scalar` before a new arbitrary admission can expose a
missing broadcast; real dispatcher broadcasts do not use this lane-local route.

Keep extracted files at most 500 lines and functions around 80 lines. Use the refactor tools for
moves, preserve emission order, and qualify relocations with token read-back and the corpus gate.

`sopp_case_routes.*` owns the existing SOPP opcode switch, including branch refusals, hint handling,
the allocation callback and barrier emission. It borrows the original emitter state and safe-branch
set; the relocation preserves the direct-emission path.

`buffer_opcode_case_routes.*` classifies the existing MUBUF/MTBUF opcode groups using references
to the emitter's original locals. Descriptor resolution, access and disposition reporting remain
in `emit_alu`; the classifier preserves the original unsupported-opcode refusal.

The DS routes guard actual retry input reads before their early returns: compute/graphics ADDTID,
plain GDS reads, implicit M0 readers, and DS shuffle inputs share the arbitrary-input boundary.
The ordinary LDS callback alone cannot protect those early routes. Existing stored values,
addresses, predicates and route order remain with their family lowerings.
`emit_alu` also checks ordinary old-destination reads before lowering, using the existing
old-word predicate. This includes inactive-lane preservation, subword preservation and implicit
accumulation; complete overwrites and slot-map writers retain their separate contracts.

`bvh_case_routes.*` owns the existing MIMG BVH branch, including guarded node loads, triangle
and box calculations, sorting and destination predication. Its continuation methods keep original
local declarations alive while later stages borrow their references and existing callbacks.

`image_resolution_routes.*` owns the existing unresolved-MIMG reports and diagnostic branch.
It borrows the parent lookup helpers and resolution state; reporting keys, static deduplication,
diagnostic selection and refusal order remain together, while descriptor lookup stays in `emit_alu`.
