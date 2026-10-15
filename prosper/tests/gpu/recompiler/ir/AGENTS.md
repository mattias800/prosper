# `tests/gpu/recompiler/ir` — tests for the recompiler IR (`src/gpu/recompiler/ir`)

One focused test per piece: `test_ir_core` (Block rules, evaluation edges, emission order),
`test_ir_lower_<fmt>` (one per lowering file: what it writes and what it refuses),
`test_ir_scalar_fold` (the constant folder, against values worked by hand from the ISA) and
`test_ir_scalar_alu_execution`, which compiles every scalar-op definition through
`recompile_compute`, runs the module in `../bpermute_spirv_oracle.hpp`'s integer interpreter and
compares against the definition's own evaluation. `vector_lane_harness.hpp` is the per-lane
counterpart every vector family's execution test calls (`test_ir_vector_alu_execution`): own
operands per lane, a partial EXEC, and inactive lanes checked for their old VDST. A new family adds its tests here and to the
`foreach(ir_test ...)` list in `prosper/CMakeLists.txt`; the recipe is
`src/gpu/recompiler/ir/HOW_TO_ADD_A_FAMILY.md`.
