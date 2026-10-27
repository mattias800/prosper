# `tests/gpu/recompiler/ir` — tests for the recompiler IR (`src/gpu/recompiler/ir`)

One focused test per piece: `test_ir_core` (Block rules, evaluation edges, emission order),
`test_ir_lower_<fmt>` (one per lowering file: what it writes and what it refuses),
`test_ir_scalar_fold` (the constant folder, against values worked by hand from the ISA) and
`test_ir_scalar_alu_execution`, which compiles every scalar-op definition through
`recompile_compute`, runs the module in `../bpermute_spirv_oracle.hpp`'s integer interpreter and
compares against the definition's own evaluation. `vector_lane_harness.hpp` is the per-lane
counterpart every vector family's execution test calls (`test_ir_vector_alu_execution`): own
operands per lane, a partial EXEC, and inactive lanes checked for their old VDST.
`vector_compare_harness.hpp` wraps it for VOPC compares, which write a lane mask instead of a VGPR:
it reads the mask back into v1 for every lane (inactive lanes must read 0) or, for V_CMPX, writes
v1 under the narrowed EXEC. A new family adds its tests here and to the
`foreach(ir_test ...)` list in `prosper/CMakeLists.txt`; the recipe is
`src/gpu/recompiler/ir/HOW_TO_ADD_A_FAMILY.md`.
The interpreter fails an out-of-range buffer access instead of returning 0, so a memory family's
robustness arm runs on Vulkan: `test_ir_buffer_address_robustness` links Vulkan and is registered
beside `test_traversal_chase`, not in that list. It cannot run an image access at all, so
`test_ir_image_coordinates_execution` runs its arms (`image_coordinates_cases.hpp`) on the live
Vulkan compute backend, registered beside `test_ir_buffer_store_execution`.
`test_ir_cfg_guest_cfg`, `test_ir_cfg_region_tree`, `test_ir_cfg_region_loops` and
`test_ir_cfg_region_shadow` cover `src/gpu/recompiler/ir/cfg/` on synthetic streams
(`cfg_streams.hpp` holds their encoders). The shadow test also runs every guest-program fixture in
`tests/data` under each stage's rules and needs the region tree's ifs and loops to agree with
`detect_forward_ifs`, `detect_divergent_loops` and `detect_counted_loop` at the pc level.

The buffer address, format, store and PC-relative bounds execution fixtures continue through
`recompile_compute` after opcode classification moves to `lower/buffer_opcode_case_routes.*`.
That mechanical move adds no semantic assertions; token read-back controls and the full corpus
comparison qualify relocation, alongside these existing execution cases.

`vector_lane_harness` leaves ordinary instruction tapes and guest PCs unchanged. Hard controls
opt into `seed_lane_hard_scalar` to seed s50 from the possible scratch-offset
launch word. Its user-word argument keeps the seed after the actual
launch prefix; an unwritten, explicitly unloaded s50 would be Arbitrary under ADR 0038.
Explicit scalar producers in a tested body replace this seed normally.
