---
kind: investigation
status: current
---

# VINTRP value migration: native parameter boundary

Source audit on main `40d95db32efe`, 2026-10-19. The
[current queue](https://github.com/mattias800/prosper/issues/5325#issuecomment-6261028018)
requests interpolation arithmetic through IR while preserving attribute/M0 resolution,
P1/P2 pairing, definedness and admission. This audit reaches its stop condition before
relocation or an IR definition: the native route has no coefficient arithmetic to move.

## Existing routes

`emit_alu`'s VINTRP case in `prosper/src/gpu/recompiler/rdna2_emit_alu.cpp` makes
P1 a no-op. P2 calls `SpirvCompute::interp_read`, which loads a component of a
Vulkan Input. It does not consume the encoded barycentric VGPR or reaching VDST.
MOV calls `SpirvCompute::interp_parameter`: geometry-backed parameter locations
can supply the selected parameter, while the ordinary fallback accepts selector 2
through `interp_read` and refuses selectors 0 and 1. Both helpers live in
`prosper/src/gpu/recompiler/spirv/shader_io.cpp`.

This native substitution deliberately relies on Vulkan interpolation. It does not
provide already-resolved P0/P10/P20 words and barycentrics to the VINTRP case.
Replacing its Input load with coefficient arithmetic needs a parameter source and
a decision about which native instruction sequences that source can represent.
Passing the final Input value through an identity definition would preserve bytes
but would not migrate the requested interpolation arithmetic.

The separate owned-input path in
`prosper/src/gpu/recompiler/fragment_packet_resource_services.cpp` does consume
coefficient tuples and barycentrics. It uses `packet_f32_mul` and `packet_f32_add`,
retains actual reaching VDST for P2, and admits only exact, normal-or-zero
intermediates with a nonzero final result. Its preflight in
`fragment_packet_resource_preflight.cpp` accepts only VINTRP P1/P2, and rejects
P1 source/destination aliasing without HALF_LDS authority. It refuses MOV and
VOP3 interpolation; expanding it changes admission.

The instruction-effects table already describes all requested forms:
`ir/effects/misc_memory.inc` describes VINTRP and its M0/old-VDST reads;
`ir/effects/vector3.inc` describes the F32 VOP3 aliases and the P1LL/P1LV/P2 F16
forms. Those VOP3 forms have no corresponding route or definition in
`ir/lower/` or `ir/ops/`; `emit_alu`'s VOP3 dispatch reaches its refusal.
Effects-table coverage does not establish lowering support.

## Hand-worked ISA boundary

[AMD RDNA2 ISA, document 70648](https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content),
sections 10.4.2/Table 53 and 12.11, gives P1 as `P10 * I + P0` and P2 as
`P20 * J + reaching_D`. MOV selects P10, P20 or P0. Section 12.12 describes the
F16 parameter halves, P1LV's additional VGPR parameter and P2's halfword result.

For exactly representable P0=1, P10=2, P20=4, I=1/2 and J=1/4, P1 produces 2
and a matching P2 produces 3. If another instruction replaces D with 7 between
the two, P2 produces 8. A single already-interpolated Input cannot distinguish
those reaching-D histories. A standalone P1 with P0=1, I=0 and old D=0 produces
1 architecturally, while the native case preserves 0. These are hand calculations
and source-derived consequences, not fresh execution results.

The earlier independently assembled `c8000002` P1 witness and its measured native
output are recorded in
[#5301](https://github.com/mattias800/prosper/issues/5301#issuecomment-6219712556).
Neither that execution nor its llvm-mc check was rerun in this audit.

## Lead disposition and verification limits

The lead [accepted the stop](https://github.com/mattias800/prosper/issues/5325#issuecomment-6277035905)
and closed Group 7 as having no value-level arithmetic to migrate. Native
parameter, pairing and admission decisions remain lead work under #5301. No M0,
attribute, pairing, definedness, admission, source or test assertion was changed
by this audit.

The requested recorded and synthetic corpus gates, positive controls and full
CTest qualification did not run during the October 19 audit. The required ps5ys distrobox entry did not
reach configuration; a separate 20-second entry probe and a 10-second read-only
container-inspect probe both timed out with exit 124. No successful build or
runtime verification is claimed for that audit. A fresh entry probe succeeded
on October 20; the earlier timeout is an observation, not a continuing blocker.

CONFIDENCE: HIGH for the current source boundaries and exact hand calculations.
CONFIDENCE: MED for the scope of a future native solution; corpus impact and
runtime behavior remain unmeasured in this audit. Group 6 resumes separately
under the lead's updated queue. Refs #5325 #5146 #5301.
