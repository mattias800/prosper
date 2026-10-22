// Native subgroup row and BPERMUTE emission.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>

namespace prosper::gpu {
uint32_t SpirvCompute::subgroup_row_shr_dynamic(uint32_t value, uint32_t active, uint32_t amount,
                                                uint32_t event, uint32_t* valid_lane) {
    mark_subgroup_min16();
    const uint32_t lane = subgroup_local_id();
    const uint32_t row_lane = ibin(Op_BitwiseAnd, lane, uconst(15));
    const uint32_t in_bounds = ucmp(Op_UGreaterThanEqual, row_lane, amount);
    // Keep the shuffle index valid even for row-leading lanes. FI=0 also makes an EXEC-inactive
    // source invalid. The caller uses `valid` to preserve VDST for BOUND_CTRL=0; returning VALUE
    // here is only a safe placeholder for the disabled lane, not its architectural result.
    const uint32_t source_lane = sel(in_bounds, ibin(Op_ISub, lane, amount), lane);
    const uint32_t shifted = subgroup_shuffle(value, source_lane);
    const uint32_t source_active = subgroup_shuffle(sel(active, uconst(1), uconst(0)), source_lane);
    uint32_t valid = land(in_bounds, ucmp(Op_INotEqual, source_active, uconst(0)));
    // A lane-local graphics dispatcher can have adjacent lanes parked at distinct static DPP
    // instructions in the same common phase.  Shuffling only ACTIVE would let one instruction
    // consume a neighbour published by another instruction.  Carry the static event tag through
    // the identical shuffle and require it to match the destination lane's event.
    if (event) {
        const uint32_t source_event = subgroup_shuffle(event, source_lane);
        valid = land(valid, ucmp(Op_IEqual, source_event, event));
    }
    if (valid_lane) *valid_lane = valid;
    return sel(valid, shifted, value);
}

uint32_t SpirvCompute::subgroup_row_xor(uint32_t value, uint32_t active, uint32_t stride,
                                        uint32_t* valid_lane) {
    mark_subgroup_min16();
    const uint32_t lane = subgroup_local_id();
    const uint32_t source_lane = ibin(Op_BitwiseXor, lane, uconst(stride & 15u));
    const uint32_t rotated = subgroup_shuffle(value, source_lane);
    // FI=0 makes an EXEC-inactive source invalid. The one admitted form has BOUND_CTRL=1,
    // whose caller substitutes zero for that invalid source before V_MIN_F32.
    const uint32_t source_active = subgroup_shuffle(sel(active, uconst(1), uconst(0)), source_lane);
    const uint32_t valid = ucmp(Op_INotEqual, source_active, uconst(0));
    if (valid_lane) *valid_lane = valid;
    return sel(valid, rotated, value);
}

uint32_t SpirvCompute::ds_bpermute_b32(uint32_t address, uint32_t value, uint32_t active,
                                       uint32_t offset, uint32_t event) {
    // RDNA2 ISA 10.4.4: ADDR is a byte address and BPERMUTE gathers DATA0 backward from
    // ((ADDR + OFFSET) >> 2) & 31 within each independent 32-lane half. Addition is deliberately
    // performed before the shift so uint32 overflow and unaligned byte addresses match hardware.
    mark_subgroup_min32();
    const uint32_t lane = subgroup_local_id();
    const uint32_t selected =
        ibin(Op_BitwiseAnd, ibin(Op_ShiftRightLogical, ibin(Op_IAdd, address, offset), uconst(2)),
             uconst(31));
    const uint32_t source_lane =
        ibin(Op_BitwiseOr, ibin(Op_BitwiseAnd, lane, uconst(~31u)), selected);
    const uint32_t shuffled = subgroup_shuffle(value, source_lane);
    const uint32_t source_active = subgroup_shuffle(sel(active, uconst(1), uconst(0)), source_lane);
    uint32_t valid = ucmp(Op_INotEqual, source_active, uconst(0));
    if (event) {
        const uint32_t source_event = subgroup_shuffle(event, source_lane);
        valid = land(valid, ucmp(Op_IEqual, source_event, event));
    }
    return sel(valid, shuffled, uconst(0));
}
}   // namespace prosper::gpu
