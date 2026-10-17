// DS shuffle-only arms moved by tools/refactor/extract_case_routes.py. Their source reads,
// destination writes and subgroup contracts retain the original ALU route order.
#include "gpu/recompiler/ir/lower/ds_cross_lane_lowering.hpp"

#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu::ir {

struct DsCrossLaneLowering {
    SpirvCompute& b;
    RegState& rs;
    const Rdna2Inst& in;
    bool& ok;

    std::optional<bool> route_bpermute();
    std::optional<bool> route_swizzle();
};

std::optional<bool> DsCrossLaneLowering::route_bpermute() {
    if (in.opcode == kDsOpcodeBpermuteB32) {
        // A native subgroup exactly matching the guest wave gives each architectural
        // 32-lane half a valid shuffle domain. Portable compute is routed through the
        // CFG common event; never put its workgroup rendezvous in this arbitrary ALU arm.
        if (!b.is_compute || in.ds_gds || !b.native_subgroup_size ||
            b.native_subgroup_size != b.wave_size) {
            ok = false;
            return true;
        }
        const auto address = rs.vreg.find(in.src[0].value);
        const auto source = rs.vreg.find(in.src[1].value);
        const uint32_t old = vreg_old(b, rs, in.dst.value);
        rs.vreg[in.dst.value] = b.ds_bpermute_b32(
            address == rs.vreg.end() ? b.uconst(0) : address->second,
            source == rs.vreg.end() ? b.uconst(0) : source->second, rs.exec, b.uconst(in.literal));
        predicate_write(b, rs, in.dst.value, old);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> DsCrossLaneLowering::route_swizzle() {
    if (in.opcode == 0x35) {                    // ds_swizzle_b32: VDST = lane-gather(ADDR)
        if (!b.is_compute && !b.is_fragment) {
            ok = false;
            return true;
        }
        uint32_t source_lane = 0;
        if (!b.ds_swizzle_source_lane(in.literal, &source_lane)) {
            ok = false;
            return true;
        }
        auto source = rs.vreg.find(in.src[0].value);
        const uint32_t source_value = source == rs.vreg.end() ? b.uconst(0) : source->second;
        const uint32_t shuffled = b.subgroup_shuffle(source_value, source_lane);
        const uint32_t active_word = b.sel(rs.exec, b.uconst(1), b.uconst(0));
        const uint32_t fetched_active = b.subgroup_shuffle(active_word, source_lane);
        const uint32_t result =
            b.sel(b.ucmp(Op_INotEqual, fetched_active, b.uconst(0)), shuffled, b.uconst(0));
        const uint32_t old = vreg_old(b, rs, in.dst.value);
        rs.vreg[in.dst.value] = result;
        predicate_write(b, rs, in.dst.value, old);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> lower_ds_cross_lane(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                                        bool& ok) {
    if (in.fmt != Rdna2Format::DS) return std::nullopt;
    DsCrossLaneLowering lowering{b, rs, in, ok};
    if (auto routed = lowering.route_bpermute()) return *routed;
    return lowering.route_swizzle();
}

} // namespace prosper::gpu::ir
