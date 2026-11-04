#pragma once
// Independent device oracle: native BroadcastFirst inside real shader divergence, followed by
// an OR reduction at its merge. It uses neither the production ballot/index nor its shuffle.
#include "gpu/recompiler/float_transport_config.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <vector>

namespace prosper::test {
inline std::vector<uint32_t> native_readfirst_reference(uint32_t width, float threshold,
                                                        gpu::FloatTransportConfig transport,
                                                        bool whole_quad_mode = false) {
    using namespace gpu;
    SpirvCompute b;
    b.wave_size = width;
    b.float_transport = transport;
    b.begin_fragment(nullptr, 1);
    std::vector<uint32_t> extension;
    b.pstr(extension, "SPV_KHR_maximal_reconvergence");
    b.putv(b.exts, Op_Extension, extension);
    b.put(b.exec, Op_ExecutionMode, {b.f_main, 6023});
    extension.clear();
    b.pstr(extension, "SPV_KHR_quad_control");
    b.putv(b.exts, Op_Extension, extension);
    b.put(b.caps, Op_Capability, {5087});
    b.put(b.exec, Op_ExecutionMode, {b.f_main, 5089});
    const auto x = b.fragcoord_component(0);
    const auto initial_low = b.fragment_wave_ballot_half(b.btrue(), 0);
    b.put(b.caps, Op_Capability, {Cap_GroupNonUniformArithmetic});
    const auto entry = b.cur_block;
    const auto selected = b.id(), merge = b.id();
    const auto pixel_valid = whole_quad_mode ? b.btrue() : b.logical_not(b.helper_invocation());
    const auto active =
        b.land(pixel_valid, b.ucmp(Op_UGreaterThanEqual, x, b.uconst(fbits(threshold))));
    b.put(b.code, Op_SelectionMerge, {merge, 0});
    b.put(b.code, Op_BranchConditional, {active, selected, merge});
    b.emit_label(selected);
    const auto first = b.id();
    b.put(b.code, 338, {b.t_u32, first, b.uconst(Scope_Subgroup), x}); // BroadcastFirst
    b.put(b.code, Op_Branch, {merge});
    b.emit_label(merge);
    const auto contribution = b.emit_phi_2way(b.t_u32, b.uconst(0), entry, first, selected);
    const auto combined = b.id();
    b.put(b.code, 360,
          {b.t_u32, combined, b.uconst(Scope_Subgroup), 0, contribution}); // BitwiseOr Reduce
    const auto lane_zero = b.id();
    b.put(b.code, 337,
          {b.t_u32, lane_zero, b.uconst(Scope_Subgroup), x, b.uconst(0)}); // Broadcast
    const auto chosen = b.sel(b.ucmp(Op_INotEqual, combined, b.uconst(0)), combined, lane_zero);
    const auto color = b.fbin(Op_FMul, chosen, b.uconst(fbits(0.125f)));
    // A lane-zero empty-EXEC observation is only meaningful when that lane exists in the cohort.
    const auto lane_zero_exists =
        b.ucmp(Op_INotEqual, b.ibin(Op_BitwiseAnd, initial_low, b.uconst(1)), b.uconst(0));
    b.export_color(0, color, color, color,
                   b.sel(lane_zero_exists, b.uconst(fbits(1.0f)), b.uconst(0)));
    return b.finish();
}
}   // namespace prosper::test
