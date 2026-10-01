#pragma once

// Shared by strict spirv-val coverage and device execution. These are synthetic shaders, not
// captured game bytes. The pixel oracle is analytic screen-space dFdx(x)=dFdy(y)=1.
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::test::fragment_vote_execution {
constexpr uint32_t width = 8, height = 8;

inline std::vector<uint32_t> edge_vertex(uint32_t x, uint32_t y) {
    using namespace prosper::gpu;
    SpirvCompute b;
    b.begin_vertex();
    const uint32_t vertex = b.vertex_invocation_id();
    const auto choose = [&](float a, float c, float d) {
        return b.sel(b.ucmp(Op_IEqual, vertex, b.uconst(0)), b.uconst(fbits(a)),
            b.sel(b.ucmp(Op_IEqual, vertex, b.uconst(1)), b.uconst(fbits(c)), b.uconst(fbits(d))));
    };
    const auto ndc = [](float p) { return p / 4.0f - 1.0f; };
    // Only pixel (x,y) is covered: a small triangle around its center, three quad neighbors outside.
    b.export_position(choose(ndc(x + .5f), ndc(x + .9f), ndc(x + .1f)),
                      choose(ndc(y + .1f), ndc(y + .9f), ndc(y + .9f)),
                      b.uconst(0), b.uconst(fbits(1.0f)));
    return b.finish();
}

inline std::vector<uint32_t> derivative_fragment(bool poison_helpers = false) {
    using namespace prosper::gpu;
    SpirvCompute b;
    b.begin_fragment();
    b.export_color(0, b.uconst(0), b.uconst(0), b.uconst(fbits(1.0f)), b.uconst(fbits(1.0f)));
    const uint32_t vote = b.fragment_wave_any(b.btrue());
    const uint32_t arm = b.id(), merge = b.id();
    SpirvCompute::put(b.code, Op_SelectionMerge, {merge, 0});
    SpirvCompute::put(b.code, Op_BranchConditional, {vote, arm, merge});
    SpirvCompute::put(b.code, Op_Label, {arm});
    uint32_t x = b.fragcoord_component(0), y = b.fragcoord_component(1);
    if (poison_helpers) {
        // Deliberate fault control: losing helper values must redden the analytic pixel checks.
        const uint32_t helper = b.helper_invocation();
        x = b.sel(helper, b.uconst(0), x);
        y = b.sel(helper, b.uconst(0), y);
    }
    const uint32_t dx = b.id(), dy = b.id();
    SpirvCompute::put(b.code, 207, {b.t_f32, dx, b.bcf(x)}); // DPdx
    SpirvCompute::put(b.code, 208, {b.t_f32, dy, b.bcf(y)}); // DPdy
    b.export_color(0, b.fbin(Op_FMul, b.bcu(dx), b.uconst(fbits(.125f))),
                      b.fbin(Op_FMul, b.bcu(dy), b.uconst(fbits(.125f))),
                      b.uconst(0), b.uconst(fbits(1.0f)));
    SpirvCompute::put(b.code, Op_Branch, {merge});
    SpirvCompute::put(b.code, Op_Label, {merge});
    return b.finish();
}

inline std::vector<uint32_t> helper_witness() {
    using namespace prosper::gpu;
    SpirvCompute b;
    b.begin_fragment();
    const uint32_t lane = b.subgroup_local_id();
    SpirvCompute::put(b.caps, Op_Capability, {Cap_GroupNonUniformQuad});
    const uint32_t helper = b.sel(b.helper_invocation(), b.uconst(1), b.uconst(0));
    uint32_t count = b.uconst(0);
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t other = b.id();
        SpirvCompute::put(b.code, 365, {b.t_u32, other, b.uconst(Scope_Subgroup), helper, b.uconst(i)});
        count = b.ibin(Op_IAdd, count, other);
    }
    const auto encode = [&](uint32_t value) {
        return b.fbin(Op_FMul, b.cvt_u2f(value), b.uconst(fbits(1.0f / 255.0f)));
    };
    b.export_color(0, encode(b.ibin(Op_BitwiseAnd, lane, b.uconst(3))), encode(count),
                      b.uconst(0), b.uconst(fbits(1.0f)));
    return b.finish();
}
} // namespace prosper::test::fragment_vote_execution
