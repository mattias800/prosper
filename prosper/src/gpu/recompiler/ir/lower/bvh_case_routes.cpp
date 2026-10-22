// Mechanical relocation of the existing MIMG BVH lowering.
#include "gpu/recompiler/ir/lower/bvh_case_routes.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <bit>
#include <cstdint>
#include <functional>
#include <optional>

namespace prosper::gpu::ir {
std::optional<bool> BvhCaseRoutes::route() {
    if (in.opcode == 0xe6u) { return load(); }
    return std::nullopt;
}

bool BvhCaseRoutes::load() {
    const ShaderResource* bvh = rt->by_fetch_pc(in.pc);
    if (in.words[0] != 0xf1989f07u || in.len_dwords != 5u || in.mimg_dmask != 0xfu ||
        !in.mimg_unorm || in.mimg_dim != 0u || in.mimg_glc || in.src[2].value != 0 ||
        (in.words[4] & 0xffff0000u) != 0u || !bvh || bvh->cls != ResourceClass::ConstantBuffer ||
        bvh->format != DataFormat::Uint32 || bvh->num_components != 1u || bvh->size < 64u ||
        (bvh->size & 3u) != 0u) {
        ok = false;
        return true;
    }

    // The front-half only creates this marker when one mapped zero qword load taints all
    // four descriptor words and the ray instruction is dominated by an EXEC-narrowing
    // guard with no external region entry. This is an explicit empty acceleration
    // structure for this dispatch, not a fallback for an unresolved descriptor.
    if (is_proven_null_bvh(*bvh)) {
        for (uint32_t k = 0; k < 4; ++k) {
            const int vd = in.dst.value + static_cast<int>(k);
            const uint32_t old = vreg_old(b, rs, vd);
            rs.vreg[vd] = b.uconst(0xffffffffu);
            predicate_write(b, rs, vd, old);
        }
        return true;
    }

    auto addr_vgpr = [&](uint32_t k) { return mimg_address_vgpr(rs, in, k); };
    uint32_t a[11]{};
    for (uint32_t k = 0; k < 11; ++k) a[k] = vread(addr_vgpr(k));

    const uint32_t node = a[0];
    const uint32_t node_type = b.ibin(Op_BitwiseAnd, node, b.uconst(7u));
    const uint32_t node_offset = b.ibin(Op_BitwiseAnd, node, b.uconst(~7u));
    const uint32_t word_base = b.ibin(Op_ShiftLeftLogical, node_offset, b.uconst(1u));
    const uint32_t dword_count = bvh->size / 4u;
    const uint32_t valid64 =
        b.ucmp(Op_ULessThanEqual, node_offset, b.uconst((bvh->size - 64u) / 8u));
    const uint32_t valid128 = bvh->size < 128u ? b.bfalse()
                                               : b.ucmp(Op_ULessThanEqual, node_offset,
                                                        b.uconst((bvh->size - 128u) / 8u));

    // Every speculative load is independently clamped to dword zero. Results from a
    // malformed/out-of-range pointer are discarded below, but the clamp also prevents
    // an invalid guest pointer from becoming an out-of-bounds Vulkan SSBO access.
    auto load_node = [&](uint32_t rel) {
        const uint32_t idx = rel ? b.ibin(Op_IAdd, word_base, b.uconst(rel)) : word_base;
        const uint32_t safe_idx =
            b.sel(b.ucmp(Op_ULessThan, idx, b.uconst(dword_count)), idx, b.uconst(0u));
        return b.cbuf_load(safe_idx, bvh->binding);
    };
    uint32_t w[28]{};
    for (uint32_t k = 0; k < 28; ++k) w[k] = load_node(k);

    return setup(node_type, valid64, valid128, w, a, bvh);
}

bool BvhCaseRoutes::setup(const uint32_t& node_type, const uint32_t& valid64,
                          const uint32_t& valid128, uint32_t (&w)[28], uint32_t (&a)[11],
                          const ShaderResource*& bvh) {
    const uint32_t zero = b.uconst(0u);
    const uint32_t one = b.uconst(fbits(1.0f));
    const uint32_t invalid = b.uconst(0xffffffffu);
    const uint32_t is_tri0 = b.ucmp(Op_IEqual, node_type, b.uconst(0u));
    const uint32_t is_tri1 = b.ucmp(Op_IEqual, node_type, b.uconst(1u));
    const uint32_t is_box16 = b.ucmp(Op_IEqual, node_type, b.uconst(4u));
    const uint32_t is_box32 = b.ucmp(Op_IEqual, node_type, b.uconst(5u));
    const uint32_t tri_valid = b.land(b.lor(is_tri0, is_tri1), valid64);
    const uint32_t box_valid = b.lor(b.land(is_box16, valid64), b.land(is_box32, valid128));

    auto fadd = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FAdd, x, y); };
    auto fsub = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FSub, x, y); };
    auto fmul = [&](uint32_t x, uint32_t y) { return b.fbin(Op_FMul, x, y); };
    auto dot3 = [&](const uint32_t x[3], const uint32_t y[3]) {
        return fadd(fadd(fmul(x[0], y[0]), fmul(x[1], y[1])), fmul(x[2], y[2]));
    };
    auto cross3 = [&](const uint32_t x[3], const uint32_t y[3], uint32_t out[3]) {
        out[0] = fsub(fmul(x[1], y[2]), fmul(x[2], y[1]));
        out[1] = fsub(fmul(x[2], y[0]), fmul(x[0], y[2]));
        out[2] = fsub(fmul(x[0], y[1]), fmul(x[1], y[0]));
    };

    return triangles(is_tri1, w, std::cref(fsub), a, std::cref(cross3), std::cref(dot3), zero, one,
                     std::cref(fadd), node_type, is_box16, std::cref(fmul), bvh, box_valid, invalid,
                     tri_valid);
}

bool BvhCaseRoutes::triangles(
    const uint32_t& is_tri1, uint32_t (&w)[28],
    const std::function<uint32_t(uint32_t, uint32_t)>& fsub, uint32_t (&a)[11],
    const std::function<void(const uint32_t*, const uint32_t*, uint32_t*)>& cross3,
    const std::function<uint32_t(const uint32_t*, const uint32_t*)>& dot3, const uint32_t& zero,
    const uint32_t& one, const std::function<uint32_t(uint32_t, uint32_t)>& fadd,
    const uint32_t& node_type, const uint32_t& is_box16,
    const std::function<uint32_t(uint32_t, uint32_t)>& fmul, const ShaderResource*& bvh,
    const uint32_t& box_valid, const uint32_t& invalid, const uint32_t& tri_valid) {
    // Triangle node types 0 and 1 share a 64-byte quad. Type 0 selects V0,V1,V2;
    // type 1 selects V1,V3,V2. The four hardware return values are the unnormalised
    // t numerator, determinant, I numerator and J numerator (mode 1).
    uint32_t tv0[3] = {
        b.sel(is_tri1, w[3], w[0]),
        b.sel(is_tri1, w[4], w[1]),
        b.sel(is_tri1, w[5], w[2]),
    };
    uint32_t tv1[3] = {
        b.sel(is_tri1, w[9], w[3]),
        b.sel(is_tri1, w[10], w[4]),
        b.sel(is_tri1, w[11], w[5]),
    };
    uint32_t tv2[3] = {w[6], w[7], w[8]};
    uint32_t edge1[3], edge2[3], from_v0[3];
    for (uint32_t k = 0; k < 3; ++k) {
        edge1[k] = fsub(tv1[k], tv0[k]);
        edge2[k] = fsub(tv2[k], tv0[k]);
        from_v0[k] = fsub(a[2u + k], tv0[k]);
    }
    uint32_t ray_dir[3] = {a[5], a[6], a[7]};
    uint32_t s1[3], s2[3];
    cross3(ray_dir, edge2, s1);
    cross3(from_v0, edge1, s2);
    const uint32_t t_num = dot3(edge2, s2);
    const uint32_t t_denom = dot3(s1, edge1);
    const uint32_t i_num = dot3(from_v0, s1);
    const uint32_t j_num = dot3(ray_dir, s2);
    const uint32_t t = b.fbin(Op_FDiv, t_num, t_denom);
    const uint32_t bary_i = b.fbin(Op_FDiv, i_num, t_denom);
    const uint32_t bary_j = b.fbin(Op_FDiv, j_num, t_denom);
    uint32_t tri_miss =
        b.lor(b.fcmp(Op_FOrdLessThan, bary_i, zero), b.fcmp(Op_FOrdGreaterThan, bary_i, one));
    tri_miss = b.lor(tri_miss, b.fcmp(Op_FOrdLessThan, bary_j, zero));
    tri_miss = b.lor(tri_miss, b.fcmp(Op_FOrdGreaterThan, fadd(bary_i, bary_j), one));
    tri_miss = b.lor(tri_miss, b.fcmp(Op_FOrdLessThan, t, zero));
    uint32_t tri_out[4] = {
        b.sel(tri_miss, b.uconst(0x7f800000u), t_num),
        b.sel(tri_miss, one, t_denom),
        i_num,
        j_num,
    };

    return barycentrics(std::cref(fsub), tri_out, i_num, j_num, node_type, w, a, is_box16,
                        std::cref(fmul), zero, bvh, box_valid, invalid, tri_valid);
}

bool BvhCaseRoutes::barycentrics(const std::function<uint32_t(uint32_t, uint32_t)>& fsub,
                                 uint32_t (&tri_out)[4], const uint32_t& i_num,
                                 const uint32_t& j_num, const uint32_t& node_type,
                                 uint32_t (&w)[28], uint32_t (&a)[11], const uint32_t& is_box16,
                                 const std::function<uint32_t(uint32_t, uint32_t)>& fmul,
                                 const uint32_t& zero, const ShaderResource*& bvh,
                                 const uint32_t& box_valid, const uint32_t& invalid,
                                 const uint32_t& tri_valid) {
    // Undo the pair-compression vertex rotation encoded in the final triangle-node
    // dword. The two 2-bit fields select from {1-I-J, I, J}; valid RTIP 1.1 nodes never
    // use selector 3, whose conservative fallback here is 1-I-J.
    const uint32_t bary0_num = fsub(fsub(tri_out[1], i_num), j_num);
    const uint32_t tri_shift =
        b.ibin(Op_ShiftLeftLogical, b.ibin(Op_BitwiseAnd, node_type, b.uconst(3u)), b.uconst(3u));
    const uint32_t i_selector =
        b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, w[15], tri_shift), b.uconst(3u));
    const uint32_t j_selector =
        b.ibin(Op_BitwiseAnd,
               b.ibin(Op_ShiftRightLogical, w[15], b.ibin(Op_IAdd, tri_shift, b.uconst(2u))),
               b.uconst(3u));
    auto swizzled_bary = [&](uint32_t selector) {
        return b.sel(b.ucmp(Op_IEqual, selector, b.uconst(1u)), i_num,
                     b.sel(b.ucmp(Op_IEqual, selector, b.uconst(2u)), j_num, bary0_num));
    };
    tri_out[2] = swizzled_bary(i_selector);
    tri_out[3] = swizzled_bary(j_selector);

    return boxes(a, w, is_box16, std::cref(fmul), std::cref(fsub), zero, bvh, box_valid, invalid,
                 tri_valid, tri_out);
}

bool BvhCaseRoutes::boxes(uint32_t (&a)[11], uint32_t (&w)[28], const uint32_t& is_box16,
                          const std::function<uint32_t(uint32_t, uint32_t)>& fmul,
                          const std::function<uint32_t(uint32_t, uint32_t)>& fsub,
                          const uint32_t& zero, const ShaderResource*& bvh,
                          const uint32_t& box_valid, const uint32_t& invalid,
                          const uint32_t& tri_valid, uint32_t (&tri_out)[4]) {
    // Convert each FP16 box payload to the same six-float representation as an FP32
    // box, then apply the slab test to all four children. BOX_SORT_EN orders valid hits
    // by increasing near intersection time; misses receive +INF and therefore remain at
    // the end. Strict compare-swaps retain physical child order for equal-time hits.
    uint32_t box_out[4]{};
    uint32_t box_key[4]{};
    uint32_t ray_inv[3] = {a[8], a[9], a[10]};
    for (uint32_t child = 0; child < 4; ++child) {
        const uint32_t hbase = 4u + child * 3u;
        uint32_t fp16_bounds[6] = {
            b.unpack_half(w[hbase], 0u),      b.unpack_half(w[hbase], 1u),
            b.unpack_half(w[hbase + 1u], 0u), b.unpack_half(w[hbase + 1u], 1u),
            b.unpack_half(w[hbase + 2u], 0u), b.unpack_half(w[hbase + 2u], 1u),
        };
        const uint32_t fbase = 4u + child * 6u;
        uint32_t bmin[3], bmax[3];
        for (uint32_t axis = 0; axis < 3; ++axis) {
            bmin[axis] = b.sel(is_box16, fp16_bounds[axis], w[fbase + axis]);
            bmax[axis] = b.sel(is_box16, fp16_bounds[3u + axis], w[fbase + 3u + axis]);
        }
        uint32_t near_axis[3], far_axis[3];
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const uint32_t t0 = fmul(fsub(bmin[axis], a[2u + axis]), ray_inv[axis]);
            const uint32_t t1 = fmul(fsub(bmax[axis], a[2u + axis]), ray_inv[axis]);
            const uint32_t positive = b.fcmp(Op_FOrdGreaterThanEqual, ray_inv[axis], zero);
            near_axis[axis] = b.sel(positive, t0, t1);
            far_axis[axis] = b.sel(positive, t1, t0);
        }
        const uint32_t near_unclamped =
            b.fext2(Glsl_FMax, b.fext2(Glsl_FMax, near_axis[0], near_axis[1]), near_axis[2]);
        const uint32_t far_unclamped =
            b.fext2(Glsl_FMin, b.fext2(Glsl_FMin, far_axis[0], far_axis[1]), far_axis[2]);
        const uint32_t near_t = b.fext2(Glsl_FMax, near_unclamped, zero);
        const uint32_t far_t = b.fext2(Glsl_FMin, far_unclamped, a[1]);
        const uint32_t nan_interval =
            b.lor(b.fcmp(Op_FUnordNotEqual, near_unclamped, near_unclamped),
                  b.fcmp(Op_FUnordNotEqual, far_unclamped, far_unclamped));
        const float box_multiplier =
            1.0f + static_cast<float>(bvh->bvh_box_grow) * (1.0f / 16777216.0f);
        const uint32_t hit =
            b.land(b.logical_not(nan_interval),
                   b.fcmp(Op_FOrdLessThanEqual, near_t,
                          fmul(far_t, b.uconst(std::bit_cast<uint32_t>(box_multiplier)))));
        const uint32_t valid_hit = b.land(box_valid, hit);
        box_out[child] = b.sel(valid_hit, w[child], invalid);
        box_key[child] = b.sel(valid_hit, near_t, b.uconst(0x7f800000u));
    }
    return finish(bvh, box_key, box_out, tri_valid, tri_out);
}

bool BvhCaseRoutes::finish(const ShaderResource*& bvh, uint32_t (&box_key)[4],
                           uint32_t (&box_out)[4], const uint32_t& tri_valid,
                           uint32_t (&tri_out)[4]) {
    if (bvh->bvh_sort_enabled) {
        auto compare_swap = [&](uint32_t left, uint32_t right) {
            const uint32_t left_key = box_key[left];
            const uint32_t right_key = box_key[right];
            const uint32_t left_out = box_out[left];
            const uint32_t right_out = box_out[right];
            const uint32_t swap = b.fcmp(Op_FOrdGreaterThan, left_key, right_key);
            box_key[left] = b.sel(swap, right_key, left_key);
            box_key[right] = b.sel(swap, left_key, right_key);
            box_out[left] = b.sel(swap, right_out, left_out);
            box_out[right] = b.sel(swap, left_out, right_out);
        };
        compare_swap(0, 1);
        compare_swap(2, 3);
        compare_swap(0, 2);
        compare_swap(1, 3);
        compare_swap(1, 2);
    }

    for (uint32_t k = 0; k < 4; ++k) {
        const uint32_t result = b.sel(tri_valid, tri_out[k], box_out[k]);
        const int vd = in.dst.value + static_cast<int>(k);
        const uint32_t old = vreg_old(b, rs, vd);
        rs.vreg[vd] = result;
        predicate_write(b, rs, vd, old);
    }
    return true;
}

}  // namespace prosper::gpu::ir
