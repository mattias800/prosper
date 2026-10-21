// Existing MIMG BVH route and its borrowed continuation state.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include <cstdint>
#include <functional>
#include <optional>

namespace prosper::gpu {
struct ShaderResource;
struct ShaderResourceTable;
}   // namespace prosper::gpu

namespace prosper::gpu::ir {
struct BvhCaseRoutes : LoweringContext {
    const ShaderResourceTable*& rt;
    std::function<uint32_t(int)> vread;
    std::optional<bool> route();
    bool load();
    bool setup(const uint32_t& node_type, const uint32_t& valid64, const uint32_t& valid128,
               uint32_t (&w)[28], uint32_t (&a)[11], const ShaderResource*& bvh);
    bool triangles(const uint32_t& is_tri1, uint32_t (&w)[28],
                   const std::function<uint32_t(uint32_t, uint32_t)>& fsub, uint32_t (&a)[11],
                   const std::function<void(const uint32_t*, const uint32_t*, uint32_t*)>& cross3,
                   const std::function<uint32_t(const uint32_t*, const uint32_t*)>& dot3,
                   const uint32_t& zero, const uint32_t& one,
                   const std::function<uint32_t(uint32_t, uint32_t)>& fadd,
                   const uint32_t& node_type, const uint32_t& is_box16,
                   const std::function<uint32_t(uint32_t, uint32_t)>& fmul,
                   const ShaderResource*& bvh, const uint32_t& box_valid, const uint32_t& invalid,
                   const uint32_t& tri_valid);
    bool barycentrics(const std::function<uint32_t(uint32_t, uint32_t)>& fsub,
                      uint32_t (&tri_out)[4], const uint32_t& i_num, const uint32_t& j_num,
                      const uint32_t& node_type, uint32_t (&w)[28], uint32_t (&a)[11],
                      const uint32_t& is_box16,
                      const std::function<uint32_t(uint32_t, uint32_t)>& fmul, const uint32_t& zero,
                      const ShaderResource*& bvh, const uint32_t& box_valid,
                      const uint32_t& invalid, const uint32_t& tri_valid);
    bool boxes(uint32_t (&a)[11], uint32_t (&w)[28], const uint32_t& is_box16,
               const std::function<uint32_t(uint32_t, uint32_t)>& fmul,
               const std::function<uint32_t(uint32_t, uint32_t)>& fsub, const uint32_t& zero,
               const ShaderResource*& bvh, const uint32_t& box_valid, const uint32_t& invalid,
               const uint32_t& tri_valid, uint32_t (&tri_out)[4]);
    bool finish(const ShaderResource*& bvh, uint32_t (&box_key)[4], uint32_t (&box_out)[4],
                const uint32_t& tri_valid, uint32_t (&tri_out)[4]);
};
}  // namespace prosper::gpu::ir
