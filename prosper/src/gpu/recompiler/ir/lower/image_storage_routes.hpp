// The storage-image arm relocated from emit_alu; coordinate/result proof callbacks stay there.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <functional>
namespace prosper::gpu::ir {
struct ImageStorageRoutes : LoweringContext {
    const ShaderResource* res;
    bool uint_texture, d16_float_resource;
    uint32_t SQ_DIM_2D, SQ_DIM_3D, SQ_DIM_2D_ARRAY;
    std::function<uint32_t(int)> vread;
    std::function<bool(const uint32_t*)> write_mimg_results;
    bool lower();
};
} // namespace prosper::gpu::ir
