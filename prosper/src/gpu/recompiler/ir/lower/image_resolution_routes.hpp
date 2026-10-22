// Existing unresolved-image reporting and diagnostic route.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include <cstdint>
#include <functional>
#include <optional>

namespace prosper::gpu {
struct RegState;
struct ShaderResource;
struct ShaderResourceTable;
enum class ImageResourceRequirement : uint32_t;
}   // namespace prosper::gpu

namespace prosper::gpu::ir {
struct ImageResolutionRoutes : LoweringContext {
    const ShaderResource*& res;
    const ShaderResourceTable*& rt;
    const ImageResourceRequirement& image_requirement;
    const bool& storage_only_op;
    bool (*sreg_srt_range_tag)(const RegState&, int, uint32_t, uint32_t&);
    bool (*sreg_range_ud_alias)(const RegState&, int, uint32_t, int&);
    bool (*sreg_range_written)(const RegState&, int, uint32_t);
    std::optional<bool> route();
    void available(const std::function<const char*(const ShaderResource*)>& cls_name);
    void report(bool& first_report);
    std::optional<bool> soften();
};
}  // namespace prosper::gpu::ir
