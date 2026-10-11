#pragma once
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <memory>
namespace prosper::gpu {
enum class ShaderProgramStage : uint8_t;
struct ShaderCompileKey;
struct FragmentCompileCase;
std::vector<uint32_t> compile_graphics_shader(
    ShaderProgramStage, const ShaderCompileKey &, const ShaderResourceTable *,
    uint64_t program_address, FragmentArithmeticObservation *,
    std::shared_ptr<const FragmentCompileCase> *producer = nullptr);
} // namespace prosper::gpu
