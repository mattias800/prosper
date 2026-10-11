#pragma once
#include "gpu/diagnostics/refused_graphics_inputs.hpp"
namespace prosper::gpu {
struct ShaderCompileKey;
struct NggSubgroupDrawRequest;
std::vector<uint32_t> observe_graphics_shader_attempt(const ShaderCompileKey&,
                                                      const ShaderResourceTable*, uint64_t address,
                                                      const RefusedGraphicsCompile&);
std::vector<uint32_t> observe_ngg_shader_attempt(const NggSubgroupDrawRequest&,
                                                 const NggSubgroupShellConfig&,
                                                 const RefusedGraphicsCompile&,
                                                 const std::string* refusal);
} // namespace prosper::gpu
