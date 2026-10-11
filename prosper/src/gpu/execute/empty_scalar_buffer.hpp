// Exact-PC zero marker from a genuinely folded, fully-OOB empty scalar-buffer use.
#pragma once

#include "gpu/resources/shader_resources.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace prosper::gpu {
struct SrtUse;
// Revalidate the observation against its exact scalar instruction and existing byte bounds.
// Reflection's normalized null resource alone never grants this marker.
std::optional<ShaderResource> empty_scalar_buffer_resource(const SrtUse& use, const uint32_t* code,
                                                           size_t dwords);
} // namespace prosper::gpu
