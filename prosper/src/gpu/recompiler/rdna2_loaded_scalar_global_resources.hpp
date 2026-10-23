#pragma once

#include "gpu/recompiler/rdna2_loaded_scalar_global.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <optional>

namespace prosper::gpu {
struct LoadedScalarGlobalResources {
    const ShaderResource* parent = nullptr;
    const ShaderResource* target = nullptr;
};

// Compilation and warm-cache admission consume the same complete exact-PC pair. The marker
// alone is not code/epoch authority; the caller must supply the immutable original-code proof.
std::optional<LoadedScalarGlobalResources> owned_loaded_scalar_global_resources(
    const ShaderResourceTable&, const LoadedScalarGlobalRead&);
} // namespace prosper::gpu
