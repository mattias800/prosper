// Paired replay bytes retain the real loaded-pointer equation without borrowing a host address.
#include "gpu/recompiler/rdna2_loaded_scalar_global_resources.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include <cstring>

namespace prosper::gpu {
std::optional<LoadedScalarGlobalResources> owned_loaded_scalar_global_resources(
        const ShaderResourceTable& table, const LoadedScalarGlobalRead& read) {
    if (read.parent_pc >= read.global_pc || (read.window_offset & 3u) ||
        (read.window_bytes != 16u && read.window_bytes != 32u)) return {};
    const auto* parent = owned_nested_snapshot_at(table, read.parent_pc, 8u,
                                                 compiler_resource_has_host_data);
    const auto* target = owned_nested_snapshot_at(table, read.global_pc, read.window_bytes,
                                                 compiler_resource_has_host_data);
    if (!parent || !target) return {};
    const auto* parent_bytes = compiler_resource_data(*parent, 8u);
    if (!parent_bytes || !compiler_resource_data(*target, read.window_bytes)) return {};
    uint64_t pointer = 0;
    std::memcpy(&pointer, parent_bytes, sizeof(pointer));
    if (pointer <= 0x10000u || pointer > UINT64_MAX - read.window_offset ||
        pointer + read.window_offset != target->gpu_addr) return {};
    return LoadedScalarGlobalResources{parent, target};
}
} // namespace prosper::gpu
