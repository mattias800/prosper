#include "gpu_dependency_graph.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace prosper::gpu {
namespace {

struct Writer {
    uint32_t operation = 0;
    uint64_t addr = 0;
    uint64_t size = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    bool image = false;
};

uint64_t span_end(uint64_t addr, uint64_t size) {
    return size > std::numeric_limits<uint64_t>::max() - addr
        ? std::numeric_limits<uint64_t>::max() : addr + size;
}

bool overlaps(uint64_t a_addr, uint64_t a_size, uint64_t b_addr, uint64_t b_size) {
    return a_addr < span_end(b_addr, b_size) && b_addr < span_end(a_addr, a_size);
}

bool writer_matches(const GpuDependencyAccess& access, const Writer& writer) {
    const bool image_access = access.width && access.height &&
        (access.resource_class == ResourceClass::Texture ||
         access.resource_class == ResourceClass::StorageImage);
    // The live RTT cache and replay seed contract identify images by their programmed base. An
    // interior byte overlap with another image allocation cannot be imported as that texture and
    // must not be advertised as a closed temporal dependency.
    if (image_access && writer.image) return access.addr == writer.addr;
    return overlaps(access.addr, access.size, writer.addr, writer.size);
}

uint64_t resource_size(const ShaderResource& resource) {
    if (resource.host_data_size) return resource.host_data_size;
    if (resource.size) return resource.size;
    if (resource.width && resource.height)
        return static_cast<uint64_t>(resource.width) * resource.height * 4;
    return 0;
}

void append_accesses(const ShaderResourceTable* table, const char* stage,
                     std::vector<GpuDependencyAccess>& accesses) {
    if (!table) return;
    for (const auto& resource : table->resources) {
        const uint64_t size = resource_size(resource);
        if (!resource.gpu_addr || !size || resource.cls == ResourceClass::Sampler) continue;
        accesses.push_back({resource.gpu_addr, size, resource.width, resource.height,
                            resource.binding, resource.cls, stage, resource.format,
                            resource.num_components, resource.srgb});
    }
}

void append_compute_accesses(const ComputeItem& compute,
                             std::vector<GpuDependencyAccess>& reads,
                             std::vector<Writer>& writes) {
    const ShaderResourceTable* table = compute.resources.get();
    if (!table) return;
    const DescriptorValidationReport reflected = validate_spirv_descriptor_interface(
        compute.spirv, table, 0, SpirvShaderStage::Compute, false);
    if (!compute.spirv.empty() && reflected.ok()) {
        for (const auto& descriptor : reflected.descriptors) {
            const ShaderResource* resource = table->by_binding(descriptor.binding);
            if (!resource) continue;  // reflected.ok() normally makes this unreachable
            const uint64_t size = resource_size(*resource);
            if (!resource->gpu_addr || !size) continue;
            if (descriptor.readable)
                reads.push_back({resource->gpu_addr, size, resource->width, resource->height,
                                 resource->binding, resource->cls, "cs", resource->format,
                                 resource->num_components, resource->srgb});
            if (descriptor.writable)
                writes.push_back({0, resource->gpu_addr, size, resource->width,
                                  resource->height,
                                  resource->cls == ResourceClass::StorageImage});
        }
        return;
    }

    // Hand-built fixtures and legacy diagnostic captures may not carry a reflectable module.
    // Preserve their historical conservative graph instead of hiding every resource.
    append_accesses(table, "cs", reads);
    for (const auto& access : reads)
        writes.push_back({0, access.addr, access.size, access.width, access.height,
                          access.resource_class == ResourceClass::StorageImage});
}

} // namespace

bool gpu_dependency_rtt_seed_matches(const GpuDependencyAccess& access,
                                     const GpuCaptureRttSeed& seed) {
    if (!access.addr || !access.width || !access.height || access.srgb ||
        seed.guest_addr != access.addr || seed.width != access.width ||
        seed.height != access.height)
        return false;
    switch (seed.format) {
        case GpuCaptureColorFormat::Rgba8Unorm:
            return access.format == DataFormat::Unorm8 && access.num_components == 4;
        case GpuCaptureColorFormat::Rgba16Float:
            return access.format == DataFormat::Float16 && access.num_components == 4;
        case GpuCaptureColorFormat::R11G11B10Float:
            return access.format == DataFormat::Float10_11_11 && access.num_components == 3;
        case GpuCaptureColorFormat::R8Unorm:
            return access.format == DataFormat::Unorm8 && access.num_components == 1;
        case GpuCaptureColorFormat::R32Uint:
            return access.format == DataFormat::Uint32 && access.num_components == 1;
    }
    return false;
}

bool build_gpu_dependency_graph(const GpuReplayFrame& replay,
                                GpuDependencyGraph& graph, std::string& error) {
    graph = {};
    error.clear();
    std::unordered_map<uint64_t, const DrawItem*> draws;
    std::unordered_map<uint64_t, const ComputeItem*> computes;
    for (const auto& draw : replay.items) draws.emplace(draw.draw_index, &draw);
    for (const auto& compute : replay.computes) computes.emplace(compute.dispatch_index, &compute);

    std::vector<Writer> writers;
    graph.nodes.reserve(replay.operations.size());
    for (size_t operation_index = 0; operation_index < replay.operations.size(); ++operation_index) {
        const auto& operation = replay.operations[operation_index];
        graph.nodes.push_back({static_cast<uint32_t>(operation_index), operation.kind,
                               operation.source_index, operation.command_order, operation.realized});
        if (!operation.realized) continue;

        std::vector<GpuDependencyAccess> reads;
        std::vector<Writer> writes;
        if (operation.kind == SubmitOperationKind::Draw) {
            auto it = draws.find(operation.source_index);
            if (it == draws.end()) {
                error = "realized draw operation has no materialized item";
                return false;
            }
            const DrawItem& draw = *it->second;
            append_accesses(draw.vrt.get(), "vs", reads);
            append_accesses(draw.prt.get(), "ps", reads);
            auto append_target = [&](uint64_t base, uint32_t width, uint32_t height,
                                     uint32_t write_mask) {
                if (!write_mask || !base || !width || !height ||
                    std::any_of(writes.begin(), writes.end(),
                        [&](const auto& write) { return write.addr == base; })) return;
                writes.push_back({0, base, static_cast<uint64_t>(width) * height * 4,
                                  width, height, true});
            };
            for (size_t slot = 0; slot < draw.color_targets.size(); ++slot) {
                const auto& target = draw.color_targets[slot];
                append_target(target.base, target.width, target.height,
                              draw.ps.color_targets[slot].write_mask);
            }
            // Direct graph callers and captures through v33 may populate only the named aliases.
            append_target(draw.color0_base, draw.color0_width, draw.color0_height,
                          draw.ps.color_write_mask);
            append_target(draw.color1_base, draw.color1_width, draw.color1_height,
                          draw.ps.color1_write_mask);
        } else if (operation.kind == SubmitOperationKind::Dispatch) {
            auto it = computes.find(operation.source_index);
            if (it == computes.end()) {
                error = "realized dispatch operation has no materialized item";
                return false;
            }
            append_compute_accesses(*it->second, reads, writes);
        } else {
            if (operation.source_index >= replay.dma_copies.size()) {
                error = "realized DMA operation has no materialized copy";
                return false;
            }
            const ReplayDmaCopy& copy = replay.dma_copies[operation.source_index];
            reads.push_back({copy.src, copy.bytes, 0, 0, 0,
                             ResourceClass::ConstantBuffer, "dma-src"});
            writes.push_back({0, copy.dst, copy.bytes, 0, 0, false});
        }

        for (const auto& access : reads) {
            auto producer = std::find_if(writers.rbegin(), writers.rend(), [&](const Writer& writer) {
                return writer_matches(access, writer);
            });
            if (producer == writers.rend()) {
                auto leaf = std::find_if(graph.external_leaves.begin(), graph.external_leaves.end(),
                    [&](const GpuDependencyLeaf& candidate) {
                        return candidate.access.addr == access.addr &&
                               candidate.access.size == access.size &&
                               candidate.access.width == access.width &&
                               candidate.access.height == access.height &&
                               candidate.access.resource_class == access.resource_class &&
                               candidate.access.format == access.format &&
                               candidate.access.num_components == access.num_components &&
                               candidate.access.srgb == access.srgb;
                    });
                if (leaf == graph.external_leaves.end())
                    graph.external_leaves.push_back({access, {static_cast<uint32_t>(operation_index)}, UINT32_MAX});
                else if (leaf->consumer_operations.empty() ||
                         leaf->consumer_operations.back() != operation_index)
                    leaf->consumer_operations.push_back(static_cast<uint32_t>(operation_index));
            } else
                graph.edges.push_back({producer->operation, static_cast<uint32_t>(operation_index), access});
        }
        for (auto write : writes)
            if (write.addr && write.size) {
                write.operation = static_cast<uint32_t>(operation_index);
                writers.push_back(write);
            }
    }
    for (auto& leaf : graph.external_leaves) {
        const uint32_t first_consumer = leaf.consumer_operations.front();
        auto future = std::find_if(writers.begin(), writers.end(), [&](const Writer& writer) {
            return writer.operation >= first_consumer &&
                   writer_matches(leaf.access, writer);
        });
        if (future != writers.end()) leaf.first_future_writer = future->operation;
    }
    return true;
}

} // namespace prosper::gpu
