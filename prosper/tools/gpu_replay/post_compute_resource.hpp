#pragma once

#include "gpu/gpu_capture.hpp"
#include "gpu/shader_resources.hpp"
#include "gpu/tile.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace prosper::tools {

struct PostComputeR11Stats {
    uint64_t texels = 0;
    uint64_t channels = 0;
    uint64_t finite = 0;
    uint64_t zero = 0;
    uint64_t above_one = 0;
    uint64_t positive_infinity = 0;
    uint64_t nan = 0;
    double finite_min = 0.0;
    double finite_max = 0.0;
    double finite_mean = 0.0;
};

struct PostComputeResourceSnapshot {
    std::vector<uint8_t> raw;
    std::vector<uint8_t> linear;
    uint64_t raw_hash = 0;
    uint64_t linear_hash = 0;
    bool has_r11_stats = false;
    PostComputeR11Stats r11;
};

struct PostComputeChangeEvidence {
    bool prefix_changed = false;
    bool selected_changed = false;
};

inline PostComputeChangeEvidence post_compute_change_evidence(
    const PostComputeResourceSnapshot& captured_seed,
    const PostComputeResourceSnapshot& selected_before,
    const PostComputeResourceSnapshot& selected_after) {
    return {
        captured_seed.linear_hash != selected_before.linear_hash,
        selected_before.linear_hash != selected_after.linear_hash,
    };
}

inline uint32_t post_compute_bytes_per_texel(const gpu::ShaderResource& resource) {
    using gpu::DataFormat;
    switch (resource.format) {
        case DataFormat::Float10_11_11:
        case DataFormat::Unorm2_10_10_10:
        case DataFormat::Snorm2_10_10_10:
        case DataFormat::Uint2_10_10_10:
        case DataFormat::Sint2_10_10_10:
            return 4;
        default:
            break;
    }
    const uint32_t component_bytes = gpu::data_format_bytes(resource.format);
    const uint32_t components = resource.num_components ? resource.num_components : 1u;
    if (!component_bytes || component_bytes > UINT32_MAX / components) return 0;
    return component_bytes * components;
}

inline bool linearize_post_compute_resource(const gpu::ShaderResource& resource,
                                            const std::vector<uint8_t>& raw,
                                            std::vector<uint8_t>& linear,
                                            std::string& error) {
    const uint32_t width = resource.width;
    const uint32_t height = resource.height;
    const uint32_t depth = resource.depth ? resource.depth : 1u;
    const uint32_t bytes_per_texel = post_compute_bytes_per_texel(resource);
    if (!width || !height || !bytes_per_texel) {
        error = "selected storage image has no supported texel layout";
        return false;
    }
    const uint64_t texels = static_cast<uint64_t>(width) * height * depth;
    if (texels > SIZE_MAX / bytes_per_texel) {
        error = "selected storage image linear size overflows";
        return false;
    }
    const size_t linear_bytes = static_cast<size_t>(texels) * bytes_per_texel;
    linear.assign(linear_bytes, 0);

    if (resource.tile_mode && depth > 1) {
        if (resource.img_dim != 2 || !gpu::tile_mode_supports_volume(resource.tile_mode)) {
            error = "selected tiled layered/volume storage layout is unsupported";
            return false;
        }
        if (!gpu::detile_volume(linear.data(), raw.data(), raw.size(), width, height, depth,
                                resource.tile_mode, bytes_per_texel)) {
            error = "selected storage volume could not be detiled";
            return false;
        }
        return true;
    }
    if (resource.tile_mode) {
        const size_t required = gpu::tiled_surface_bytes(
            width, height, resource.tile_mode, 0, bytes_per_texel);
        if (!required || raw.size() < required) {
            error = "selected storage surface has incomplete tiled backing";
            return false;
        }
        gpu::detile_surface(linear.data(), raw.data(), width, height, resource.tile_mode, 0,
                            bytes_per_texel);
        return true;
    }

    const size_t tight_row = static_cast<size_t>(width) * bytes_per_texel;
    const size_t row_pitch = resource.linear_row_pitch_bytes
        ? resource.linear_row_pitch_bytes : tight_row;
    if (row_pitch < tight_row || height > SIZE_MAX / row_pitch ||
        depth > SIZE_MAX / (row_pitch * height) ||
        raw.size() < row_pitch * height * depth) {
        error = "selected storage image has incomplete linear backing";
        return false;
    }
    for (uint32_t z = 0; z < depth; ++z)
        for (uint32_t y = 0; y < height; ++y)
            std::memcpy(linear.data() +
                            (static_cast<size_t>(z) * height + y) * tight_row,
                        raw.data() +
                            (static_cast<size_t>(z) * height + y) * row_pitch,
                        tight_row);
    return true;
}

inline PostComputeR11Stats analyze_post_compute_r11(const std::vector<uint8_t>& linear) {
    PostComputeR11Stats stats;
    stats.texels = linear.size() / sizeof(uint32_t);
    stats.channels = stats.texels * 3;
    double finite_sum = 0.0;
    double finite_min = std::numeric_limits<double>::infinity();
    double finite_max = -std::numeric_limits<double>::infinity();
    for (size_t texel = 0; texel < stats.texels; ++texel) {
        uint32_t packed = 0;
        std::memcpy(&packed, linear.data() + texel * sizeof(packed), sizeof(packed));
        const float values[3] = {
            gpu::f11_to_float(static_cast<uint16_t>(packed)),
            gpu::f11_to_float(static_cast<uint16_t>(packed >> 11)),
            gpu::f10_to_float(static_cast<uint16_t>(packed >> 22)),
        };
        for (float value : values) {
            if (std::isnan(value)) {
                ++stats.nan;
            } else if (std::isinf(value)) {
                ++stats.positive_infinity;
            } else {
                ++stats.finite;
                stats.zero += value == 0.0f;
                stats.above_one += value > 1.0f;
                finite_min = std::min(finite_min, static_cast<double>(value));
                finite_max = std::max(finite_max, static_cast<double>(value));
                finite_sum += value;
            }
        }
    }
    if (stats.finite) {
        stats.finite_min = finite_min;
        stats.finite_max = finite_max;
        stats.finite_mean = finite_sum / static_cast<double>(stats.finite);
    }
    return stats;
}

inline bool snapshot_post_compute_resource(const gpu::ShaderResourceTable* table,
                                           uint32_t binding,
                                           PostComputeResourceSnapshot& snapshot,
                                           std::string& error) {
    snapshot = {};
    if (!table) {
        error = "selected compute has no resource table";
        return false;
    }
    const gpu::ShaderResource* found = nullptr;
    for (const auto& resource : table->resources) {
        if (resource.binding != binding) continue;
        if (found) {
            error = "selected compute binding is duplicated";
            return false;
        }
        found = &resource;
    }
    if (!found || found->cls != gpu::ResourceClass::StorageImage ||
        !found->host_data || !found->host_data_size ||
        found->host_data_size > SIZE_MAX) {
        error = "selected binding is not a captured storage image";
        return false;
    }
    snapshot.raw.assign(found->host_data,
                        found->host_data + static_cast<size_t>(found->host_data_size));
    snapshot.raw_hash = gpu::gpu_capture_hash(snapshot.raw);
    if (!linearize_post_compute_resource(*found, snapshot.raw, snapshot.linear, error))
        return false;
    snapshot.linear_hash = gpu::gpu_capture_hash(snapshot.linear);
    if (found->format == gpu::DataFormat::Float10_11_11 &&
        (found->num_components ? found->num_components : 1u) == 3u) {
        snapshot.has_r11_stats = true;
        snapshot.r11 = analyze_post_compute_r11(snapshot.linear);
    }
    return true;
}

} // namespace prosper::tools
