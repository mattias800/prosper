#pragma once

#include "../../src/gpu/diagnostic_selectors.hpp"
#include "../../src/gpu/gpu_capture.hpp"
#include "realized_shader_dump.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace prosper::tools {

enum class ResourceOverrideStage {
    Vertex,
    Pixel,
};

struct ResourceOverrideSelector {
    uint64_t draw_index = 0;
    ResourceOverrideStage stage = ResourceOverrideStage::Vertex;
    uint32_t binding = 0;
};

struct ResourceOverrideTarget {
    size_t item_index = SIZE_MAX;
    size_t resource_index = SIZE_MAX;
    uint64_t gpu_addr = 0;
    uint64_t captured_size = 0;
};

// Retain this object for as long as the replay may execute. The selected copied resource points at
// replacement_bytes; keeping that ownership outside GpuReplayFrame avoids changing capture/runtime
// data structures for a diagnostic-only operation.
struct AppliedResourceOverride {
    ResourceOverrideSelector selector;
    ResourceOverrideTarget target;
    uint64_t original_hash = 0;
    uint64_t replacement_hash = 0;
    std::shared_ptr<std::vector<uint8_t>> replacement_bytes;
};

inline const char* resource_override_stage_name(ResourceOverrideStage stage) {
    return stage == ResourceOverrideStage::Vertex ? "vs" : "ps";
}

inline bool parse_resource_override_selector(std::string_view text,
                                             ResourceOverrideSelector& selector) {
    const size_t first_colon = text.find(':');
    const size_t second_colon = first_colon == std::string_view::npos
        ? std::string_view::npos : text.find(':', first_colon + 1);
    if (first_colon == std::string_view::npos || second_colon == std::string_view::npos ||
        text.find(':', second_colon + 1) != std::string_view::npos)
        return false;

    uint64_t draw_index = 0;
    uint64_t binding = 0;
    if (!gpu::parse_diagnostic_draw_id(text.substr(0, first_colon), draw_index) ||
        !gpu::parse_diagnostic_uint64(text.substr(second_colon + 1), binding) ||
        binding > std::numeric_limits<uint32_t>::max())
        return false;

    const std::string_view stage = text.substr(first_colon + 1,
                                               second_colon - first_colon - 1);
    if (stage != "vs" && stage != "ps") return false;
    selector.draw_index = draw_index;
    selector.stage = stage == "vs" ? ResourceOverrideStage::Vertex
                                    : ResourceOverrideStage::Pixel;
    selector.binding = static_cast<uint32_t>(binding);
    return true;
}

inline bool find_resource_override_target(const gpu::GpuReplayFrame& replay,
                                          const ResourceOverrideSelector& selector,
                                          ResourceOverrideTarget& target,
                                          std::string& error) {
    const size_t item_index = replay_item_index_for_draw(replay, selector.draw_index);
    if (item_index == SIZE_MAX) {
        error = "realized draw " + std::to_string(selector.draw_index) + " not found";
        return false;
    }
    const auto& draw = replay.items[item_index];
    const auto& table = selector.stage == ResourceOverrideStage::Vertex ? draw.vrt : draw.prt;
    if (!table) {
        error = "draw " + std::to_string(selector.draw_index) + " " +
                resource_override_stage_name(selector.stage) + " resource table is absent";
        return false;
    }

    size_t resource_index = SIZE_MAX;
    for (size_t index = 0; index < table->resources.size(); ++index) {
        if (table->resources[index].binding != selector.binding) continue;
        if (resource_index != SIZE_MAX) {
            error = "draw " + std::to_string(selector.draw_index) + " " +
                    resource_override_stage_name(selector.stage) + " binding " +
                    std::to_string(selector.binding) + " is ambiguous";
            return false;
        }
        resource_index = index;
    }
    if (resource_index == SIZE_MAX) {
        error = "draw " + std::to_string(selector.draw_index) + " " +
                resource_override_stage_name(selector.stage) + " binding " +
                std::to_string(selector.binding) + " not found";
        return false;
    }

    const auto& resource = table->resources[resource_index];
    if (!resource.host_data || resource.host_data_size == 0) {
        error = "draw " + std::to_string(selector.draw_index) + " " +
                resource_override_stage_name(selector.stage) + " binding " +
                std::to_string(selector.binding) + " has no captured bytes";
        return false;
    }
    if (resource.host_data_size > std::numeric_limits<size_t>::max()) {
        error = "selected captured resource span is too large for this host";
        return false;
    }

    target.item_index = item_index;
    target.resource_index = resource_index;
    target.gpu_addr = resource.gpu_addr;
    target.captured_size = resource.host_data_size;
    error.clear();
    return true;
}

inline bool read_exact_resource_override_file(const std::string& path, uint64_t expected_size,
                                              std::vector<uint8_t>& bytes,
                                              std::string& error) {
    if (!expected_size || expected_size > std::numeric_limits<size_t>::max() ||
        expected_size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        error = "selected captured resource span cannot be read on this host";
        return false;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        error = "cannot read resource override " + path;
        return false;
    }
    const std::streampos end = file.tellg();
    if (end < 0 || static_cast<uint64_t>(end) != expected_size) {
        error = "resource override " + path + " has " +
                (end < 0 ? std::string("an unreadable size")
                         : std::to_string(static_cast<uint64_t>(end)) + " bytes") +
                "; selected captured span is " + std::to_string(expected_size) + " bytes";
        return false;
    }
    file.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(expected_size));
    if (!file.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
        error = "cannot read complete resource override " + path;
        bytes.clear();
        return false;
    }
    error.clear();
    return true;
}

inline bool apply_resource_override(gpu::GpuReplayFrame& replay,
                                    const ResourceOverrideSelector& selector,
                                    std::vector<uint8_t> replacement,
                                    AppliedResourceOverride& applied,
                                    std::string& error) {
    ResourceOverrideTarget target;
    if (!find_resource_override_target(replay, selector, target, error)) return false;
    if (replacement.size() != target.captured_size) {
        error = "replacement has " + std::to_string(replacement.size()) +
                " bytes; selected captured span is " +
                std::to_string(target.captured_size) + " bytes";
        return false;
    }

    auto& draw = replay.items[target.item_index];
    auto& selected_table = selector.stage == ResourceOverrideStage::Vertex ? draw.vrt : draw.prt;
    const auto& original_resource = selected_table->resources[target.resource_index];

    AppliedResourceOverride result;
    result.selector = selector;
    result.target = target;
    result.original_hash = gpu::gpu_capture_hash(
        original_resource.host_data, static_cast<size_t>(original_resource.host_data_size));
    result.replacement_bytes =
        std::make_shared<std::vector<uint8_t>>(std::move(replacement));
    result.replacement_hash = gpu::gpu_capture_hash(*result.replacement_bytes);

    // Draws can deliberately share a resource-table object, while different table entries can
    // point into one content-deduplicated capture instance. Clone the table and redirect only this
    // copied entry: mutating either original object would contaminate sibling draws/resources.
    auto copied_table = std::make_shared<gpu::ShaderResourceTable>(*selected_table);
    copied_table->resources[target.resource_index].host_data = result.replacement_bytes->data();
    selected_table = std::move(copied_table);
    applied = std::move(result);
    error.clear();
    return true;
}

inline bool apply_resource_override_file(gpu::GpuReplayFrame& replay,
                                         const ResourceOverrideSelector& selector,
                                         const std::string& path,
                                         AppliedResourceOverride& applied,
                                         std::string& error) {
    ResourceOverrideTarget target;
    if (!find_resource_override_target(replay, selector, target, error)) return false;
    std::vector<uint8_t> replacement;
    if (!read_exact_resource_override_file(path, target.captured_size, replacement, error))
        return false;
    return apply_resource_override(replay, selector, std::move(replacement), applied, error);
}

} // namespace prosper::tools
