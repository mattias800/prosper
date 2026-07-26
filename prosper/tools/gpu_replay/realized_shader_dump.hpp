#pragma once

#include "../../src/gpu/gpu_capture.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace prosper::tools {

struct RealizedShaderSelector {
    uint64_t draw_index = 0;
    bool vertex = false;
};

struct RecompiledShaderView {
    const std::vector<uint32_t>* words = nullptr;
    bool shared = false;
};

inline RecompiledShaderView recompiled_shader_view(const gpu::DrawItem& draw,
                                                    bool vertex) {
    return vertex ? RecompiledShaderView{&draw.vs_words(), static_cast<bool>(draw.vs_shared)}
                  : RecompiledShaderView{&draw.fs_words(), static_cast<bool>(draw.fs_shared)};
}

inline size_t replay_item_index_for_draw(const gpu::GpuReplayFrame& replay,
                                         uint64_t draw_index) {
    for (size_t item_index = 0; item_index < replay.items.size(); ++item_index)
        if (replay.items[item_index].draw_index == draw_index) return item_index;
    return SIZE_MAX;
}

inline size_t replay_operation_index_for_draw(const gpu::GpuReplayFrame& replay,
                                              uint64_t draw_index) {
    for (size_t operation_index = 0; operation_index < replay.operations.size();
         ++operation_index) {
        const auto& operation = replay.operations[operation_index];
        if (operation.kind == gpu::SubmitOperationKind::Draw && operation.realized &&
            operation.source_index == draw_index)
            return operation_index;
    }
    return SIZE_MAX;
}

inline bool parse_realized_shader_selector(const std::string& spec,
                                           RealizedShaderSelector& selector) {
    const size_t colon = spec.find(':');
    if (colon == std::string::npos || colon == 0 ||
        spec.find(':', colon + 1) != std::string::npos || spec[0] == '-')
        return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long draw = std::strtoull(spec.c_str(), &end, 0);
    if (errno == ERANGE || end != spec.c_str() + colon)
        return false;
    const std::string stage = spec.substr(colon + 1);
    if (stage != "vs" && stage != "fs") return false;
    selector.draw_index = static_cast<uint64_t>(draw);
    selector.vertex = stage == "vs";
    return true;
}

inline const std::vector<uint32_t>* select_recompiled_shader(
    const gpu::GpuReplayFrame& replay, const std::string& spec, bool& shared,
    std::string& error) {
    RealizedShaderSelector selector;
    if (!parse_realized_shader_selector(spec, selector)) {
        error = "invalid shader selector " + spec;
        return nullptr;
    }
    const size_t item_index = replay_item_index_for_draw(replay, selector.draw_index);
    if (item_index == SIZE_MAX) {
        error = "realized draw " + std::to_string(selector.draw_index) + " not found";
        return nullptr;
    }
    const auto selection = recompiled_shader_view(replay.items[item_index], selector.vertex);
    shared = selection.shared;
    error.clear();
    return selection.words;
}

inline const gpu::GpuCaptureRawShaderVersion* select_realized_raw_shader(
    const gpu::GpuReplayFrame& replay, const std::string& spec, std::string& error) {
    RealizedShaderSelector selector;
    if (!parse_realized_shader_selector(spec, selector)) {
        error = "invalid realized-shader selector " + spec;
        return nullptr;
    }
    const size_t item_index = replay_item_index_for_draw(replay, selector.draw_index);
    if (item_index == SIZE_MAX) {
        error = "realized draw " + std::to_string(selector.draw_index) + " not found";
        return nullptr;
    }
    const auto& draw = replay.items[item_index];
    const uint32_t raw_index = selector.vertex
        ? draw.vs_raw_shader_index : draw.fs_raw_shader_index;
    if (raw_index >= replay.raw_shader_versions.size()) {
        error = "realized shader " + spec +
                " has no captured raw stream (capture predates v19 or source was unreadable)";
        return nullptr;
    }
    error.clear();
    return &replay.raw_shader_versions[raw_index];
}

} // namespace prosper::tools
