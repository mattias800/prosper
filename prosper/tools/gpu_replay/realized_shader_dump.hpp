#pragma once

#include "../../src/gpu/gpu_capture.hpp"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>

namespace prosper::tools {

struct RealizedShaderSelector {
    size_t draw_index = 0;
    bool vertex = false;
};

inline bool parse_realized_shader_selector(const std::string& spec,
                                           RealizedShaderSelector& selector) {
    const size_t colon = spec.find(':');
    if (colon == std::string::npos || colon == 0 ||
        spec.find(':', colon + 1) != std::string::npos || spec[0] == '-')
        return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long draw = std::strtoull(spec.c_str(), &end, 0);
    if (errno == ERANGE || end != spec.c_str() + colon ||
        draw > std::numeric_limits<size_t>::max())
        return false;
    const std::string stage = spec.substr(colon + 1);
    if (stage != "vs" && stage != "fs") return false;
    selector.draw_index = static_cast<size_t>(draw);
    selector.vertex = stage == "vs";
    return true;
}

inline const gpu::GpuCaptureRawShaderVersion* select_realized_raw_shader(
    const gpu::GpuReplayFrame& replay, const std::string& spec, std::string& error) {
    RealizedShaderSelector selector;
    if (!parse_realized_shader_selector(spec, selector) ||
        selector.draw_index >= replay.items.size()) {
        error = "invalid realized-shader selector " + spec;
        return nullptr;
    }
    const auto& draw = replay.items[selector.draw_index];
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
