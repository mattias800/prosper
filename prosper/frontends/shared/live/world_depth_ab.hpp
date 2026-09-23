#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>

namespace prosper::frontend {

// Opt-in #2790 experiment, never a guest-depth rule. Every field is required so an old route or
// reallocated surface fails visibly instead of silently targeting a shadow map. The addresses
// identify the retained allocations; the dimensions and draw-state checks below verify their role.
struct WorldDepthAbConfig {
    int mode = 0;                  // 0 (control), 4 or 5 (existing diagnostic policies)
    int64_t capture_pad_flip = -1;
    uint64_t depth_base = 0, color_base = 0;
    uint32_t width = 0, height = 0;
};

inline bool parse_world_depth_ab(std::string_view input, WorldDepthAbConfig& out) {
    WorldDepthAbConfig parsed;
    std::string_view fields[6];
    for (unsigned i = 0; i < 6; ++i) {
        const size_t cut = input.find(':');
        fields[i] = input.substr(0, cut);
        if (fields[i].empty() || (i < 5 && cut == input.npos) ||
            (i == 5 && cut != input.npos))
            return false;
        if (i < 5) input.remove_prefix(cut + 1);
    }
    auto decimal = [](std::string_view s, uint64_t& value) {
        const char* const end = s.data() + s.size();
        const auto result = std::from_chars(s.data(), end, value, 10);
        return result.ec == std::errc{} && result.ptr == end;
    };
    auto hex_address = [](std::string_view s, uint64_t& value) {
        if (s.size() < 3 || s.substr(0, 2) != "0x") return false;
        s.remove_prefix(2);
        const char* const end = s.data() + s.size();
        const auto result = std::from_chars(s.data(), end, value, 16);
        return result.ec == std::errc{} && result.ptr == end && value != 0;
    };
    uint64_t mode = 0, flip = 0, width = 0, height = 0;
    if (!decimal(fields[0], mode) || (mode != 0 && mode != 4 && mode != 5) ||
        !decimal(fields[1], flip) || flip > INT64_MAX ||
        !hex_address(fields[2], parsed.depth_base) ||
        !hex_address(fields[3], parsed.color_base) ||
        !decimal(fields[4], width) || !decimal(fields[5], height) ||
        !width || !height || width > 8192 || height > 8192 ||
        width * height > (64u << 20) / sizeof(float))
        return false;
    parsed.mode = static_cast<int>(mode);
    parsed.capture_pad_flip = static_cast<int64_t>(flip);
    parsed.width = static_cast<uint32_t>(width);
    parsed.height = static_cast<uint32_t>(height);
    out = parsed;
    return true;
}

struct WorldDepthDrawFact {
    uint64_t depth_read_base = 0, depth_write_base = 0, htile_base = 0;
    uint64_t color_base = 0;
    uint32_t depth_width = 0, depth_height = 0;
    uint32_t compare_op = 0, topology = 0, shader_control = 0;
    uint32_t vertex_count = 0, index_count = 0;
    uint32_t color_write_mask = 0, all_color_write_masks = 0;
    bool depth_test = false, depth_write = false, clear_enable = false;
    bool has_scissor = false;
    int32_t scissor_left = 0, scissor_top = 0, scissor_right = 0, scissor_bottom = 0;
};

enum class WorldDepthPassKind { Unrelated, Clear, Geometry, OtherWorld, Ambiguous };

// This is the pad value sampled at callback entry. It identifies when a group was observed;
// it does not prove that its produced pixels reached a later native presentation.
constexpr bool world_census_observation_allowed(bool armed, bool census_only,
                                                 int64_t requested_pad, int64_t callback_pad) {
    return armed && census_only && requested_pad == callback_pad;
}

// Metadata-only first-divergence preflight. A bound target is not proof of a pixel write, and a
// same-address producer in another frame is not the version sampled by the present composite.
// The state deliberately permits no readback decision; it only reports whether this pad contains
// an ordered, single large world candidate worth tracing further. One line is reserved for the
// final summary, so all group and callback rows together can never exceed kMaximumLines - 1.
struct WorldProducerCensus {
    static constexpr uint64_t kMinimumWorldDraws = 100;
    static constexpr uint64_t kMaximumLines = 64;
    uint64_t depth_groups = 0, world_groups = 0, large_world_groups = 0, composite_groups = 0;
    uint64_t composite_after_large_world = 0, max_world_draws = 0;
    uint64_t relevant_groups = 0, reported_lines = 0, omitted_lines = 0;
    uint64_t unstable_callbacks = 0;

    bool reserve_detail_line() {
        if (reported_lines < kMaximumLines - 1) {
            ++reported_lines;
            return true;
        }
        ++omitted_lines;
        return false;
    }

    void observe(bool depth_target_bound, bool world_target_bound, bool composite_target_bound,
                 WorldDepthPassKind kind, uint64_t draws) {
        const bool large_preceded_this_group = large_world_groups != 0;
        if (depth_target_bound) ++depth_groups;
        if (world_target_bound) {
            ++world_groups;
            if (draws > max_world_draws) max_world_draws = draws;
            if (kind == WorldDepthPassKind::Geometry && draws >= kMinimumWorldDraws)
                ++large_world_groups;
        }
        if (composite_target_bound) {
            ++composite_groups;
            if (large_preceded_this_group) ++composite_after_large_world;
        }
        if (depth_target_bound || world_target_bound || composite_target_bound) {
            ++relevant_groups;
        }
    }

    bool observe_at(bool armed, bool census_only, int64_t requested_pad, int64_t callback_pad,
                    bool depth_target_bound, bool world_target_bound, bool composite_target_bound,
                    WorldDepthPassKind kind, uint64_t draws) {
        if (!world_census_observation_allowed(armed, census_only, requested_pad, callback_pad))
            return false;
        observe(depth_target_bound, world_target_bound, composite_target_bound, kind, draws);
        return true;
    }

    bool has_single_ordered_candidate() const {
        return large_world_groups == 1 && composite_after_large_world != 0 &&
               omitted_lines == 0 && unstable_callbacks == 0;
    }
};

constexpr bool world_depth_override_enabled(bool armed, bool census_only,
                                            WorldDepthPassKind kind) {
    return armed && !census_only &&
           (kind == WorldDepthPassKind::Clear || kind == WorldDepthPassKind::Geometry);
}

constexpr bool world_depth_scope_allowed(WorldDepthPassKind kind) {
    return kind == WorldDepthPassKind::Clear || kind == WorldDepthPassKind::Geometry;
}

// The callback-tail readback is attributable to one selected geometry group only if no other
// group can replace either attachment afterward. Surface_match includes color base and extent;
// conservative attachment identity counts as "touch" even when pipeline admission is unknown.
constexpr bool world_depth_capture_pass_ambiguous(bool prior_geometry,
                                                   WorldDepthPassKind kind,
                                                   bool attachment_touched,
                                                   bool surface_match) {
    return (prior_geometry && attachment_touched) ||
           (kind == WorldDepthPassKind::Ambiguous && attachment_touched) ||
           (kind == WorldDepthPassKind::Geometry && (!surface_match || prior_geometry));
}

inline WorldDepthPassKind classify_world_depth_pass(
    const WorldDepthAbConfig& config, std::span<const WorldDepthDrawFact> draws) {
    if (draws.empty()) return WorldDepthPassKind::Unrelated;
    bool saw_target = false, saw_other = false, clear = true, geometry = true;
    bool saw_full_scissor = false, saw_scene_scissor = false;
    for (const WorldDepthDrawFact& draw : draws) {
        const bool identity = draw.depth_write_base == config.depth_base &&
            (draw.depth_read_base == 0 || draw.depth_read_base == config.depth_base) &&
            draw.depth_width == config.width && draw.depth_height == config.height &&
            draw.htile_base == 0 && draw.depth_test && draw.depth_write;
        const bool full_scissor = draw.has_scissor && draw.scissor_left == 0 &&
            draw.scissor_top == 0 && draw.scissor_right == static_cast<int32_t>(config.width) &&
            draw.scissor_bottom == static_cast<int32_t>(config.height);
        const bool scene_scissor = draw.has_scissor && draw.scissor_left == 0 &&
            draw.scissor_top == 0 &&
            draw.scissor_right == static_cast<int32_t>(config.width * 3 / 4) &&
            draw.scissor_bottom == static_cast<int32_t>(config.height * 3 / 4);
        const bool target = identity && (full_scissor || scene_scissor);
        saw_target |= target;
        saw_other |= !target;
        if (!target) continue;
        saw_full_scissor |= full_scissor;
        saw_scene_scissor |= scene_scissor;
        // Vulkan enum values: ALWAYS=7, GREATER_OR_EQUAL=6, TRIANGLE_STRIP=4.
        clear &= full_scissor && draw.clear_enable && draw.compare_op == 7 && draw.topology == 4 &&
            draw.vertex_count == 4 && draw.index_count == 0 &&
            draw.all_color_write_masks == 0 && (draw.shader_control & 0x31u) == 0;
        geometry &= scene_scissor && draw.clear_enable && draw.compare_op == 6 &&
            draw.color_base == config.color_base && draw.color_write_mask != 0;
    }
    if (!saw_target) return WorldDepthPassKind::Unrelated;
    if (saw_other || (saw_full_scissor && saw_scene_scissor))
        return WorldDepthPassKind::Ambiguous;
    if (clear) return WorldDepthPassKind::Clear;
    if (geometry) return WorldDepthPassKind::Geometry;
    return WorldDepthPassKind::OtherWorld;
}

} // namespace prosper::frontend
