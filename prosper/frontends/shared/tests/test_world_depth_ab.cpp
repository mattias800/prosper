#include "shared/live/world_depth_ab.hpp"

#include <array>
#include <cstdio>
#include <initializer_list>

using prosper::frontend::WorldDepthAbConfig;
using prosper::frontend::WorldDepthDrawFact;
using prosper::frontend::WorldDepthPassKind;

static bool check(bool condition, const char* reason) {
    if (!condition) std::fprintf(stderr, "world-depth-ab: %s\n", reason);
    return condition;
}

int main() {
    WorldDepthAbConfig config;
    if (!check(prosper::frontend::parse_world_depth_ab(
                   "4:3930:0x2053960000:0x2049a00000:3840:2160", config),
               "valid config declined") ||
        !check(config.mode == 4 && config.capture_pad_flip == 3930 &&
                   config.depth_base == 0x2053960000ull &&
                   config.color_base == 0x2049a00000ull,
               "valid config parsed incorrectly"))
        return 1;
    for (const char* malformed : {
             "4:3930:0x2053960000:0x2049a00000:3840",
             "1:3930:0x2053960000:0x2049a00000:3840:2160",
             "4:3930:2053960000:0x2049a00000:3840:2160",
             "4:-1:0x2053960000:0x2049a00000:3840:2160",
             "4:3930:0x2053960000:0x2049a00000:16384:2160",
             "4:3930:0x2053960000:0x2049a00000:3840:2160:extra",
         })
        if (!check(!prosper::frontend::parse_world_depth_ab(malformed, config),
                   "malformed config admitted"))
            return 1;

    WorldDepthDrawFact clear;
    clear.depth_read_base = clear.depth_write_base = 0x2053960000ull;
    clear.depth_width = 3840; clear.depth_height = 2160;
    clear.depth_test = clear.depth_write = clear.clear_enable = true;
    clear.has_scissor = true; clear.scissor_right = 3840; clear.scissor_bottom = 2160;
    clear.compare_op = 7; clear.topology = 4; clear.vertex_count = 4;
    std::array<WorldDepthDrawFact, 1> one{clear};
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Clear,
               "world clear did not arm"))
        return 1;

    one[0].depth_width = one[0].depth_height = 1024;
    one[0].depth_read_base = one[0].depth_write_base = 0x2048a00000ull;
    one[0].scissor_right = one[0].scissor_bottom = 1024;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Unrelated,
               "shadow clear armed"))
        return 1;
    one[0] = clear;
    one[0].scissor_right = 3000;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Unrelated,
               "wrong scissor armed"))
        return 1;
    one[0] = clear;
    one[0].depth_read_base = 0x2048a00000ull;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Unrelated,
               "aliasing read identity armed"))
        return 1;

    WorldDepthDrawFact geometry = clear;
    geometry.compare_op = 6;
    geometry.topology = 3;
    geometry.vertex_count = 270;
    geometry.index_count = 270;
    geometry.color_base = config.color_base;
    geometry.color_write_mask = 0xf;
    geometry.scissor_right = 2880;
    geometry.scissor_bottom = 1620;
    one[0] = geometry;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Geometry,
               "world geometry did not arm"))
        return 1;
    one[0].color_base ^= 0x1000;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::OtherWorld,
               "different world color mislabeled as target geometry"))
        return 1;
    const std::array<WorldDepthDrawFact, 2> mixed{clear, WorldDepthDrawFact{}};
    if (!check(prosper::frontend::classify_world_depth_pass(config, mixed) ==
                   WorldDepthPassKind::Ambiguous,
               "mixed target pass armed"))
        return 1;
    const std::array<WorldDepthDrawFact, 2> mixed_scissors{clear, geometry};
    if (!check(prosper::frontend::classify_world_depth_pass(config, mixed_scissors) ==
                   WorldDepthPassKind::Ambiguous,
               "mixed clear and scene scissor armed as one pass"))
        return 1;
    if (!check(prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::Clear),
               "selected world clear lost its override") ||
        !check(prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::Geometry),
               "selected world geometry lost its override") ||
        !check(!prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::OtherWorld),
               "unproven same-depth pass gained an override") ||
        !check(!prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::Ambiguous),
               "mixed pass gained an override") ||
        !check(!prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::Unrelated),
               "shadow or unrelated pass gained an override"))
        return 1;
    const auto ambiguous = prosper::frontend::world_depth_capture_pass_ambiguous;
    if (!check(!ambiguous(false, WorldDepthPassKind::Geometry, true, true),
               "first matching world geometry declined") ||
        !check(ambiguous(true, WorldDepthPassKind::Geometry, true, true),
               "duplicate matching world geometry accepted") ||
        !check(ambiguous(true, WorldDepthPassKind::OtherWorld, true, false),
               "later depth target overwrite accepted") ||
        !check(ambiguous(true, WorldDepthPassKind::Unrelated, true, false),
               "later color target overwrite accepted") ||
        !check(ambiguous(false, WorldDepthPassKind::Ambiguous, true, false),
               "ambiguous target pass accepted") ||
        !check(!ambiguous(true, WorldDepthPassKind::Unrelated, false, false),
               "independent later pass invalidated world capture"))
        return 1;
    return 0;
}
