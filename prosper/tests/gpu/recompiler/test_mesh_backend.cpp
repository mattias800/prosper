// Exercises the shipping offscreen backend's optional mesh stage. The positive shader writes three
// vertices only after two peer-lane shared-memory exchanges; two SPIR-V variants alter one source
// expression each so a private-LDS or incorrect prefix projection loses the center pixel.
// These are host workgroup controls, not a claim that an RDNA merged-NGG program recompiles yet.
#include "fixtures/render_runner.h"
#include "fixtures/mesh_workgroup_spirv.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <vector>

static std::array<uint8_t, 4> center(const std::vector<uint8_t>& pixels) {
    constexpr size_t at = (32u * 64u + 32u) * 4u;
    if (pixels.size() <= at + 3u) return {};
    return {pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]};
}

int main() {
    const auto& ctx = prosper::test::render_vk_ctx();
    if (!ctx.ok) return 77;
    using namespace prosper::test::mesh_fixture;
    const std::vector<uint32_t> fs(std::begin(fragment), std::end(fragment));
    const std::vector<uint32_t> exact_module(std::begin(exact), std::end(exact));
    const std::vector<uint32_t> one_lane_module(std::begin(one_lane), std::end(one_lane));
    const std::vector<uint32_t> wrong_prefix_module(
        std::begin(wrong_prefix), std::end(wrong_prefix));
    constexpr float clear[4] = {0, 0, 1, 1};
    auto render = [&](const std::vector<uint32_t>& ms,
                      std::array<uint32_t, 3> groups = {1, 1, 1}) {
        prosper::test::BackendDraw draw;
        draw.mesh_draw = true;
        draw.mesh_groups = groups;
        draw.vs = ms;
        draw.fs = fs;
        return center(prosper::test::render_draws_rgba({draw}, 64, 64, nullptr, clear));
    };
    const auto positive = render(exact_module);
    const bool red = positive[0] > 200 && positive[1] < 50 && positive[2] < 50;
    const bool blue = positive[0] < 50 && positive[1] < 50 && positive[2] > 200;
    std::printf("mesh feature=%d center=%u,%u,%u,%u\n",
                ctx.mesh_shader_enabled, positive[0], positive[1], positive[2], positive[3]);
    if (ctx.mesh_shader_enabled) {
        if (!red) return 1;
        const auto lane = render(one_lane_module);
        const auto prefix = render(wrong_prefix_module);
        const auto zero = render(exact_module, {0, 1, 1});
        const auto is_blue = [](std::array<uint8_t, 4> p) {
            return p[0] < 50 && p[1] < 50 && p[2] > 200;
        };
        if (!is_blue(lane) || !is_blue(prefix) || !is_blue(zero)) {
            std::fprintf(stderr, "mesh negative control failed: lane=%u,%u,%u "
                                 "prefix=%u,%u,%u zero=%u,%u,%u\n",
                         lane[0], lane[1], lane[2], prefix[0], prefix[1], prefix[2],
                         zero[0], zero[1], zero[2]);
            return 1;
        }
    } else if (!blue) {
        std::fprintf(stderr, "unavailable mesh stage changed the clear image\n");
        return 1;
    }
    return 0;
}
