#include "../tools/gpu_replay/realized_shader_dump.hpp"

#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_gpu_replay_shader_dump ==\n");

    tools::RealizedShaderSelector selector;
    CHECK(tools::parse_realized_shader_selector("0:vs", selector) &&
              selector.draw_index == 0 && selector.vertex,
          "realized VS selector parses exactly");
    CHECK(tools::parse_realized_shader_selector("0x1:fs", selector) &&
              selector.draw_index == 1 && !selector.vertex,
          "realized FS selector accepts an explicit numeric base");
    CHECK(!tools::parse_realized_shader_selector("-1:vs", selector) &&
              !tools::parse_realized_shader_selector("1junk:fs", selector) &&
              !tools::parse_realized_shader_selector("1:ps", selector) &&
              !tools::parse_realized_shader_selector("1:vs:extra", selector),
          "malformed realized-shader selectors are rejected without partial parsing");

    gpu::GpuReplayFrame replay;
    replay.raw_shader_versions = {
        {11, true, {0x11111111u, 0xbf810000u}},
        {22, true, {0x22222222u, 0xbf810000u}},
    };
    gpu::DrawItem draw;
    draw.vs_raw_shader_index = 1;
    draw.fs_raw_shader_index = 0;
    replay.items.push_back(draw);

    std::string error;
    const auto* vs = tools::select_realized_raw_shader(replay, "0:vs", error);
    const auto* fs = tools::select_realized_raw_shader(replay, "0:fs", error);
    CHECK(vs == &replay.raw_shader_versions[1] && fs == &replay.raw_shader_versions[0],
          "selector resolves the realized draw's exact raw stage identities");
    CHECK(!tools::select_realized_raw_shader(replay, "2:vs", error) &&
              error.find("invalid realized-shader selector") != std::string::npos,
          "out-of-range realized draw reports a selector error");
    replay.items[0].fs_raw_shader_index = 0xFFFFFFFFu;
    CHECK(!tools::select_realized_raw_shader(replay, "0:fs", error) &&
              error.find("capture predates v19 or source was unreadable") != std::string::npos,
          "missing legacy raw source is explicit instead of selecting unrelated bytes");

    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
