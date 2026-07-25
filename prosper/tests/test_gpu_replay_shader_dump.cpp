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
    draw.draw_index = 7;
    draw.vs_raw_shader_index = 1;
    draw.fs_raw_shader_index = 0;
    replay.items.push_back(draw);

    gpu::DrawItem later = draw;
    later.draw_index = 11;
    later.vs_raw_shader_index = 0;
    later.fs_raw_shader_index = 1;
    replay.items.push_back(later);
    replay.operations = {
        {gpu::SubmitOperationKind::Draw, 7, 100, true},
        {gpu::SubmitOperationKind::Dispatch, 3, 101, true},
        {gpu::SubmitOperationKind::Draw, 11, 102, true},
    };

    std::string error;
    const auto* vs = tools::select_realized_raw_shader(replay, "7:vs", error);
    const auto* fs = tools::select_realized_raw_shader(replay, "7:fs", error);
    CHECK(vs == &replay.raw_shader_versions[1] && fs == &replay.raw_shader_versions[0],
          "selector resolves a semantic draw ID instead of a compact item offset");
    CHECK(tools::replay_item_index_for_draw(replay, 11) == 1 &&
              tools::replay_operation_index_for_draw(replay, 11) == 2 &&
              tools::replay_item_index_for_draw(replay, 1) == SIZE_MAX,
          "draw IDs map across compact-item holes and mixed operation indices");
    CHECK(!tools::select_realized_raw_shader(replay, "2:vs", error) &&
              error.find("realized draw 2 not found") != std::string::npos,
          "missing semantic draw reports a selector error");
    replay.items[0].fs_raw_shader_index = 0xFFFFFFFFu;
    CHECK(!tools::select_realized_raw_shader(replay, "7:fs", error) &&
              error.find("capture predates v19 or source was unreadable") != std::string::npos,
          "missing legacy raw source is explicit instead of selecting unrelated bytes");

    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
