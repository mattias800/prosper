#include "../tools/gpu_replay/realized_shader_dump.hpp"
#include "../src/gpu/diagnostic_selectors.hpp"

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
    CHECK(tools::parse_realized_shader_selector("7:vs-main", selector) &&
              selector.draw_index == 7 && selector.vertex && selector.vertex_main,
          "linked vertex-main selector parses exactly");
    CHECK(!tools::parse_realized_shader_selector("-1:vs", selector) &&
              !tools::parse_realized_shader_selector("1junk:fs", selector) &&
              !tools::parse_realized_shader_selector("1:ps", selector) &&
              !tools::parse_realized_shader_selector("1:vs:extra", selector),
          "malformed realized-shader selectors are rejected without partial parsing");

    uint64_t diagnostic_draw = 0;
    CHECK(gpu::parse_diagnostic_draw_id("0x481", diagnostic_draw) && diagnostic_draw == 1153,
          "geometry probe accepts the same explicit-base semantic ID in every layer");
    CHECK(!gpu::parse_diagnostic_draw_id("1153junk", diagnostic_draw) &&
              !gpu::parse_diagnostic_draw_id("-1", diagnostic_draw) &&
              !gpu::parse_diagnostic_draw_id("18446744073709551616", diagnostic_draw),
          "geometry probe rejects partial, signed, and overflowing draw IDs");

    uint64_t draw_first = 0, draw_last = 0;
    CHECK(gpu::parse_diagnostic_draw_range("0x7:013", draw_first, draw_last) &&
              draw_first == 7 && draw_last == 11 &&
              gpu::parse_diagnostic_draw_range("1153", draw_first, draw_last) &&
              draw_first == 1153 && draw_last == 1153,
          "draw selection parses complete semantic IDs and explicit numeric bases");
    CHECK(!gpu::parse_diagnostic_draw_range("7junk", draw_first, draw_last) &&
              !gpu::parse_diagnostic_draw_range("7:", draw_first, draw_last) &&
              !gpu::parse_diagnostic_draw_range(":11", draw_first, draw_last) &&
              !gpu::parse_diagnostic_draw_range("7:11:12", draw_first, draw_last),
          "draw selection rejects partial and incomplete ranges");

    uint32_t tap_pc = 0;
    CHECK(gpu::parse_fragment_tap_selector("0x481:0x18f", diagnostic_draw, tap_pc) &&
              diagnostic_draw == 1153 && tap_pc == 399,
          "fragment tap parses a complete semantic draw and 32-bit PC");
    CHECK(!gpu::parse_fragment_tap_selector("1153", diagnostic_draw, tap_pc) &&
              !gpu::parse_fragment_tap_selector("1153:399junk", diagnostic_draw, tap_pc) &&
              !gpu::parse_fragment_tap_selector("1153:0x100000000", diagnostic_draw, tap_pc) &&
              !gpu::parse_fragment_tap_selector("1153:399:1", diagnostic_draw, tap_pc),
          "fragment tap rejects missing, partial, overflowing, and extra components");

    gpu::GpuReplayFrame replay;
    replay.raw_shader_versions = {
        {11, true, {0x11111111u, 0xbf810000u}},
        {22, true, {0x22222222u, 0xbf810000u}},
        {33, true, {0x33333333u, 0xbf810000u}},
    };
    gpu::DrawItem draw;
    draw.draw_index = 7;
    draw.vs_raw_shader_index = 1;
    draw.fs_raw_shader_index = 0;
    draw.vs_chain_raw_shader_index = 2;
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
    const auto* main = tools::select_realized_raw_shader(replay, "7:vs-main", error);
    CHECK(vs == &replay.raw_shader_versions[1] && fs == &replay.raw_shader_versions[0] &&
              main == &replay.raw_shader_versions[2],
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
    replay.items[0].vs_chain_raw_shader_index = 0xFFFFFFFFu;
    CHECK(!tools::select_realized_raw_shader(replay, "7:vs-main", error) &&
              error.find("capture predates v31 or draw is not linked") != std::string::npos,
          "missing linked main source reports its versioned capture requirement");

    gpu::GpuReplayFrame shared_replay;
    gpu::DrawItem shared_draw;
    shared_draw.draw_index = 19;
    shared_draw.vs = {0xaaaaaaaa};
    shared_draw.fs = {0xbbbbbbbb};
    shared_draw.vs_shared =
        std::make_shared<const std::vector<uint32_t>>(std::vector<uint32_t>{0x07230203, 0x11});
    shared_draw.fs_shared =
        std::make_shared<const std::vector<uint32_t>>(std::vector<uint32_t>{0x07230203, 0x22, 0x33});
    shared_replay.items.push_back(shared_draw);
    bool shared = false;
    const auto* stored_vs = tools::select_recompiled_shader(shared_replay, "19:vs", shared, error);
    CHECK(stored_vs == shared_draw.vs_shared.get() && shared,
          "recompiled VS selection reads the shared words that rendering consumes");
    const auto* stored_fs = tools::select_recompiled_shader(shared_replay, "19:fs", shared, error);
    CHECK(stored_fs == shared_draw.fs_shared.get() && shared,
          "recompiled FS selection reads the shared words that rendering consumes");
    shared_replay.items[0].vs_shared.reset();
    const auto* owned_vs = tools::select_recompiled_shader(shared_replay, "19:vs", shared, error);
    CHECK(owned_vs == &shared_replay.items[0].vs && !shared,
          "recompiled shader selection retains the ordinary owned-vector path");
    CHECK(!tools::select_recompiled_shader(shared_replay, "20:vs", shared, error) &&
              error.find("realized draw 20 not found") != std::string::npos,
          "recompiled shader selection rejects an absent semantic draw");

    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
