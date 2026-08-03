#include "../tools/gpu_replay/resource_override.hpp"
#include "test_scratch.h"

#include <cstdio>
#include <cstring>
#include <fstream>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_gpu_replay_resource_override ==\n");

    tools::ResourceOverrideSelector selector;
    CHECK(tools::parse_resource_override_selector("0x481:ps:040", selector) &&
              selector.draw_index == 1153 &&
              selector.stage == tools::ResourceOverrideStage::Pixel &&
              selector.binding == 32,
          "resource override uses strict semantic draw and explicit-base binding grammar");
    CHECK(!tools::parse_resource_override_selector("1153:fs:32", selector) &&
              !tools::parse_resource_override_selector("1153:ps", selector) &&
              !tools::parse_resource_override_selector("1153:ps:32:extra", selector) &&
              !tools::parse_resource_override_selector("1153junk:ps:32", selector) &&
              !tools::parse_resource_override_selector("-1:ps:32", selector) &&
              !tools::parse_resource_override_selector("1153:ps:0x100000000", selector),
          "malformed, partial, signed, and overflowing override selectors fail closed");

    std::vector<uint8_t> captured = {0x00, 0x00, 0x00, 0x00};
    const std::vector<uint8_t> captured_expected = captured;
    std::vector<uint8_t> neighbour = {0xaa, 0xbb, 0xcc, 0xdd};
    auto shared_table = std::make_shared<gpu::ShaderResourceTable>();
    gpu::ShaderResource selected_resource{};
    selected_resource.cls = gpu::ResourceClass::ConstantBuffer;
    selected_resource.binding = 32;
    selected_resource.gpu_addr = 0x12345000;
    selected_resource.size = 4;
    selected_resource.host_data = captured.data();
    selected_resource.host_data_size = captured.size();
    gpu::ShaderResource neighbouring_resource = selected_resource;
    neighbouring_resource.binding = 33;
    neighbouring_resource.gpu_addr = 0x12346000;
    neighbouring_resource.host_data = neighbour.data();
    neighbouring_resource.host_data_size = neighbour.size();
    shared_table->resources = {selected_resource, neighbouring_resource};

    gpu::GpuReplayFrame replay;
    gpu::DrawItem selected_draw;
    selected_draw.draw_index = 1153;
    selected_draw.vrt = shared_table;
    selected_draw.prt = shared_table;
    gpu::DrawItem sibling_draw;
    sibling_draw.draw_index = 1154;
    sibling_draw.prt = shared_table;
    replay.items = {selected_draw, sibling_draw};

    CHECK(tools::parse_resource_override_selector("1153:ps:32", selector),
          "decimal override selector parses");
    const auto* original_table = shared_table.get();
    uint8_t* const original_selected_pointer = shared_table->resources[0].host_data;
    uint8_t* const original_neighbour_pointer = shared_table->resources[1].host_data;
    const uint64_t original_hash = gpu::gpu_capture_hash(captured);
    const std::vector<uint8_t> replacement = {0x00, 0x3c, 0x00, 0x3c};
    const uint64_t expected_replacement_hash = gpu::gpu_capture_hash(replacement);
    tools::AppliedResourceOverride applied;
    std::string error;
    CHECK(tools::apply_resource_override(replay, selector, replacement, applied, error),
          "draw-scoped resource override installs into a captured replay");

    const auto& changed = replay.items[0].prt->resources[0];
    CHECK(replay.items[0].prt.get() != original_table &&
              changed.host_data == applied.replacement_bytes->data() &&
              changed.host_data != original_selected_pointer &&
              std::memcmp(changed.host_data, replacement.data(), replacement.size()) == 0 &&
              gpu::gpu_capture_hash(changed.host_data,
                                    static_cast<size_t>(changed.host_data_size)) ==
                  expected_replacement_hash &&
              applied.original_hash == original_hash &&
              applied.replacement_hash == expected_replacement_hash &&
              applied.original_hash != applied.replacement_hash,
          "selected draw bytes and inspect-visible hash change");

    const auto& sibling = replay.items[1].prt->resources[0];
    CHECK(replay.items[1].prt.get() == original_table &&
              sibling.host_data == original_selected_pointer &&
              captured == captured_expected &&
              std::memcmp(sibling.host_data, captured_expected.data(), captured_expected.size()) == 0 &&
              gpu::gpu_capture_hash(sibling.host_data,
                                    static_cast<size_t>(sibling.host_data_size)) == original_hash,
          "shared sibling draw remains byte-identical after selected override");
    CHECK(replay.items[0].vrt.get() == original_table &&
              replay.items[0].prt->resources[1].host_data == original_neighbour_pointer &&
              neighbour == std::vector<uint8_t>({0xaa, 0xbb, 0xcc, 0xdd}),
          "unselected stage and binding retain their original objects and bytes");
    CHECK(applied.target.item_index == 0 && applied.target.resource_index == 0 &&
              applied.target.gpu_addr == 0x12345000 && applied.target.captured_size == 4,
          "installed override reports exact draw-resource identity and captured span");

    tools::AppliedResourceOverride rejected;
    auto missing_draw = selector;
    missing_draw.draw_index = 9999;
    CHECK(!tools::apply_resource_override(replay, missing_draw, replacement, rejected, error) &&
              error.find("realized draw 9999 not found") != std::string::npos,
          "missing semantic draw fails visibly");
    auto missing_binding = selector;
    missing_binding.binding = 99;
    CHECK(!tools::apply_resource_override(replay, missing_binding, replacement, rejected, error) &&
              error.find("binding 99 not found") != std::string::npos,
          "missing binding fails visibly");
    CHECK(!tools::apply_resource_override(replay, selector, {1, 2, 3}, rejected, error) &&
              error.find("selected captured span is 4 bytes") != std::string::npos,
          "replacement size mismatch fails before changing replay state");
    CHECK(replay.items[0].prt->resources[0].host_data == changed.host_data &&
              gpu::gpu_capture_hash(replay.items[0].prt->resources[0].host_data, 4) ==
                  expected_replacement_hash,
          "failed override leaves the previously selected bytes unchanged");

    const auto exact_path = prosper_test::test_scratch_dir() / "resource-override-exact.bin";
    const auto short_path = prosper_test::test_scratch_dir() / "resource-override-short.bin";
    {
        std::ofstream exact(exact_path, std::ios::binary);
        exact.write(reinterpret_cast<const char*>(replacement.data()), replacement.size());
        std::ofstream short_file(short_path, std::ios::binary);
        short_file.write(reinterpret_cast<const char*>(replacement.data()),
                         replacement.size() - 1);
    }
    std::vector<uint8_t> loaded_override;
    CHECK(tools::read_exact_resource_override_file(
              exact_path.string(), replacement.size(), loaded_override, error) &&
              loaded_override == replacement,
          "replacement file reader accepts exactly the selected captured span");
    CHECK(!tools::read_exact_resource_override_file(
              short_path.string(), replacement.size(), loaded_override, error) &&
              error.find("has 3 bytes") != std::string::npos,
          "replacement file size mismatch fails visibly");
    CHECK(!tools::read_exact_resource_override_file(
              (prosper_test::test_scratch_dir() / "missing.bin").string(),
              replacement.size(), loaded_override, error) &&
              error.find("cannot read resource override") != std::string::npos,
          "missing replacement file fails visibly");

    gpu::GpuReplayFrame ambiguous;
    gpu::DrawItem ambiguous_draw;
    ambiguous_draw.draw_index = 12;
    ambiguous_draw.prt = std::make_shared<gpu::ShaderResourceTable>();
    ambiguous_draw.prt->resources = {selected_resource, selected_resource};
    ambiguous.items.push_back(ambiguous_draw);
    auto ambiguous_selector = selector;
    ambiguous_selector.draw_index = 12;
    CHECK(!tools::apply_resource_override(
              ambiguous, ambiguous_selector, replacement, rejected, error) &&
              error.find("binding 32 is ambiguous") != std::string::npos,
          "duplicate binding fails instead of selecting an arbitrary resource");

    gpu::GpuReplayFrame absent_table;
    gpu::DrawItem absent_draw;
    absent_draw.draw_index = 13;
    absent_table.items.push_back(absent_draw);
    auto absent_selector = selector;
    absent_selector.draw_index = 13;
    CHECK(!tools::apply_resource_override(
              absent_table, absent_selector, replacement, rejected, error) &&
              error.find("resource table is absent") != std::string::npos,
          "missing stage table fails visibly");

    gpu::GpuReplayFrame absent_bytes;
    gpu::DrawItem byte_less_draw;
    byte_less_draw.draw_index = 14;
    byte_less_draw.prt = std::make_shared<gpu::ShaderResourceTable>();
    auto byte_less_resource = selected_resource;
    byte_less_resource.host_data = nullptr;
    byte_less_resource.host_data_size = 0;
    byte_less_draw.prt->resources.push_back(byte_less_resource);
    absent_bytes.items.push_back(byte_less_draw);
    auto absent_bytes_selector = selector;
    absent_bytes_selector.draw_index = 14;
    CHECK(!tools::apply_resource_override(
              absent_bytes, absent_bytes_selector, replacement, rejected, error) &&
              error.find("has no captured bytes") != std::string::npos,
          "resource without captured bytes fails visibly");

    std::printf("%s\n", fails ? "FAILED" : "ALL TESTS PASSED");
    return fails ? 1 : 0;
}
