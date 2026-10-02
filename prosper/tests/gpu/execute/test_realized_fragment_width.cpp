// #4041: retain the actual guest compiler input, never reconstruct it from emitted SPIR-V.
// All inputs are project-owned synthetic ISA; realization/collection/codec/replay are CPU-only.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/capture/gpu_capture_bundle.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
static int failures = 0;
#define CHECK(c, m) do { if (c) std::printf("  [ok]   %s\n", m); \
    else { std::printf("  [FAIL] %s\n", m); ++failures; } } while (0)

namespace {
alignas(256) constexpr uint32_t vertex_words[] = {
    0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u, 0x7e0c02f2u,
    0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u, 0x7e060280u, 0x7e0802f2u,
    0xf80008cfu, 0x04030201u, 0xbf810000u,
};
alignas(256) constexpr uint32_t fragment_words[] = {
    0xd7660000u, 0x00010081u, // v_mbcnt_hi_u32_b32 v0, 1, 0
    0x7e000d00u, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u,
    0xf800180fu, 0x03020100u, 0xbf810000u,
};
alignas(256) constexpr uint32_t dcc_helper_words[] = {
    0x7e000280u, 0xf8001803u, 0x00000000u, 0xbf810000u,
};

void set_program(GpuState& state, uint32_t lo, uint32_t hi, const void* program) {
    const auto address = reinterpret_cast<uint64_t>(program);
    state.sh[lo] = static_cast<uint32_t>(address >> 8);
    state.sh[hi] = static_cast<uint32_t>((address >> 40) & 0xffu);
}

GpuState state_for(bool wave32, const uint32_t* fragment = fragment_words) {
    GpuState state;
    set_program(state, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, vertex_words);
    set_program(state, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, fragment);
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4u;
    state.cx[P::CB_TARGET_MASK] = 0xfu;
    state.cx[P::CB_COLOR_CONTROL] =
        P::CB_COLOR_CONTROL_MODE_NORMAL << P::CB_COLOR_CONTROL_MODE_SHIFT;
    state.cx[P::SPI_PS_IN_CONTROL] = wave32 ? 1u << P::SPI_PS_IN_CONTROL_PS_W32_EN_SHIFT : 0u;
    return state;
}

size_t read_shader(uint64_t address, uint8_t* destination, size_t bytes) {
    for (const auto& span : {std::pair{vertex_words, sizeof(vertex_words)},
                             std::pair{fragment_words, sizeof(fragment_words)},
                             std::pair{dcc_helper_words, sizeof(dcc_helper_words)}}) {
        const auto base = reinterpret_cast<uint64_t>(span.first);
        if (address < base || address - base >= span.second) continue;
        const size_t offset = static_cast<size_t>(address - base);
        const size_t copied = std::min(bytes, span.second - offset);
        std::memcpy(destination, reinterpret_cast<const uint8_t*>(span.first) + offset, copied);
        return copied;
    }
    return 0;
}

void set_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) bytes[offset + i] = static_cast<uint8_t>(value >> (i * 8));
}
} // namespace

int main() {
    std::printf("== test_realized_fragment_width ==\n");
    clear_shader_recompile_cache();
    std::vector<DrawItem> realized;
    for (bool shared : {false, true}) {
        for (bool wave32 : {false, true, false, true}) {
            const auto state = state_for(wave32);
            DrawItem draw;
            GpuState::Draw packet;
            packet.index_count = 3u;
            CHECK(realize_draw_item(state, &packet, 3u, std::size(vertex_words), false,
                                    draw, nullptr, shared),
                  "actual successful fragment realization executes the compiler boundary");
            CHECK(draw.fragment_wave_config_available && draw.ps_wave32 == wave32 &&
                      fragment_spirv_required_subgroup_size(draw.fs_words()) == (wave32 ? 32u : 64u),
                  "cold/warm, copied/shared modules retain the actual guest width argument");
            draw.draw_index = realized.size();
            draw.command_order = 100u + realized.size();
            realized.push_back(std::move(draw));
        }
    }

    // The authoritative source is the same draw snapshot that feeds compilation, not the
    // submit's final sticky register state. Force serial realization; no Vulkan is initialized.
    GpuState mixed = state_for(false);
    for (bool wave32 : {true, false}) {
        GpuState::Draw packet;
        packet.index_count = 3u;
        packet.command_order = 200u + mixed.draws.size();
        packet.state = std::make_shared<const GpuState>(state_for(wave32));
        mixed.draws.push_back(std::move(packet));
    }
    const auto mixed_draws = realize_gpustate_draws(
        mixed, std::size(vertex_words), 1.0f, 1.0f, nullptr, true, false);
    CHECK(mixed_draws.size() == 2u && mixed_draws[0].fragment_wave_config_available &&
              mixed_draws[0].ps_wave32 && mixed_draws[1].fragment_wave_config_available &&
              !mixed_draws[1].ps_wave32,
          "mixed-width draws retain their own compile snapshot rather than submit-end width");

    // The existing DCC utility path replaces the effective module with a null export. Width
    // provenance must still name the guest compile input, not infer from that replacement.
    auto utility = state_for(true, dcc_helper_words);
    utility.cx[P::CB_COLOR_CONTROL] =
        P::CB_COLOR_CONTROL_MODE_DCC_DECOMPRESS << P::CB_COLOR_CONTROL_MODE_SHIFT;
    DrawItem utility_draw;
    CHECK(realize_draw_item(utility, nullptr, 3u, std::size(vertex_words), false, utility_draw) &&
              utility_draw.fragment_wave_config_available && utility_draw.ps_wave32 &&
              fragment_spirv_required_subgroup_size(utility_draw.fs_words()) == 0u,
          "guest Wave32 provenance survives an effective width-independent utility replacement");

    GpuCaptureMetadata metadata;
    metadata.width = metadata.height = 1u;
    std::string error;
    GpuCaptureFile capture;
    CHECK(capture_draw_items(realized, metadata, read_shader, capture, error) &&
              capture.draws.size() == realized.size(),
          "production collector retains successfully realized draws");
    std::vector<uint8_t> bytes;
    GpuCaptureFile loaded;
    CHECK(serialize_gpu_capture(capture, bytes, error) &&
              deserialize_gpu_capture(bytes, loaded, error) && loaded.format_version == 66u,
          "current realized-width tail round-trips through production codecs");
    GpuReplayFrame replay;
    CHECK(materialize_gpu_replay(loaded, replay, error) && replay.items.size() == realized.size(),
          "production replay materializes captured launch ABI");
    for (size_t i = 0; i < realized.size(); ++i) {
        CHECK(i < capture.draws.size() && capture.draws[i].fragment_wave_config_available &&
                  capture.draws[i].ps_wave32 == realized[i].ps_wave32 &&
                  i < replay.items.size() && replay.items[i].fragment_wave_config_available &&
                  replay.items[i].ps_wave32 == realized[i].ps_wave32 &&
                  replay.items[i].fs_words() == realized[i].fs_words(),
              "collection/serialization/materialization preserve known32 and known64 without rewriting modules");
    }
    GpuCaptureBundle bundle;
    GpuCaptureFile bundled, manifest;
    CHECK(append_gpu_capture_bundle(bundle, capture, error) &&
              materialize_gpu_capture_bundle_submit(bundle, 0u, bundled, error) &&
              materialize_gpu_capture_bundle_manifest(bundle, 0u, manifest, error) &&
              bundled.draws.size() == realized.size() && manifest.draws.size() == realized.size(),
          "bundle payload and manifest materialization retain the appended width ABI");
    for (const auto* result : {&bundled, &manifest}) {
        for (size_t i = 0; i < realized.size(); ++i)
            CHECK(i < result->draws.size() && result->draws[i].fragment_wave_config_available &&
                      result->draws[i].ps_wave32 == realized[i].ps_wave32,
                  "full and metadata-only bundle paths preserve each captured guest width");
    }
    if (bytes.size() < 4u + realized.size() || loaded.draws.size() != realized.size()) {
        CHECK(false, "codec controls require the successful bounded realized fixture");
        return 1;
    }
    const size_t mode_tail_size = 8u + 2u * realized.size();
    const size_t transport_tail_size = 12u + realized.size();
    const size_t flags_tail_size = 8u + 3u * realized.size();
    const size_t tail = bytes.size() - flags_tail_size - transport_tail_size - mode_tail_size - 4u - realized.size();

    auto legacy_bytes = bytes;
    legacy_bytes.resize(tail);
    set_u32(legacy_bytes, 8u, 62u);
    GpuCaptureFile legacy;
    CHECK(deserialize_gpu_capture(legacy_bytes, legacy, error) && legacy.format_version == 62u &&
              legacy.draws.size() == realized.size(),
          "genuine v62 prefix remains readable after removing only the appended v63 tail");
    for (const auto& draw : legacy.draws)
        CHECK(!draw.fragment_wave_config_available && !draw.ps_wave32,
              "legacy source/effective SPIR-V width markers never manufacture captured width");
    std::vector<uint8_t> rewritten;
    GpuCaptureFile upgraded;
    CHECK(serialize_gpu_capture(legacy, rewritten, error) &&
              deserialize_gpu_capture(rewritten, upgraded, error) &&
              std::all_of(upgraded.draws.begin(), upgraded.draws.end(), [](const auto& draw) {
                  return !draw.fragment_wave_config_available && !draw.ps_wave32;
              }), "legacy-to-current rewrite preserves unavailable width");

    auto unknown = capture;
    for (auto& draw : unknown.draws) {
        draw.fragment_wave_config_available = false;
        draw.ps_wave32 = false;
    }
    CHECK(serialize_gpu_capture(unknown, rewritten, error) &&
              deserialize_gpu_capture(rewritten, upgraded, error) &&
              !upgraded.draws[0].fragment_wave_config_available,
          "current external modules can explicitly retain unavailable guest provenance");
    unknown.draws[0].ps_wave32 = true;
    CHECK(!serialize_gpu_capture(unknown, rewritten, error) &&
              error == "invalid realized-draw fragment wave config",
          "writer rejects a width value whose provenance is unavailable");
    CHECK(!materialize_gpu_replay(unknown, replay, error) &&
              error == "invalid realized-draw fragment wave config",
          "direct materializer also rejects noncanonical unavailable state");
    auto invalid_draws = realized;
    invalid_draws[0].fragment_wave_config_available = false;
    invalid_draws[0].ps_wave32 = true;
    CHECK(!capture_draw_items(invalid_draws, metadata, read_shader, upgraded, error) &&
              error == "invalid realized-draw fragment wave config",
          "collector rejects noncanonical external draw state");

    for (uint32_t count : {0u, 1u, static_cast<uint32_t>(realized.size() + 1u), UINT32_MAX}) {
        auto malformed = bytes;
        set_u32(malformed, tail, count);
        CHECK(!deserialize_gpu_capture(malformed, upgraded, error) &&
                  error == "invalid realized-draw fragment wave count",
              "reader rejects mismatched/hostile width counts without allocating a second draw list");
    }
    for (uint8_t tag : {uint8_t{3}, uint8_t{128}, uint8_t{255}}) {
        auto malformed = bytes;
        malformed[tail + 4u] = tag;
        CHECK(!deserialize_gpu_capture(malformed, upgraded, error) &&
                  error == "invalid realized-draw fragment wave config",
              "reader rejects reserved width tags rather than treating them as a Boolean");
    }
    for (size_t end : {tail, tail + 1u, tail + 3u, tail + 4u, bytes.size() - 1u}) {
        auto malformed = bytes;
        malformed.resize(end);
        CHECK(!deserialize_gpu_capture(malformed, upgraded, error),
              "reader refuses truncated count or per-draw width tags");
    }
    auto trailing = bytes;
    trailing.push_back(0u);
    CHECK(!deserialize_gpu_capture(trailing, upgraded, error) && error == "capture has trailing data",
          "reader refuses data after the complete v63 tail");
    GpuCaptureFile empty;
    CHECK(serialize_gpu_capture(empty, rewritten, error) &&
              deserialize_gpu_capture(rewritten, upgraded, error) && upgraded.draws.empty(),
          "empty capture has a canonical zero-count width tail");
    std::printf("== %s (%d failures) ==\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
