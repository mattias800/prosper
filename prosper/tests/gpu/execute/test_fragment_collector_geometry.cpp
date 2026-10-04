// #4398: an original helper PS may enable unused center IJ before its consumed POS_X.
// The genuine collector still needs the retained coefficient GS. Observe its actual first
// quad's numerical IJ after normal completion AND all attachment pixels; native fallback or
// an unmatched/unwritten varying cannot satisfy both rails. No fabricated guest coefficient.
#include "fixtures/fragment_raster_fixture.hpp"
#include "fixtures/fragment_draw_source.hpp"
#include "fixtures/render_runner.h"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {
namespace g = prosper::gpu;
namespace r = prosper::test;
namespace f = r::fragment_draw;
namespace h = r::fragment_raster;

#ifdef _WIN32
int stream_fd(FILE* stream) {
    return _fileno(stream);
}
int copy_fd(int fd) {
    return _dup(fd);
}
int replace_fd(int from, int to) {
    return _dup2(from, to);
}
int close_fd(int fd) {
    return _close(fd);
}
#else
int stream_fd(FILE* stream) {
    return fileno(stream);
}
int copy_fd(int fd) {
    return dup(fd);
}
int replace_fd(int from, int to) {
    return dup2(from, to);
}
int close_fd(int fd) {
    return close(fd);
}
#endif

// Keep observation artifacts on the process-owned test scratch disk, not gtest's Linux /tmp.
class StderrCapture {
public:
    explicit StderrCapture(const std::string& name) {
        file_ = std::fopen(prosper_test::test_scratch_file(name).c_str(), "w+b");
        if (!file_) return;
        std::fflush(stderr);
        saved_ = copy_fd(stream_fd(stderr));
        if (saved_ < 0 || replace_fd(stream_fd(file_), stream_fd(stderr)) < 0) {
            if (saved_ >= 0) close_fd(saved_);
            saved_ = -1;
        }
    }
    ~StderrCapture() {
        if (saved_ >= 0) restore();
        if (file_) std::fclose(file_);
    }
    bool active() const { return saved_ >= 0; }
    std::string finish() {
        if (!active()) return {};
        restore();
        std::rewind(file_);
        std::string result;
        char bytes[1024];
        while (const size_t count = std::fread(bytes, 1, sizeof(bytes), file_))
            result.append(bytes, count);
        return std::ferror(file_) ? std::string{} : result;
    }

private:
    void restore() {
        std::fflush(stderr);
        if (replace_fd(saved_, stream_fd(stderr)) < 0) std::abort();
        close_fd(saved_);
        saved_ = -1;
    }
    FILE* file_ = nullptr;
    int saved_ = -1;
};

constexpr const char* collector_prefix =
    "[fragment-draw-observe] draw=0 plane=collector capacity_wave=4294967295 word=";

std::vector<uint32_t> collector_sample(const std::string& log, uint32_t count) {
    std::vector<uint32_t> words(count);
    std::vector<bool> seen(count);
    std::istringstream lines(log);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.starts_with(collector_prefix)) continue;
        std::istringstream values(line.substr(std::strlen(collector_prefix)));
        uint32_t at = UINT32_MAX;
        if (!(values >> at)) return {};
        std::string token;
        while (values >> token) {
            uint32_t word = 0;
            const auto parsed =
                std::from_chars(token.data(), token.data() + token.size(), word, 16);
            if (token.size() != 8 || parsed.ec != std::errc{} ||
                parsed.ptr != token.data() + token.size() || at >= count || seen[at])
                return {};
            words[at] = word;
            seen[at++] = true;
        }
    }
    return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })
               ? words
               : std::vector<uint32_t>{};
}

TEST(FragmentCollectorGeometry, ActualCenterIjProducerSurvivesHelperPackingAndOrderedReplay) {
    // Independent parser controls: missing, duplicate and malformed words must not read as zero.
    std::ostringstream calibration;
    calibration << collector_prefix << 0;
    std::vector<uint32_t> expected_words;
    for (uint32_t word = 0; word < 48; ++word) {
        expected_words.push_back(word + 1);
        calibration << ' ' << std::hex << std::setw(8) << std::setfill('0') << word + 1;
    }
    const auto valid_log = calibration.str() + '\n';
    ASSERT_EQ(collector_sample(valid_log, 48), expected_words);
    EXPECT_TRUE(collector_sample(valid_log, 49).empty());
    EXPECT_TRUE(collector_sample(valid_log + valid_log, 48).empty());
    EXPECT_TRUE(collector_sample(valid_log + collector_prefix + "0 not-hex\n", 48).empty());

    const auto& context = r::render_vk_ctx();
    ASSERT_TRUE(context.ok && context.shader_int64_enabled && context.geometry_shader_enabled);
    ASSERT_TRUE(context.float_transport.explicit_nonfinite32());
    ASSERT_TRUE(r::fragment_draw_observation_enabled())
        << "CTest arms the diagnostic at process start";
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(context.phys, VK_FORMAT_R32G32B32A32_SFLOAT, &properties);
    ASSERT_TRUE(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);

    auto physical = h::context();
    for (auto& [reg, value] : physical)
        if (reg == prosper::agc::Pm4::SPI_PS_INPUT_ENA ||
            reg == prosper::agc::Pm4::SPI_PS_INPUT_ADDR)
            value = (1u << 1) | (1u << 8);   // actual center IJ occupies v0:v1, POS_X is v2
    g::DrawItem draw;
    ASSERT_TRUE(f::realize(draw, f::color_a, h::original(2), f::ieee_rsrc1, 15, physical));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    ASSERT_EQ(*draw.fragment_draw_inputs->raw_code, h::original(2));
    ASSERT_TRUE(draw.gs_words().empty()) << "the native original does not consume center IJ";
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared);
    const g::FragmentPacketDeviceContract device{
        reinterpret_cast<uintptr_t>(context.dev), context.shader_int64_enabled, false,
        g::FragmentPacketRasterDeviceContract{
            context.geometry_shader_enabled, context.detile_limits.maxVertexOutputComponents,
            context.detile_limits.maxGeometryInputComponents,
            context.detile_limits.maxGeometryOutputComponents,
            context.detile_limits.maxGeometryTotalOutputComponents,
            context.detile_limits.maxGeometryOutputVertices,
            context.detile_limits.maxGeometryShaderInvocations,
            context.detile_limits.maxFragmentInputComponents}};
    const auto plan =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, device, 4096);
    ASSERT_TRUE(plan);
    ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
    ASSERT_TRUE(plan->raster_launch_collection());
    const auto& collection = *plan->raster_launch_collection();
    ASSERT_FALSE(collection.geometry.empty());
    ASSERT_TRUE(collection.parameters.requires_geometry);
    ASSERT_EQ(collection.shape.fields.size(), 1u);
    ASSERT_EQ(collection.shape.fields[0].kind, g::RasterQuadFieldKind::SystemInterpolation);
    ASSERT_EQ(collection.shape.fields[0].index, 1u);
    ASSERT_EQ(collection.shape.fields[0].words, 2u);
    ASSERT_EQ(collection.shape.lane_words, 11u);
    ASSERT_EQ(collection.shape.record_words, 44u);
    // Opt-in retention of the actual emitted modules for SDK validation (no effect otherwise).
    f::retain_source(collection.geometry, "coefficient_gs");
    f::retain_source(plan->collect_words(), "collector_fs");

    // Same immutable plan cannot execute against another observed device profile or VS generation.
    std::string refusal;
    const auto before = g::fragment_draw_cache_stats();
    auto smaller = context;
    smaller.detile_limits.maxGeometryOutputVertices = 2;
    EXPECT_FALSE(r::FragmentDrawCollectGpuProgram::acquire(smaller, plan, draw.vs_shared, {}, false,
                                                           refusal));
    EXPECT_EQ(refusal, "fragment-draw-collector-enabled-raster-budget-mismatch");
    const auto foreign = std::make_shared<const std::vector<uint32_t>>(*draw.vs_shared);
    EXPECT_FALSE(
        r::FragmentDrawCollectGpuProgram::acquire(context, plan, foreign, {}, false, refusal));
    EXPECT_EQ(refusal, "fragment-draw-collector-producing-vertex-unavailable");
    EXPECT_EQ(g::fragment_draw_cache_stats().vk_object_create_calls, before.vk_object_create_calls);
    const auto collect = r::FragmentDrawCollectGpuProgram::acquire(context, plan, draw.vs_shared,
                                                                   {}, false, refusal);
    ASSERT_TRUE(collect) << refusal;
    EXPECT_EQ(g::fragment_draw_cache_stats().checked_shader_module_calls -
                  before.checked_shader_module_calls,
              3u)
        << "actual shipping VS, collector FS, and retained coefficient GS modules";

    r::BackendDraw original;
    original.vs_shared = draw.vs_shared;
    original.fs_shared = draw.fs_shared;
    original.vs = draw.vs;
    original.fs = draw.fs;
    original.fragment_draw_inputs = draw.fragment_draw_inputs;
    original.vcount = draw.vertex_count;
    original.vertex_offset = draw.vertex_offset;
    original.instance_count = 1;
    original.ps = &draw.ps;
    r::BackendColorTarget target;
    target.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    constexpr std::array<float, 4> seed{.125f, .25f, .5f, 1.f};
    uint32_t frame = 0;
    for (uint32_t width : {f::width, f::width - 1, f::width}) {
        const auto observations = r::fragment_draw_observation_stats();
        StderrCapture capture("collector-center-" + std::to_string(frame++) + ".log");
        ASSERT_TRUE(capture.active());
        const auto raw = r::render_draws_rgba({original}, width, f::height, nullptr, seed.data(),
                                              false, &target);
        const auto log = capture.finish();
        std::fwrite(log.data(), 1, log.size(),
                    stderr);   // preserve complete raw receipts outside capture
        const auto stats = r::fragment_draw_backend_stats();
        EXPECT_EQ(stats.planned, 1u);
        EXPECT_EQ(stats.recorded, 1u) << "real transaction, never native fallback";
        EXPECT_EQ(stats.refused, 0u);
        const auto completed = r::fragment_draw_observation_stats();
        EXPECT_EQ(completed.recorded, observations.recorded + 1);
        EXPECT_EQ(completed.reported, observations.reported + 1);
        EXPECT_EQ(completed.unavailable, observations.unavailable);
        std::fprintf(stderr,
                     "[fragment-draw-gpu] original_transactions=%llu recorded=%llu extent=%ux%u "
                     "raw_bytes=%zu\n",
                     static_cast<unsigned long long>(stats.planned),
                     static_cast<unsigned long long>(stats.recorded), width, f::height, raw.size());
        ASSERT_EQ(raw.size(), size_t(width) * f::height * sizeof(seed));
        for (uint32_t y = 0; y < f::height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                std::array<float, 4> actual{};
                std::memcpy(actual.data(), raw.data() + (size_t(y) * width + x) * sizeof(actual),
                            sizeof(actual));
                for (float channel : actual)
                    EXPECT_EQ(channel, float((x & ~1u) + 1) + .5f) << "pixel=" << x << ',' << y;
            }
        const auto sample = collector_sample(log, 48);
        ASSERT_EQ(sample.size(), 48u) << "complete actual collector first quad must be observed";
        ASSERT_GT(sample[0], 0u);
        EXPECT_EQ(sample[1], 0u);
        ASSERT_EQ(sample[2], collection.shape.record_words);
        ASSERT_EQ(sample[3], g::kRasterQuadMagic);
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const uint32_t row = 4 + lane * collection.shape.lane_words;
            const float x = std::bit_cast<float>(sample[row + 5]);
            const float y = std::bit_cast<float>(sample[row + 6]);
            ASSERT_TRUE(std::isfinite(x) && std::isfinite(y));
            ASSERT_GT(x, 0.f);
            ASSERT_GT(y, 0.f);
            EXPECT_EQ(x - std::floor(x), .5f);
            EXPECT_EQ(y - std::floor(y), .5f);
            EXPECT_EQ(sample[row + 4], 0u) << "GS preserves the actual original primitive identity";
            const float i = std::bit_cast<float>(sample[row + 9]);
            const float j = std::bit_cast<float>(sample[row + 10]);
            ASSERT_TRUE(std::isfinite(i) && std::isfinite(j));
            // Independent geometry oracle: original vertices (-1,-1),(3,-1),(-1,3), W=1,
            // normal positive full-target viewport. Do not derive expected IJ from emitted GS.
            EXPECT_NEAR(i, x / (2.f * width), 1e-6f) << "actual center I lane=" << lane;
            EXPECT_NEAR(j, y / (2.f * f::height), 1e-6f) << "actual center J lane=" << lane;
        }
    }
    EXPECT_EQ(g::fragment_draw_cache_stats().collector_cold_builds,
              before.collector_cold_builds + 1)
        << "same retained collector/GS pipeline remains warm across changing framebuffer extents";
}
}   // namespace
