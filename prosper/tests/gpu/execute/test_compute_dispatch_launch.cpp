// Raw dispatch controls are part of physical entry-mask provenance. Normalized extents lose the
// independent partial-TG flag and register halves, so they cannot grant original guest EXEC words.
#include "fixtures/portable_entry_exec.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/execute/compute_dispatch_launch.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::portable_entry_exec;
namespace {
GpuState::Dispatch packet(bool indirect = false) {
    auto state = std::make_shared<GpuState>();
    state->sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_X] = 31u << 16u | 64u;
    state->sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_Y] = 1;
    state->sh[prosper::agc::Pm4::COMPUTE_NUM_THREAD_Z] = 1;
    GpuState::Dispatch d;
    d.modifier = 1;
    d.threads_x = 4;
    d.threads_y = d.threads_z = 1;
    d.state = state;
    d.indirect = indirect;
    d.indirect_args_addr = indirect ? 0x1000u : 0u;
    return d;
}

struct ComputeShaderBlob {
    AgcShaderHeader header;
    ShaderReg registers[2];
};

bool register_compute(const uint32_t* words, size_t dwords, ComputeShaderBlob& blob) {
    prosper::register_agc_hle();
    const auto create = prosper::Hle::lookup("f3dg2CSgRKY");
    if (!create) return false;
    blob.registers[0] = {prosper::agc::Pm4::COMPUTE_PGM_LO, 0};
    blob.registers[1] = {prosper::agc::Pm4::COMPUTE_PGM_HI, 0};
    blob.header = {};
    blob.header.file_header = 0x34333231u;
    blob.header.version = 0x18;
    blob.header.sh_registers =
        reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(blob.registers) -
                                      reinterpret_cast<uintptr_t>(&blob.header.sh_registers));
    blob.header.shader_size = static_cast<uint32_t>(dwords * sizeof(uint32_t));
    blob.header.num_sh_registers = 2;
    void* registered = nullptr;
    return create(reinterpret_cast<uint64_t>(&registered), reinterpret_cast<uint64_t>(&blob.header),
                  reinterpret_cast<uint64_t>(words), 0, 0, 0) == 0 &&
           registered == &blob.header;
}
}   // namespace

TEST(ComputeDispatchLaunch, DirectPacketRetainsPackedWordsAndDimensions) {
    const auto launch = compute_dispatch_launch_state(packet());
    EXPECT_EQ(launch.known, ComputeDispatchLaunchState::kAll);
    EXPECT_EQ(launch.initiator, 1u);
    EXPECT_EQ(launch.num_thread, (std::array<uint32_t, 3>{31u << 16u | 64u, 1u, 1u}));
    EXPECT_EQ(launch.dimensions, (std::array<uint32_t, 3>{4u, 1u, 1u}));
}
TEST(ComputeDispatchLaunch, MissingSnapshotCannotAcquireRegisterPresence) {
    auto d = packet();
    d.state.reset();
    const auto launch = compute_dispatch_launch_state(d);
    EXPECT_EQ(launch.known,
              ComputeDispatchLaunchState::kInitiator | ComputeDispatchLaunchState::kDimX |
                  ComputeDispatchLaunchState::kDimY | ComputeDispatchLaunchState::kDimZ);
    EXPECT_EQ(launch.num_thread, (std::array<uint32_t, 3>{}));
}
TEST(ComputeDispatchLaunch, MissingRegisterIsUnknownEvenWhenResolverDefaultsOne) {
    auto d = packet();
    auto missing = std::make_shared<GpuState>(*d.state);
    missing->sh.erase(prosper::agc::Pm4::COMPUTE_NUM_THREAD_Y);
    d.state = missing;
    const auto launch = compute_dispatch_launch_state(d);
    EXPECT_EQ(launch.known,
              ComputeDispatchLaunchState::kAll & ~ComputeDispatchLaunchState::kNumThreadY);
    EXPECT_EQ(launch.num_thread[1], 0u);
}
TEST(ComputeDispatchLaunch, ActualExecutorRetainsProducingWordsAndIndependentAbsence) {
    // This no-resource program needs no entry-word specialization. The real executor must retain
    // its launch observations even when normalization defaults an absent register to one.
    alignas(256) static const uint32_t code[] = {0x7e000280u, 0xbf810000u};
    static ComputeShaderBlob blob;
    ASSERT_TRUE(register_compute(code, std::size(code), blob));
    clear_shader_recompile_cache();
    for (uint32_t variant = 0; variant < 3; ++variant) {
        auto d = packet();
        auto producing = std::make_shared<GpuState>(*d.state);
        for (const auto& reg : blob.registers) producing->sh[reg.offset] = reg.value;
        if (variant == 1) producing->sh.erase(prosper::agc::Pm4::COMPUTE_NUM_THREAD_Y);
        if (variant == 2) d.modifier |= 2;
        d.state = producing;
        GpuState state;
        state.dispatches.push_back(d);
        std::vector<OperationRealizationFailure> failures;
        const auto items = realize_compute_dispatches(state, 4930, &failures);
        ASSERT_TRUE(failures.empty()) << variant;
        ASSERT_EQ(items.size(), 1u) << variant;
        ASSERT_TRUE(items[0].recompile_config_available) << variant;
        const auto& config = items[0].recompile_config;
        EXPECT_EQ(config.dispatch_launch, compute_dispatch_launch_state(d)) << variant;
        EXPECT_EQ(config.local_y, 1u) << variant;
        EXPECT_EQ(compute_dispatch_has_complete_groups(config), variant == 0) << variant;
    }
    clear_shader_recompile_cache();
}
TEST(ComputeDispatchLaunch, UnresolvedAndResolvedIndirectArgumentsRemainUntrusted) {
    auto d = packet(true);
    for (bool resolved : {false, true}) {
        d.indirect = !resolved;
        const auto launch = compute_dispatch_launch_state(d);
        EXPECT_EQ(launch.known, ComputeDispatchLaunchState::kInitiator |
                                    ComputeDispatchLaunchState::kNumThreadX |
                                    ComputeDispatchLaunchState::kNumThreadY |
                                    ComputeDispatchLaunchState::kNumThreadZ);
        EXPECT_EQ(launch.dimensions, (std::array<uint32_t, 3>{}));
    }
}
TEST(ComputeDispatchLaunch, CompleteLocalGroupDoesNotImplyCompleteGuestWave) {
    const auto c = fixture::config(65);
    EXPECT_TRUE(compute_dispatch_has_complete_groups(c));
    EXPECT_TRUE(fixture::compile(fixture::program(65), c).empty());
}
TEST(ComputeDispatchLaunch, IndependentPartialFlagInvalidatesOtherwiseIdenticalGeometry) {
    auto c = fixture::config(64);
    c.dispatch_launch.num_thread[0] |= 31u << 16u;
    ASSERT_TRUE(compute_dispatch_has_complete_groups(c));
    c.dispatch_launch.initiator |= 2;
    EXPECT_FALSE(compute_dispatch_has_complete_groups(c));
}
TEST(ComputeDispatchLaunch, Wave32HasCompleteGroupsButCannotGrantWave64Words) {
    auto c = fixture::config(64);
    c.wave_size = 32;
    c.dispatch_launch.initiator |= 1u << 15u;
    EXPECT_TRUE(compute_dispatch_has_complete_groups(c));
    c.wave_size = 64;
    EXPECT_FALSE(compute_dispatch_has_complete_groups(c));
}
// ADR 0006: the request's switch-type choices change the emitted module, so they are part of the key;
// otherwise a module compiled under one request would be served for another.
TEST(ComputeDispatchLaunch, CacheSeparatesRequestChoices) {
    auto config = fixture::config();
    ShaderCompileKey unset;
    unset.has_compute_config = true;
    populate_compute_shader_key(unset, config);
    config.choices.set(CompilerChoice::MimgSoft, 1);
    ShaderCompileKey soft;
    soft.has_compute_config = true;
    populate_compute_shader_key(soft, config);
    config.choices.set(CompilerChoice::MimgSoft, 0);
    ShaderCompileKey hard;
    hard.has_compute_config = true;
    populate_compute_shader_key(hard, config);
    EXPECT_FALSE(unset == soft);
    EXPECT_FALSE(soft == hard) << "set to 1 and set to 0 are different requests";
    EXPECT_FALSE(unset == hard) << "an unset entry (live read) is not the same request as 0";
    EXPECT_NE(ShaderCompileKeyHash::compute(soft), ShaderCompileKeyHash::compute(hard));
    EXPECT_NE(ShaderCompileKeyHash::compute(unset), ShaderCompileKeyHash::compute(soft));
    auto same = soft;
    EXPECT_TRUE(same == soft);
    EXPECT_EQ(ShaderCompileKeyHash::compute(same), ShaderCompileKeyHash::compute(soft));
}
TEST(ComputeDispatchLaunch, CacheSeparatesAllRawWordsAndPresenceBits) {
    ShaderCompileKey a;
    a.has_compute_config = true;
    populate_compute_shader_key(a, fixture::config());
    const auto hash = ShaderCompileKeyHash::compute(a);
    for (uint32_t field = 0; field < 8; ++field) {
        auto b = a;
        if (field == 0) b.compute_dispatch_launch.known ^= 1;
        if (field == 1) b.compute_dispatch_launch.initiator ^= 2;
        if (field >= 2 && field < 5) ++b.compute_dispatch_launch.num_thread[field - 2];
        if (field >= 5) ++b.compute_dispatch_launch.dimensions[field - 5];
        EXPECT_FALSE(a == b) << field;
        EXPECT_NE(hash, ShaderCompileKeyHash::compute(b)) << field;
    }
    auto b = a;
    b.compute_dispatch_launch.initiator |= uint64_t{1} << 32u;
    EXPECT_FALSE(a == b);
    EXPECT_NE(hash, ShaderCompileKeyHash::compute(b));
}
TEST(ComputeDispatchLaunch, WarmCacheCannotReuseFullMaskForChangedRawControl) {
    clear_shader_recompile_cache();
    const auto p = fixture::program(64);
    const auto resources = fixture::table(64);
    auto c = fixture::config(64);
    const auto first = recompile_compute_shader_cached(p.data(), p.size(), &resources, c);
    ASSERT_FALSE(first.empty());
    const auto warm = recompile_compute_shader_cached(p.data(), p.size(), &resources, c);
    EXPECT_EQ(first, warm);
    auto stats = shader_recompile_cache_stats();
    EXPECT_EQ(stats.misses, 1u);
    EXPECT_EQ(stats.hits, 1u);
    c.dispatch_launch.initiator |= 2;
    EXPECT_TRUE(recompile_compute_shader_cached(p.data(), p.size(), &resources, c).empty());
    c.dispatch_launch = {};
    EXPECT_TRUE(recompile_compute_shader_cached(p.data(), p.size(), &resources, c).empty());
    stats = shader_recompile_cache_stats();
    EXPECT_EQ(stats.misses, 3u);
    EXPECT_EQ(stats.hits, 1u);
    clear_shader_recompile_cache();
}

// The producer helper must remain wired at the real register-file realization boundary.
TEST(ComputeDispatchLaunch, ActualExecutorRetainsIndependentFp16OverflowKnownness) {
    alignas(256) static const uint32_t code[] = {0x7e000280u, 0xbf810000u};
    static ComputeShaderBlob blob;
    ASSERT_TRUE(register_compute(code, std::size(code), blob));
    clear_shader_recompile_cache();
    for (const auto expected :
         {Fp16OverflowMode{}, Fp16OverflowMode{true, false}, Fp16OverflowMode{true, true}}) {
        auto dispatch = packet();
        auto producing = std::make_shared<GpuState>(*dispatch.state);
        for (const auto& reg : blob.registers) producing->sh.set(reg.offset, reg.value);
        if (expected.available)
            producing->sh.set(prosper::agc::Pm4::COMPUTE_PGM_RSRC1,
                              expected.saturate ? 1u << 26 : 0u);
        dispatch.state = producing;
        GpuState state;
        state.dispatches.push_back(dispatch);
        std::vector<OperationRealizationFailure> failures;
        const auto items = realize_compute_dispatches(state, 4096, &failures);
        ASSERT_TRUE(failures.empty());
        ASSERT_EQ(items.size(), 1u);
        ASSERT_TRUE(items[0].recompile_config_available);
        EXPECT_EQ(items[0].recompile_config.fp16_overflow, expected);
        EXPECT_EQ(items[0].recompile_config.compute_pgm_rsrc1, expected.saturate ? 1u << 26 : 0u);
        auto captured_items = items;
        captured_items[0].command_order = 1;
        captured_items[0].dispatch_index = 0;
        GpuCaptureMetadata metadata;
        metadata.width = metadata.height = 1;
        GpuCaptureFile capture, decoded;
        std::string error;
        const auto reader = [&](uint64_t address, uint8_t* out, size_t count) {
            const uint64_t base = reinterpret_cast<uint64_t>(code);
            if (address < base || address - base >= sizeof(code)) return size_t{0};
            const size_t offset = size_t(address - base);
            const size_t size = std::min(count, sizeof(code) - offset);
            std::memcpy(out, reinterpret_cast<const uint8_t*>(code) + offset, size);
            return size;
        };
        ASSERT_TRUE(capture_submit_items({}, captured_items,
                                         {{SubmitOperationKind::Dispatch, 0, 1}}, metadata, reader,
                                         capture, error))
            << error;
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
        ASSERT_TRUE(deserialize_gpu_capture(bytes, decoded, error)) << error;
        ASSERT_EQ(decoded.computes.size(), 1u);
        EXPECT_EQ(decoded.computes[0].recompile_config.fp16_overflow, expected);
        GpuReplayFrame replay;
        ASSERT_TRUE(materialize_gpu_replay(decoded, replay, error)) << error;
        ASSERT_EQ(replay.computes.size(), 1u);
        EXPECT_EQ(replay.computes[0].recompile_config.fp16_overflow, expected);
        if (expected.available) {
            ASSERT_EQ(decoded.format_version, 78u);
            // v78: empty graphics counts, one compute fact, then empty failure count.
            bytes.resize(bytes.size() - 18);
            for (unsigned i = 0; i < 4; ++i) bytes.at(8 + i) = uint8_t(77u >> (8 * i));
            ASSERT_TRUE(deserialize_gpu_capture(bytes, decoded, error)) << error;
            EXPECT_EQ(decoded.computes[0].recompile_config.fp16_overflow, Fp16OverflowMode{});
            EXPECT_EQ(decoded.computes[0].recompile_config.compute_pgm_rsrc1,
                      expected.saturate ? 1u << 26 : 0u);
        }
    }
    clear_shader_recompile_cache();
}
