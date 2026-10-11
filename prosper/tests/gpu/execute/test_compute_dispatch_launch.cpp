// Raw dispatch controls are part of physical entry-mask provenance. Normalized extents lose the
// independent partial-TG flag and register halves, so they cannot grant original guest EXEC words.
#include "fixtures/portable_entry_exec.hpp"
#include "gpu/execute/compute_dispatch_launch.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
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
