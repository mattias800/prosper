// Observed CS overflow state must survive missing raw words, cache reuse, and failed compiles.
// This is a metadata contract; the fixtures never execute GPU arithmetic.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/shader_cache_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr std::array<Fp16OverflowMode, 3> kModes{{{}, {true, false}, {true, true}}};
constexpr uint32_t kRsrc1 = prosper::agc::Pm4::COMPUTE_PGM_RSRC1;

std::array<ShaderCompileKey, kModes.size()> mode_keys() {
    std::array<ShaderCompileKey, kModes.size()> keys;
    for (size_t i = 0; i < kModes.size(); ++i) {
        auto& key = keys[i];
        key.stage = ShaderProgramStage::Compute;
        key.has_compute_config = true;
        key.code = std::make_shared<const std::vector<uint32_t>>(
            std::initializer_list<uint32_t>{0x7e000280u, 0xbf810000u});
        key.code_hash = hash_shader_code(*key.code);
        ComputeShaderConfig config;
        config.compute_pgm_rsrc1 = 0;
        config.fp16_overflow = kModes[i];
        populate_compute_shader_key(key, config);
        key.cached_hash = ShaderCompileKeyHash::compute(key);
    }
    return keys;
}
} // namespace

TEST(ComputeFp16Overflow, RawZeroDoesNotProveAnObservation) {
    RegisterFile registers;
    ComputeShaderConfig config;
    populate_compute_launch_rsrc1(config, registers);
    EXPECT_EQ(config.compute_pgm_rsrc1, 0u) << "preserve the historical missing raw zero";
    EXPECT_EQ(config.fp16_overflow, Fp16OverflowMode{});
    registers.set(kRsrc1, 0);
    populate_compute_launch_rsrc1(config, registers);
    EXPECT_EQ(config.compute_pgm_rsrc1, 0u);
    EXPECT_EQ(config.fp16_overflow, (Fp16OverflowMode{true, false}));
    registers.set(kRsrc1, (1u << 26) | 0x12345u);
    populate_compute_launch_rsrc1(config, registers);
    EXPECT_EQ(config.compute_pgm_rsrc1, (1u << 26) | 0x12345u);
    EXPECT_EQ(config.fp16_overflow, (Fp16OverflowMode{true, true}));
    registers.erase(kRsrc1);
    registers.set(prosper::agc::Pm4::SPI_SHADER_PGM_RSRC1_PS, 1u << 29);
    populate_compute_launch_rsrc1(config, registers);
    EXPECT_EQ(config.compute_pgm_rsrc1, 0u);
    EXPECT_EQ(config.fp16_overflow, Fp16OverflowMode{})
        << "neither stale nor PS state is CS authority";
}

TEST(ComputeFp16Overflow, RawBitWithoutPresenceAndWrongStageBitsGrantNothing) {
    EXPECT_EQ(compute_fp16_overflow_from_rsrc1(false, 1u << 26), Fp16OverflowMode{});
    EXPECT_EQ(compute_fp16_overflow_from_rsrc1(true, 0), (Fp16OverflowMode{true, false}));
    EXPECT_EQ(compute_fp16_overflow_from_rsrc1(true, 1u << 26), (Fp16OverflowMode{true, true}));
    for (const auto bit : {29u, 30u, 31u})
        EXPECT_EQ(compute_fp16_overflow_from_rsrc1(true, 1u << bit),
                  (Fp16OverflowMode{true, false}))
            << "other stage's bit " << bit;
    EXPECT_FALSE((Fp16OverflowMode{false, true}).canonical());
}

TEST(ComputeFp16Overflow, EqualityAndBothHashesDistinguishThreeProducingStates) {
    const auto keys = mode_keys();
    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(keys[i].compute_fp16_overflow, kModes[i]);
        for (size_t j = 0; j < keys.size(); ++j) {
            EXPECT_EQ(keys[i] == keys[j], i == j) << i << ',' << j;
            EXPECT_EQ(ShaderCompileKeyHash::compute_impl<true>(keys[i]) ==
                          ShaderCompileKeyHash::compute_impl<true>(keys[j]),
                      i == j)
                << "word hash " << i << ',' << j;
            EXPECT_EQ(ShaderCompileKeyHash::compute_impl<false>(keys[i]) ==
                          ShaderCompileKeyHash::compute_impl<false>(keys[j]),
                      i == j)
                << "byte hash " << i << ',' << j;
        }
    }
}

TEST(ComputeFp16Overflow, EqualBytesReuseOnlyTheirOwnProducingKey) {
    const auto keys = mode_keys();
    std::unordered_map<ShaderCompileKey, size_t, ShaderCompileKeyHash> entries;
    for (size_t i = 0; i < keys.size(); ++i) entries.emplace(keys[i], i);
    ASSERT_EQ(entries.size(), kModes.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        auto repeated = keys[i];
        repeated.code = std::make_shared<const std::vector<uint32_t>>(*keys[i].code);
        repeated.cached_hash = ShaderCompileKeyHash::compute(repeated);
        const auto hit = entries.find(repeated);
        ASSERT_NE(hit, entries.end());
        EXPECT_EQ(hit->second, i);
    }
}

TEST(ComputeFp16Overflow, WarmSuccessfulAndFailedCompilesRetainSeparateIdentities) {
    const std::array<std::vector<uint32_t>, 2> programs{{
        {0x7e000280u, 0xbf810000u}, // v_mov_b32 v0, 0; s_endpgm
        {0xc8000100u, 0xbf810000u}, // interpolation has no compute lowering
    }};
    for (size_t program = 0; program < programs.size(); ++program) {
        clear_shader_recompile_cache();
        std::array<uint64_t, kModes.size()> identities{};
        std::vector<uint32_t> original;
        for (size_t state = 0; state < kModes.size(); ++state) {
            ComputeShaderConfig config;
            config.compute_pgm_rsrc1 = 0;
            config.fp16_overflow = kModes[state];
            const auto& code = programs[program];
            const auto compiled = recompile_compute_shader_cached(code.data(), code.size(), nullptr,
                                                                  config, &identities[state]);
            EXPECT_EQ(compiled.empty(), program == 1) << "success/failure control " << program;
            if (state == 0) original = compiled;
            EXPECT_EQ(compiled, original) << "metadata alone changes no arithmetic or refusal";
            ASSERT_NE(identities[state], 0u);
            uint64_t repeat_identity = 0;
            const auto repeat = recompile_compute_shader_cached(code.data(), code.size(), nullptr,
                                                                config, &repeat_identity);
            EXPECT_EQ(repeat, compiled);
            EXPECT_EQ(repeat_identity, identities[state]);
            for (size_t previous = 0; previous < state; ++previous)
                EXPECT_NE(identities[previous], identities[state]);
        }
        const auto stats = shader_recompile_cache_stats();
        EXPECT_EQ(stats.misses, kModes.size());
        EXPECT_EQ(stats.hits, kModes.size());
        EXPECT_EQ(stats.entries, kModes.size());
    }
    clear_shader_recompile_cache();
}
