// #5047: a test-only flat-load offset must be supplied by a real instruction. Missing words
// still reject at the fresh carry producer, before an unrelated descriptor check can hide it.
#include "fixtures/compute_coverage_inputs.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <array>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace inputs = prosper::test::coverage_inputs;

struct Compilation {
    std::vector<uint32_t> module;
    std::string reason;
};
Compilation compile(const uint32_t* code, std::size_t count, uint64_t address, bool window) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(16);
    const auto analysis = analyze_flat_loads(code, count, 16);
    EXPECT_TRUE(analysis.any && analysis.all_resolved);
    ShaderResourceTable table;
    if (window && !analysis.loads.empty()) {
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.binding = 4;
        resource.gpu_addr = 0x1000000000ull;
        resource.size = 4096;
        resource.fetch_pc = analysis.loads.front().load_pc;
        resource.flat_base_sgpr = static_cast<uint32_t>(analysis.loads.front().base_sgpr);
        table.resources.push_back(resource);
    }
    auto module = recompile_compute(code, count, window ? &table : nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

Compilation compile_consumer(const uint32_t* code, std::size_t count, uint32_t wave,
                             bool dispatcher = false, bool native = true,
                             std::size_t user_words = 0) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(user_words);
    config.wave_size = wave;
    config.native_subgroup_size = native ? wave : 0;
    if (wave == 64) {
        config.local_x = config.threads_x = 64;
        prosper::test::complete_compute_launch(config);
    }
    ShaderResourceTable table;
    if (dispatcher) {
        ShaderResource resource;
        resource.cls = ResourceClass::VertexBuffer;
        resource.binding = 3;
        resource.sgpr_base = 8;
        resource.stride = 16;
        resource.format = DataFormat::Float32;
        resource.num_components = 4;
        table.resources.push_back(resource);
    }
    static uint64_t next_address = 0x5047001000ull;
    const auto address = next_address++;
    auto module = recompile_compute(code, count, dispatcher ? &table : nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

// #5174: a consumer whose producer is simply absent reads a VGPR compute never loads and nothing
// wrote. Hardware holds an earlier wave's value there, so the guard admits it as arbitrary; these
// arms were refusals before that decision made them over-refusals. #5185 extends that to values
// computed from such words, so relaying one (v_mov from v31) is admitted too. Each test keeps a
// twin with a HARD producer that reaches the guard at the actual consumer: the old LDS word an
// atomic returns at the never-written address v31 (ds_add_rtn_u32). An atomic at an arbitrary
// address stays hard (#5193); a plain LDS read there would be arbitrary.
constexpr std::array<uint32_t, 2> lds_from_v31(uint32_t vgpr) {
    return {0xd8800000u, (vgpr << 24) | 31u};
}

void missing_predicate(const Compilation& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("fabricated-fresh-mask-source"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reason;
}

template <std::size_t N>
auto with_dispatcher(const std::array<uint32_t, N>& prefix) {
    std::vector<uint32_t> result(prefix.begin(), prefix.end());
    result.insert(result.end(), std::begin(inputs::dispatch_tail), std::end(inputs::dispatch_tail));
    return result;
}
} // namespace

TEST(ComputeCoverageInputs, FlatOffsetHasAnActualProducer) {
    const auto result = compile(inputs::flat_with_offset.data(), inputs::flat_with_offset.size(),
                                0x5047000100ull, true);
    EXPECT_FALSE(result.module.empty()) << result.reason;
}

TEST(ComputeCoverageInputs, MissingOffsetRefusesAtTheActualCarry) {
    const auto absent =
        compile(inputs::flat_consumer.data(), inputs::flat_consumer.size(), 0x5047000200ull, true);
    EXPECT_FALSE(absent.module.empty()) << absent.reason;
    const auto relayed = inputs::with_inputs(inputs::flat_consumer, lds_from_v31(6));
    const auto result = compile(relayed.data(), relayed.size(), 0x5047000201ull, true);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("fabricated-fresh-mask-source"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=2 "), std::string::npos) << result.reason;
}

TEST(ComputeCoverageInputs, AnOffsetProducerDoesNotSupplyAMissingWindow) {
    const auto result = compile(inputs::flat_with_offset.data(), inputs::flat_with_offset.size(),
                                0x5047000300ull, false);
    EXPECT_TRUE(result.module.empty());
    EXPECT_EQ(result.reason.find("fabricated-fresh-mask-source"), std::string::npos)
        << result.reason;
}

TEST(ComputeCoverageInputs, AScalarCompareNeedsItsActualScalarWord) {
    const auto& supplied = inputs::wave32_half_mask;
    const auto& missing = inputs::wave32_half_mask_consumer;
    const auto positive = compile_consumer(supplied.data(), supplied.size(), 32);
    EXPECT_FALSE(positive.module.empty()) << positive.reason;
    missing_predicate(compile_consumer(missing.data(), missing.size(), 32), 0);
}

TEST(ComputeCoverageInputs, APhysicalVccVoteNeedsItsActualVectorWord) {
    const auto& supplied = inputs::wave32_vcc_gt_zero;
    const auto& missing = inputs::wave32_vcc_gt_zero_consumer;
    const auto positive = compile_consumer(supplied.data(), supplied.size(), 32);
    EXPECT_FALSE(positive.module.empty()) << positive.reason;
    const auto absent = compile_consumer(missing.data(), missing.size(), 32);
    EXPECT_FALSE(absent.module.empty()) << absent.reason;
    const auto relayed = inputs::with_inputs(missing, lds_from_v31(20));
    missing_predicate(compile_consumer(relayed.data(), relayed.size(), 32), 2);
}

TEST(ComputeCoverageInputs, SavedMaskMbcntKeepsItsRealCompareOnBothRoutes) {
    const auto& supplied = inputs::compute_structured_saved_mbcnt_b64;
    const auto& missing = inputs::compute_structured_saved_mbcnt_b64_consumer;
    for (const bool native : {true, false}) {
        const auto positive = compile_consumer(supplied.data(), supplied.size(), 64, false, native);
        EXPECT_FALSE(positive.module.empty()) << positive.reason;
        // v1 comes from s30, which no launch loads: hard on every route. (Without a producer v1
        // is never loaded at TIDIG_COMP_CNT 0, so arbitrary under #5185 on every route, #5039.)
        std::vector<uint32_t> hard{0x7e02021eu};   // v_mov_b32 v1, s30
        hard.insert(hard.end(), missing.begin(), missing.end());
        const auto negative = compile_consumer(hard.data(), hard.size(), 64, false, native);
        EXPECT_TRUE(negative.module.empty());
        // This dispatcher emits the guard diagnostic but its terminal record retains only the
        // unresolved operand. Pin that actual consumer, rather than accepting a later rejection.
        EXPECT_NE(negative.reason.find("mode=unresolved-operand"), std::string::npos)
            << negative.reason;
        EXPECT_NE(negative.reason.find("pc=3 words=7c020300 "), std::string::npos)
            << negative.reason;
    }
}

TEST(ComputeCoverageInputs, MaskNorNeedsItsActualVectorCompareSource) {
    const auto& supplied = inputs::mask_nor;
    const auto& missing = inputs::mask_nor_consumer;
    const auto positive = compile_consumer(supplied.data(), supplied.size(), 64);
    EXPECT_FALSE(positive.module.empty()) << positive.reason;
    const auto absent = compile_consumer(missing.data(), missing.size(), 64);
    EXPECT_FALSE(absent.module.empty()) << absent.reason;
    const auto relayed = inputs::with_inputs(missing, lds_from_v31(6));
    missing_predicate(compile_consumer(relayed.data(), relayed.size(), 64), 2);
}

TEST(ComputeCoverageInputs, SavedMaskNotKeepsGenuineSccBeforeTheMerge) {
    const auto supplied = with_dispatcher(inputs::compute_cfg_dispatch_saved_mask_not);
    const auto missing = with_dispatcher(inputs::compute_cfg_dispatch_saved_mask_not_consumer);
    const auto positive = compile_consumer(supplied.data(), supplied.size(), 32, true);
    EXPECT_FALSE(positive.module.empty()) << positive.reason;
    const auto negative = compile_consumer(missing.data(), missing.size(), 32, true);
    EXPECT_TRUE(negative.module.empty());
    EXPECT_NE(negative.reason.find("poisoned-scc"), std::string::npos) << negative.reason;
    EXPECT_NE(negative.reason.find("pc=3 "), std::string::npos) << negative.reason;
}

TEST(ComputeCoverageInputs, SavedMaskNotKeepsGenuineSccAcrossABlockBoundary) {
    // Isolate the saved-mask boundary. The longer legacy fixture separately checks the subsequent
    // dispatcher loop, whose emptied EXEC has an independent confinement-proof refusal.
    const auto supplied = inputs::with_inputs(std::array{0xbf810000u},
                                              inputs::compute_cfg_dispatch_saved_mask_not_boundary);
    const auto missing = inputs::with_inputs(
        std::array{0xbf810000u}, inputs::compute_cfg_dispatch_saved_mask_not_boundary_consumer);
    const auto positive = compile_consumer(supplied.data(), supplied.size(), 32);
    EXPECT_FALSE(positive.module.empty()) << positive.reason;
    const auto negative = compile_consumer(missing.data(), missing.size(), 32);
    EXPECT_TRUE(negative.module.empty());
    EXPECT_NE(negative.reason.find("poisoned-scc"), std::string::npos) << negative.reason;
    EXPECT_NE(negative.reason.find("pc=3 "), std::string::npos) << negative.reason;
}

namespace {
uint64_t next_x16_address = 0x5047002000ull;
template <typename Code>
Compilation compile_x16(const Code& code, std::size_t input_words, uint32_t wave = 64) {
    ComputeShaderConfig config;
    config.local_x = config.threads_x = 64;
    config.wave_size = config.native_subgroup_size = wave;
    config.user_sgprs.resize(20);
    prosper::test::complete_compute_launch(config);
    ShaderResourceTable table;
    for (const auto [binding, consumer_pc] : {std::pair{4u, 2u}, std::pair{5u, 4u}}) {
        ShaderResource resource;
        resource.cls = ResourceClass::Texture;
        resource.format = DataFormat::Float32;
        resource.num_components = 4;
        resource.binding = binding;
        resource.fetch_pc = consumer_pc + static_cast<uint32_t>(input_words);
        resource.img_dim = 1;
        resource.width = resource.height = 2;
        table.resources.push_back(resource);
    }
    const auto address = next_x16_address++;
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
}   // namespace

TEST(ComputeCoverageInputs, DescriptorLifetimeNeedsBothRealCompareInputsAndItsBranchCondition) {
    const auto positive =
        compile_x16(inputs::x16_wave_mask_lifetime, inputs::x16_wave_mask_inputs.size());
    EXPECT_FALSE(positive.module.empty()) << positive.reason;
    const auto missing_vectors =
        inputs::with_inputs(inputs::x16_wave_mask_consumer, std::array{0xbf068080u});
    const auto absent = compile_x16(missing_vectors, 1);
    EXPECT_FALSE(absent.module.empty()) << absent.reason;
    const auto lds = lds_from_v31(10);
    const auto relayed_vectors = inputs::with_inputs(
        inputs::x16_wave_mask_consumer, std::array{lds[0], lds[1], 0x7e240280u, 0xbf068080u});
    missing_predicate(compile_x16(relayed_vectors, 4), 12);
    const auto missing_scc_prefix =
        inputs::with_inputs(inputs::x16_wave_mask_consumer, std::array{0x7e140300u, 0x7e240280u});
    std::vector<uint32_t> missing_scc(missing_scc_prefix.begin(), missing_scc_prefix.end());
    // The Wave32 control explicitly replaces s5 before observing it, so the descriptor lifetime
    // cannot reject first and hide the missing condition at the earlier scalar branch.
    missing_scc.insert(missing_scc.begin() + 12, 0xbe850380u);
    const auto negative = compile_x16(missing_scc, 2, 32);
    EXPECT_TRUE(negative.module.empty());
    EXPECT_NE(negative.reason.find("poisoned-scc"), std::string::npos) << negative.reason;
    EXPECT_NE(negative.reason.find("pc=8 "), std::string::npos) << negative.reason;
    missing_scc.insert(missing_scc.begin(), 0xbf068080u);
    const auto restored_condition = compile_x16(missing_scc, 3, 32);
    EXPECT_FALSE(restored_condition.module.empty()) << restored_condition.reason;
}

TEST(ComputeCoverageInputs, AScalarDescriptorOffsetCompareNeedsTheDeclaredVectorInput) {
    const auto supplied =
        inputs::with_inputs(inputs::scalar_offset_compare, inputs::scalar_offset_input);
    const auto positive = compile_consumer(supplied.data(), supplied.size(), 64, false, true, 25);
    EXPECT_FALSE(positive.module.empty()) << positive.reason;
    const auto& missing = inputs::scalar_offset_compare;
    const auto absent = compile_consumer(missing.data(), missing.size(), 64, false, true, 25);
    EXPECT_FALSE(absent.module.empty()) << absent.reason;
    const auto relayed = inputs::with_inputs(missing, lds_from_v31(68));
    missing_predicate(compile_consumer(relayed.data(), relayed.size(), 64, false, true, 25), 2);
}

TEST(ComputeCoverageInputs, EachFreshMaskNeedsTheActualComparedWords) {
    // Keep the three actual comparisons. The legacy fixture separately checks their later SCC
    // intersection, which still has a production condition-proof refusal after these input repairs.
    std::array<uint32_t, 7> consumer{};
    for (std::size_t i = 0; i < 6; ++i) consumer[i] = inputs::mask_scc_consumer[i];
    consumer.back() = 0xbf810000u;
    const auto supplied = inputs::with_inputs(consumer, inputs::mask_scc_inputs);
    const auto positive = compile_consumer(supplied.data(), supplied.size(), 32);
    EXPECT_FALSE(positive.module.empty()) << positive.reason;
    missing_predicate(compile_consumer(consumer.data(), consumer.size(), 32), 1);
}
