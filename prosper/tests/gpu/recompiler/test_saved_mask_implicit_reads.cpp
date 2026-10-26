// Implicit SDST reads need the producer's physical mask words just like explicit DATA reads.
// Otherwise CMPK consumes missing storage, or an RMW instruction cannot compile at all.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;

ComputeShaderConfig config() {
    ComputeShaderConfig result;
    result.local_x = 64;
    result.wave_size = result.native_subgroup_size = 64;
    prosper::test::complete_compute_launch(result);
    return result;
}

Words program(uint32_t consumer, uint32_t peer = 63) {
    return {
        0x7d8400f9u, 0x06869480u | peer, // v_cmp_eq_u32_sdwa s[20:21], peer, v0
        consumer,
        0x8516968bu,              // s_cselect_b32 s22, 11, 22
        0x7e020216u,              // v_mov_b32 v1, s22
        0xe0702000u, 0x80020100u, // buffer_store_dword v1, v0, s[8:11] idxen
        0xbf810000u,
    };
}

struct Compilation {
    Words module;
    std::string reason;
};

Compilation compile(const Words& words, const ComputeShaderConfig& launch = config()) {
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    ShaderResourceTable table;
    table.resources.push_back(output);
    static std::atomic<uint64_t> next{0x5039000000ull};
    const uint64_t address = next.fetch_add(0x100, std::memory_order_relaxed);
    auto module = recompile_compute(words.data(), words.size(), &table, launch,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
} // namespace

TEST(SavedMaskImplicitReads, CompareObservesTheHighBallotWord) {
    constexpr uint32_t compare_high = 0xb2150000u; // s_cmpk_lg_i32 s21, 0
    for (uint32_t peer : {63u, 64u}) {
        const auto words = program(compare_high, peer);
        auto producer = rdna2_decode_one(words.data(), words.size());
        ASSERT_EQ(producer.dst.kind, OperandKind::SGPR);
        ASSERT_EQ(producer.dst.value, 20);
        ASSERT_EQ(producer.src[0].kind, OperandKind::InlineInt);
        ASSERT_EQ(producer.src[0].value, peer);
        const auto result = compile(words);
        ASSERT_FALSE(result.module.empty()) << result.reason;
        bpermute_oracle::Interpreter vm(result.module);
        vm.native_subgroup_width = 64;
        const auto actual = vm.run(64, Words(64, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(actual, Words(64, peer == 63 ? 11u : 22u));
    }
}

TEST(SavedMaskImplicitReads, ReadModifyWriteRequiresTheOldBallotWord) {
    for (uint32_t consumer : {0xb7940001u, 0xbe941d81u}) {
        // s_addk_i32 s20,1 / s_bitset1_b32 s20,1. Neither names SDST as a source.
        auto words = program(consumer);
        words[3] = 0xbf800000u;
        words[4] = 0x7e020281u;   // independent output; keep the implicit read the only DATA demand
        const auto result = compile(words);
        EXPECT_FALSE(result.module.empty()) << result.reason;
    }
}

TEST(SavedMaskImplicitReads, AddkOverflowObservesTheLowBallotWord) {
    // ISA 70648 section 12.2: ADDK sets SCC on signed overflow. INT_MIN - 1 overflows;
    // 0 - 1 does not. Moving the matching lane across the low/high boundary changes SCC.
    for (uint32_t peer : {31u, 32u}) {
        const auto result = compile(program(0xb794ffffu, peer));   // s_addk_i32 s20, -1
        ASSERT_FALSE(result.module.empty()) << result.reason;
        bpermute_oracle::Interpreter vm(result.module);
        vm.native_subgroup_width = 64;
        const auto actual = vm.run(64, Words(64, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(actual, Words(64, peer == 31 ? 11u : 22u));
    }
}

TEST(SavedMaskImplicitReads, MissingRawLaunchStillRefusesTheSameRead) {
    auto launch = config();
    launch.dispatch_launch = {};
    const auto result = compile(program(0xb2150000u), launch);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("saved-mask"), std::string::npos) << result.reason;
}

TEST(SavedMaskImplicitReads, PartialEntryStillRefusesTheSameRead) {
    auto launch = config();
    launch.dispatch_launch.initiator |= 2u;
    const auto result = compile(program(0xb2150000u), launch);
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("saved-mask"), std::string::npos) << result.reason;
}

TEST(SavedMaskImplicitReads, LateVccCopyCannotInventProducerParticipation) {
    // Keep the original legacy join's missing-participation shape as a negative control.
    const Words words{
        0x7e040280u, 0x7d840100u, 0xbf860001u, 0x7e040281u, 0x7d840100u, 0xbf870001u,
        0xbf82fffdu, 0xbe80038bu, 0xbe81038cu, 0xbf800000u, 0xbf07800cu, 0xbf840002u,
        0xbe84046au, 0xbf820001u, 0xbe840400u, 0xbf138004u, 0xbf810000u,
    };
    auto launch = config();
    launch.user_sgprs.resize(13);
    TerminalRejectCapture attempts;
    const auto result = compile(words, launch);
    EXPECT_TRUE(result.module.empty());
    // A later straight-line fallback replaces the final reason with its unrelated PC2 refusal.
    std::string rejected_joins;
    for (const auto& [tag, payload] : attempts.take())
        if (tag == "compute-cfg-reject") rejected_joins += payload;
    EXPECT_NE(rejected_joins.find("pc=15 reason=wave64-ambiguous-mask-read"), std::string::npos)
        << rejected_joins << result.reason;
}
