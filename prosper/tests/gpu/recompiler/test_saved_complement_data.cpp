// Saving exact B=P&~A, narrowing for an inner region, then restoring unchanged B must retain
// the first-half DATA obligation under A. Complete genuine writes under both halves cover P;
// an unrelated restore or an incomplete half must never authenticate P's untouched lanes.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
struct Result {
    std::vector<uint32_t> module;
    std::string reject;
};

Result compile(const std::vector<uint32_t>& code) {
    ComputeShaderConfig config;
    config.user_sgprs.assign(1, 0);
    config.local_x = 64;
    config.wave_size = config.native_subgroup_size = 64;
    config.force_exchange_dispatcher = true;
    prosper::test::complete_compute_launch(config, 2);
    static uint64_t next = 0x5075000100ull;
    const auto address = next++;
    auto module = recompile_compute(code.data(), code.size(), nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

void dispatcher(const Result& result) {
    ASSERT_FALSE(result.module.empty()) << result.reject;
    bool found = false;
    for (std::size_t word = 5; word < result.module.size();) {
        const auto count = result.module[word] >> 16;
        ASSERT_NE(count, 0u);
        ASSERT_LE(word + count, result.module.size());
        found |= (result.module[word] & 0xffffu) == 251;
        word += count;
    }
    EXPECT_TRUE(found);
}

void missing(const Result& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reject.find("mode=unresolved-operand"), std::string::npos) << result.reject;
    EXPECT_NE(result.reject.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reject;
}

std::vector<uint32_t> saved_complement(bool first = true, bool second = true) {
    return {
        0xbe84047eu, // P is the actual producer EXEC, not a width-derived full-wave certificate
        0x7da40080u, // A=P & (local-ID==0)
        first ? 0x7e040281u : 0x7e040309u,
        0x8afe7e04u, // exact B=P & ~A
        0xbe86047eu, // save actual B before the nested EXEC change
        0x7da40081u, // C=B & (local-ID==1), a strict subset for the supplied 64-lane workgroup
        0xbefe0406u, // restore unchanged B, including its lanes outside C
        second ? 0x7e040282u : 0x7e040309u,
        0xbefe0404u,
        0x7d280480u,
        0xbf810000u,
    };
}
} // namespace

TEST(SavedComplementData, RestoringTheSavedExactComplementRetainsTheFirstHalfData) {
    dispatcher(compile(saved_complement()));
}

TEST(SavedComplementData, MissingFirstHalfCannotBeRepairedByTheSecondHalf) {
    missing(compile(saved_complement(false, true)), 9);
}

TEST(SavedComplementData, MissingSecondHalfCannotAuthenticateTheSavedParent) {
    missing(compile(saved_complement(true, false)), 9);
}

TEST(SavedComplementData, InnerSubsetWithoutRestoreCannotCoverTheSavedComplement) {
    auto code = saved_complement();
    code[6] = 0xbf800000u;
    missing(compile(code), 9);
}

TEST(SavedComplementData, AnUnrelatedSavedMaskCannotRestoreComplementMembership) {
    auto code = saved_complement();
    code.insert(code.begin() + 3, 0xbe88047eu); // retain A separately before selecting B
    code[7] = 0xbefe0408u; // restoring A again leaves B's lanes unwritten
    missing(compile(code), 10);
}
