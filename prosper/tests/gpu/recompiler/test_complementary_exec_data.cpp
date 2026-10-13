// Complete writes under A and parent&~A cover the saved parent. Neither an absent source in one
// half nor a register-file placeholder may authenticate that whole-parent DATA at a fresh compare.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;

std::vector<uint32_t> partition(bool first = true, bool second = true) {
    return {
        0xbe84047eu, // save actual parent EXEC in s[4:5]
        0x7dea0080u, // A = parent & (genuine local-ID v0 != 0)
        0xbf880001u, // an empty A skips its writer, without creating a missing active lane
        first ? 0x7e040281u : 0x7e040309u,
        0x8afe7e04u, // B = parent & ~A
        0xbf880001u, // an empty B skips its writer
        second ? 0x7e040282u : 0x7e040309u,
        0xbefe0404u, // restore the same parent value
        0x7d280480u, // fresh predicate consumes the entire parent word v2
        0xbf810000u,
    };
}

struct Result {
    std::vector<uint32_t> module;
    std::string reject;
};

Result compile(const std::vector<uint32_t>& code) {
    ComputeShaderConfig config;
    config.user_sgprs.assign(1, 0);
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.force_exchange_dispatcher = true;
    prosper::test::complete_compute_launch(config);
    static uint64_t next = 0x5066000100ull;
    const uint64_t address = next++;
    auto module = recompile_compute(code.data(), code.size(), nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}

void dispatcher(const Result& result) {
    ASSERT_FALSE(result.module.empty()) << result.reject;
    bool found = false;
    for (std::size_t word = 5; word < result.module.size();) {
        const uint32_t count = result.module[word] >> 16;
        ASSERT_NE(count, 0u);
        ASSERT_LE(word + count, result.module.size());
        found |= (result.module[word] & 0xffffu) == 251; // OpSwitch
        word += count;
    }
    EXPECT_TRUE(found);
}

void missing(const Result& result, uint32_t pc = 8) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reject.find("mode=unresolved-operand"), std::string::npos) << result.reject;
    EXPECT_NE(result.reject.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reject;
}
} // namespace

TEST(ComplementaryExecData, BothActiveHalvesDefineTheRestoredParent) {
    dispatcher(compile(partition()));
}

TEST(ComplementaryExecData, MissingFirstHalfCannotDefineTheRestoredParent) {
    missing(compile(partition(false, true)));
}

TEST(ComplementaryExecData, MissingSecondHalfCannotDefineTheRestoredParent) {
    missing(compile(partition(true, false)));
}

TEST(ComplementaryExecData, MissingBothHalvesCannotDefineTheRestoredParent) {
    missing(compile(partition(false, false)));
}

namespace {
std::vector<uint32_t> scalar_first_half(bool alternate_genuine) {
    return {
        0xbe84047eu, 0x7dea0080u, 0xbf060000u,
        0xbf840002u, // independent genuine SCC, not an EXEC-empty condition
        0x7e040281u, 0xbf820001u, alternate_genuine ? 0x7e040281u : 0x7e040309u,
        0x8afe7e04u, 0xbf880001u, 0x7e040282u,
        0xbefe0404u, 0x7d280480u, 0xbf810000u,
    };
}
} // namespace

TEST(ComplementaryExecData, BothScalarPredecessorsCanSupplyTheFirstHalf) {
    dispatcher(compile(scalar_first_half(true)));
}

TEST(ComplementaryExecData, ScalarBranchCannotInventAnEmptyFirstHalf) {
    missing(compile(scalar_first_half(false)), 11);
}

TEST(ComplementaryExecData, Andn2SaveexecCoversTheSameSavedParent) {
    auto code = partition();
    code[4] = 0xbe862704u; // EXEC = saved parent & ~old EXEC; s[6:7] receives old EXEC
    dispatcher(compile(code));
}

TEST(ComplementaryExecData, Andn2SaveexecCannotReplaceAMissingFirstHalf) {
    auto code = partition(false, true);
    code[4] = 0xbe862704u;
    missing(compile(code));
}

TEST(ComplementaryExecData, PartialSecondWriterCannotDefineTheRestoredParent) {
    auto code = partition();
    code[5] = 0xbf880002u; // skip the entire two-dword SDWA writer when B is empty
    code[6] = 0x7e0402f9u;
    code.insert(code.begin() + 7, 0x00851480u); // WORD_0 / UNUSED_PRESERVE
    missing(compile(code), 9);
}

namespace {
std::vector<uint32_t> partition_backedge(bool genuine) {
    auto code = partition();
    code.resize(8); // complete both halves and restore the parent before the loop header
    code.insert(code.end(),
                {
                    0xbe860380u, // genuine scalar counter s6=0
                    0x7d280480u, // header: compare current v2 under the restored parent
                    genuine ? 0x7e040281u : 0x7e040309u,
                    0x80068106u,
                    0xbf0a8206u,
                    0xbf85fffbu, // counter++; counter<2; backedge -> compare
                    0xbf810000u,
                });
    return code;
}
} // namespace

TEST(ComplementaryExecData, BackedgeRetainsGenuineCurrentReplacement) {
    dispatcher(compile(partition_backedge(true)));
}

TEST(ComplementaryExecData, BackedgeCannotReviveAnEarlierPartitionWord) {
    missing(compile(partition_backedge(false)), 9);
}
