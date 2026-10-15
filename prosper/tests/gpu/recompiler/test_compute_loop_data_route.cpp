// A confined preheader word can lose its owner at a structured EXEC-loop phi. Retrying the
// exact wave dispatcher must retain scalar wave-maximum state and real masked memory effects;
// missing DATA or backing remains a refusal. This is CPU compilation and a SPIR-V interpreter.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr uint32_t kNop = 0xbf800000u;
constexpr uint32_t kEnd = 0xbf810000u;
struct Program {
    std::vector<uint32_t> words;
    uint32_t header;
};
Program program() {
    return {{0xbe94047eu, // save real entry EXEC s[20:21]
             0x7daa0080u, // entry & (local ID != 0)
             0x7e020280u, // v1=0 under that EXEC
             0x7e040300u, // v2=local ID under that EXEC
             0xbe860380u, // wave scalar counter=0
             0x7e0e02c0u, // genuine v7=64 for matched backedge relays
             0x7da20501u, // header: CMPX_LT_U32 v1,v2
             0xbf880005u, // EXECZ -> saved entry restore pc13
             0x80068106u, // s6++ for the whole guest wave
             0xe0102000u, 0x80030100u, // output3[local ID]=v1, FORMAT_X
             0x4a020281u, // v1++ only in current EXEC
             0xbf82fff9u, // -> header pc6
             0xbefe0414u, // restore actual saved entry EXEC
             0x7e060206u, // v3=wave scalar counter
             0xe0702000u, 0x80020300u, // output2[local ID]=v3, DWORD
             kEnd},
            6};
}
ComputeShaderConfig config(bool force = false) {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0);
    c.tgid_x_en = true;
    c.force_exchange_dispatcher = force;
    prosper::test::complete_compute_launch(c);
    return c;
}
ShaderResourceTable table() {
    ShaderResourceTable t;
    for (uint32_t n = 0; n < 2; ++n) {
        ShaderResource r;
        r.cls = ResourceClass::ConstantBuffer;
        r.format = DataFormat::Uint32;
        r.num_components = 1;
        r.binding = 2 + n;
        r.sgpr_base = 8 + 4 * n;
        r.stride = 4;
        r.size = 256;
        t.resources.push_back(r);
    }
    return t;
}
struct Result {
    std::vector<uint32_t> module;
    std::string reason;
};
Result compile(const Program& p, const ComputeShaderConfig& c = config(),
               const ShaderResourceTable& t = table()) {
    static uint64_t next = 0x5140001000ull;
    const uint64_t address = next++;
    auto m = recompile_compute(p.words.data(), p.words.size(), &t, c,
                               {RecompileDiagnosticStage::Compute, address});
    if (!m.empty()) {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): test observer fixed before process start.
        if (const char* dir = std::getenv("PROSPER_COMPUTE_LOOP_DATA_SPV_DIRECTORY")) {
            const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
            std::ofstream out(std::filesystem::path(dir) / (std::string(test->name()) + "-" +
                                                            std::to_string(address) + ".spv"),
                              std::ios::binary);
            out.write(reinterpret_cast<const char*>(m.data()),
                      static_cast<std::streamsize>(m.size() * 4));
            EXPECT_TRUE(out.good());
        }
    }
    return {std::move(m), last_terminal_reject_reason(address)};
}
bool has_op(const std::vector<uint32_t>& m, uint32_t op) {
    for (size_t i = 5; i < m.size();) {
        const uint32_t n = m[i] >> 16;
        if (!n || n > m.size() - i) return false;
        if ((m[i] & 65535u) == op) return true;
        i += n;
    }
    return false;
}
void wave_maximum(const Result& r) {
    ASSERT_FALSE(r.module.empty()) << r.reason;
    ASSERT_TRUE(has_op(r.module, 251u)) << "exact synchronized dispatcher route";
    bpermute_oracle::Interpreter vm(r.module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config().user_sgprs;
    vm.extra_writable_bindings = {3};
    const auto values = vm.run_buffers(
        64,
        {{2, std::vector<uint32_t>(64, 0xdeadbeefu)}, {3, std::vector<uint32_t>(64, 0xdeadbeefu)}},
        2);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(values[lane], 63u)
            << "lane " << lane << " observes the wave maximum, including zero-trip lane 0";
    const auto stores = vm.writable_result(3);
    ASSERT_EQ(stores.size(), 64u);
    EXPECT_EQ(stores[0], 0xdeadbeefu) << "inactive zero-trip lane executes no buffer store";
    for (uint32_t lane = 1; lane < 64; ++lane)
        EXPECT_EQ(stores[lane], lane - 1u)
            << "lane " << lane << " retains its own final masked iteration";
}
// The old value of `vgpr` in every lane before the program: hard on every route. s30 is neither a
// user SGPR nor written by the programs here, so hardware never loads it. A never-written VGPR
// would not do: under #5185 it holds an arbitrary value hardware could hold, which a fresh
// predicate may read on any route (#5039). Shifts every pc by one.
Program hard_old(Program p, uint32_t vgpr) {
    p.words.insert(p.words.begin(), 0x7e00021eu | (vgpr << 17u));   // v_mov_b32 vN, s30
    ++p.header;
    return p;
}
void missing(const Result& r, uint32_t pc) {
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reason.find("fabricated-fresh-mask-source"), std::string::npos) << r.reason;
    EXPECT_NE(r.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos) << r.reason;
}
Program predecessor(bool both) {
    auto p = program();
    p.words.erase(p.words.begin() + 2);
    p.words.insert(p.words.begin() + 2, {0xbf068010u, 0xbf840002u, 0x7e020280u, 0xbf820001u,
                                         both ? 0x7e020280u : kNop, kNop});
    p.header += 5;
    return p;
}
Program memory_producer(bool address) {
    auto p = program();
    // Genuine full entry v1 precedes the narrowing; the load replaces its active generation.
    p.words.insert(p.words.begin() + 1, 0x7e020280u);
    p.words.erase(p.words.begin() + 3);
    p.words.insert(p.words.begin() + 3, {0xe0002000u, address ? 0x80010100u : 0x80010109u});
    p.header += 2;
    return p;
}
ShaderResourceTable input_table() {
    auto t = table();
    auto r = t.resources.front();
    r.binding = 4;
    r.sgpr_base = 4;
    t.resources.push_back(r);
    return t;
}
void cfg_missing(const Result& r, uint32_t pc, uint32_t word) {
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reason.find("cfg-recompile-reject"), std::string::npos) << r.reason;
    EXPECT_NE(r.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos) << r.reason;
    char hex[9];
    std::snprintf(hex, sizeof(hex), "%08x", word);
    EXPECT_NE(r.reason.find("words=" + std::string(hex)), std::string::npos) << r.reason;
}
} // namespace

TEST(ComputeLoopDataRoute, ClosedNativeLoopRetainsWaveScalarAndMaskedStores) {
    wave_maximum(compile(program()));
}
TEST(ComputeLoopDataRoute, ExplicitExactDispatcherCalibratesWaveMaximum) {
    wave_maximum(compile(program(), config(true)));
}
TEST(ComputeLoopDataRoute, UnwrittenPreheaderWordStaysMissing) {
    auto p = program();
    p.words[2] = kNop;
    p = hard_old(p, 1);
    missing(compile(p), p.header);
}
TEST(ComputeLoopDataRoute, FabricatedPreheaderRelayStaysMissing) {
    auto p = program();
    p.words[2] = 0x7e02021eu;   // v1 = s30, never loaded: hard on every route (#5039)
    missing(compile(p), p.header);
}
TEST(ComputeLoopDataRoute, MissingScalarBoundIsNotVectorData) {
    auto p = program();
    p.words[3] = 0x7e04021eu;
    missing(compile(p), p.header);
}
TEST(ComputeLoopDataRoute, UnwrittenBackedgeWordStaysMissing) {
    auto p = program();
    p.words[11] = 0x7e02021eu;   // v1 = s30, never loaded: hard on every route (#5039)
    missing(compile(p), p.header);
}
TEST(ComputeLoopDataRoute, ExplicitDispatcherDoesNotInventBackedgeData) {
    auto p = program();
    p.words[11] = 0x7e02021eu;   // v1 = s30, never loaded: hard on every route (#5039)
    const auto result = compile(p, config(true));
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(
        result.reason.find("cfg-recompile-reject mode=unresolved-operand pc=6 words=7da20501"),
        std::string::npos)
        << result.reason;
}
TEST(ComputeLoopDataRoute, WideningBackedgeDoesNotUseNarrowingRetry) {
    auto p = program();
    p.words[8] = 0xbefe04c1u;
    missing(compile(p), p.header);
}
TEST(ComputeLoopDataRoute, NativeWidthWithoutActualLaunchDoesNotGrantRetry) {
    auto c = config();
    c.dispatch_launch.known = 0;
    missing(compile(program(), c), 6);
}
TEST(ComputeLoopDataRoute, MalformedLaunchDoesNotGrantRetry) {
    auto c = config();
    c.dispatch_launch.initiator |= 0x80000000ull;
    missing(compile(program(), c), 6);
}

TEST(ComputeLoopDataRoute, BothPredecessorsSupplyTheCurrentGeneration) {
    wave_maximum(compile(predecessor(true)));
}
TEST(ComputeLoopDataRoute, OnePredecessorCannotSupplyMissingLoopEntryData) {
    const auto p = hard_old(predecessor(false), 1);
    missing(compile(p), p.header);
}
TEST(ComputeLoopDataRoute, GenuineBackedgeRelayRetainsActiveData) {
    auto p = program();
    p.words[11] = 0x7e020307u;
    const auto r = compile(p);
    ASSERT_FALSE(r.module.empty()) << r.reason;
    ASSERT_TRUE(has_op(r.module, 251u));
}
TEST(ComputeLoopDataRoute, GenuineFormatLoadRetainsItsCurrentProducer) {
    const auto r = compile(memory_producer(true), config(), input_table());
    ASSERT_FALSE(r.module.empty()) << r.reason;
    ASSERT_TRUE(has_op(r.module, 251u));
}
TEST(ComputeLoopDataRoute, AFormatBindingCannotSupplyAnUnwrittenAddress) {
    // The address v9 must be hard: a never-written one is arbitrary, and a bounded format load
    // there loads a value hardware could have loaded (#5193).
    const auto p = hard_old(memory_producer(false), 9);
    missing(compile(p, config(), input_table()), p.header);
}
TEST(ComputeLoopDataRoute, AFormatLoadStillNeedsItsActualResource) {
    const auto r = compile(memory_producer(true));
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reason.find("pc=3 words=e0002000"), std::string::npos) << r.reason;
}
TEST(ComputeLoopDataRoute, AFormatLoadStillNeedsItsActualFormat) {
    auto t = input_table();
    t.resources.back().format = DataFormat::Unknown;
    const auto r = compile(memory_producer(true), config(), t);
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reason.find("pc=3 words=e0002000"), std::string::npos) << r.reason;
}
TEST(ComputeLoopDataRoute, FailedExactAttemptRetainsTheOriginalHeaderReason) {
    auto t = table();
    t.resources.pop_back();
    missing(compile(program(), config(), t), 6);
}
TEST(ComputeLoopDataRoute, ExactStoreStillNeedsItsActualResource) {
    auto t = table();
    t.resources.pop_back();
    cfg_missing(compile(program(), config(true), t), 9, 0xe0102000u);
}
TEST(ComputeLoopDataRoute, RestoredExecCanReadGenuineEntryData) {
    auto p = program();
    p.words.insert(p.words.end() - 1, 0x7d840080u);
    ASSERT_FALSE(compile(p).module.empty());
}
TEST(ComputeLoopDataRoute, RestoredExecCannotReadInactiveBoundWords) {
    auto p = program();
    p.words.insert(p.words.end() - 1, 0x7d840480u);
    p = hard_old(p, 2);
    missing(compile(p), 7);
    cfg_missing(compile(p, config(true)), 18, 0x7d840480u);
}
TEST(ComputeLoopDataRoute, AStatusBearingStoreDoesNotSelectThisRetry) {
    auto p = program();
    p.words[10] |= 1u << 23;
    missing(compile(p), 6);
    cfg_missing(compile(p, config(true)), 9, 0xe0102000u);
}
TEST(ComputeLoopDataRoute, PartialLocalWaveCannotSelectThisRetry) {
    auto c = config();
    c.local_x = 65;
    prosper::test::complete_compute_launch(c);
    missing(compile(program(), c), 6);
}
TEST(ComputeLoopDataRoute, APartialWorkgroupCannotSelectThisRetry) {
    auto c = config();
    c.exact_thread_extent = true;
    c.threads_x = 65;
    prosper::test::complete_compute_launch(c);
    missing(compile(program(), c), 6);
}
TEST(ComputeLoopDataRoute, AnUncertifiedPortableWaveKeepsTheOriginalRefusal) {
    auto c = config();
    c.native_subgroup_size = 0;
    missing(compile(program(), c), 6);
}

TEST(ComputeLoopDataRoute, AnEstablishedPureLoopKeepsItsStructuredRoute) {
    auto p = program();
    p.words[8] = p.words[9] = p.words[10] = kNop;
    p.words[14] = 0x7e060280u;
    const auto r = compile(p);
    ASSERT_FALSE(r.module.empty()) << r.reason;
    EXPECT_TRUE(has_op(r.module, 246u));
    EXPECT_FALSE(has_op(r.module, 251u)) << "successful established modules are never retried";
}
TEST(ComputeLoopDataRoute, AnOriginalExtraHeaderEdgeCannotBorrowItsFoldedAlias) {
    auto p = program();
    p.words[5] = 0xbf068080u;
    p.words.insert(p.words.begin() + 6, 0xbf840000u);
    p.header += 1;
    missing(compile(p), p.header);
}
TEST(ComputeLoopDataRoute, AClosedOriginalWithTheSameConstantConditionCanRetry) {
    auto p = program();
    p.words[5] = 0xbf068080u;
    p.words.insert(p.words.begin() + 6, kNop);
    p.header += 1;
    wave_maximum(compile(p));
}
