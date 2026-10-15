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

// #5201: the same loop with every vector word DEFINED (v1, v2 written under the full entry EXEC).
// No header read refuses, so what routes it is what the loop is: no narrowing certificate (its body
// stores) and a wave scalar, s6, live after it. Per-invocation trip counts would leave s6 at each
// lane's own count; hardware leaves the wave maximum.
namespace {
Program defined_data() {
    return {{0xbe94047eu, 0x7e020280u, 0x7e040300u, 0x7daa0080u, 0xbe860380u, 0x7e0e02c0u,
             0x7da20501u, 0xbf880005u, 0x80068106u, 0xe0102000u, 0x80030100u, 0x4a020281u,
             0xbf82fff9u, 0xbefe0414u, 0x7e060206u, 0xe0702000u, 0x80020300u, kEnd},
            6};
}
// The structured route, and output2 (binding 2) for every lane.
std::vector<uint32_t> structured(const Result& r, std::vector<uint32_t>* stores = nullptr) {
    EXPECT_FALSE(r.module.empty()) << r.reason;
    if (r.module.empty()) return {};
    EXPECT_TRUE(has_op(r.module, 246u)) << "OpLoopMerge";
    EXPECT_FALSE(has_op(r.module, 251u)) << "no dispatcher OpSwitch";
    bpermute_oracle::Interpreter vm(r.module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config().user_sgprs;
    vm.extra_writable_bindings = {3};
    auto values = vm.run_buffers(
        64,
        {{2, std::vector<uint32_t>(64, 0xdeadbeefu)}, {3, std::vector<uint32_t>(64, 0xdeadbeefu)}},
        2);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    if (stores) *stores = vm.writable_result(3);
    return values;
}
} // namespace

TEST(LoopWaveScalarRoute, ALiveWaveScalarTakesTheExactRouteWithDefinedData) {
    wave_maximum(compile(defined_data()));
}
TEST(LoopWaveScalarRoute, ADeadWaveScalarKeepsTheStructuredRoute) {
    auto p = defined_data();
    p.words[14] = 0x7e060280u;   // v3 = 0: s6 (and the SCC its add writes) dead after the loop
    const auto values = structured(compile(p));
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(values[lane], 0u) << lane;
}
TEST(LoopWaveScalarRoute, VectorResultsAloneKeepTheStructuredRoute) {
    auto p = defined_data();
    p.words[8] = kNop;           // no scalar write in the body
    p.words[14] = 0x7e060301u;   // v3 = v1, each lane's own final count
    const auto values = structured(compile(p));
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(values[lane], lane) << lane;
}
TEST(LoopWaveScalarRoute, ACertifiedNarrowingLoopKeepsTheStructuredRoute) {
    auto p = defined_data();
    p.words[8] = p.words[9] = p.words[10] = kNop;   // a lane-local body: the certificate holds
    p.words[14] = 0x7e060280u;
    const auto values = structured(compile(p));
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(values[lane], 0u) << lane;
}
TEST(LoopWaveScalarRoute, ALiveSccTakesTheExactRoute) {
    auto p = defined_data();
    // s7 = SCC ? 1 : 0 after the loop: SCC from the body's s6 add is live; s6 itself is dead.
    p.words[14] = 0x7e060207u;   // v3 = s7
    p.words.insert(p.words.begin() + 14, 0x85078180u);   // s_cselect_b32 s7, 1, 0
    const auto r = compile(p);
    ASSERT_FALSE(r.module.empty()) << r.reason;
    EXPECT_TRUE(has_op(r.module, 251u)) << "exact synchronized dispatcher route";
}
TEST(LoopWaveScalarRoute, WithoutANativeEntryALiveWaveScalarIsRefused) {
    auto c = config();
    c.native_subgroup_size = 0;
    const auto r = compile(defined_data(), c);
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reason.find("wave-scalar-loop-live-out"), std::string::npos) << r.reason;
}

// #5206: a compute EXEC loop whose body can re-enable a lane that already left it. Hardware runs
// that lane again in every later wave iteration; the left invocation cannot. The scalars are dead
// after these loops (v3 = 0), so only the widening routes them.
namespace {
Program widening(uint32_t restore) {
    auto p = defined_data();
    p.words[8] = restore;   // in the body, after the exit test
    p.words[14] = 0x7e060280u;   // v3 = 0
    return p;
}
// Exact route; output3 (binding 3) holds `last` in every lane of `lanes`, untouched elsewhere.
void exact_stores(const Result& r, uint32_t first_lane, uint32_t last) {
    ASSERT_FALSE(r.module.empty()) << r.reason;
    ASSERT_TRUE(has_op(r.module, 251u)) << "exact synchronized dispatcher route";
    bpermute_oracle::Interpreter vm(r.module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config().user_sgprs;
    vm.extra_writable_bindings = {3};
    vm.run_buffers(
        64,
        {{2, std::vector<uint32_t>(64, 0xdeadbeefu)}, {3, std::vector<uint32_t>(64, 0xdeadbeefu)}},
        2);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto stores = vm.writable_result(3);
    ASSERT_EQ(stores.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(stores[lane], lane < first_lane ? 0xdeadbeefu : last) << "lane " << lane;
}
}   // namespace

TEST(LoopExecWidening, AnAllLanesWriteReEnablesLeftLanes) {
    // The #5206 reproduction: s_mov_b64 exec, -1. Every lane, lane 0 included, stores in every
    // wave iteration; the last one (iteration 63) stores 62.
    exact_stores(compile(widening(0xbefe04c1u)), 0, 62);
}
TEST(LoopExecWidening, AHighToLowExecCopyReEnablesDepartedLowLanes) {
    // High lane N+32 can still be active after low lane N has left. Moving its bit into
    // EXEC_LO re-enables N; a bound on the other physical half is not a subset proof.
    auto p = widening(0xbe98037fu);   // s_mov_b32 s24, exec_hi
    p.words.insert(p.words.begin() + 9, {0xbe99037fu, 0xbefe0418u});
    p.words[7] = 0xbf880007u;
    p.words[14] = 0xbf82fff7u;
    const auto r = compile(p);   // s25 = exec_hi; exec = s[24:25]
    ASSERT_FALSE(r.module.empty()) << r.reason;
    EXPECT_TRUE(has_op(r.module, 251u)) << "cross-half restore needs the exact route";
    bpermute_oracle::Interpreter vm(r.module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config().user_sgprs;
    vm.extra_writable_bindings = {3};
    vm.run_buffers(
        64,
        {{2, std::vector<uint32_t>(64, 0xdeadbeefu)}, {3, std::vector<uint32_t>(64, 0xdeadbeefu)}},
        2);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const auto stores = vm.writable_result(3);
    ASSERT_EQ(stores.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(stores[lane], lane < 32 ? lane + 31u : lane - 1u) << lane;
}
TEST(LoopExecWidening, MatchingExecHalvesMayTravelThroughOppositeRegisterParities) {
    auto p = widening(0xbe99037eu);   // s25 = exec_lo
    p.words.insert(p.words.begin() + 9, {0xbe98037fu, 0xbe9a0319u, 0xbe9b0318u, 0xbefe041au});
    p.words[7] = 0xbf880009u;
    p.words[16] = 0xbf82fff5u;
    const auto r = compile(p);   // s24 = exec_hi; s26 = s25; s27 = s24; exec = s[26:27]
    ASSERT_FALSE(r.module.empty()) << r.reason;
    EXPECT_TRUE(has_op(r.module, 246u)) << "matching physical halves stay structured";
    EXPECT_FALSE(has_op(r.module, 251u)) << "SGPR parity must not select the dispatcher";
}
TEST(LoopExecWidening, ARestoreOfAMaskSavedBeforeTheLoopWidens) {
    exact_stores(compile(widening(0xbefe0414u)), 0, 62);   // s_mov_b64 exec, s[20:21]
}
TEST(LoopExecWidening, ARestoreOfAMaskSavedBeforeTheExitTestWidens) {
    // s[24:25] = EXEC at the header, before the CMPX test; restored in the body. Lanes 1..63 keep
    // iterating with the wave until iteration 63; lane 0 never entered.
    auto p = widening(0xbefe0418u);   // s_mov_b64 exec, s[24:25]
    p.words.insert(p.words.begin() + 6, 0xbe98047eu);   // header: s_mov_b64 s[24:25], exec
    p.words[13] = 0xbf82fff8u;   // back edge -> the new header pc6
    exact_stores(compile(p), 1, 62);
}
TEST(LoopExecWidening, ANestedIfRestoreKeepsTheStructuredRoute) {
    // v_cmp_ne vcc,0,v1; s_and_saveexec s[24:25],vcc; v8=0; s_or exec,exec,s[24:25]: the body
    // restores only what it narrowed after the exit test.
    auto p = widening(0x7d8a0280u);
    p.words.insert(p.words.begin() + 9, {0xbe98246au, 0x7e100280u, 0x88fe187eu});
    p.words[7] = 0xbf880008u;   // exit -> restore at pc16
    p.words[15] = 0xbf82fff6u;   // back edge -> header pc6
    std::vector<uint32_t> stores;
    const auto values = structured(compile(p), &stores);
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(values[lane], 0u) << lane;
    ASSERT_EQ(stores.size(), 64u);
    EXPECT_EQ(stores[0], 0xdeadbeefu) << "lane 0 never enters the body";
    for (uint32_t lane = 1; lane < 64; ++lane)
        EXPECT_EQ(stores[lane], lane - 1u) << "last masked store for lane " << lane;
}
TEST(LoopExecWidening, WithoutANativeEntryAWideningLoopIsRefused) {
    auto c = config();
    c.native_subgroup_size = 0;
    const auto r = compile(widening(0xbefe04c1u), c);
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reason.find("exec-loop-widens"), std::string::npos) << r.reason;
}

TEST(LoopExecWidening, APortableBarrierPhaseKeepsAnExactRouteForNestedExecRestores) {
    // The outer body restores pre-loop EXEC after its inner loop exits. Its portable phase
    // dispatcher can reproduce that widening; a late structured refusal must not hide the route.
    const std::vector<uint32_t> words = {
        0xbe82047eu, 0x7e040280u, 0x7e060280u, 0xbf880003u, 0x7e040281u, 0xbefe0480u,
        0xbf82fffcu, 0xbefe0402u, 0x7e040287u, 0xbefe0480u, 0xbf89fff7u, 0xbefe0402u,
        0xbf8a0000u, 0xbefe0402u, 0xbf880004u, 0xd8d80000u, 0x03000000u, 0xbefe0480u,
        0xbf82fffbu, 0xbefe0402u, 0xbefe0402u, 0xbf880004u, 0xd8d80000u, 0x04000000u,
        0xbefe0480u, 0xbf82fffbu, 0xbefe0402u, 0x7e040d02u, kEnd};
    const auto module = recompile_valu(words.data(), words.size(), 1, 2);
    ASSERT_FALSE(module.empty());
    EXPECT_TRUE(has_op(module, 251u)) << "the phase uses the exact dispatcher";
    EXPECT_TRUE(has_op(module, 224u)) << "the proved guest barrier separates phases";
}
