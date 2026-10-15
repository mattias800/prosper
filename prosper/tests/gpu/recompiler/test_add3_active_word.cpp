// A supported ADD3 fully replaces a lane's word only when all three current inputs are genuine.
// Its address feeds a real buffer load and synchronized scalar selection; absent inputs must
// never become buffer offsets or observable stores, including a padded final workgroup.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
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
constexpr uint32_t kCanary = 0xdeadbeefu;
constexpr uint32_t kAdd3 = 0xd76d0002u;
constexpr uint32_t kSources = 0x100u | (12u << 9u) | (131u << 18u); // v0+s12+3
std::vector<uint32_t> program() {
    return {
        0xbe84047eu, // save ENTRY EXEC
        0x7da20084u, // CMPX_LT_U32 4,v0: lanes above4
        0x7da800b0u, // CMPX_GT_U32 48,v0: lanes below48
        kAdd3,       kSources, // v2=localID+1+3
        0x7daa0480u, // actual ADD3-derived predicate controls the load's enabled lanes
        0xe0302000u, 0x80020302u, // load v3 at index v2, descriptor s8
        0x7daa0680u, // the loaded word determines the service's active source lanes
        0x7e3c0503u, // synchronized READFIRST s30=v3
        0xbefe0404u, // restore ENTRY; scalar service result is common to all guest workers
        0x7e08021eu, // v4=s30
        0xe0702000u, 0x80040400u, // observable store v4 at localID, descriptor s16
        0xbf810000u,
    };
}
struct Attempt {
    std::vector<uint32_t> module;
    std::string reason;
};
Attempt compile(const std::vector<uint32_t>& code, uint32_t threads = 64) {
    ComputeShaderConfig c;
    c.wave_size = 64;
    c.local_x = 64;
    c.tgid_x_en = true;
    c.exact_thread_extent = true;
    c.threads_x = threads;
    c.threads_y = c.threads_z = 1;
    c.force_exchange_dispatcher = true;
    c.user_sgprs.assign(21, 0);
    c.user_sgprs[12] = 1;
    ShaderResourceTable table;
    ShaderResource input;
    input.cls = ResourceClass::ConstantBuffer;
    input.format = DataFormat::Uint32;
    input.num_components = 1;
    input.stride = 4;
    input.binding = 4;
    input.sgpr_base = 8;
    input.size = 512;
    table.resources.push_back(input);
    auto output = input;
    output.binding = 3;
    output.sgpr_base = 16;
    table.resources.push_back(output);
    static uint64_t address = 0x5141000100ull;
    const auto id = address++;
    auto module = recompile_compute(code.data(), code.size(), &table, c,
                                    {RecompileDiagnosticStage::Compute, id});
    if (!module.empty()) {
        // Each CTest process has a fixed optional artifact directory; no admission reads it.
        // NOLINTNEXTLINE(concurrency-mt-unsafe)
        if (const char* directory = std::getenv("PROSPER_ADD3_SPV_DIRECTORY")) {
            const auto* test = ::testing::UnitTest::GetInstance()->current_test_info();
            std::ofstream file(std::filesystem::path(directory) /
                                   (std::string(test->name()) + "-" + std::to_string(id) + ".spv"),
                               std::ios::binary);
            file.write(reinterpret_cast<const char*>(module.data()),
                       static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
            EXPECT_TRUE(file.good());
        }
    }
    return {std::move(module), last_terminal_reject_reason(id)};
}
void producing(const std::vector<uint32_t>& code, uint32_t threads = 64, uint32_t group = 0,
               uint32_t selected_index = 9) {
    const auto attempt = compile(code, threads);
    ASSERT_FALSE(attempt.module.empty()) << attempt.reason;
    std::vector<uint32_t> input(128);
    for (uint32_t i = 0; i < input.size(); ++i) input[i] = 1000u + 17u * i;
    bpermute_oracle::Interpreter oracle(attempt.module);
    oracle.supplied_push_constants.assign(21, 0);
    oracle.supplied_push_constants[12] = 1;
    const auto output =
        oracle.run_buffers(64, {{4, input}, {3, std::vector<uint32_t>(64, kCanary)}}, 3, group);
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    std::vector<uint32_t> expected(64, kCanary);
    const uint32_t active = std::min(64u, threads - group * 64u);
    std::fill(expected.begin(), expected.begin() + active, input[selected_index]);
    EXPECT_EQ(output, expected) << "the selected input follows the actual Bool and lane sources; "
                                   "padded workers never store";
}
// The absent inputs below must be HARD: a never-written v2 or v9 is arbitrary, and a bounded
// buffer read at an arbitrary address loads a value hardware could have loaded (#5185, #5193).
// So every refused arm first relays s30 -- which no launch loads -- into v2 and v9.
void missing(const std::vector<uint32_t>& body) {
    std::vector<uint32_t> code{0x7e04021eu, 0x7e12021eu};   // v2 = s30; v9 = s30
    code.insert(code.end(), body.begin(), body.end());
    const auto attempt = compile(code);
    EXPECT_TRUE(attempt.module.empty())
        << "missing current input reached an observable buffer sink";
    EXPECT_FALSE(attempt.reason.empty());
}
std::vector<uint32_t> arms(bool first, bool second) {
    auto code = program();
    code.erase(code.begin() + 3, code.begin() + 5);
    const std::vector<uint32_t> body{
        0xbf060014u, 0xbf850003u,           first ? kAdd3 : kNop,     first ? kSources : kNop,
        0xbf820002u, second ? kAdd3 : kNop, second ? kSources : kNop,
    };
    code.insert(code.begin() + 3, body.begin(), body.end());
    return code;
}
std::vector<uint32_t> selected_copy(bool add3, bool produced_condition) {
    auto code = program();
    code.erase(code.begin() + 3, code.begin() + 5);
    // The existing initial Bool condition and a genuine compare are separate premises.
    // Their selected DATA must survive a real copy, address, load and scalar/store sink.
    std::vector<uint32_t> body{0x7e020280u};   // v1=0
    if (produced_condition) body.push_back(0x7d8a0080u);   // VCC=(0!=v0), true on these lanes
    body.insert(body.end(), {0x020c0300u, 0x7e0e0306u});   // v6=CNDMASK(v0,v1); v7=v6
    if (add3) {
        body.insert(body.end(), {kAdd3, (kSources & ~511u) | 263u});
    } else {
        body.push_back(0x4a040e84u);   // existing ADD_U32 v2=4+v7
    }
    code.insert(code.begin() + 3, body.begin(), body.end());
    return code;
}
}   // namespace
TEST(Add3ActiveWord, CurrentOperandsFeedLoadedScalarAndRealStores) {
    producing(program());
}
TEST(Add3ActiveWord, ExactPartialLastWorkgroupKeepsPaddedCanaries) {
    producing(program(), 2064, 32);
}
TEST(Add3ActiveWord, BothPredecessorsDefineTheCurrentAddress) {
    producing(arms(true, true));
}
TEST(Add3ActiveWord, OnePredecessorCannotDefineTheJoinedAddress) {
    missing(arms(true, false));
}
TEST(Add3ActiveWord, MissingVectorSourceCannotAuthenticateTheAddress) {
    auto code = program();
    code[4] = (kSources & ~511u) | 265u;
    missing(code);
}
TEST(Add3ActiveWord, MissingScalarSourceCannotAuthenticateTheAddress) {
    auto code = program();
    code[4] = (kSources & ~(511u << 9u)) | (22u << 9u);
    missing(code);
}
TEST(Add3ActiveWord, MissingThirdSourceCannotAuthenticateTheAddress) {
    auto code = program();
    code[4] = (kSources & ~(511u << 18u)) | (265u << 18u);
    missing(code);
}
TEST(Add3ActiveWord, RelayOfAnUnwrittenSourceStaysUnresolved) {
    auto code = program();
    code.insert(code.begin() + 3, 0x7e0c0309u);
    code[5] = (kSources & ~511u) | 262u;
    missing(code);
}
TEST(Add3ActiveWord, LaterMissingGenerationCannotReuseTheAddress) {
    auto code = program();
    code.insert(code.begin() + 5, 0x7e040309u);
    missing(code);
}
TEST(Add3ActiveWord, LaterGenuineReplacementKeepsItsOwnLoadedValue) {
    auto code = program();
    code.insert(code.begin() + 5, {kAdd3, kSources});
    producing(code);
}
TEST(Add3ActiveWord, AHalfwordWriterCannotReplaceTheWholeAddress) {
    auto code = program();
    code[3] = 0xd3030002u;
    missing(code);
}
TEST(Add3ActiveWord, RestoredInactiveLanesCannotReadTheNarrowAddress) {
    auto code = program();
    code.insert(code.begin() + 5, 0xbefe0404u);
    missing(code);
}
TEST(Add3ActiveWord, ScalarSelectionNeedsTheActualLoadedGeneration) {
    auto code = program();
    code.insert(code.begin() + 8, 0x7e06021eu);   // v3 = s30: never loaded, hard (#5039)
    missing(code);
}
TEST(Add3ActiveWord, NoServiceProducerCannotBorrowItsOldScalarStorage) {
    auto code = program();
    code[9] = kNop;
    code.insert(code.begin() + 12, 0x7daa0880u);
    missing(code);
}
TEST(Add3ActiveWord, ARealConstantScalarProducerKeepsTheMatchedPredicateSink) {
    auto code = program();
    code[9] = 0xbe9e03ffu;
    code.insert(code.begin() + 10, 1153u);
    code.insert(code.begin() + 13, 0x7daa0880u);
    producing(code);
}
TEST(Add3ActiveWord, SelectedScalarDoesNotGainAnUnprovedRestoredPredicateDomain) {
    auto code = program();
    code.insert(code.begin() + 12, 0x7daa0880u);
    missing(code);
}
TEST(Add3ActiveWord, GenuineLiteralThirdOperandFeedsTheSameLoadedSink) {
    auto code = program();
    code[4] = (kSources & ~(511u << 18u)) | (255u << 18u);
    code.insert(code.begin() + 5, 3u);
    producing(code);
}
TEST(Add3ActiveWord, AbsModifierCannotSupplyAnIndependentWord) {
    auto code = program();
    code[3] |= 1u << 8u;
    missing(code);
}
TEST(Add3ActiveWord, ClampModifierCannotSupplyAnIndependentWord) {
    auto code = program();
    code[3] |= 1u << 15u;
    missing(code);
}
TEST(Add3ActiveWord, ReservedOpSelCannotSupplyAnIndependentWord) {
    auto code = program();
    code[3] |= 1u << 11u;
    missing(code);
}
TEST(Add3ActiveWord, NegModifierCannotSupplyAnIndependentWord) {
    auto code = program();
    code[4] |= 1u << 29u;
    missing(code);
}
TEST(Add3ActiveWord, OutputModifierCannotSupplyAnIndependentWord) {
    auto code = program();
    code[4] |= 1u << 27u;
    missing(code);
}
TEST(Add3ActiveWord, InitialConditionCopyFeedsTheExistingAddSink) {
    producing(selected_copy(false, false));
}
TEST(Add3ActiveWord, InitialConditionCopyFeedsTheAdd3Sink) {
    producing(selected_copy(true, false));
}
TEST(Add3ActiveWord, ProducedConditionCopyFeedsTheExistingAddSink) {
    producing(selected_copy(false, true), 64, 0, 4);
}
TEST(Add3ActiveWord, ProducedConditionCopyFeedsTheAdd3Sink) {
    producing(selected_copy(true, true), 64, 0, 4);
}
TEST(Add3ActiveWord, InitialVccNumericWordCannotFeedTheExistingAddSink) {
    auto code = program();
    code.erase(code.begin() + 3, code.begin() + 5);
    code.insert(code.begin() + 3, 0x4a04006au);   // v2=VCC_LO+v0, requires actual numeric DATA
    missing(code);
}
TEST(Add3ActiveWord, InitialVccNumericWordCannotFeedTheAdd3Sink) {
    auto code = program();
    code[4] = (kSources & ~(511u << 9u)) | (106u << 9u);
    missing(code);
}
