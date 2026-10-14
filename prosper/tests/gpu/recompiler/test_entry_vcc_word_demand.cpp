// #4878: a dead entry VCC need not have both physical words overwritten before a B32 read.
// Demand the actual read word while preserving pair, conditional-write and all-path contracts.
// A synthetic CPU oracle executes production dispatcher storage; it does not claim live NGG inputs.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_entry_vcc.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;

namespace {

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& words) {
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(words.data(), words.size(), ins), words.size());
    return ins;
}

// MOV publishes one real scalar word; BRANCH creates a dispatcher boundary before the read.
// V_LSHL_ADD_U32's three source words are already pinned by the production width inventory.
std::vector<uint32_t> code(int written = 106, int read = 106, bool complete = false) {
    std::vector<uint32_t> words{0xbe8003aau | (static_cast<uint32_t>(written) << 16u)};
    if (complete) words.push_back(0xbeeb0387u);   // the independent complete-pair control
    words.insert(words.end(), {0xbf820000u, 0xd7460000u, 0x02150600u | static_cast<uint32_t>(read),
                               0xbf810000u});
    return words;
}

struct Compiled {
    std::vector<uint32_t> module;
    std::vector<std::pair<std::string, std::string>> rejects;
};

Compiled compile(const std::vector<uint32_t>& words, uint32_t native = 0, bool partial = false) {
    SpirvCompute b;
    static uint64_t next_program = 0x48780000u;
    b.diagnostic = {RecompileDiagnosticStage::Compute, next_program++};
    b.native_subgroup_size = native;
    b.begin(4u, nullptr, 64u, 1u, 1u, 64u, 0u, true, true);
    RegState state;
    state.exec = b.btrue();
    state.scc = b.bfalse();
    state.max_vgpr = 0;
    state.sreg_input[0] = b.groupid[0];
    // Start precisely at the gate, with ABSENT VCC; the whole-program wrapper seeds false.
    const uint32_t active = partial ? b.ucmp(Op_ULessThan, b.gidx, b.uconst(32u)) : 0u;
    TerminalRejectCapture capture;
    Compiled result;
    if (emit_cfg_state_machine(
            b, state, decode(words), {}, nullptr, true, false,
            [](RegState&, const Rdna2Inst&) { return false; }, words.data(), words.size(),
            active)) {
        const auto output = state.vreg.find(0);
        if (output == state.vreg.end()) {
            ADD_FAILURE() << "dispatcher did not publish the actual v0 result";
        } else {
            b.store_output_word(output->second, 1u, 0u);
            result.module = b.finish();
        }
    }
    result.rejects = capture.take();
    return result;
}

bool has_switch(const std::vector<uint32_t>& module) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16u;
        if (!count || count > module.size() - at) return false;
        if ((module[at] & 0xffffu) == Op_Switch) return true;
        at += count;
    }
    return false;
}

void expect_missing(const std::vector<uint32_t>& words, bool partial = false) {
    for (uint32_t native : {0u, 64u}) {
        SCOPED_TRACE(native);
        const auto result = compile(words, native, partial);
        EXPECT_TRUE(result.module.empty());
        const std::pair<std::string, std::string> expected{"compute-cfg-reject",
                                                           "pc=0 reason=missing-entry-vcc"};
        EXPECT_NE(std::find(result.rejects.begin(), result.rejects.end(), expected),
                  result.rejects.end())
            << testing::PrintToString(result.rejects);
    }
}

TEST(EntryVccWordDemand, DecodeAndExistingB32WidthPreconditions) {
    const auto ins = decode(code());
    ASSERT_EQ(ins.size(), 4u);
    EXPECT_EQ(ins[0].dst.value, 106);
    EXPECT_EQ(ins[0].src[0].value, 42);
    ASSERT_EQ(ins[2].fmt, Rdna2Format::VOP3);
    ASSERT_EQ(ins[2].opcode, kVop3OpcodeLshlAddU32);
    ASSERT_EQ(ins[2].src[0].kind, OperandKind::Special);
    ASSERT_EQ(ins[2].src[0].value, 106);
    EXPECT_EQ(ins[2].src[1].value, 3);
    EXPECT_EQ(ins[2].src[2].value, 5);
    for (uint32_t source = 0; source < 3; ++source)
        EXPECT_EQ(scalar_alu_source_words(ins[2], source), 1u);
}

TEST(EntryVccWordDemand, OnlyTheReadWordNeedsAnEarlierDefinition) {
    for (int word : {106, 107}) {
        SCOPED_TRACE(word);
        const auto ins = decode(code(word, word));
        EXPECT_FALSE(entry_block_defines_vcc_before_any_read(ins, 0u, 2u));
        EXPECT_TRUE(region_defines_vcc_before_any_read(ins));
        const auto native = compile(code(word, word), 64u);
        EXPECT_FALSE(native.module.empty()) << testing::PrintToString(native.rejects);
        EXPECT_TRUE(has_switch(native.module));
    }
}

TEST(EntryVccWordDemand, CompletePairControlAlreadyCompiles) {
    const auto words = code(106, 106, true);
    ASSERT_TRUE(region_defines_vcc_before_any_read(decode(words)));
    for (uint32_t native : {0u, 64u}) {
        SCOPED_TRACE(native);
        const auto result = compile(words, native);
        ASSERT_FALSE(result.module.empty()) << testing::PrintToString(result.rejects);
        EXPECT_TRUE(has_switch(result.module));
    }
}

TEST(EntryVccWordDemand, ProductionDispatcherPreservesTheDefinedWord) {
    for (int word : {106, 107}) {
        SCOPED_TRACE(word);
        const auto result = compile(code(word, word));
        ASSERT_FALSE(result.module.empty()) << testing::PrintToString(result.rejects);
        ASSERT_TRUE(has_switch(result.module));
        bpermute_oracle::Interpreter vm(result.module);
        const auto output =
            vm.run_buffers(64u, {{1u, std::vector<uint32_t>(64u, 0xdeadbeefu)}}, 1u);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(output, std::vector<uint32_t>(64u, 341u));   // (42 << 3) + 5, never false/zero
    }
}

TEST(EntryVccWordDemand, ARealOppositeHalfReadStillRequiresItsOwnDefinition) {
    EXPECT_FALSE(region_defines_vcc_before_any_read(decode(code(106, 107))));
    EXPECT_FALSE(region_defines_vcc_before_any_read(decode(code(107, 106))));
    expect_missing(code(106, 107));
    expect_missing(code(107, 106));
}

TEST(EntryVccWordDemand, ExplicitAndImplicitMaskReadsDemandTheCompletePair) {
    for (const auto& reader :
         {std::vector<uint32_t>{0xd5010000u, 0x01aa0100u}, std::vector<uint32_t>{0x02000100u}}) {
        auto words = code();
        words.erase(words.begin() + 2, words.begin() + 4);
        words.insert(words.begin() + 2, reader.begin(), reader.end());
        EXPECT_FALSE(region_defines_vcc_before_any_read(decode(words)));
        expect_missing(words);
    }
}

TEST(EntryVccWordDemand, OneArmDefinitionCannotCrossARealJoin) {
    const std::vector<uint32_t> words{0xbf068000u, 0xbf850001u, 0xbeea03aau,
                                      0xd7460000u, 0x0215066au, 0xbf810000u};
    EXPECT_FALSE(region_defines_vcc_before_any_read(decode(words)));
    expect_missing(words);
    auto control = words;
    control[1] = 0xbf850000u;   // both successors now pass the real scalar write
    EXPECT_TRUE(region_defines_vcc_before_any_read(decode(control)));
}

TEST(EntryVccWordDemand, ABackEdgeCannotDefineTheFirstIterationRead) {
    const std::vector<uint32_t> words{0xd7460000u, 0x0215066au, 0xbeea03aau, 0xbf82fffcu,
                                      0xbf810000u};
    EXPECT_FALSE(region_defines_vcc_before_any_read(decode(words)));
    expect_missing(words);
    const std::vector<uint32_t> control{0xbeea03aau, 0xd7460000u, 0x0215066au, 0xbf82fffdu,
                                        0xbf810000u};
    EXPECT_TRUE(region_defines_vcc_before_any_read(decode(control)));
}

TEST(EntryVccWordDemand, ConditionalMovesCannotDefineAnAbsentEntryWord) {
    for (uint32_t move : {0xbeea05aau, 0xbeea06aau, 0xb16a002au}) {
        auto words = code();
        words[0] = move;   // CMOV_B32 / CMOV_B64 / CMOVK preserve old bits on SCC=false
        EXPECT_FALSE(region_defines_vcc_before_any_read(decode(words)));
        expect_missing(words);
    }
}

TEST(EntryVccWordDemand, ImplicitDestinationReadsDemandOnlyTheirEstablishedWord) {
    for (uint32_t move : {0xbeea05aau, 0xb16a002au}) {
        auto words = code();
        words.insert(words.begin() + 1, move);
        const auto ins = decode(words);
        ASSERT_EQ(scalar_implicit_destination_read_width(ins[1]), 1u);
        EXPECT_TRUE(region_defines_vcc_before_any_read(ins));
        words[1] = 0xbeea06aau;   // CMOV_B64 also keeps the absent high word on SCC=false
        EXPECT_FALSE(region_defines_vcc_before_any_read(decode(words)));
    }
}

TEST(EntryVccWordDemand, CmpxAndUnmodeledSmemDoNotPublishVccWords) {
    auto words = code();
    words[0] = 0x7da20080u;   // CMPX writes EXEC, never VCC
    const auto cmpx = decode(words);
    ASSERT_TRUE(vopc_is_cmpx(cmpx.front().opcode));
    EXPECT_FALSE(region_defines_vcc_before_any_read(cmpx));
    // SMEM opcode 0x0d is unassigned in gfx1030 and gfx1013. (This fixture used 0x05, which is
    // gfx10.1's s_scratch_load_dword and does write its SDATA.)
    const std::vector<uint32_t> unknown{0xf4341a84u, 0xfa000000u, 0xd7460000u, 0x0215066au,
                                        0xbf810000u};
    const auto ins = decode(unknown);
    ASSERT_EQ(ins.front().fmt, Rdna2Format::SMEM);
    ASSERT_EQ(ins.front().opcode, 0x0du);
    EXPECT_FALSE(region_defines_vcc_before_any_read(ins));
    auto control = unknown;
    control[0] = 0xf4201a84u;   // a supported buffer-load dword really writes VCC_LO
    EXPECT_TRUE(region_defines_vcc_before_any_read(decode(control)));
}

TEST(EntryVccWordDemand, UnclassifiedWideReadersKeepThePairDemand) {
    auto words = code();
    // An unassigned VOP3 opcode (0x3ff) is the one VALU source the effects table cannot classify,
    // so it keeps the conservative pair width. (V_XAD_U32, used here before #5146, is B32.)
    words[2] = 0xd7ff0000u;
    ASSERT_EQ(scalar_alu_source_words(decode(words)[2], 0u), 2u);
    EXPECT_FALSE(region_defines_vcc_before_any_read(decode(words)));
    const std::vector<uint32_t> unknown{0xbeea03aau, 0xf4000504u, 0xd4000000u, 0xbf810000u};
    EXPECT_FALSE(region_defines_vcc_before_any_read(decode(unknown)));
}

TEST(EntryVccWordDemand, PartialWorkgroupsKeepTheMissingEntryGate) {
    expect_missing(code(), true);
    expect_missing(code(106, 106, true), true);
}

}   // namespace
