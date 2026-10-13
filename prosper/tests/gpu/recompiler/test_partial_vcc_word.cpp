// A B32 scalar write replaces one VCC word, not the sibling comparison bits. Losing those bits
// either refuses a genuine scalar branch or sends every guest lane down the wrong branch.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_partial_vcc_word.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr std::array<uint32_t, 9> original{0xbe800380u, 0x7e000280u, 0x7e020284u,
                                           0x7d020200u, 0xbeea0380u, 0xbf860002u,
                                           0x060000ffu, 0x3e800000u, 0xbf810000u};
ShaderResourceTable table() {
    ShaderResource out{};
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.sgpr_base = 8;
    out.stride = 4;
    ShaderResourceTable result;
    result.resources.push_back(out);
    return result;
}
std::vector<uint32_t> branch(uint32_t lane, bool high_write = false, uint32_t value = 0) {
    // The real local ID creates a single old VCC bit. Scalar VCCZ makes one decision for the
    // complete logical wave; all64 output lanes therefore receive the same branch result.
    return {
        0x7e060280u, 0x7d840080u | lane, (high_write ? 0xbeeb0300u : 0xbeea0300u) | (0x80u + value),
        0xbf860001u, 0x7e060281u,        0xe0702000u,
        0x80020300u, 0xbf810000u};
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address,
                              const ComputeShaderConfig& config = {}) {
    const auto resources = table();
    auto module = recompile_compute(code.data(), code.size(), &resources, config,
                                    {RecompileDiagnosticStage::Compute, address});
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* directory = std::getenv("PROSPER_PARTIAL_VCC_SPV_DIRECTORY");
        directory && !module.empty()) {
        std::ofstream output(std::filesystem::path(directory) / (std::to_string(address) + ".spv"),
                             std::ios::binary);
        output.write(reinterpret_cast<const char*>(module.data()),
                     static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
        EXPECT_TRUE(output.good()) << "retain actual translated SOURCE for strict validation";
    }
    return module;
}
void output_is(const std::vector<uint32_t>& code, const std::vector<uint32_t>& expected,
               uint64_t address, const ComputeShaderConfig& config = {}) {
    const auto module = compile(code, address, config);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = config.native_subgroup_size;
    const auto count = static_cast<uint32_t>(expected.size());
    const auto values = vm.run(count, std::vector<uint32_t>(count, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(values.size(), expected.size());
    for (size_t lane = 0; lane < expected.size(); ++lane)
        EXPECT_EQ(values[lane], expected[lane]) << "lane=" << lane;
}
void observes(uint32_t selected, bool high_write, uint32_t expected, uint64_t address) {
    // Default portable logical64 API: the observer supplies no native/launch authority.
    output_is(branch(selected, high_write), std::vector<uint32_t>(64, expected), address);
}
std::vector<uint32_t> bitmap(uint32_t selected, bool high_write, uint32_t value) {
    return {0x7e020281u,
            0x7d840080u | selected,
            (high_write ? 0xbeeb0300u : 0xbeea0300u) | (0x80u + value),
            0x02060280u, // v_cndmask_b32 v3, 0, v1, vcc
            0xe0702000u,
            0x80020300u,
            0xbf810000u};
}
std::set<uint32_t> proved(const std::vector<uint32_t>& code, bool partial = false) {
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64, 0);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code.data(), code.size(), instructions);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), partial);
    std::set<uint32_t> result;
    for (const auto& [pc, in] : b.partial_vcc_word_pcs) {
        EXPECT_EQ(in.pc, pc);
        result.insert(pc);
    }
    return result;
}
std::vector<uint32_t> join(uint32_t right) {
    // Both actual CFG predecessors define s20, then one MOV consumes it at pc5. The proof
    // examines both edges independently of any other constant-branch specialization.
    return {0x7d8400a8u, 0xbf860002u, 0xbe940380u, 0xbf820001u, right, 0xbeea0314u, 0xbf810000u};
}
}   // namespace
TEST(PartialVccWord, TheUnchangedLegacyCurrentPredicateStillProduces) {
    const auto replaced = rdna2_decode_one(original.data() + 4, original.size() - 4);
    ASSERT_EQ(replaced.fmt, Rdna2Format::SOP1);
    ASSERT_EQ(replaced.opcode, 3u);
    ASSERT_EQ(replaced.dst.value, 106);
    EXPECT_FALSE(recompile_valu(original.data(), original.size(), 0, 0).empty());
}
TEST(PartialVccWord, LowZeroWriteKeepsTheHighComparisonBitForTheWholeWaveBranch) {
    observes(40, false, 1, 0x5135000010ull);
}
TEST(PartialVccWord, LowZeroWriteReplacesTheOldLowComparisonBit) {
    observes(5, false, 0, 0x5135000020ull);
}
TEST(PartialVccWord, HighZeroWriteKeepsTheLowComparisonBitForTheWholeWaveBranch) {
    observes(5, true, 1, 0x5135000030ull);
}
TEST(PartialVccWord, HighZeroWriteReplacesTheOldHighComparisonBit) {
    observes(40, true, 0, 0x5135000040ull);
}
TEST(PartialVccWord, APartialWriteDoesNotProvideTheAbsentOldSiblingMask) {
    auto code = branch(40);
    code[1] = 0xbf800000u;   // no VCC compare, not a supplied initial predicate
    constexpr uint64_t address = 0x5135001010ull;
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("missing VCC at branch pc=3"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}
TEST(PartialVccWord, MissingScalarSourceDoesNotBecomeAZeroReplacementMask) {
    auto code = branch(40);
    code[2] = 0xbeea0314u;   // s20 is absent, with no user-data or instruction producer
    constexpr uint64_t address = 0x5135001020ull;
    EXPECT_TRUE(compile(code, address).empty());
}
TEST(PartialVccWord, LowReplacementPublishesOnlyIts32Bits) {
    auto expected = std::vector<uint32_t>(64);
    expected[0] = expected[40] = 1;
    output_is(bitmap(40, false, 1), expected, 0x5135000050ull);
}
TEST(PartialVccWord, HighReplacementPublishesOnlyIts32Bits) {
    auto expected = std::vector<uint32_t>(64);
    expected[5] = expected[32] = 1;
    output_is(bitmap(5, true, 1), expected, 0x5135000060ull);
}
TEST(PartialVccWord, ReplacementRevokesThePreviousUniformFalseMirror) {
    auto code = branch(40, false, 1);
    code.insert(code.begin(), 0x7e020280u); // actual uniform numeric v1=0
    code[2] = 0x7d840281u; // v_cmp_eq_u32 1,v1: genuine uniform false, no old set bits
    output_is(code, std::vector<uint32_t>(64, 1), 0x5135000070ull);
}
TEST(PartialVccWord, ANewCompleteComparisonReplacesBothWords) {
    auto code = bitmap(40, false, 1);
    code.insert(code.begin() + 3, 0x7d840087u); // local ID==7 replaces all of old VCC
    auto expected = std::vector<uint32_t>(64);
    expected[7] = 1;
    output_is(code, expected, 0x5135000080ull);
}
TEST(PartialVccWord, ARealScalarConstantCopySuppliesTheReplacementWord) {
    auto code = branch(40);
    code.insert(code.begin() + 2, 0xbe940380u); // s20=0
    code[3] = 0xbeea0314u;
    output_is(code, std::vector<uint32_t>(64, 1), 0x5135000090ull);
}
TEST(PartialVccWord, AReadfirstlaneRelayIsNotAUniformReplacementWord) {
    auto code = branch(40);
    code.insert(code.begin() + 2, 0x7e280500u); // v_readfirstlane_b32 s20,v0
    code.insert(code.begin() + 3,
                0x80158014u); // s_add_u32 s21,s20,0, loses direct lane-local tag
    code[4] = 0xbeea0315u;
    constexpr uint64_t address = 0x5135001030ull;
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("pc=5 reason=missing-vcc-mask"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}
TEST(PartialVccWord, CmpxDoesNotSupplyAnAbsentVccProducer) {
    auto code = branch(40);
    code[1] = 0x7da400a8u; // cmpx_eq_u32 40,v0 changes EXEC only
    constexpr uint64_t address = 0x5135001040ull;
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, APhysicalWordSelfCopyDoesNotInheritMissingWordAuthority) {
    auto code = branch(40);
    code[2] = 0xbeea036au; // read VCC_LO DATA without a native physical-word contract
    constexpr uint64_t address = 0x5135001050ull;
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, BothReachingConstantsProveTheSameWord) {
    EXPECT_EQ(proved(join(0xbe940380u)), (std::set<uint32_t>{5}));
}
TEST(PartialVccWord, AOneArmMissingWordCannotBorrowTheOtherPredecessor) {
    EXPECT_TRUE(proved(join(0xbf800000u)).empty());
}
TEST(PartialVccWord, DifferentReachingConstantsDoNotInventUniformity) {
    EXPECT_TRUE(proved(join(0xbe940381u)).empty());
}
TEST(PartialVccWord, BothReachingWordsMissingRemainUnproved) {
    auto code = join(0xbf800000u);
    code[2] = 0xbf800000u;
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, AOneArmOverwriteOfTheOldMaskRetiresItsSibling) {
    auto code = join(0xb06b0001u); // MOVK VCC_HI: outside this audited partial MOV32 route
    code[2] = 0xbf800000u;
    code[5] = 0xbeea0380u;
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, ANewComparisonEndsTheOldPhysicalConstantGeneration) {
    const std::vector<uint32_t> code{0xbeea0380u, 0x7d8400a8u, 0xbeeb036au, 0xbf810000u};
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, BackedgesCannotReuseAnOldLogicalHighWord) {
    auto code = branch(40);
    code.back() = 0xbf82fff8u; // back to pc0; new proof is deliberately acyclic
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, PartialEntryDoesNotProvideMissingParticipation) {
    EXPECT_TRUE(proved(branch(40), true).empty());
    EXPECT_EQ(proved(branch(40)), (std::set<uint32_t>{2}));
}
TEST(PartialVccWord, AChangedOriginalStreamCannotReuseTheOldConsumerProof) {
    auto code = branch(40);
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64, 0);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code.data(), code.size(), instructions);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    ASSERT_EQ(b.partial_vcc_word_pcs.size(), 1u);
    const auto count = instructions.size();
    code[1] = 0xbf800000u; // remove the real producer, keeping the MOV consumer byte-identical
    instructions
        .clear(); // rdna2_walk appends; this must remain the actual new instruction stream
    rdna2_walk(code.data(), code.size(), instructions);
    ASSERT_EQ(instructions.size(), count);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    EXPECT_TRUE(b.partial_vcc_word_pcs.empty());
    RegState absent;
    absent.vcc = b.bfalse(); // the new shell's placeholder is not the old producer's high word
    EXPECT_EQ(partial_vcc_word_predicate(b, absent, instructions[2], b.uconst(0)), 0u);
}
TEST(PartialVccWord, AHighDataWriteRetainsTheIndependentTaggedWave32LowMask) {
    ComputeShaderConfig c;
    c.wave_size = c.native_subgroup_size = c.local_x = 32;
    output_is(branch(5, true, 1), std::vector<uint32_t>(32, 1), 0x51350000a0ull, c);
}
TEST(PartialVccWord, AWholeWaveFalseProducerIsStillGenuine) {
    auto code = branch(40);
    code.insert(code.begin(), 0x7e020280u);
    code[2] = 0x7d840281u; // genuine uniform false compare
    output_is(code, std::vector<uint32_t>(64, 0), 0x51350000b0ull);
}
TEST(PartialVccWord, BothWordReplacementsEndAllOldComparisonBits) {
    auto code = branch(40);
    code.insert(code.begin() + 3, 0xbeeb0380u);
    output_is(code, std::vector<uint32_t>(64, 0), 0x51350000c0ull);
}
TEST(PartialVccWord, ACurrentNumericWordSelfCopyKeepsTheOtherLogicalWord) {
    auto code = bitmap(40, false, 1);
    code.insert(code.begin() + 3, 0xbeea036au); // real earlier scalar VCC_LO=1, not a ballot
    auto expected = std::vector<uint32_t>(64);
    expected[0] = expected[40] = 1;
    output_is(code, expected, 0x51350000d0ull);
}
TEST(PartialVccWord, LowLiteralBit31DoesNotShiftPastItsWord) {
    auto code = bitmap(40, false, 0);
    code[2] = 0xbeea03ffu;
    code.insert(code.begin() + 3, 0x80000000u);
    auto expected = std::vector<uint32_t>(64);
    expected[31] = expected[40] = 1;
    output_is(code, expected, 0x51350000e0ull);
}
TEST(PartialVccWord, HighLiteralBit31PublishesLogicalLane63) {
    auto code = bitmap(5, true, 0);
    code[2] = 0xbeeb03ffu;
    code.insert(code.begin() + 3, 0x80000000u);
    auto expected = std::vector<uint32_t>(64);
    expected[5] = expected[63] = 1;
    output_is(code, expected, 0x51350000f0ull);
}
TEST(PartialVccWord, CmpxPreservesTheOldHighVccBitWhileScalarWritesIgnoreExec) {
    auto code = bitmap(40, false, 1);
    code.insert(code.begin() + 2, 0x7da40085u);   // EXEC = lanes with local ID==5; VCC is unchanged
    code.insert(code.begin() + 4,
                0xbefe04c1u);   // restore full EXEC before observing the predicate
    auto expected = std::vector<uint32_t>(64);
    expected[0] = expected[40] = 1;
    output_is(code, expected, 0x5135000100ull);
}
TEST(PartialVccWord, ALogicalPredicateDoesNotSupplyUnprovedPhysicalHighData) {
    auto code = bitmap(40, false, 0);
    code[3] = 0x7e06026bu;   // read the untouched VCC_HI as ordinary numeric DATA
    constexpr uint64_t address = 0x5135001060ull;
    EXPECT_TRUE(compile(code, address).empty());   // portable API has no physical ballot contract
    ComputeShaderConfig c;
    c.native_subgroup_size = 64;   // native width alone still has no original raw launch authority
    EXPECT_TRUE(compile(code, address + 1, c).empty());
}
TEST(PartialVccWord, AnActualCompareMustReachEveryIncomingEdge) {
    const std::vector<uint32_t> code{0xbf068080u, 0xbf840002u, 0x7d8400a8u, 0xbf820001u,
                                     0xbf800000u, 0xbeea0380u, 0xbf810000u};
    EXPECT_TRUE(proved(code).empty());
    auto genuine = code;
    genuine[4] = 0x7d8400a8u;
    EXPECT_EQ(proved(genuine), (std::set<uint32_t>{5}));
}
TEST(PartialVccWord, ARecursiveSliceCannotReseedTheOriginalProof) {
    auto code = branch(40);
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64, 0);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code.data(), code.size(), instructions);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    ASSERT_EQ(b.partial_vcc_word_pcs.size(), 1u);
    instructions.erase(instructions.begin(), instructions.begin() + 2);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    EXPECT_TRUE(b.partial_vcc_word_pcs.empty());
}
TEST(PartialVccWord, TheProofDoesNotAuthorizeAnotherInstructionAtTheSamePc) {
    auto code = branch(40);
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64, 0);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code.data(), code.size(), instructions);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    RegState rs;
    rs.vcc = b.btrue();
    const auto source = b.uconst(0);
    const auto proofs = b.partial_vcc_word_pcs;
    ASSERT_NE(partial_vcc_word_predicate(b, rs, instructions[2], source), 0u);
    auto changed = instructions[2];
    changed.src[0].value = 1;
    EXPECT_EQ(partial_vcc_word_predicate(b, rs, changed, source), 0u);
    b.partial_vcc_word_pcs = proofs;
    b.native_subgroup_size = 32;
    EXPECT_EQ(partial_vcc_word_predicate(b, rs, instructions[2], source), 0u)
        << "a native32 physical shell cannot stand for this logical64 predicate";
    b.native_subgroup_size = 0;
    b.partial_vcc_word_pcs = proofs;
    b.is_fragment = true;
    EXPECT_EQ(partial_vcc_word_predicate(b, rs, instructions[2], source), 0u)
        << "ordinary fragment invocation has no compute logical lane identity";
}
TEST(PartialVccWord, CurrentWordMarksStillRejectFabricatedAndDescriptorSources) {
    auto code = branch(40);
    code.insert(code.begin() + 2, 0xbe940380u);
    code[3] = 0xbeea0314u;
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64, 0);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code.data(), code.size(), instructions);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    RegState genuine;
    genuine.vcc = b.btrue();
    const auto source = b.uconst(0);
    genuine.sreg[20] = source;
    const auto consumer = instructions[3];
    const auto proofs = b.partial_vcc_word_pcs;
    const auto probe = [&](const RegState& state) {
        b.partial_vcc_word_pcs = proofs;   // each independent current-source challenge starts fresh
        return partial_vcc_word_predicate(b, state, consumer, source);
    };
    ASSERT_NE(probe(genuine), 0u);
    auto missing = genuine;
    missing.sreg.erase(20);
    EXPECT_EQ(probe(missing), 0u);
    auto fabricated = genuine;
    fabricated.sreg_merge_placeholder.insert(20);
    EXPECT_EQ(probe(fabricated), 0u);
    auto memory = genuine;
    memory.sreg_memory_pattern.insert(20);
    EXPECT_EQ(probe(memory), 0u);
    auto descriptor = genuine;
    descriptor.sreg_srt[20] = 64;
    EXPECT_EQ(probe(descriptor), 0u);
    auto lane_local = genuine;
    lane_local.lane_local_scalars.insert(source);
    EXPECT_EQ(probe(lane_local), 0u);
    auto physical = genuine;
    physical.unavailable_saved_mask_words.insert(20);
    EXPECT_EQ(probe(physical), 0u);
}
TEST(PartialVccWord, PredicateReplacementExplicitlyRevokesTheUniformMirror) {
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64, 0);
    RegState rs;
    rs.vcc = rs.vcc_wave_uniform = b.bfalse();
    const auto code = branch(40, false, 1);
    const auto consumer = rdna2_decode_one(code.data() + 2, code.size() - 2);
    const auto predicate = b.btrue(), source = b.uconst(1);
    publish_s_mov_b32_word(rs, consumer, source, predicate);
    EXPECT_EQ(rs.vcc, predicate);
    EXPECT_EQ(rs.vcc_wave_uniform, 0u);
    EXPECT_EQ(rs.sreg.at(106), source);
    EXPECT_FALSE(rs.sreg_bool.contains(106)) << "no complete scalar saved-mask alias was invented";
}
TEST(PartialVccWord, OrdinaryScalarMoveKeepsDescriptorSelectionAndSelfCopy) {
    RegState rs;
    rs.sreg_srt[20] = 64;
    rs.sreg_srt[21] = 128;
    const uint32_t code = 0xbe950314u;   // s_mov_b32 s21,s20
    const auto consumer = rdna2_decode_one(&code, 1);
    publish_s_mov_b32_word(rs, consumer, 11, 0);
    EXPECT_EQ(rs.sreg.at(21), 11u);
    EXPECT_EQ(rs.sreg_srt.at(21), 64u);
    const uint32_t copy = 0xbe950315u;   // s_mov_b32 s21,s21
    publish_s_mov_b32_word(rs, rdna2_decode_one(&copy, 1), 11, 0);
    EXPECT_EQ(rs.sreg_srt.at(21), 64u);
    const uint32_t replace = 0xbe950380u;   // inline zero has no descriptor tag
    publish_s_mov_b32_word(rs, rdna2_decode_one(&replace, 1), 12, 0);
    EXPECT_EQ(rs.sreg.at(21), 12u);
    EXPECT_FALSE(rs.sreg_srt.contains(21));
}
TEST(PartialVccWord, GenuineRawEntryStillSuppliesTheSeparatePhysicalHighWord) {
    auto code = bitmap(40, false, 0);
    code[3] = 0x7e06026bu;
    ComputeShaderConfig c;
    c.native_subgroup_size = 64;
    prosper::test::complete_compute_launch(c);
    output_is(code, std::vector<uint32_t>(64, 1u << 8u), 0x5135000110ull, c);
}
TEST(PartialVccWord, ObservationAndRelayUseTheActualDecodedInstructions) {
    const auto code = bitmap(40, false, 1);
    const auto compare = rdna2_decode_one(code.data() + 1, code.size() - 1);
    EXPECT_EQ(compare.fmt, Rdna2Format::VOPC);
    EXPECT_EQ(compare.opcode, 0xc2u);
    EXPECT_FALSE(vopc_is_cmpx(compare.opcode));
    const auto select = rdna2_decode_one(code.data() + 3, code.size() - 3);
    EXPECT_EQ(select.fmt, Rdna2Format::VOP2);
    EXPECT_EQ(select.opcode, 1u);
    EXPECT_EQ(select.dst.value, 3);
    EXPECT_EQ(select.src[0].kind, OperandKind::InlineInt);
    EXPECT_EQ(select.src[0].value, 0);
    EXPECT_EQ(select.src[1].kind, OperandKind::VGPR);
    EXPECT_EQ(select.src[1].value, 1);
    const uint32_t read = 0x7e280500u;
    const auto relay = rdna2_decode_one(&read, 1);
    EXPECT_EQ(relay.fmt, Rdna2Format::VOP1);
    EXPECT_EQ(relay.opcode, 2u);
    EXPECT_EQ(relay.dst.value, 20);
    const uint32_t add = 0x80158014u;
    const auto copied = rdna2_decode_one(&add, 1);
    EXPECT_EQ(copied.fmt, Rdna2Format::SOP2);
    EXPECT_EQ(copied.opcode, 0u);
    EXPECT_EQ(copied.dst.value, 21);
    EXPECT_EQ(copied.src[0].kind, OperandKind::SGPR);
    EXPECT_EQ(copied.src[0].value, 20);
}
TEST(PartialVccWord, ADeclinedGenerationCannotBorrowALaterCfgPlaceholder) {
    const std::vector<uint32_t> code{0xbe940380u, 0x7d8400a8u, 0xbeea0314u,
                                     0xbf820000u, 0xbeeb0380u, 0xbf810000u};
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64, 0);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code.data(), code.size(), instructions);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    ASSERT_EQ(b.partial_vcc_word_pcs.size(), 2u);
    RegState poisoned;
    poisoned.vcc = b.btrue();
    poisoned.sreg[20] = b.uconst(0);
    poisoned.sreg_merge_placeholder.insert(20);
    EXPECT_EQ(partial_vcc_word_predicate(b, poisoned, instructions[2], poisoned.sreg.at(20)), 0u);
    EXPECT_TRUE(b.partial_vcc_word_pcs.empty());
    poisoned.vcc = b.bfalse();   // later reload's nonzero false SSA is not the departed sibling
    EXPECT_EQ(partial_vcc_word_predicate(b, poisoned, instructions[4], b.uconst(0)), 0u);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    EXPECT_TRUE(b.partial_vcc_word_pcs.empty()) << "same-stream initialization never reseeds";
}
TEST(PartialVccWord, BothRepresentedGenerationsRemainAvailableAcrossTheSameBoundary) {
    const std::vector<uint32_t> code{0xbe940380u, 0x7d8400a8u, 0xbeea0314u,
                                     0xbf820000u, 0xbeeb0380u, 0xbf810000u};
    SpirvCompute b;
    b.begin(1, nullptr, 64, 1, 1, 64, 0);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code.data(), code.size(), instructions);
    initialize_partial_vcc_word_proof(b, instructions, code.data(), code.size(), false);
    RegState genuine;
    genuine.vcc = b.btrue();
    genuine.sreg[20] = b.uconst(0);
    const auto first = partial_vcc_word_predicate(b, genuine, instructions[2], genuine.sreg.at(20));
    ASSERT_NE(first, 0u);
    publish_s_mov_b32_word(genuine, instructions[2], genuine.sreg.at(20), first);
    const auto second = partial_vcc_word_predicate(b, genuine, instructions[4], b.uconst(0));
    EXPECT_NE(second, 0u);
    EXPECT_EQ(b.partial_vcc_word_pcs.size(), 2u);
}
TEST(PartialVccWord, ASecondPartialWriteDoesNotRepairAnAbsentFirstWord) {
    auto code = branch(40);
    code[2] = 0xbeea0314u;   // no instruction supplied s20
    code.insert(code.begin() + 3, 0xbf820000u);   // explicit CFG edge before the next replacement
    code.insert(code.begin() + 4, 0xbeeb0380u);
    constexpr uint64_t address = 0x5135001070ull;
    EXPECT_TRUE(compile(code, address).empty());
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, AHighWordCopySnapshotsTheGenuineLowDataBeforePublication) {
    auto code = bitmap(40, false, 1);
    code.insert(code.begin() + 3, 0xbeeb036au);   // S_MOV_B32 VCC_HI,VCC_LO, both are current DATA
    auto expected = std::vector<uint32_t>(64);
    expected[0] = expected[32] = 1;
    output_is(code, expected, 0x5135000120ull);
}
namespace {
std::vector<uint32_t> scalar_branch(bool readfirst) {
    return {0x7e020281u, 0x7d8400a8u, readfirst ? 0x7e280500u : 0xbe940380u,
            0xbf068014u, 0xbf840001u, 0xbeea03c1u,
            0x02060280u, 0xbf810000u};
}
void scalar_branch_output(const std::vector<uint32_t>& module) {
    std::vector<uint32_t> input(64), output(64, 0xdeadbeefu);
    for (uint32_t lane = 0; lane < 64; ++lane) input[lane] = lane;
    bpermute_oracle::Interpreter vm(module);
    vm.legacy_float_word_storage = true;
    const auto values = vm.run_buffers(64, {{0, input}, {1, output}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(values[lane], lane < 32 || lane == 40 ? 1u : 0u) << "lane=" << lane;
}
}   // namespace
TEST(PartialVccWord, AnInvocationLocalScalarBranchCannotAdmitANewWordReplacement) {
    const auto code = scalar_branch(true);
    const auto module = recompile_valu(code.data(), code.size(), 1, 3);
    EXPECT_TRUE(module.empty()) << "READFIRST's scalar branch needs an exact whole-wave route";
    // A mistaken admission must expose the actual wrong picture, not merely count a module.
    if (!module.empty()) scalar_branch_output(module);
}
TEST(PartialVccWord, ARealConstantScalarBranchRetainsTheSameProducingRegion) {
    const auto code = scalar_branch(false);
    const auto module = recompile_valu(code.data(), code.size(), 1, 3);
    ASSERT_FALSE(module.empty());
    scalar_branch_output(module);
}
TEST(PartialVccWord, LegacyFloatSlotsRequireTheExplicitExactAbi) {
    const auto code = scalar_branch(false);
    const auto module = recompile_valu(code.data(), code.size(), 1, 3);
    ASSERT_FALSE(module.empty());
    std::vector<uint32_t> input(64), output(64, 0xdeadbeefu);
    for (uint32_t lane = 0; lane < 64; ++lane) input[lane] = lane;
    bpermute_oracle::Interpreter strict(module);
    strict.run_buffers(64, {{0, input}, {1, output}}, 1);
    EXPECT_EQ(strict.error, "unsupported live storage ABI");
    for (uint32_t variant = 0; variant < 4; ++variant) {
        auto malformed = module;
        bool changed = false;
        for (size_t at = 5; at < malformed.size();) {
            const uint32_t size = malformed[at] >> 16, opcode = malformed[at] & 0xffffu;
            ASSERT_GT(size, 0u);
            ASSERT_LE(at + size, malformed.size());
            if (variant == 0 && opcode == 22 && size == 3) {
                malformed[at + 2] = 64;   // float64 slots cannot borrow scalar-word transport
                changed = true;
            } else if (opcode == 71 && size == 4 && malformed[at + 2] == 6 && variant == 1) {
                malformed[at + 3] = 8;   // actual ArrayStride must remain exactly four bytes
                changed = true;
            } else if (opcode == 71 && size == 4 && malformed[at + 2] == 34 && variant == 2) {
                malformed[at + 3] = 1;   // the shell's supplied buffers belong to set zero
                changed = true;
            } else if (opcode == 71 && size == 4 && malformed[at + 2] == 33 && variant == 3) {
                malformed[at + 3] += 2;   // float-word allowance belongs only to bindings 0/1
                changed = true;
            }
            at += size;
        }
        ASSERT_TRUE(changed);
        bpermute_oracle::Interpreter vm(malformed);
        vm.legacy_float_word_storage = true;
        if (variant == 3)
            vm.run_buffers(64, {{2, input}, {3, output}}, 3);
        else
            vm.run_buffers(64, {{0, input}, {1, output}}, 1);
        EXPECT_FALSE(vm.error.empty()) << "malformed ABI variant=" << variant;
    }
    auto foreign_set = module;
    for (size_t at = 5; at < foreign_set.size(); at += foreign_set[at] >> 16)
        if ((foreign_set[at] & 0xffffu) == 71 && (foreign_set[at] >> 16) == 4 &&
            foreign_set[at + 2] == 34)
            foreign_set[at + 3] = 2;
    bpermute_oracle::Interpreter foreign(foreign_set);
    foreign.storage_descriptor_set = 2;
    foreign.legacy_float_word_storage = true;
    foreign.run_buffers(64, {{0, input}, {1, output}}, 1);
    EXPECT_EQ(foreign.error, "unsupported live storage ABI");
    scalar_branch_output(module);   // same actual module preserves raw input and output bits
}
TEST(PartialVccWord, ScalarControlNeedsAnActualUniformSccProducer) {
    auto code = scalar_branch(false);
    EXPECT_EQ(proved(code), (std::set<uint32_t>{5}));
    code[2] = 0xbf800000u;   // absent numeric s20 cannot supply the SOPC's DATA
    EXPECT_TRUE(proved(code).empty());
    code = scalar_branch(false);
    code[3] = 0xbf800000u;   // initial nonzero false SCC SSA is not a producer
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, ANewCompareCannotRepairEarlierInvocationLocalControl) {
    auto code = scalar_branch(true);
    code.insert(code.begin() + 5, 0x7d8400a8u);   // fresh VCC inside the unproved scalar arm
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, AScalarSccOverwriteEndsTheEarlierUniformComparisonProof) {
    auto code = scalar_branch(false);
    code.insert(code.begin() + 4, 0xbe960714u);   // NOT_B32 s22,s20 writes SCC
    EXPECT_TRUE(proved(code).empty());
    code[4] = 0xbe960314u;   // MOV_B32 preserves the genuinely uniform SCC
    EXPECT_EQ(proved(code), (std::set<uint32_t>{6}));
}
TEST(PartialVccWord, EveryIncomingScalarBranchMustHaveUniformControl) {
    // The actual VCC producer drives a whole-wave split. Only one arm supplies uniform SCC;
    // the later SCC branch must not borrow it from the other predecessor.
    const std::vector<uint32_t> code{0x7d8400a8u, 0xbf860002u, 0xbf068080u, 0xbf820001u,
                                     0xbf800000u, 0xbf840001u, 0xbeea03c1u, 0xbf810000u};
    EXPECT_TRUE(proved(code).empty());
    auto genuine = code;
    genuine[4] = 0xbf068080u;
    EXPECT_EQ(proved(genuine), (std::set<uint32_t>{6}));
}
TEST(PartialVccWord, AnUnprovedPredecessorCannotBorrowTheOtherArmsWaveControl) {
    // Both arms retain a genuine VCC and eventually uniform SCC, but one first crosses an
    // unproved SCC branch. A later genuine compare cannot erase that control-domain loss.
    const std::vector<uint32_t> code{0x7d8400a8u, 0xbf860002u, 0xbf840000u, 0xbf820001u,
                                     0xbf800000u, 0xbeea03c1u, 0xbf810000u};
    EXPECT_TRUE(proved(code).empty());
    auto genuine = code;
    genuine[2] = 0xbf800000u;
    EXPECT_EQ(proved(genuine), (std::set<uint32_t>{5}));
}
TEST(PartialVccWord, CopyingAnInvocationLocalScalarDoesNotMakeItsControlUniform) {
    auto code = scalar_branch(true);
    code.insert(code.begin() + 3, 0xbe950314u);   // MOV_B32 s21,s20 preserves the unproved source
    code[4] = 0xbf068015u;   // compare the relay, not the original word
    EXPECT_TRUE(proved(code).empty());
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 1, 3).empty());
}
TEST(PartialVccWord, NativeWidthAloneDoesNotAuthorizeInvocationLocalControl) {
    ComputeShaderConfig config;
    config.native_subgroup_size = 64;
    auto code = scalar_branch(true);
    code.insert(code.end() - 1, {0xe0702000u, 0x80020300u});
    constexpr uint64_t address = 0x5135001080ull;
    EXPECT_TRUE(compile(code, address, config).empty());
    // A genuine straight-line replacement still produces in the same native profile. Its
    // logical predicate adds no physical ballot or raw-launch authority. Constant scalar arms
    // can be sliced by the native entry prepass; this proof never reseeds a recursive slice.
    auto expected = std::vector<uint32_t>(64);
    expected[0] = expected[40] = 1;
    output_is(bitmap(40, false, 1), expected, address + 1, config);
}
namespace {
std::vector<uint32_t> scalar_predicate(bool readfirst) {
    return {0x7e020281u, readfirst ? 0x7e280500u : 0xbe940380u,
            0x7d840014u, 0xbeea0380u,
            0x02060280u, 0xbf810000u};
}
void zero_scalar_predicate_output(const std::vector<uint32_t>& module) {
    std::vector<uint32_t> input(64), output(64, 0xdeadbeefu);
    for (uint32_t lane = 0; lane < 64; ++lane) input[lane] = lane;
    bpermute_oracle::Interpreter vm(module);
    vm.legacy_float_word_storage = true;
    const auto values = vm.run_buffers(64, {{0, input}, {1, output}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(values[lane], 0u) << "lane=" << lane;
}
}   // namespace
TEST(PartialVccWord, AnApproximateScalarSourceCannotSupplyTheOldComparisonBits) {
    const auto code = scalar_predicate(true);
    const auto module = recompile_valu(code.data(), code.size(), 1, 3);
    EXPECT_TRUE(module.empty()) << "READFIRST's predicate lineage needs an exact service";
    if (!module.empty()) zero_scalar_predicate_output(module);
}
TEST(PartialVccWord, ARealConstantScalarSourceSuppliesTheSamePredicateConsumer) {
    const auto code = scalar_predicate(false);
    const auto module = recompile_valu(code.data(), code.size(), 1, 3);
    ASSERT_FALSE(module.empty());
    zero_scalar_predicate_output(module);
}
TEST(PartialVccWord, AnExistingPortableServiceDoesNotInventANewRecoveryRoute) {
    auto code = scalar_predicate(true);
    code.insert(code.end() - 1, {0xe0702000u, 0x80020300u});
    EXPECT_TRUE(compile(code, 0x5135001090ull).empty());
    // The old exact service still produces without a partial replacement. A builder flag alone
    // cannot prove it runs in every body/slice, so this new route deliberately declines READFIRST.
    code[3] = 0xbf800000u;
    auto expected = std::vector<uint32_t>(64);
    expected[0] = 1;
    output_is(code, expected, 0x5135001091ull);
}
namespace {
std::vector<uint32_t> implicit_scc_predicate(bool carry, bool supplied) {
    auto code = scalar_predicate(false);
    code[1] = carry ? 0x821480a8u : 0x8514a9a8u;
    if (supplied) code.insert(code.begin() + 1, 0xbf06a8a8u);   // real CMP_EQ_U32 40,40: SCC=true
    return code;
}
void implicit_scc_output(const std::vector<uint32_t>& module, uint32_t selected) {
    std::vector<uint32_t> input(64), output(64, 0xdeadbeefu);
    for (uint32_t lane = 0; lane < 64; ++lane) input[lane] = lane;
    bpermute_oracle::Interpreter vm(module);
    vm.legacy_float_word_storage = true;
    const auto values = vm.run_buffers(64, {{0, input}, {1, output}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(values[lane], lane == selected ? 1u : 0u) << "lane=" << lane;
}
}   // namespace
TEST(PartialVccWord, InitialSccStorageDoesNotAuthenticateASelectPredicate) {
    const auto code = implicit_scc_predicate(false, false);
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 1, 3).empty());
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, InitialSccStorageDoesNotAuthenticateACarryPredicate) {
    const auto code = implicit_scc_predicate(true, false);
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 1, 3).empty());
    EXPECT_TRUE(proved(code).empty());
}
TEST(PartialVccWord, ARealSccProducerSuppliesTheSelectAndCarryPredicate) {
    for (bool carry : {false, true}) {
        const auto code = implicit_scc_predicate(carry, true);
        const auto module = recompile_valu(code.data(), code.size(), 1, 3);
        ASSERT_FALSE(module.empty());
        implicit_scc_output(module, carry ? 41u : 40u);
        auto false_producer = code;
        false_producer[1] = 0xbf06a9a8u;   // actual CMP_EQ_U32 40,41: SCC=false
        const auto false_module =
            recompile_valu(false_producer.data(), false_producer.size(), 1, 3);
        ASSERT_FALSE(false_module.empty());
        implicit_scc_output(false_module, carry ? 40u : 41u);
    }
}
TEST(PartialVccWord, AnExplicitInitialSccWordDoesNotAuthenticateComparisonBits) {
    auto code = scalar_predicate(false);
    code[1] = 0xbe9403fdu;   // S_MOV_B32 s20,SCC: explicit numeric DATA read of absent entry SCC
    EXPECT_TRUE(proved(code).empty());
    EXPECT_TRUE(recompile_valu(code.data(), code.size(), 1, 3).empty());
    code.insert(code.begin() + 1, 0xbf06a8a8u);
    const auto module = recompile_valu(code.data(), code.size(), 1, 3);
    ASSERT_FALSE(module.empty());
    // Actual SCC=true becomes scalar1; comparison only has a low bit, removed by LO=0.
    zero_scalar_predicate_output(module);
}
TEST(PartialVccWord, AFlagWithoutAConsumedReadfirstServiceCannotAuthenticateThePredicate) {
    auto code = scalar_predicate(true);
    code.insert(code.end() - 1, {0xe0702000u, 0x80020300u});
    code.pop_back();   // no END: the builder flag cannot force the existing exact service route
    const auto module = compile(code, 0x5135001100ull);
    EXPECT_TRUE(module.empty());
    if (!module.empty()) {
        bpermute_oracle::Interpreter vm(module);
        const auto output = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(output[lane], 0u) << "lane=" << lane;
    }
    code[1] = 0xbe940380u;   // genuine scalar0 keeps the same prefix-store API producing
    output_is(code, std::vector<uint32_t>(64, 0u), 0x5135001101ull);
}
