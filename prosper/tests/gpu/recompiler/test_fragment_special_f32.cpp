#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_special_f32_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_special_f32;
uint32_t dumped = 0;
void retain(const FragmentResourcePacketProgram& p) {
    // Existing diagnostic-only CPU SOURCE retention switch, never a guest execution selector.
    const char* root = std::getenv("PROSPER_RESOURCE_PACKET_SPV_DIRECTORY");
    if (!root || !*root || p.packet.spirv.empty()) return;
    const std::filesystem::path directory(root);
    std::filesystem::create_directories(directory);
    std::ofstream f(directory / ("special_f32_" + std::to_string(dumped++) + ".spv"),
                    std::ios::binary);
    f.write(reinterpret_cast<const char*>(p.packet.spirv.data()), p.packet.spirv.size() * 4);
    f.close();
    ASSERT_TRUE(f);
}
struct Execution {
    FragmentResourcePacketProgram program;
    FragmentResourcePacketResult result;
    std::vector<uint32_t> raw;
};
Execution execute(const FragmentResourcePacket& p) {
    Execution e;
    e.program = recompile_fragment_resource_packet(p, {RecompileDiagnosticStage::Fragment, 0x4224});
    EXPECT_FALSE(e.program.packet.spirv.empty()) << e.program.packet.rejection;
    if (e.program.packet.spirv.empty()) return e;
    retain(e.program);
    bpermute_oracle::Interpreter vm(e.program.packet.spirv);
    e.raw = vm.run_packet(e.program.packet.input_words, e.program.packet.output_words);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    e.result = decode_fragment_resource_packet(e.program, e.raw, true, p.device.device_identity);
    return e;
}
void expect(const FragmentResourcePacket& p, uint32_t op) {
    const auto e = execute(p);
    ASSERT_TRUE(e.result.rejection.empty()) << e.result.rejection;
    const auto wanted = fixture::expected(p, op);
    ASSERT_EQ(e.result.exports.size(), wanted.size());
    for (size_t i = 0; i < wanted.size(); ++i)
        EXPECT_EQ(e.result.exports[i], wanted[i]) << "word=" << i;
}
void reject(const FragmentResourcePacket& p, const char* reason) {
    const auto e = recompile_fragment_resource_packet(p);
    EXPECT_TRUE(e.packet.spirv.empty());
    EXPECT_TRUE(e.packet.input_words.empty());
    EXPECT_TRUE(e.packet.output_words.empty());
    EXPECT_EQ(e.packet.rejection, reason);
}
} // namespace

TEST(FragmentSpecialF32, EveryNormalExponentAndGuestRounding) {
    for (uint32_t op : {0x2au, 0x2eu, 0x33u})
        for (uint32_t mode = 0; mode < 4; ++mode) {
            for (uint32_t batch = 0; batch < 4; ++batch) {
                std::array<uint32_t, 64> words{};
                for (uint32_t lane = 0; lane < 64; ++lane) {
                    const uint32_t e = std::min(254u, 1 + batch * 64 + lane);
                    constexpr uint32_t fractions[]{1, 0x155555, 0x3fffff, 0x7fffff};
                    words[lane] = (e << 23) | fractions[lane % 4];
                    if (op == 0x2a && lane & 1) words[lane] |= 0x80000000u;
                }
                SCOPED_TRACE("op=" + std::to_string(op) + " round=" + std::to_string(mode) +
                             " batch=" + std::to_string(batch));
                expect(fixture::packet(op, words, uint8_t(0x30 | mode)), op);
            }
        }
}
TEST(FragmentSpecialF32, OpcodeFlushOverridesAllDenormalModesAndInactivePeer) {
    for (uint32_t op : {0x2au, 0x2eu, 0x33u})
        for (uint32_t denorm = 0; denorm < 4; ++denorm)
            for (uint32_t round = 0; round < 4; ++round) {
                const auto mode = uint8_t((denorm << 4) | round);
                SCOPED_TRACE("op=" + std::to_string(op) + " mode=" + std::to_string(mode));
                expect(fixture::packet(op, fixture::rails(op), mode, round == 3), op);
            }
}
TEST(FragmentSpecialF32, KnownProducingTrapFieldsAndWholeProgramObservation) {
    for (uint32_t op : {0x2au, 0x2eu, 0x33u}) {
        auto p = fixture::packet(op, fixture::rails(op));
        auto missing = p;
        missing.entry_facts = {};
        reject(missing, "packet-special-f32-producing-rsrc2-unavailable");
        missing = p;
        missing.entry_facts.rsrc2_available = false;
        reject(missing, "packet-special-f32-producing-rsrc2-unavailable");
        missing = p;
        missing.entry_facts.observed = false;
        reject(missing, "packet-special-f32-producing-rsrc2-unavailable");
        // Unknown is not mode0, and a retained field is not proof of some different raw launch.
        missing = p;
        missing.invocation.float_mode.available = false;
        reject(missing, "packet-launch-state-invalid");   // noncanonical unknown retains 0x30
        missing.invocation.float_mode = {};   // genuine canonical absence, never a known mode0
        reject(missing, "packet-f32-launch-mode-or-flags-unavailable");
        missing = p;
        missing.launch_rsrc1.value ^= 1u << 12;
        reject(missing, "packet-f32-launch-rsrc1-association-mismatch");
        p.entry_facts.rsrc2 = 1u << 6;   // handler exists, floating enables all disabled
        expect(p, op);
        p.entry_facts.rsrc2 |= 7u << 22;   // unrelated EXCP_EN high three bits are not FP enables
        expect(p, op);
        for (uint32_t bit = 16; bit < 22; ++bit) {
            auto enabled = p;
            enabled.entry_facts.rsrc2 |= 1u << bit;
            reject(enabled, "packet-special-f32-enabled-exception-handler-unimplemented");
            enabled.entry_facts.rsrc2 &= ~(1u << 6);
            expect(enabled, op);   // no handler: ignored
        }
        p.launch_rsrc1.value |= 1u << 22;
        reject(p, "packet-special-f32-debug-handler-unimplemented");
        p.entry_facts.rsrc2 &= ~(1u << 6);
        expect(p, op);   // DEBUG only works with trap_en
        // Genuine decoded SETREG is rejected even when skipped structurally at entry. A no-op
        // twin preserves original special opcode and raw sink, so unsupported scan is not vacuous.
        auto dead = p;
        dead.invocation.guest_code.insert(dead.invocation.guest_code.begin(),
                                          {0xbf820001u, 0xb9930001u});
        reject(dead, "packet-mode-write");
        dead.invocation.guest_code[1] = 0xbf800000u;
        expect(dead, op);
    }
}
TEST(FragmentSpecialF32, ExceptionalHighLaneStickyFirstPcAndTransactionalConsumer) {
    for (uint32_t op : {0x2au, 0x2eu, 0x33u})
        for (uint32_t bad : {0x7fc00001u, 0x7f800001u, 0xffc00002u}) {
            auto inputs = fixture::rails();
            inputs[63] = bad;
            auto p = fixture::packet(op, inputs);
            // Add another actual failing special instruction AFTER a reached EXP. No later failure
            // can overwrite the original PC0, and no early reached record may escape the transaction.
            p.invocation.guest_code.insert(p.invocation.guest_code.end() - 1,
                                           {0x7e000000u | (9u << 17) | (0x33u << 9) | 256u});
            std::vector<Rdna2Inst> ins;
            ASSERT_TRUE(
                rdna2_walk(p.invocation.guest_code.data(), p.invocation.guest_code.size(), ins));
            ASSERT_EQ(ins.front().pc, 0u);
            ASSERT_EQ(ins.front().opcode, op);
            auto e = execute(p);
            EXPECT_TRUE(e.result.exports.empty());
            EXPECT_EQ(e.result.failure, FragmentPacketRuntimeFailure::SpecialNanOrNegativeRoot);
            EXPECT_EQ(e.result.lane, 63u);
            EXPECT_EQ(e.result.pc, ins.front().pc);
            EXPECT_EQ(e.result.rejection,
                      "packet-runtime-special-f32-nan-or-negative-root-unimplemented");
            auto inactive = p;
            inactive.invocation.exec_mask &= ~(uint64_t{1} << 63);
            expect(inactive,
                   op);   // unconsumed exceptional value is not a failure or a fabricated write
            if (!e.raw.empty()) {
                EXPECT_TRUE(decode_fragment_resource_packet(e.program, e.raw, false,
                                                            p.device.device_identity)
                                .exports.empty());
                auto forged = e.raw;
                forged[e.program.status_offset + 63 * 3 + 1] = 0x12345678;
                EXPECT_EQ(decode_fragment_resource_packet(e.program, forged, true,
                                                          p.device.device_identity)
                              .rejection,
                          "packet-status-record-invalid");
            }
        }
    for (uint32_t op : {0x2eu, 0x33u})
        for (uint32_t bad : {0xbf800000u, 0xff800000u}) {
            auto inputs = fixture::rails();
            inputs[40] = bad;
            const auto e = execute(fixture::packet(op, inputs));
            EXPECT_TRUE(e.result.exports.empty());
            EXPECT_EQ(e.result.lane, 40u);
            EXPECT_EQ(e.result.pc, 0u);
            EXPECT_EQ(e.result.failure, FragmentPacketRuntimeFailure::SpecialNanOrNegativeRoot);
        }
}
TEST(FragmentSpecialF32, OriginalOperandAndTypedIntegerOutputAreLoadBearing) {
    auto p = fixture::packet(0x2e, fixture::rails());
    const auto e = execute(p);
    ASSERT_TRUE(e.result.rejection.empty());
    auto changed = p;
    changed.invocation.vgprs[0].words[63] = 0x40400001;
    const auto other = execute(changed);
    EXPECT_EQ(e.program.packet.spirv, other.program.packet.spirv);
    EXPECT_NE(e.result.exports, other.result.exports);
    // An actual malformed uint64 division result must fail the typed SOURCE evaluator, not
    // fall through to host arithmetic or assume the compiler's types are correct.
    auto mutant = e.program.packet.spirv;
    uint32_t boolean = 0, wide = 0;
    size_t site = 0;
    for (size_t pc = 5; pc < mutant.size();) {
        const uint32_t n = mutant[pc] >> 16, op = mutant[pc] & 0xffff;
        ASSERT_GT(n, 0u);
        ASSERT_LE(n, mutant.size() - pc);
        if (op == 20) boolean = mutant[pc + 1];
        if (op == 21 && n == 4 && mutant[pc + 2] == 64 && mutant[pc + 3] == 0)
            wide = mutant[pc + 1];
        if (op == 134 && mutant[pc + 1] == wide && !site) site = pc;
        pc += n;
    }
    ASSERT_NE(site, 0u);
    ASSERT_NE(boolean, 0u);
    mutant[site + 1] = boolean;
    bpermute_oracle::Interpreter vm(mutant);
    (void)vm.run_packet(e.program.packet.input_words, e.program.packet.output_words);
    EXPECT_EQ(vm.error, "integer result type");
}
TEST(FragmentSpecialF32, DirectRsqIsNotComposedRoundedSqrtAndRcp) {
    // This distinguishes our correctly-rounded direct software algorithm. A one-ULP difference
    // alone is NOT proof that the composed approximation violates AMD's published 1-ULP bound.
    constexpr uint32_t input = 0x3f801eef;
    EXPECT_EQ(fixture::oracle(0x2e, input, 0x30), 0x3f7fe117u);
    EXPECT_EQ(fixture::oracle(0x2a, fixture::oracle(0x33, input, 0x30), 0x30), 0x3f7fe116u);
    auto words = fixture::rails();
    words.fill(input);
    expect(fixture::packet(0x2e, words), 0x2e);
}
TEST(FragmentSpecialF32, OriginalScalarLiteralAndInlineSourceForms) {
    std::array<uint32_t, 64> words{};
    words.fill(0x40000000);
    for (uint32_t op : {0x2au, 0x2eu, 0x33u}) {
        const auto original = fixture::packet(op, words);
        auto scalar = original;
        scalar.invocation.sgprs.emplace_back(16, 0x40000000);
        scalar.invocation.guest_code[0] = (scalar.invocation.guest_code[0] & ~0x1ffu) | 16u;
        std::vector<Rdna2Inst> ins;
        rdna2_walk(scalar.invocation.guest_code.data(), scalar.invocation.guest_code.size(), ins);
        ASSERT_EQ(ins.front().src[0].kind, OperandKind::SGPR);
        expect(scalar, op);
        auto missing = scalar;
        missing.invocation.sgprs.clear();
        reject(missing, "packet-sgpr-read-before-definition");
        auto literal = original;
        literal.invocation.guest_code[0] = (literal.invocation.guest_code[0] & ~0x1ffu) | 255u;
        literal.invocation.guest_code.insert(literal.invocation.guest_code.begin() + 1, 0x40000000);
        ins.clear();
        rdna2_walk(literal.invocation.guest_code.data(), literal.invocation.guest_code.size(), ins);
        ASSERT_EQ(ins.front().len_dwords, 2u);
        ASSERT_EQ(ins.front().src[0].kind, OperandKind::Literal);
        expect(literal, op);
        auto inline_float = original;
        inline_float.invocation.guest_code[0] =
            (inline_float.invocation.guest_code[0] & ~0x1ffu) | 244u;
        ins.clear();
        rdna2_walk(inline_float.invocation.guest_code.data(),
                   inline_float.invocation.guest_code.size(), ins);
        ASSERT_EQ(ins.front().src[0].kind, OperandKind::InlineFloat);
        expect(inline_float, op);
        auto special = original;
        special.invocation.guest_code[0] = (special.invocation.guest_code[0] & ~0x1ffu) | 106u;
        reject(special, "packet-valu-op-unimplemented");   // no numeric cast of a physical VCC mask
    }
    auto unsupported = fixture::packet(0x2b, words);   // RCP_IFLAG is NOT RCP's exception contract
    reject(unsupported, "packet-valu-op-unimplemented");
    unsupported =
        fixture::packet(0x25, words);   // EXP/LOG/COS are deliberately separate obligations
    reject(unsupported, "packet-valu-op-unimplemented");
}
TEST(FragmentSpecialF32, FullOriginalPcSurvivesSkippedRegion) {
    const auto p = fixture::high_pc_failure();
    std::vector<Rdna2Inst> ins;
    rdna2_walk(p.invocation.guest_code.data(), p.invocation.guest_code.size(), ins);
    const auto first = std::find_if(ins.begin(), ins.end(),
                                    [](const auto& in) { return in.fmt == Rdna2Format::VOP1; });
    ASSERT_NE(first, ins.end());
    ASSERT_EQ(first->pc, 257u);
    ASSERT_EQ(first->opcode, 0x2eu);
    const auto e = execute(p);
    EXPECT_TRUE(e.result.exports.empty());
    EXPECT_EQ(e.result.lane, 63u);
    EXPECT_EQ(e.result.pc, first->pc);
    EXPECT_EQ(e.result.failure, FragmentPacketRuntimeFailure::SpecialNanOrNegativeRoot);
    auto twin = p;
    twin.invocation.vgprs[0].words[63] = 0x40400000;
    expect(twin, 0x2e);
}
