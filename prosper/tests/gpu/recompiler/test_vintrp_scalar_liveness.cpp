// #5091: ordinary interpolation cannot keep an unrelated scalar mask live at a
// structured merge. A false live claim rejects real draws; a false dead claim can
// discard a surviving mask. Keep implicit EXEC/M0 and unknown forms unproved.
#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::fragment_compare_kill;
constexpr uint32_t interp(uint32_t opcode, uint32_t dst = 4, uint32_t src = 0, uint32_t attr = 0,
                          uint32_t channel = 0) {
    return 0xc8000000u | (opcode << 16u) | (dst << 18u) | (attr << 10u) | (channel << 8u) | src;
}
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), result), code.size());
    return result;
}
bool dead(const std::vector<uint32_t>& code, int reg) {
    return sgpr_dead_at_merge(decode(code), 0, reg);
}
std::vector<uint32_t> fragment(bool defined_mask = false, bool read_mask = false) {
    // Genuine FragCoord inputs feed the comparison. Only the active arm writes s20;
    // both paths restore EXEC before interpolation and export the original v0/v1.
    std::vector<uint32_t> code;
    if (defined_mask) code.insert(code.end(), {fixture::compare(), fixture::compare_sd(20)});
    code.insert(code.end(), {0x7d820300u, 0xbe983c6au, 0xbf880002u, fixture::compare(),
                             fixture::compare_sd(20), fixture::mov(126, 24), interp(0), interp(1)});
    if (read_mask) {
        code.push_back(fixture::mov(126, 20));
        code.push_back(fixture::mov(126, 193)); // restore all Wave32 lanes before EXP
    }
    code.insert(code.end(), {0xf800180fu, 0x01000100u, fixture::end});
    return code;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address,
                              bool supplied_inputs = true) {
    const auto inputs = fixture::position_inputs();
    return recompile_fragment(code.data(), code.size(), nullptr,
                              supplied_inputs ? &inputs : nullptr, UINT32_MAX, nullptr, true,
                              {RecompileDiagnosticStage::Fragment, address});
}
void retain_module(const std::vector<uint32_t>& module, const char* name) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): inherited before tests; no test changes this.
    const char* directory = std::getenv("PROSPER_STRUCTURED_MASK_JOIN_SPV_DIRECTORY");
    if (!directory) return;
    std::ofstream out(std::filesystem::path(directory) / name, std::ios::binary);
    ASSERT_TRUE(out.good());
    out.write(reinterpret_cast<const char*>(module.data()),
              static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    ASSERT_TRUE(out.good());
}
} // namespace

TEST(VintrpScalarLiveness, PublishedFormsHaveOnlyVectorOperands) {
    for (uint32_t opcode : {0u, 1u, 2u}) {
        const auto ins = decode({interp(opcode), fixture::end});
        ASSERT_EQ(ins.size(), 2u);
        EXPECT_EQ(ins[0].fmt, Rdna2Format::VINTRP);
        EXPECT_EQ(ins[0].opcode, opcode);
        EXPECT_EQ(ins[0].n_src, 1);
        EXPECT_EQ(ins[0].src[0].kind, OperandKind::VGPR);
        EXPECT_EQ(ins[0].dst.kind, OperandKind::VGPR);
        EXPECT_EQ(ins[0].vintrp_attr, 0u);
        EXPECT_EQ(ins[0].vintrp_chan, 0u);
    }
}

TEST(VintrpScalarLiveness, P1P2AndMovDoNotReadOrdinaryScalarWords) {
    for (uint32_t opcode : {0u, 1u, 2u})
        for (int reg : {0, 1, 2, 3, 105, 106, 107})
            EXPECT_TRUE(dead({interp(opcode), fixture::end}, reg))
                << "opcode=" << opcode << " scalar=" << reg;
}

TEST(VintrpScalarLiveness, LaterLowAndHighPhysicalReadsRemainLive) {
    for (uint32_t opcode : {0u, 1u, 2u})
        for (uint32_t reg : {0u, 1u, 2u, 3u, 106u, 107u})
            EXPECT_FALSE(dead({interp(opcode), fixture::mov(30, reg), fixture::end}, reg))
                << "opcode=" << opcode << " scalar=" << reg;
}

TEST(VintrpScalarLiveness, BothConditionalSuccessorsMustBeDead) {
    EXPECT_FALSE(dead({interp(0), 0xbf880001u, fixture::mov(30, 0), fixture::end}, 0));
    EXPECT_TRUE(dead({interp(1), 0xbf880001u, fixture::mov(30, 4), fixture::end}, 0));
}

TEST(VintrpScalarLiveness, M0ExecAndUnnamedTargetsRemainUnproved) {
    for (uint32_t opcode : {0u, 1u, 2u})
        for (int reg : {-1, 108, 124, 126, 127, 253})
            EXPECT_FALSE(dead({interp(opcode), fixture::end}, reg))
                << "opcode=" << opcode << " scalar=" << reg;
}

TEST(VintrpScalarLiveness, ReservedOpcodeAndParameterFieldsRemainUnproved) {
    EXPECT_FALSE(dead({interp(3), fixture::end}, 0));
    EXPECT_FALSE(dead({interp(0, 4, 0, 33), fixture::end}, 0));
    EXPECT_FALSE(dead({interp(2, 4, 3), fixture::end}, 0));
}

TEST(VintrpScalarLiveness, InconsistentDecodedFormsRemainUnproved) {
    const auto original = decode({interp(0), fixture::end});
    for (int mutation = 0; mutation < 13; ++mutation) {
        auto ins = original;
        auto& in = ins[0];
        switch (mutation) {
            case 0: in.src[0] = {OperandKind::SGPR, 0}; break;
            case 1: in.dst = {OperandKind::SGPR, 0}; break;
            case 2: in.n_src = 2; break;
            case 3: in.len_dwords = 2; break;
            case 4: in.words[0] ^= 1u; break;
            case 5: in.has_modifier = true; break;
            case 6: in.has_sdwa = true; break;
            case 7: in.has_dpp = true; break;
            case 8: in.sdst = {OperandKind::SGPR, 0}; break;
            case 9: in.has_literal = true; break;
            case 10: in.src[0].value = -1; break;
            case 11: in.dst.value = 256; break;
            case 12: in.vintrp_chan = 4; break;
        }
        EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 0)) << "mutation=" << mutation;
    }
}

TEST(VintrpScalarLiveness, LaterUnknownFormatAndMaskConsumerStillBlock) {
    auto unknown = decode({interp(0), fixture::end});
    unknown[1].fmt = Rdna2Format::Unknown;
    unknown[1].is_end = false;
    EXPECT_FALSE(sgpr_dead_at_merge(unknown, 0, 0));
    EXPECT_FALSE(dead({interp(1), 0x02000080u, fixture::end}, 106));
    const auto ins = decode({interp(2), fixture::end});
    EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 0, ScalarMergeProof::MaskDomainOnly));
    EXPECT_TRUE(sgpr_dead_at_merge(ins, 0, 106, ScalarMergeProof::MaskDomainOnly));
}

TEST(VintrpScalarLiveness, GenuineFragmentInputsAllowDeadArmMaskAcrossInterpolation) {
    const auto module = compile(fragment(), 0x509101u);
    ASSERT_FALSE(module.empty());
    retain_module(module, "interpolation-fragment-dead-mask.spv");
}

TEST(VintrpScalarLiveness, MissingFragmentInputsCannotAuthenticateArmCompare) {
    EXPECT_TRUE(compile(fragment(), 0x509102u, false).empty());
    EXPECT_NE(last_terminal_reject_reason(0x509102u).find("fabricated-fresh-mask-source"),
              std::string::npos);
}

TEST(VintrpScalarLiveness, MissingSavedMaskStillRefusesItsLaterConsumer) {
    EXPECT_TRUE(compile(fragment(false, true), 0x509103u).empty());
    // Native Wave32 checks its B32 domains before the B64 saved-mask merger.
    EXPECT_NE(last_terminal_reject_reason(0x509103u).find("live b32 mask domain differs"),
              std::string::npos)
        << last_terminal_reject_reason(0x509103u);
    const auto module = compile(fragment(true, true), 0x509104u);
    ASSERT_FALSE(module.empty())
        << "a genuinely defined skipped-edge mask supplies the matched control";
    retain_module(module, "interpolation-fragment-defined-mask.spv");
}
} // namespace prosper::gpu
