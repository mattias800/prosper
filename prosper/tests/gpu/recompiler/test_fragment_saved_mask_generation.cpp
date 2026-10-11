// #4956: a fresh fragment Bool mask must not inherit physical-word demand from other
// register generations. Losing real DATA demand corrupts masks after partial writes or relays.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fragment_predicate_oracle.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <string>
#include <vector>
namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000u;
constexpr uint32_t mov(uint32_t dst, uint32_t src, bool pair = false) {
    return 0xbe800000u | (dst << 16u) | ((pair ? 4u : 3u) << 8u) | src;
}
constexpr uint32_t vmov(uint32_t dst, uint32_t src) {
    return 0x7e000200u | (dst << 17u) | src;
}
std::vector<uint32_t> compare() {
    return {0x7c0200f9u, 0x06869880u};
}
std::vector<uint32_t> select(uint32_t root = 24u, uint32_t a = 128u, uint32_t c = 240u) {
    return {0xd5010000u, a | (c << 9u) | (root << 18u)};
}
void append(std::vector<uint32_t>& code, const std::vector<uint32_t>& words) {
    code.insert(code.end(), words.begin(), words.end());
}
size_t ballots(const std::vector<uint32_t>& words, size_t at = 0) {
    size_t count = 0;
    while (at < words.size()) {
        const size_t width = words[at] >> 16u;
        if (!width || width > words.size() - at) return SIZE_MAX;
        count += (words[at] & 0xffffu) == 339u;
        at += width;
    }
    return count;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter(float input = 0.5f) {
        b.wave_size = 64;
        b.begin_fragment();
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.scc = b.btrue();
        rs.vreg[0] = b.fconst(input);
    }
    bool run(const std::vector<uint32_t>& code, bool partial = false, bool dispatcher = false,
             const std::function<void(std::vector<Rdna2Inst>&)>& rewrite = {}) {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        if (rewrite) rewrite(ins);
        const auto exp = [](RegState&, const Rdna2Inst&) { return true; };
        const bool ok =
            dispatcher ? emit_cfg_state_machine(b, rs, ins, {}, nullptr, true, false, exp,
                                                code.data(), code.size(), partial ? b.btrue() : 0)
                       : emit_body(b, rs, ins, {}, nullptr, true, false, exp, code.data(),
                                   code.size(), nullptr, true, partial ? b.btrue() : 0);
        std::printf("OBS emitter ok=%d ballots=%zu required=%u reasons=0x%x root24=%d low=%d "
                    "high=%d reject_pc=%u reject=%s\n",
                    ok, ballots(b.code), b.fragment_required_subgroup_size, b.fragment_wave_reasons,
                    b.saved_mask_word_roots.contains(24), rs.sreg.contains(24),
                    rs.sreg.contains(25), b.stage_reject_pc, b.stage_reject_reason.c_str());
        return ok;
    }
};
std::vector<uint32_t> minimal(const std::vector<uint32_t>& tail = {}) {
    auto code = compare();
    append(code, select());
    append(code, tail);
    code.push_back(end);
    return code;
}
void expect_bool_only(Emitter& e, const std::vector<uint32_t>& code) {
    ASSERT_TRUE(e.run(code));
    EXPECT_EQ(ballots(e.b.code), 0u)
        << "fresh current Bool condition has no physical DATA observer";
    EXPECT_EQ(e.b.fragment_required_subgroup_size, 0u);
    EXPECT_EQ(e.b.fragment_wave_reasons & kFragmentWaveReasonWaveBallot, 0u);
}
void expect_numeric(Emitter& e, const std::vector<uint32_t>& code) {
    ASSERT_TRUE(e.run(code));
    EXPECT_EQ(ballots(e.b.code), 2u)
        << "both actual physical words must exist at original producer";
    EXPECT_EQ(e.b.fragment_required_subgroup_size, 64u);
    EXPECT_NE(e.b.fragment_wave_reasons & kFragmentWaveReasonWaveBallot, 0u);
}
void retain(const char* name, const std::vector<uint32_t>& words) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): inherited before process start; no test mutates it.
    const char* dir = std::getenv("PROSPER_SAVED_MASK_GENERATION_SPV_DIRECTORY");
    if (!dir) return;
    std::ofstream out(std::filesystem::path(dir) / name, std::ios::binary);
    ASSERT_TRUE(out.good());
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    ASSERT_TRUE(out.good());
}
}   // namespace
TEST(FragmentSavedMaskGeneration, FreshConditionNeedsNoPhysicalWords) {
    for (float v : {0.5f, 0.0f}) {
        SCOPED_TRACE(v);
        Emitter e(v);
        expect_bool_only(e, minimal());
    }
}
TEST(FragmentSavedMaskGeneration, EarlierNumericGenerationDoesNotDemandFreshMask) {
    Emitter e;
    std::vector<uint32_t> code{mov(24, 135), mov(25, 137), vmov(1, 24)};
    append(code, minimal());
    expect_bool_only(e, code);
}
TEST(FragmentSavedMaskGeneration, LaterCompleteReplacementKillsMaskNumericDemand) {
    Emitter e;
    expect_bool_only(e, minimal({mov(24, 128, true), vmov(1, 24), vmov(2, 25)}));
}
TEST(FragmentSavedMaskGeneration, BoolOnlyPairRelayNeedsNoPhysicalWords) {
    Emitter e;
    auto code = compare();
    code.push_back(mov(30, 24, true));
    append(code, select(30));
    code.push_back(end);
    expect_bool_only(e, code);
}
TEST(FragmentSavedMaskGeneration, ExistingCopyProofAlreadySeesBooleanOnlyCondition) {
    Emitter e;
    auto code = compare();
    const uint32_t copy_pc = static_cast<uint32_t>(code.size());
    code.push_back(mov(106, 24, true));
    append(code, select(106));
    code.push_back(end);
    ASSERT_TRUE(e.run(code));
    EXPECT_TRUE(e.b.fragment_bool_only_vcc_copy_pcs.contains(copy_pc))
        << "existing instruction-order generation proof already exempts only the true Bool "
           "condition";
    EXPECT_EQ(ballots(e.b.code), 0u);
}
TEST(FragmentSavedMaskGeneration, CurrentLowNumericObserverStillNeedsWords) {
    Emitter e;
    expect_numeric(e, minimal({vmov(1, 24)}));
}
TEST(FragmentSavedMaskGeneration, CurrentHighNumericObserverStillNeedsWords) {
    Emitter e;
    expect_numeric(e, minimal({vmov(1, 25)}));
}
TEST(FragmentSavedMaskGeneration, CndmaskDataSourceZeroStillNeedsWords) {
    Emitter e;
    auto code = compare();
    append(code, select(24, 24, 240));
    code.push_back(end);
    expect_numeric(e, code);
}
TEST(FragmentSavedMaskGeneration, CndmaskDataSourceOneStillNeedsWords) {
    Emitter e;
    auto code = compare();
    append(code, select(24, 128, 25));
    code.push_back(end);
    expect_numeric(e, code);
}
TEST(FragmentSavedMaskGeneration, RelayedNumericObserverStillNeedsOriginalWords) {
    Emitter e;
    auto code = compare();
    code.push_back(mov(30, 24, true));
    append(code, select(30));
    code.push_back(vmov(1, 31));
    code.push_back(end);
    expect_numeric(e, code);
    ASSERT_TRUE(e.rs.sreg.contains(25) && e.rs.sreg.contains(31));
    EXPECT_EQ(e.rs.sreg.at(25), e.rs.sreg.at(31));
}
TEST(FragmentSavedMaskGeneration, PartialReplacementPreservesOtherNumericWord) {
    Emitter e;
    auto code = compare();
    code.push_back(mov(24, 135));
    code.push_back(vmov(1, 25));
    code.push_back(end);
    expect_numeric(e, code);
    ASSERT_TRUE(e.rs.sreg.contains(25));
    EXPECT_EQ(e.rs.vreg.at(1), e.rs.sreg.at(25));
}
TEST(FragmentSavedMaskGeneration, PartialProducerCannotInventNumericWords) {
    Emitter e;
    EXPECT_FALSE(e.run(minimal({vmov(1, 25)}), true));
    EXPECT_EQ(ballots(e.b.code), 0u);
}
TEST(FragmentSavedMaskGeneration, OneArmNumericObserverStillDemandsGeneration) {
    Emitter e;
    auto code = compare();
    append(code, select());
    code.push_back(0xbf840001u);
    code.push_back(vmov(1, 25));
    code.push_back(mov(24, 128, true));
    code.push_back(end);
    expect_numeric(e, code);
}
TEST(FragmentSavedMaskGeneration, MinimalPublicModulesHaveNoPhysicalWidthContract) {
    uint64_t program = 0x1ee495602ull;
    for (uint32_t input : {240u, 128u}) {
        SCOPED_TRACE(input);
        std::vector<uint32_t> code{vmov(0, input)};
        append(code, compare());
        append(code, select());
        append(code, {0x7e0202ffu, 0x3f000000u, 0x7e0402ffu, 0x3f400000u, 0x7e0602f2u, 0xf800180fu,
                      0x03020100u, end});
        const auto module =
            recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                               false, {RecompileDiagnosticStage::Fragment, program++});
        retain(input == 240 ? "minimal-half.bin" : "minimal-zero.bin", code);
        retain(input == 240 ? "minimal-half.spv" : "minimal-zero.spv", module);
        std::printf("OBS minimal input=%u module_words=%zu ballots=%zu required=%u reasons=0x%x\n",
                    input, module.size(), ballots(module, 5),
                    fragment_spirv_required_subgroup_size(module),
                    fragment_spirv_required_subgroup_reasons(module));
        ASSERT_FALSE(module.empty());
        EXPECT_EQ(ballots(module, 5), 0u);
        EXPECT_EQ(fragment_spirv_required_subgroup_size(module), 0u);
    }
}
TEST(FragmentSavedMaskGeneration, DescriptorThenFreshConditionAddsNoBallotContract) {
    alignas(16) std::array<uint32_t, 128> data{};
    std::vector<uint32_t> code{0xbe940380u, 0xf4080600u, 0x28000000u,
                               0x8f1b801bu, 0xe0300000u, 0x80060000u};
    append(code, compare());
    append(code, select());
    append(code, {0x7e0202ffu, 0x3f000000u, 0x7e0402ffu, 0x3f400000u, 0x7e0602f2u, 0xf800180fu,
                  0x03020100u, end});
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    ASSERT_EQ(rdna2_raw_wide_data_loads(ins).size(), 0u);
    ShaderResourceTable table;
    ShaderResource r;
    r.cls = ResourceClass::VertexBuffer;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.binding = 4;
    for (const auto& in : ins)
        if (in.fmt == Rdna2Format::MUBUF) r.fetch_pc = in.pc;
    r.gpu_addr = reinterpret_cast<uint64_t>(data.data());
    r.size = sizeof(data);
    table.resources.push_back(r);
    const auto module =
        recompile_fragment(code.data(), code.size(), &table, nullptr, UINT32_MAX, nullptr, false,
                           {RecompileDiagnosticStage::Fragment, 0x1ee495601ull});
    retain("descriptor.bin", code);
    retain("descriptor.spv", module);
    std::printf("OBS public module_words=%zu ballots=%zu required=%u reasons=0x%x reject=%s\n",
                module.size(), ballots(module, 5), fragment_spirv_required_subgroup_size(module),
                fragment_spirv_required_subgroup_reasons(module),
                last_terminal_reject_reason(0x1ee495601ull).c_str());
    ASSERT_FALSE(module.empty());
    EXPECT_EQ(ballots(module, 5), 0u);
    EXPECT_EQ(fragment_spirv_required_subgroup_size(module), 0u);
    // An absent width contract has no reason marker; the parser returns its unknown sentinel.
    EXPECT_EQ(fragment_spirv_required_subgroup_reasons(module), UINT32_MAX);
}
TEST(FragmentSavedMaskGeneration, DynamicSccOneArmCannotProveCopyGeneration) {
    Emitter e;
    e.b.declare_cbufs();
    e.rs.sreg[20] = e.b.cbuf_load(e.b.uconst(0), 2);
    std::vector<uint32_t> code{0xbf068014u, 0xbf840002u};
    append(code, compare());
    const uint32_t copy_pc = static_cast<uint32_t>(code.size());
    code.push_back(mov(106, 24, true));
    append(code, select(106));
    code.push_back(end);
    EXPECT_FALSE(e.run(code));
    EXPECT_FALSE(e.b.fragment_bool_only_vcc_copy_pcs.contains(copy_pc))
        << "copy input has no all-path current-mask fact";
}
TEST(FragmentSavedMaskGeneration, DynamicSccPredefinedMaskKeepsCopyProof) {
    Emitter e;
    e.b.declare_cbufs();
    e.rs.sreg[20] = e.b.cbuf_load(e.b.uconst(0), 2);
    auto code = compare();
    append(code, {0xbf068014u, 0xbf840001u, vmov(2, 128)});
    const uint32_t copy_pc = static_cast<uint32_t>(code.size());
    code.push_back(mov(106, 24, true));
    append(code, select(106));
    code.push_back(end);
    ASSERT_TRUE(e.run(code));
    EXPECT_TRUE(e.b.fragment_bool_only_vcc_copy_pcs.contains(copy_pc));
    EXPECT_EQ(ballots(e.b.code), 0u);
}
TEST(FragmentSavedMaskGeneration, DynamicSccOneArmOverwriteCannotKillNumericDemand) {
    Emitter e;
    e.b.declare_cbufs();
    e.rs.sreg[20] = e.b.cbuf_load(e.b.uconst(0), 2);
    auto code = compare();
    append(code, {0xbf068014u, 0xbf840001u, mov(24, 128, true), vmov(1, 25), end});
    expect_numeric(e, code);
}
TEST(FragmentSavedMaskGeneration, CurrentBoolSaveSelectsActualLaneValues) {
    constexpr uint64_t pattern = 0x80000001abcdef01ull;
    Emitter e;
    e.rs.exec = constant_mask_bit(e.b, static_cast<uint32_t>(pattern),
                                  static_cast<uint32_t>(pattern >> 32u));
    std::vector<uint32_t> code{mov(24, 126, true)};
    append(code, select(24, 129, 137));
    code.push_back(end);
    ASSERT_TRUE(e.run(code));
    EXPECT_EQ(ballots(e.b.code), 0u);
    fragment_predicate_oracle::Values oracle(e.b);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto actual = oracle.evaluate(e.rs.vreg.at(0), lane);
        ASSERT_TRUE(actual.has_value()) << oracle.error;
        EXPECT_EQ(actual.value_or(UINT32_MAX), (pattern >> lane) & 1u ? 9u : 1u)
            << "guest lane " << lane;
    }
}
TEST(FragmentSavedMaskGeneration, ValueOracleRejectsUnsupportedOperation) {
    Emitter e;
    const uint32_t unsupported = e.b.ibin(Op_IAdd, e.b.uconst(1), e.b.uconst(2));
    fragment_predicate_oracle::Values oracle(e.b);
    EXPECT_FALSE(oracle.evaluate(unsupported, 0).has_value());
}
TEST(FragmentSavedMaskGeneration, RewrittenConsumerCannotInheritOriginalGenerationProof) {
    Emitter e;
    EXPECT_FALSE(e.run(minimal(), false, false, [](std::vector<Rdna2Inst>& ins) {
        for (auto& in : ins)
            if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101)
                in.src[0] = {OperandKind::SGPR, 24};
    }));
    EXPECT_EQ(ballots(e.b.code), 0u);
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-physical-word-unavailable");
}
TEST(FragmentSavedMaskGeneration, UnqualifiedExecutionDomainRetainsConservativeWords) {
    Emitter e;
    e.b.physical_domain = PhysicalExecutionDomain::Workgroup;
    expect_numeric(e, minimal());
}
TEST(FragmentSavedMaskGeneration, ChangedStreamCannotReseedPhysicalProducerAuthority) {
    Emitter e;
    ASSERT_TRUE(e.run(minimal()));
    const size_t before = ballots(e.b.code);
    EXPECT_FALSE(e.run(minimal({vmov(1, 24)})));
    EXPECT_EQ(ballots(e.b.code), before);
    EXPECT_EQ(e.b.stage_reject_reason, "saved-mask-physical-word-unavailable");
}
TEST(FragmentSavedMaskGeneration, UnknownEffectDoesNotSupplyGenerationDeadness) {
    Emitter e;
    auto code = minimal();
    code.insert(code.end() - 1, 0xffffffffu);
    EXPECT_FALSE(e.run(code));
    EXPECT_EQ(ballots(e.b.code), 0u);
}
TEST(FragmentSavedMaskGeneration, ConditionalHighReplacementDoesNotKillNumericDemand) {
    Emitter e;
    e.b.declare_cbufs();
    e.rs.sreg[20] = e.b.cbuf_load(e.b.uconst(0), 2);
    auto code = compare();
    append(code, select());
    append(code, {0xbf068014u, 0xbe800000u | (25u << 16u) | (kSop1OpcodeCmovB32 << 8u) | 135u,
                  vmov(1, 25), end});
    // The current fragment emitter refuses CMOV. Its conditional writer must still not
    // become a definite kill of the preceding mask generation.
    EXPECT_FALSE(e.run(code));
    EXPECT_EQ(ballots(e.b.code), 2u);
}
TEST(FragmentSavedMaskGeneration, DynamicOneArmMaskPreservesRawBackingObligation) {
    alignas(16) std::array<uint32_t, 128> data{};
    uint64_t program = 0x1ee495608ull;
    for (bool before_branch : {true, false}) {
        SCOPED_TRACE(before_branch);
        std::vector<uint32_t> code{
            vmov(0, 240), mov(20, 128), 0xf4080600u, 0x28000000u, 0x8f1b801bu,
            0xe0300000u,  0x80060000u,  0x7e280500u};   // s20 <- actual resolved buffer value
        if (before_branch) append(code, compare());
        append(code, {0xbf068014u, before_branch ? 0xbf840001u : 0xbf840002u});
        if (before_branch)
            code.push_back(vmov(2, 128));
        else
            append(code, compare());
        append(code, select());
        append(code, {0x7e020280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, end});
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        const auto raw = rdna2_raw_wide_data_loads(ins);
        EXPECT_EQ(raw, before_branch ? std::vector<uint32_t>{} : std::vector<uint32_t>{2u});
        ShaderResourceTable table;
        ShaderResource r;
        r.cls = ResourceClass::VertexBuffer;
        r.format = DataFormat::Uint32;
        r.num_components = 1;
        r.binding = 4;
        for (const auto& in : ins)
            if (in.fmt == Rdna2Format::MUBUF) r.fetch_pc = in.pc;
        r.gpu_addr = reinterpret_cast<uint64_t>(data.data());
        r.size = sizeof(data);
        table.resources.push_back(r);
        const auto module =
            recompile_fragment(code.data(), code.size(), &table, nullptr, UINT32_MAX, nullptr,
                               false, {RecompileDiagnosticStage::Fragment, program});
        if (before_branch) {
            ASSERT_FALSE(module.empty());
            EXPECT_EQ(ballots(module, 5), 0u);
            EXPECT_EQ(fragment_spirv_required_subgroup_size(module), 0u);
            EXPECT_EQ(fragment_spirv_required_subgroup_reasons(module), UINT32_MAX);
            retain("dynamic-predefined-mask.spv", module);
        } else {
            EXPECT_TRUE(module.empty());
            const auto reason = last_terminal_reject_reason(program);
            EXPECT_NE(reason.find("pc=2 "), std::string::npos) << reason;
            EXPECT_NE(reason.find("op=0x2"), std::string::npos) << reason;
        }
        ++program;
    }
}
}   // namespace prosper::gpu
