// An automatic PS entry word is a real hardware input, not unloaded zero. Only a closed
// interpolation-only entry use can consume the existing native fragment Input service.
#include "fixtures/native_fragment_spirv_oracle.hpp"
#include "fixtures/native_pixel_parameter.hpp"
#include "gpu/recompiler/rdna2_native_parameter_base.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <stdexcept>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::native_pixel_parameter;
std::vector<Rdna2Inst> decode(const fixture::Words& words) {
    std::vector<Rdna2Inst> result;
    rdna2_walk(words.data(), words.size(), result);
    return result;
}
auto plan(const fixture::Words& words, uint32_t count = 30) {
    return native_parameter_base_plan(fixture::mapping(count).parameter_launch, decode(words));
}
} // namespace

TEST(NativePixelParameter, ExactRecordedPrefixLocatesTheFirstSystemWord) {
    for (uint32_t count = 0; count <= 32; ++count)
        EXPECT_EQ(pixel_parameter_sgpr({true, fixture::rsrc2(count)}), count);
    EXPECT_FALSE(pixel_parameter_sgpr({}));
    EXPECT_FALSE(pixel_parameter_sgpr({false, 60}));
    EXPECT_FALSE(pixel_parameter_sgpr({true, fixture::rsrc2(33)}));
    EXPECT_FALSE(pixel_parameter_sgpr({true, fixture::rsrc2(63)}));
    auto malformed = fixture::mapping();
    malformed.parameter_launch = {false, 60};
    EXPECT_TRUE(fixture::compile(fixture::program(), malformed).module.empty());
}

TEST(NativePixelParameter, ActualFragmentInputAndReusedScalarLifetimeExecute) {
    for (const uint32_t count : {0u, 16u, 20u, 30u, 32u}) {
        SCOPED_TRACE(count);
        const auto code = fixture::program(count);
        const auto instructions = decode(code);
        ASSERT_EQ(instructions.size(), 11u);
        EXPECT_EQ(instructions[0].dst.value, 124);
        EXPECT_EQ(instructions[0].src[0].value, count);
        EXPECT_EQ(instructions[1].dst.value, count);
        EXPECT_EQ(instructions[7].src[0].value, count);
        for (const bool wave32 : {false, true}) {
            const auto result = fixture::compile(code, fixture::mapping(count), wave32);
            ASSERT_FALSE(result.module.empty()) << result.reason;
            EXPECT_GT(arbitrary_placeholder_reads(result.module).total(), 0u);
            for (const float red : {0.125f, 0.25f, 0.5f, 0.875f}) {
                const auto r = std::bit_cast<uint32_t>(red);
                const auto actual = prosper::test::native_fragment_words(
                    result.module, {{0, {r, 0, 0, 0x3f800000u}}},
                    {{15, {0x3f000000u, 0x3f000000u, 0, 0x3f800000u}}}, wave32 ? 32 : 64);
                EXPECT_EQ(actual, (std::array<uint32_t, 4>{r, 0x3f000000u, r, 0x3f800000u}));
            }
            EXPECT_THROW(prosper::test::native_fragment_words(
                             result.module, {}, {{15, {0x3f000000u, 0x3f000000u, 0, 0x3f800000u}}},
                             wave32 ? 32 : 64),
                         std::runtime_error);
        }
    }
}

TEST(NativePixelParameter, SmoothP1P2ReadsTheSameRealNativeInput) {
    for (const bool wave32 : {false, true}) {
        const auto result =
            fixture::compile(fixture::program(30, true), fixture::mapping(30, true), wave32);
        ASSERT_FALSE(result.module.empty()) << result.reason;
        EXPECT_GT(arbitrary_placeholder_reads(result.module).total(), 0u);
        for (const uint32_t red : {0x3e000000u, 0x3f200000u}) {
            const auto actual = prosper::test::native_fragment_words(
                result.module, {{0, {red, 0, 0, 0x3f800000u}}},
                {{15, {0x3f000000u, 0x3f000000u, 0, 0x3f800000u}}}, wave32 ? 32 : 64);
            EXPECT_EQ(actual, (std::array<uint32_t, 4>{red, 0x3f000000u, red, 0x3f800000u}));
        }
    }
}

TEST(NativePixelParameter, ASuccessfulOriginalRouteKeepsItsExactModuleBytes) {
    auto code = fixture::program();
    code[6] = 0x7d840f07u; // replace the scalar-relay comparison with v7 == v7
    code.erase(code.begin() + 4, code.begin() + 6);   // remove READLANE, retaining the legacy route
    auto unknown = fixture::mapping();
    unknown.parameter_launch = {};
    for (const bool wave32 : {false, true}) {
        const auto original = fixture::compile(code, unknown, wave32);
        const auto known = fixture::compile(code, fixture::mapping(), wave32);
        ASSERT_FALSE(original.module.empty()) << original.reason;
        ASSERT_FALSE(known.module.empty()) << known.reason;
        EXPECT_EQ(arbitrary_placeholder_reads(original.module).total(), 0u);
        EXPECT_EQ(arbitrary_placeholder_reads(known.module).total(), 0u);
        EXPECT_EQ(known.module, original.module);
    }
}

TEST(NativePixelParameter, UnknownAndWrongPrefixesCannotBorrowTheService) {
    const auto code = fixture::program();
    auto inputs = fixture::mapping();
    inputs.parameter_launch = {};
    const auto unknown = fixture::compile(code, inputs);
    EXPECT_TRUE(unknown.module.empty());
    EXPECT_NE(unknown.reason.find("origin=unknown-launch word=s30"), std::string::npos);
    const auto wrong = fixture::compile(code, fixture::mapping(31));
    EXPECT_TRUE(wrong.module.empty());
    EXPECT_NE(wrong.reason.find("origin=launch-input word=s30"), std::string::npos);
}

TEST(NativePixelParameter, OrdinaryUserDataAndNumericM0ObserversRemainHard) {
    auto code = fixture::program();
    code[3] = 0x7e0e021du;   // v7 = s29, an actual missing user-SGPR input
    const auto user = fixture::compile(code, fixture::mapping());
    EXPECT_TRUE(user.module.empty());
    EXPECT_NE(user.reason.find("origin=launch-input word=s29"), std::string::npos);
    code = fixture::program();
    code[9] = 0x7e12027cu;   // v9 = M0, a numerical observer
    EXPECT_FALSE(plan(code));
    const auto numeric = fixture::compile(code, fixture::mapping());
    EXPECT_TRUE(numeric.module.empty());
}

TEST(NativePixelParameter, EntryLifetimeRequiresAnUnconditionalCompleteOverwrite) {
    auto code = fixture::program();
    ASSERT_TRUE(plan(code));
    code[1] =
        0xbf800000u;   // remove the overwrite; later numerical source is still the entry input
    EXPECT_FALSE(plan(code));
    EXPECT_TRUE(fixture::compile(code, fixture::mapping()).module.empty());
    code = fixture::program();
    code[1] = 0xbe9e05f0u;   // s_cmov_b32 s30,0.5: SCC may preserve the original word
    EXPECT_FALSE(plan(code));
    code = fixture::program();
    code.insert(code.begin(), 0xbe9e0380u);   // an earlier scalar write destroys entry identity
    EXPECT_FALSE(plan(code));
}

TEST(NativePixelParameter, M0WritesLdsAndRelativeUsesAreOutsideTheClosedService) {
    for (const uint32_t instruction : {0xbefc0380u, 0x7e12027cu, 0x7e1202feu}) {
        auto code = fixture::program();
        code.insert(code.begin() + 3, instruction);
        EXPECT_FALSE(plan(code)) << std::hex << instruction;
    }
    auto original = decode(fixture::program());
    ASSERT_TRUE(native_parameter_base_plan(fixture::mapping().parameter_launch, original));
    auto ds = original;
    ds[3].fmt = Rdna2Format::DS;
    EXPECT_FALSE(native_parameter_base_plan(fixture::mapping().parameter_launch, ds));
    auto redirect = original;
    redirect[3].mubuf_lds = true;
    EXPECT_FALSE(native_parameter_base_plan(fixture::mapping().parameter_launch, redirect));
}

TEST(NativePixelParameter, EncodedScalarMemoryPayloadReadsStayOutsideTheService) {
    // LLVM gfx1013: s_store_dword / s_atomic_add s30, s[0:1], 0. SDATA is encoded
    // in the destination field, but consumes the real entry word before its overwrite.
    for (const std::array<uint32_t, 2> payload : {
             std::array<uint32_t, 2>{0xf4400780u, 0xfa000000u},
             std::array<uint32_t, 2>{0xf6080780u, 0xfa000000u},
         }) {
        auto code = fixture::program();
        code.insert(code.begin() + 1, payload.begin(), payload.end());
        const auto instructions = decode(code);
        ASSERT_EQ(instructions[1].fmt, Rdna2Format::SMEM);
        ASSERT_EQ(instructions[1].dst.value, 30);
        EXPECT_FALSE(plan(code));
    }
}

TEST(NativePixelParameter, EncodedWaitCounterSourcesCannotBorrowTheEntryWord) {
    // LLVM gfx1013: s_waitcnt_{vscnt,vmcnt,expcnt,lgkmcnt} s30, 0. These source
    // fields are outside the explicit source array; do not certify their numeric use.
    for (const uint32_t wait : {0xbb9e0000u, 0xbc1e0000u, 0xbc9e0000u, 0xbd1e0000u}) {
        auto code = fixture::program();
        code.insert(code.begin() + 1, wait);
        const auto instructions = decode(code);
        ASSERT_EQ(instructions[1].fmt, Rdna2Format::SOPK);
        ASSERT_EQ(instructions[1].dst.value, 30);
        EXPECT_FALSE(plan(code));
        code = fixture::program();
        code.insert(code.begin() + 2, wait);   // the complete overwrite supplies this threshold
        EXPECT_TRUE(plan(code));
    }
}

TEST(NativePixelParameter, EntryReentryAndUninventoriedControlNeverAcquireAuthority) {
    auto code = fixture::program();
    code.insert(code.end() - 1, 0xbf82fff2u);   // branch back to the entry M0 load
    EXPECT_FALSE(plan(code));
    code = fixture::program();
    code.insert(code.end() - 1, 0xbf820010u);   // branch outside the original decoded prefix
    EXPECT_FALSE(plan(code));
    code = fixture::program();
    code.insert(code.begin(), 0xbf820000u);   // even a fallthrough branch ends entry dominance
    EXPECT_FALSE(plan(code));
}

TEST(NativePixelParameter, CertificateBindsOriginalBytesAndTheNativeFragmentContext) {
    const auto original = decode(fixture::program());
    const auto certificate = plan(fixture::program());
    ASSERT_NE(certificate, nullptr);
    EXPECT_TRUE(native_parameter_base_region(*certificate, original));
    EXPECT_TRUE(native_parameter_base_load(*certificate, original.front()));
    auto changed = original;
    changed[0].src[0].value = 29;
    EXPECT_FALSE(native_parameter_base_load(*certificate, changed[0]));
    EXPECT_FALSE(native_parameter_base_region(*certificate, changed));
    RegState rs;
    SpirvCompute compute;
    compute.is_compute = true;
    compute.native_parameter_base = certificate;
    EXPECT_FALSE(native_parameter_base_region(compute, original));
    EXPECT_FALSE(emit_native_parameter_base_load(compute, rs, original.front()));
}

TEST(NativePixelParameter, TheServicePublishesNoScalarBitsAndCannotCrossExecutionDomains) {
    const auto original = decode(fixture::program());
    const auto certificate = plan(fixture::program());
    ASSERT_NE(certificate, nullptr);
    for (unsigned variant = 0; variant < 7; ++variant) {
        SpirvCompute b;
        b.begin_fragment();
        b.native_parameter_base = certificate;
        RegState rs;
        rs.sreg[124] = b.uconst(0xdeadbeefu);
        if (variant == 1) b.is_compute = true;
        if (variant == 2) b.is_vertex = true;
        if (variant == 3) b.ngg_workgroup_shell = true;
        if (variant == 4) b.ngg_one_lane = true;
        if (variant == 5) b.guest_stage = GuestShaderStage::Compute;
        if (variant == 6) b.physical_domain = PhysicalExecutionDomain::Workgroup;
        EXPECT_EQ(emit_native_parameter_base_load(b, rs, original.front()), variant == 0);
        EXPECT_EQ(native_parameter_base_region(b, original), variant == 0);
        EXPECT_EQ(rs.sreg.contains(124), variant != 0);
        EXPECT_FALSE(rs.sreg.contains(30)) << "the automatic input has no numerical publication";
    }
}

TEST(NativePixelParameter, AnUnwrittenColourOutputCannotPassTheOracle) {
    auto result = fixture::compile(fixture::program(), fixture::mapping());
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bpermute_oracle::Interpreter vm(result.module, 4);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    uint32_t output = 0;
    for (const auto& global : vm.globals)
        if (global.a.size() == 3 && global.a[2] == 3 &&
            vm.decorations[{global.a[1], 30}] == std::vector<uint32_t>{0})
            output = global.a[1];
    ASSERT_NE(output, 0u);
    unsigned removed = 0;
    for (size_t pc = 5; pc < result.module.size();) {
        const auto count = result.module[pc] >> 16;
        ASSERT_GT(count, 0u);
        if ((result.module[pc] & 0xffffu) == 62 && result.module[pc + 1] == output) {
            result.module.erase(result.module.begin() + pc, result.module.begin() + pc + count);
            ++removed;
        } else {
            pc += count;
        }
    }
    ASSERT_EQ(removed, 1u);
    EXPECT_THROW(prosper::test::native_fragment_words(
                     result.module, {{0, {0x3e000000u, 0, 0, 0x3f800000u}}},
                     {{15, {0x3f000000u, 0x3f000000u, 0, 0x3f800000u}}}, 64),
                 std::runtime_error);
}
