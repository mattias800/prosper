// #4886: an absent ordinary VGPR or preserved old word must not acquire physical lane-mask
// authority through READFIRSTLANE. These project-owned streams use the production entry points;
// defined zero, actual entry values and complete replacements are separate positive controls.
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/compute_quad_or.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
using Words = std::vector<uint32_t>;
enum class Stage { Fragment, Compute };
std::atomic<uint64_t> next_address{0xa4886001ULL};
thread_local std::string terminal_reason;

Words concatenate(std::initializer_list<Words> pieces) {
    Words result;
    for (const auto& piece : pieces) result.insert(result.end(), piece.begin(), piece.end());
    return result;
}

Words compile(Stage stage, const Words& body, bool guarded_phases = false) {
    const uint64_t address = next_address.fetch_add(1, std::memory_order_relaxed);
    const Words project = stage == Stage::Fragment ? Words{0xbefe042cu} : Words{0x87fec12cu};
    const Words tail = stage == Stage::Fragment
                           ? Words{0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
                                   0xf800000fu, 0x03020100u, 0xbf810000u}
                           : Words{0x7e0602f2u, 0xe0702000u, 0x80020300u, 0xbf810000u};
    Words code = concatenate({{0xbefe04c1u}, body, {0xbead03c1u}, project, tail});
    if (guarded_phases) {
        code.insert(code.begin() + 1, {0xbe820380u, 0xbf068002u, 0xbf840000u});
        // A genuine uniform SCC guard selects the whole barrier-separated body or its END.
        code[3] |= static_cast<uint32_t>(code.size() - 5u);
    }
    ShaderResourceTable table;
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    table.resources.push_back(output);
    if (stage == Stage::Fragment) {
        Words module =
            recompile_fragment(code.data(), code.size(), &table, nullptr, UINT32_MAX, nullptr,
                               false, {RecompileDiagnosticStage::Fragment, address});
        terminal_reason = last_terminal_reject_reason(address);
        return module;
    }
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    prosper::test::complete_compute_launch(config, 2);
    // s4 is the actual workgroup X id. Two complete workgroups make both SCC edges observable;
    // a literal selector would let scalar folding legitimately remove the missing edge.
    config.user_sgprs.resize(4);
    config.tgid_x_en = true;
    config.threads_x = 128;
    Words module = recompile_compute(code.data(), code.size(), &table, config,
                                     {RecompileDiagnosticStage::Compute, address});
    terminal_reason = last_terminal_reject_reason(address);
    return module;
}

void expect_refused(const Words& body) {
    for (Stage stage : {Stage::Fragment, Stage::Compute}) {
        SCOPED_TRACE(stage == Stage::Fragment ? "fragment" : "compute");
        EXPECT_TRUE(compile(stage, body).empty()) << terminal_reason;
        EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
            << terminal_reason;
    }
}

void expect_admitted(const Words& body) {
    for (Stage stage : {Stage::Fragment, Stage::Compute}) {
        SCOPED_TRACE(stage == Stage::Fragment ? "fragment" : "compute");
        EXPECT_FALSE(compile(stage, body).empty()) << terminal_reason;
    }
}
}   // namespace

TEST(VectorWordDefinedness, EncodingsNameTheAbsentSourceAndScalarConsumer) {
    const Words code{0x7e0c0314u, 0x7e580506u};   // v6=v20; s44=first(v6)
    std::vector<Rdna2Inst> instructions;
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    ASSERT_EQ(instructions.size(), 2u);
    EXPECT_EQ(instructions[0].src[0].kind, OperandKind::VGPR);
    EXPECT_EQ(instructions[0].src[0].value, 20);
    EXPECT_EQ(instructions[0].dst.value, 6);
    EXPECT_EQ(instructions[1].opcode, 2u);
    // VOP1's decoded VDST field retains its generic VGPR tag; opcode 2 writes the scalar number.
    EXPECT_EQ(instructions[1].dst.kind, OperandKind::VGPR);
    EXPECT_EQ(instructions[1].dst.value, 44);
}

TEST(VectorWordDefinedness, DirectAbsentVectorCannotSupplyMaskWord) {
    expect_refused({0x7e580514u});   // s44=first(v20), no entry or preceding writer
}

TEST(VectorWordDefinedness, CopyOfAbsentVectorCannotSupplyMaskWord) {
    expect_refused({0x7e0c0314u, 0x7e580506u});
}

TEST(VectorWordDefinedness, PartialWriteCannotInventPreservedOldBits) {
    expect_refused({0x7e0c02f9u, 0x00051480u, 0x7e580506u});
}

TEST(VectorWordDefinedness, PredicatedWriteCannotInventInactiveOldLanes) {
    expect_refused({0xbefe0480u, 0x7e0c02c1u, 0xbefe04c1u, 0x7e580506u});
}

TEST(VectorWordDefinedness, GenuineWrittenZeroRemainsAValue) {
    expect_admitted({0x7e0c0280u, 0x7e580506u});
}

TEST(VectorWordDefinedness, GenuineCopyRemainsAValue) {
    expect_admitted({0x7e280280u, 0x7e0c0314u, 0x7e580506u});   // v20=0
}

TEST(VectorWordDefinedness, CompleteOverwriteRetiresTheMissingGeneration) {
    expect_admitted({0x7e0c0314u, 0x7e0c0280u, 0x7e580506u});
}

TEST(VectorWordDefinedness, PartialWriteKeepsGenuineOldBits) {
    // S0_S is clear: the SDWA field 128 names v128, not inline zero. Supply the actual source.
    expect_admitted({0x7f000280u, 0x7e0c02c1u, 0x7e0c02f9u, 0x00051480u, 0x7e580506u});
}

TEST(VectorWordDefinedness, PredicatedWriteKeepsGenuineOldLanes) {
    expect_admitted({0x7e0c0280u, 0xbefe0480u, 0x7e0c02c1u, 0xbefe04c1u, 0x7e580506u});
}

TEST(VectorWordDefinedness, GenuineComputeEntryVectorRemainsAValue) {
    EXPECT_FALSE(compile(Stage::Compute, {0x7e580500u}).empty()) << terminal_reason;
}

TEST(VectorWordDefinedness, SdwaScalarZeroReallyIsAnInlineSource) {
    const Words code{0x7e0c02f9u, 0x00851480u};
    std::vector<Rdna2Inst> instructions;
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    ASSERT_EQ(instructions.size(), 1u);
    EXPECT_EQ(instructions[0].src[0].kind, OperandKind::InlineInt);
    EXPECT_EQ(instructions[0].src[0].value, 0);
    EXPECT_EQ(instructions[0].sdwa_dst_sel, 4u);
    EXPECT_EQ(instructions[0].sdwa_dst_unused, 2u);
}

TEST(VectorWordDefinedness, GenuinePartialSourceCannotInventOldBits) {
    expect_refused({0x7e0c02f9u, 0x00851480u, 0x7e580506u});
}

TEST(VectorWordDefinedness, MissingPartialSourceCannotBorrowGenuineOldBits) {
    expect_refused({0x7e0c02c1u, 0x7e0c02f9u, 0x00051480u, 0x7e580506u});
}

TEST(VectorWordDefinedness, GenuinePartialSourceAndOldWordRemainAValue) {
    expect_admitted({0x7e0c02c1u, 0x7e0c02f9u, 0x00851480u, 0x7e580506u});
}

TEST(VectorWordDefinedness, FullSdwaReplacementRetiresOldBits) {
    // BYTE_0 source extraction into DWORD/PAD writes every destination bit.
    expect_admitted({0x7e0c0314u, 0x7e0c02f9u, 0x00800680u, 0x7e580506u});
}

TEST(VectorWordDefinedness, HalfConvertCannotInventTheOtherHalf) {
    expect_refused({0x7e0c14f2u, 0x7e580506u});   // cvt_f16_f32 v6,1.0
}

TEST(VectorWordDefinedness, HalfConvertKeepsGenuineOldBits) {
    expect_admitted({0x7e0c0280u, 0x7e0c14f2u, 0x7e580506u});
}

TEST(VectorWordDefinedness, ImplicitAccumulatorCannotInventOldData) {
    expect_refused({0x7e0402f2u, 0x3e0c04f2u, 0x7e580506u});   // mac v6,1.0,v2
}

TEST(VectorWordDefinedness, ImplicitAccumulatorKeepsGenuineOldData) {
    expect_admitted({0x7e0402f2u, 0x7e0c0280u, 0x3e0c04f2u, 0x7e580506u});
}

TEST(VectorWordDefinedness, OneArmWriterCannotSupplyTheMissingEdge) {
    EXPECT_TRUE(
        compile(Stage::Compute, {0xbf800000u, 0xbf068004u, 0xbf840001u, 0x7e0c0280u, 0x7e580506u})
            .empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, BothArmWritersSupplyTheCurrentWord) {
    EXPECT_FALSE(compile(Stage::Compute, {0xbf800000u, 0xbf068004u, 0xbf840002u, 0x7e0c0280u,
                                          0xbf820001u, 0x7e0c02c1u, 0x7e580506u})
                     .empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, OneNarrowedExecEdgePreservesMissingLanes) {
    EXPECT_TRUE(
        compile(Stage::Compute, {0xbf800000u, 0xbf068004u, 0xbf840002u, 0xbefe0480u, 0xbf820001u,
                                 0xbf800000u, 0x7e0c02c1u, 0xbefe04c1u, 0x7e580506u})
            .empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, FullExecResetAfterTheJoinRetiresMissingLanes) {
    EXPECT_FALSE(
        compile(Stage::Compute, {0xbf800000u, 0xbf068004u, 0xbf840002u, 0xbefe0480u, 0xbf820001u,
                                 0xbf800000u, 0xbefe04c1u, 0x7e0c02c1u, 0x7e580506u})
            .empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, MissingRelaySurvivesBarrierPhaseReload) {
    EXPECT_TRUE(compile(Stage::Compute, {0x7e0c0314u, 0xbf8a0000u, 0x7e580506u}, true).empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, GenuineRelaySurvivesBarrierPhaseReload) {
    EXPECT_FALSE(
        compile(Stage::Compute, {0x7e280280u, 0x7e0c0314u, 0xbf8a0000u, 0x7e580506u}, true).empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, CompleteReplacementAfterBarrierRetiresMissingRelay) {
    EXPECT_FALSE(
        compile(Stage::Compute, {0x7e0c0314u, 0xbf8a0000u, 0x7e0c0280u, 0x7e580506u}, true).empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, MissingRelaySurvivesRepeatedBarrierReloads) {
    EXPECT_TRUE(
        compile(Stage::Compute, {0x7e0c0314u, 0xbf8a0000u, 0xbf8a0000u, 0x7e580506u}, true).empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, GenuineRelaySurvivesRepeatedBarrierReloads) {
    EXPECT_FALSE(compile(Stage::Compute,
                         {0x7e280280u, 0x7e0c0314u, 0xbf8a0000u, 0xbf8a0000u, 0x7e580506u}, true)
                     .empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, NopCannotInventItsDecodedDestination) {
    expect_refused({0x7e0c0080u, 0x7e580506u});
}

TEST(VectorWordDefinedness, NopKeepsItsGenuineOldDestination) {
    expect_admitted({0x7e0c0280u, 0x7e0c0080u, 0x7e580506u});
}

TEST(VectorWordDefinedness, NativeInterpolationP1CannotInventItsDestination) {
    // Native P1 is an emitter no-op; its encoded source/destination are not actual varying reads.
    EXPECT_TRUE(compile(Stage::Fragment, {0xc8180014u, 0x7e580506u}).empty()) << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, NativeInterpolationP1KeepsGenuineOldData) {
    EXPECT_FALSE(compile(Stage::Fragment, {0x7e0c0280u, 0xc8180014u, 0x7e580506u}).empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, NativeInterpolationP2UsesTheGenuineVarying) {
    // Native P2 consumes the declared varying, not the encoded (absent here) v20/P1 result.
    EXPECT_FALSE(compile(Stage::Fragment, {0xc8190014u, 0x7e580506u}).empty()) << terminal_reason;
}

TEST(VectorWordDefinedness, SwizzleUsesOnlyItsActualSingleWordSource) {
    EXPECT_FALSE(compile(Stage::Compute, {0xd8d4001fu, 0x06000000u, 0x7e580506u}).empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, SwizzleCannotInventItsActualSource) {
    EXPECT_TRUE(compile(Stage::Compute, {0xd8d4001fu, 0x06000014u, 0x7e580506u}).empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, BpermuteUsesOnlyItsAddressAndDataWords) {
    EXPECT_FALSE(
        compile(Stage::Compute, {0x7e100280u, 0xdacc0000u, 0x06000800u, 0x7e580506u}).empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, BpermuteCannotInventItsActualData) {
    EXPECT_TRUE(compile(Stage::Compute, {0xdacc0000u, 0x06001400u, 0x7e580506u}).empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, LdsReadUsesOnlyItsActualAddressWord) {
    EXPECT_FALSE(compile(Stage::Compute, {0x7e100280u, 0x7e180287u, 0xd8340000u, 0x00000c08u,
                                          0xd8d80000u, 0x06000008u, 0x7e580506u})
                     .empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, LdsReadCannotInventItsActualAddress) {
    EXPECT_TRUE(compile(Stage::Compute, {0xd8d80000u, 0x06000008u, 0x7e580506u}).empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, RelativeReadCannotInventTheWordBeyondItsEncodedBase) {
    // v20 is genuine; M0=1 selects the absent architectural v21 beyond the storage inventory.
    expect_refused({0x7e280280u, 0xbefc0381u, 0x7e0c8714u, 0x7e580506u});
}

TEST(VectorWordDefinedness, RelativeReadKeepsItsGenuineSelectedBaseWord) {
    expect_admitted({0x7e280280u, 0xbefc0380u, 0x7e0c8714u, 0x7e580506u});
}

TEST(VectorWordDefinedness, RelativeReadKeepsItsGenuineSelectedOffsetWord) {
    expect_admitted({0x7e280280u, 0x7e2a0280u, 0xbefc0381u, 0x7e0c8714u, 0x7e580506u});
}

TEST(VectorWordDefinedness, RelativeReadCannotTrustAFabricatedSelector) {
    expect_refused({0x7e280280u, 0xbefc033cu, 0x7e0c8714u, 0x7e580506u});
}

TEST(VectorWordDefinedness, DynamicRelativeReadCannotAssumeEveryCandidateExists) {
    EXPECT_TRUE(
        compile(Stage::Compute, {0x7e280280u, 0x7e2a0280u, 0xbefc0304u, 0x7e0c8714u, 0x7e580506u})
            .empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, FullReplacementRetiresAnUnprovedRelativeRead) {
    EXPECT_FALSE(compile(Stage::Compute, {0x7e280280u, 0x7e2a0280u, 0xbefc0304u, 0x7e0c8714u,
                                          0x7e0c0280u, 0x7e580506u})
                     .empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, RelativeReadRetainsTheSelectedWordsMissingOrigin) {
    expect_refused({0x7e280280u, 0x7e2a031eu, 0xbefc0381u, 0x7e0c8714u, 0x7e580506u});
}

TEST(VectorWordDefinedness, RelativeReadCannotTrustAnEqualValuedMarkedSelectorMerge) {
    EXPECT_TRUE(compile(Stage::Compute, {0x7e280280u, 0xbe910380u, 0xbf068004u, 0xbf840001u,
                                         0xbe91033cu, 0xbefc0311u, 0x7e0c8714u, 0x7e580506u})
                    .empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, RelativeReadKeepsAnEqualValuedGenuineSelectorMerge) {
    EXPECT_FALSE(compile(Stage::Compute, {0x7e280280u, 0xbe910380u, 0xbf068004u, 0xbf840001u,
                                          0xbe910380u, 0xbefc0311u, 0x7e0c8714u, 0x7e580506u})
                     .empty())
        << terminal_reason;
}

TEST(VectorWordDefinedness, RelativeReadCannotBorrowOneArmsSelectedWord) {
    EXPECT_TRUE(compile(Stage::Compute, {0x7e280280u, 0xbf068004u, 0xbf840001u, 0x7e2a0280u,
                                         0xbefc0381u, 0x7e0c8714u, 0x7e580506u})
                    .empty())
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("scalar-fabricated-lane-mask"), std::string::npos)
        << terminal_reason;
}

TEST(VectorWordDefinedness, RelativeReadKeepsBothArmsGenuineSelectedWord) {
    EXPECT_FALSE(
        compile(Stage::Compute, {0x7e280280u, 0xbf068004u, 0xbf840002u, 0x7e2a0280u, 0xbf820001u,
                                 0x7e2a02c1u, 0xbefc0381u, 0x7e0c8714u, 0x7e580506u})
            .empty())
        << terminal_reason;
}

namespace {
Words compile_quad_projection(prosper::test::compute_quad_or::Shape shape, uint32_t native) {
    namespace fixture = prosper::test::compute_quad_or;
    auto code = fixture::program(shape);
    code.resize(code.size() - 3);   // replace the fixture's final store/END with a mask consumer
    const Words tail{0x7e580503u, 0xbead03c1u, 0x87fec12cu, 0xe0702000u, 0x80020300u, 0xbf810000u};
    code.insert(code.end(), tail.begin(), tail.end());
    const auto address = next_address.fetch_add(1, std::memory_order_relaxed);
    const auto module = fixture::compile(code, fixture::table(), fixture::config(native), address);
    terminal_reason = last_terminal_reject_reason(address);
    return module;
}
}   // namespace

TEST(VectorWordDefinedness, ProvenComplementaryQuadWritersSupplyTheCurrentResult) {
    for (uint32_t native : {0u, 64u}) {
        SCOPED_TRACE(native);
        EXPECT_FALSE(
            compile_quad_projection(prosper::test::compute_quad_or::Shape::Complement, native)
                .empty())
            << terminal_reason;
    }
}

TEST(VectorWordDefinedness, IncompleteQuadWritersCannotCertifyTheirResult) {
    for (uint32_t native : {0u, 64u}) {
        SCOPED_TRACE(native);
        EXPECT_TRUE(
            compile_quad_projection(prosper::test::compute_quad_or::Shape::MissingArm, native)
                .empty());
        EXPECT_NE(terminal_reason.find("dpp-quad-or-source-uninitialized"), std::string::npos)
            << terminal_reason;
    }
}

TEST(VectorWordDefinedness, FullExecOrSaveexecKeepsCompleteWriterGenuine) {
    // OR_SAVEEXEC with zero keeps a known full mask; the complete write initializes v20.
    expect_admitted({0xbea02580u, 0x7e280280u, 0xbefe04c1u, 0x7e580514u});
}

TEST(VectorWordDefinedness, EmptyOrn2SaveexecCannotInventInactiveOldLanes) {
    // ORN2_SAVEEXEC with zero empties EXEC; restoring it cannot supply the skipped definition.
    expect_refused({0xbea02880u, 0x7e280280u, 0xbefe04c1u, 0x7e580514u});
}

TEST(VectorWordDefinedness, EmptyOrSaveexecCannotClaimACompleteWriter) {
    expect_refused({0xbefe0480u, 0xbea02580u, 0x7e280280u, 0xbefe04c1u, 0x7e580514u});
}
