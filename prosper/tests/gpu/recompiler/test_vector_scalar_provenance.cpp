// #4787: fabricated scalar words must not become genuine lane-mask bits after a VGPR hop.
// These independent instruction streams exercise actual fragment/compute compilation; positive
// controls retain real words and complete overwrites. No entry authority or lane selection is added.
#include "fixtures/checked_compute_launch.hpp"
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
std::atomic<uint64_t> next_address{0xa4787001ULL};
thread_local std::string terminal_reason;
enum class Stage { Fragment, Compute };

Words cat(std::initializer_list<Words> parts) {
    Words words;
    for (const auto& part : parts) words.insert(words.end(), part.begin(), part.end());
    return words;
}

Words prefix() {
    // The later dispatcher/CMPX comparisons consume v0/v1 before the payload export writes
    // them. Supply genuine words at full entry EXEC so those comparisons test the relay below.
    return {0x7e000280u, 0x7e020280u, 0x7e0a0280u, 0x7e000505u, 0xd4c20002u, 0x00010080u};
}
Words spill_source() {
    return {0xd7610015u, 0x00010032u, 0xd7610015u, 0x00010233u};
}
Words scalar_reload() {
    return {0xd760002cu, 0x00010115u, 0xd760002du, 0x00010315u};
}
Words vector_copy() {
    return {0x7e0c022cu, 0x7e0e022du};
}   // v6=s44; v7=s45
Words firstlane() {
    return {0x7e580506u, 0x7e5a0507u};
}   // s44=first(v6); s45=first(v7)
Words restore(Stage stage = Stage::Fragment) {
    // MOV projection is admitted in fragment; AND projects the same pair in compute.
    return stage == Stage::Fragment ? Words{0xbefe042cu} : Words{0x87fec12cu};
}
Words tail(Stage stage) {
    if (stage == Stage::Compute) return {0x7e0602f2u, 0xe0702000u, 0x80020300u, 0xbf810000u};
    return {0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
            0xf800000fu, 0x03020100u, 0xbf810000u};
}
Words compile(Stage stage, const Words& body) {
    const uint64_t address = next_address.fetch_add(1, std::memory_order_relaxed);
    const Words code = cat({body, tail(stage)});
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
    prosper::test::complete_compute_launch(config);
    Words module = recompile_compute(code.data(), code.size(), &table, config,
                                     {RecompileDiagnosticStage::Compute, address});
    terminal_reason = last_terminal_reject_reason(address);
    return module;
}
Words relay(const Words& source = {}) {
    return cat({prefix(), source, spill_source(), scalar_reload(), vector_copy()});
}
void expect_refused(Stage stage, const Words& body,
                    const char* cause = "scalar-fabricated-lane-mask") {
    EXPECT_TRUE(compile(stage, body).empty());
    EXPECT_NE(terminal_reason.find(cause), std::string::npos) << terminal_reason;
}
Words dispatcher() {
    // Crossed VCCZ regions force the shipping fragment CFG dispatcher.
    return {0x7e040280u, 0x7e060280u, 0x7e080280u, 0x7e0a0280u, 0x7c020300u,
            0xbf860003u, 0x7c020300u, 0xbf860002u, 0x7e040281u, 0x7e060281u,
            0x7c020300u, 0xbf860002u, 0x7e080281u, 0xbf820001u, 0x7e0a0281u};
}
Words loop(Stage stage, const Words& rewrite) {
    const Words pre = {0x7e0c02c1u, 0x7e0e02c1u, 0xbe960380u, 0xbe970383u};
    const Words read = cat({firstlane(), restore(stage), {0xbefe04c1u}});
    const uint32_t body_size = static_cast<uint32_t>(read.size() + rewrite.size() + 2);
    const Words test = {0xbf0a1716u, 0xbf840000u | body_size};
    const Words latch = {0x80168116u, 0xbf820000u | ((0x10000u - body_size - 2u) & 0xffffu)};
    return cat({prefix(), pre, test, read, rewrite, latch});
}
}   // namespace

TEST(VectorScalarProvenance, EncodingsReallyCrossVectorAndScalarDomains) {
    const Words code = cat({vector_copy(), firstlane()});
    std::vector<Rdna2Inst> decoded;
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
    ASSERT_EQ(decoded.size(), 4u);
    EXPECT_EQ(decoded[0].opcode, 1u);
    EXPECT_EQ(decoded[0].dst.value, 6);
    EXPECT_EQ(decoded[0].src[0].kind, OperandKind::SGPR);
    EXPECT_EQ(decoded[0].src[0].value, 44);
    EXPECT_EQ(decoded[2].opcode, 2u);
    EXPECT_EQ(decoded[2].dst.value, 44);
    EXPECT_EQ(decoded[2].src[0].kind, OperandKind::VGPR);
    EXPECT_EQ(decoded[2].src[0].value, 6);
}

TEST(VectorScalarProvenance, FabricatedSpillCannotBeLaunderedThroughReadfirstlane) {
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        expect_refused(stage, cat({relay(), firstlane(), restore(stage)}));
}

TEST(VectorScalarProvenance, GenuineExecWordsKeepCompilingThroughTheSameHop) {
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        EXPECT_FALSE(
            compile(stage, cat({relay({0xbeb2047eu}), firstlane(), restore(stage)})).empty())
            << terminal_reason;
}

TEST(VectorScalarProvenance, FullOverwriteEndsTheFabricatedVectorLifetime) {
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        EXPECT_FALSE(compile(stage, cat({relay(),
                                         {0xbefe04c1u, 0x7e0c02c1u, 0x7e0e02c1u},
                                         firstlane(),
                                         restore(stage)}))
                         .empty())
            << terminal_reason;
}

TEST(VectorScalarProvenance, PredicatedOverwriteDoesNotCleanUnwrittenLanes) {
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        expect_refused(
            stage,
            cat({relay(), {0xbefe0480u, 0x7e0c02c1u, 0x7e0e02c1u}, firstlane(), restore(stage)}));
}

TEST(VectorScalarProvenance, PartialOverwriteDoesNotCleanPreservedBits) {
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        expect_refused(stage, cat({relay(),
                                   {0x7e0c02f9u, 0x00051480u, 0x7e0e02f9u, 0x00051480u},
                                   firstlane(),
                                   restore(stage)}));
}

TEST(VectorScalarProvenance, VectorAluAndCopiesRetainSourceProvenance) {
    // v5=0; v6=v6^v5; v7=v7^v5; v8=v6; v9=v7; s44=first(v8); s45=first(v9).
    const Words chain = {0x7e0a0280u, 0x3a0c0b06u, 0x3a0e0b07u, 0x7e100306u,
                         0x7e120307u, 0x7e580508u, 0x7e5a0509u};
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        expect_refused(stage, cat({relay(), chain, restore(stage)}));
}

TEST(VectorScalarProvenance, OrdinaryDynamicReadlaneKeepsFabricatedProvenance) {
    // A real scalar lane selector does not establish the value selected from the vector.
    const Words reads = {0xbe900380u, 0xd760002cu, 0x00002106u, 0xd760002du, 0x00002107u};
    expect_refused(Stage::Compute, cat({relay(), reads, restore(Stage::Compute)}));
    EXPECT_FALSE(
        compile(Stage::Compute, cat({relay({0xbeb2047eu}), reads, restore(Stage::Compute)}))
            .empty())
        << terminal_reason;
}

TEST(VectorScalarProvenance, DispatcherReloadDoesNotTurnVectorBitsIntoGenuineWords) {
    const Words body = cat({relay(), dispatcher(), firstlane(), restore()});
    EXPECT_TRUE(compile(Stage::Fragment, body).empty());
    const std::string reason = terminal_reason;
    EXPECT_NE(reason.find("mode=unresolved-operand"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=" + std::to_string(body.size() - 1u) + " "), std::string::npos)
        << reason;
    EXPECT_FALSE(
        compile(Stage::Fragment, cat({relay({0xbeb2047eu}), dispatcher(), firstlane(), restore()}))
            .empty())
        << terminal_reason;
}

TEST(VectorScalarProvenance, OneArmVectorOverwriteCannotEraseTheOtherArmsMark) {
    // Unknown scalar branch: only its taken arm overwrites v6/v7 with genuine words.
    const Words branch = {0xbf068000u, 0xbf850002u, 0x7e0c02c1u, 0x7e0e02c1u};
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        expect_refused(stage, cat({relay(), branch, firstlane(), restore(stage)}));
}

TEST(VectorScalarProvenance, LoopBackedgeChecksTheVectorAssumptionUsedByLaneBits) {
    const Words fabricated = {0x7e0c0232u, 0x7e0e0233u};   // absent s50/s51
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        expect_refused(stage, loop(stage, fabricated), "fabricated on the back edge");
}

TEST(VectorScalarProvenance, DefinedVectorLoopCarryKeepsCompiling) {
    const Words defined = {0x7e0c02c1u, 0x7e0e02c1u};
    for (Stage stage : {Stage::Fragment, Stage::Compute})
        EXPECT_FALSE(compile(stage, loop(stage, defined)).empty()) << terminal_reason;
}

TEST(VectorScalarProvenance, MemoryPatternKeepsItsFragmentProjectionRestrictionAfterTheHop) {
    // Genuine buffer data: s_buffer_load_dwordx2 s[44:45],s[8:11],0. Compute may use its actual
    // guest-lane mask bits; fragment cannot infer pixel-to-lane authority from this memory pattern.
    const Words load = {0xf4240b04u, 0xfa000000u};
    const Words body = cat({prefix(), load, vector_copy(), firstlane(), restore()});
    expect_refused(Stage::Fragment, body);
    EXPECT_FALSE(compile(Stage::Compute,
                         cat({prefix(), load, vector_copy(), firstlane(), restore(Stage::Compute)}))
                     .empty())
        << terminal_reason;
}

TEST(VectorScalarProvenance, DispatcherRetainsMemoryMarksAcrossTheVectorReload) {
    const Words load = {0xf4240b04u, 0xfa000000u};
    const Words body = cat({prefix(), load, vector_copy(), dispatcher(), firstlane(), restore()});
    EXPECT_TRUE(compile(Stage::Fragment, body).empty());
    const std::string reason = terminal_reason;
    EXPECT_NE(reason.find("mode=unresolved-operand"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=" + std::to_string(body.size() - 1u) + " "), std::string::npos)
        << reason;
}

TEST(VectorScalarProvenance, ImplicitCmpxNarrowingCannotCleanDispatcherVectorMarks) {
    // After an explicit full EXEC, CMPX narrows it implicitly. The clean moves retain the old
    // fabricated words outside active lanes, including when the next block reloads the VGPRs.
    const Words overwrite = {0x7e0c02c1u, 0x7e0e02c1u};
    const Words body =
        cat({relay(), {0xbefe04c1u, 0x7da80090u}, overwrite, dispatcher(), firstlane(), restore()});
    EXPECT_TRUE(compile(Stage::Fragment, body).empty());
    const std::string reason = terminal_reason;
    EXPECT_NE(reason.find("pc=" + std::to_string(body.size() - 1u) + " "), std::string::npos)
        << reason;
    EXPECT_FALSE(
        compile(Stage::Fragment,
                cat({relay(), {0xbefe04c1u}, overwrite, dispatcher(), firstlane(), restore()}))
            .empty())
        << terminal_reason;
}

TEST(VectorScalarProvenance, Mad64ChargesTheHighWordOfItsVectorAddend) {
    for (uint32_t opcode : {0x176u, 0x177u}) {
        // v6 is genuine; only high addend v7 comes from an absent scalar. The low/high MAD result
        // is written to a fresh pair, so neither old destination can accidentally retain the mark.
        const Words mad = {0xd4006a0au | (opcode << 16u), 129u | (129u << 9u) | (262u << 18u)};
        const auto instruction = rdna2_decode_one(mad.data(), mad.size());
        ASSERT_EQ(instruction.opcode, opcode);
        EXPECT_EQ(rdna2_vgpr_source_span(instruction, 2), 2u);
        const Words read = {0x7e58050bu, 0xbead03c1u};   // s44=first(v11); s45=-1
        for (Stage stage : {Stage::Fragment, Stage::Compute}) {
            expect_refused(stage,
                           cat({prefix(), {0x7e0c0280u, 0x7e0e0233u}, mad, read, restore(stage)}),
                           "fabricated-fresh-mask-source");
            EXPECT_NE(terminal_reason.find("pc=" + std::to_string(prefix().size() + 2u) + " "),
                      std::string::npos)
                << terminal_reason;
            EXPECT_FALSE(
                compile(stage,
                        cat({prefix(), {0x7e0c0280u, 0x7e0e02c1u}, mad, read, restore(stage)}))
                    .empty())
                << terminal_reason;
            // Only the addend is a pair: missing s51/s53 cannot poison genuine B32 multipliers.
            const Words scalar_mad = {mad[0], 50u | (52u << 9u) | (262u << 18u)};
            EXPECT_FALSE(compile(stage, cat({prefix(),
                                             {0xbeb203c1u, 0xbeb40381u, 0x7e0c0280u, 0x7e0e02c1u},
                                             scalar_mad,
                                             read,
                                             restore(stage)}))
                             .empty())
                << terminal_reason;
            const Words scalar_addend = {mad[0], 129u | (129u << 9u) | (50u << 18u)};
            expect_refused(stage,
                           cat({prefix(), {0xbeb20380u}, scalar_addend, read, restore(stage)}),
                           "fabricated-fresh-mask-source");
            EXPECT_NE(terminal_reason.find("pc=" + std::to_string(prefix().size() + 1u) + " "),
                      std::string::npos)
                << terminal_reason;
            EXPECT_FALSE(compile(stage, cat({prefix(),
                                             {0xbeb20380u, 0xbeb303c1u},
                                             scalar_addend,
                                             read,
                                             restore(stage)}))
                             .empty())
                << terminal_reason;
        }
    }
}

TEST(VectorScalarProvenance, CarryDestinationCannotCertifyItsOwnVectorSourceAtAnEdge) {
    // V_ADD_CO_U32 reads OLD absent s50 before publishing its genuine carry mask into s[50:51].
    // Static transfer must snapshot that input before the new SDST fact is inserted.
    const Words add = {0xd70f3206u, 50u | (129u << 9u)};
    const Words body = cat({prefix(), add, dispatcher(), {0x7e580506u, 0xbead03c1u}, restore()});
    EXPECT_TRUE(compile(Stage::Fragment, body).empty());
    // Its carry output is itself a fresh predicate, so refuse the fabricated old source before
    // that output can certify it. The CFG terminal wrapper retains the original writer bytes.
    EXPECT_NE(terminal_reason.find("pc=" + std::to_string(prefix().size()) + " "),
              std::string::npos)
        << terminal_reason;
    EXPECT_NE(terminal_reason.find("words=d70f3206,00010232"), std::string::npos)
        << terminal_reason;
    EXPECT_FALSE(compile(Stage::Fragment, cat({prefix(),
                                               {0xbeb203c1u},
                                               add,
                                               dispatcher(),
                                               {0x7e580506u, 0xbead03c1u},
                                               restore()}))
                     .empty())
        << terminal_reason;
}
