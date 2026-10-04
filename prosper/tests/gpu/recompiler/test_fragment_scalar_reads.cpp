// #4334: an empty normalized table must not hide original scalar-buffer demands, and a folded
// native32 value must not masquerade as a genuine PS user-prefix descriptor origin. CPU only.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/fragment_packet_scalar_reads.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>

namespace {
namespace g = prosper::gpu;
constexpr std::array<uint32_t, 4> Descriptor{0x12345000u, 0x12u, 64u, 0u};
// GFX10 SMEM: opcode8/9/10, SDST20/22/24, SBASE0, null SOFFSET125, immediate16/24/32.
const std::vector<uint32_t> ThreeReads{0xf4200500u, 0xfa000010u, 0xf4240580u, 0xfa000018u,
                                       0xf4280600u, 0xfa000020u, 0xbf810000u};
g::FragmentPacketScalarReadRequirements facts(const std::vector<uint32_t>& code) {
    std::vector<g::Rdna2Inst> decoded;
    g::rdna2_walk(code.data(), code.size(), decoded);
    return g::fragment_packet_scalar_read_requirements(code, decoded);
}
std::shared_ptr<g::RasterQuadInputs> inputs(const std::vector<uint32_t>& code, uint32_t count = 4) {
    auto result = std::make_shared<g::RasterQuadInputs>();
    const auto analysis = g::acquire_shader_analysis(code.data(), code.size());
    result->raw_code = g::shader_analysis_owned_words(analysis);
    result->vgpr_requirements = g::shader_analysis_packet_vgpr_requirements(analysis);
    result->source_fs =
        std::make_shared<const std::vector<uint32_t>>(std::initializer_list<uint32_t>{
            0x07230203u}); // source-association token, not a GPU module
    result->raw_matches_producing_source = true;
    result->entry.observed = true;
    result->entry.rsrc2_available = true;
    result->entry.rsrc2 = ((count & 31u) << 1) | ((count >> 5) << 27);
    result->entry.user_data_available = UINT32_MAX;
    std::copy(Descriptor.begin(), Descriptor.end(), result->entry.user_data.begin());
    result->ps_resources = g::own_fragment_packet_resources(nullptr);
    return result;
}
bool gap(const g::FragmentPacketPreparation& prepared, const char* reason) {
    return std::find(prepared.unmet.begin(), prepared.unmet.end(), reason) != prepared.unmet.end();
}
TEST(FragmentScalarReads, OriginalPcWidthsAndImmediateDemands) {
    const auto plan = facts(ThreeReads);
    ASSERT_TRUE(plan.has_smem);
    ASSERT_TRUE(plan.rejection.empty()) << plan.rejection;
    ASSERT_EQ(plan.source_words, &ThreeReads);
    ASSERT_EQ(plan.sites.size(), 3u);
    const uint32_t pcs[]{0, 2, 4}, opcodes[]{8, 9, 10}, words[]{1, 2, 4}, offsets[]{16, 24, 32};
    for (uint32_t index = 0; index < 3; ++index) {
        EXPECT_EQ(plan.sites[index].pc, pcs[index]);
        EXPECT_EQ(plan.sites[index].opcode, opcodes[index]);
        EXPECT_EQ(plan.sites[index].words, words[index]);
        EXPECT_EQ(plan.sites[index].byte_offset, offsets[index]);
        EXPECT_EQ(plan.sites[index].entry_words, (std::array<uint32_t, 4>{0, 1, 2, 3}));
    }
}
TEST(FragmentScalarReads, CanonicalPrefetchPreservesOriginsButReservedHintsAndClausesRefuse) {
    for (uint32_t mode : {1u, 2u, 3u}) {
        SCOPED_TRACE(mode);
        auto code = ThreeReads;
        code.insert(code.begin(), 0xbfa00000u | mode);
        const auto plan = facts(code);
        ASSERT_TRUE(plan.has_smem);
        ASSERT_TRUE(plan.rejection.empty()) << plan.rejection;
        ASSERT_EQ(plan.sites.size(), 3u);
        for (uint32_t index = 0; index < plan.sites.size(); ++index) {
            EXPECT_EQ(plan.sites[index].pc, 1u + 2u * index);
            EXPECT_EQ(plan.sites[index].entry_words, (std::array<uint32_t, 4>{0, 1, 2, 3}));
        }
    }
    for (uint32_t hint : {0xbfa00000u, 0xbfa00004u, 0xbfa00103u, 0xbfa08003u, 0xbfa10001u}) {
        SCOPED_TRACE(hint);
        auto code = ThreeReads;
        code.insert(code.begin(), hint);
        const auto refused = facts(code);
        EXPECT_TRUE(refused.has_smem);
        EXPECT_EQ(refused.rejection, "packet-scalar-instruction-effects-unimplemented:pc=0");
        EXPECT_TRUE(refused.sites.empty());
    }
}

TEST(FragmentScalarReads, GenuineMovOriginsAndReadBeforeOverwrite) {
    // s_mov_b64 s[8:9],s[0:1]; s_mov_b32 s10,s2; s_mov_b32 s11,s3; two loads via s[8:11].
    const std::vector<uint32_t> moved{0xbe880400u, 0xbe8a0302u, 0xbe8b0303u, 0xf4200504u,
                                      0xfa000000u, 0xf4240584u, 0xfa000010u, 0xbf810000u};
    const auto plan = facts(moved);
    ASSERT_TRUE(plan.rejection.empty()) << plan.rejection;
    ASSERT_EQ(plan.sites.size(), 2u);
    EXPECT_EQ(plan.sites[0].pc, 3u);
    EXPECT_EQ(plan.sites[1].pc, 5u);
    for (const auto& site : plan.sites)
        EXPECT_EQ(site.entry_words, (std::array<uint32_t, 4>{0, 1, 2, 3}));
    auto prepared = g::prepare_fragment_packet_inputs(inputs(moved), true);
    ASSERT_EQ(prepared->scalar_descriptors.size(), 2u);
    for (const auto& observation : prepared->scalar_descriptors)
        EXPECT_EQ(observation.descriptor, Descriptor);
    // First load legitimately consumes descriptor s[0:3] before it overwrites s0; a second load
    // cannot reuse that original entry origin. No partially certified first site survives refusal.
    const std::vector<uint32_t> overwritten{0xf4200000u, 0xfa000000u, 0xf4200500u, 0xfa000004u,
                                            0xbf810000u};
    const auto rejected = facts(overwritten);
    EXPECT_TRUE(rejected.has_smem);
    EXPECT_TRUE(rejected.sites.empty());
    EXPECT_EQ(rejected.rejection, "packet-scalar-descriptor-origin-unproved:pc=2");
}
TEST(FragmentScalarReads, ScalarAluConditionalAndWaveDefinitionsCannotLaunderOrigin) {
    // s_add_u32 s0,s0,0; s_cmov_b32 s0,s0; v_readfirstlane_b32 s0,v0;
    // inline MOV s0,0; genuine unchanged MOV s0,s0 is the paired positive.
    for (const uint32_t writer : {0x80008000u, 0xbe800500u, 0x7e000500u, 0xbe800380u}) {
        SCOPED_TRACE(writer);
        const std::vector<uint32_t> code{writer, 0xf4200500u, 0xfa000000u, 0xbf810000u};
        const auto rejected = facts(code);
        EXPECT_TRUE(rejected.has_smem);
        EXPECT_EQ(rejected.rejection, "packet-scalar-descriptor-origin-unproved:pc=1");
        EXPECT_TRUE(rejected.sites.empty());
    }
    const std::vector<uint32_t> unchanged{0xbe800300u, 0xf4200500u, 0xfa000000u, 0xbf810000u};
    const auto positive = facts(unchanged);
    EXPECT_TRUE(positive.rejection.empty()) << positive.rejection;
    ASSERT_EQ(positive.sites.size(), 1u);
    EXPECT_EQ(positive.sites[0].entry_words, (std::array<uint32_t, 4>{0, 1, 2, 3}));
    const std::vector<uint32_t> literal{0xbe8003ffu, 0u, 0xf4200500u, 0xfa000000u, 0xbf810000u};
    EXPECT_EQ(facts(literal).rejection, "packet-scalar-descriptor-origin-unproved:pc=2");
}
TEST(FragmentScalarReads, UnsupportedControlFormsAndTruncationStayNamed) {
    const std::vector<uint32_t> branch{0xbf820000u, 0xf4200500u, 0xfa000000u, 0xbf810000u};
    EXPECT_EQ(facts(branch).rejection, "packet-scalar-control-flow-unimplemented:pc=0");
    auto controls = ThreeReads;
    controls[0] |= 1u << 16;   // nonplain GLC control, not silently ignored
    EXPECT_EQ(facts(controls).rejection, "packet-smem-form-unimplemented:pc=0");
    auto truncated = ThreeReads;
    truncated.pop_back();
    EXPECT_EQ(facts(truncated).rejection, "packet-scalar-program-inventory-incomplete:pc=6");
    const std::vector<uint32_t> no_memory{0xbe800300u, 0xbf810000u};
    const auto empty = facts(no_memory);
    EXPECT_FALSE(empty.has_smem);
    EXPECT_TRUE(empty.sites.empty());
    EXPECT_TRUE(empty.rejection.empty());
}
TEST(FragmentScalarReads, ActualPreparationNamesDemandsWithoutNormalizedResources) {
    const auto in = inputs(ThreeReads);
    ASSERT_TRUE(in->ps_resources.table->resources.empty());
    const auto prepared = g::prepare_fragment_packet_inputs(in, true);
    ASSERT_TRUE(prepared->scalar_read_requirements);
    EXPECT_EQ(prepared->scalar_read_requirements->source_words, in->raw_code.get());
    ASSERT_EQ(prepared->scalar_descriptors.size(), 3u);
    for (uint32_t site = 0; site < 3; ++site) {
        EXPECT_EQ(prepared->scalar_descriptors[site].site_index, site);
        EXPECT_EQ(prepared->scalar_descriptors[site].descriptor, Descriptor);
    }
    EXPECT_TRUE(gap(*prepared, "packet-resource-guest-fetch-and-read-point-unproved"));
    EXPECT_TRUE(gap(*prepared, "packet-scalar-live-read-lease-unimplemented"));
    EXPECT_FALSE(prepared->ready);
}
TEST(FragmentScalarReads, MissingWordAndSystemPrefixNeverBorrowPhysicalUserData) {
    auto in = inputs(ThreeReads);
    const auto present_zero = g::prepare_fragment_packet_inputs(in, true);
    ASSERT_EQ(present_zero->scalar_descriptors.size(), 3u);
    EXPECT_EQ(present_zero->scalar_descriptors[0].descriptor[3], 0u);
    in->entry.user_data_available &= ~(1u << 3);
    const auto absent = g::prepare_fragment_packet_inputs(in, true);
    EXPECT_TRUE(absent->scalar_descriptors.empty());
    EXPECT_TRUE(gap(*absent, "packet-scalar-descriptor-user-word-unavailable:pc=0"));
    const auto system = g::prepare_fragment_packet_inputs(inputs(ThreeReads, 3), true);
    EXPECT_TRUE(system->scalar_descriptors.empty());
    EXPECT_TRUE(gap(*system, "packet-scalar-descriptor-user-word-unavailable:pc=0"));
    const auto no_count = [&] {
        auto missing = inputs(ThreeReads);
        missing->entry.rsrc2_available = false;
        return g::prepare_fragment_packet_inputs(missing, true);
    }();
    EXPECT_TRUE(no_count->scalar_descriptors.empty());
    EXPECT_TRUE(gap(*no_count, "packet-scalar-descriptor-user-prefix-unavailable"));
}
TEST(FragmentScalarReads, ExactCachedOwnerAndDynamicDescriptorsRemainSeparate) {
    auto code = ThreeReads;
    const auto a = inputs(code), b = inputs(code);
    ASSERT_EQ(a->vgpr_requirements, b->vgpr_requirements);
    ASSERT_EQ(a->raw_code, b->raw_code);
    const auto first = g::prepare_fragment_packet_inputs(a, true);
    b->entry.user_data[0] = 0x76543000u;
    const auto second = g::prepare_fragment_packet_inputs(b, true);
    ASSERT_EQ(first->scalar_descriptors.size(), 3u);
    ASSERT_EQ(second->scalar_descriptors.size(), 3u);
    EXPECT_EQ(first->scalar_read_requirements, second->scalar_read_requirements);
    EXPECT_EQ(first->scalar_descriptors[0].descriptor, Descriptor);
    EXPECT_EQ(second->scalar_descriptors[0].descriptor[0], 0x76543000u);
    // Rewriting the same source VA creates another exact-byte owner. Earlier requirements cannot
    // authorize a later original version, even if its operation/shape happens to remain supported.
    code[1] = 0xfa000014u;
    const auto rewritten = inputs(code);
    ASSERT_NE(a->raw_code, rewritten->raw_code);
    auto borrowed = std::make_shared<g::RasterQuadInputs>(*rewritten);
    borrowed->vgpr_requirements = a->vgpr_requirements;
    const auto refused = g::prepare_fragment_packet_inputs(borrowed, true);
    EXPECT_FALSE(refused->scalar_read_requirements);
    EXPECT_TRUE(refused->scalar_descriptors.empty());
    EXPECT_TRUE(gap(*refused, "packet-scalar-program-requirements-unavailable"));
    const auto current = g::prepare_fragment_packet_inputs(rewritten, true);
    ASSERT_EQ(current->scalar_descriptors.size(), 3u);
    EXPECT_EQ(current->scalar_read_requirements->sites[0].byte_offset, 20u);
    EXPECT_EQ(first->scalar_read_requirements->sites[0].byte_offset, 16u);
}
TEST(FragmentScalarReads, ForeignRawOwnerAndModuleMismatchGrantNoObservations) {
    auto in = inputs(ThreeReads);
    in->raw_code = std::make_shared<const std::vector<uint32_t>>(ThreeReads);
    const auto foreign = g::prepare_fragment_packet_inputs(in, true);
    EXPECT_FALSE(foreign->scalar_read_requirements);
    EXPECT_TRUE(foreign->scalar_descriptors.empty());
    EXPECT_TRUE(gap(*foreign, "packet-scalar-program-requirements-unavailable"));
    const auto mismatch = g::prepare_fragment_packet_inputs(inputs(ThreeReads), false);
    EXPECT_FALSE(mismatch->scalar_read_requirements);
    EXPECT_TRUE(mismatch->scalar_descriptors.empty());
    EXPECT_TRUE(gap(*mismatch, "packet-producing-source-unavailable"));
}
}   // namespace
