// #5041: DATA dependencies of an SCC-preserving instruction cannot poison an earlier
// comparison. A genuine SCC writer replaces old provenance; a missing source stays missing.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
using namespace prosper::gpu;

void transfer(SpilledMaskHalves& state, uint32_t encoding, bool missing) {
    const auto in = rdna2_decode_one(&encoding, 1);
    ASSERT_NE(in.fmt, Rdna2Format::Unknown);
    advance_defined_slots(state, in, {}, {}, {{in.dst.value, 1}}, {}, {.scalar_relay = missing},
                          true, 64);
}

std::vector<uint32_t> relative_read_program(uint32_t compare, bool boundary) {
    std::vector<uint32_t> words{
        0xbe840385u, // s4 = 5
        0xbe850386u, // s5 = 6
        0xbe860387u, // s6 = 7
        0xbefc0381u, // M0 = 1
        compare,
        0xbe802e04u, // s0 = s[4+M0]; SCC unchanged, although later candidates are absent
    };
    if (boundary) words.push_back(0xbf880000u); // EXECZ +0 puts the next PC in another block
    words.insert(words.end(), {0xbf850001u, 0xbe800389u, 0x7e000200u, 0xbf810000u});
    return words;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code) {
    return recompile_valu(code.data(), code.size(), 1, 0, nullptr, 0, kDefaultComputePgmRsrc1,
                          true);
}
} // namespace

TEST(ScalarConditionPreservation, RelativeReadKeepsEarlierComparisonAcrossDispatcher) {
    for (const uint32_t compare : {0xbf018005u, 0xbf008005u})
        for (const bool boundary : {false, true}) {
            SCOPED_TRACE(compare);
            SCOPED_TRACE(boundary);
            EXPECT_FALSE(compile(relative_read_program(compare, boundary)).empty());
        }
}

TEST(ScalarConditionPreservation, MissingCompareSourceStillRefusesAcrossTheRelativeRead) {
    for (const bool boundary : {false, true}) {
        SCOPED_TRACE(boundary);
        EXPECT_TRUE(compile(relative_read_program(0xbf018014u, boundary)).empty())
            << "the same comparison now reads absent s20";
    }
}

TEST(ScalarConditionPreservation, PreservingInstructionsDoNotPoisonGenuineScc) {
    // Same DATA taint, different architectural preservers: relative MOV, CSELECT and BFM.
    for (const uint32_t encoding : {0xbe802e04u, 0x85010414u, 0x93010414u}) {
        SCOPED_TRACE(encoding);
        SpilledMaskHalves state;
        transfer(state, encoding, true);
        EXPECT_TRUE(state.relayed_words.contains(1) || state.relayed_words.contains(0))
            << "the destination's missing DATA dependency remains tracked";
        EXPECT_FALSE(state.relayed_words.contains(kRelayedScc));
    }
}

TEST(ScalarConditionPreservation, PreservingInstructionsCannotRepairPoisonedScc) {
    for (const uint32_t encoding : {0xbe802e04u, 0x85010414u, 0x93010414u}) {
        SCOPED_TRACE(encoding);
        SpilledMaskHalves state;
        state.relayed_words.insert(kRelayedScc);
        transfer(state, encoding, false);
        EXPECT_TRUE(state.relayed_words.contains(kRelayedScc));
    }
}

TEST(ScalarConditionPreservation, GenuineWritersReplaceOldSccProvenance) {
    // ADD and AND write a fresh scalar condition from their current genuine sources.
    for (const uint32_t encoding : {0x80018481u, 0x87018481u, 0xbe810704u}) {
        SCOPED_TRACE(encoding);
        SpilledMaskHalves state;
        state.relayed_words.insert(kRelayedScc);
        transfer(state, encoding, false);
        EXPECT_FALSE(state.relayed_words.contains(kRelayedScc));
    }
}

TEST(ScalarConditionPreservation, WritersWithMissingSourcesKeepSccPoisoned) {
    for (const uint32_t encoding : {0x80018481u, 0x87018481u, 0xbe810704u}) {
        SCOPED_TRACE(encoding);
        SpilledMaskHalves state;
        transfer(state, encoding, true);
        EXPECT_TRUE(state.relayed_words.contains(kRelayedScc));
    }
}

TEST(ScalarConditionPreservation, MissingCarryInCannotBecomeAFreshCondition) {
    for (const uint32_t encoding : {0x82018481u, 0x82818481u}) {
        SCOPED_TRACE(encoding); // ADDC and SUBB: explicit constants, implicit incoming SCC
        const auto in = rdna2_decode_one(&encoding, 1);
        for (const bool missing : {false, true}) {
            SCOPED_TRACE(missing);
            SpilledMaskHalves state;
            if (missing) state.relayed_words.insert(kRelayedScc);
            const auto source = spilled_vector_source_marks(state, in, {}, {}, !missing, 64);
            EXPECT_EQ(source.scalar_relay, missing);
            advance_defined_slots(state, in, {}, {}, {{in.dst.value, 1}}, {}, source, !missing, 64);
            EXPECT_EQ(state.relayed_words.contains(kRelayedScc), missing);
        }
    }
}
