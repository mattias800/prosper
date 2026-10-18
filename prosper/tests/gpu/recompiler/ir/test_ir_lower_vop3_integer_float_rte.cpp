// Guest RTE and a capable published device are independent necessary authorities.
#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstddef>
#include "gpu/recompiler/float32_rte_support.hpp"
#include "gpu/recompiler/ir/integer_float_rte_e64_cases.hpp"
#include "gpu/recompiler/ir/lower/vop3_integer_float_rte_lowering.hpp"
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include <gtest/gtest.h>
#include <cstdint>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::integer_float_rte;
namespace {
class IrLowerVop3IntegerFloatRte : public ::testing::Test {
    void SetUp() override { reset_float32_rte_support_for_test(); }
    void TearDown() override { reset_float32_rte_support_for_test(); }
};
bool has_mode(const SpirvCompute& b) {
    for (size_t i = 0; i < b.exec.size();) {
        const uint32_t word = b.exec[i], count = word >> 16;
        if (!count) return false;
        if ((word & 0xffffu) == Op_ExecutionMode && count == 4 && b.exec[i + 2] == 4462 &&
            b.exec[i + 3] == 32)
            return true;
        i += count;
    }
    return false;
}
}   // namespace
TEST_F(IrLowerVop3IntegerFloatRte, UnpublishedOrAnyWeakerDeviceCannotBeOverridden) {
    EXPECT_FALSE(float32_rte_admitted());
    for (uint32_t missing = 0; missing < 3; ++missing) {
        reset_float32_rte_support_for_test();
        publish_float32_rte_support(missing != 0, missing != 1, missing != 2);
        publish_float32_rte_support(true, true, true);
        EXPECT_FALSE(float32_rte_admitted());
        LoweringFixture t;
        t.b.is_compute = true;
        t.rs.vreg[2] = t.b.load_input(0);
        auto in = LoweringFixture::decode(cases::words({}));
        const auto before = t.b.code;
        EXPECT_FALSE(lower_vop3_integer_float_rte(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_EQ(t.b.code, before);
        EXPECT_FALSE(t.rs.vreg.contains(1));
        EXPECT_FALSE(has_mode(t.b));
    }
}
TEST_F(IrLowerVop3IntegerFloatRte, AllModeBytesRequireKnownRteAndDeclareItOnlyOnce) {
    publish_float32_rte_support(true, true, true);
    for (uint32_t opcode : {0x185u, 0x186u})
        for (uint32_t mode = 0; mode < 256; ++mode) {
            LoweringFixture t;
            t.b.is_compute = true;
            t.b.compute_pgm_rsrc1 = mode << 12;
            t.rs.vreg[2] = t.b.load_input(0);
            const auto in = LoweringFixture::decode(cases::words({opcode}));
            const auto before = t.b.code;
            EXPECT_EQ(lower_vop3_integer_float_rte(t.b, t.rs, in, t.ok), (mode & 3u) == 0);
            EXPECT_TRUE(t.ok);
            EXPECT_EQ(t.rs.vreg.contains(1), (mode & 3u) == 0);
            EXPECT_EQ(has_mode(t.b), (mode & 3u) == 0);
            if (mode & 3u)
                EXPECT_EQ(t.b.code, before);
            else {
                const auto exec = t.b.exec, caps = t.b.caps;
                EXPECT_TRUE(lower_vop3_integer_float_rte(t.b, t.rs, in, t.ok));
                EXPECT_EQ(t.b.exec, exec);
                EXPECT_EQ(t.b.caps, caps);
            }
        }
}
TEST_F(IrLowerVop3IntegerFloatRte, UnknownModeModifiersAndSpecialSourcesNeverPublish) {
    publish_float32_rte_support(true, true, true);
    for (const auto& c : cases::cases())
        if (!c.accepted) {
            LoweringFixture t;
            t.b.is_compute = true;
            t.b.compute_pgm_rsrc1 = c.mode;
            t.rs.vreg[2] = t.b.load_input(0);
            const auto before = t.b.code;
            const auto in = LoweringFixture::decode(cases::words(c));
            const bool routed = lower_vop3_integer_float_rte(t.b, t.rs, in, t.ok);
            EXPECT_EQ(routed, c.modifier != 0);
            EXPECT_EQ(t.ok, c.modifier == 0);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_EQ(t.b.code, before);
            EXPECT_FALSE(has_mode(t.b));
        }
    // Entry/source definedness belongs to the whole-program caller, tested in execution.
    // This lowerer alone excludes special/constant encodings, not raw VGPR storage slots.
    for (uint32_t selector : {106u, 107u, 124u, 125u, 126u, 127u, 128u, 242u, 255u}) {
        LoweringFixture t;
        t.b.is_compute = true;
        auto c = cases::Case{};
        c.source = selector;
        auto raw = cases::words(c);
        if (selector == 255) raw.push_back(0);
        const auto in = LoweringFixture::decode(raw);
        lower_vop3_integer_float_rte(t.b, t.rs, in, t.ok);
        EXPECT_FALSE(t.rs.vreg.contains(1));
        EXPECT_FALSE(has_mode(t.b));
    }
}

TEST_F(IrLowerVop3IntegerFloatRte, GraphicsAndNggUseTheirProducingModeInsteadOfComputeMode) {
    publish_float32_rte_support(true, true, true);
    for (uint32_t stage : {0u, 1u, 2u})
        for (uint32_t opcode : {0x185u, 0x186u})
            for (uint32_t arm : {0u, 1u, 2u}) {
                LoweringFixture t;
                t.b.is_fragment = stage == 0;
                t.b.is_vertex = stage == 1;
                t.b.is_compute = stage == 2;
                t.b.ngg_workgroup_shell = stage == 2;
                t.b.compute_pgm_rsrc1 = arm == 0 ? 0x33000u : 0x30000u;   // Deliberately opposite.
                const uint8_t mode = arm == 1 ? 0x33 : 0x30;
                t.b.fragment_float_mode = {stage == 0 && arm != 2, mode};
                t.b.vertex_float_mode = {stage != 0 && arm != 2, mode, false};
                t.rs.vreg[2] = t.b.load_input(0);
                const auto in = LoweringFixture::decode(cases::words({opcode}));
                EXPECT_EQ(lower_vop3_integer_float_rte(t.b, t.rs, in, t.ok), arm == 0);
                EXPECT_TRUE(t.ok);
                EXPECT_EQ(t.rs.vreg.contains(1), arm == 0);
                EXPECT_EQ(has_mode(t.b), arm == 0);
            }
    for (uint32_t selector : {0u, 50u, 102u, 105u}) {
        LoweringFixture t;
        t.b.is_compute = true;
        t.rs.sreg[static_cast<int>(selector)] = t.b.load_input(0);
        auto c = cases::Case{};
        c.source = selector;
        const auto in = LoweringFixture::decode(cases::words(c));
        EXPECT_TRUE(lower_vop3_integer_float_rte(t.b, t.rs, in, t.ok));
        EXPECT_TRUE(t.ok);
        EXPECT_TRUE(t.rs.vreg.contains(1));
    }
}

TEST_F(IrLowerVop3IntegerFloatRte, PresentStorageDoesNotBypassTheExistingMaskAndSpillResolver) {
    publish_float32_rte_support(true, true, true);
    for (uint32_t opcode : {0x185u, 0x186u})
        for (bool scalar : {false, true}) {
            LoweringFixture t;
            t.b.is_compute = true;
            cases::Case c;
            c.opcode = opcode;
            c.source = scalar ? 50u : 258u;
            const auto value = t.b.load_input(0);
            if (scalar) {
                t.rs.sreg[50] = value;
                t.rs.unavailable_saved_mask_words.insert(50);
            } else {
                t.rs.vreg[2] = value;
                t.rs.vgpr_lane_slots[2][0] = value;
            }
            const auto in = LoweringFixture::decode(cases::words(c));
            EXPECT_TRUE(lower_vop3_integer_float_rte(t.b, t.rs, in, t.ok));
            EXPECT_FALSE(t.ok);
            EXPECT_FALSE(t.rs.vreg.contains(1));
            EXPECT_FALSE(has_mode(t.b));
        }
}

TEST_F(IrLowerVop3IntegerFloatRte, RecordedCapabilityAndRequestSurviveOppositeAmbientPublication) {
    for (bool capable : {false, true}) {
        reset_float32_rte_support_for_test();
        publish_float32_rte_support(capable, true, true);
        const auto in = LoweringFixture::decode(cases::words({0x185}));
        LoweringFixture first;
        first.b.compute_pgm_rsrc1 = 0x30000;
        first.rs.vreg[2] = first.b.load_input(0);
        CompilerChoiceTrace recorded;
        {
            CompilerChoiceScope recording(recorded);
            EXPECT_EQ(lower_vop3_integer_float_rte(first.b, first.rs, in, first.ok), capable);
        }
        ASSERT_TRUE(recorded.error.empty());
        ASSERT_EQ(recorded.reads.size(), 1u);
        EXPECT_EQ(recorded.reads[0], (CompilerChoiceRead{CompilerChoice::Float32Rte, capable}));
        reset_float32_rte_support_for_test();
        publish_float32_rte_support(!capable, true, true);
        LoweringFixture replayed;
        replayed.b.compute_pgm_rsrc1 = 0x30000;
        replayed.rs.vreg[2] = replayed.b.load_input(0);
        {
            CompilerChoiceScope replay(static_cast<const CompilerChoiceTrace&>(recorded));
            EXPECT_EQ(lower_vop3_integer_float_rte(replayed.b, replayed.rs, in, replayed.ok),
                      capable);
            EXPECT_NO_THROW(replay.finish_replay());
        }
        EXPECT_EQ(replayed.b.code, first.b.code);
        EXPECT_EQ(replayed.b.exec, first.b.exec);
        EXPECT_EQ(has_mode(replayed.b), capable);
        CompilerChoiceSet request;
        request.set(CompilerChoice::Float32Rte, capable);
        LoweringFixture supplied;
        supplied.b.compute_pgm_rsrc1 = 0x30000;
        supplied.rs.vreg[2] = supplied.b.load_input(0);
        CompilerChoiceScope use_request(request);
        EXPECT_EQ(lower_vop3_integer_float_rte(supplied.b, supplied.rs, in, supplied.ok), capable);
        EXPECT_EQ(supplied.b.code, first.b.code);
        EXPECT_EQ(has_mode(supplied.b), capable);
    }
}
