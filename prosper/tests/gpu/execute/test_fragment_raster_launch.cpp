// Actual registered producer and immutable draw-source association. No Vulkan queue is used;
// observing these words does not authorize guest workitem-valid, helper or scheduling inputs.
#include "fixtures/fragment_draw_fixture.hpp"
#include "fixtures/fragment_raster_fixture.hpp"
#include "fixtures/fragment_draw_source.hpp"
#include "gpu/execute/fragment_raster_contract.hpp"
#include "gpu/execute/fragment_raster_launch.hpp"
#include "gpu/execute/fragment_raster_program.hpp"
#include "gpu/execute/fragment_raster_launch_collection.hpp"
#include "gpu/recompiler/fragment_raster_interface.hpp"
#include "gpu/recompiler/fragment_packet_mask_requirements.hpp"
#include "gpu/execute/fragment_draw_plan.hpp"
#include "bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <map>

namespace {
namespace g = prosper::gpu;
namespace f = prosper::test::fragment_draw;
g::FragmentPacketDeviceContract raster_device() {
    // Explicit offline profile for SOURCE/interface controls, not a queried physical device.
    return {1, true, false,
            g::FragmentPacketRasterDeviceContract{true, 128, 128, 128, 1024, 3, 1, 128}};
}
class FragmentRasterLaunch : public ::testing::Test {
protected:
    void SetUp() override {
        g::reset_float_transport_config_for_test();
        g::publish_float_transport_config({g::FloatTransportProfile::ExplicitNonFinite32});
        g::publish_float_controls_support(true, true);
    }
    void TearDown() override { g::reset_float_transport_config_for_test(); }
};
} // namespace

namespace {
std::vector<uint32_t> helper_original(uint32_t position = 0) {
    return {0xbe94047eu, 0xbefe0a7eu, 0xd8d480ffu, (position << 24) | position,
            0xbf8cc07fu, 0xbefe0414u, 0xf800180fu, position * 0x01010101u,
            0xbf810000u};
}
std::vector<uint32_t> position_free_original() {
    // Genuine WQM literal writer defines the selected peer before aliasing DS; no entry VGPR.
    return {0xbe94047eu, 0xbefe0a7eu, 0x7e0002ffu, 0x3f000000u, 0xd8d480ffu, 0u,
            0xbf8cc07fu, 0xbefe0414u, 0xf800180fu, 0u,          0xbf810000u};
}
std::vector<std::pair<uint32_t, uint32_t>> position_free_context() {
    auto context = prosper::test::fragment_raster::context();
    for (auto& [reg, value] : context)
        if (reg == prosper::agc::Pm4::SPI_PS_INPUT_ENA ||
            reg == prosper::agc::Pm4::SPI_PS_INPUT_ADDR)
            value = 0;
    return context;
}
g::FragmentRasterProgram helper_requirements(const std::vector<uint32_t>& code,
                                             g::PixelSystemInputMapping mapping) {
    std::vector<g::Rdna2Inst> instructions;
    g::rdna2_walk(code.data(), code.size(), instructions);
    return g::fragment_raster_program(instructions, uint32_t(code.size()), mapping, 0);
}
} // namespace

TEST_F(FragmentRasterLaunch, OriginalSavedLiveHelperProgramDemandsRealPackedPositionWords) {
    const auto code = helper_original();
    std::vector<g::Rdna2Inst> decoded;
    g::rdna2_walk(code.data(), code.size(), decoded);
    ASSERT_GE(decoded.size(), 2u);
    for (size_t index : {size_t(0), size_t(1)}) {
        EXPECT_EQ(decoded[index].src[0].kind, g::OperandKind::Special);
        EXPECT_EQ(decoded[index].src[0].value, 126);
    }
    const auto direct = helper_requirements(helper_original(), {1u << 8, 1u << 8});
    ASSERT_TRUE(direct.rejection.empty()) << direct.rejection;
    ASSERT_EQ(direct.positions.size(), 1u);
    EXPECT_EQ(direct.positions[0], (g::FragmentRasterPositionInput{0, 5}));
    // ADDR reserves a disabled perspective-sample pair before the genuinely enabled POS_X.
    const auto packed = helper_requirements(helper_original(2), {1u << 8, (1u << 0) | (1u << 8)});
    ASSERT_TRUE(packed.rejection.empty()) << packed.rejection;
    ASSERT_EQ(packed.positions.size(), 1u);
    EXPECT_EQ(packed.positions[0], (g::FragmentRasterPositionInput{2, 5}));
    const auto disabled = helper_requirements(helper_original(2), {0, (1u << 0) | (1u << 8)});
    EXPECT_EQ(disabled.rejection, "fragment-raster-genuine-quad-helper-source-unproved:pc=2");
    EXPECT_TRUE(disabled.positions.empty());
}

TEST_F(FragmentRasterLaunch, CanonicalPrefetchRetainsTheRegisteredOriginalThroughFinalCompilation) {
    namespace helper = prosper::test::fragment_raster;
    for (uint32_t mode : {1u, 2u, 3u}) {
        SCOPED_TRACE(mode);
        auto code = helper::original();
        code.insert(code.begin(), 0xbfa00000u | mode);
        ASSERT_TRUE(helper_requirements(code, {1u << 8, 1u << 8}).rejection.empty());
        std::vector<g::Rdna2Inst> decoded;
        g::rdna2_walk(code.data(), code.size(), decoded);
        const auto masks = g::fragment_packet_mask_requirements(code, decoded);
        ASSERT_TRUE(masks.rejection.empty()) << masks.rejection;
        EXPECT_EQ(masks.demanded, g::kPacketInitialExec);
        g::DrawItem draw;
        const auto context = helper::context();
        ASSERT_TRUE(f::realize(draw, f::color_a, code, f::ieee_rsrc1, 15, context));
        ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
        const auto& original = *draw.fragment_draw_inputs;
        EXPECT_EQ(*original.raw_code, code);
        EXPECT_TRUE(original.launch_source->matches(original));
        const auto prepared = f::prepare(draw);
        ASSERT_TRUE(prepared && prepared->launch_source);
        const auto plan = g::cached_fragment_draw_program(original, *prepared, raster_device(), 48);
        ASSERT_TRUE(plan);
        ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
        ASSERT_TRUE(plan->capacity_owner() && plan->capacity_owner()->kernel());
        const auto& kernel = *plan->capacity_owner()->kernel();
        EXPECT_EQ(kernel.guest_code, code);
        EXPECT_EQ(kernel.program.packet.initial_mask_availability, g::kPacketInitialExec);
        EXPECT_EQ(kernel.program.packet.export_observation,
                  g::FragmentPacketExportObservation::Architectural);
        EXPECT_FALSE(kernel.program.packet.spirv.empty());
        EXPECT_EQ(plan, g::cached_fragment_draw_program(original, *prepared, raster_device(), 48));
    }
}

TEST_F(FragmentRasterLaunch, ReservedPrefetchAndUnprovedClauseDoNotMintPendingOriginalPermission) {
    for (uint32_t mode : {0u, 4u, 0x103u, 0x8003u}) {
        SCOPED_TRACE(mode);
        auto code = helper_original();
        code.insert(code.begin(), 0xbfa00000u | mode);
        const auto refused = helper_requirements(code, {1u << 8, 1u << 8});
        EXPECT_EQ(refused.rejection, "fragment-raster-prefetch-mode-unimplemented:pc=0");
        EXPECT_TRUE(refused.positions.empty());
    }
    auto clause = helper_original();
    clause.insert(clause.begin(), 0xbfa10001u);
    const auto refused = helper_requirements(clause, {1u << 8, 1u << 8});
    EXPECT_EQ(refused.rejection, "fragment-raster-clause-unimplemented:pc=0");
    EXPECT_TRUE(refused.positions.empty());
}

TEST_F(FragmentRasterLaunch, PositionFreeOriginalRetainsItsRegisteredRasterEntrySchema) {
    const auto code = position_free_original();
    const auto proof = helper_requirements(code, {0, 0});
    ASSERT_TRUE(proof.rejection.empty()) << proof.rejection;
    ASSERT_TRUE(proof.positions.empty());
    g::DrawItem draw;
    const auto context = position_free_context();
    ASSERT_TRUE(f::realize(draw, f::color_a, code, f::ieee_rsrc1, 15, context));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    const auto& original = *draw.fragment_draw_inputs;
    EXPECT_EQ(*original.raw_code, code);
    ASSERT_TRUE(original.launch_source->matches(original));
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared && prepared->launch_source == original.launch_source);
    const auto plan = g::cached_fragment_draw_program(original, *prepared, raster_device(), 16);
    ASSERT_TRUE(plan);
    ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
    ASSERT_TRUE(plan->capacity_owner());
    const auto& capacity = *plan->capacity_owner();
    EXPECT_EQ(plan->entry_schema(), g::FragmentDrawEntryRecipe::DrawBoundRasterSystemAndQuadMasks);
    EXPECT_EQ(capacity.entry_recipe(), plan->entry_schema());
    EXPECT_TRUE(capacity.raster_inputs().empty());
    EXPECT_EQ(capacity.kernel()->guest_code, code);
    EXPECT_EQ(capacity.kernel()->program.packet.initial_mask_availability, g::kPacketInitialExec);
    EXPECT_FALSE(plan->assembly_words().empty());
    EXPECT_FALSE(plan->validation_words().empty());
    EXPECT_FALSE(plan->replay_words().empty());
    EXPECT_EQ(plan, g::cached_fragment_draw_program(original, *prepared, raster_device(), 16));
    const auto transaction = g::instantiate_fragment_draw_transaction(
        plan, draw.fragment_draw_inputs, *prepared, 18, 2, 1, 1);
    ASSERT_TRUE(transaction.rejection().empty()) << transaction.rejection();
    for (uint8_t availability :
         {uint8_t(0), uint8_t(g::kPacketInitialExec | g::kPacketInitialVcc)}) {
        auto kernel = std::make_shared<g::FragmentPacketKernel>(*capacity.kernel());
        kernel->program.packet.initial_mask_availability = availability;
        std::string rejection;
        EXPECT_FALSE(g::fragment_draw_capacity(kernel, capacity.collector(), rejection,
                                               capacity.entry_recipe()));
        EXPECT_EQ(rejection, "fragment-draw-raster-entry-schema-invalid");
    }
    std::string rejection;
    EXPECT_FALSE(g::fragment_draw_capacity(capacity.kernel(), capacity.collector(), rejection,
                                           static_cast<g::FragmentDrawEntryRecipe>(255)));
    EXPECT_EQ(rejection, "fragment-draw-entry-recipe-invalid");
}
TEST_F(FragmentRasterLaunch,
       PositionFreeLiveMaskSurvivesAssemblyOriginalExecutionAndWholeDrawValidation) {
    g::DrawItem draw;
    const auto code = position_free_original();
    const auto context = position_free_context();
    ASSERT_TRUE(f::realize(draw, f::color_a, code, f::ieee_rsrc1, 15, context));
    ASSERT_TRUE(draw.fragment_draw_inputs);
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared);
    const auto plan =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, raster_device(), 16);
    ASSERT_TRUE(plan);
    ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
    ASSERT_TRUE(plan->capacity_owner());
    const auto& capacity = *plan->capacity_owner();
    const auto& kernel = *capacity.kernel();
    ASSERT_TRUE(capacity.raster_inputs().empty());
    ASSERT_EQ(kernel.layout.vgprs.size(), 1u); // writer-only v0, not an incoming position
    const auto transaction = g::instantiate_fragment_draw_transaction(
        plan, draw.fragment_draw_inputs, *prepared, 18, 2, 1, 1);
    ASSERT_TRUE(transaction.rejection().empty()) << transaction.rejection();
    const auto& shape = plan->collector_shape();
    std::vector<uint32_t> source(capacity.collector_words(), 0);
    const uint32_t header[]{9, 0, shape.record_words, g::kRasterQuadMagic};
    std::copy(std::begin(header), std::end(header), source.begin());
    // Hand-owned offline observation, NOT a device witness: upper32, helpers, and padding.
    for (uint32_t quad = 0; quad < 9; ++quad)
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const bool live = !(lane & 1u);
            const uint32_t words[]{uint32_t(!live),
                                   uint32_t(live),
                                   uint32_t(live),
                                   1,
                                   0,
                                   std::bit_cast<uint32_t>(float(2 * quad + (lane & 1u)) + .5f),
                                   std::bit_cast<uint32_t>(float(lane >> 1u) + .5f),
                                   0,
                                   std::bit_cast<uint32_t>(1.f)};
            std::copy(std::begin(words), std::end(words),
                      source.begin() + 4 + (4 * quad + lane) * shape.lane_words);
        }
    const std::vector<uint32_t> authority(capacity.authority().begin(), capacity.authority().end());
    const std::vector<uint32_t> zero_commit(capacity.commit_words(), 0);
    bpermute_oracle::Interpreter count(plan->count_words());
    count.extra_writable_bindings = {0, 4};
    auto input = count.run_buffers(1,
                                   {{0, source},
                                    {1, std::vector<uint32_t>(capacity.input_words(), 0)},
                                    {2, authority},
                                    {3, transaction.entry_words()},
                                    {4, zero_commit}},
                                   1);
    ASSERT_TRUE(count.error.empty()) << count.error;
    source = count.writable_result(0);
    ASSERT_EQ(input.size(), capacity.input_words());
    ASSERT_EQ(input[1], 1u);
    ASSERT_EQ(input[4], 9u);
    ASSERT_EQ(input[8], 0u);
    bpermute_oracle::Interpreter assemble(plan->assembly_words());
    input = assemble.run_buffers(
        64, {{0, source}, {1, input}, {2, authority}, {3, transaction.entry_words()}}, 1);
    ASSERT_TRUE(assemble.error.empty()) << assemble.error;
    for (uint32_t worker = 0; worker < 64; ++worker) {
        const bool live = worker < 36 && !(worker & 1u);
        const auto row =
            capacity.placement(0).input_base + 2 + worker * kernel.program.packet.input_stride;
        ASSERT_LT(row + 5, input.size());
        EXPECT_EQ(input[row + 1], 0u) << "writer-only storage is initially unavailable";
        EXPECT_EQ(input[row + 2], uint32_t(live)) << "live mask without any position rows";
        EXPECT_EQ(input[row + 3], 0u);
        EXPECT_EQ(input[row + 4], 0u);
        EXPECT_EQ(input[row + 5], uint32_t(live));
    }
    bpermute_oracle::Interpreter original(kernel.program.packet.spirv);
    const auto output = original.run_buffers(
        64, {{0, input}, {1, std::vector<uint32_t>(capacity.output_words(), 0)}, {2, authority}},
        1);
    ASSERT_TRUE(original.error.empty()) << original.error;
    ASSERT_EQ(output.size(), capacity.output_words());
    for (uint32_t worker = 0; worker < 64; ++worker) {
        const bool live = worker < 36 && !(worker & 1u);
        const auto record =
            g::kPacketWaveOutputPrefix + worker * g::kFragmentPacketArchitecturalExportWords;
        EXPECT_EQ(output[record + 1], uint32_t(live));
        EXPECT_EQ(output[record + 2], uint32_t(live));
        EXPECT_EQ(output[record + 13], live ? 15u : 0u);
        for (uint32_t channel = 0; channel < 4; ++channel)
            EXPECT_EQ(output[record + 8 + channel], live ? 0x3f000000u : 0u);
    }
    const auto validate = [&](const std::vector<uint32_t>& words) {
        bpermute_oracle::Interpreter validator(plan->validation_words());
        auto commit = validator.run_buffers(
            1, {{0, input}, {1, zero_commit}, {2, authority}, {3, words}, {4, source}}, 1);
        EXPECT_TRUE(validator.error.empty()) << validator.error;
        return commit;
    };
    const auto accepted = validate(output);
    ASSERT_EQ(accepted.size(), capacity.commit_words());
    EXPECT_EQ(accepted[0], 1u);
    EXPECT_EQ(accepted[1], 0u);
    uint32_t indexed = 0;
    for (size_t slot = g::kFragmentDrawCommitHeaderWords; slot < accepted.size(); ++slot)
        if (accepted[slot]) {
            ++indexed;
            EXPECT_LT(accepted[slot] - 1, 36u);
            EXPECT_EQ((accepted[slot] - 1) & 1u, 0u);
        }
    EXPECT_EQ(indexed, 18u);
    auto late = output;
    late[g::kPacketWaveOutputPrefix + 32 * g::kFragmentPacketArchitecturalExportWords + 12] ^= 1;
    const auto denied = validate(late);
    ASSERT_EQ(denied.size(), capacity.commit_words());
    EXPECT_EQ(denied[0], 0u) << "upper32 original-site failure prevents all replay";
    EXPECT_EQ(denied[1], uint32_t(g::FragmentDrawFailure::GuestExport));
    const std::array<const std::vector<uint32_t>*, 6> forms{
        &plan->collect_words(),       &plan->count_words(),      &plan->assembly_words(),
        &kernel.program.packet.spirv, &plan->validation_words(), &plan->replay_words()};
    const char* names[]{"position_free_collector",  "position_free_count",
                        "position_free_assembly",   "position_free_original_ps",
                        "position_free_validation", "position_free_replay"};
    for (size_t index = 0; index < forms.size(); ++index)
        f::retain_source(*forms[index], names[index]);
}
TEST_F(FragmentRasterLaunch, SavedMaskNumericalExposureAndPartialAliasReplacementStayUnproved) {
    // Canonical EXEC alone carries the incoming mask: VCC, M0, unknown specials, numeric zero
    // and ordinary pairs without a saved-mask definition cannot borrow that authority.
    for (uint32_t source : {106u, 107u, 108u, 124u, 125u, 127u, 128u, 20u, 105u}) {
        SCOPED_TRACE(source);
        auto foreign = helper_original();
        foreign[0] = (foreign[0] & ~0xffu) | source;
        const auto refused = helper_requirements(foreign, {1u << 8, 1u << 8});
        EXPECT_EQ(refused.rejection, "fragment-raster-mask-source-unproved:pc=0");
        EXPECT_TRUE(refused.positions.empty());
    }
    const auto canonical = helper_original();
    std::vector<g::Rdna2Inst> decoded;
    g::rdna2_walk(canonical.data(), canonical.size(), decoded);
    ASSERT_FALSE(decoded.empty());
    decoded[0].src[0] = {g::OperandKind::SGPR, 126};   // noncanonical forged IR, not a source SGPR
    EXPECT_EQ(g::fragment_raster_program(decoded, uint32_t(canonical.size()), {1u << 8, 1u << 8}, 0)
                  .rejection,
              "fragment-raster-mask-source-unproved:pc=0");
    auto code = helper_original();
    code.insert(code.begin() + 2, 0x7e020214u); // v_mov_b32 v1,s20 exposes actual mask bits
    EXPECT_EQ(helper_requirements(code, {1u << 8, 1u << 8}).rejection,
              "fragment-raster-vector-source-unproved:pc=2");
    code = helper_original();
    code.insert(code.begin() + 5, 0xbe950380u);   // s_mov_b32 s21,0 expires the saved s20:21 pair
    std::vector<g::Rdna2Inst> replacement;
    g::rdna2_walk(code.data(), code.size(), replacement);
    ASSERT_GT(replacement.size(), 4u);
    EXPECT_EQ(replacement[4].pc, 5u);
    EXPECT_EQ(replacement[4].fmt, g::Rdna2Format::SOP1);
    EXPECT_EQ(replacement[4].opcode, g::kSop1OpcodeMovB32);
    EXPECT_EQ(replacement[4].dst.kind, g::OperandKind::SGPR);
    EXPECT_EQ(replacement[4].dst.value, 21);
    EXPECT_EQ(replacement[4].src[0].kind, g::OperandKind::InlineInt);
    EXPECT_EQ(replacement[4].src[0].value, 0);
    EXPECT_EQ(helper_requirements(code, {1u << 8, 1u << 8}).rejection,
              "fragment-raster-mask-source-unproved:pc=6");
    ASSERT_TRUE(helper_requirements(helper_original(), {1u << 8, 1u << 8}).rejection.empty());
}

TEST_F(FragmentRasterLaunch, HelperExpansionIsNotFinalExportEligibilityOrWaitCompletion) {
    auto code = helper_original();
    code.erase(code.begin() + 5); // do not restore original live EXEC before VM export
    EXPECT_EQ(helper_requirements(code, {1u << 8, 1u << 8}).rejection,
              "fragment-raster-original-live-export-recipe-unimplemented:pc=5");
    code = helper_original();
    code[4] = 0xbf8c3f70u; // real VMCNT0 cannot complete DS's LGKM result
    EXPECT_EQ(helper_requirements(code, {1u << 8, 1u << 8}).rejection,
              "packet-quad-swizzle-result-read-before-lgkm-wait:pc=6");
    const auto positive = helper_requirements(helper_original(), {1u << 8, 1u << 8});
    ASSERT_TRUE(positive.rejection.empty()) << positive.rejection;
}

TEST_F(FragmentRasterLaunch, NormalOriginalRealizerRetainsItsPhysicalEntryAndSelectedSources) {
    g::DrawItem draw;
    ASSERT_TRUE(f::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs);
    const auto& in = *draw.fragment_draw_inputs;
    ASSERT_TRUE(in.launch_source);
    EXPECT_TRUE(in.launch_source->matches(in));
    EXPECT_EQ(in.launch_source->entry(), draw.ps_entry);
    EXPECT_EQ(in.launch_source->launch(), draw.ps_raster_launch);
    EXPECT_EQ(*in.raw_code, f::fragment_words());
    EXPECT_EQ(in.entry.user_data_available, 15u);
    EXPECT_EQ(in.entry.user_data[0], 0x3e800000u);
    EXPECT_FALSE(in.launch.sc_shader_control_available);
    EXPECT_EQ(in.launch.coverage.available, 0u);
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared);
    EXPECT_EQ(prepared->launch_source, in.launch_source);
    EXPECT_FALSE(prepared->ready);
    EXPECT_NE(std::find(prepared->unmet.begin(), prepared->unmet.end(),
                        "packet-logical64-composition-unproved"),
              prepared->unmet.end());
    const auto foreign_modules =
        g::prepare_fragment_packet_inputs(draw.fragment_draw_inputs, false);
    ASSERT_TRUE(foreign_modules);
    EXPECT_FALSE(foreign_modules->launch_source);
}

TEST_F(FragmentRasterLaunch, SameOriginalProgramCannotBorrowAnotherDrawsEntryObservation) {
    const auto* programs = f::register_programs();
    ASSERT_NE(programs, nullptr);
    g::DrawItem first, second;
    ASSERT_TRUE(f::realize_registered(first, *programs, f::color_a));
    ASSERT_TRUE(f::realize_registered(second, *programs, f::color_b));
    ASSERT_TRUE(first.fragment_draw_inputs && second.fragment_draw_inputs);
    const auto& a = *first.fragment_draw_inputs;
    const auto& b = *second.fragment_draw_inputs;
    ASSERT_TRUE(a.launch_source && b.launch_source);
    EXPECT_EQ(a.raw_code, b.raw_code) << "normal original analysis reuse remains allowed";
    EXPECT_NE(a.entry, b.entry);
    EXPECT_FALSE(a.launch_source->matches(b));
    EXPECT_FALSE(b.launch_source->matches(a));
    EXPECT_TRUE(a.launch_source->matches(a));
    EXPECT_TRUE(b.launch_source->matches(b));
}

TEST_F(FragmentRasterLaunch, CopiedOrSameAddressForeignOwnersNeverMatchTheCapturedVersion) {
    g::DrawItem draw;
    ASSERT_TRUE(f::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    const auto& original = *draw.fragment_draw_inputs;
    auto replaced = std::make_shared<g::RasterQuadInputs>(original);
    replaced->raw_code = std::make_shared<const std::vector<uint32_t>>(*original.raw_code);
    EXPECT_FALSE(original.launch_source->matches(*replaced));
    // A borrowed pointer has identical address and bytes, but a different control block.
    replaced->raw_code = std::shared_ptr<const std::vector<uint32_t>>(
        original.raw_code.get(), [](const std::vector<uint32_t>*) {});
    EXPECT_FALSE(original.launch_source->matches(*replaced));
    replaced->raw_code = original.raw_code;
    replaced->source_vs = std::make_shared<const std::vector<uint32_t>>(*original.source_vs);
    EXPECT_FALSE(original.launch_source->matches(*replaced));
    replaced->source_vs = original.source_vs;
    replaced->vgpr_requirements =
        std::make_shared<const g::FragmentPacketVgprRequirements>(*original.vgpr_requirements);
    EXPECT_FALSE(original.launch_source->matches(*replaced));
    const auto prepared = g::prepare_fragment_packet_inputs(replaced, true);
    ASSERT_TRUE(prepared);
    EXPECT_FALSE(prepared->launch_source);
    EXPECT_FALSE(prepared->ready);
}

TEST_F(FragmentRasterLaunch, RawKnownnessAndParameterRoutingRemainPartOfTheDrawAssociation) {
    g::DrawItem draw;
    ASSERT_TRUE(f::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    const auto& original = *draw.fragment_draw_inputs;
    for (uint32_t kind = 0; kind < 7; ++kind) {
        auto replaced = original;
        if (kind == 0) replaced.entry.user_data_available &= ~1u;
        if (kind == 1) replaced.launch.sc_shader_control_available = true;
        if (kind == 2) replaced.launch.coverage.available = 1u << 26; // observed clear control
        if (kind == 3) replaced.interpolation.parameter_locations[0][2] = 8;
        if (kind == 4) replaced.system_inputs.addr = 2;
        if (kind == 5) replaced.generated_interpolation_geometry = true;
        if (kind == 6) replaced.owned_wave_pending = true;
        EXPECT_FALSE(original.launch_source->matches(replaced)) << kind;
    }
    EXPECT_TRUE(original.launch_source->matches(original));
}

TEST_F(FragmentRasterLaunch, StalePhysicalPrefixWithoutAProducingPsStaysUnobserved) {
    g::GpuState state;
    state.sh[0x00c] = 0x12345678u; // actual PS USER_DATA0
    state.sh[0x00b] = 2u; // actual PS RSRC2 USER_SGPR=1
    g::DrawItem draw;
    EXPECT_FALSE(g::realize_draw_item(state, nullptr, 3, 64, false, draw, nullptr, true));
    EXPECT_EQ(draw.ps_entry, g::FragmentEntryFacts{});
    EXPECT_FALSE(draw.fragment_draw_inputs);
}

TEST_F(FragmentRasterLaunch,
       OriginalProducingPullAndLinearInputsCompileTheCompleteCollectorProfile) {
    namespace pull = prosper::test::ps_pull;
    for (const auto& original : pull::cases) {
        g::DrawItem draw;
        ASSERT_TRUE(pull::realize(original, draw));
        ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
        const auto& in = *draw.fragment_draw_inputs;
        ASSERT_TRUE(in.launch_source->matches(in));
        const auto collection =
            g::compile_fragment_raster_launch_collection(in, raster_device(), 16);
        ASSERT_TRUE(collection.rejection.empty()) << original.name << ':' << collection.rejection;
        EXPECT_FALSE(collection.collector.empty());
        EXPECT_FALSE(collection.geometry.empty());
        EXPECT_TRUE(collection.parameters.requires_geometry);
        EXPECT_EQ(collection.parameters.attribute_mask, in.interpolation.attribute_mask);
        EXPECT_EQ(collection.parameters.smooth_mask, in.interpolation.smooth_mask);
        const uint32_t supplied_words = original.words == pull::Words::Pull              ? 3u
                                        : original.words == pull::Words::CenterAndLinear ? 16u
                                                                                         : 14u;
        EXPECT_EQ(collection.shape.lane_words, 9u + supplied_words);
        EXPECT_EQ(collection.geometry,
                  g::recompile_interpolation_geometry(collection.parameters, false, false,
                                                      in.float_transport, true));
        auto disabled = raster_device();
        disabled.raster->geometry_shader_enabled = false;
        const auto missing_geometry =
            g::compile_fragment_raster_launch_collection(in, disabled, 16);
        EXPECT_EQ(missing_geometry.rejection,
                  "fragment-raster-launch-enabled-geometry-unavailable");
        EXPECT_TRUE(missing_geometry.collector.empty());
        EXPECT_TRUE(missing_geometry.geometry.empty());
        auto replaced = in;
        replaced.launch.input_addr ^= 2u;
        EXPECT_EQ(
            g::compile_fragment_raster_launch_collection(replaced, raster_device(), 16).rejection,
            "fragment-raster-launch-producing-observation-unavailable");
    }
}

TEST_F(FragmentRasterLaunch, SameEnabledDeviceOwnsEveryPreRasterAndFragmentBudget) {
    namespace pull = prosper::test::ps_pull;
    g::DrawItem draw;
    ASSERT_TRUE(pull::realize(pull::cases[1], draw));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    const auto& in = *draw.fragment_draw_inputs;
    const auto complete = g::compile_fragment_raster_launch_collection(in, raster_device(), 16);
    ASSERT_TRUE(complete.rejection.empty()) << complete.rejection;
    const auto vertex = g::fragment_raster_output_interface(*in.source_vs, 0);
    const auto geometry = g::fragment_raster_output_interface(complete.geometry, 3);
    ASSERT_TRUE(vertex.available && geometry.available);
    ASSERT_GT(vertex.components, 0u);
    ASSERT_GT(geometry.components, 0u);
    EXPECT_EQ(geometry.output_vertices, 3u);
    for (uint32_t kind = 0; kind < 8; ++kind) {
        auto device = raster_device();
        if (kind == 0) device.device_identity = 0;
        if (kind == 1) device.raster.reset();
        if (kind == 2)
            device.raster->max_vertex_output_components = uint32_t(vertex.components - 1);
        if (kind == 3) device.raster->max_geometry_input_components = 0;
        if (kind == 4)
            device.raster->max_geometry_output_components = uint32_t(geometry.components - 1);
        if (kind == 5)
            device.raster->max_geometry_total_output_components =
                uint32_t(geometry.components * 3 - 1);
        if (kind == 6) device.raster->max_geometry_output_vertices = 2;
        if (kind == 7) device.raster->max_fragment_input_components = 4;
        const auto refused = g::compile_fragment_raster_launch_collection(in, device, 16);
        EXPECT_FALSE(refused.rejection.empty()) << kind;
        EXPECT_TRUE(refused.collector.empty()) << kind;
        EXPECT_TRUE(refused.geometry.empty()) << kind;
    }
    EXPECT_EQ(g::compile_fragment_raster_launch_collection(in, {1, false, false}, 16).rejection,
              "fragment-raster-launch-enabled-device-observation-unavailable")
        << "old three-field callers do not acquire the new recipe";
}

TEST_F(FragmentRasterLaunch, WiderParameterProgramStillNeedsItsCompleteEntryAndExecutionRecipe) {
    namespace pull = prosper::test::ps_pull;
    g::DrawItem unavailable;
    ASSERT_TRUE(pull::realize(pull::cases[1], unavailable));
    ASSERT_TRUE(unavailable.fragment_draw_inputs);
    EXPECT_FALSE(unavailable.fragment_draw_inputs->float_mode.available);
    const auto unavailable_prepared =
        g::prepare_fragment_packet_inputs(unavailable.fragment_draw_inputs, true);
    ASSERT_TRUE(unavailable_prepared);
    const auto unavailable_plan = g::cached_fragment_draw_program(
        *unavailable.fragment_draw_inputs, *unavailable_prepared, raster_device(), 16);
    ASSERT_TRUE(unavailable_plan);
    EXPECT_EQ(unavailable_plan->rejection_reason(),
              "fragment-draw-original-float-mode-unavailable");
    EXPECT_FALSE(unavailable_plan->capacity_owner());
    g::DrawItem draw;
    ASSERT_TRUE(pull::realize(pull::cases[1], draw, f::ieee_rsrc1));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    EXPECT_TRUE(draw.fragment_draw_inputs->float_mode.available);
    const auto prepared = g::prepare_fragment_packet_inputs(draw.fragment_draw_inputs, true);
    ASSERT_TRUE(prepared && prepared->launch_source);
    const auto first =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, raster_device(), 16);
    ASSERT_TRUE(first);
    EXPECT_EQ(first->rejection_reason(), "fragment-raster-vector-source-unproved:pc=0");
    EXPECT_FALSE(first->raster_launch_collection());
    EXPECT_FALSE(first->capacity_owner());
    const auto again =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, raster_device(), 16);
    EXPECT_EQ(first, again) << "warm code/profile reuse, not a per-draw compilation";
    auto smaller = raster_device();
    smaller.raster->max_geometry_total_output_components = 1;
    const auto short_device =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, smaller, 16);
    EXPECT_NE(first, short_device);
    EXPECT_EQ(short_device->rejection_reason(), first->rejection_reason());
    EXPECT_FALSE(short_device->raster_launch_collection());
    const auto transaction = g::instantiate_fragment_draw_transaction(
        first, draw.fragment_draw_inputs, *prepared, 16, 16, 2, 1);
    EXPECT_FALSE(transaction.rejection().empty());
    EXPECT_TRUE(transaction.entry_words().empty());
    EXPECT_FALSE(prepared->ready);
}

TEST_F(FragmentRasterLaunch, GenuineDrawBoundSavedLiveHelperRecipeConnectsTheOriginalKernel) {
    namespace helper = prosper::test::fragment_raster;
    g::DrawItem draw;
    ASSERT_TRUE(helper::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    const auto& in = *draw.fragment_draw_inputs;
    EXPECT_EQ(*in.raw_code, helper::original());
    EXPECT_EQ(in.system_inputs, (g::PixelSystemInputMapping{1u << 8, 1u << 8}));
    EXPECT_FALSE(in.launch.baryc_cntl & (1u << 24));
    EXPECT_FALSE(in.launch.coverage.has(g::RasterCoverageControl::CentroidPriority0));
    if (draw.fs_words().empty()) {
        EXPECT_TRUE(in.owned_wave_pending);
        EXPECT_TRUE(in.launch_source->pending_original_has_no_external_effects());
    }
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared && prepared->launch_source);
    EXPECT_FALSE(prepared->ready) << "generic preparation is not draw transaction authority";
    const auto plan = g::cached_fragment_draw_program(in, *prepared, raster_device(), 48);
    ASSERT_TRUE(plan);
    ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
    ASSERT_TRUE(plan->capacity_owner());
    ASSERT_EQ(plan->capacity_owner()->raster_inputs().size(), 1u);
    EXPECT_EQ(plan->capacity_owner()->raster_inputs().front(), (g::FragmentDrawRasterInput{0, 5}));
    EXPECT_EQ(plan->capacity_owner()->kernel()->guest_code, helper::original());
    const auto& packet = plan->capacity_owner()->kernel()->program.packet;
    EXPECT_EQ(packet.initial_mask_availability, g::kPacketInitialExec);
    EXPECT_EQ(packet.export_observation, g::FragmentPacketExportObservation::Architectural);
    ASSERT_EQ(packet.export_sites.size(), 1u);
    EXPECT_EQ(packet.export_sites.front(), (g::FragmentPacketExportSite{6, 0, 15, 0, 1, 1}));
    for (const auto* source :
         {&plan->collect_words(), &plan->count_words(), &plan->assembly_words(), &packet.spirv,
          &plan->validation_words(), &plan->replay_words()})
        EXPECT_FALSE(source->empty());
    // Independent structural rail for the actual collector: all four constant-index gathers
    // must execute before any election/budget divergence. Helper memory stores cannot substitute
    // for this communication. Real helper values and final pixels have separate sink oracles.
    const auto& collector = plan->collect_words();
    std::map<uint32_t, uint32_t> constants;
    std::array<uint32_t, 4> gathers{};
    bool branched = false;
    for (size_t pc = 5; pc < collector.size();) {
        const uint32_t count = collector[pc] >> 16, opcode = collector[pc] & 0xffffu;
        ASSERT_NE(count, 0u);
        ASSERT_LE(count, collector.size() - pc);
        if (opcode == 43 && count == 4) constants[collector[pc + 2]] = collector[pc + 3];
        if (opcode == 249 || opcode == 250 || opcode == 251) branched = true;
        if (opcode == 365) {   // OpGroupNonUniformQuadBroadcast
            ASSERT_EQ(count, 6u);
            EXPECT_FALSE(branched);
            ASSERT_TRUE(constants.contains(collector[pc + 5]));
            const uint32_t index = constants.at(collector[pc + 5]);
            ASSERT_LT(index, gathers.size());
            ++gathers[index];
        }
        EXPECT_NE(opcode, 333u) << "no subgroup Elect grants helper publication authority";
        pc += count;
    }
    for (const auto gathered : gathers) EXPECT_EQ(gathered, plan->collector_shape().lane_words);
    EXPECT_EQ(plan, g::cached_fragment_draw_program(in, *prepared, raster_device(), 48));
    const auto transaction = g::instantiate_fragment_draw_transaction(
        plan, draw.fragment_draw_inputs, *prepared, 15, 12, 1, 1);
    EXPECT_TRUE(transaction.rejection().empty()) << transaction.rejection();
    EXPECT_FALSE(transaction.entry_words().empty());
}

TEST_F(FragmentRasterLaunch, WorkitemBridgeRejectsUnknownOrCoverageChangingPhysicalFacts) {
    namespace helper = prosper::test::fragment_raster;
    g::DrawItem draw;
    ASSERT_TRUE(helper::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    const auto& launch = draw.fragment_draw_inputs->launch;
    ASSERT_EQ(g::fragment_raster_workitem_gap(launch), nullptr);
    for (uint32_t kind = 0; kind < 11; ++kind) {
        auto changed = launch;
        if (kind == 0) changed.sc_shader_control_available = false;
        if (kind == 1) changed.sc_mode_cntl_0 = 1;
        if (kind == 2) changed.db_shader_control = 1u << 6;   // actual shader kill
        if (kind == 3) changed.db_shader_control = 1u << 4;   // EARLY_Z
        if (kind == 4) changed.sc_aa_config = 1;
        if (kind == 5) changed.coverage.available &= ~(1u << 0);
        if (kind == 6) changed.coverage.words[uint32_t(g::RasterCoverageControl::AaMaskTop)] = 0;
        if (kind == 7)
            changed.coverage.words[uint32_t(g::RasterCoverageControl::RenderControl)] = 1;
        if (kind == 8) changed.coverage.words[uint32_t(g::RasterCoverageControl::Conservative)] = 1;
        if (kind == 9)
            changed.coverage.words[uint32_t(g::RasterCoverageControl::VertexControl)] = 0;
        if (kind == 10) changed.db_shader_control &= ~(1u << 11);
        EXPECT_NE(g::fragment_raster_workitem_gap(changed), nullptr) << kind;
    }
    auto different_break = launch;
    different_break.sc_shader_control = 0x6fu;
    EXPECT_EQ(g::fragment_raster_workitem_gap(different_break), nullptr)
        << "only the separately proved quad-local program is insensitive to interquad packing";
}

TEST_F(FragmentRasterLaunch, PendingEmptyFsCannotBorrowWarmCodeOrAnotherDrawObservation) {
    namespace helper = prosper::test::fragment_raster;
    g::DrawItem draw;
    ASSERT_TRUE(helper::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs && draw.fragment_draw_inputs->launch_source);
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared);
    auto cold_missing = std::make_shared<g::RasterQuadInputs>(*draw.fragment_draw_inputs);
    cold_missing->launch_source.reset();
    const auto cold_prepared = g::prepare_fragment_packet_inputs(cold_missing, true);
    ASSERT_TRUE(cold_prepared && !cold_prepared->launch_source);
    const auto cold_refusal =
        g::cached_fragment_draw_program(*cold_missing, *cold_prepared, raster_device(), 48);
    ASSERT_TRUE(cold_refusal);
    EXPECT_EQ(cold_refusal->rejection_reason(),
              "fragment-draw-draw-bound-launch-source-unavailable");
    EXPECT_FALSE(cold_refusal->capacity_owner());
    const auto warm =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, raster_device(), 48);
    ASSERT_TRUE(warm);
    ASSERT_TRUE(warm->rejection_reason().empty()) << warm->rejection_reason();
    for (uint32_t kind = 0; kind < 5; ++kind) {
        auto forged = std::make_shared<g::RasterQuadInputs>(*draw.fragment_draw_inputs);
        if (kind == 0) forged->launch_source.reset();
        if (kind == 1)
            forged->raw_code = std::make_shared<const std::vector<uint32_t>>(*forged->raw_code);
        if (kind == 2) forged->launch.sc_shader_control ^= 4;
        if (kind == 3) forged->source_fs = std::make_shared<const std::vector<uint32_t>>();
        if (kind == 4) forged->owned_wave_pending = !forged->owned_wave_pending;
        const auto missing = g::prepare_fragment_packet_inputs(forged, true);
        ASSERT_TRUE(missing);
        EXPECT_FALSE(missing->launch_source) << kind;
        const auto refusal =
            g::cached_fragment_draw_program(*forged, *missing, raster_device(), 48);
        ASSERT_TRUE(refusal);
        EXPECT_NE(refusal, warm) << kind;
        EXPECT_FALSE(refusal->rejection_reason().empty()) << kind;
        EXPECT_FALSE(refusal->capacity_owner()) << kind;
        if (kind == 0)
            EXPECT_EQ(refusal->rejection_reason(),
                      "fragment-draw-draw-bound-launch-source-unavailable");
        const auto compile_calls = g::fragment_draw_cache_stats().program_compile_calls;
        EXPECT_EQ(warm, g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared,
                                                        raster_device(), 48));
        EXPECT_EQ(g::fragment_draw_cache_stats().program_compile_calls, compile_calls)
            << "invalid draw authority must neither poison nor recompile the valid warm plan";
    }
    const auto transaction = g::instantiate_fragment_draw_transaction(
        warm, draw.fragment_draw_inputs, *prepared, 15, 12, 1, 1);
    EXPECT_TRUE(transaction.rejection().empty()) << transaction.rejection();
}

TEST_F(FragmentRasterLaunch, FreshRegisteredGenerationAndPhysicalLaunchCannotReuseOldPermission) {
    namespace helper = prosper::test::fragment_raster;
    g::DrawItem first, replacement, changed_launch;
    ASSERT_TRUE(helper::realize(first));
    ASSERT_TRUE(helper::realize(replacement));
    auto context = helper::context();
    for (auto& [reg, value] : context)
        if (reg == prosper::agc::Pm4::PA_SC_SHADER_CONTROL) value = 4;
    ASSERT_TRUE(
        f::realize(changed_launch, f::color_a, helper::original(), f::ieee_rsrc1, 15, context));
    ASSERT_TRUE(first.fragment_draw_inputs && replacement.fragment_draw_inputs &&
                changed_launch.fragment_draw_inputs);
    const auto prepared = f::prepare(first);
    ASSERT_TRUE(prepared);
    const auto warm = g::cached_fragment_draw_program(*first.fragment_draw_inputs, *prepared,
                                                      raster_device(), 48);
    ASSERT_TRUE(warm);
    ASSERT_TRUE(warm->rejection_reason().empty()) << warm->rejection_reason();
    const auto& original = *first.fragment_draw_inputs;
    ASSERT_TRUE(original.launch_source);
    EXPECT_EQ(*original.raw_code, *replacement.fragment_draw_inputs->raw_code);
    EXPECT_NE(original.raw_code, replacement.fragment_draw_inputs->raw_code)
        << "a fresh registered mapping is a genuine independently captured source generation";
    for (const auto* draw : {&replacement, &changed_launch}) {
        ASSERT_TRUE(draw->fragment_draw_inputs->launch_source);
        EXPECT_FALSE(original.launch_source->matches(*draw->fragment_draw_inputs));
        auto forged = std::make_shared<g::RasterQuadInputs>(*draw->fragment_draw_inputs);
        forged->launch_source = original.launch_source;
        const auto missing = g::prepare_fragment_packet_inputs(forged, true);
        ASSERT_TRUE(missing);
        EXPECT_FALSE(missing->launch_source);
        const auto refused =
            g::cached_fragment_draw_program(*forged, *missing, raster_device(), 48);
        ASSERT_TRUE(refused);
        EXPECT_FALSE(refused->rejection_reason().empty());
        EXPECT_FALSE(refused->capacity_owner());
        const auto actual = f::prepare(*draw);
        ASSERT_TRUE(actual && actual->launch_source);
        const auto old_transaction = g::instantiate_fragment_draw_transaction(
            warm, draw->fragment_draw_inputs, *actual, 15, 12, 1, 1);
        // Identical code/launch may reuse copied compiled SOURCE, but every draw must retain its
        // OWN genuine observation. A different physical launch cannot borrow the old profile.
        if (draw == &changed_launch) {
            EXPECT_EQ(old_transaction.rejection(), "fragment-draw-producing-profile-mismatch");
            EXPECT_TRUE(old_transaction.entry_words().empty());
        } else {
            EXPECT_TRUE(old_transaction.rejection().empty()) << old_transaction.rejection();
        }
    }
}

TEST_F(FragmentRasterLaunch, PublicEmptyFsAndPendingFlagsNeverMintAnOriginalBodyProof) {
    g::DrawItem native;
    ASSERT_TRUE(f::realize(native));
    ASSERT_TRUE(native.fragment_draw_inputs && native.fragment_draw_inputs->launch_source);
    ASSERT_FALSE(native.fs_words().empty());
    auto forged = std::make_shared<g::RasterQuadInputs>(*native.fragment_draw_inputs);
    forged->source_fs = std::make_shared<const std::vector<uint32_t>>();
    forged->owned_wave_pending = true;
    forged->raw_matches_producing_source = true;
    EXPECT_FALSE(forged->launch_source->matches(*forged));
    EXPECT_FALSE(forged->launch_source->pending_original_has_no_external_effects());
    const auto missing = g::prepare_fragment_packet_inputs(forged, true);
    ASSERT_TRUE(missing);
    EXPECT_FALSE(missing->launch_source);
    const auto refused = g::cached_fragment_draw_program(*forged, *missing, raster_device(), 48);
    ASSERT_TRUE(refused);
    EXPECT_EQ(refused->rejection_reason(), "fragment-draw-producing-owner-unavailable");
    EXPECT_FALSE(refused->capacity_owner());
}

TEST_F(FragmentRasterLaunch, SeventeenActualRegisteredHelperProfilesStayWarmAcrossNewDrawOwners) {
    namespace helper = prosper::test::fragment_raster;
    constexpr uint32_t count = 17;
    std::array<g::DrawItem, count> retained;
    std::array<std::shared_ptr<const g::FragmentDrawProgramPlan>, count> plans;
    const auto context = helper::context();
    const auto before = g::fragment_draw_cache_stats().program_compile_calls;
    for (uint32_t variant = 0; variant < count; ++variant) {
        auto original = helper::original();
        original.insert(original.begin(), 0xbf800000u | (variant + 100));   // genuine original NOP
        ASSERT_TRUE(
            f::realize(retained[variant], f::color_a, original, f::ieee_rsrc1, 15, context));
        ASSERT_TRUE(retained[variant].fragment_draw_inputs);
        const auto prepared = f::prepare(retained[variant]);
        ASSERT_TRUE(prepared);
        plans[variant] = g::cached_fragment_draw_program(*retained[variant].fragment_draw_inputs,
                                                         *prepared, raster_device(), 48);
        ASSERT_TRUE(plans[variant]);
        ASSERT_TRUE(plans[variant]->rejection_reason().empty())
            << variant << ':' << plans[variant]->rejection_reason();
    }
    const auto warmed = g::fragment_draw_cache_stats().program_compile_calls;
    EXPECT_EQ(warmed - before, count);
    for (uint32_t repeat = 0; repeat < 4; ++repeat)
        for (uint32_t variant = 0; variant < count; ++variant) {
            g::DrawItem next;
            auto original = helper::original();
            original.insert(original.begin(), 0xbf800000u | (variant + 100));
            ASSERT_TRUE(f::realize(next, f::color_b, original, f::ieee_rsrc1, 15, context));
            ASSERT_TRUE(next.fragment_draw_inputs && next.fragment_draw_inputs->launch_source);
            EXPECT_NE(next.fragment_draw_inputs->launch_source,
                      retained[variant].fragment_draw_inputs->launch_source);
            EXPECT_NE(next.fragment_draw_inputs->raw_code,
                      retained[variant].fragment_draw_inputs->raw_code);
            const auto prepared = f::prepare(next);
            ASSERT_TRUE(prepared);
            const auto reused = g::cached_fragment_draw_program(*next.fragment_draw_inputs,
                                                                *prepared, raster_device(), 48);
            EXPECT_EQ(reused, plans[variant]) << variant;
            const auto transaction = g::instantiate_fragment_draw_transaction(
                reused, next.fragment_draw_inputs, *prepared, 15, 12, 1, 1);
            EXPECT_TRUE(transaction.rejection().empty()) << transaction.rejection();
        }
    EXPECT_EQ(g::fragment_draw_cache_stats().program_compile_calls, warmed)
        << "no recurring cold emit for an actually realized 17-key helper working set";
}

TEST_F(FragmentRasterLaunch, RasterRowsCannotForgeMaskAvailabilityOrPhysicalEntryShape) {
    namespace helper = prosper::test::fragment_raster;
    g::DrawItem draw;
    ASSERT_TRUE(helper::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs);
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared);
    const auto plan =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, raster_device(), 48);
    ASSERT_TRUE(plan);
    ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
    ASSERT_TRUE(plan->capacity_owner());
    const auto& capacity = *plan->capacity_owner();
    for (uint32_t kind = 0; kind < 7; ++kind) {
        auto kernel = std::make_shared<g::FragmentPacketKernel>(*capacity.kernel());
        auto rows = capacity.raster_inputs();
        ASSERT_EQ(rows.size(), 1u);
        if (kind == 0) kernel->program.packet.initial_mask_availability |= g::kPacketInitialVcc;
        if (kind == 1) kernel->program.packet.input_stride -= 1;
        if (kind == 2) kernel->topology = g::FragmentPacketQuadTopology::Unknown;
        if (kind == 3) rows[0].collector_word = 7;   // POS_Z is not established by this entry shape
        if (kind == 4) rows[0].reg = 255;   // not a demanded packed position column
        if (kind == 5) rows.push_back(rows[0]);
        if (kind == 6) rows[0].reg = 256;
        std::string rejection;
        const auto refused = g::fragment_draw_capacity(kernel, capacity.collector(), rejection,
                                                       capacity.entry_recipe(), std::move(rows));
        EXPECT_FALSE(refused) << kind;
        EXPECT_EQ(rejection, "fragment-draw-raster-entry-schema-invalid") << kind;
    }
}

TEST_F(FragmentRasterLaunch, AssembledHelperBackingRunsSavedLiveOriginalBeforeWholeDrawCommit) {
    namespace helper = prosper::test::fragment_raster;
    g::DrawItem draw;
    ASSERT_TRUE(helper::realize(draw));
    ASSERT_TRUE(draw.fragment_draw_inputs);
    const auto prepared = f::prepare(draw);
    ASSERT_TRUE(prepared);
    const auto plan =
        g::cached_fragment_draw_program(*draw.fragment_draw_inputs, *prepared, raster_device(), 48);
    ASSERT_TRUE(plan);
    ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
    ASSERT_TRUE(plan->capacity_owner());
    const auto& capacity = *plan->capacity_owner();
    const auto& kernel = *capacity.kernel();
    const auto transaction = g::instantiate_fragment_draw_transaction(
        plan, draw.fragment_draw_inputs, *prepared, 16, 10, 1, 1);
    ASSERT_TRUE(transaction.rejection().empty()) << transaction.rejection();
    // Explicit offline host-observation input, not a claim of a device raster launch. Every
    // occupied quad has live lanes0/2 and genuine backed helpers1/3. The original selects lane3's
    // nonconstant POS_X with VDST==ADDR, so guessed helper zeros or live-only copies cannot pass.
    const auto& shape = plan->collector_shape();
    std::vector<uint32_t> source(capacity.collector_words(), 0);
    const uint32_t header[]{40, 0, shape.record_words, g::kRasterQuadMagic};
    std::copy(std::begin(header), std::end(header), source.begin());
    for (uint32_t quad = 0; quad < 40; ++quad)
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const bool live = !(lane & 1u);
            const uint32_t words[]{
                uint32_t(!live),
                uint32_t(live),
                uint32_t(live),
                1,
                0,
                std::bit_cast<uint32_t>(float(2 * (quad % 8) + (lane & 1u)) + .5f),
                std::bit_cast<uint32_t>(float(2 * (quad / 8) + (lane >> 1u)) + .5f),
                0,
                std::bit_cast<uint32_t>(1.f)};
            std::copy(std::begin(words), std::end(words),
                      source.begin() + 4 + (4 * quad + lane) * shape.lane_words);
        }
    const std::vector<uint32_t> authority(capacity.authority().begin(), capacity.authority().end());
    const std::vector<uint32_t> zero_commit(capacity.commit_words(), 0);
    bpermute_oracle::Interpreter count(plan->count_words());
    count.extra_writable_bindings = {0, 4};
    auto input = count.run_buffers(1,
                                   {{0, source},
                                    {1, std::vector<uint32_t>(capacity.input_words(), 0)},
                                    {2, authority},
                                    {3, transaction.entry_words()},
                                    {4, zero_commit}},
                                   1);
    ASSERT_TRUE(count.error.empty()) << count.error;
    source = count.writable_result(0);
    EXPECT_EQ(count.writable_result(4), zero_commit);
    ASSERT_EQ(input.size(), capacity.input_words());
    EXPECT_EQ(input[1], 3u);
    EXPECT_EQ(input[4], 40u);
    EXPECT_EQ(input[8], 0u);
    for (uint32_t wave : {2u, 0u, 1u}) {
        bpermute_oracle::Interpreter assemble(plan->assembly_words());
        input = assemble.run_buffers(
            64, {{0, source}, {1, input}, {2, authority}, {3, transaction.entry_words()}}, 1, wave);
        ASSERT_TRUE(assemble.error.empty()) << assemble.error;
    }
    const uint32_t columns = uint32_t(kernel.layout.vgprs.size());
    ASSERT_EQ(columns, 1u);
    for (uint32_t worker = 0; worker < 192; ++worker) {
        const bool occupied = worker < 160, live = occupied && !(worker & 1u);
        const auto row = capacity.placement(worker / 64).input_base + 2 +
                         (worker % 64) * kernel.program.packet.input_stride;
        ASSERT_LT(row + 2 * columns + 3, input.size());
        EXPECT_EQ(input[row + columns], uint32_t(occupied));
        EXPECT_EQ(input[row + 2 * columns], uint32_t(live));
        EXPECT_EQ(input[row + 2 * columns + 1], 0u) << "no initial VCC authority";
        EXPECT_EQ(input[row + 2 * columns + 2], 0u) << "no initial SCC authority";
        EXPECT_EQ(input[row + 2 * columns + 3], uint32_t(live));
        if (occupied) EXPECT_EQ(input[row], source[4 + worker * shape.lane_words + 5]);
    }
    const auto execute = [&](const std::vector<uint32_t>& current) {
        std::vector<uint32_t> output(capacity.output_words(), 0);
        for (uint32_t wave : {2u, 0u, 1u}) {
            bpermute_oracle::Interpreter original(kernel.program.packet.spirv);
            output = original.run_buffers(64, {{0, current}, {1, output}, {2, authority}}, 1, wave);
            EXPECT_TRUE(original.error.empty()) << original.error;
        }
        return output;
    };
    const auto output = execute(input);
    ASSERT_EQ(output.size(), capacity.output_words());
    for (uint32_t worker = 0; worker < 192; ++worker) {
        const bool live = worker < 160 && !(worker & 1u);
        const auto record = capacity.placement(worker / 64).output_base +
                            g::kPacketWaveOutputPrefix +
                            (worker % 64) * g::kFragmentPacketArchitecturalExportWords;
        ASSERT_LT(record + 13, output.size());
        EXPECT_EQ(output[record], 1u);
        EXPECT_EQ(output[record + 1], uint32_t(live));
        EXPECT_EQ(output[record + 2], uint32_t(live));
        EXPECT_EQ(output[record + 13], live ? 15u : 0u);
        for (uint32_t channel = 0; channel < 4; ++channel)
            EXPECT_EQ(output[record + 8 + channel],
                      live ? source[4 + ((worker & ~3u) + 3) * shape.lane_words + 5] : 0u)
                << "original nonconstant helper word at worker=" << worker;
    }
    const auto validate = [&](const auto& current_source, const auto& current_output) {
        bpermute_oracle::Interpreter validator(plan->validation_words());
        auto commit = validator.run_buffers(1,
                                            {{0, input},
                                             {1, zero_commit},
                                             {2, authority},
                                             {3, current_output},
                                             {4, current_source}},
                                            1);
        EXPECT_TRUE(validator.error.empty()) << validator.error;
        return commit;
    };
    const auto accepted = validate(source, output);
    ASSERT_EQ(accepted.size(), capacity.commit_words());
    EXPECT_EQ(accepted[0], 1u);
    EXPECT_EQ(accepted[1], 0u);
    uint32_t indexed = 0;
    for (size_t slot = g::kFragmentDrawCommitHeaderWords; slot < accepted.size(); ++slot) {
        if (!accepted[slot]) continue;
        ++indexed;
        EXPECT_EQ((accepted[slot] - 1) & 1u, 0u) << "helpers never enter the replay index";
    }
    EXPECT_EQ(indexed, 80u);
    // Recheck retained provenance at the final gate, independently of a prior good assembly.
    // These deliberate private-plane corruptions do not fabricate another guest launch.
    auto malformed_source = source;
    malformed_source[4 + shape.lane_words + 1] = 1;   // helper coverage remains unavailable
    const auto bad_coverage = validate(malformed_source, output);
    ASSERT_EQ(bad_coverage.size(), capacity.commit_words());
    EXPECT_EQ(bad_coverage[0], 0u);
    EXPECT_EQ(bad_coverage[1], uint32_t(g::FragmentDrawFailure::HelperEntryUnavailable));
    malformed_source = source;
    malformed_source[4 + shape.lane_words + 4] = 1;   // primitive identity differs inside the quad
    const auto bad_primitive = validate(malformed_source, output);
    ASSERT_EQ(bad_primitive.size(), capacity.commit_words());
    EXPECT_EQ(bad_primitive[0], 0u);
    EXPECT_EQ(bad_primitive[1], uint32_t(g::FragmentDrawFailure::CollectionRecord));
    malformed_source = source;
    for (uint32_t lane = 0; lane < 4; ++lane) {
        malformed_source[4 + lane * shape.lane_words] = 1;
        malformed_source[4 + lane * shape.lane_words + 1] = 0;
        malformed_source[4 + lane * shape.lane_words + 2] = 0;
    }
    bpermute_oracle::Interpreter helper_only(plan->count_words());
    helper_only.extra_writable_bindings = {0, 4};
    const auto rejected_count =
        helper_only.run_buffers(1,
                                {{0, malformed_source},
                                 {1, std::vector<uint32_t>(capacity.input_words(), 0)},
                                 {2, authority},
                                 {3, transaction.entry_words()},
                                 {4, zero_commit}},
                                1);
    ASSERT_TRUE(helper_only.error.empty()) << helper_only.error;
    ASSERT_EQ(rejected_count.size(), capacity.input_words());
    EXPECT_EQ(rejected_count[1], 0u);
    EXPECT_EQ(rejected_count[4], 0u);
    EXPECT_EQ(rejected_count[8], uint32_t(g::FragmentDrawFailure::CollectionRecord));
    auto absent_helper = input;
    const auto helper63 =
        capacity.placement(1).input_base + 2 + 63 * kernel.program.packet.input_stride;
    ASSERT_LT(helper63 + columns, absent_helper.size());
    absent_helper[helper63 + columns] = 0;   // upper-half helper, selected by live logical lane60
    const auto failed_words = execute(absent_helper);
    const auto denied = validate(source, failed_words);
    ASSERT_EQ(denied.size(), capacity.commit_words());
    EXPECT_EQ(denied[0], 0u);
    EXPECT_EQ(denied[1], uint32_t(g::FragmentDrawFailure::GuestRuntime));
    auto late = output;
    const auto late_record = capacity.placement(2).output_base + g::kPacketWaveOutputPrefix +
                             28 * g::kFragmentPacketArchitecturalExportWords;
    ASSERT_LT(late_record + 12, late.size());
    late[late_record + 12] ^= 1u;
    const auto refused = validate(source, late);
    ASSERT_EQ(refused.size(), capacity.commit_words());
    EXPECT_EQ(refused[0], 0u) << "late original wave failure cannot publish earlier pixels";
    EXPECT_EQ(refused[1], uint32_t(g::FragmentDrawFailure::GuestExport));
    const auto& packet = kernel.program.packet;
    const std::array<const std::vector<uint32_t>*, 6> forms{
        &plan->collect_words(), &plan->count_words(),      &plan->assembly_words(),
        &packet.spirv,          &plan->validation_words(), &plan->replay_words()};
    const char* names[]{"helper_collector",   "helper_count",      "helper_assembly",
                        "helper_original_ps", "helper_validation", "helper_replay"};
    for (size_t index = 0; index < forms.size(); ++index)
        f::retain_source(*forms[index], names[index]);
}
