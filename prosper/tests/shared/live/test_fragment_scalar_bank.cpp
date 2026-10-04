// Actual AGC registration -> ordered issuer/pre-fold seal -> immutable bank/native lineage.
// Uses the normal live backend's real pending/failure/allocation queries without registering a
// renderer or creating a device. No fabricated manifests, reflection or complete=true authority.
// These CPU observations prove ownership/refusal, not GPU execution or CPU-writer exclusion.
#include "fixtures/fragment_scalar_bank_fixture.hpp"
#include "fixtures/render_runner.h"
#include "gpu/execute/ordered_graphics_read_point_internal.hpp"
#include "gpu/recompiler/original_fragment_producer.hpp"
#include "gpu/recompiler/original_graphics_draw_effects.hpp"
#include "shared/live/live_renderer.hpp"
#include <gtest/gtest.h>

namespace {
namespace g = prosper::gpu;
namespace f = prosper::test::scalar_bank;
namespace r = prosper::test;
namespace p = prosper::agc::Pm4;
namespace live = prosper::frontend;
class FragmentScalarBank : public testing::Test {
protected:
    f::Scene scene;
    std::vector<g::DrawItem> observed;
    uint32_t render_calls = 0;
    void SetUp() override {
        ASSERT_TRUE(scene.create());
        ASSERT_EQ(r::backend_pending_submission_batches().load(), 0);
        ASSERT_FALSE(r::backend_has_unproven_submission());
        g::set_graphics_producer_status_query(live::live_graphics_producer_status);
        g::set_graphics_raw_source_authority(live::live_graphics_raw_source_current);
        g::set_graphics_raw_allocation_authority(live::live_graphics_raw_allocation_current);
        g::set_submit_renderer([&](const std::vector<g::DrawItem>& items, uint32_t, uint32_t) {
            ++render_calls;
            observed.insert(observed.end(), items.begin(), items.end());
            return g::RenderedFrame{}; // no pixel publication or GPU work is claimed
        });
    }
    void TearDown() override {
        g::set_submit_renderer({});
        g::set_submit_compute({});
        g::set_graphics_raw_allocation_authority({});
        g::set_graphics_raw_source_authority({});
        g::set_graphics_producer_status_query({});
        EXPECT_EQ(r::backend_pending_submission_batches().load(), 0);
        EXPECT_TRUE(scene.data.close());
        EXPECT_TRUE(scene.color.close());
    }
    void run(const g::GpuState& state, uint64_t submit = 4801) {
        observed.clear();
        render_calls = 0;
        EXPECT_FALSE(g::execute_ordered_and_present(state, r::fragment_draw::width,
                                                    r::fragment_draw::height, submit, false));
    }
    static std::shared_ptr<const g::FragmentScalarBank> bank(const g::DrawItem& draw) {
        return draw.fragment_draw_inputs ? draw.fragment_draw_inputs->scalar_bank : nullptr;
    }
    static std::array<float, 4> rgba(const g::FragmentScalarBank& value) {
        std::array<float, 4> result{};
        if (value.intervals().size() == 1 && value.intervals()[0].bytes &&
            value.intervals()[0].bytes->size() == sizeof(result))
            std::memcpy(result.data(), value.intervals()[0].bytes->data(), sizeof(result));
        return result;
    }
    static void add_draw(g::GpuState& state, const g::GpuState& actual, uint64_t order) {
        auto draw = state.draws[0];
        draw.command_order = order;
        draw.state = std::make_shared<g::GpuState>(actual);
        state.draws.push_back(std::move(draw));
    }
};

class FragmentScalarBankPlan : public FragmentScalarBank {
protected:
    void SetUp() override {
        // Explicit offline compiler inputs, NOT evidence of actual enabled device features.
        // Publish before realization so the real producing capsule carries the same profile.
        g::reset_float_transport_config_for_test();
        g::publish_float_transport_config({g::FloatTransportProfile::ExplicitNonFinite32});
        g::publish_float_controls_support(true, true);
        FragmentScalarBank::SetUp();
    }
    void TearDown() override {
        FragmentScalarBank::TearDown();
        g::reset_float_transport_config_for_test();
    }
};

TEST_F(FragmentScalarBankPlan, CounterSpecificWaitSeparatesSealedBytesFromPlanReadiness) {
    const g::FragmentPacketDeviceContract source_device{0x1234, true, false};
    for (const uint16_t immediate : {0u, 0xc07fu, 0x3f70u, 0xff7fu, 0x0100u, 0x0001u}) {
        SCOPED_TRACE(immediate);
        auto words = f::fragment_words();
        ASSERT_EQ(words[3], 0xbf8c0000u);
        words[3] |= immediate;
        const auto source = f::register_original(false, words);
        ASSERT_TRUE(source);
        auto state = scene.state();
        for (const auto& reg : source->registers) state.sh[reg.offset] = reg.value;
        run(state);
        ASSERT_EQ(observed.size(), 1u);
        const auto value = bank(observed[0]);
        ASSERT_TRUE(value) << "checked byte ownership is not ISA memory-result readiness";
        EXPECT_EQ(rgba(*value), r::fragment_draw::color_a);
        const auto& input = observed[0].fragment_draw_inputs;
        ASSERT_TRUE(input);
        ASSERT_EQ(*input->raw_code, words);
        const auto prepared = r::fragment_draw::prepare(observed[0]);
        ASSERT_TRUE(prepared);
        const auto plan = g::cached_fragment_draw_program(*input, *prepared, source_device, 48);
        ASSERT_TRUE(plan);
        if (immediate == 0 || immediate == 0xc07f) {
            ASSERT_TRUE(plan->rejection_reason().empty()) << plan->rejection_reason();
            ASSERT_TRUE(plan->requires_scalar_bank());
            const auto transaction = g::instantiate_fragment_draw_transaction(
                plan, input, *prepared, r::fragment_draw::width, r::fragment_draw::height, 1,
                source_device.device_identity);
            ASSERT_TRUE(transaction.rejection().empty()) << transaction.rejection();
            EXPECT_EQ(transaction.scalar_bank(), value)
                << "the actual consumer retains the sealed bytes after permission expiry";
        } else {
            // VM1/LGKM0 also reaches the scalar drain if the execution-domain guard is
            // deleted: the exact packing refusal must precede later compiler guards.
            EXPECT_EQ(plan->rejection_reason(),
                      "fragment-draw-entry-and-composition-recipe-unproved")
                << "VM-only/no-drain/partial threshold cannot complete pending scalar results";
            EXPECT_FALSE(plan->capacity_owner());
        }
    }
}

TEST_F(FragmentScalarBank, CodeFreeHintObservesOnlyPhysicalLaunchAndCannotMintAuthority) {
    auto state = scene.state();
    const auto before = g::shader_decode_cache_stats();
    ASSERT_TRUE(g::draw_requires_original_scalar_bank(state));
    state.cx.erase(p::SPI_PS_IN_CONTROL);
    EXPECT_TRUE(g::draw_requires_original_scalar_bank(state));
    state.cx[p::SPI_PS_IN_CONTROL] = 1u << p::SPI_PS_IN_CONTROL_PS_W32_EN_SHIFT;
    EXPECT_FALSE(g::draw_requires_original_scalar_bank(state));
    state.cx[p::SPI_PS_IN_CONTROL] = 0;
    state.sh[p::SPI_SHADER_PGM_LO_PS] = 0xdeadbe00u;
    state.sh[p::SPI_SHADER_PGM_HI_PS] = 1;
    EXPECT_TRUE(g::draw_requires_original_scalar_bank(state))
        << "missing code is routing, not permission";
    const auto after = g::shader_decode_cache_stats();
    EXPECT_EQ(after.hits, before.hits);
    EXPECT_EQ(after.misses, before.misses);
    EXPECT_EQ(after.bypasses, before.bypasses);
    run(state);
    for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
}
TEST_F(FragmentScalarBank, TwoAdjacentOriginalReadsSealOncePerDrawWithoutAnExtraRenderBoundary) {
    auto state = scene.state();
    add_draw(state, state, 200);
    run(state);
    ASSERT_EQ(observed.size(), 2u);
    EXPECT_EQ(render_calls, 1u) << "a pending unsubmitted disjoint span is not a nested-read flush";
    for (const auto& draw : observed) {
        const auto value = bank(draw);
        ASSERT_TRUE(value);
        EXPECT_EQ(value->payload_bytes(), sizeof(r::fragment_draw::color_a));
        EXPECT_EQ(rgba(*value), r::fragment_draw::color_a);
        ASSERT_EQ(value->sources().size(), 1u);
        EXPECT_EQ(value->sources()[0].descriptor, scene.descriptor);
        EXPECT_EQ(value->sources()[0].allocation.physical_end -
                      value->sources()[0].allocation.physical_begin,
                  f::page)
            << "the certified read is small but output exclusion retains the full allocation";
        EXPECT_NE(value->sources()[0].allocation.identity, 0u);
        ASSERT_TRUE(draw.original_graphics_effects);
        EXPECT_TRUE(draw.original_graphics_effects->matches_draw(
            4801, draw.command_order, draw.vs_shared, draw.fs_shared, draw.gs, draw.fs_words()));
        ASSERT_TRUE(draw.fragment_draw_inputs->original_fragment_producer);
        EXPECT_TRUE(draw.fragment_draw_inputs->original_fragment_producer->matches(
            *draw.fragment_draw_inputs));
        EXPECT_TRUE(draw.fs_words().empty()) << "no fake normalized FS module/reflection";
        EXPECT_FALSE(
            draw.ordered_read_point &&
            draw.ordered_read_point->valid_for(4801, draw.command_order, draw.fs_guest_addr,
                                               draw.ordered_read_point->source(draw.fs_guest_addr)))
            << "legacy RAW remains separate/expired";
    }
}
TEST_F(FragmentScalarBank,
       WarmRealizationReusesOneCanonicalObservationPerStageAndOwnsChangedBytes) {
    const auto state = scene.state();
    run(state);
    ASSERT_EQ(observed.size(), 1u);
    const auto old = bank(observed[0]);
    ASSERT_TRUE(old);
    const auto code = old->original_words();
    scene.write(r::fragment_draw::color_b);
    const auto before = g::shader_decode_cache_stats();
    run(state, 4802);
    const auto after = g::shader_decode_cache_stats();
    ASSERT_EQ(observed.size(), 1u);
    const auto fresh = bank(observed[0]);
    ASSERT_TRUE(fresh);
    EXPECT_EQ(after.hits - before.hits, 2u)
        << "one existing validation each for original VS/PS; no repair relookup";
    EXPECT_EQ(after.misses, before.misses);
    EXPECT_EQ(fresh->original_words(), code);
    EXPECT_NE(fresh, old);
    EXPECT_EQ(rgba(*old), r::fragment_draw::color_a);
    EXPECT_EQ(rgba(*fresh), r::fragment_draw::color_b);
    ASSERT_TRUE(scene.data.close());
    EXPECT_EQ(rgba(*old), r::fragment_draw::color_a)
        << "sealed bytes survive expired point and guest unmap";
    EXPECT_EQ(rgba(*fresh), r::fragment_draw::color_b);
}
TEST_F(FragmentScalarBank, GenuineLargeDeclaredExtentCopiesOnlyTheCertifiedSmallDemand) {
    auto state = scene.state();
    state.sh[p::SPI_SHADER_USER_DATA_PS_0 + 1] |= 16u << 16u;
    state.sh[p::SPI_SHADER_USER_DATA_PS_0 + 2] = UINT32_MAX;
    run(state);
    ASSERT_EQ(observed.size(), 1u);
    const auto value = bank(observed[0]);
    ASSERT_TRUE(value);
    ASSERT_EQ(value->sources().size(), 1u);
    EXPECT_EQ(value->sources()[0].declared_bytes, uint64_t(16) * UINT32_MAX);
    EXPECT_GT(value->sources()[0].declared_bytes, f::page);
    EXPECT_EQ(value->payload_bytes(), 16u);
    EXPECT_EQ(rgba(*value), r::fragment_draw::color_a);
}
TEST_F(FragmentScalarBank, MissingEntryWordAndPartialOobNeverBorrowPaddedBacking) {
    auto state = scene.state();
    state.sh.erase(p::SPI_SHADER_USER_DATA_PS_0 + 3);
    run(state);
    for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
    state = scene.state();
    state.sh[p::SPI_SHADER_USER_DATA_PS_0 + 2] = 12;
    run(state, 4802);
    for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
    state = scene.state();
    state.sh[p::SPI_SHADER_USER_DATA_PS_0 + 1] |= 0x80000000u;
    run(state, 4803);
    for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
    run(scene.state(), 4804);
    ASSERT_EQ(observed.size(), 1u);
    ASSERT_TRUE(bank(observed[0])) << "matched genuine in-bounds positive is not optional";
}
TEST_F(FragmentScalarBank, CurrentAndPriorFullAllocationAliasesRefuseEvenDisjointByteWindows) {
    auto alias = scene.state();
    alias.cx[p::CB_COLOR0_BASE] = uint32_t((scene.data.address + 0x1000) >> 8u);
    alias.cx[p::CB_COLOR0_BASE_EXT] = uint32_t(scene.data.address >> 40u);
    run(alias);
    for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
    const auto native = f::register_original(false, r::fragment_draw::fragment_words());
    ASSERT_TRUE(native);
    auto first = alias;
    for (const auto& reg : native->registers) first.sh[reg.offset] = reg.value;
    first.cx[p::SPI_PS_IN_CONTROL] = 1u << p::SPI_PS_IN_CONTROL_PS_W32_EN_SHIFT;
    add_draw(first, scene.state(), 200);
    run(first, 4802);
    ASSERT_EQ(observed.size(), 2u);
    ASSERT_TRUE(observed[0].original_graphics_effects)
        << "real preceding native draw effects were retained";
    EXPECT_FALSE(bank(observed[1]))
        << "prior unsubmitted output excludes the FULL original allocation";
    run(scene.state(), 4803);
    ASSERT_EQ(observed.size(), 1u);
    ASSERT_TRUE(bank(observed[0]));
}
TEST_F(FragmentScalarBank, MissingOrActiveStagesStreamoutAndMetadataRefuseWithoutInventedDefaults) {
    for (const auto reg :
         {p::VGT_SHADER_STAGES_EN, p::VGT_STRMOUT_CONFIG, p::VGT_STRMOUT_BUFFER_CONFIG,
          p::DB_DEPTH_CONTROL, p::CB_COLOR0_CMASK, p::CB_COLOR0_FMASK, p::CB_COLOR0_DCC_BASE_EXT}) {
        SCOPED_TRACE(reg);
        auto state = scene.state();
        state.cx.erase(reg);
        run(state);
        for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
        state = scene.state();
        state.cx[reg] = 1;
        run(state, 4802);
        for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
    }
    run(scene.state(), 4803);
    ASSERT_EQ(observed.size(), 1u);
    ASSERT_TRUE(bank(observed[0]));
}
TEST_F(FragmentScalarBank, RegisteredPostEndPaddingAndMeaningfulBranchTailAreNotSilentlyTrimmed) {
    for (const auto tail : {std::vector<uint32_t>{0xbf800000u},
                            std::vector<uint32_t>{0xbf820000u, 0xe0700000u, 0x80000000u}}) {
        auto words = f::fragment_words();
        words.insert(words.end(), tail.begin(), tail.end());
        const auto source = f::register_original(false, words);
        ASSERT_TRUE(source);
        auto state = scene.state();
        for (const auto& reg : source->registers) state.sh[reg.offset] = reg.value;
        run(state);
        for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
    }
    run(scene.state(), 4803);
    ASSERT_EQ(observed.size(), 1u);
    ASSERT_TRUE(bank(observed[0]));
}
TEST_F(FragmentScalarBank, MixedNativeAndBankDrawsRetainExactOriginalModuleAndOrderLineage) {
    const auto native = f::register_original(false, r::fragment_draw::fragment_words());
    ASSERT_TRUE(native);
    auto state = scene.state();
    auto native_state = state;
    for (const auto& reg : native->registers) native_state.sh[reg.offset] = reg.value;
    native_state.cx[p::SPI_PS_IN_CONTROL] = 1u << p::SPI_PS_IN_CONTROL_PS_W32_EN_SHIFT;
    add_draw(state, native_state, 200);
    add_draw(state, scene.state(), 300);
    run(state);
    ASSERT_EQ(observed.size(), 3u);
    EXPECT_EQ(render_calls, 1u);
    EXPECT_TRUE(bank(observed[0]));
    EXPECT_FALSE(bank(observed[1]));
    EXPECT_FALSE(observed[1].fs_words().empty());
    EXPECT_TRUE(bank(observed[2]));
    for (const auto& draw : observed) {
        const auto& effects = draw.original_graphics_effects;
        ASSERT_TRUE(effects);
        EXPECT_TRUE(effects->matches_draw(4801, draw.command_order, draw.vs_shared, draw.fs_shared,
                                          draw.gs, draw.fs_words()));
        EXPECT_FALSE(effects->matches_draw(4802, draw.command_order, draw.vs_shared, draw.fs_shared,
                                           draw.gs, draw.fs_words()));
        EXPECT_FALSE(effects->matches_draw(4801, draw.command_order + 1, draw.vs_shared,
                                           draw.fs_shared, draw.gs, draw.fs_words()));
        const auto equal_but_foreign =
            std::make_shared<const std::vector<uint32_t>>(draw.vs_words());
        EXPECT_FALSE(effects->matches_draw(4801, draw.command_order, equal_but_foreign,
                                           draw.fs_shared, draw.gs, draw.fs_words()));
    }
}
TEST_F(FragmentScalarBank, ActualFailureGenerationDuringFinalPublicationRevokesTheCopy) {
    uint32_t source_checks = 0;
    const auto before = r::backend_failed_publication_generation().load();
    g::set_graphics_raw_allocation_authority(
        [&](const prosper::GuestMappingLease& lease, const prosper::GuestDirectAllocation& source) {
            const bool current = live::live_graphics_raw_allocation_current(lease, source);
            if (++source_checks == 2) { const r::BackendProducerAttempt actual_failed_publication; }
            return current;
        });
    run(scene.state());
    ASSERT_EQ(source_checks, 2u)
        << "final source-current validation is actually reached after copying";
    EXPECT_EQ(r::backend_failed_publication_generation().load(), before + 1);
    for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
    g::set_graphics_raw_allocation_authority(live::live_graphics_raw_allocation_current);
    run(scene.state(), 4802);
    ASSERT_EQ(observed.size(), 1u);
    ASSERT_TRUE(bank(observed[0]))
        << "a new clean baseline is distinct from reviving an old ticket";
}
TEST_F(FragmentScalarBank, SuccessfulPublicProducerEntryCannotReuseAnEarlierCopyPermission) {
    uint32_t source_checks = 0;
    const auto before = r::backend_failed_publication_generation().load();
    g::set_graphics_raw_allocation_authority(
        [&](const prosper::GuestMappingLease& lease, const prosper::GuestDirectAllocation& source) {
            const bool current = live::live_graphics_raw_allocation_current(lease, source);
            if (++source_checks == 2) {
                // A real publicly callable successful executor entry participates even when it
                // has no draws. It changes shared order/version, not the failure generation.
                EXPECT_TRUE(g::render_submit_items({}, 0, 0).empty());
            }
            return current;
        });
    run(scene.state());
    ASSERT_EQ(source_checks, 2u);
    EXPECT_EQ(r::backend_failed_publication_generation().load(), before);
    for (const auto& draw : observed) EXPECT_FALSE(bank(draw));
    g::set_graphics_raw_allocation_authority(live::live_graphics_raw_allocation_current);
    run(scene.state(), 4802);
    ASSERT_EQ(observed.size(), 1u);
    ASSERT_TRUE(bank(observed[0]));
}
TEST(FragmentScalarBankVisibility, ExactRecordedBankRangeAccessAndStagesAreRequired) {
    // Pure inspection of the exact Vulkan-call argument shape, not a device/visibility claim.
    // Actual frontend GPU cases additionally require the recording observer to increment.
    const VkDescriptorBufferInfo bank{reinterpret_cast<VkBuffer>(uintptr_t(1)), 0, 104};
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.buffer = bank.buffer;
    barrier.offset = bank.offset;
    barrier.size = bank.range;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    const auto source = VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    const auto destination = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    const auto matches = [&](const auto& value,
                             VkPipelineStageFlags src =
                                 VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VkPipelineStageFlags dst = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        return r::fragment_scalar_bank_visibility_matches(src, dst, std::span(&value, 1), bank);
    };
    ASSERT_TRUE(matches(barrier));
    EXPECT_FALSE(matches(barrier, VK_PIPELINE_STAGE_TRANSFER_BIT));
    EXPECT_FALSE(matches(barrier, source, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT));
    for (uint32_t change = 0; change < 7; ++change) {
        auto wrong = barrier;
        switch (change) {
            case 0: wrong.buffer = reinterpret_cast<VkBuffer>(uintptr_t(2)); break;
            case 1: wrong.offset = 4; break;
            case 2: wrong.size -= 4; break;
            case 3: wrong.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; break;
            case 4: wrong.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; break;
            case 5: wrong.srcQueueFamilyIndex = 0; break;
            case 6: wrong.dstQueueFamilyIndex = 0; break;
        }
        EXPECT_FALSE(matches(wrong)) << "wrong actual recorded property=" << change;
    }
    EXPECT_FALSE(r::fragment_scalar_bank_visibility_matches(source, destination, {}, bank));
    EXPECT_FALSE(r::fragment_scalar_bank_visibility_matches(source, destination,
                                                            std::span(&barrier, 1), {}));
}
}   // namespace
