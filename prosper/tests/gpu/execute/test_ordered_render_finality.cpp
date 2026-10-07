// Internal snapshot flushes are producer boundaries, not submit ends. A premature final callback
// can publish an earlier draw and suppress the terminal scanout/timing callback for later work.
#include "diagnostics/env_submit.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/present/videoout_present.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "shared/present/single_framebuffer_submit_policy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

struct Program {
    alignas(256) std::array<uint32_t, 32> code{};
    AgcShaderHeader header{};
    std::array<ShaderReg, 2> registers{};
};

class OrderedRenderFinality : public testing::Test {
protected:
    static void SetUpTestSuite() {
        prosper::register_builtin_hle();
        const std::vector<uint32_t> vs{
            0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
            0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
            0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
        };
        const std::vector<uint32_t> ps{
            0xf4080a00u, 0xfa000004u, // parent s[40:43] from entry s[0:1]+4
            0xf4080b14u, 0xfa000010u, // child s[44:47] from s[40:41]+16
            0x7e00022fu,             // numeric consumer of the final child word s47
            0x7e0202f2u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u,
        };
        // The same export with no scalar loads: v0 = 1.0 instead of the nested child word, so a draw
        // using it has no owned nested input and can take the graphics-only (unordered) path.
        const std::vector<uint32_t> flat_ps{
            0x7e0002f2u, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u,
            0xf800180fu, 0x03020100u, 0xbf810000u,
        };
        const std::vector<uint32_t> cs{0x7e000280u, 0xbf810000u};
        const auto create = prosper::Hle::lookup("f3dg2CSgRKY");
        ASSERT_TRUE(create);
        const auto register_program = [&](Program& program, const auto& code, uint32_t type,
                                          uint32_t lo, uint32_t hi) {
            std::copy(code.begin(), code.end(), program.code.begin());
            program.registers = {{{lo, 0}, {hi, 0}}};
            program.header.file_header = 0x34333231u;
            program.header.version = 0x18u;
            program.header.type = type;
            program.header.shader_size = static_cast<uint32_t>(code.size() * sizeof(uint32_t));
            // Static test storage can live below 4 GiB, where CreateShader expects a self-relative
            // SDK pointer. Put registers after the header and use that actual constructor format.
            program.header.sh_registers = reinterpret_cast<const void*>(
                reinterpret_cast<uintptr_t>(program.registers.data()) -
                reinterpret_cast<uintptr_t>(&program.header.sh_registers));
            program.header.num_sh_registers = 2;
            void* registered = nullptr;
            EXPECT_EQ(create(reinterpret_cast<uint64_t>(&registered),
                             reinterpret_cast<uint64_t>(&program.header),
                             reinterpret_cast<uint64_t>(program.code.data()), 0, 0, 0),
                      0u);
            EXPECT_EQ(registered, &program.header);
            EXPECT_EQ(program.header.sh_registers, program.registers.data());
            const uint64_t address = reinterpret_cast<uint64_t>(program.code.data());
            EXPECT_EQ(program.registers[0].value, static_cast<uint32_t>(address >> 8u));
            EXPECT_EQ(program.registers[1].value, static_cast<uint32_t>((address >> 40u) & 0xffu));
        };
        register_program(vertex_, vs, 2, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES);
        register_program(fragment_, ps, 1, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS);
        register_program(compute_, cs, 0, P::COMPUTE_PGM_LO, P::COMPUTE_PGM_HI);
        register_program(flat_fragment_, flat_ps, 1, P::SPI_SHADER_PGM_LO_PS,
                         P::SPI_SHADER_PGM_HI_PS);
    }

    void SetUp() override {
        active_ = this;
        const auto allocate =
            prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
        ASSERT_TRUE(allocate && map);
        ASSERT_EQ(
            allocate(0, 0x200000000ull, page_, page_, 0, reinterpret_cast<uint64_t>(&physical_)),
            0u);
        allocated_ = true;
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&guest_), page_, 3, 0, physical_, page_), 0u);
        ASSERT_NE(guest_, 0u);
        const uint64_t child = guest_ + 0x1000u;
        std::memcpy(reinterpret_cast<void*>(guest_ + 4u), &child, sizeof(child));
        *child_word() =
            0x3e800000u;   // 0.25, copied into an immutable snapshot before each callback
        set_graphics_producer_status_query(
            [&] { return GraphicsProducerStatus{true, pending_, 0}; });
        set_graphics_raw_source_authority(
            [](const prosper::GuestMappingLease& lease, uint64_t address, uint32_t bytes) {
                return prosper::guest_memory_direct_allocation_relation(lease, address, bytes,
                                                                        address, bytes) ==
                       prosper::GuestMemoryTopologyRelation::Overlap;
            });
        set_deferred_graphics_retirer(+[] {
            active_->pending_ = false;
            active_->events_.push_back('T');
        });
        set_graphics_deferred_wait_for_test(0);
        present_reset();
        set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t w, uint32_t h) {
            Observation observed{live_render_phase(), {}, {}};
            single_frame_.begin_span(observed.phase.first_span, observed.phase.source_submit, w, h);
            for (const auto& item : items) {
                observed.draws.push_back(item.draw_index);
                const auto* child =
                    item.prt ? owned_nested_snapshot_at(*item.prt, 2u, 16u) : nullptr;
                EXPECT_NE(child, nullptr) << "the real registered child load must be admitted";
                if (child && child->host_data && child->host_data_size == 16u) {
                    uint32_t word = 0;
                    std::memcpy(&word, child->host_data + 12u, sizeof(word));
                    observed.words.push_back(word);
                }
            }
            if (observed.phase.final_span) EXPECT_FALSE(pending_);
            pending_ = observed.phase.defer_batch_completion;
            events_.push_back(observed.phase.final_span ? 'F' : 'R');
            observations_.push_back(std::move(observed));
            if (mutate_after_first_ && observations_.size() == 1u) *child_word() = 0x3f400000u;
            prosper::frontend::PixelSourceCandidate current;
            if (!items.empty()) {
                // Only an actual nonempty producer supplies pixels. The empty terminal cannot
                // fabricate its own frame; it must recover this exact immutable owner and source.
                const auto value = static_cast<uint8_t>(items.back().draw_index + 1u);
                current = {std::make_shared<const std::vector<uint8_t>>(
                               std::initializer_list<uint8_t>{value, 2, 3, 255}),
                           live_render_phase().source_submit};
                produced_pixels_.push_back(current.pixels);
            }
            auto selected = single_frame_.select(live_render_phase().final_span, !items.empty(),
                                                 std::move(current));
            if (selected.pixels) {
                selected_pixels_ = selected.pixels;
                selected_source_ = selected.source_submit;
            }
            RenderedFrame frame(std::move(selected.pixels));
            frame.source_submit = selected.source_submit;
            return frame;
        });
        set_submit_compute([&](const std::vector<ComputeItem>& items) {
            EXPECT_EQ(items.size(), 1u);
            ++compute_calls_;
            events_.push_back('C');
            return compute_succeeds_;
        });
    }

    void TearDown() override {
        set_submit_renderer({});
        set_submit_compute({});
        set_graphics_producer_status_query({});
        set_graphics_raw_source_authority({});
        set_deferred_graphics_retirer(nullptr);
        set_graphics_deferred_wait_for_test(-1);
        present_reset();
        if (guest_) {
            const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
            EXPECT_EQ(unmap(guest_, page_, 0, 0, 0, 0), 0u);
        }
        if (allocated_) {
            const auto release =
                prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
            EXPECT_EQ(release(physical_, page_, 0, 0, 0, 0), 0u);
        }
        active_ = nullptr;
    }

    GpuState two_draws() const {
        GpuState state;
        for (const Program* program : {&vertex_, &fragment_, &compute_})
            for (const auto& reg : program->registers) state.sh[reg.offset] = reg.value;
        state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
        state.cx[P::CB_TARGET_MASK] = state.cx[P::CB_SHADER_MASK] = 15;
        state.sh[P::SPI_SHADER_USER_DATA_PS_0] = static_cast<uint32_t>(guest_);
        state.sh[P::SPI_SHADER_USER_DATA_PS_0 + 1u] = static_cast<uint32_t>(guest_ >> 32u);
        state.sh[P::SPI_SHADER_PGM_RSRC2_PS] = 2u << P::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_SHIFT;
        for (uint64_t order : {100u, 200u}) {
            GpuState::Draw draw;
            draw.index_count = 3;
            draw.instance_count = 1;
            draw.command_order = order;
            state.draws.push_back(draw);
        }
        return state;
    }

    void execute(const GpuState& state) {
        EXPECT_TRUE(draw_requires_owned_nested_snapshot(state));
        EXPECT_TRUE(execute_ordered_and_present(state, 1, 1, submit_, true));
        ASSERT_FALSE(observations_.empty());
        EXPECT_TRUE(observations_.front().phase.first_span);
        for (size_t i = 0; i < observations_.size(); ++i) {
            EXPECT_EQ(observations_[i].phase.final_span, i + 1u == observations_.size());
            EXPECT_EQ(observations_[i].phase.first_span, i == 0u);
            EXPECT_EQ(observations_[i].phase.source_submit, submit_);
        }
        EXPECT_FALSE(pending_);
        EXPECT_FALSE(live_render_phase().defer_batch_completion);
        ASSERT_FALSE(produced_pixels_.empty());
        EXPECT_EQ(selected_pixels_, produced_pixels_.back());
        EXPECT_EQ(selected_source_, submit_);
        EXPECT_EQ(present_frame_seq(), 1u) << "only the actual final callback publishes";
        PresentFrameLease published;
        ASSERT_TRUE(present_acquire_rendered_frame(published));
        EXPECT_EQ(published.rgba, produced_pixels_.back());
        EXPECT_EQ(*published.rgba, (std::vector<uint8_t>{2, 2, 3, 255}));
    }

    uint32_t* child_word() const { return reinterpret_cast<uint32_t*>(guest_ + 0x1000u + 28u); }
    struct Observation {
        LiveRenderPhase phase;
        std::vector<uint32_t> draws;
        std::vector<uint32_t> words;
    };
    static inline Program vertex_, fragment_, compute_, flat_fragment_;
    static inline OrderedRenderFinality* active_ = nullptr;
    static constexpr uint64_t page_ = 0x10000u;
    static constexpr uint64_t submit_ = 4280u;
    uint64_t guest_ = 0, physical_ = 0;
    bool allocated_ = false, pending_ = false, mutate_after_first_ = false;
    bool compute_succeeds_ = true;
    size_t compute_calls_ = 0;
    std::vector<Observation> observations_;
    std::vector<char> events_;
    prosper::frontend::SingleFramebufferSubmitFrame single_frame_;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> produced_pixels_;
    std::shared_ptr<const std::vector<uint8_t>> selected_pixels_;
    uint64_t selected_source_ = 0;
};

TEST_F(OrderedRenderFinality, AdjacentSnapshotDrawsFinalizeTheSecondDraw) {
    mutate_after_first_ = true;
    execute(two_draws());
    ASSERT_EQ(observations_.size(), 2u);
    EXPECT_EQ(observations_[0].draws, std::vector<uint32_t>{0});
    EXPECT_EQ(observations_[1].draws, std::vector<uint32_t>{1});
    EXPECT_EQ(observations_[0].words, std::vector<uint32_t>{0x3e800000u});
    EXPECT_EQ(observations_[1].words, std::vector<uint32_t>{0x3f400000u});
    EXPECT_TRUE(observations_[0].phase.authoritative_readback);
    EXPECT_FALSE(observations_[1].phase.authoritative_readback);
}

TEST_F(OrderedRenderFinality, TrailingRefusedDrawUsesOneEmptyTerminalCallback) {
    auto state = two_draws();
    auto refused = state.draws.back();
    refused.command_order = 300;
    auto missing_parent = std::make_shared<GpuState>(state);
    missing_parent->sh[P::SPI_SHADER_USER_DATA_PS_0] = 0;
    missing_parent->sh[P::SPI_SHADER_USER_DATA_PS_0 + 1u] = 0;
    refused.state = missing_parent;   // actual backing refusal, with a nonzero draw count
    state.draws.push_back(refused);
    execute(state);
    ASSERT_EQ(observations_.size(), 3u);
    EXPECT_EQ(observations_[0].draws, std::vector<uint32_t>{0});
    EXPECT_EQ(observations_[1].draws, std::vector<uint32_t>{1});
    EXPECT_TRUE(observations_[2].draws.empty());
    EXPECT_TRUE(observations_[1].phase.authoritative_readback);
    EXPECT_FALSE(observations_[2].phase.authoritative_readback);
}

TEST_F(OrderedRenderFinality, TrailingNoopDispatchDoesNotInventAnotherSpan) {
    auto state = two_draws();
    GpuState::Dispatch noop;
    noop.command_order = 300;
    state.dispatches.push_back(noop);   // zero group extent is a proven no-op
    execute(state);
    ASSERT_EQ(observations_.size(), 2u);
    EXPECT_EQ(observations_.back().draws, std::vector<uint32_t>{1});
    EXPECT_EQ(compute_calls_, 0u);
}

TEST_F(OrderedRenderFinality, DispatchTailRetiresBeforeEmptyFinalPublication) {
    auto state = two_draws();
    GpuState::Dispatch dispatch;
    dispatch.threads_x = dispatch.threads_y = dispatch.threads_z = 1;
    dispatch.command_order = 300;
    state.dispatches.push_back(dispatch);
    set_graphics_deferred_wait_for_test(1);
    execute(state);
    ASSERT_EQ(observations_.size(), 3u);
    EXPECT_EQ(compute_calls_, 1u);
    EXPECT_TRUE(observations_[1].phase.defer_batch_completion);
    EXPECT_FALSE(observations_[1].phase.authoritative_readback);
    EXPECT_TRUE(observations_[2].draws.empty());
    EXPECT_FALSE(observations_[2].phase.defer_batch_completion);
    const auto compute = std::find(events_.begin(), events_.end(), 'C');
    ASSERT_NE(compute, events_.end());
    const auto retired = std::find(compute + 1, events_.end(), 'T');
    ASSERT_NE(retired, events_.end());
    const auto final = std::find(events_.begin(), events_.end(), 'F');
    ASSERT_NE(final, events_.end());
    EXPECT_LT(retired, final);
}

TEST_F(OrderedRenderFinality, RefusedDispatchTailStillFinalizesEarlierDraws) {
    auto state = two_draws();
    state.sh.erase(P::COMPUTE_PGM_LO);
    state.sh.erase(P::COMPUTE_PGM_HI);
    GpuState::Dispatch dispatch;
    dispatch.threads_x = dispatch.threads_y = dispatch.threads_z = 1;
    dispatch.command_order = 300;
    state.dispatches.push_back(dispatch);
    execute(state);
    ASSERT_EQ(observations_.size(), 3u);
    EXPECT_TRUE(observations_.back().draws.empty());
    EXPECT_EQ(compute_calls_, 0u);
}

TEST_F(OrderedRenderFinality, DeclinedDeferredDispatchStillRetiresBeforeFinalPublication) {
    auto state = two_draws();
    GpuState::Dispatch dispatch;
    dispatch.threads_x = dispatch.threads_y = dispatch.threads_z = 1;
    dispatch.command_order = 300;
    state.dispatches.push_back(dispatch);
    compute_succeeds_ = false;
    set_graphics_deferred_wait_for_test(1);
    execute(state);
    ASSERT_EQ(observations_.size(), 3u);
    EXPECT_TRUE(observations_[1].phase.defer_batch_completion);
    EXPECT_TRUE(observations_[2].draws.empty());
    EXPECT_EQ(compute_calls_, 1u);
    const auto compute = std::find(events_.begin(), events_.end(), 'C');
    ASSERT_NE(compute, events_.end());
    const auto retired = std::find(compute + 1, events_.end(), 'T');
    const auto final = std::find(events_.begin(), events_.end(), 'F');
    ASSERT_NE(retired, events_.end());
    ASSERT_NE(final, events_.end());
    EXPECT_LT(retired, final);
}

TEST_F(OrderedRenderFinality, DmaTailRetainsAuthoritativeReadbackBeforeFinalizing) {
    auto state = two_draws();
    // #4439: the tail copy lands inside the span's 16x8 RGBA8 colour target, so the span before
    // it must still be published authoritatively. The target gets its own direct allocation: the
    // owned snapshot refuses an output that shares the allocation of its scalar inputs.
    struct TargetPage {
        uint64_t physical = 0, address = 0;
        ~TargetPage() {
            if (address)
                prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"))(address, page_, 0, 0, 0, 0);
            if (physical)
                prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"))(
                    physical, page_, 0, 0, 0, 0);
        }
    } page;
    ASSERT_EQ(prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"))(
                  0, 0x200000000ull, page_, page_, 0, reinterpret_cast<uint64_t>(&page.physical)),
              0u);
    ASSERT_EQ(prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"))(
                  reinterpret_cast<uint64_t>(&page.address), page_, 3, 0, page.physical, page_),
              0u);
    const uint64_t rt = page.address;
    state.cx[P::CB_COLOR0_BASE] = static_cast<uint32_t>(rt >> 8u);
    state.cx[P::CB_COLOR0_BASE_EXT] = static_cast<uint32_t>(rt >> 40u);
    state.cx[P::CB_COLOR0_INFO] = 0xau << P::CB_COLOR0_INFO_FORMAT_SHIFT;
    state.cx[P::CB_COLOR0_ATTRIB2] = (15u << P::CB_COLOR0_ATTRIB2_MIP0_WIDTH_SHIFT) | 7u;
    state.cx[P::CB_COLOR0_ATTRIB3] = 1u << P::CB_COLOR0_ATTRIB3_RESOURCE_TYPE_SHIFT;
    uint32_t source = 0x4280u;
    auto* destination = reinterpret_cast<uint32_t*>(rt + 16u);
    *destination = 0;
    state.dma_copies.push_back({rt + 16u, reinterpret_cast<uint64_t>(&source), sizeof(source),
                                0, 300, 0});
    execute(state);
    ASSERT_EQ(observations_.size(), 3u);
    EXPECT_TRUE(observations_[1].phase.authoritative_readback);
    EXPECT_TRUE(observations_[2].draws.empty());
    EXPECT_FALSE(observations_[2].phase.authoritative_readback);
    EXPECT_EQ(*destination, source);
}

// #4439: the span renders no colour target, so a tail copy can neither observe nor overwrite
// rendered bytes: the span flushes without the authoritative readback and the copy still lands.
TEST_F(OrderedRenderFinality, DmaTailTouchingNoTargetSkipsAuthoritativeReadback) {
    auto state = two_draws();
    uint32_t source = 0x4439u, destination = 0;
    state.dma_copies.push_back({reinterpret_cast<uint64_t>(&destination),
                                reinterpret_cast<uint64_t>(&source), sizeof(source), 0, 300, 0});
    execute(state);
    ASSERT_EQ(observations_.size(), 3u);
    EXPECT_FALSE(observations_[1].phase.authoritative_readback);
    EXPECT_TRUE(observations_[2].draws.empty());
    EXPECT_EQ(destination, source);
}

TEST_F(OrderedRenderFinality, NoSuccessfulDrawDoesNotPublishAnEmptyFinalCallback) {
    auto state = two_draws();
    for (auto& draw : state.draws) draw.index_count = 0;
    EXPECT_FALSE(execute_ordered_and_present(state, 1, 1, submit_, true));
    EXPECT_TRUE(observations_.empty());
    EXPECT_FALSE(present_has_frame());
}

// #4677: execute_ordered_and_present opens ONE per-submit env window around both branches. A
// graphics-only submit (no dispatch, DMA, indirect, nested or scalar-bank input) renders through
// execute_ordered_guest_items, outside the window realize_gpustate_draws opens and closes, so
// without the outer scope every PROSPER_ENV_ON_PER_SUBMIT site in the backend read getenv live per
// pass. Mutation: delete that scope and the window observed inside the render callback is 0.
TEST_F(OrderedRenderFinality, GraphicsOnlySubmitRendersInsideOneEnvWindow) {
    GpuState state;
    for (const Program* program : {&vertex_, &flat_fragment_})
        for (const auto& reg : program->registers) state.sh[reg.offset] = reg.value;
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[P::CB_TARGET_MASK] = state.cx[P::CB_SHADER_MASK] = 15;
    // Observed Wave32 launch metadata: without it the draw is routed ordered for the scalar bank.
    state.cx[P::SPI_PS_IN_CONTROL] = 1u << P::SPI_PS_IN_CONTROL_PS_W32_EN_SHIFT;
    GpuState::Draw draw;
    draw.index_count = 3;
    draw.instance_count = 1;
    draw.command_order = 100;
    state.draws.push_back(draw);
    // Preconditions for the graphics-only branch (needs_ordered_realization false).
    ASSERT_FALSE(draw_requires_owned_nested_snapshot(state));
    ASSERT_FALSE(draw_requires_original_scalar_bank(state));
    ASSERT_EQ(prosper::diag::submit_env_window(), 0u) << "no submit may be open before the call";

    std::vector<uint64_t> windows;
    size_t rendered_items = 0;
    set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
        windows.push_back(prosper::diag::submit_env_window());
        rendered_items += items.size();
        return RenderedFrame{};
    });
    (void)execute_ordered_and_present(state, 1, 1, submit_, false);
    ASSERT_FALSE(windows.empty()) << "the graphics-only submit never reached the renderer";
    EXPECT_GT(rendered_items, 0u) << "the draw must actually be handed to the renderer";
    for (const uint64_t window : windows)
        EXPECT_NE(window, 0u) << "the backend rendered with no per-submit env window open";
    EXPECT_EQ(prosper::diag::submit_env_window(), 0u) << "the window must close with the submit";
}

TEST(SingleFramebufferSubmitFrame, FinalConsumesExactProducerOnce) {
    prosper::frontend::SingleFramebufferSubmitFrame frame;
    const auto pixels = std::make_shared<const std::vector<uint8_t>>(8u, 42u);
    frame.begin_span(true, 4280u, 2u, 1u);
    EXPECT_FALSE(frame.select(false, true, {pixels, 4280u}).pixels);
    frame.begin_span(false, 4280u, 2u, 1u);
    const auto selected = frame.select(true, false, {});
    EXPECT_EQ(selected.pixels, pixels);
    EXPECT_EQ(selected.source_submit, 4280u);
    EXPECT_FALSE(frame.select(true, false, {}).pixels);
}

TEST(SingleFramebufferSubmitFrame, NewFirstDropsAbandonedPriorSubmit) {
    prosper::frontend::SingleFramebufferSubmitFrame frame;
    const auto pixels = std::make_shared<const std::vector<uint8_t>>(4u, 42u);
    frame.begin_span(true, 4280u, 1u, 1u);
    frame.select(false, true, {pixels, 4280u});
    frame.begin_span(true, 4281u, 1u, 1u);   // also reached before render-window early returns
    EXPECT_FALSE(frame.select(true, false, {}).pixels);
}

TEST(SingleFramebufferSubmitFrame, BufferedFinalOwnPixelsReplaceEarlierSpan) {
    prosper::frontend::SingleFramebufferSubmitFrame frame;
    const auto earlier = std::make_shared<const std::vector<uint8_t>>(4u, 42u);
    const auto final = std::make_shared<const std::vector<uint8_t>>(4u, 43u);
    frame.begin_span(true, 4280u, 1u, 1u);
    frame.select(false, true, {earlier, 4280u});
    frame.begin_span(false, 4280u, 1u, 1u);
    EXPECT_EQ(frame.select(true, true, {final, 4280u}).pixels, final);
}

TEST(SingleFramebufferSubmitFrame, RefusesMismatchedExtentSourceAndEmptyProducer) {
    prosper::frontend::SingleFramebufferSubmitFrame frame;
    const auto pixels = std::make_shared<const std::vector<uint8_t>>(8u, 42u);
    frame.begin_span(true, 4280u, 2u, 1u);
    frame.select(false, true, {pixels, 4280u});
    frame.begin_span(false, 4280u, 1u, 2u);   // equal byte count is not the same image extent
    EXPECT_FALSE(frame.select(true, false, {}).pixels);
    frame.begin_span(true, 4280u, 2u, 1u);
    EXPECT_FALSE(frame.select(true, true, {pixels, 4281u}).pixels);
    frame.begin_span(true, 4280u, 1u, 1u);
    EXPECT_FALSE(frame.select(true, true, {pixels, 4280u}).pixels);
    frame.begin_span(true, 4280u, 2u, 1u);
    EXPECT_FALSE(frame.select(true, false, {pixels, 4280u}).pixels);
}
}   // namespace
