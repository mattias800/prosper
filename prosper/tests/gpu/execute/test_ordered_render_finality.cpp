// Internal snapshot flushes are producer boundaries, not submit ends. A premature final callback
// can publish an earlier draw and suppress the terminal scanout/timing callback for later work.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/present/videoout_present.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

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
        set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
            Observation observed{live_render_phase(), {}, {}};
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
            return live_render_phase().final_span
                       ? RenderedFrame(std::vector<uint8_t>{1, 2, 3, 255})
                       : RenderedFrame{};
        });
        set_submit_compute([&](const std::vector<ComputeItem>& items) {
            EXPECT_EQ(items.size(), 1u);
            ++compute_calls_;
            events_.push_back('C');
            return true;
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
    }

    uint32_t* child_word() const { return reinterpret_cast<uint32_t*>(guest_ + 0x1000u + 28u); }
    struct Observation {
        LiveRenderPhase phase;
        std::vector<uint32_t> draws;
        std::vector<uint32_t> words;
    };
    static inline Program vertex_, fragment_, compute_;
    static inline OrderedRenderFinality* active_ = nullptr;
    static constexpr uint64_t page_ = 0x10000u;
    static constexpr uint64_t submit_ = 4280u;
    uint64_t guest_ = 0, physical_ = 0;
    bool allocated_ = false, pending_ = false, mutate_after_first_ = false;
    size_t compute_calls_ = 0;
    std::vector<Observation> observations_;
    std::vector<char> events_;
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

TEST_F(OrderedRenderFinality, DmaTailRetainsAuthoritativeReadbackBeforeFinalizing) {
    auto state = two_draws();
    uint32_t source = 0x4280u, destination = 0;
    state.dma_copies.push_back({reinterpret_cast<uint64_t>(&destination),
                                reinterpret_cast<uint64_t>(&source), sizeof(source), 0, 300, 0});
    execute(state);
    ASSERT_EQ(observations_.size(), 3u);
    EXPECT_TRUE(observations_[1].phase.authoritative_readback);
    EXPECT_TRUE(observations_[2].draws.empty());
    EXPECT_FALSE(observations_[2].phase.authoritative_readback);
    EXPECT_EQ(destination, source);
}

TEST_F(OrderedRenderFinality, NoSuccessfulDrawDoesNotPublishAnEmptyFinalCallback) {
    auto state = two_draws();
    for (auto& draw : state.draws) draw.index_count = 0;
    EXPECT_FALSE(execute_ordered_and_present(state, 1, 1, submit_, true));
    EXPECT_TRUE(observations_.empty());
    EXPECT_FALSE(present_has_frame());
}
}   // namespace
