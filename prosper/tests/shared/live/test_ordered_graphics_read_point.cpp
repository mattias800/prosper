// An idle backend is not proof that prior ordered producers succeeded. Exercise the real ordered
// executor and checked exact-PC RAW consumer with registered sources and HLE allocation topology.
// No renderer registration, device creation, GPU submission or caller-supplied complete ticket.
// Entry-version tests detect stale permissions only. They never overlap a CPU writer with memcpy
// or claim the activity counter provides byte coherence; submitted input stability remains required.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "shared/live/live_renderer.hpp"
#include "fixtures/render_runner.h"
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace {
using namespace prosper;
using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
namespace Backend = prosper::test;
namespace Live = prosper::frontend;
constexpr uint64_t Page = 0x10000, Submit = 901, Order = 20;
constexpr std::array<uint32_t, 6> Nested{
    0xf4080a00u, 0xfa000000u,   // pc0 parent x4 from genuine entry pointer
    0xf4080b14u, 0xfa000010u,   // pc2 child x4 at parent pointer +16
    0x7e00022eu, 0xbf810000u};   // numeric observer, no writer
constexpr std::array<uint32_t, 4> Values{0x12345678u, 0x89abcdefu, 0x10203040u, 0xfedcba98u};
constexpr std::array<uint32_t, 15> ProceduralVertex{
    0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
    0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
    0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u};
constexpr std::array<uint32_t, 11> NestedExport{0xf4080a00u, 0xfa000000u, 0xf4080b14u, 0xfa000010u,
                                                0x7e00022fu, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u,
                                                0xf800180fu, 0x03020100u, 0xbf810000u};
constexpr std::array<uint32_t, 7> ResourceFreeExport{
    0x7e000280u, 0x7e0202f2u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u};

struct Program {
    alignas(256) std::array<uint32_t, 64> code{};
    AgcShaderUserData user{};
    std::array<ShaderReg, 2> registers{};
    AgcShaderHeader header{};
};

class OrderedGraphicsReadPointTest : public ::testing::Test {
protected:
    std::vector<Backend::PersistentColorTargetKey> retained_keys;
    std::vector<uint64_t> mappings, allocations;
    uint64_t parent = 0, child = 0, output = 0, parent_physical = 0;
    Program* vertex = nullptr;
    Program* fragment = nullptr;
    bool fail_during_source_check = false;
    uint64_t generation_before = 0;

    void map_allocation(uint64_t& address, uint64_t* physical_out = nullptr) {
        const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
        const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
        ASSERT_TRUE(allocate && map);
        uint64_t physical = 0;
        ASSERT_EQ(allocate(0, 0x200000000ull, Page, Page, 0, reinterpret_cast<uint64_t>(&physical)),
                  0u);
        ASSERT_NE(physical, 0u);
        allocations.push_back(physical);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&address), Page, 3, 0, physical, Page), 0u);
        ASSERT_NE(address, 0u);
        mappings.push_back(address);
        if (physical_out) *physical_out = physical;
    }

    void register_program(Program*& result, bool pixel, std::span<const uint32_t> supplied = {}) {
        // AGC retains raw header/code pointers for process lifetime. Never leave stale fixture
        // pointers or reuse a low address interpreted as a blob-relative pointer.
        static std::vector<std::unique_ptr<Program>> owners;
        owners.push_back(std::make_unique<Program>());
        result = owners.back().get();
        ASSERT_GT(reinterpret_cast<uint64_t>(result), UINT32_MAX);
        if (!supplied.empty()) {
            ASSERT_LE(supplied.size(), result->code.size());
            std::copy(supplied.begin(), supplied.end(), result->code.begin());
        } else if (pixel)
            std::copy(Nested.begin(), Nested.end(), result->code.begin());
        else
            result->code[0] = 0xbf810000u;
        result->registers[0].offset = pixel ? P::SPI_SHADER_PGM_LO_PS : P::SPI_SHADER_PGM_LO_ES;
        result->registers[1].offset = pixel ? P::SPI_SHADER_PGM_HI_PS : P::SPI_SHADER_PGM_HI_ES;
        result->header.file_header = 0x34333231u;
        result->header.version = 0x18u;
        result->header.user_data = &result->user;
        result->header.sh_registers = result->registers.data();
        result->header.shader_size =
            (supplied.empty() ? (pixel ? Nested.size() : 1u) : supplied.size()) * sizeof(uint32_t);
        result->header.type = pixel ? 1u : 2u;
        result->header.num_sh_registers = 2u;
        const auto create = Hle::lookup("f3dg2CSgRKY");
        ASSERT_TRUE(create);
        void* registered = nullptr;
        ASSERT_EQ(create(reinterpret_cast<uint64_t>(&registered),
                         reinterpret_cast<uint64_t>(&result->header),
                         reinterpret_cast<uint64_t>(result->code.data()), 0, 0, 0),
                  0u);
        ASSERT_EQ(registered, &result->header);
        EXPECT_EQ(result->header.sh_registers, result->registers.data());
        EXPECT_EQ(result->registers[0].value,
                  uint32_t(reinterpret_cast<uint64_t>(result->code.data()) >> 8u));
    }

    void SetUp() override {
        register_builtin_hle();
        generation_before = Backend::backend_failed_publication_generation().load();
        ASSERT_EQ(Backend::backend_pending_submission_batches().load(), 0);
        ASSERT_FALSE(Backend::backend_has_unproven_submission());
        ASSERT_NO_FATAL_FAILURE(map_allocation(parent, &parent_physical));
        ASSERT_NO_FATAL_FAILURE(map_allocation(child));
        ASSERT_NO_FATAL_FAILURE(map_allocation(output));
        std::memcpy(reinterpret_cast<void*>(parent), &child, sizeof(child));
        std::memcpy(reinterpret_cast<void*>(child + 16), Values.data(), sizeof(Values));
        ASSERT_NO_FATAL_FAILURE(register_program(vertex, false));
        ASSERT_NO_FATAL_FAILURE(register_program(fragment, true));
        set_graphics_producer_status_query(Live::live_graphics_producer_status);
        set_graphics_raw_source_authority(
            [this](const GuestMappingLease& lease, uint64_t address, uint32_t bytes) {
                const bool current = Live::live_graphics_raw_source_current(lease, address, bytes);
                if (fail_during_source_check) {
                    fail_during_source_check = false;
                    const Backend::BackendProducerAttempt failed;
                }
                return current;
            });
    }

    void TearDown() override {
        set_submit_compute({});
        set_submit_renderer({});
        set_graphics_raw_source_authority({});
        set_graphics_producer_status_query({});
        for (const auto& key : retained_keys) Backend::persistent_color_target_cache().erase(key);
        EXPECT_EQ(Backend::backend_pending_submission_batches().load(), 0);
        // Do not reset the real monotonic failure generation to hide a historical failure.
        const auto unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
        const auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
        if (unmap)
            for (const auto address : mappings) EXPECT_EQ(unmap(address, Page, 0, 0, 0, 0), 0u);
        if (release)
            for (const auto physical : allocations)
                EXPECT_EQ(release(physical, Page, 0, 0, 0, 0), 0u);
    }

    DrawItem draw(uint64_t index = 0, uint64_t order = Order) const {
        DrawItem item;
        item.draw_index = index;
        item.command_order = order;
        item.vs_guest_addr = reinterpret_cast<uint64_t>(vertex->code.data());
        item.fs_guest_addr = reinterpret_cast<uint64_t>(fragment->code.data());
        return item;
    }

    std::vector<std::shared_ptr<const OrderedGraphicsReadPoint>>
    execute(std::vector<SubmitOperation> operations = {{SubmitOperationKind::Draw, 0, Order}},
            std::vector<DrawItem> draws = {}, std::vector<ComputeItem> computes = {},
            LiveComputeFn compute = {}, uint64_t submit = Submit,
            std::function<void(const DrawItem&)> consume = {}) {
        if (draws.empty()) draws.push_back(draw());
        std::vector<std::shared_ptr<const OrderedGraphicsReadPoint>> points;
        const LiveRenderFn render = [&](const std::vector<DrawItem>& span, uint32_t, uint32_t) {
            for (const auto& item : span) {
                points.push_back(item.ordered_read_point);
                if (consume) consume(item);
            }
            return RenderedFrame{};
        };
        execute_ordered_items(operations, draws, computes, std::vector<GpuState::DmaCopy>{}, render,
                              compute, 8, 8, submit);
        return points;
    }

    void inspect(const std::function<void(const DrawItem&)>& consume) {
        const auto points =
            execute({{SubmitOperationKind::Draw, 0, Order}}, {}, {}, {}, Submit, consume);
        ASSERT_EQ(points.size(), 1u);
        ASSERT_TRUE(points[0]);
        EXPECT_FALSE(points[0]->valid_for(
            Submit, Order, reinterpret_cast<uint64_t>(fragment->code.data()),
            points[0]->source(reinterpret_cast<uint64_t>(fragment->code.data()))))
            << "An escaped ticket expires at executor return, even with an unchanged backend";
    }

    GraphicsRawSnapshotContext
    context(const std::shared_ptr<const OrderedGraphicsReadPoint>& point) {
        GraphicsRawSnapshotContext result;
        // This legacy bit is not the authority: the checked overload requires the opaque ticket.
        result.producers_complete = bool(point);
        result.output_allocations.emplace_back(output, Page);
        result.ordered_read_point = point;
        result.source_submit = Submit;
        result.requires_ordered_read_point = true;
        return result;
    }

    std::unique_ptr<GraphicsNestedWideReader> reader(const GraphicsRawSnapshotContext& context,
                                                     SharedShaderWords source = {},
                                                     uint64_t submit = Submit,
                                                     uint64_t order = Order) {
        const uint64_t address = reinterpret_cast<uint64_t>(fragment->code.data());
        if (!source) source = registered_graphics_original(address);
        std::vector<Rdna2Inst> decoded;
        rdna2_walk(source->data(), source->size(), decoded);
        return std::make_unique<GraphicsNestedWideReader>(rdna2_owned_nested_wide_chains(decoded),
                                                          &context, address, std::move(source),
                                                          submit, order);
    }
};

TEST_F(OrderedGraphicsReadPointTest, CleanOrderedSourceOwnsExactBytes) {
    ShaderResourceTable table;
    inspect([&](const DrawItem& item) {
        ASSERT_TRUE(item.ordered_read_point);
        EXPECT_NE(item.ordered_read_point->identity(), 0u);
        const auto ctx = context(item.ordered_read_point);
        GpuState state;
        state.sh[P::SPI_SHADER_USER_DATA_PS_0] = uint32_t(parent);
        state.sh[P::SPI_SHADER_USER_DATA_PS_0 + 1] = uint32_t(parent >> 32u);
        state.sh[P::SPI_SHADER_PGM_RSRC2_PS] = 2u << P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT;
        const auto materialized =
            build_stage_table(state, item.fs_guest_addr, true, 3, Order, &ctx);
        ASSERT_TRUE(materialized);
        ASSERT_EQ(materialized->owned_nested_snapshot_requirements,
                  (std::vector<std::pair<uint32_t, uint32_t>>{{0, 16}, {2, 16}}));
        const auto* numeric = materialized->by_fetch_pc(2);
        ASSERT_TRUE(numeric);
        ASSERT_NE(numeric->host_data, nullptr);
        ASSERT_EQ(numeric->host_data_size, sizeof(Values));
        EXPECT_EQ(std::memcmp(numeric->host_data, Values.data(), sizeof(Values)), 0)
            << "The actual normal checked stage materializer consumed the issued source owner";
        auto checked = reader(context(item.ordered_read_point));
        ASSERT_TRUE(checked->probe(FoldProbe::Raw, 0, parent, 16));
        ASSERT_TRUE(checked->probe(FoldProbe::Raw, 2, child + 16, 16));
        std::array<uint32_t, 4> copied{};
        checked->prefix(2, child + 16, copied.data(), sizeof(copied));
        EXPECT_EQ(copied, Values);
        ASSERT_TRUE(checked->publish(table));
        ASSERT_EQ(table.resources.size(), 2u);
        EXPECT_EQ(table.resources[1].fetch_pc, 2u);
        EXPECT_EQ(table.resources[1].size, 16u);
        // One observed owner serves fold and publication; guest resource content is not recopied.
        std::memset(reinterpret_cast<void*>(child + 16), 0, sizeof(Values));
        EXPECT_TRUE(checked->probe(FoldProbe::Raw, 2, child + 16, 16));
        EXPECT_EQ(checked->word(2, child + 16), Values[0]);
    });
    ASSERT_EQ(table.resources.size(), 2u);
    EXPECT_EQ(std::memcmp(table.resources[1].host_data, Values.data(), sizeof(Values)), 0)
        << "Published immutable bytes outlive the ticket, not its permission to reread guest bytes";
}

TEST_F(OrderedGraphicsReadPointTest, FailedPriorComputeCannotMintFromIdleBackend) {
    ComputeItem compute;
    compute.dispatch_index = 0;
    const auto points =
        execute({{SubmitOperationKind::Dispatch, 0, 10}, {SubmitOperationKind::Draw, 0, Order}}, {},
                {compute}, [](const auto&) { return false; });
    ASSERT_EQ(points.size(), 1u);
    EXPECT_FALSE(points[0]);
    EXPECT_FALSE(Live::live_graphics_producer_status().pending);
}

TEST_F(OrderedGraphicsReadPointTest, MissingProducerAndQueuedDrawAreNotCompletedWork) {
    const auto missing =
        execute({{SubmitOperationKind::Dispatch, 9, 10}, {SubmitOperationKind::Draw, 0, Order}});
    ASSERT_EQ(missing.size(), 1u);
    EXPECT_FALSE(missing[0]);
    const auto queued = execute(
        {{SubmitOperationKind::Draw, 0, Order}, {SubmitOperationKind::Draw, 1, Order + 1}},
        {draw(), draw(1, Order + 1)}, {}, {}, Submit, [&](const DrawItem& item) {
            if (!item.ordered_read_point) return;
            EXPECT_FALSE(item.ordered_read_point->valid_for(
                Submit, item.command_order, item.fs_guest_addr,
                item.ordered_read_point->source(item.fs_guest_addr)))
                << "A queued nonnull point already expired before the batch callback";
            EXPECT_FALSE(
                reader(context(item.ordered_read_point))->probe(FoldProbe::Raw, 0, parent, 16));
        });
    ASSERT_EQ(queued.size(), 2u);
    ASSERT_TRUE(queued[0]);
    EXPECT_FALSE(queued[1]);
}

TEST_F(OrderedGraphicsReadPointTest, PendingBackendAndPhysicalAliasRefuse) {
    Backend::backend_pending_submission_batches().fetch_add(1);
    const auto pending = execute();
    Backend::backend_pending_submission_batches().fetch_sub(1);
    ASSERT_EQ(pending.size(), 1u);
    EXPECT_FALSE(pending[0]);
    uint64_t alias = 0;
    const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    ASSERT_TRUE(map);
    ASSERT_EQ(map(reinterpret_cast<uint64_t>(&alias), Page, 3, 0, parent_physical, Page), 0u);
    mappings.push_back(alias);
    Backend::PersistentColorTargetKey key{alias, 8, 8, VK_FORMAT_R8G8B8A8_UNORM};
    Backend::PersistentColorTargetImage retained_surface;
    retained_surface.guest_producer_seen = true;
    retained_surface.guest_origins.allocations.push_back(
        Backend::retain_guest_allocation(alias, Page));
    Backend::persistent_color_target_cache().emplace(key, std::move(retained_surface));
    retained_keys.push_back(key);
    inspect([&](const DrawItem& item) {
        auto checked = reader(context(item.ordered_read_point));
        EXPECT_FALSE(checked->probe(FoldProbe::Raw, 0, parent, 16))
            << "A different VA in the same actual retained physical allocation is not current "
               "backing";
    });
}

TEST_F(OrderedGraphicsReadPointTest, NewFailureBeforeOrDuringCopyCannotPublish) {
    inspect([&](const DrawItem& item) {
        auto checked = reader(context(item.ordered_read_point));
        { const Backend::BackendProducerAttempt failed; }
        EXPECT_FALSE(checked->probe(FoldProbe::Raw, 0, parent, 16));
    });
    // Historical failures alone do not poison a new clean submit epoch.
    inspect([&](const DrawItem& item) {
        auto checked = reader(context(item.ordered_read_point));
        fail_during_source_check = true;
        EXPECT_FALSE(checked->probe(FoldProbe::Raw, 0, parent, 16));
        ShaderResourceTable table;
        EXPECT_FALSE(checked->publish(table));
        EXPECT_TRUE(table.resources.empty());
    });
    EXPECT_GE(Backend::backend_failed_publication_generation().load(), generation_before + 2);
}

TEST_F(OrderedGraphicsReadPointTest, LaterFailureInvalidatesOwnedObservationAndPublication) {
    inspect([&](const DrawItem& item) {
        auto checked = reader(context(item.ordered_read_point));
        ASSERT_TRUE(checked->probe(FoldProbe::Raw, 0, parent, 16));
        ASSERT_TRUE(checked->probe(FoldProbe::Raw, 2, child + 16, 16));
        { const Backend::BackendProducerAttempt failed; }
        EXPECT_FALSE(checked->probe(FoldProbe::Raw, 2, child + 16, 16));
        EXPECT_THROW(checked->word(2, child + 16), std::runtime_error);
        ShaderResourceTable table;
        EXPECT_FALSE(checked->publish(table));
        EXPECT_TRUE(table.resources.empty());
    });
}

TEST_F(OrderedGraphicsReadPointTest, ForeignSourceWrongOrderAndReplayCannotBorrowAuthority) {
    const auto points = execute(
        {{SubmitOperationKind::Draw, 0, Order}}, {}, {}, {}, Submit, [&](const DrawItem& item) {
            ASSERT_TRUE(item.ordered_read_point);
            const auto ctx = context(item.ordered_read_point);
            const auto foreign =
                std::make_shared<const std::vector<uint32_t>>(Nested.begin(), Nested.end());
            EXPECT_FALSE(reader(ctx, foreign)->probe(FoldProbe::Raw, 0, parent, 16));
            EXPECT_FALSE(reader(ctx, {}, Submit + 1)->probe(FoldProbe::Raw, 0, parent, 16));
            EXPECT_FALSE(reader(ctx, {}, Submit, Order + 1)->probe(FoldProbe::Raw, 0, parent, 16));
            const auto source = registered_graphics_original(item.fs_guest_addr);
            GraphicsNestedWideReader wrong_pc({{4, 2, 16, 16, 0, 16}}, &ctx, item.fs_guest_addr,
                                              source, Submit, Order);
            EXPECT_FALSE(wrong_pc.probe(FoldProbe::Raw, 4, parent, 16));
            // A new byte-validated version at the same address cannot use the earlier source owner.
            fragment->code[4] = 0x7e00022fu;
            EXPECT_FALSE(reader(ctx)->probe(FoldProbe::Raw, 0, parent, 16));
        });
    ASSERT_EQ(points.size(), 1u);
    ASSERT_TRUE(points[0]);
    DrawItem supplied = draw();
    supplied.ordered_read_point = points[0];
    const auto replay = execute({{SubmitOperationKind::Draw, 0, Order}}, {supplied}, {}, {}, 0);
    ASSERT_EQ(replay.size(), 1u);
    EXPECT_FALSE(replay[0]);
}

TEST_F(OrderedGraphicsReadPointTest, SuccessfulLaterComputeExpiresEarlierReadPoint) {
    ComputeItem compute;
    compute.dispatch_index = 0;
    std::shared_ptr<const OrderedGraphicsReadPoint> earlier;
    set_submit_compute([&](const auto&) {
        EXPECT_FALSE(reader(context(earlier))->probe(FoldProbe::Raw, 0, parent, 16))
            << "The earlier read point expires before the next actual compute callback";
        std::memset(reinterpret_cast<void*>(child + 16), 0, sizeof(Values));
        return true;
    });
    const auto points = execute(
        {{SubmitOperationKind::Draw, 0, Order},
         {SubmitOperationKind::Dispatch, 0, Order + 1},
         {SubmitOperationKind::Draw, 1, Order + 2}},
        {draw(), draw(1, Order + 2)}, {compute},
        [](const auto& items) { return execute_compute_items(items); }, Submit,
        [&](const DrawItem& item) {
            ASSERT_TRUE(item.ordered_read_point);
            if (!earlier) {
                earlier = item.ordered_read_point;
                EXPECT_TRUE(reader(context(earlier))->probe(FoldProbe::Raw, 0, parent, 16));
            } else {
                EXPECT_FALSE(reader(context(earlier))->probe(FoldProbe::Raw, 0, parent, 16));
                auto current = reader(context(item.ordered_read_point), {}, Submit, Order + 2);
                ASSERT_TRUE(current->probe(FoldProbe::Raw, 0, parent, 16));
                ASSERT_TRUE(current->probe(FoldProbe::Raw, 2, child + 16, 16));
                EXPECT_EQ(current->word(2, child + 16), 0u);
                EXPECT_NE(earlier->identity(), item.ordered_read_point->identity());
            }
        });
    EXPECT_EQ(points.size(), 2u);
    EXPECT_EQ(Backend::backend_failed_publication_generation().load(), generation_before);
}

TEST_F(OrderedGraphicsReadPointTest, SuccessfulReentrantOrderedWorkPermanentlyExpiresOldPoint) {
    ComputeItem compute;
    compute.dispatch_index = 0;
    inspect([&](const DrawItem& item) {
        auto old = reader(context(item.ordered_read_point));
        ASSERT_TRUE(old->probe(FoldProbe::Raw, 0, parent, 16));
        const auto nested = execute(
            {{SubmitOperationKind::Dispatch, 0, 10}, {SubmitOperationKind::Draw, 0, Order}}, {},
            {compute},
            [&](const auto&) {
                EXPECT_FALSE(old->probe(FoldProbe::Raw, 2, child + 16, 16));
                std::memset(reinterpret_cast<void*>(child + 16), 0, sizeof(Values));
                return true;
            },
            Submit + 1);
        ASSERT_EQ(nested.size(), 1u);
        EXPECT_FALSE(nested[0]) << "An overlapping executor cannot mint even after successful work";
        EXPECT_FALSE(old->probe(FoldProbe::Raw, 2, child + 16, 16))
            << "A successful peer's exit does not revive the outer point";
        ShaderResourceTable table;
        EXPECT_FALSE(old->publish(table));
        EXPECT_TRUE(table.resources.empty());
    });
    inspect([&](const DrawItem& item) {
        auto fresh = reader(context(item.ordered_read_point));
        ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 0, parent, 16));
        ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 2, child + 16, 16));
        EXPECT_EQ(fresh->word(2, child + 16), 0u);
    });
    EXPECT_EQ(Backend::backend_failed_publication_generation().load(), generation_before);
}

TEST_F(OrderedGraphicsReadPointTest, PublicComputeAndRenderEntriesExpireOldPoint) {
    set_submit_compute([&](const auto&) {
        std::memset(reinterpret_cast<void*>(child + 16), 0, sizeof(Values));
        return true;
    });
    set_submit_renderer([&](const auto&, uint32_t, uint32_t) {
        std::memcpy(reinterpret_cast<void*>(child + 16), Values.data(), sizeof(Values));
        return RenderedFrame{};
    });
    for (bool compute : {true, false}) {
        inspect([&](const DrawItem& item) {
            auto old = reader(context(item.ordered_read_point));
            ASSERT_TRUE(old->probe(FoldProbe::Raw, 0, parent, 16));
            if (compute)
                EXPECT_TRUE(execute_compute_items({ComputeItem{}}));
            else
                EXPECT_TRUE(render_submit_items({draw()}, 8, 8).empty());
            EXPECT_FALSE(old->probe(FoldProbe::Raw, 2, child + 16, 16));
            ShaderResourceTable table;
            EXPECT_FALSE(old->publish(table));
            EXPECT_TRUE(table.resources.empty());
        });
        inspect([&](const DrawItem& item) {
            auto fresh = reader(context(item.ordered_read_point));
            ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 0, parent, 16));
            ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 2, child + 16, 16));
            EXPECT_EQ(fresh->word(2, child + 16), compute ? 0u : Values[0]);
        });
    }
    EXPECT_EQ(Backend::backend_failed_publication_generation().load(), generation_before);
}

TEST_F(OrderedGraphicsReadPointTest, PeerEntryRefusesBeforeProbeAndNeverRevivesOldPoint) {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, release = false, completed = false;
    set_submit_compute([&](const auto&) {
        std::unique_lock lock(mutex);
        entered = true;
        changed.notify_all();
        if (!changed.wait_for(lock, std::chrono::seconds(5), [&] { return release; })) return false;
        // No read overlaps this write: the main thread has finished all probe refusals and joins
        // this producer before inspecting a fresh point. This is not a memcpy race-coherence test.
        std::memset(reinterpret_cast<void*>(child + 16), 0, sizeof(Values));
        return true;
    });
    inspect([&](const DrawItem& item) {
        auto old = reader(context(item.ordered_read_point));
        ASSERT_TRUE(old->probe(FoldProbe::Raw, 0, parent, 16));
        std::jthread peer([&] { completed = execute_compute_items({ComputeItem{}}); });
        bool observed = false;
        {
            std::unique_lock lock(mutex);
            observed = changed.wait_for(lock, std::chrono::seconds(5), [&] { return entered; });
        }
        EXPECT_TRUE(observed);
        if (observed) {
            EXPECT_FALSE(old->probe(FoldProbe::Raw, 2, child + 16, 16));
            const auto overlapping = execute();
            EXPECT_EQ(overlapping.size(), 1u);
            if (!overlapping.empty()) EXPECT_FALSE(overlapping[0]);
        }
        {
            std::lock_guard lock(mutex);
            release = true;
        }
        changed.notify_all();
        peer.join();
        EXPECT_TRUE(completed);
        EXPECT_FALSE(old->probe(FoldProbe::Raw, 2, child + 16, 16));
        ShaderResourceTable table;
        EXPECT_FALSE(old->publish(table));
    });
    inspect([&](const DrawItem& item) {
        auto fresh = reader(context(item.ordered_read_point));
        ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 0, parent, 16));
        ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 2, child + 16, 16));
        EXPECT_EQ(fresh->word(2, child + 16), 0u);
    });
    EXPECT_EQ(Backend::backend_failed_publication_generation().load(), generation_before);
}

TEST_F(OrderedGraphicsReadPointTest, DirectOrderedDmaExpiresOldPointAndFreshPointSeesBytes) {
    std::memset(reinterpret_cast<void*>(output), 0, sizeof(Values));
    inspect([&](const DrawItem& item) {
        auto old = reader(context(item.ordered_read_point));
        ASSERT_TRUE(old->probe(FoldProbe::Raw, 0, parent, 16));
        GpuState::DmaCopy copy;
        copy.src = output;
        copy.dst = child + 16;
        copy.bytes = sizeof(Values);
        copy.sels = 3u | (3u << 8u) | kDmaDataAddressSource;
        copy.command_order = Order + 1;
        ASSERT_TRUE(execute_ordered_dma_copy(copy));
        EXPECT_EQ(std::memcmp(reinterpret_cast<const void*>(child + 16),
                              reinterpret_cast<const void*>(output), sizeof(Values)),
                  0);
        EXPECT_FALSE(old->probe(FoldProbe::Raw, 2, child + 16, 16));
        ShaderResourceTable table;
        EXPECT_FALSE(old->publish(table));
    });
    inspect([&](const DrawItem& item) {
        auto fresh = reader(context(item.ordered_read_point));
        ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 0, parent, 16));
        ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 2, child + 16, 16));
        EXPECT_EQ(fresh->word(2, child + 16), 0u);
    });
    EXPECT_EQ(Backend::backend_failed_publication_generation().load(), generation_before);
}

TEST_F(OrderedGraphicsReadPointTest, LegacyExecutionEntriesExpirePermissionBeforeEmptyReturn) {
    for (unsigned entry = 0; entry < 3; ++entry) {
        inspect([&](const DrawItem& item) {
            auto old = reader(context(item.ordered_read_point));
            ASSERT_TRUE(old->probe(FoldProbe::Raw, 0, parent, 16));
            const GpuState empty;
            if (entry == 0)
                EXPECT_FALSE(execute_and_present(empty, 8, 8, false));
            else if (entry == 1)
                EXPECT_FALSE(execute_nonrender_submit_work(empty, Submit + 1));
            else
                EXPECT_TRUE(execute_gpustate(empty, RenderFn{}).empty());
            EXPECT_FALSE(old->probe(FoldProbe::Raw, 2, child + 16, 16));
        });
        inspect([&](const DrawItem& item) {
            auto fresh = reader(context(item.ordered_read_point));
            ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 0, parent, 16));
            ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 2, child + 16, 16));
            EXPECT_EQ(fresh->word(2, child + 16), Values[0]);
        });
    }
}

TEST_F(OrderedGraphicsReadPointTest, RetainedWriteDataExpiresOldPointAndFreshPointSeesBytes) {
    const uint32_t replacement = 0x55667788u;
    Pm4Command write{};
    write.kind = Pm4Command::Kind::WriteData;
    write.wd_addr = child + 16;
    write.wd_declared_num = write.wd_num = 1;
    write.wd_data = &replacement;
    write.wd_valid = true;
    const GpuState::MemoryEffect effect(write, Order + 1);
    ASSERT_NE(effect.cmd.wd_data, write.wd_data);
    ASSERT_EQ(effect.write_data, (std::vector<uint32_t>{replacement}));
    inspect([&](const DrawItem& item) {
        auto old = reader(context(item.ordered_read_point));
        ASSERT_TRUE(old->probe(FoldProbe::Raw, 0, parent, 16));
        execute_ordered_memory_effect(effect);
        uint32_t actual = 0;
        std::memcpy(&actual, reinterpret_cast<const void*>(child + 16), sizeof(actual));
        ASSERT_EQ(actual, replacement)
            << "The actual mapped retained WRITE_DATA API changed the numeric source";
        EXPECT_FALSE(old->probe(FoldProbe::Raw, 2, child + 16, 16));
        ShaderResourceTable table;
        EXPECT_FALSE(old->publish(table));
    });
    inspect([&](const DrawItem& item) {
        auto fresh = reader(context(item.ordered_read_point));
        ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 0, parent, 16));
        ASSERT_TRUE(fresh->probe(FoldProbe::Raw, 2, child + 16, 16));
        EXPECT_EQ(fresh->word(2, child + 16), replacement);
    });
    EXPECT_EQ(Backend::backend_failed_publication_generation().load(), generation_before);
}

TEST_F(OrderedGraphicsReadPointTest, OrderedVoidEffectRemainsUnknownAfterParserStall) {
    ASSERT_NO_FATAL_FAILURE(register_program(vertex, false, ProceduralVertex));
    ASSERT_NO_FATAL_FAILURE(register_program(fragment, true, NestedExport));
    Program* resource_free = nullptr;
    ASSERT_NO_FATAL_FAILURE(register_program(resource_free, true, ResourceFreeExport));
    GpuState clean;
    for (const Program* program : {vertex, fragment})
        for (const auto& reg : program->registers) clean.sh[reg.offset] = reg.value;
    clean.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    clean.cx[P::CB_TARGET_MASK] = clean.cx[P::CB_SHADER_MASK] = 15;
    clean.sh[P::SPI_SHADER_USER_DATA_PS_0] = uint32_t(parent);
    clean.sh[P::SPI_SHADER_USER_DATA_PS_0 + 1] = uint32_t(parent >> 32u);
    clean.sh[P::SPI_SHADER_PGM_RSRC2_PS] = 2u << P::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT;
    GpuState::Draw first;
    first.index_count = 3;
    first.instance_count = 1;
    first.command_order = 100;
    clean.draws.push_back(first);
    ASSERT_TRUE(draw_requires_owned_nested_snapshot(clean))
        << "The genuine nested PS selects checked ordered realization, not the eager route";

    struct Observation {
        uint64_t index;
        bool has_point, live_point;
        std::array<uint32_t, 4> words{};
        bool has_snapshot = false;
    };
    std::vector<Observation> observations;
    const auto observe = [&](const DrawItem& item) {
        Observation observed{item.draw_index, bool(item.ordered_read_point), false};
        if (item.ordered_read_point) {
            const auto& point = item.ordered_read_point;
            EXPECT_NE(point->identity(), 0u);
            EXPECT_EQ(point->source(item.fs_guest_addr),
                      registered_graphics_original(item.fs_guest_addr));
            observed.live_point =
                point->valid_for(live_render_phase().source_submit, item.command_order,
                                 item.fs_guest_addr, point->source(item.fs_guest_addr));
        }
        if (item.fs_guest_addr == reinterpret_cast<uint64_t>(fragment->code.data())) {
            EXPECT_TRUE(item.prt);
            if (item.prt) {
                const auto* snapshot = owned_nested_snapshot_at(*item.prt, 2, sizeof(Values));
                EXPECT_NE(snapshot, nullptr)
                    << "Normal checked materialization must own the actual child before rendering";
                if (snapshot && snapshot->host_data && snapshot->host_data_size == sizeof(Values)) {
                    std::memcpy(observed.words.data(), snapshot->host_data, sizeof(Values));
                    observed.has_snapshot = true;
                }
            }
        }
        observations.push_back(observed);
    };
    set_submit_renderer([&](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
        for (const auto& item : items) observe(item);
        return RenderedFrame{};
    });
    // No pixels are supplied or published: the positive is genuine realization and byte ownership.
    EXPECT_FALSE(execute_ordered_and_present(clean, 8, 8, Submit + 1, false));
    ASSERT_EQ(observations.size(), 1u);
    EXPECT_TRUE(observations[0].has_point);
    EXPECT_TRUE(observations[0].live_point);
    ASSERT_TRUE(observations[0].has_snapshot);
    EXPECT_EQ(observations[0].words, Values);

    auto with_effect = clean;
    auto later_state = std::make_shared<GpuState>(clean);
    for (const auto& reg : resource_free->registers) later_state->sh[reg.offset] = reg.value;
    ASSERT_FALSE(draw_requires_owned_nested_snapshot(*later_state));
    auto second = first;
    second.command_order = 200;
    second.state = std::move(later_state);
    with_effect.draws.push_back(std::move(second));
    const uint32_t replacement = 0x55667788u;
    Pm4Command write{};
    write.kind = Pm4Command::Kind::WriteData;
    write.wd_addr = child + 16;
    write.wd_declared_num = write.wd_num = 1;
    write.wd_data = &replacement;
    write.wd_valid = true;
    with_effect.ordered_memory_effects.emplace_back(write, 150);
    with_effect.parser_stalls.push_back({175});
    observations.clear();
    EXPECT_FALSE(execute_ordered_and_present(with_effect, 8, 8, Submit + 2, false));
    ASSERT_EQ(observations.size(), 2u) << "Legacy resource-free draw realization remains intact";
    EXPECT_EQ(observations[0].index, 0u);
    EXPECT_TRUE(observations[0].has_point);
    EXPECT_FALSE(observations[0].live_point)
        << "The next operation advances the local epoch before flushing the queued first draw";
    ASSERT_TRUE(observations[0].has_snapshot);
    EXPECT_EQ(observations[0].words, Values);
    EXPECT_EQ(observations[1].index, 1u);
    EXPECT_FALSE(observations[1].has_point)
        << "A parser stall cannot turn the void retained effect outcome into completion proof";
    uint32_t actual = 0;
    std::memcpy(&actual, reinterpret_cast<const void*>(child + 16), sizeof(actual));
    ASSERT_EQ(actual, replacement) << "The retained mapped WRITE_DATA actually executed";

    observations.clear();
    EXPECT_FALSE(execute_ordered_and_present(clean, 8, 8, Submit + 3, false));
    ASSERT_EQ(observations.size(), 1u);
    EXPECT_TRUE(observations[0].has_point);
    EXPECT_TRUE(observations[0].live_point);
    ASSERT_TRUE(observations[0].has_snapshot);
    auto changed = Values;
    changed[0] = replacement;
    EXPECT_EQ(observations[0].words, changed)
        << "Only a later independent clean submit can own the genuinely changed child";
    EXPECT_EQ(Backend::backend_failed_publication_generation().load(), generation_before);
}

}   // namespace
