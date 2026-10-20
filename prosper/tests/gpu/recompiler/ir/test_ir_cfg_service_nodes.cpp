// test_ir_cfg_service_nodes -- step 11 shadow groundwork, with no consumer. The independent
// per-block and dispatch-tail observations must retain selected service pc, kind and order.
// Hand-built missing/wrong/reordered events keep a clean census from hiding a broken comparator.
#include "gpu/recompiler/ir/cfg/service_census.hpp"
#include "gpu/recompiler/ir/cfg/service_nodes.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include "fixtures/dpp_row_max.hpp"
#include "gpu/recompiler/float32_rte_support.hpp"
#include "gpu/recompiler/float64_support.hpp"
#include "gpu/recompiler/ir/cfg_fixtures.hpp"
#include "gpu/recompiler/ir/cfg_streams.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir::cfg;
using namespace prosper::test::cfg_streams;
using namespace prosper::test::cfg_fixtures;

namespace {

struct Example {
    std::vector<Rdna2Inst> ins = walk({kVNop, kVNop, kEndpgm});
    std::unordered_set<uint32_t> sites{0, 1};
    SplitPolicy policy{true, &sites, nullptr};
    GuestCfg cfg = build_guest_cfg(ins, policy);
    std::vector<BlockServiceSelection> selected{{0, 0, {0, ServiceKind::Mbcnt}},
                                                {1, 1, {1, ServiceKind::Readlane}}};
    std::vector<DispatchServiceEvent> dispatcher{{0, {0, ServiceKind::Mbcnt}},
                                                 {1, {1, ServiceKind::Readlane}}};
};

TEST(IrCfgServices, ProjectsEveryKindOntoItsOwnServiceBlockEnd) {
    std::vector<uint32_t> words(kServiceKindCount, kVNop);
    words.push_back(kEndpgm);
    const auto ins = walk(words);
    std::unordered_set<uint32_t> sites;
    std::vector<BlockServiceSelection> selected;
    std::vector<DispatchServiceEvent> oracle;
    std::set<std::string> names;
    for (uint32_t k = 0; k < kServiceKindCount; ++k) {
        const auto kind = static_cast<ServiceKind>(k);
        sites.insert(k);
        selected.push_back({k, k, {k, kind}});
        oracle.push_back({k, {k, kind}});
        names.insert(service_kind_name(kind));
    }
    EXPECT_EQ(kServiceKindCount, 21u);
    EXPECT_EQ(names.size(), 21u);
    EXPECT_FALSE(names.contains("invalid"));
    const auto cfg = build_guest_cfg(ins, {true, &sites, nullptr});
    const ServiceNodes model = build_service_nodes(cfg, ins, selected);
    EXPECT_TRUE(model.difference.empty()) << model.difference;
    ASSERT_EQ(model.nodes.size(), 21u);
    for (size_t k = 0; k < model.nodes.size(); ++k) EXPECT_EQ(model.nodes[k].block, k);
    EXPECT_TRUE(service_nodes_difference(model, oracle).empty());
}

TEST(IrCfgServices, ComparisonDetectsMissingExtraWrongPcWrongKindAndOrder) {
    Example e;
    const ServiceNodes model = build_service_nodes(e.cfg, e.ins, e.selected);
    ASSERT_TRUE(service_nodes_difference(model, e.dispatcher).empty());
    auto oracle = e.dispatcher;
    oracle.pop_back();
    EXPECT_FALSE(service_nodes_difference(model, oracle).empty());
    oracle = e.dispatcher;
    oracle.push_back({2, {2, ServiceKind::Swizzle}});
    EXPECT_FALSE(service_nodes_difference(model, oracle).empty());
    oracle = e.dispatcher;
    oracle[0].event.pc = 99;
    EXPECT_FALSE(service_nodes_difference(model, oracle).empty());
    oracle = e.dispatcher;
    oracle[0].event.kind = ServiceKind::Append;
    EXPECT_FALSE(service_nodes_difference(model, oracle).empty());
    oracle = e.dispatcher;
    std::reverse(oracle.begin(), oracle.end());
    EXPECT_FALSE(service_nodes_difference(model, oracle).empty());
    oracle = e.dispatcher;
    ++oracle[1].dispatch;
    EXPECT_FALSE(service_nodes_difference(model, oracle).empty());
}

TEST(IrCfgServices, ProjectionRefusesWrongBlockInteriorPcAndMissingServiceSplit) {
    Example e;
    auto selected = e.selected;
    selected[0].block_pc = 99;
    EXPECT_FALSE(build_service_nodes(e.cfg, e.ins, selected).difference.empty());
    selected = e.selected;
    selected[0].event.pc = 1;
    EXPECT_FALSE(build_service_nodes(e.cfg, e.ins, selected).difference.empty());
    const auto no_split = build_guest_cfg(e.ins, {true, nullptr, nullptr});
    EXPECT_FALSE(build_service_nodes(no_split, e.ins, e.selected).difference.empty());
    selected = e.selected;
    selected[0].event.kind = ServiceKind::Count;
    EXPECT_FALSE(build_service_nodes(e.cfg, e.ins, selected).difference.empty());
}

TEST(IrCfgServices, DispatchIdentityRejectsTwoServiceTailsAndPreservesNonPcOrder) {
    Example e;
    e.selected[1].dispatch = 0;
    EXPECT_FALSE(build_service_nodes(e.cfg, e.ins, e.selected).difference.empty());
    // Forward-chain fusion can visit a higher-pc tail before a later group's lower-pc tail.
    e.selected = {{0, 1, {1, ServiceKind::Readlane}}, {1, 0, {0, ServiceKind::Mbcnt}}};
    e.dispatcher = {{0, {1, ServiceKind::Readlane}}, {1, {0, ServiceKind::Mbcnt}}};
    const auto model = build_service_nodes(e.cfg, e.ins, e.selected);
    EXPECT_TRUE(service_nodes_difference(model, e.dispatcher).empty());
    std::reverse(e.selected.begin(), e.selected.end());
    EXPECT_FALSE(build_service_nodes(e.cfg, e.ins, e.selected).difference.empty());
}

TEST(IrCfgServices, CapturesEachPointerSlotAndRestoresNestedObservers) {
    Example e;
    ServiceCapture outer;
    {
        ServiceCapture inner;
        for (size_t k = 0; k < kServiceKindCount; ++k) {
            ServiceObservation observation(e.ins, e.policy);
            SelectedServicePointers selected{};
            selected[k] = &e.ins[0];
            observation.block(0, 0, selected);
            observation.group(0, selected);
            observation.complete();
        }
        ASSERT_EQ(inner.records().size(), 21u);
        for (size_t k = 0; k < inner.records().size(); ++k) {
            const auto& record = inner.records()[k];
            ASSERT_EQ(record.model.nodes.size(), 1u);
            EXPECT_EQ(record.model.nodes[0].event.kind, static_cast<ServiceKind>(k));
            EXPECT_TRUE(record.difference.empty()) << record.difference;
            EXPECT_TRUE(record.complete);
        }
    }
    {
        ServiceObservation observation(e.ins, e.policy, 2);
        SelectedServicePointers selected{};
        selected[0] = &e.ins[0];
        observation.block(0, 0, selected);
        // Simulates refusal before the dispatch-tail assignment; never calls group/complete.
    }
    ASSERT_EQ(outer.records().size(), 1u);
    const auto& record = outer.records()[0];
    EXPECT_FALSE(record.complete);
    EXPECT_EQ(record.pending_selections, 1u);
    EXPECT_EQ(record.auxiliary_raw_mask_sites, 2u);
    EXPECT_TRUE(record.model.nodes.empty());
    EXPECT_TRUE(record.difference.empty());
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& words, int stage) {
    if (stage == 0)
        return recompile_compute(words.data(), words.size(), nullptr, ComputeShaderConfig{});
    if (stage == 1) return recompile_fragment(words.data(), words.size());
    if (stage == 3)
        return recompile_valu(words.data(), words.size(), 3, 1, nullptr, 0, kDefaultComputePgmRsrc1,
                              true);
    return recompile_vertex(words.data(), words.size());
}

TEST(IrCfgServices, CommittedFixturesMatchTheDispatcherWithoutChangingModules) {
    publish_float64_support(true, true, true);
    publish_float32_rte_support(true, true, true);
    const auto fixtures = guest_program_fixtures(__FILE__);
    ASSERT_GE(fixtures.size(), 10u);
    size_t regions = 0, nodes = 0, complete = 0;
    std::array<size_t, kServiceKindCount> kinds{};
    for (const auto& path : fixtures) {
        const auto words = load_words(path);
        ASSERT_FALSE(words.empty()) << path;
        const bool synthetic = path.filename() == "rdna2_cfg_services.hex";
        if (synthetic) ASSERT_EQ(words, prosper::test::dpp_row_max_program({1, 2}));
        // The existing row-service test's VALU profile supplies three genuine input VGPRs and
        // selects the dispatcher. The ordinary fixture profiles remain unchanged.
        for (int stage = 0; stage < (synthetic ? 4 : 3); ++stage) {
            const auto without = compile(words, stage);
            ServiceCapture capture;
            EXPECT_EQ(compile(words, stage), without) << path.filename() << " stage " << stage;
            if (stage == 3) {
                ASSERT_EQ(capture.records().size(), 1u);
                const auto& record = capture.records().front();
                ASSERT_EQ(record.model.nodes.size(), 2u);
                EXPECT_TRUE(record.complete);
                EXPECT_EQ(record.model.nodes[0].event,
                          (ServiceEvent{0, ServiceKind::ComputeDppRowShr}));
                EXPECT_EQ(record.model.nodes[1].event,
                          (ServiceEvent{2, ServiceKind::ComputeDppRowShr}));
            }
            for (const auto& record : capture.records()) {
                EXPECT_TRUE(record.difference.empty())
                    << path.filename() << " stage " << stage << " entry " << record.entry_pc << ": "
                    << record.difference;
                ++regions;
                nodes += record.model.nodes.size();
                complete += record.complete;
                for (const auto& node : record.model.nodes)
                    ++kinds[static_cast<size_t>(node.event.kind)];
                // Use an actual selected fixture event as well as the independent manual controls.
                if (!record.model.nodes.empty()) {
                    auto missing = record.model;
                    missing.nodes.pop_back();
                    EXPECT_FALSE(service_nodes_difference(missing, record.dispatcher).empty());
                }
            }
        }
    }
    EXPECT_GE(regions, 8u);
    EXPECT_GT(nodes, 0u);
    EXPECT_GT(complete, 0u);
    std::printf("cfg-services-fixtures regions=%zu nodes=%zu complete=%zu\n", regions, nodes,
                complete);
    for (size_t k = 0; k < kinds.size(); ++k)
        std::printf("cfg-services-fixtures-kind kind=%s nodes=%zu\n",
                    service_kind_name(static_cast<ServiceKind>(k)), kinds[k]);
}

}   // namespace
