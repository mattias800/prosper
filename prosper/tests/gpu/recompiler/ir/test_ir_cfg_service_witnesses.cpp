#include "gpu/recompiler/ir/cfg_service_witness_corpus.hpp"
#include "gpu/recompiler/ir/cfg_service_witnesses.hpp"
#include "gpu/recompiler/ir/cfg/service_census.hpp"
#include "gpu/recompiler/ir/cfg/service_nodes.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ios>
#include <regex>
#include <set>
#include <stdexcept>

namespace {
namespace fixture = prosper::test::cfg_services;
using namespace prosper::gpu;
using namespace prosper::gpu::ir::cfg;

class IrCfgServiceWitness : public testing::TestWithParam<size_t> {};

TEST_P(IrCfgServiceWitness, NodeMatchesDispatcherAndLeavesModuleUnchanged) {
    const auto& w = fixture::witnesses()[GetParam()];
    ASSERT_LE(w.code.size(), 32u) << "keep witnesses small and project-owned";
    std::set<uint32_t> starts;
    for (size_t pc = 0; pc < w.code.size();) {
        const auto in = rdna2_decode_one(w.code.data() + pc, w.code.size() - pc);
        ASSERT_GT(in.len_dwords, 0u);
        ASSERT_LE(in.len_dwords, w.code.size() - pc);
        const auto& effects = rdna2_instruction_effects(in);
        ASSERT_NE(effects.name, nullptr) << pc;
        ASSERT_NE(effects.status, Rdna2EffectsStatus::Unknown) << pc;
        ASSERT_NE(effects.status, Rdna2EffectsStatus::Reserved) << pc;
        starts.insert(static_cast<uint32_t>(pc));
        pc += in.len_dwords;
    }
    for (const auto& event : w.expected) ASSERT_TRUE(starts.contains(event.pc));
    fixture::publish_support();
    const auto without = fixture::compile(w);
    ASSERT_FALSE(without.empty());
    ServiceCapture capture;
    EXPECT_EQ(fixture::compile(w), without);
    ASSERT_EQ(capture.records().size(), 1u);
    const auto& record = capture.records().front();
    ASSERT_TRUE(record.complete);
    ASSERT_TRUE(record.difference.empty()) << record.difference;
    ASSERT_EQ(record.model.nodes.size(), w.expected.size());
    ASSERT_EQ(record.dispatcher.size(), w.expected.size());
    for (size_t i = 0; i < w.expected.size(); ++i) {
        const auto& node = record.model.nodes[i];
        EXPECT_EQ(node.event, w.expected[i]);
        EXPECT_EQ(node.event, record.dispatcher[i].event);
        EXPECT_EQ(node.dispatch, record.dispatcher[i].dispatch);
        std::printf("cfg-services-witness name=%s pc=%u kind=%s dispatch=%u\n", w.name,
                    node.event.pc, service_kind_name(node.event.kind), node.dispatch);
    }
    const auto selected = std::find_if(record.model.nodes.begin(), record.model.nodes.end(),
                                       [&](const auto& n) { return n.event.kind == w.target; });
    ASSERT_NE(selected, record.model.nodes.end());
    auto missing = record.model;
    missing.nodes.erase(missing.nodes.begin() + (selected - record.model.nodes.begin()));
    EXPECT_FALSE(service_nodes_difference(missing, record.dispatcher).empty())
        << "the actual selected event must discriminate a missing node";
}

INSTANTIATE_TEST_SUITE_P(PreviouslyUnobserved, IrCfgServiceWitness,
                         testing::Range(size_t(0), fixture::witnesses().size()),
                         [](const testing::TestParamInfo<size_t>& p) {
                             return fixture::witnesses()[p.param].name;
                         });

TEST(IrCfgServiceWitnesses, CoversEveryPreviouslyUnobservedKind) {
    std::set<ServiceKind> actual;
    for (const auto& w : fixture::witnesses()) actual.insert(w.target);
    const std::set<ServiceKind> expected = {ServiceKind::Append,
                                            ServiceKind::LdsStore,
                                            ServiceKind::LdsFminmax,
                                            ServiceKind::Bpermute,
                                            ServiceKind::FragmentDppMinRowShr,
                                            ServiceKind::ComputeDppRowRor8,
                                            ServiceKind::MaskFfbh,
                                            ServiceKind::MaskReduction,
                                            ServiceKind::PacketWqm,
                                            ServiceKind::SavedMaskPairCompare,
                                            ServiceKind::VopcMaskZeroCompare};
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(fixture::witnesses().size(), 11u);
}

TEST(IrCfgServiceCorpus, ExplicitProfilesRoundTripAndRejectDrift) {
    fixture::publish_support();
    for (const auto& w : fixture::witnesses()) {
        SCOPED_TRACE(w.name);
        const auto stem = prosper_test::test_scratch_path(w.name);
        const auto code = stem.string() + ".bin", profile = stem.string() + ".inputs";
        ASSERT_TRUE(fixture::write_witness(stem, w));
        const auto replay = fixture::replay_witness(code, profile);
        ASSERT_TRUE(replay.produced) << replay.reject << ' ' << replay.incomplete;
        ASSERT_TRUE(replay.incomplete.empty());
        EXPECT_EQ(replay.spirv, fixture::compile(w));
        if (w.profile != fixture::Profile::Packet) continue;
        EXPECT_EQ(&fixture::read_packet_witness(code, profile), &w);
        auto drift = w.code;
        drift.front() ^= 1u;
        ASSERT_TRUE(fixture::write_words(code, drift));
        EXPECT_THROW(fixture::read_packet_witness(code, profile), std::runtime_error);
        ASSERT_TRUE(fixture::write_witness(stem, w));
        std::ofstream invalid(profile, std::ios::binary);
        invalid << fixture::kProfileMagic << "UnknownProfile\n";
        invalid.close();
        EXPECT_THROW(fixture::read_packet_witness(code, profile), std::runtime_error);
    }
}

TEST(IrCfgServiceCorpus, ExportUsesTheHarvestersDumpNamingContract) {
    fixture::publish_support();
    const auto root = prosper_test::test_scratch_path("service-export");
    fixture::export_corpus(root);
    // shader_corpus.py's DUMP_FILE contract: address and synthetic ID must be hexadecimal.
    const std::regex accepted("^(cs|ps|vsmain|vs)_([0-9a-f]+)_([0-9a-f]+)\\.bin$");
    size_t count = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.path().extension() != ".bin") continue;
        EXPECT_TRUE(std::regex_match(entry.path().filename().string(), accepted));
        auto profile = entry.path();
        profile.replace_extension(".inputs");
        EXPECT_TRUE(fixture::replay_witness(entry.path(), profile).produced);
        ++count;
    }
    EXPECT_EQ(count, fixture::witnesses().size());
    EXPECT_THROW(fixture::export_corpus(root), std::runtime_error);
}
} // namespace
