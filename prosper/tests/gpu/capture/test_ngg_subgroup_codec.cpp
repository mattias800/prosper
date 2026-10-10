// The v72 capture record of a merged-NGG draw description (#3135 P5, ngg_subgroup_codec.hpp):
// Kena's LUT producer description round-trips field for field, an absent description is one byte,
// and a malformed record is refused rather than replayed. Pure CPU (the shell is compiled, not run).
#include "fixtures/ngg_merged_lut_fixture.hpp"
#include "fixtures/test_data.h"
#include "gpu/capture/serialize/ngg_subgroup_codec.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test;

namespace {

std::shared_ptr<const NggSubgroupDraw> kena(uint32_t vertices, uint32_t instances) {
    static const auto linked = ngg::kena_linked(tests_root(__FILE__) / "data");
    static const ShaderResourceTable table = ngg::kena_resources(4);
    NggSubgroupDrawRequest request;
    request.linked_code = linked.data();
    request.dwords = linked.size();
    request.resources = &table;
    request.shell.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(ngg::kKenaRsrc2Gs);
    request.shell.user_sgprs = ngg::kKenaUserSgprs;
    request.limits = ngg::kena_limits();
    request.shape = {NggInputTopology::TriangleStrip, vertices, instances, 0};
    request.raster.topology = NggOutputTopology::TriangleList;
    request.raster.layer_from_pos1 = true;
    request.raster.layer_slices = 32;
    request.raster.route = NggLayerRoute::ForwardingGeometry;
    request.push_constants = {1, 2, 3, 4, 5, 6, 7, 8};
    return build_ngg_subgroup_draw(request);
}

std::vector<uint8_t> encode(const std::shared_ptr<const NggSubgroupDraw>& draw) {
    Writer w;
    EXPECT_TRUE(write_ngg_subgroup_draw(w, draw));
    return w.data;
}

bool decode(const std::vector<uint8_t>& bytes, std::shared_ptr<const NggSubgroupDraw>& out) {
    std::string error;
    Reader r{bytes.data(), bytes.size(), &error};
    return read_ngg_subgroup_draw(r, out) && r.left == 0;
}

TEST(NggSubgroupCodec, KenaRoundTripsFieldForField) {
    const auto draw = kena(76, 2);   // two wave-count groups, four runs, a geometry stage
    ASSERT_TRUE(draw);
    ASSERT_EQ(draw->groups.size(), 2u);
    std::shared_ptr<const NggSubgroupDraw> back;
    ASSERT_TRUE(decode(encode(draw), back));
    ASSERT_TRUE(back);
    EXPECT_EQ(back->topology, draw->topology);
    EXPECT_EQ(back->route, NggLayerRoute::ForwardingGeometry);
    EXPECT_EQ(back->vertices_per_primitive, 3u);
    EXPECT_EQ(back->count_violations, draw->count_violations);
    EXPECT_EQ(back->native_wave64, draw->native_wave64);
    EXPECT_EQ(back->lds_bytes, draw->lds_bytes);
    EXPECT_EQ(back->guest_bindings, draw->guest_bindings);
    EXPECT_EQ(back->push_constants, draw->push_constants);
    EXPECT_EQ(back->layout.words_per_lane, draw->layout.words_per_lane);
    EXPECT_EQ(back->layout.param_targets, draw->layout.param_targets);
    EXPECT_EQ(back->layout.pos1_word, draw->layout.pos1_word);
    ASSERT_EQ(back->groups.size(), draw->groups.size());
    for (size_t g = 0; g < draw->groups.size(); ++g) {
        const auto& a = draw->groups[g];
        const auto& b = back->groups[g];
        EXPECT_EQ(b.waves, a.waves);
        EXPECT_EQ(b.blocks, a.blocks);
        EXPECT_EQ(b.export_words, a.export_words);
        EXPECT_EQ(b.launch_words, a.launch_words);
        EXPECT_EQ(*b.stages->shell, *a.stages->shell);
        EXPECT_EQ(*b.stages->raster_vertex, *a.stages->raster_vertex);
        EXPECT_EQ(b.stages->raster_geometry, a.stages->raster_geometry);
        EXPECT_FALSE(b.stages->raster_geometry.empty());
        EXPECT_EQ(b.stages->shell_hash, a.stages->shell_hash) << "recomputed on read";
        EXPECT_EQ(b.stages->guest_bindings, a.stages->guest_bindings);
    }
    ASSERT_EQ(back->runs.size(), draw->runs.size());
    for (size_t i = 0; i < draw->runs.size(); ++i) {
        EXPECT_EQ(back->runs[i].group, draw->runs[i].group);
        EXPECT_EQ(back->runs[i].first_block, draw->runs[i].first_block);
        EXPECT_EQ(back->runs[i].blocks, draw->runs[i].blocks);
    }
}

TEST(NggSubgroupCodec, AbsentIsOneZeroByte) {
    const auto bytes = encode(nullptr);
    EXPECT_EQ(bytes, std::vector<uint8_t>{0});
    std::shared_ptr<const NggSubgroupDraw> back = kena(4, 1);
    ASSERT_TRUE(decode(bytes, back));
    EXPECT_FALSE(back);
}

TEST(NggSubgroupCodec, MalformedRecordsAreRefused) {
    const auto draw = kena(4, 4);
    ASSERT_TRUE(draw);
    const auto good = encode(draw);
    std::shared_ptr<const NggSubgroupDraw> back;
    ASSERT_TRUE(decode(good, back));

    // Truncated anywhere: refused.
    for (size_t cut : {size_t{1}, good.size() / 2, good.size() - 1}) {
        const std::vector<uint8_t> truncated(good.begin(), good.begin() + cut);
        EXPECT_FALSE(decode(truncated, back)) << "cut at " << cut;
    }
    // A presence byte other than 0/1.
    auto bad = good;
    bad[0] = 2;
    EXPECT_FALSE(decode(bad, back));
    // The last run's block count past its group's blocks (the final four bytes).
    bad = good;
    bad[bad.size() - 4] = 0xff;
    EXPECT_FALSE(decode(bad, back)) << "a run past its group's export blocks";
    // A launch record count that disagrees with the blocks: shorten the description by changing
    // the group's block count (it follows the waves word).
    auto altered = std::make_shared<NggSubgroupDraw>(*draw);
    altered->groups[0].blocks += 1;
    Writer w;
    ASSERT_TRUE(write_ngg_subgroup_draw(w, altered));
    EXPECT_FALSE(decode(w.data, back)) << "launch records sized for other blocks";
    altered = std::make_shared<NggSubgroupDraw>(*draw);
    altered->groups.clear();
    Writer empty;
    EXPECT_FALSE(write_ngg_subgroup_draw(empty, altered)) << "a description with no group";
}

// #4808: a Wave32 description (wave_lanes 32) round-trips its width through flags bit 2, with its
// launch records and export blocks sized for 32-lane waves, and up to eight waves. The same record
// read as Wave64 would mis-size both, so the flag is what the reader checks them against.
TEST(NggSubgroupCodec, AWave32DescriptionKeepsItsWidth) {
    const auto draw = kena(4, 4);
    ASSERT_TRUE(draw);
    auto w32 = std::make_shared<NggSubgroupDraw>(*draw);
    w32->wave_lanes = 32;
    for (NggSubgroupWaveGroup& group : w32->groups) {
        group.waves = 5;   // more than Wave64's four
        group.launch_words.assign(size_t{group.blocks} * 32u * 5u * kNggLaunchWordsPerLane, 7u);
        group.export_words = group.blocks * w32->layout.block_words(5, 32);
    }
    std::shared_ptr<const NggSubgroupDraw> back;
    const auto bytes = encode(w32);
    ASSERT_TRUE(decode(bytes, back));
    ASSERT_TRUE(back);
    EXPECT_EQ(back->wave_lanes, 32u);
    EXPECT_EQ(back->groups.at(0).waves, 5u);
    EXPECT_EQ(back->groups.at(0).launch_words, w32->groups.at(0).launch_words);

    ASSERT_TRUE(decode(encode(draw), back));
    EXPECT_EQ(back->wave_lanes, 64u) << "a Wave64 description reads back as Wave64";
    auto w64 = std::make_shared<NggSubgroupDraw>(*w32);
    w64->wave_lanes = 64;   // the same 32-lane-sized records, claimed as Wave64
    EXPECT_FALSE(decode(encode(w64), back)) << "five Wave64 waves, and records sized for 32 lanes";
}

// A capture is v72 only when a draw carries a description; every other capture stays a v71 file,
// byte for byte (the v71 tail stays last).
TEST(NggSubgroupCodec, OnlyACaptureWithADescriptionIsV72) {
    GpuCaptureFile file;
    file.draws.resize(2);
    EXPECT_EQ(gpu_capture_version_for(file), 71u);
    file.draws[1].ngg_subgroup = kena(4, 1);
    ASSERT_TRUE(file.draws[1].ngg_subgroup);
    EXPECT_EQ(gpu_capture_version_for(file), 72u);
}

}   // namespace
