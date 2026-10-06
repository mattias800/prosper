// The merged-NGG subgroup shell (#3135 P2) executed offline on Vulkan.
//
// A synthetic two-wave program pins the shell's own contract: launch words reach v0..v8 and s3,
// set-2 shell I/O coexists with a guest binding 0, EXEC predicates every record word and flag, and
// GS_ALLOC_REQ lands in the block header once, from wave 0 only.
//
// Kena's captured LUT producer (tests/data) then runs under the P1 planner's launch for the draw it
// was captured with: a 4-vertex strip x 32 instances, one subgroup per instance. The primitives it
// exports must be 64 non-null triangles whose indices hit written vertices, one layer per triangle,
// covering layers 0..31, with PARAM.xy equal to the vertex's normalized screen position. Four
// launch-model controls must break that: GS offsets not scaled by VGT_ESGS_RING_ITEMSIZE, a 1-wave
// subgroup where the guest needs more output lanes, v5/v8 swapped, and a mutated vertex input.
#include "gpu/execute/ngg_subgroup_plan.hpp"
#include "gpu/recompiler/ngg_export_record.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/ngg_merged_lut_fixture.hpp"
#include "fixtures/ngg_subgroup_runner.h"
#include "fixtures/test_data.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace prosper::gpu;
using prosper::test::NggSubgroupDispatch;
using prosper::test::run_ngg_subgroup;

namespace {

using namespace prosper::test::ngg;

using LaneEdit = std::function<void(NggLaneLaunch&, uint32_t& s3)>;

// Launch records for every subgroup in `plan`, all dispatched with `waves` waves.
std::vector<uint32_t> launch_records(const NggSubgroupPlan& plan, const NggSubgroupLimits& limits,
                                     uint32_t waves, const LaneEdit& edit = {}) {
    std::vector<uint32_t> words;
    for (const NggSubgroup& subgroup : plan.subgroups)
        for (uint32_t wave = 0; wave < waves; ++wave) {
            uint32_t s3 = ngg_merged_wave_info(subgroup, wave);
            if (waves != subgroup.waves) s3 = (s3 & 0x0fffffffu) | (waves << 28);
            for (uint32_t lane = 0; lane < 64; ++lane) {
                NggLaneLaunch launch = ngg_lane_launch(subgroup, limits, wave, lane);
                uint32_t lane_s3 = s3;
                if (edit) edit(launch, lane_s3);
                words.insert(words.end(), launch.v, launch.v + 9);
                words.push_back(lane_s3);
            }
        }
    return words;
}

struct Primitive {
    uint32_t block = 0, slot = 0, layer = 0;
    std::array<std::array<float, 4>, 3> pos{};
};

struct LutSummary {
    uint32_t blocks = 0, valid_headers = 0, non_null = 0, bad_connectivity = 0;
    uint32_t mixed_layer = 0, degenerate = 0, param_mismatch = 0;
    std::map<uint32_t, uint32_t> prims_per_layer;
    std::vector<Primitive> primitives;
};

float as_float(uint32_t bits) {
    return std::bit_cast<float>(bits);
}

// Decode the export buffer the way the P3 pass-through draw will: connectivity is valid only when
// p < prims_alloc, the prim was written, the null bit is clear, and all three indices are below
// verts_alloc and hit written vertices.
LutSummary summarize(const std::vector<uint32_t>& out, const NggExportRecordLayout& layout,
                     uint32_t waves, uint32_t blocks) {
    LutSummary summary;
    summary.blocks = blocks;
    const uint32_t lanes = 64u * waves;
    for (uint32_t block = 0; block < blocks; ++block) {
        const uint32_t base = block * layout.block_words(waves);
        const uint32_t verts = out[base + kNggHeaderVertsAlloc];
        const uint32_t prims = out[base + kNggHeaderPrimsAlloc];
        if (out[base + kNggHeaderAllocRequests] != 1 || out[base + kNggHeaderStrayRequests] != 0)
            continue;
        ++summary.valid_headers;
        const auto record = [&](uint32_t thread) {
            return &out[layout.record_offset(waves, block, thread)];
        };
        for (uint32_t slot = 0; slot < std::min(prims, lanes); ++slot) {
            const uint32_t* prim = record(slot);
            if (!(prim[kNggRecordFlagsWord] & kNggFlagPrim)) {
                ++summary.bad_connectivity;
                continue;
            }
            const uint32_t word = prim[kNggRecordPrimWord];
            if (word & 0x80000000u) continue;
            ++summary.non_null;
            const uint32_t index[3] = {word & 0x1ffu, (word >> 10) & 0x1ffu, (word >> 20) & 0x1ffu};
            bool connected = true;
            Primitive primitive{block, slot, 0, {}};
            std::set<uint32_t> layers;
            for (uint32_t k = 0; k < 3; ++k) {
                if (index[k] >= verts || index[k] >= lanes ||
                    (record(index[k])[kNggRecordFlagsWord] & (kNggFlagPos0 | kNggFlagPos1)) !=
                        (kNggFlagPos0 | kNggFlagPos1)) {
                    connected = false;
                    break;
                }
                const uint32_t* vertex = record(index[k]);
                for (uint32_t c = 0; c < 4; ++c)
                    primitive.pos[k][c] = as_float(vertex[kNggRecordPos0Word + c]);
                layers.insert(vertex[layout.pos1_word + 2]);
                // PARAM.xy is the vertex's normalized screen position.
                const float w = primitive.pos[k][3];
                const float sx = (primitive.pos[k][0] / w + 1.0f) * 0.5f;
                const float sy = (1.0f - primitive.pos[k][1] / w) * 0.5f;
                const float px = as_float(vertex[layout.param_word(0)]);
                const float py = as_float(vertex[layout.param_word(0) + 1]);
                if (!(std::fabs(px - sx) < 1e-5f && std::fabs(py - sy) < 1e-5f))
                    ++summary.param_mismatch;
            }
            if (!connected) {
                ++summary.bad_connectivity;
                continue;
            }
            if (layers.size() != 1) {
                ++summary.mixed_layer;
                continue;
            }
            primitive.layer = *layers.begin();
            if (primitive.pos[0] == primitive.pos[1] || primitive.pos[1] == primitive.pos[2] ||
                primitive.pos[0] == primitive.pos[2])
                ++summary.degenerate;
            ++summary.prims_per_layer[primitive.layer];
            summary.primitives.push_back(primitive);
        }
    }
    return summary;
}

// Everything the captured LUT draw must produce.
::testing::AssertionResult is_complete_lut(const LutSummary& s, uint32_t layers,
                                           uint32_t prims_per_layer) {
    if (s.valid_headers != s.blocks)
        return ::testing::AssertionFailure()
               << "valid headers " << s.valid_headers << "/" << s.blocks;
    if (s.non_null != layers * prims_per_layer)
        return ::testing::AssertionFailure() << "non-null primitives " << s.non_null;
    if (s.bad_connectivity || s.mixed_layer || s.degenerate || s.param_mismatch)
        return ::testing::AssertionFailure()
               << "bad=" << s.bad_connectivity << " mixed=" << s.mixed_layer
               << " degenerate=" << s.degenerate << " param-mismatch=" << s.param_mismatch;
    if (s.prims_per_layer.size() != layers)
        return ::testing::AssertionFailure() << "layers covered " << s.prims_per_layer.size();
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const auto it = s.prims_per_layer.find(layer);
        if (it == s.prims_per_layer.end() || it->second != prims_per_layer)
            return ::testing::AssertionFailure() << "layer " << layer << " incomplete";
    }
    return ::testing::AssertionSuccess();
}

struct KenaRun {
    std::vector<uint32_t> module;
    NggExportRecordLayout layout;
    std::string refusal;
};

KenaRun compile_kena(uint32_t waves, bool native, uint32_t vertices = 4) {
    KenaRun run;
    const auto linked = kena_linked(prosper::test::tests_root(__FILE__) / "data");
    const ShaderResourceTable table = kena_resources(vertices);
    NggSubgroupShellConfig config;
    config.waves = waves;
    config.lds_bytes = kKenaLds;
    config.user_sgprs = kKenaUserSgprs;
    config.native_wave64 = native;
    run.module =
        recompile_ngg_subgroup(linked.data(), linked.size(), &table, config, &run.layout,
                               {RecompileDiagnosticStage::Vertex, 0x5009440000ull}, &run.refusal);
    return run;
}

std::vector<uint32_t> dispatch_kena(const KenaRun& compiled, uint32_t waves, bool native,
                                    uint32_t workgroups, std::vector<uint32_t> launch,
                                    const std::vector<uint32_t>& records) {
    NggSubgroupDispatch dispatch;
    dispatch.spirv = compiled.module;
    dispatch.local_size = 64 * waves;
    dispatch.workgroups = workgroups;
    dispatch.launch = std::move(launch);
    dispatch.export_words = workgroups * compiled.layout.block_words(waves);
    dispatch.guest_buffers = kena_buffers(records);
    dispatch.push_constants.assign(kKenaUserSgprs, 0u);
    dispatch.required_subgroup_size = native ? 64u : 0u;
    return run_ngg_subgroup(dispatch).value_or(std::vector<uint32_t>{});
}

NggSubgroupPlan lut_plan() {
    NggDrawShape draw;
    draw.topology = NggInputTopology::TriangleStrip;
    draw.vertex_count = 4;
    draw.instance_count = 32;
    return plan_ngg_subgroups(draw, kena_limits());
}

// ---- The synthetic two-wave contract -------------------------------------------------------

uint32_t launch_word(uint32_t block, uint32_t thread, uint32_t reg) {
    return 0x10000000u + block * 0x100000u + thread * 0x100u + reg;
}

std::vector<uint32_t> synthetic_launch(uint32_t blocks, uint32_t waves) {
    std::vector<uint32_t> words;
    for (uint32_t block = 0; block < blocks; ++block)
        for (uint32_t thread = 0; thread < 64u * waves; ++thread) {
            for (uint32_t reg = 0; reg < 9; ++reg) words.push_back(launch_word(block, thread, reg));
            words.push_back(waves << 28 | (thread / 64u) << 24 | 64u << 8 | 64u);
        }
    return words;
}

std::vector<uint32_t> run_synthetic(const std::vector<uint32_t>& code, uint32_t waves, bool native,
                                    uint32_t blocks, NggExportRecordLayout* layout,
                                    std::string* refusal = nullptr) {
    const ShaderResourceTable table = synthetic_resources();
    NggSubgroupShellConfig config;
    config.waves = waves;
    config.user_sgprs = 4;
    config.native_wave64 = native;
    std::string why;
    const auto module =
        recompile_ngg_subgroup(code.data(), code.size(), &table, config, layout,
                               {RecompileDiagnosticStage::Vertex, 0x5a5a0000ull}, &why);
    if (refusal) *refusal = why;
    if (module.empty()) return {};
    NggSubgroupDispatch dispatch;
    dispatch.spirv = module;
    dispatch.local_size = 64 * waves;
    dispatch.workgroups = blocks;
    dispatch.launch = synthetic_launch(blocks, waves);
    dispatch.export_words = blocks * layout->block_words(waves);
    dispatch.guest_buffers = {{0, {kSyntheticCbuf, 2, 3, 4}}};
    dispatch.push_constants.assign(4, 0u);
    dispatch.required_subgroup_size = native ? 64u : 0u;
    return run_ngg_subgroup(dispatch).value_or(std::vector<uint32_t>{});
}

}   // namespace

TEST(NggSubgroupShell, SyntheticTwoWaveRecordsCarryLaunchValuesAndTheHeader) {
    NggExportRecordLayout layout;
    std::string why;
    const auto out = run_synthetic(synthetic_program(), 2, false, 3, &layout, &why);
    ASSERT_FALSE(out.empty()) << why;
    ASSERT_EQ(layout.words_per_lane, 10u);
    for (uint32_t block = 0; block < 3; ++block) {
        const uint32_t base = block * layout.block_words(2);
        EXPECT_EQ(out[base + kNggHeaderVertsAlloc], 5u) << "block " << block;
        EXPECT_EQ(out[base + kNggHeaderPrimsAlloc], 3u) << "block " << block;
        EXPECT_EQ(out[base + kNggHeaderAllocRequests], 1u) << "block " << block;
        EXPECT_EQ(out[base + kNggHeaderStrayRequests], 0u) << "block " << block;
        uint32_t bad = 0;
        for (uint32_t thread = 0; thread < 128; ++thread) {
            const uint32_t* r = &out[layout.record_offset(2, block, thread)];
            const bool right =
                r[0] == (kNggFlagPrim | kNggFlagPos0 | (1u << kNggFlagParamShift)) &&
                r[1] == launch_word(block, thread, 2) && r[2] == launch_word(block, thread, 0) &&
                r[3] == launch_word(block, thread, 1) && r[4] == launch_word(block, thread, 2) &&
                r[5] == launch_word(block, thread, 3) && r[6] == launch_word(block, thread, 5) &&
                r[7] == launch_word(block, thread, 8) && r[8] == kSyntheticCbuf &&
                r[9] == launch_word(block, thread, 3);
            if (!right && bad++ < 3)
                ADD_FAILURE() << "block " << block << " thread " << thread << " flags=" << r[0]
                              << " prim=" << std::hex << r[1] << " param.z=" << r[8];
        }
        EXPECT_EQ(bad, 0u) << "block " << block;
    }
}

TEST(NggSubgroupShell, AllocationRequestsAreCountedPerWave) {
    NggExportRecordLayout layout;
    const auto every_wave = run_synthetic(synthetic_program(false), 2, false, 2, &layout);
    ASSERT_FALSE(every_wave.empty());
    EXPECT_EQ(every_wave[kNggHeaderAllocRequests], 1u);
    EXPECT_EQ(every_wave[kNggHeaderStrayRequests], 1u) << "wave 1's request is a stray";
    const auto silent = run_synthetic(synthetic_program(true, false), 2, false, 2, &layout);
    ASSERT_FALSE(silent.empty());
    EXPECT_EQ(silent[kNggHeaderAllocRequests], 0u);
    EXPECT_EQ(silent[kNggHeaderVertsAlloc], 0u);
}

TEST(NggSubgroupShell, ExecPredicatesRecordWordsAndFlags) {
    NggExportRecordLayout layout;
    std::string why;
    const auto out = run_synthetic(synthetic_program(true, true, true), 2, false, 1, &layout, &why);
    ASSERT_FALSE(out.empty()) << why;
    for (uint32_t thread : {0u, 31u, 32u, 63u, 64u, 95u, 96u, 127u}) {
        const uint32_t* r = &out[layout.record_offset(2, 0, thread)];
        const bool active = thread % 64u < 32u;
        EXPECT_EQ(r[0] & (1u << kNggFlagParamShift), active ? (1u << kNggFlagParamShift) : 0u)
            << "thread " << thread;
        EXPECT_EQ(r[6], active ? launch_word(0, thread, 5) : 0u) << "thread " << thread;
        EXPECT_EQ(r[0] & kNggFlagPos0, kNggFlagPos0) << "POS0 ran under full EXEC";
    }
}

// A fetch through an untouched ABI index reads VertexID from v5 and InstanceID from v8: one
// vertex-rate and one instance-rate stream, each returning a value that names its index.
TEST(NggSubgroupShell, VertexAndInstanceFetchesUseTheirLaunchIndices) {
    const std::vector<uint32_t> code = {
        0xbefe04c1u,   // s_mov_b64 exec, -1
        0xe0002000u,
        0x80020a00u,   // buffer_load_format_x v10, v0, s[8:11], 0 idxen (vertex rate)
        0xe0002000u,
        0x80030b00u,   // buffer_load_format_x v11, v0, s[12:15], 0 idxen (instance rate)
        0xbf8c0000u,   // s_waitcnt 0
        0x7e120280u,   // v_mov_b32 v9, 0
        0xf8000941u,
        0x00000009u,   // exp prim v9
        0xf80000cfu,
        0x03020100u,   // exp pos0 v0..v3
        0xf800020fu,
        0x08050b0au,   // exp param0 v10, v11, v5, v8
        0xbf810000u,
    };
    ShaderResourceTable table;
    std::map<uint32_t, std::vector<uint32_t>> buffers;
    for (const auto& [binding, pc, base, mode] :
         {std::tuple<uint32_t, uint32_t, float, VertexFetchIndexMode>{3, 1, 100.0f,
                                                                      VertexFetchIndexMode::Vertex},
          {4, 3, 1000.0f, VertexFetchIndexMode::Instance}}) {
        ShaderResource stream;
        stream.cls = ResourceClass::VertexBuffer;
        stream.binding = binding;
        stream.size = 4 * 64;
        stream.stride = 4;
        stream.format = DataFormat::Float32;
        stream.num_components = 1;
        stream.fetch_pc = pc;
        stream.sgpr_base = binding == 3 ? 8 : 12;
        stream.fetch_index_mode = mode;
        table.resources.push_back(stream);
        auto& words = buffers[binding];
        for (uint32_t i = 0; i < 64; ++i)
            words.push_back(std::bit_cast<uint32_t>(base + static_cast<float>(i)));
    }
    NggSubgroupShellConfig config;
    config.user_sgprs = 8;
    NggExportRecordLayout layout;
    std::string why;
    const auto module =
        recompile_ngg_subgroup(code.data(), code.size(), &table, config, &layout,
                               {RecompileDiagnosticStage::Vertex, 0x5a5a1000ull}, &why);
    ASSERT_FALSE(module.empty()) << why;
    const auto vertex_id = [](uint32_t block, uint32_t lane) { return (lane * 7u + block) % 41u; };
    const auto instance_id = [](uint32_t block, uint32_t lane) {
        return 3u + block * 5u + lane % 2u;
    };
    NggSubgroupDispatch dispatch;
    dispatch.spirv = module;
    dispatch.workgroups = 2;
    for (uint32_t block = 0; block < 2; ++block)
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t v[9] = {
                0, 0, 0, 0, 0, vertex_id(block, lane), 0, 0, instance_id(block, lane)};
            dispatch.launch.insert(dispatch.launch.end(), v, v + 9);
            dispatch.launch.push_back(1u << 28 | 64u << 8 | 64u);
        }
    dispatch.export_words = 2 * layout.block_words(1);
    dispatch.guest_buffers = buffers;
    dispatch.push_constants.assign(8, 0u);
    const auto out = run_ngg_subgroup(dispatch).value_or(std::vector<uint32_t>{});
    ASSERT_FALSE(out.empty());
    uint32_t bad = 0;
    for (uint32_t block = 0; block < 2; ++block)
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t* r = &out[layout.record_offset(1, block, lane) + layout.param_word(0)];
            const bool right =
                r[0] ==
                    std::bit_cast<uint32_t>(100.0f + static_cast<float>(vertex_id(block, lane))) &&
                r[1] == std::bit_cast<uint32_t>(1000.0f +
                                                static_cast<float>(instance_id(block, lane))) &&
                r[2] == vertex_id(block, lane) && r[3] == instance_id(block, lane);
            if (!right && bad++ < 3)
                ADD_FAILURE() << "block " << block << " lane " << lane << " vertex fetch "
                              << std::bit_cast<float>(r[0]) << " instance fetch "
                              << std::bit_cast<float>(r[1]);
        }
    EXPECT_EQ(bad, 0u);
}

TEST(NggSubgroupShell, NativeWave64MatchesPortableOnTheSyntheticProgram) {
    if (!prosper::test::ngg_native_wave64_supported(2))
        GTEST_SKIP() << "device cannot require full 64-lane compute subgroups";
    NggExportRecordLayout portable_layout, native_layout;
    const auto portable = run_synthetic(synthetic_program(), 2, false, 2, &portable_layout);
    const auto native = run_synthetic(synthetic_program(), 2, true, 2, &native_layout);
    ASSERT_FALSE(portable.empty() || native.empty());
    EXPECT_EQ(portable, native);
}

TEST(NggSubgroupShell, KenaLutUnderThePlannerLaunchIsComplete) {
    const NggSubgroupPlan plan = lut_plan();
    ASSERT_TRUE(plan.ok()) << plan.refusal;
    ASSERT_EQ(plan.subgroups.size(), 32u);
    const KenaRun compiled = compile_kena(1, false);
    ASSERT_FALSE(compiled.module.empty()) << compiled.refusal;
    ASSERT_EQ(compiled.layout.words_per_lane, 14u);
    const auto out = dispatch_kena(compiled, 1, false, 32, launch_records(plan, kena_limits(), 1),
                                   vertex_records(kLutQuad));
    ASSERT_FALSE(out.empty());
    const LutSummary summary = summarize(out, compiled.layout, 1, 32);
    EXPECT_TRUE(is_complete_lut(summary, 32, 2));
}

TEST(NggSubgroupShell, KenaNativeWave64OutputIsIdenticalToPortable) {
    if (!prosper::test::ngg_native_wave64_supported(1))
        GTEST_SKIP() << "device cannot require full 64-lane compute subgroups";
    const NggSubgroupPlan plan = lut_plan();
    const KenaRun portable = compile_kena(1, false);
    const KenaRun native = compile_kena(1, true);
    ASSERT_FALSE(portable.module.empty()) << portable.refusal;
    ASSERT_FALSE(native.module.empty()) << native.refusal;
    ASSERT_NE(portable.module, native.module) << "the two shells must differ to compare anything";
    const auto launch = launch_records(plan, kena_limits(), 1);
    const auto a = dispatch_kena(portable, 1, false, 32, launch, vertex_records(kLutQuad));
    const auto b = dispatch_kena(native, 1, true, 32, launch, vertex_records(kLutQuad));
    ASSERT_FALSE(a.empty() || b.empty());
    EXPECT_EQ(a, b);
    EXPECT_TRUE(is_complete_lut(summarize(b, native.layout, 1, 32), 32, 2));
}

TEST(NggSubgroupShell, KenaControlUnscaledGsOffsetsBreaksTheLut) {
    const NggSubgroupPlan plan = lut_plan();
    NggSubgroupLimits unscaled = kena_limits();
    unscaled.esgs_item_size = 1;
    const KenaRun compiled = compile_kena(1, false);
    ASSERT_FALSE(compiled.module.empty()) << compiled.refusal;
    const auto out = dispatch_kena(compiled, 1, false, 32, launch_records(plan, unscaled, 1),
                                   vertex_records(kLutQuad));
    ASSERT_FALSE(out.empty());
    const LutSummary summary = summarize(out, compiled.layout, 1, 32);
    EXPECT_FALSE(is_complete_lut(summary, 32, 2));
    // Measured: every GS lane reads the same ES slot, so all 64 triangles collapse to a point.
    EXPECT_EQ(summary.degenerate, 64u);
}

TEST(NggSubgroupShell, KenaControlSwappedVertexAndInstanceIdBreaksTheLut) {
    const NggSubgroupPlan plan = lut_plan();
    const KenaRun compiled = compile_kena(1, false);
    ASSERT_FALSE(compiled.module.empty()) << compiled.refusal;
    const auto out = dispatch_kena(compiled, 1, false, 32,
                                   launch_records(plan, kena_limits(), 1,
                                                  [](NggLaneLaunch& launch, uint32_t&) {
                                                      std::swap(launch.v[5], launch.v[8]);
                                                  }),
                                   vertex_records(kLutQuad));
    ASSERT_FALSE(out.empty());
    const LutSummary summary = summarize(out, compiled.layout, 1, 32);
    EXPECT_FALSE(is_complete_lut(summary, 32, 2));
    // Measured: the layer becomes the vertex index, so every triangle spans several layers.
    EXPECT_EQ(summary.mixed_layer, 64u);
}

TEST(NggSubgroupShell, KenaControlMutatedVertexInputChangesOnlyParam) {
    const NggSubgroupPlan plan = lut_plan();
    const KenaRun compiled = compile_kena(1, false);
    ASSERT_FALSE(compiled.module.empty()) << compiled.refusal;
    const auto launch = launch_records(plan, kena_limits(), 1);
    auto records = vertex_records(kLutQuad);
    const auto before = dispatch_kena(compiled, 1, false, 32, launch, records);
    for (uint32_t vertex = 0; vertex < 4; ++vertex)
        records[vertex * 4u + 3u] = std::bit_cast<uint32_t>(0.25f);   // PARAM.y only
    const auto after = dispatch_kena(compiled, 1, false, 32, launch, records);
    ASSERT_FALSE(before.empty() || after.empty());
    const LutSummary a = summarize(before, compiled.layout, 1, 32);
    const LutSummary b = summarize(after, compiled.layout, 1, 32);
    EXPECT_TRUE(is_complete_lut(a, 32, 2));
    EXPECT_FALSE(is_complete_lut(b, 32, 2)) << "PARAM no longer matches the screen position";
    EXPECT_GT(b.param_mismatch, 0u);
    ASSERT_EQ(a.primitives.size(), b.primitives.size());
    for (size_t i = 0; i < a.primitives.size(); ++i) {
        EXPECT_EQ(a.primitives[i].pos, b.primitives[i].pos) << "POS0 must not move";
        EXPECT_EQ(a.primitives[i].layer, b.primitives[i].layer) << "the layer must not move";
    }
}

// A 34-vertex strip has 32 triangles and 96 output vertices per instance, so the planner gives it
// two waves. A 1-wave launch of the same subgroup has no lanes for output vertices 64..95.
TEST(NggSubgroupShell, KenaControlOneWaveLosesOutputVertices) {
    std::vector<std::array<float, 2>> strip;
    strip.reserve(34);
    for (uint32_t k = 0; k < 34; ++k)
        strip.push_back({-1.0f + static_cast<float>(k >> 1u) / 8.0f, (k & 1u) ? 1.0f : -1.0f});
    NggDrawShape draw;
    draw.topology = NggInputTopology::TriangleStrip;
    draw.vertex_count = 34;
    draw.instance_count = 2;
    const NggSubgroupPlan plan = plan_ngg_subgroups(draw, kena_limits());
    ASSERT_TRUE(plan.ok()) << plan.refusal;
    ASSERT_EQ(plan.subgroups.size(), 2u);
    ASSERT_EQ(plan.subgroups[0].waves, 2u);

    const KenaRun one_wave = compile_kena(1, false, 34);
    ASSERT_FALSE(one_wave.module.empty()) << one_wave.refusal;
    const auto lost = dispatch_kena(one_wave, 1, false, 2, launch_records(plan, kena_limits(), 1),
                                    vertex_records(strip));
    ASSERT_FALSE(lost.empty());
    const LutSummary lost_summary = summarize(lost, one_wave.layout, 1, 2);
    // Measured: the guest sizes its own allocation from s3's wave count, so it requests 64
    // vertices for 32 triangles and the triangles past the 21st lose their vertices (they
    // reference the wrong, repeated slots).
    EXPECT_FALSE(is_complete_lut(lost_summary, 2, 32));
    EXPECT_GT(lost_summary.degenerate + lost_summary.bad_connectivity, 0u)
        << "a 1-wave subgroup has no lanes for output vertices 64..95";

    if (!prosper::test::ngg_native_wave64_supported(2))
        GTEST_SKIP() << "the two-wave positive needs native Wave64";
    const KenaRun two_wave = compile_kena(2, true, 34);
    ASSERT_FALSE(two_wave.module.empty()) << two_wave.refusal;
    const auto complete = dispatch_kena(
        two_wave, 2, true, 2, launch_records(plan, kena_limits(), 2), vertex_records(strip));
    ASSERT_FALSE(complete.empty());
    EXPECT_TRUE(is_complete_lut(summarize(complete, two_wave.layout, 2, 2), 2, 32));
}
