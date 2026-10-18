// The MIMG spatial-coordinate definition (ir/ops/image_coordinates.cpp) through the real
// recompiler and the live Vulkan compute backend: every arm of image_coordinates_cases.hpp in
// Wave64 and Wave32, each lane with its own coordinates under the harness's partial EXEC.
// The integer interpreter cannot run an image access, so this binary links Vulkan.
//
// Contracts it pins: a lane reads the texel at ITS coordinates (Separate, NSA, A16 and 3D forms);
// an EXEC-inactive lane keeps its old VDST; a coordinate past the image reads 0, the existing
// out-of-range contract prosper delegates to robustImageAccess (#3531, #131); and a
// FORCE_UNNORMALIZED sample at a texel centre returns that texel (the coordinate normalized by
// 1/extent). Also exports the arms as the synthetic witness corpus (HOW_TO_ADD_A_FAMILY.md,
// "Image instructions (MIMG)").
#include "gpu/recompiler/ir/image_coordinates_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <system_error>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::image_coordinates;

namespace {

// Every allocation outlives the process's compute caches, so no two runs share an address.
std::vector<uint32_t>& keep_alive(std::vector<uint32_t> words) {
    static std::vector<std::unique_ptr<std::vector<uint32_t>>> owned;
    owned.push_back(std::make_unique<std::vector<uint32_t>>(std::move(words)));
    return *owned.back();
}

class IrImageCoordinatesExecution : public testing::Test {
protected:
    void SetUp() override {
        prosper::frontend::execute_live_compute_items({});
        ASSERT_TRUE(prosper::frontend::live_compute_storage_image_device().initialized)
            << "production Vulkan initialization failed";
    }
};

// Runs one arm and returns what each lane stored, or {} if it did not compile or execute.
std::vector<uint32_t> run(cases::Arm arm, uint32_t wave, const LaneInputs& in, uint32_t index) {
    std::vector<uint32_t> words;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d}) words.insert(words.end(), array->begin(), array->end());
    for (bool active : in.active) words.push_back(active ? 1u : 0u);
    words.insert(words.end(), in.s2.begin(), in.s2.end());
    auto& buffer = keep_alive(std::move(words));
    auto& image = keep_alive(cases::image_words(arm));
    const LaneProgram p = cases::program(arm, wave, reinterpret_cast<uint64_t>(buffer.data()),
                                         reinterpret_cast<uint64_t>(image.data()));
    const uint64_t address = 0x1a6e000000ull + uint64_t{index} * 0x100u;
    ComputeItem item;
    item.spirv = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                   {RecompileDiagnosticStage::Compute, address});
    if (item.spirv.empty()) {
        ADD_FAILURE() << cases::name(arm) << " wave " << wave
                      << " refused: " << last_terminal_reject_reason(address);
        return {};
    }
    item.user_sgprs = p.config.user_sgprs;
    item.resources = std::make_shared<ShaderResourceTable>(p.table);
    item.code_addr = address;
    item.dispatch_index = index;
    item.command_order = uint64_t{index} * 10u;
    item.launch.threads_x = item.launch.local_x = cases::kLanes;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    if (!prosper::frontend::execute_live_compute_items({item})) {
        ADD_FAILURE() << cases::name(arm) << " wave " << wave << " did not execute";
        return {};
    }
    return {buffer.begin(), buffer.begin() + cases::kLanes};
}

TEST_F(IrImageCoordinatesExecution, EveryLaneReadsItsOwnCoordinates) {
    uint32_t index = 1;
    for (const uint32_t wave : {64u, 32u}) {
        for (const cases::Arm arm : cases::kArms) {
            SCOPED_TRACE(testing::Message() << cases::name(arm) << " wave " << wave);
            const LaneInputs in = cases::inputs(arm);
            const auto got = run(arm, wave, in, index++);
            ASSERT_EQ(got.size(), cases::kLanes);
            uint32_t data = 0, outside = 0, inactive = 0;
            for (uint32_t lane = 0; lane < cases::kLanes; ++lane) {
                const uint32_t want = cases::expected(arm, in, lane);
                EXPECT_EQ(got[lane], want) << "lane " << lane << std::hex << " s0 " << in.s0[lane]
                                           << " s1 " << in.s1[lane] << " s2 " << in.s2[lane];
                if (!in.active[lane]) ++inactive;
                else if (want == 0u) ++outside;
                else ++data;
            }
            // The arm can see each property: lanes reading data, lanes past the image (the
            // integer arms), and lanes EXEC leaves alone.
            EXPECT_GE(data, 16u);
            EXPECT_GE(inactive, 4u);
            if (!cases::sampled(arm)) EXPECT_GE(outside, 4u);
        }
    }
}

// The coordinates differ per lane, so a definition that read one lane's (or one axis's) value for
// all of them fails above. This checks the premise: no two active in-range lanes of an arm share a
// texel, and both axes vary.
TEST(IrImageCoordinatesCases, LanesUseDifferentCoordinates) {
    for (const cases::Arm arm : cases::kArms) {
        const LaneInputs in = cases::inputs(arm);
        std::vector<uint32_t> seen;
        uint32_t xs = 0, ys = 0;
        for (uint32_t lane = 0; lane < cases::kLanes; ++lane) {
            const auto c = cases::sampled(arm) ? cases::sampled_coordinates(lane)
                                               : cases::coordinates(lane);
            xs |= 1u << (c.x % 32u);
            ys |= 1u << c.y;
            const uint32_t want = cases::expected(arm, in, lane);
            if (in.active[lane] && want != 0u) seen.push_back(want);
        }
        std::sort(seen.begin(), seen.end());
        EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end()) << cases::name(arm);
        EXPECT_GE(std::popcount(xs), 16) << cases::name(arm);
        EXPECT_EQ(ys, cases::sampled(arm) ? 0xfu : 0x1fu) << cases::name(arm);
    }
}

// The synthetic witness corpus: every arm in both wave sizes. Every program must replay.
TEST(IrImageCoordinatesExecutionWitness, ExportsReplayableWitnessPrograms) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_image_coordinates_witness");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump("ir_image_coordinates_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (const uint32_t wave : {64u, 32u})
        for (const cases::Arm arm : cases::kArms) {
            write_compare_witness_program(dump, 0x1a6f000000ull + count * 0x100, count,
                                          cases::program(arm, wave));
            ++count;
        }
    EXPECT_EQ(count, 12u);
}

}   // namespace
