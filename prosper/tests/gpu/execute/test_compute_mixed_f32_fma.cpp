// #5225: compare every mixed-F32 operand form to a separately built native GLSL Fma.
// Strict RTE is qualified by that driver's independent control, never a driver-name whitelist.
#include "fixtures/compute_runner.h"
#include "fixtures/f32_fma_device_receipt.hpp"
#include "fixtures/f32_fma_native_control.hpp"
#include "fixtures/mixed_f32_fma.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::mixed_f32;
namespace {
std::filesystem::path artifact_root() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_MIXED_F32_GPU_ARTIFACTS");
    const auto root = configured ? std::filesystem::path(configured)
                                 : prosper_test::test_scratch_dir() / "mixed-f32-gpu";
    std::filesystem::create_directories(root);
    return root;
}

std::vector<uint32_t> native(Shape shape, const LaneInputs& in) {
    std::vector<uint32_t> data(4 * in.lanes());
    for (std::size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto src = sources(shape, in, lane);
        for (std::size_t slot = 0; slot < 3; ++slot) data[slot * in.lanes() + lane] = src[slot];
    }
    std::vector<uint32_t> result;
    const auto dispatch =
        prosper::test::run_compute(prosper::test::fma_rounding::native_fma_control(),
                                   std::vector<float>(64), 64, 64, {}, data, &result);
    EXPECT_EQ(dispatch.size(), 64u);
    if (result.size() != data.size()) return {};
    return {result.begin() + 192, result.end()};
}

std::vector<uint32_t> guest(Shape shape, uint32_t wave, const std::string& stem) {
    auto p = program(shape, wave);
    const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
    EXPECT_FALSE(module.empty());
    if (module.empty()) return {};
    const auto root = artifact_root();
    std::ofstream spv(root / (stem + ".spv"), std::ios::binary);
    spv.write(reinterpret_cast<const char*>(module.data()),
              static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    spv.close();
    EXPECT_TRUE(spv.good());
    std::ofstream bin(root / (stem + ".bin"), std::ios::binary);
    bin.write(reinterpret_cast<const char*>(p.code.data()),
              static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
    bin.close();
    EXPECT_TRUE(bin.good());
    EXPECT_TRUE(write_refused_compute_inputs(root / (stem + ".inputs"), 0, &p.table, p.config,
                                             "synthetic-mixed-f32-gpu-witness"));
    const auto in = inputs(shape);
    std::vector<uint32_t> data;
    for (const auto* values : {&in.s0, &in.s1, &in.old_d})
        data.insert(data.end(), values->begin(), values->end());
    for (bool active : in.active) data.push_back(active);
    data.insert(data.end(), in.s2.begin(), in.s2.end());
    std::vector<uint32_t> output;
    const auto dispatch =
        prosper::test::run_compute(module, std::vector<float>(64), 64, 64, {}, data, &output);
    EXPECT_EQ(dispatch.size(), 64u);
    if (output.size() != data.size()) return {};
    output.resize(in.lanes());
    return output;
}

uint32_t clamped(uint32_t value) {
    return std::bit_cast<uint32_t>(std::fmin(std::fmax(std::bit_cast<float>(value), 0.0f), 1.0f));
}
}   // namespace

TEST(ComputeMixedF32Fma, EveryGuestFormMatchesTheIndependentNativeControl) {
    const auto receipt = prosper::test::fma_rounding::device_receipt();
    ASSERT_FALSE(receipt.empty());
    RecordProperty("executing_driver", receipt);
    std::printf("MIXED_F32_FMA_EXECUTING_DRIVER %s\n", receipt.c_str());
    std::ofstream driver(artifact_root() / "driver.txt");
    driver << receipt << '\n';
    driver.close();
    ASSERT_TRUE(driver.good());
    const auto control = native({}, inputs());
    ASSERT_EQ(control.size(), 64u);
    EXPECT_EQ(control[1], 0x40600000u);   // 1.5*2+0.5 = 3.5, independent positive controls
    EXPECT_EQ(control[2], 0x3f800000u);   // -1*2+3 = 1
    std::size_t index = 0;
    for (const Shape shape : shapes()) {
        SCOPED_TRACE(std::to_string(index));
        const auto in = inputs(shape);
        const auto reference = native(shape, in);
        ASSERT_EQ(reference.size(), 64u);
        for (uint32_t wave : {32u, 64u}) {
            const auto actual = guest(
                shape, wave, "mapping_" + std::to_string(index) + "_wave" + std::to_string(wave));
            ASSERT_EQ(actual.size(), 64u);
            for (std::size_t lane = 0; lane < in.lanes(); ++lane) {
                const uint32_t active = shape.clamp ? clamped(reference[lane]) : reference[lane];
                EXPECT_EQ(actual[lane], in.active[lane] ? active : in.old_d[lane])
                    << "wave " << wave << " lane " << lane;
            }
        }
        ++index;
    }
    EXPECT_EQ(index, 84u);
}

TEST(ComputeMixedF32Fma, QualifiedDriverKeepsTheHandWorkedFusedWitnesses) {
    RecordProperty("executing_driver", prosper::test::fma_rounding::device_receipt());
    const auto full = native({}, inputs());
    const Shape mixed{2, 2};
    const auto half = native(mixed, inputs(mixed));
    ASSERT_EQ(full.size(), 64u);
    ASSERT_EQ(half.size(), 64u);
    RecordProperty("native_full_witness", std::to_string(full[4]));
    RecordProperty("native_mixed_witness", std::to_string(half[4]));
    if (full[4] != 0xa8800000u || half[4] != 0x33ffc000u)
        GTEST_SKIP() << "Independent native Fma lacks the selected strict RTE accuracy; "
                        "mapping coverage remains unconditional (#5225).";
    for (const Shape shape : {Shape{}, mixed}) {
        const auto in = inputs(shape);
        for (uint32_t wave : {32u, 64u}) {
            const auto actual =
                guest(shape, wave,
                      "strict_" + std::to_string(shape.half) + "_wave" + std::to_string(wave));
            ASSERT_EQ(actual.size(), 64u);
            for (std::size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(actual[lane], expected(shape, in, lane)) << "lane " << lane;
            EXPECT_EQ(actual[0], in.old_d[0]);
        }
    }
}
