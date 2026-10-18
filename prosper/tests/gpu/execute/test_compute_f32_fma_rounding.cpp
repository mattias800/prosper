// #5225: execute actual guest GLSL Fma modules. This proves this driver's finite RTE
// witness only; Vulkan allows Mul+Add precision and the old RADV graph also contracted.
// #5331: keep guest-to-native mapping active when an independent driver control lacks
// strict fused accuracy; qualify only the strict witness and retain the #5225 gap.
#include "fixtures/f32_fma_rounding.hpp"
#include "fixtures/f32_fma_native_control.hpp"
#include "fixtures/f32_fma_device_receipt.hpp"
#include "fixtures/f32_mad_rounding.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <vulkan/vulkan.h>
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::fma_rounding;
namespace {
std::vector<uint32_t> native_fma_inputs(const Encoding& encoding, const LaneInputs& in) {
    std::vector<uint32_t> data(4 * in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const auto source = prosper::test::mad_rounding::sources(encoding, {}, in, lane);
        for (size_t k = 0; k < source.size(); ++k) data[k * in.lanes() + lane] = source[k];
    }
    return data;
}

std::vector<uint32_t> native_fma_results(const Encoding& encoding, const LaneInputs& in) {
    const auto data = native_fma_inputs(encoding, in);
    std::vector<uint32_t> result;
    const auto dispatch = prosper::test::run_compute(native_fma_control(), std::vector<float>(64),
                                                     64, 64, {}, data, &result);
    EXPECT_EQ(dispatch.size(), 64u);
    if (result.size() != data.size()) return {};
    return {result.begin() + 192, result.end()};
}

std::filesystem::path artifact_root() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_F32_FMA_GPU_ARTIFACTS");
    const auto root = configured ? std::filesystem::path(configured)
                                 : prosper_test::test_scratch_dir() / "f32-fma-gpu";
    std::filesystem::create_directories(root);
    return root;
}
std::vector<uint32_t> gpu_run(const Encoding& e, uint32_t wave,
                              const LaneInputs* independent_inputs = nullptr,
                              bool legacy_prefix = false) {
    auto p = program(e, {}, wave, 0);
    if (e.format == Rdna2Format::VOP3 && e.opcode == 0x12b && !legacy_prefix) {
        // Use the exact LLVM gfx1030 spelling, with unused SRC2 zero and implicit oldD.
        const auto instruction = std::find(p.code.begin(), p.code.end(), 0xd12b0001u);
        EXPECT_NE(instruction, p.code.end());
        if (instruction == p.code.end()) return {};
        *instruction = 0xd52b0001u;
    }
    const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
    EXPECT_FALSE(module.empty());
    if (module.empty()) return {};
    const auto stem = std::string(e.name) + "_wave" + std::to_string(wave) +
                      (independent_inputs ? "_distinct" : "") +
                      (legacy_prefix ? "_legacy-prefix" : "");
    const auto root = artifact_root();
    std::ofstream spv(root / (stem + ".spv"), std::ios::binary);
    spv.write(reinterpret_cast<const char*>(module.data()),
              static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    spv.close();
    if (!spv.good()) return {};
    std::ofstream guest(root / (stem + ".bin"), std::ios::binary);
    guest.write(reinterpret_cast<const char*>(p.code.data()),
                static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
    guest.close();
    if (!guest.good() || !write_refused_compute_inputs(root / (stem + ".inputs"), 0, &p.table,
                                                       p.config, "synthetic-f32-fma-gpu-witness"))
        return {};
    const auto in = independent_inputs ? *independent_inputs : inputs(&e);
    std::vector<uint32_t> output;
    const auto dispatch =
        prosper::test::run_compute(module, std::vector<float>(64), 64, 64, {}, buffer(in), &output);
    EXPECT_EQ(dispatch.size(), 64u) << "dispatch, fence, and host availability must complete";
    output.resize(std::min(output.size(), size_t(64)));
    return output;
}
} // namespace
TEST(ComputeF32FmaRounding, IndependentControlKeepsTheHandWorkedFusedWitness) {
    const auto data = native_fma_inputs(kEncodings.front(), inputs(&kEncodings.front()));
    const auto module = native_fma_control();
    std::ofstream saved(artifact_root() / "native-fma-control.spv", std::ios::binary);
    saved.write(reinterpret_cast<const char*>(module.data()),
                static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    saved.close();
    ASSERT_TRUE(saved.good());
    bpermute_oracle::Interpreter interpreter(module);
    const auto output = interpreter.run_buffers(64, {{3, data}}, 3);
    ASSERT_TRUE(interpreter.error.empty()) << interpreter.error;
    ASSERT_EQ(output.size(), data.size());
    // (1+2^-23)*(1-2^-23)-1 = -2^-46, independent of the guest IR evaluator.
    EXPECT_EQ(output[3 * 64 + 4], 0xa8800000u);
    EXPECT_TRUE(std::equal(data.begin(), data.begin() + 192, output.begin()));
}

TEST(ComputeF32FmaRounding, FourFusedGuestGraphsMatchTheFiniteRteReferenceOnTheExecutingDriver) {
    const auto receipt = device_receipt();
    ASSERT_FALSE(receipt.empty()) << "a real Vulkan dispatch is required for this witness";
    RecordProperty("executing_driver", receipt);
    std::printf("F32_FMA_EXECUTING_DRIVER %s\n", receipt.c_str());
    std::ofstream driver(artifact_root() / "driver.txt");
    driver << receipt << '\n';
    driver.close();
    ASSERT_TRUE(driver.good());
    const auto control = native_fma_results(kEncodings.front(), inputs(&kEncodings.front()));
    ASSERT_EQ(control.size(), 64u);
    RecordProperty("native_fma_witness", control[4]);
    if (control[4] == 0)
        GTEST_SKIP() << "Independent native GLSL Fma control also rounds separately; "
                        "strict guest accuracy is unsupported here (#5225). "
                     << receipt;
    ASSERT_EQ(control[4], 0xa8800000u) << "unexpected native-control accuracy";
    for (const auto& e : kEncodings)
        for (uint32_t wave : {32u, 64u}) {
            SCOPED_TRACE(std::string(e.name) + " wave " + std::to_string(wave));
            const auto output = gpu_run(e, wave);
            const auto in = inputs(&e);
            ASSERT_EQ(output.size(), 64u);
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(output[lane], fused_expected(e, {}, in, lane, 0)) << "lane " << lane;
            EXPECT_EQ(output[4], 0xa8800000u);
            EXPECT_EQ(output[0], in.old_d[0]) << "inactive witness retains old VDST";
        }
}

TEST(ComputeF32FmaRounding, EveryGuestFormMatchesAnIndependentNativeFmaControl) {
    for (const auto& encoding : kEncodings) {
        auto in = inputs(&encoding);
        // Distinguish accumulator, SRC1 and SRC2. The cancellation dataset deliberately
        // aliases these values, so it cannot detect selecting the wrong addend register.
        for (size_t lane = 0; lane < in.lanes(); ++lane) {
            in.old_d[lane] = lane % 2 ? 0xc0500000u : 0x40500000u; // +/-3.25
            in.s2[lane] = lane % 2 ? 0xbf400000u : 0x3f400000u; // +/-0.75
        }
        const auto reference = native_fma_results(encoding, in);
        ASSERT_EQ(reference.size(), 64u);
        for (uint32_t wave : {32u, 64u}) {
            SCOPED_TRACE(std::string(encoding.name) + " wave " + std::to_string(wave));
            const auto output = gpu_run(encoding, wave, &in);
            ASSERT_EQ(output.size(), 64u);
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(output[lane], in.active[lane] ? reference[lane] : in.old_d[lane])
                    << "lane " << lane;
        }
    }
}
TEST(ComputeF32FmaRounding, CanonicalE64FmacAlwaysMatchesTheIndependentNativeControl) {
    constexpr Encoding e{"fmac_e64", Rdna2Format::VOP3, 0x12b, Addend::Destination, false};
    auto in = inputs(&e);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.old_d[lane] = lane % 2 ? 0xc0500000u : 0x40500000u;
        in.s2[lane] = lane % 2 ? 0xbf400000u : 0x3f400000u;
    }
    const auto reference = native_fma_results(e, in);
    ASSERT_EQ(reference.size(), 64u);
    for (bool legacy_prefix : {false, true})
        for (uint32_t wave : {32u, 64u}) {
            const auto output = gpu_run(e, wave, &in, legacy_prefix);
            ASSERT_EQ(output.size(), 64u);
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(output[lane], in.active[lane] ? reference[lane] : in.old_d[lane])
                    << "wave " << wave << " lane " << lane;
        }
}
TEST(ComputeF32FmaRounding, CanonicalE64FmacKeepsTheQualifiedFiniteRteWitness) {
    constexpr Encoding e{"fmac_e64", Rdna2Format::VOP3, 0x12b, Addend::Destination, false};
    const auto receipt = device_receipt();
    ASSERT_FALSE(receipt.empty());
    RecordProperty("executing_driver", receipt);
    const auto in = inputs(&e);
    const auto control = native_fma_results(e, in);
    ASSERT_EQ(control.size(), 64u);
    RecordProperty("native_fma_witness", control[4]);
    if (control[4] == 0)
        GTEST_SKIP() << "Independent native Fma also rounds separately; strict guest "
                        "accuracy remains unsupported (#5225). "
                     << receipt;
    ASSERT_EQ(control[4], 0xa8800000u) << receipt;
    for (bool legacy_prefix : {false, true})
        for (uint32_t wave : {32u, 64u}) {
            const auto output = gpu_run(e, wave, nullptr, legacy_prefix);
            ASSERT_EQ(output.size(), 64u);
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(output[lane], fused_expected(e, {}, in, lane, 0)) << lane;
            EXPECT_EQ(output[4], 0xa8800000u);
            EXPECT_EQ(output[0], in.old_d[0]);
        }
}
