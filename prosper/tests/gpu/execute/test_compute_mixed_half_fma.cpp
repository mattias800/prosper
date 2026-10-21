// #5225: execute exactly representable mixed-half residues against independent native Fma.
#include "fixtures/compute_runner.h"
#include "fixtures/f32_fma_device_receipt.hpp"
#include "fixtures/f32_fma_native_control.hpp"
#include "fixtures/mixed_f32_fma.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::mixed_f32;

TEST(ComputeMixedHalfFma, ExactResiduesAndIndependentNativeFmaControl) {
    const auto receipt = prosper::test::fma_rounding::device_receipt();
    ASSERT_FALSE(receipt.empty());
    std::printf("MIXED_HALF_FMA_DRIVER %s\n", receipt.c_str());
    RecordProperty("executing_driver", receipt);
    constexpr std::array<uint32_t, 4> halves{0xbc00, 0xc880, 0xce40, 0xd220};
    auto in = inputs();
    std::vector<uint32_t> control_data(256);
    for (std::size_t lane = 0; lane < in.lanes(); ++lane) {
        const float t = float(2 * (lane % 4) + 1);
        in.s0[lane] = std::bit_cast<uint32_t>(8192.0f + t);
        in.s1[lane] = std::bit_cast<uint32_t>(8192.0f - t);
        in.s2[lane] = 0xcc800000u;
        control_data[lane] = in.s0[lane];
        control_data[64 + lane] = in.s1[lane];
        control_data[128 + lane] = in.s2[lane];
    }
    std::vector<uint32_t> control_out;
    const auto control_dispatch =
        prosper::test::run_compute(prosper::test::fma_rounding::native_fma_control(),
                                   std::vector<float>(64), 64, 64, {}, control_data, &control_out);
    ASSERT_EQ(control_dispatch.size(), 64u);
    ASSERT_EQ(control_out.size(), 256u);
    for (std::size_t lane = 0; lane < in.lanes(); ++lane) {
        const float t = float(2 * (lane % 4) + 1);
        if (control_out[192 + lane] != std::bit_cast<uint32_t>(-t * t))
            GTEST_SKIP() << "Independent native Fma lacks the selected strict RTE accuracy; "
                            "CPU regression and instruction mapping remain unconditional.";
    }
    std::size_t wrong = 0;
    for (uint32_t opcode : {0x21u, 0x22u})
        for (uint32_t wave : {32u, 64u})
            for (bool empty : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "op=" << opcode << " wave=" << wave << " empty=" << empty);
                const auto p = program({}, wave, opcode);
                const auto module =
                    recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
                ASSERT_FALSE(module.empty());
                const auto stem =
                    "mixed-half-op" + std::to_string(opcode) + "-wave" + std::to_string(wave);
                std::ofstream spv(prosper_test::test_scratch_dir() / (stem + ".spv"),
                                  std::ios::binary);
                spv.write(reinterpret_cast<const char*>(module.data()),
                          std::streamsize(module.size() * sizeof(uint32_t)));
                spv.close();
                ASSERT_TRUE(spv.good());
                std::vector<uint32_t> data;
                for (const auto* source : {&in.s0, &in.s1, &in.old_d})
                    data.insert(data.end(), source->begin(), source->end());
                for (bool active : in.active) data.push_back(!empty && active);
                data.insert(data.end(), in.s2.begin(), in.s2.end());
                std::vector<uint32_t> actual;
                const auto dispatch = prosper::test::run_compute(module, std::vector<float>(64), 64,
                                                                 64, {}, data, &actual);
                ASSERT_EQ(dispatch.size(), 64u);
                ASSERT_EQ(actual.size(), data.size());
                for (std::size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto half = halves[lane % 4];
                    const auto written = opcode == 0x21 ? (in.old_d[lane] & 0xffff0000u) | half
                                                        : (in.old_d[lane] & 0xffffu) | (half << 16);
                    const auto expected = !empty && in.active[lane] ? written : in.old_d[lane];
                    wrong += actual[lane] != expected;
                    EXPECT_EQ(actual[lane], expected) << "lane " << lane;
                }
            }
    std::printf("MIXED_HALF_WRONG_LANES %zu\n", wrong);
}
