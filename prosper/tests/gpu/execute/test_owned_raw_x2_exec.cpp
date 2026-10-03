// Execute the owned pair through scalar multiplication, partial VCC reuse and a fresh mask.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>
using namespace prosper::gpu;
TEST(OwnedRawX2Exec, NumericPairAndFreshPredicate) {
    const std::vector<uint32_t> original = {0xf4041a80u, 0xfa000004u, 0xf4041ab5u, 0xfa000038u,
                                            0x93046b6au, 0x906a8404u, 0x7e000c6au,
                                            0x7c020300u, // fresh cmp_lt_f32 v0,v1
                                            0x02020280u, // v_cndmask_b32 v1,0,v1,VCC
                                            0xbf810000u};
    for (bool prior_mask : {false, true})
        for (auto words : {std::array<uint32_t, 2>{60, 68}, {7, 11}, {3, 5}}) {
            auto code = original;
            if (prior_mask) code.insert(code.begin(), 0x7c040100u); // v_cmp_eq_f32 vcc,v0,v0
            std::array<uint32_t, 2> pointer{0x200000, 0x20};
            ShaderResourceTable table;
            for (uint32_t k = 0; k < 2; ++k) {
                ShaderResource r;
                r.cls = ResourceClass::ConstantBuffer;
                r.format = DataFormat::Uint32;
                r.num_components = 1;
                r.fetch_pc = k * 2 + uint32_t(prior_mask);
                r.binding = k + 2;
                r.size = 8;
                r.gpu_addr = k ? 0x2000200038ull : 0x2000100004ull;
                r.host_data = reinterpret_cast<uint8_t*>(k ? words.data() : pointer.data());
                r.host_data_size = 8;
                r.owned_nested_snapshot_bytes = 8;
                table.resources.push_back(r);
            }
            const uint32_t value = (words[0] * words[1]) >> 4;
            for (uint32_t threshold : {value ? value - 1 : 0, value + 1}) {
                // A full wave, with two interleaved input channels and one selected output per lane.
                std::vector<float> input(128);
                for (uint32_t lane = 0; lane < 64; ++lane) input[lane * 2 + 1] = float(threshold);
                for (uint32_t output : {0u, 1u}) {
                    const auto spv = recompile_valu(code.data(), code.size(), 2, output, &table);
                    ASSERT_FALSE(spv.empty());
                    const auto result = prosper::test::run_compute(
                        spv, input, 64, 64, {pointer[0], pointer[1]}, {words[0], words[1]});
                    ASSERT_EQ(result.size(), 64u);
                    for (float actual : result)
                        EXPECT_FLOAT_EQ(actual, output == 0         ? float(value)
                                                : value < threshold ? float(threshold)
                                                                    : 0.0f);
                }
            }
        }
}
