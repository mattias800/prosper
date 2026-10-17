// The out-of-range contract of the MUBUF address (ir/ops/buffer_address.cpp) on real Vulkan:
// RDNA2 returns 0 for a buffer load past the V#'s range, and prosper delegates that ENTIRELY to
// robustBufferAccess over the bound descriptor range (test_traversal_chase.cpp, #2795). The
// address definition must therefore produce the guest's address unclamped, so a lane whose
// address leaves the buffer reads 0 while its neighbours read data.
//
//   v_cvt_u32_f32 v1, v1                                ; index = input[1]
//   buffer_load_dword v2, v1, s[0:3], 0 idxen offset:4  ; byte 4 + index * 4: links[index + 1]
//   s_waitcnt vmcnt(0)
//   v_cvt_f32_u32 v2, v2
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLanes = 64;
const uint32_t kProgram[] = {
    0x7e020f01u,               // v_cvt_u32_f32 v1, v1
    0xe0302004u, 0x80000201u,  // buffer_load_dword v2, v1, s[0:3], 0 idxen offset:4
    0xbf8c3f70u,               // s_waitcnt vmcnt(0)
    0x7e040d02u,               // v_cvt_f32_u32 v2, v2
    0xbf810000u,               // s_endpgm
};

TEST(IrBufferAddressRobustness, AnAddressPastTheRangeReadsZero) {
    ShaderResourceTable table;
    ShaderResource words{};
    words.cls = ResourceClass::ConstantBuffer;
    words.format = DataFormat::Uint32;
    words.num_components = 1;
    words.binding = 2;            // compute_runner binds `cbuf` here
    words.stride = 4;
    words.size = kLanes * 4;      // NUM_RECORDS * STRIDE
    words.sgpr_base = 0;
    table.resources.push_back(words);
    const auto spv = recompile_valu(kProgram, std::size(kProgram), 2, 2, &table);
    ASSERT_FALSE(spv.empty());

    std::vector<uint32_t> data(kLanes);
    for (uint32_t i = 0; i < kLanes; ++i) data[i] = 3u * i + 7u;   // never 0
    // Lanes 0..31 ask for index l and read data[l + 1]. Lanes 32..62 ask for l + 1000, far past
    // the range; lane 63 asks for 63, whose address (byte 256) is the first byte past it.
    std::vector<float> input(kLanes * 2, 0.0f);
    std::vector<float> want(kLanes);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t index = lane < 32 ? lane : lane + 1000u;
        input[lane * 2 + 1] = static_cast<float>(index);
        want[lane] = index + 1 < kLanes ? static_cast<float>(data[index + 1]) : 0.0f;
    }
    input[63 * 2 + 1] = 63.0f;
    want[63] = 0.0f;

    const std::vector<float> got = prosper::test::run_compute(spv, input, kLanes, kLanes, data);
    ASSERT_EQ(got.size(), kLanes) << "the module runs on Vulkan";
    uint32_t in_range = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
        EXPECT_EQ(got[lane], want[lane]) << "lane " << lane;
        in_range += want[lane] != 0.0f;
    }
    EXPECT_EQ(in_range, 32u) << "half the lanes read data, so a load that always returned 0 fails";
}

}   // namespace
