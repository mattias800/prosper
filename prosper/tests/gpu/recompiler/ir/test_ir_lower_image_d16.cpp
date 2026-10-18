// Selected components compact in DMASK order; the caller's callback stays between output words.
// Gather retains its separate one-PackHalf2x16 path with exact component pairing.
#include "gpu/recompiler/ir/lower/image_d16_lowering.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace {
uint32_t packs(const std::vector<uint32_t>& code) {
    uint32_t count = 0;
    for (size_t at = 0; at < code.size(); at += code[at] >> 16)
        if ((code[at] & 0xffffu) == Op_ExtInst && code[at + 4] == Glsl_PackHalf2x16) ++count;
    return count;
}
TEST(IrLowerImageD16, SelectedComponentsAndOddHighHalvesStayInOrder) {
    constexpr std::array<uint32_t, 4> source{0x3f800000, 0xc0000000, 0x3f000000, 0x40800000};
    constexpr std::array<uint32_t, 4> halves{0x3c00, 0xc000, 0x3800, 0x4400};
    for (uint32_t mask = 1; mask < 16; ++mask) {
        SpirvCompute b;
        b.begin(4, nullptr, 1, 1, 1, 64, 0, true, true);
        std::array<uint32_t, 4> ids{};
        for (uint32_t c = 0; c < 4; ++c) ids[c] = b.load_input(c);
        Rdna2Inst in{};
        in.mimg_dmask = mask;
        uint32_t writes = 0;
        lower_image_d16_results(b, in, ids.data(), [&](uint32_t value) {
            EXPECT_EQ(packs(b.code), std::min(2u * (writes + 1u), uint32_t(std::popcount(mask))))
                << "packing the next word must follow this callback";
            b.store_output_word(value, 2, writes++);
        });
        EXPECT_EQ(writes, (std::popcount(mask) + 1u) / 2u);
        std::vector<uint32_t> selected;
        for (uint32_t c = 0; c < 4; ++c)
            if (mask & (1u << c)) selected.push_back(halves[c]);
        std::vector<uint32_t> expected(2, 0xdeadbeefu);
        for (size_t at = 0; at < selected.size(); at += 2)
            expected[at / 2] =
                selected[at] | (at + 1 < selected.size() ? selected[at + 1] << 16 : 0);
        bpermute_oracle::Interpreter vm(b.finish());
        vm.half_rte_preserve_denorm = true;
        const auto result =
            vm.run_buffers(1,
                           {{0, std::vector<uint32_t>(source.begin(), source.end())},
                            {1, {0xdeadbeefu, 0xdeadbeefu}}},
                           1);
        EXPECT_TRUE(vm.error.empty()) << vm.error;
        EXPECT_EQ(result, expected) << "mask " << mask;
    }
}
TEST(IrLowerImageD16, GatherPairsStayDistinctAndOrdered) {
    SpirvCompute b;
    b.begin(4, nullptr, 1, 1, 1, 64, 0, true, true);
    std::array<uint32_t, 4> ids{};
    for (uint32_t c = 0; c < 4; ++c) ids[c] = b.load_input(c);
    for (uint32_t word = 0; word < 2; ++word)
        b.store_output_word(lower_image_d16_gather_word(b, ids.data(), word), 2, word);
    bpermute_oracle::Interpreter vm(b.finish());
    vm.half_rte_preserve_denorm = true;
    const auto result =
        vm.run_buffers(1, {{0, {0x3f800000, 0xc0000000, 0x3f000000, 0x40800000}}, {1, {0, 0}}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(result, (std::vector<uint32_t>{0xc0003c00u, 0x44003800u}));
}
} // namespace
