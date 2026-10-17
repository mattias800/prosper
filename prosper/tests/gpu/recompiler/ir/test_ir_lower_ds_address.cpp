// RDNA2 DS wrappers preserve the guest source read and the byte-to-dword/offset arithmetic order.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/lower/ds_address_lowering.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {
using Emit = std::function<std::vector<uint32_t>(SpirvCompute&, uint32_t)>;
using Read = std::function<uint32_t(int)>;
using PairEmit = std::function<std::vector<uint32_t>(SpirvCompute&, const Read&)>;

std::vector<uint32_t> run(const std::vector<uint32_t>& inputs, uint32_t outputs, const Emit& emit) {
    const auto lanes = static_cast<uint32_t>(inputs.size());
    SpirvCompute b;
    b.begin(1, nullptr, lanes, 1, 1, lanes, 0, true, true);
    const uint32_t source = b.load_input(0);
    const auto ids = emit(b, source);
    EXPECT_EQ(ids.size(), outputs);
    for (uint32_t k = 0; k < ids.size(); ++k) b.store_output_word(ids[k], outputs, k);
    bpermute_oracle::Interpreter vm(b.finish());
    const auto out = vm.run_buffers(
        lanes, {{0, inputs}, {1, std::vector<uint32_t>(size_t{lanes} * outputs)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return out;
}

std::vector<uint32_t> lanes() {
    std::vector<uint32_t> out;
    out.reserve(64);
    for (uint32_t lane = 0; lane < 64; ++lane)
        out.push_back(lane < 8 ? std::array{0u, 1u, 3u, 4u, 0xfffffffcu, 0xfffffffdu, 0xfffffffeu,
                                            0xffffffffu}[lane]
                               : (0x81234567u ^ lane * 0x01020305u));
    return out;
}

std::vector<uint32_t> pair_result(const std::vector<uint32_t>& input, const char* name,
                                  const PairEmit& emit, const std::array<uint32_t, 2>& offsets) {
    std::vector<int> reads;
    const auto out = run(input, 2, [&](SpirvCompute& b, uint32_t source) {
        const Read read = [&](int reg) {
            reads.push_back(reg);
            return source;
        };
        return emit(b, read);
    });
    EXPECT_EQ(reads, (std::vector<int>{27})) << name;
    EXPECT_EQ(out.size(), size_t{input.size()} * 2) << name;
    for (size_t lane = 0; lane < input.size(); ++lane) {
        const uint32_t base = input[lane] / 4u;
        EXPECT_EQ(out[size_t{lane} * 2], base + offsets[0]) << name << " lane=" << lane;
        EXPECT_EQ(out[size_t{lane} * 2 + 1], base + offsets[1]) << name << " lane=" << lane;
    }
    return out;
}

TEST(IrLowerDsAddress, ByteAddressReadsOnlyAddressRegisterAndHonorsGdsMask) {
    for (uint32_t wave : {32u, 64u})
        for (bool gds : {false, true}) {
            Rdna2Inst in{};
            in.src[0].value = 27;
            in.literal = 0xffffu;
            in.ds_gds = gds;
            std::vector<int> reads;
            const auto input = lanes();
            const auto out =
                run(std::vector<uint32_t>(input.begin(), input.begin() + wave), 1,
                    [&](SpirvCompute& b, uint32_t source) {
                        const Read read = [&](int reg) {
                            reads.push_back(reg);
                            return source;
                        };
                        return std::vector<uint32_t>{lower_ds_byte_address(b, in, read)};
                    });
            EXPECT_EQ(reads, std::vector<int>{27});
            ASSERT_EQ(out.size(), wave);
            for (uint32_t lane = 0; lane < wave; ++lane) {
                uint32_t sum = input[lane] + in.literal;
                if (gds) sum &= 0xffffu;
                EXPECT_EQ(out[lane], sum / 4u) << "wave=" << wave << " lane=" << lane;
            }
        }
}

TEST(IrLowerDsAddress, PairedWrappersRetainOffsetUnitsAndSourceReadOrder) {
    const auto input = lanes();
    Rdna2Inst in{};
    in.src[0].value = 27;
    in.literal = 0x0503u;
    pair_result(input, "write2-b32",
                [in](SpirvCompute& b, const Read& read) {
                    const auto pair = lower_ds_write2_b32_address(b, in, read);
                    return std::vector<uint32_t>{pair.first, pair.second};
                },
                {3, 5});
    pair_result(input, "write2-st64-b32",
                [in](SpirvCompute& b, const Read& read) {
                    const auto pair = lower_ds_write2st64_b32_address(b, in, read, 192, 320);
                    return std::vector<uint32_t>{pair.first, pair.second};
                },
                {192, 320});
    pair_result(input, "write2-b64",
                [in](SpirvCompute& b, const Read& read) {
                    const auto pair = lower_ds_write2_b64_address(b, in, read);
                    return std::vector<uint32_t>{pair.first, pair.second};
                },
                {6, 10});
    pair_result(input, "read2-b32",
                [in](SpirvCompute& b, const Read& read) {
                    const auto pair = lower_ds_read2_b32_address(b, in, read);
                    return std::vector<uint32_t>{pair[0], pair[1]};
                },
                {3, 5});
    pair_result(input, "read2-st64-b32",
                [in](SpirvCompute& b, const Read& read) {
                    const auto pair = lower_ds_read2st64_b32_address(b, in, read, 192, 320);
                    return std::vector<uint32_t>{pair[0], pair[1]};
                },
                {192, 320});
}

TEST(IrLowerDsAddress, Read2B64BuildsBothAdjacentWordsAndEqualOffsetsStayEqual) {
    for (uint32_t wave : {32u, 64u}) {
        Rdna2Inst in{};
        in.src[0].value = 27;
        in.literal = 0x0503u;
        const auto input = lanes();
        std::vector<int> reads;
        const auto out = run(std::vector<uint32_t>(input.begin(), input.begin() + wave), 4,
                             [&](SpirvCompute& b, uint32_t source) {
                                 const Read read = [&](int reg) {
                                     reads.push_back(reg);
                                     return source;
                                 };
                                 const auto values = lower_ds_read2_b64_address(b, in, read);
                                 return std::vector<uint32_t>(values.begin(), values.end());
                             });
        EXPECT_EQ(reads, std::vector<int>{27});
        ASSERT_EQ(out.size(), size_t{wave} * 4);
        for (uint32_t lane = 0; lane < wave; ++lane) {
            const uint32_t base = input[lane] / 4u;
            const std::array<uint32_t, 4> expected{base + 6u, base + 7u, base + 10u, base + 11u};
            for (size_t k = 0; k < expected.size(); ++k)
                EXPECT_EQ(out[size_t{lane} * 4 + k], expected[k]);
        }

        in.literal = 0x0202u;
        reads.clear();
        const auto equal = run(std::vector<uint32_t>(input.begin(), input.begin() + wave), 4,
                               [&](SpirvCompute& b, uint32_t source) {
                                   const Read read = [&](int reg) {
                                       reads.push_back(reg);
                                       return source;
                                   };
                                   const auto values = lower_ds_read2_b64_address(b, in, read);
                                   return std::vector<uint32_t>(values.begin(), values.end());
                               });
        EXPECT_EQ(reads, std::vector<int>{27});
        for (uint32_t lane = 0; lane < wave; ++lane) {
            const uint32_t base = input[lane] / 4u;
            const std::array<uint32_t, 4> expected{base + 4u, base + 5u, base + 4u, base + 5u};
            for (size_t k = 0; k < expected.size(); ++k)
                EXPECT_EQ(equal[size_t{lane} * 4 + k], expected[k]);
        }
    }
}

TEST(IrLowerDsAddress, WordIndexWrappersDoNotReadGuestRegistersAndWrap) {
    const std::vector<uint32_t> input{0u, 1u, 0xfffffffeu, 0xffffffffu};
    for (uint32_t component : {0u, 1u, 2u, 3u, 0xffffffffu})
        for (bool store : {false, true}) {
            const int k = component == 0xffffffffu ? -1 : static_cast<int>(component);
            const auto out = run(input, 1, [=](SpirvCompute& b, uint32_t index) {
                return std::vector<uint32_t>{store ? lower_ds_store_word_index(b, index, k)
                                                   : lower_ds_read_word_index(b, index, k)};
            });
            for (size_t lane = 0; lane < input.size(); ++lane)
                EXPECT_EQ(out[lane], input[lane] + component);
        }
}

} // namespace
