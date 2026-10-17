// Bind VDATA components at their old read sites; lowering owns no access or mask state.
#include "gpu/recompiler/ir/lower/buffer_store_values.hpp"
#include "gpu/recompiler/ir/buffer_store_cases.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace fixture = prosper::test::ir::buffer_store;
namespace {
using Row = std::array<uint32_t, 5>;
using Emit =
    std::function<std::vector<uint32_t>(SpirvCompute&, const Row&, const BufferStoreRead&)>;
std::vector<uint32_t> run(const std::vector<Row>& rows, uint32_t width, const Emit& emit,
                          std::vector<int>& reads) {
    SpirvCompute b;
    const auto n = static_cast<uint32_t>(rows.size());
    b.begin(5, nullptr, n, 1, 1, n, 0, true, true);
    Row ids{};
    for (uint32_t k = 0; k < 5; ++k) ids[k] = b.load_input(k);
    const BufferStoreRead read = [&](int reg) {
        reads.push_back(reg);
        EXPECT_GE(reg, 21);
        EXPECT_LT(reg, 25);
        return ids[static_cast<size_t>(reg - 21)];
    };
    const auto outputs = emit(b, ids, read);
    EXPECT_EQ(outputs.size(), width);
    for (uint32_t k = 0; k < outputs.size(); ++k) b.store_output_word(outputs[k], width, k);
    std::vector<uint32_t> input;
    input.reserve(rows.size() * 5);
    for (const auto& row : rows) input.insert(input.end(), row.begin(), row.end());
    bpermute_oracle::Interpreter vm(b.finish());
    vm.half_rte_preserve_denorm = true;
    auto out = vm.run_buffers(n, {{0, input}, {1, std::vector<uint32_t>(size_t{n} * width)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return out;
}
Rdna2Inst instruction() {
    Rdna2Inst in{};
    in.dst.value = 21;
    return in;
}
TEST(IrLowerBufferStoreValues, IntegerFieldsReadOnlyTheirComponentAndRetainByteResidues) {
    for (uint32_t bytes : {1u, 2u})
        for (uint32_t k = 0; k < 4; ++k) {
            std::vector<Row> rows;
            rows.reserve(64);
            for (uint32_t lane = 0; lane < 64; ++lane)
                rows.push_back({0x8192a3b4u ^ lane, 0xc5d6e7f8u ^ lane * 3, 0xabcdef17u ^ lane * 5,
                                0xfedcba98u ^ lane * 7,
                                lane < 4 ? 0xfffffffcu + lane : lane * 17u + 3});
            std::vector<int> reads;
            const uint32_t mask = bytes == 1 ? 255 : 65535;
            const auto out = run(
                rows, 3,
                [=](SpirvCompute& b, const Row& x, const auto& read) {
                    const auto result =
                        lower_buffer_store_fields(b, instruction(), x[4], k, bytes, mask, read);
                    return std::vector<uint32_t>{result.index, result.mask, result.value};
                },
                reads);
            ASSERT_EQ(out.size(), 192u);
            EXPECT_EQ(reads, (std::vector<int>{21 + static_cast<int>(k)}));
            for (uint32_t lane = 0; lane < 64; ++lane) {
                const uint32_t addr = rows[lane][4] + k * bytes, shift = (addr % 4) * 8;
                EXPECT_EQ(out[size_t{lane} * 3], addr / 4);
                EXPECT_EQ(out[size_t{lane} * 3 + 1], mask << shift);
                EXPECT_EQ(out[size_t{lane} * 3 + 2], (rows[lane][k] & mask) << shift);
            }
        }
}
TEST(IrLowerBufferStoreValues, RawDwordIndicesDoNotReadData) {
    for (uint32_t k = 0; k < 4; ++k) {
        std::vector<Row> rows;
        rows.reserve(64);
        for (uint32_t lane = 0; lane < 32; ++lane) rows.push_back({0, 0, 0, 0, 0xfffffff0u + lane});
        std::vector<int> reads;
        const auto out = run(
            rows, 1,
            [k](SpirvCompute& b, const Row& x, const auto&) {
                return std::vector<uint32_t>{lower_buffer_store_index(b, x[4], k)};
            },
            reads);
        ASSERT_EQ(out.size(), 32u);
        EXPECT_TRUE(reads.empty());
        for (uint32_t lane = 0; lane < 32; ++lane) EXPECT_EQ(out[lane], rows[lane][4] + k);
    }
}
TEST(IrLowerBufferStoreValues, PackedWordsHonorFormatFlagsAndReadOnlyPhysicalFields) {
    for (uint32_t f = 9; f < fixture::formats.size(); ++f) {
        const fixture::Case c{f, 64, 0, false};
        const auto& fmt = fixture::formats[f];
        std::vector<Row> rows;
        rows.reserve(64);
        for (uint32_t lane = 0; lane < c.wave; ++lane) {
            Row row{};
            for (uint32_t k = 0; k < 4; ++k) row[k] = fixture::input_word(c, lane, k);
            rows.push_back(row);
        }
        std::vector<int> reads;
        const auto out = run(
            rows, 1,
            [=](SpirvCompute& b, const auto&, const auto& read) {
                return std::vector<uint32_t>{lower_buffer_store_packed_word(
                    b, instruction(), 4, fmt.components, f == 13, f == 9, f == 10, f == 12, read)};
            },
            reads);
        ASSERT_EQ(out.size(), 64u);
        std::vector<int> expected_reads;
        expected_reads.reserve(4);
        for (uint32_t k = 0; k < fmt.components; ++k)
            expected_reads.push_back(21 + static_cast<int>(k));
        EXPECT_EQ(reads, expected_reads);
        for (uint32_t lane = 0; lane < c.wave; ++lane) {
            uint32_t want = 0;
            for (uint32_t k = 0; k < fmt.components; ++k)
                want |= fixture::field(c, lane, k) << (f == 13 ? k * 11 : k * 10);
            EXPECT_EQ(out[lane], want);
        }
    }
}
TEST(IrLowerBufferStoreValues, HalfAndNormalizedDwordsKeepReadOrderAndIndex) {
    for (uint32_t f : {4u, 5u, 6u, 7u, 8u})
        for (uint32_t d = 0; d < (4 * fixture::formats[f].bytes + 3) / 4; ++d) {
            const fixture::Case c{f, 64, 0, false};
            const auto fmt = fixture::formats[f];
            std::vector<Row> rows;
            rows.reserve(64);
            for (uint32_t lane = 0; lane < c.wave; ++lane) {
                Row row{};
                for (uint32_t k = 0; k < 4; ++k) row[k] = fixture::input_word(c, lane, k);
                row[4] = 0xfffffffeu + lane;
                rows.push_back(row);
            }
            std::vector<int> reads;
            const auto out = run(
                rows, 2,
                [=](SpirvCompute& b, const Row& x, const auto& read) {
                    const auto result = lower_buffer_store_packed_dword(
                        b, instruction(), x[4], d, 4, fmt.bytes, f == 4, f == 6 || f == 8,
                        f == 5   ? 255.0f
                        : f == 6 ? 127.0f
                        : f == 7 ? 65535.0f
                                 : 32767.0f,
                        read);
                    return std::vector<uint32_t>{result.index, result.value};
                },
                reads);
            ASSERT_EQ(out.size(), 128u);
            std::vector<int> expected_reads;
            expected_reads.reserve(4);
            for (uint32_t k = 0; k < 4; ++k)
                if (k * fmt.bytes / 4 == d) expected_reads.push_back(21 + static_cast<int>(k));
            EXPECT_EQ(reads, expected_reads);
            for (uint32_t lane = 0; lane < c.wave; ++lane) {
                uint32_t want = 0;
                for (uint32_t k = 0; k < 4; ++k)
                    if (k * fmt.bytes / 4 == d)
                        want |= fixture::field(c, lane, k) << ((k * fmt.bytes % 4) * 8);
                EXPECT_EQ(out[size_t{lane} * 2], rows[lane][4] + d);
                EXPECT_EQ(out[size_t{lane} * 2 + 1], want);
            }
        }
}
} // namespace
