// The MUBUF/MTBUF address definition (ir/ops/buffer_address.cpp): ISA values worked by hand,
// the input order each form reads, and every form's SPIR-V run per lane in the integer
// interpreter against its own evaluation.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/buffer_address.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {

using In = BufferAddressInput;
constexpr BufferAddressShape faithful(uint32_t offset, uint32_t stride, bool index, bool voffset) {
    return {BufferAddressForm::Faithful, offset, stride, index, voffset};
}
constexpr BufferAddressShape kFoldedElement{BufferAddressForm::FoldedElement, 0, 12, false, false};
constexpr BufferAddressShape kFoldedZero{BufferAddressForm::FoldedZero, 0, 0, false, false};

// words: Index, Voffset, Soffset, Element.
std::array<Known, 2> eval(const BufferAddressShape& shape, uint32_t index, uint32_t voffset,
                          uint32_t soffset, uint32_t element = 0) {
    return eval_buffer_address(shape, {index, voffset, soffset, element});
}

TEST(IrBufferAddress, HandWorkedAddresses) {
    // OFFSET + VADDR[0] * STRIDE + VADDR[1] + SOFFSET = 12 + 3 * 16 + 8 + 20 = 88, dword 22.
    EXPECT_EQ(eval(faithful(12, 16, true, true), 3, 8, 20), (std::array<Known, 2>{88u, 22u}));
    // Only OFFSET and SOFFSET: 4095 + 1 = 4096, dword 1024.
    EXPECT_EQ(eval(faithful(4095, 0, false, false), 99, 99, 1), (std::array<Known, 2>{4096u, 1024u}));
    // OFFEN alone: a byte offset; the dword index truncates (7 >> 2 = 1).
    EXPECT_EQ(eval(faithful(0, 0, false, true), 99, 7, 0), (std::array<Known, 2>{7u, 1u}));
    // IDXEN alone, stride 4: 8 + 5 * 4 + 0 = 28, dword 7.
    EXPECT_EQ(eval(faithful(8, 4, true, false), 5, 99, 0), (std::array<Known, 2>{28u, 7u}));
    // The sum is modulo 2^32: 4 + 0x20000000 * 8 (= 2^32) + 0xfffffff0 + 0x10 = 2^33 + 4 -> 4.
    EXPECT_EQ(eval(faithful(4, 8, true, true), 0x20000000u, 0xfffffff0u, 0x10u),
              (std::array<Known, 2>{4u, 1u}));
    // A folded per-vertex fetch: ELEMENT * STRIDE = 5 * 12 = 60, dword 15; OFFSET/SOFFSET unread.
    EXPECT_EQ(eval(kFoldedElement, 99, 99, 99, 5), (std::array<Known, 2>{60u, 15u}));
    // A folded zero-stride fetch: no dynamic term remains.
    EXPECT_EQ(eval(kFoldedZero, 99, 99, 99, 99), (std::array<Known, 2>{0u, 0u}));
}

TEST(IrBufferAddress, EachFormReadsItsOperandsInTheOldOrder) {
    using V = std::vector<In>;
    EXPECT_EQ(build_buffer_address(faithful(0, 4, true, true)).inputs,
              (V{In::Index, In::Voffset, In::Soffset}));
    EXPECT_EQ(build_buffer_address(faithful(0, 4, true, false)).inputs, (V{In::Index, In::Soffset}));
    EXPECT_EQ(build_buffer_address(faithful(0, 0, false, true)).inputs, (V{In::Voffset, In::Soffset}));
    EXPECT_EQ(build_buffer_address(faithful(0, 0, false, false)).inputs, (V{In::Soffset}));
    EXPECT_EQ(build_buffer_address(kFoldedElement).inputs, (V{In::Element}));
    EXPECT_TRUE(build_buffer_address(kFoldedZero).inputs.empty());
}

TEST(IrBufferAddress, AnUnknownWordLeavesTheAddressUnknown) {
    const auto values = eval_buffer_address(faithful(4, 4, true, false), {3u, {}, std::nullopt, {}});
    EXPECT_FALSE(values[0].has_value());
    EXPECT_FALSE(values[1].has_value());
    // A word the form does not read cannot make it unknown.
    const auto zero = eval_buffer_address(kFoldedZero, {});
    EXPECT_EQ(zero, (std::array<Known, 2>{0u, 0u}));
}

// Per lane: the shape's Block emitted through SpirvCompute with every input loaded from the input
// buffer (so nothing folds), run in the interpreter; each lane stores {addr, idx}.
std::vector<uint32_t> run_block(const BufferAddressShape& shape,
                                const std::vector<std::array<uint32_t, 4>>& lanes) {
    const BufferAddressBlock built = build_buffer_address(shape);
    const uint32_t n = static_cast<uint32_t>(lanes.size());
    const uint32_t slots = built.block.input_count();
    SpirvCompute b;
    static uint64_t next_address = 0x5b0a0000u;
    b.diagnostic = {RecompileDiagnosticStage::Compute, next_address++};
    b.begin(slots == 0 ? 1u : slots, nullptr, n, 1u, 1u, n, 0u, true, true);
    std::vector<uint32_t> ids;
    for (uint32_t slot = 0; slot < slots; ++slot) ids.push_back(b.load_input(slot));
    const auto emitted = emit_spirv(b, built.block, ids);
    b.store_output_word(emitted[built.addr.index], 2u, 0u);
    b.store_output_word(emitted[built.idx.index], 2u, 1u);
    std::vector<uint32_t> buffer;
    for (const auto& lane : lanes)
        for (uint32_t slot = 0; slot < (slots == 0 ? 1u : slots); ++slot)
            buffer.push_back(slots == 0 ? 0u : lane[static_cast<size_t>(built.inputs[slot])]);
    bpermute_oracle::Interpreter vm(b.finish());
    auto out = vm.run_buffers(n, {{0u, buffer}, {1u, std::vector<uint32_t>(2 * n, 0u)}}, 1u);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    out.resize(2 * n);
    return out;
}

TEST(IrBufferAddress, EverySpirvFormMatchesItsEvaluationPerLane) {
    std::vector<std::array<uint32_t, 4>> lanes;
    uint32_t seed = 0x5b0au;
    const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
    const uint32_t edges[] = {0u, 1u, 3u, 4u, 0x7fffffffu, 0x80000000u, 0xfffffffcu, 0xffffffffu};
    for (uint32_t lane = 0; lane < 64; ++lane) {
        if (lane < 8) lanes.push_back({edges[lane], edges[7 - lane], edges[(lane + 3) % 8], edges[lane]});
        else lanes.push_back({next(), next(), next(), next()});
    }
    const BufferAddressShape shapes[] = {
        faithful(12, 16, true, true), faithful(4095, 0, false, false), faithful(0, 0, false, true),
        faithful(8, 4, true, false),  faithful(4, 0x10001, true, true), kFoldedElement, kFoldedZero};
    for (const auto& shape : shapes) {
        const auto out = run_block(shape, lanes);
        ASSERT_EQ(out.size(), 2 * lanes.size());
        for (size_t lane = 0; lane < lanes.size(); ++lane) {
            const auto& w = lanes[lane];
            const auto want = eval_buffer_address(shape, {w[0], w[1], w[2], w[3]});
            ASSERT_TRUE(want[0] && want[1]);
            EXPECT_EQ(out[2 * lane], *want[0]) << "form " << int(shape.form) << " lane " << lane;
            EXPECT_EQ(out[2 * lane + 1], *want[1]) << "form " << int(shape.form) << " lane " << lane;
        }
    }
}

}   // namespace
