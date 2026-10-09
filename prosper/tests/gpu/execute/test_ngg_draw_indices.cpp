// The index buffer of an indexed merged-NGG draw (ngg_draw_indices.hpp, #3135 P6): 16- and 32-bit
// decode, the index-count cap, and primitive restart (refused whenever it could change the draw).
// Pure CPU.
#include "gpu/execute/ngg_draw_indices.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

namespace {

template <typename T>
std::vector<unsigned char> bytes_of(const std::vector<T>& values) {
    std::vector<unsigned char> out(values.size() * sizeof(T));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

TEST(NggDrawIndices, SixteenAndThirtyTwoBitBuffersDecodeToTheSameValues) {
    const std::vector<uint16_t> narrow = {6, 1, 4, 4, 1, 3};
    const auto a = decode_ngg_draw_indices(bytes_of(narrow).data(), 2, 6, {});
    ASSERT_TRUE(a.indices) << a.refusal;
    EXPECT_EQ(*a.indices, (std::vector<uint32_t>{6, 1, 4, 4, 1, 3}));
    EXPECT_EQ(a.max_index, 6u);

    // Values above 0xffff survive a 32-bit buffer; read at 2 bytes the same bytes would give
    // (0x11, 1, 0x22, 1, ...), so the element size is what decides.
    const std::vector<uint32_t> wide = {0x10011, 0x10022, 0x10033};
    const auto b = decode_ngg_draw_indices(bytes_of(wide).data(), 4, 3, {});
    ASSERT_TRUE(b.indices) << b.refusal;
    EXPECT_EQ(*b.indices, wide);
    EXPECT_EQ(b.max_index, 0x10033u);
    const auto misread = decode_ngg_draw_indices(bytes_of(wide).data(), 2, 3, {});
    ASSERT_TRUE(misread.indices);
    EXPECT_EQ(*misread.indices, (std::vector<uint32_t>{0x11, 1, 0x22})) << "the 2-byte reading";
}

TEST(NggDrawIndices, ShapeRefusals) {
    const std::vector<uint16_t> narrow = {0, 1, 2};
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(narrow).data(), 1, 3, {}).refusal,
                 "ngg-index-element-size");
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(narrow).data(), 0, 3, {}).refusal,
                 "ngg-index-element-size");
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(narrow).data(), 2, 0, {}).refusal,
                 "ngg-index-count");
    EXPECT_STREQ(decode_ngg_draw_indices(nullptr, 2, 3, {}).refusal, "ngg-index-count");
    const std::vector<uint16_t> many(kNggMaxIndices + 1u, 0);
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(many).data(), 2, kNggMaxIndices + 1u, {}).refusal,
                 "ngg-index-count");
    const auto at_cap = decode_ngg_draw_indices(bytes_of(many).data(), 2, kNggMaxIndices, {});
    EXPECT_TRUE(at_cap.indices) << "the cap itself is admitted";
}

// #461's arm for the NGG path: one garbage 32-bit index would size the fold and every vertex
// buffer by max_index + 1. Refused by name, at the same 2^20 bound the ordinary path clamps to;
// 0xffffffff (which would wrap the range to 0) too. The largest admitted index is 2^20 - 1.
TEST(NggDrawIndices, AGarbageIndexValueIsRefusedNotClamped) {
    const std::vector<uint32_t> garbage = {0, 1, 0x0F000000u};
    const auto a = decode_ngg_draw_indices(bytes_of(garbage).data(), 4, 3, {});
    EXPECT_STREQ(a.refusal, "ngg-index-range");
    EXPECT_FALSE(a.indices);
    const std::vector<uint32_t> wraps = {0, 1, 0xffffffffu};
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(wraps).data(), 4, 3, {}).refusal,
                 "ngg-index-range");
    const std::vector<uint32_t> at_bound = {0, 1, kNggMaxIndices};
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(at_bound).data(), 4, 3, {}).refusal,
                 "ngg-index-range");
    const std::vector<uint32_t> below = {0, 1, kNggMaxIndices - 1u};
    const auto b = decode_ngg_draw_indices(bytes_of(below).data(), 4, 3, {});
    ASSERT_TRUE(b.indices) << b.refusal;
    EXPECT_EQ(b.max_index, kNggMaxIndices - 1u);
}

// Restart is refused whenever it could change the draw -- an index equal to the restart value, or a
// restart value the draw state does not hold -- and admitted when it provably cannot.
TEST(NggDrawIndices, PrimitiveRestartIsRefusedWhereItCouldFire) {
    const std::vector<uint16_t> with_ffff = {0, 1, 0xffff, 2, 3, 4};
    const std::vector<uint16_t> without = {0, 1, 2, 2, 3, 4};
    NggIndexRestart off;
    EXPECT_TRUE(decode_ngg_draw_indices(bytes_of(with_ffff).data(), 2, 6, off).indices)
        << "disabled: 0xffff is an ordinary index";
    NggIndexRestart unknown;
    unknown.enabled = true;
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(without).data(), 2, 6, unknown).refusal,
                 "ngg-index-restart-unknown");
    NggIndexRestart on;
    on.enabled = on.value_known = true;
    on.value = 0xffffffffu;   // compared at the element width: its low 16 bits for a 16-bit buffer
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(with_ffff).data(), 2, 6, on).refusal,
                 "ngg-index-restart");
    EXPECT_TRUE(decode_ngg_draw_indices(bytes_of(without).data(), 2, 6, on).indices)
        << "no index matches: restart cannot fire";
    const std::vector<uint32_t> wide = {0, 0xffff, 2};
    EXPECT_TRUE(decode_ngg_draw_indices(bytes_of(wide).data(), 4, 3, on).indices)
        << "a 32-bit buffer compares the full value";
    on.value = 2;
    EXPECT_STREQ(decode_ngg_draw_indices(bytes_of(wide).data(), 4, 3, on).refusal,
                 "ngg-index-restart");
}

TEST(NggDrawIndices, RestartIsReadFromTheDrawState) {
    namespace P = prosper::agc::Pm4;
    GpuState state;
    NggIndexRestart r = read_ngg_index_restart(state);
    EXPECT_FALSE(r.enabled) << "reset value 0";
    EXPECT_FALSE(r.value_known);
    state.uc[P::GE_MULTI_PRIM_IB_RESET_EN] = 0x2u;   // a bit other than RESET_EN
    EXPECT_FALSE(read_ngg_index_restart(state).enabled);
    state.uc[P::GE_MULTI_PRIM_IB_RESET_EN] = 0x1u;
    state.cx[P::VGT_MULTI_PRIM_IB_RESET_INDX] = 0xfffeu;
    r = read_ngg_index_restart(state);
    EXPECT_TRUE(r.enabled);
    EXPECT_TRUE(r.value_known);
    EXPECT_EQ(r.value, 0xfffeu);
}

}   // namespace
