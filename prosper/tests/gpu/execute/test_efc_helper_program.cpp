// AGC's eliminate-fast-clear rectangle (efc_helper_program.hpp, #1588). A CB_COLOR_CONTROL MODE=2
// draw whose vertex program is AGC's own rectangle is a colour-block metadata operation and writes
// no colour; the same MODE latched onto an ordinary draw must still write (Astro Bot programs MODE
// 0/2/6 and never 1). The matcher arms pin the identification; the realization arms pin what the
// executor does with it: the operation (helper rectangle under MODE=2) against an ordinary draw
// under the same latched MODE and under NORMAL, and the operation with a depth write, which must
// still execute for that effect.
#include "gpu/execute/efc_helper_program.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

namespace {

constexpr uint32_t mode_word(uint32_t mode) {
    return (mode << P::CB_COLOR_CONTROL_MODE_SHIFT) | (0xCCu << P::CB_COLOR_CONTROL_ROP3_SHIFT);
}

// A live program block: the helper's code, then AGC shader metadata (not compared).
std::vector<uint32_t> with_metadata(const uint32_t* code, size_t dwords) {
    std::vector<uint32_t> block(code, code + dwords);
    for (uint32_t extra : {0x00000000u, 0x30306c73u, 0x0000004cu, 0x00000064u})
        block.push_back(extra);
    return block;
}

TEST(EfcHelperProgram, RecognisesBothObservedHelpersOnlyUnderEliminateFastClear) {
    for (const auto& helper : kAgcEfcRectVertexPrograms) {
        const auto block = with_metadata(helper.words, helper.dwords);
        EXPECT_TRUE(is_agc_efc_rect_vertex_program(block.data(), block.size()));
        EXPECT_TRUE(is_agc_eliminate_fast_clear_operation(
            mode_word(P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR), block.data(), block.size()));
        for (uint32_t other : {0u, 1u, 3u, 6u})
            EXPECT_FALSE(
                is_agc_eliminate_fast_clear_operation(mode_word(other), block.data(), block.size()))
                << "mode " << other << " is not the operation, even on its rectangle";
    }
}

TEST(EfcHelperProgram, ExactWordsOnly) {
    std::vector<uint32_t> mutated(std::begin(kAgcEfcRectVertexA), std::end(kAgcEfcRectVertexA));
    mutated[19] = 0x7e020280u;   // v_mov_b32 v1, 0 instead of 1.0: another program
    EXPECT_FALSE(is_agc_efc_rect_vertex_program(mutated.data(), mutated.size()));
    EXPECT_FALSE(
        is_agc_efc_rect_vertex_program(kAgcEfcRectVertexA, std::size(kAgcEfcRectVertexA) - 1))
        << "a window shorter than the program cannot match";
    EXPECT_FALSE(is_agc_efc_rect_vertex_program(nullptr, 64));
}

// Realization arms. Fullscreen-triangle VS + solid-green PS (llvm-mc gfx1030), the blobs
// test_gpu_execute uses; 256-aligned so the SHADER_PGM encoding round-trips.
alignas(256) const uint32_t kVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u,
    0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u,
    0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
};
alignas(256) const uint32_t kPs[] = {
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
alignas(256) uint32_t kHelperBlock[64];

void set_pgm(GpuState& st, uint32_t lo, uint32_t hi, const void* p) {
    const uint64_t a = (uint64_t)(uintptr_t)p;
    st.sh[lo] = (uint32_t)((a >> 8) & 0xFFFFFFFFu);
    st.sh[hi] = (uint32_t)((a >> 40) & 0xFFu);
}

GpuState state(const void* vs, uint32_t prim, uint32_t mode) {
    GpuState st;
    set_pgm(st, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, vs);
    set_pgm(st, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kPs);
    st.uc[P::VGT_PRIMITIVE_TYPE] = prim;
    st.cx[P::CB_TARGET_MASK] = 0xF;
    st.cx[P::CB_COLOR_CONTROL] = mode_word(mode);
    GpuState::Draw draw;
    draw.index_count = 3;
    st.draws.push_back(draw);
    return st;
}

struct Realized {
    bool made = false;
    uint32_t mask = 0;
    RealizationFailureReason reason{};
};
Realized realize(const GpuState& st) {
    Realized r;
    DrawItem item;
    OperationRealizationFailure failure;
    r.made = realize_draw_item(st, &st.draws[0], 3u, 0x10000u, false, item, &failure);
    r.mask = item.ps.color_write_mask;
    r.reason = failure.reason;
    return r;
}

TEST(EfcHelperProgram, TheOperationWritesNoColourAndOrdinaryDrawsUnderAStaleModeStillDo) {
    std::copy(std::begin(kAgcEfcRectVertexA), std::end(kAgcEfcRectVertexA), kHelperBlock);
    constexpr uint32_t kRectList = 7, kTriangleList = 4;
    const uint32_t efc = P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR;

    const Realized operation = realize(state(kHelperBlock, kRectList, efc));
    EXPECT_FALSE(operation.made) << "the eliminate pass has no colour and no depth/stencil effect";
    EXPECT_EQ(operation.reason, RealizationFailureReason::NoEffect);

    const Realized stale = realize(state(kVs, kTriangleList, efc));
    EXPECT_TRUE(stale.made) << "an ordinary program under a latched MODE=2 still draws";
    EXPECT_EQ(stale.mask, 0xFu);

    const Realized normal = realize(state(kVs, kTriangleList, P::CB_COLOR_CONTROL_MODE_NORMAL));
    EXPECT_TRUE(normal.made);
    EXPECT_EQ(normal.mask, 0xFu);
}

// AGC's "Decompress Htile" helper draws the same rectangle with the colour block DISABLED and no
// colour state of its own, so it inherits the parent's target and masks. Hardware writes no colour;
// an ordinary program under a latched MODE=0 must still write (#1724).
TEST(EfcHelperProgram, TheRectangleUnderColourDisableWritesNoColour) {
    std::copy(std::begin(kAgcEfcRectVertexA), std::end(kAgcEfcRectVertexA), kHelperBlock);
    constexpr uint32_t kRectList = 7, kTriangleList = 4;
    const uint32_t disable = P::CB_COLOR_CONTROL_MODE_DISABLE;
    const auto block = with_metadata(kAgcEfcRectVertexA, std::size(kAgcEfcRectVertexA));
    EXPECT_TRUE(is_agc_colour_disabled_helper_operation(mode_word(disable), block.data(),
                                                        block.size()));
    for (uint32_t other : {1u, 2u, 3u, 6u})
        EXPECT_FALSE(
            is_agc_colour_disabled_helper_operation(mode_word(other), block.data(), block.size()))
            << "mode " << other;

    const Realized helper = realize(state(kHelperBlock, kRectList, disable));
    EXPECT_FALSE(helper.made) << "the helper has no colour and no depth/stencil effect";
    EXPECT_EQ(helper.reason, RealizationFailureReason::NoEffect);

    const Realized stale = realize(state(kVs, kTriangleList, disable));
    EXPECT_TRUE(stale.made) << "an ordinary program under a latched MODE=0 still draws";
    EXPECT_EQ(stale.mask, 0xFu);
}

TEST(EfcHelperProgram, TheOperationKeepsItsDepthStencilEffect) {
    std::copy(std::begin(kAgcEfcRectVertexA), std::end(kAgcEfcRectVertexA), kHelperBlock);
    GpuState st = state(kHelperBlock, 7u, P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR);
    st.cx[P::DB_DEPTH_CONTROL] = (1u << P::DB_DEPTH_CONTROL_Z_ENABLE_SHIFT) |
                                 (1u << P::DB_DEPTH_CONTROL_Z_WRITE_ENABLE_SHIFT);
    const Realized with_depth = realize(st);
    EXPECT_TRUE(with_depth.made) << "a depth write is an effect: the draw still executes";
    EXPECT_EQ(with_depth.mask, 0u) << "...and writes no colour";
}

}  // namespace
