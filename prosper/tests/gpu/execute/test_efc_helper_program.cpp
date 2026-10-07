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

// Compiles of AGC's helper rectangle that are not exact entries (#4610). The Oregon Trail
// (PPSA19244) and Kena (PPSA01802) bind these for their "Eliminate Fast Clear" and "Decompress
// Htile" segments. Unrecognised, each eliminate pass painted the inherited pixel shader over the
// finished scanout once BOOL64 predication ran the pass the title asked for, and both titles went
// black. Words copied from the live programs, up to s_endpgm.
constexpr uint32_t kOregonRect[] = {
    0xbfa00001u, 0x93eaff03u, 0x00080008u, 0x876bff03u, 0x000000ffu, 0x8f6a8c6au,
    0x887c6a6bu, 0xbf800000u, 0xbf900009u, 0x906a8803u, 0x81ea6a80u, 0x90fe6ac1u,
    0xf8000941u, 0x00000000u, 0x81ea0380u, 0xbf8cff0fu, 0x90fe6ac1u, 0x36040a81u,
    0x2c060a81u, 0x7e000280u, 0x7e0202f2u, 0x7e040d02u, 0x7e060d03u, 0xd5410002u,
    0x03ce04f4u, 0xd5410003u, 0x03ce06f4u, 0xf80008cfu, 0x01000302u, 0xbf810000u,
};
constexpr uint32_t kKenaRect[] = {
    0xbfa00001u, 0x93ebff03u, 0x00080008u, 0x8f6a8c6bu, 0x8700ff03u, 0x000000ffu,
    0x887c6a00u, 0xbf900009u, 0x81ea6bc0u, 0x90fe6ac1u, 0xf8000941u, 0x00000000u,
    0x81ea00c0u, 0xbf8cff0fu, 0x90fe6ac1u, 0x36040a81u, 0x2c060a81u, 0x7e000280u,
    0x7e0202f2u, 0x7e040d02u, 0x7e060d03u, 0xd5410002u, 0x03ce04f4u, 0xd5410003u,
    0x03ce06f4u, 0xf80008cfu, 0x01000302u, 0xbf810000u,
};

TEST(EfcHelperProgram, RecognisesOtherCompilesOfTheRectangleFamily) {
    for (const auto& words : {std::vector<uint32_t>(std::begin(kOregonRect), std::end(kOregonRect)),
                              std::vector<uint32_t>(std::begin(kKenaRect), std::end(kKenaRect))}) {
        const auto block = with_metadata(words.data(), words.size());
        EXPECT_TRUE(is_agc_efc_rect_vertex_program(block.data(), block.size()));
        EXPECT_TRUE(is_agc_eliminate_fast_clear_operation(
            mode_word(P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR), block.data(), block.size()));
        EXPECT_TRUE(is_agc_decompress_htile_operation(mode_word(P::CB_COLOR_CONTROL_MODE_DISABLE),
                                                      0x60, block.data(), block.size()));
        EXPECT_FALSE(is_agc_eliminate_fast_clear_operation(
            mode_word(P::CB_COLOR_CONTROL_MODE_NORMAL), block.data(), block.size()));
    }
}

TEST(EfcHelperProgram, TheFamilyIsTheRectangleAndNothingNear) {
    std::vector<uint32_t> w(std::begin(kKenaRect), std::end(kKenaRect));
    auto matches = [](std::vector<uint32_t> v) {
        return is_agc_efc_rect_vertex_program(v.data(), v.size());
    };
    ASSERT_TRUE(matches(w));
    {   // the position arithmetic differs (v_mov v1, 0 instead of 1.0): another program
        auto v = w;
        v[18] = 0x7e020280u;
        EXPECT_FALSE(matches(v));
    }
    {   // a memory instruction before the core (s_load_dwordx4): a resource-reading program
        auto v = w;
        v[7] = 0xf4080000u;
        EXPECT_FALSE(matches(v));
    }
    {   // no NGG primitive export
        auto v = w;
        v[10] = 0xbf800000u;
        EXPECT_FALSE(matches(v));
    }
    {   // the same core after a long prologue: past the 32-dword bound
        std::vector<uint32_t> v(8, 0xbf800000u);
        v.insert(v.end(), w.begin(), w.end());
        EXPECT_FALSE(matches(v));
    }
    EXPECT_FALSE(is_agc_efc_rect_vertex_program(kKenaRect, std::size(kKenaRect) - 1))
        << "a window that ends before s_endpgm cannot match";
}

// The family branch needs the helper's draw shape; the exact entries do not.
TEST(EfcHelperProgram, TheFamilyBranchNeedsTheHelperDrawShape) {
    const auto kena = with_metadata(kKenaRect, std::size(kKenaRect));
    const auto exact = with_metadata(kAgcEfcRectVertexA, std::size(kAgcEfcRectVertexA));
    const AgcHelperDrawShape helper{7u, 3u, false};
    EXPECT_TRUE(is_agc_efc_rect_vertex_program(kena.data(), kena.size(), helper));
    for (const AgcHelperDrawShape other : {AgcHelperDrawShape{4u, 3u, false},    // triangle list
                                           AgcHelperDrawShape{5u, 4u, false},    // triangle strip
                                           AgcHelperDrawShape{7u, 4u, false},    // 4 vertices
                                           AgcHelperDrawShape{7u, 3u, true}}) {  // indexed
        EXPECT_FALSE(is_agc_efc_rect_vertex_program(kena.data(), kena.size(), other))
            << "prim " << other.prim_type << " vertices " << other.vertices;
        EXPECT_TRUE(is_agc_efc_rect_vertex_program(exact.data(), exact.size(), other))
            << "the exact entries are not shape-gated";
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

// AGC's "Decompress Htile" helper draws the same rectangle with the colour block DISABLED, the
// depth/stencil compress-disable bits set in DB_RENDER_CONTROL, and no colour state of its own, so
// it inherits the parent's target and masks. It writes no colour. Each half of the signature is
// load-bearing: without the decompress bits, or with an ordinary program, a latched MODE=0 draw
// still writes (#1724).
TEST(EfcHelperProgram, TheDecompressHtileHelperWritesNoColour) {
    std::copy(std::begin(kAgcEfcRectVertexA), std::end(kAgcEfcRectVertexA), kHelperBlock);
    constexpr uint32_t kRectList = 7, kTriangleList = 4;
    constexpr uint32_t kDecompress = 0x60;   // Dragon Quest VII's helper value: both bits
    const uint32_t disable = P::CB_COLOR_CONTROL_MODE_DISABLE;
    const auto block = with_metadata(kAgcEfcRectVertexA, std::size(kAgcEfcRectVertexA));
    EXPECT_TRUE(is_agc_decompress_htile_operation(mode_word(disable), kDecompress, block.data(),
                                                  block.size()));
    EXPECT_TRUE(is_agc_decompress_htile_operation(mode_word(disable), 0x40, block.data(),
                                                  block.size()))
        << "depth compress-disable alone";
    EXPECT_TRUE(is_agc_decompress_htile_operation(mode_word(disable), 0x20, block.data(),
                                                  block.size()))
        << "stencil compress-disable alone";
    EXPECT_FALSE(is_agc_decompress_htile_operation(mode_word(disable), 0x0, block.data(),
                                                   block.size()))
        << "MODE=0 on the rectangle without the decompress signature is not the helper";
    for (uint32_t other : {1u, 2u, 3u, 6u})
        EXPECT_FALSE(is_agc_decompress_htile_operation(mode_word(other), kDecompress, block.data(),
                                                       block.size()))
            << "mode " << other;

    GpuState helper_state = state(kHelperBlock, kRectList, disable);
    helper_state.cx[P::DB_RENDER_CONTROL] = kDecompress;
    const Realized helper = realize(helper_state);
    EXPECT_FALSE(helper.made) << "the helper has no colour and no depth/stencil effect";
    EXPECT_EQ(helper.reason, RealizationFailureReason::NoEffect);

    const Realized no_signature = realize(state(kHelperBlock, kRectList, disable));
    EXPECT_TRUE(no_signature.made) << "the rectangle under MODE=0 alone still draws";
    EXPECT_EQ(no_signature.mask, 0xFu);

    GpuState stale_state = state(kVs, kTriangleList, disable);
    stale_state.cx[P::DB_RENDER_CONTROL] = kDecompress;
    const Realized stale = realize(stale_state);
    EXPECT_TRUE(stale.made) << "an ordinary program under a latched MODE=0 and 0x60 still draws";
    EXPECT_EQ(stale.mask, 0xFu);
}

// The regression #4610's cross-title A/B caught: Oregon's eliminate pass, under MODE=2 on its own
// compile of the rectangle, must write no colour; it ran after the composite and painted it black.
TEST(EfcHelperProgram, OtherCompilesWriteNoColourEither) {
    constexpr uint32_t kRectList = 7;
    for (const auto& words : {std::vector<uint32_t>(std::begin(kOregonRect), std::end(kOregonRect)),
                              std::vector<uint32_t>(std::begin(kKenaRect), std::end(kKenaRect))}) {
        std::fill(std::begin(kHelperBlock), std::end(kHelperBlock), 0u);
        std::copy(words.begin(), words.end(), kHelperBlock);
        const Realized eliminate =
            realize(state(kHelperBlock, kRectList, P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR));
        EXPECT_FALSE(eliminate.made);
        EXPECT_EQ(eliminate.reason, RealizationFailureReason::NoEffect);
        GpuState decompress = state(kHelperBlock, kRectList, P::CB_COLOR_CONTROL_MODE_DISABLE);
        decompress.cx[P::DB_RENDER_CONTROL] = 0x60;
        EXPECT_FALSE(realize(decompress).made);
    }
}

// The residual class the family relaxation leaves, pinned: a position-only full-screen quad with
// exactly the family's shader, drawn as an ordinary triangle list under a stale helper MODE, still
// writes its colour. So does the same program under MODE=1.
TEST(EfcHelperProgram, APositionOnlyQuadOutsideTheHelperShapeStillDraws) {
    constexpr uint32_t kTriangleList = 4;
    std::fill(std::begin(kHelperBlock), std::end(kHelperBlock), 0u);
    std::copy(std::begin(kKenaRect), std::end(kKenaRect), kHelperBlock);
    const Realized stale_eliminate =
        realize(state(kHelperBlock, kTriangleList, P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR));
    EXPECT_TRUE(stale_eliminate.made);
    EXPECT_EQ(stale_eliminate.mask, 0xFu);
    GpuState stale_decompress =
        state(kHelperBlock, kTriangleList, P::CB_COLOR_CONTROL_MODE_DISABLE);
    stale_decompress.cx[P::DB_RENDER_CONTROL] = 0x60;
    const Realized decompress = realize(stale_decompress);
    EXPECT_TRUE(decompress.made);
    EXPECT_EQ(decompress.mask, 0xFu);
    const Realized normal =
        realize(state(kHelperBlock, 7u, P::CB_COLOR_CONTROL_MODE_NORMAL));
    EXPECT_TRUE(normal.made) << "the helper shape under MODE=1 is an ordinary draw";
    EXPECT_EQ(normal.mask, 0xFu);
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

// #4680: the helper binds no pixel shader of its own; it runs with whatever the previous draw left
// bound, together with that draw's stale pixel user data. Kena's live helpers inherited pixel
// shaders whose raw buffer_load_dword reads a V# at s[24:27], and the stale words there
// (0x92 0x00fff000 0x05000000, then an unwritten register) are not a descriptor the fold can
// publish, so the recompiler refused the shader and the helper counted as a dropped draw -- several
// hundred per run -- although a helper without a depth/stencil effect is "no effect" whatever its
// shader does. The decision is now taken before any shader work. The ordinary-draw arm proves the
// shader really is refused under those words, so the helper arms cannot pass by compiling it; the
// depth-write arm proves a helper that still has an effect keeps going through the compiler.
alignas(256) const uint32_t kStaleVSharpPs[] = {
    0xE0302004u, 0x80061701u,   // buffer_load_dword v23, v1, s[24:27], 0 offen
    0xBF8C3F70u,                // s_waitcnt vmcnt(0)
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};

GpuState with_stale_vsharp_ps(GpuState st) {
    set_pgm(st, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kStaleVSharpPs);
    st.sh[P::SPI_SHADER_USER_DATA_PS_0 + 24] = 0x00000092u;   // Kena's live words, PPSA01802
    st.sh[P::SPI_SHADER_USER_DATA_PS_0 + 25] = 0x00fff000u;
    st.sh[P::SPI_SHADER_USER_DATA_PS_0 + 26] = 0x05000000u;
    return st;
}

TEST(EfcHelperProgram, AHelperWithoutDepthStencilEffectNeverCompilesItsInheritedPixelShader) {
    constexpr uint32_t kRectList = 7, kTriangleList = 4;
    std::fill(std::begin(kHelperBlock), std::end(kHelperBlock), 0u);
    std::copy(std::begin(kKenaRect), std::end(kKenaRect), kHelperBlock);

    const Realized ordinary = realize(
        with_stale_vsharp_ps(state(kVs, kTriangleList, P::CB_COLOR_CONTROL_MODE_NORMAL)));
    EXPECT_FALSE(ordinary.made);
    ASSERT_EQ(ordinary.reason, RealizationFailureReason::ShaderRecompile)
        << "positive control: the inherited shader is refused under the stale user data";

    GpuState decompress = with_stale_vsharp_ps(
        state(kHelperBlock, kRectList, P::CB_COLOR_CONTROL_MODE_DISABLE));
    decompress.cx[P::DB_RENDER_CONTROL] = 0x60;
    decompress.cx[P::DB_DEPTH_CONTROL] = 0x70;   // Kena's helper: ZFUNC=ALWAYS, Z test/write off
    const Realized decompress_helper = realize(decompress);
    EXPECT_FALSE(decompress_helper.made);
    EXPECT_EQ(decompress_helper.reason, RealizationFailureReason::NoEffect)
        << "Decompress Htile: no effect, not a refused shader";

    const Realized eliminate = realize(with_stale_vsharp_ps(
        state(kHelperBlock, kRectList, P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR)));
    EXPECT_FALSE(eliminate.made);
    EXPECT_EQ(eliminate.reason, RealizationFailureReason::NoEffect)
        << "Eliminate Fast Clear: no effect, not a refused shader";

    GpuState with_depth = with_stale_vsharp_ps(
        state(kHelperBlock, kRectList, P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR));
    with_depth.cx[P::DB_DEPTH_CONTROL] = (1u << P::DB_DEPTH_CONTROL_Z_ENABLE_SHIFT) |
                                         (1u << P::DB_DEPTH_CONTROL_Z_WRITE_ENABLE_SHIFT);
    const Realized depth_helper = realize(with_depth);
    EXPECT_FALSE(depth_helper.made);
    EXPECT_EQ(depth_helper.reason, RealizationFailureReason::ShaderRecompile)
        << "a helper with a depth write still needs its shader, so the refusal stays visible";
}

}  // namespace
