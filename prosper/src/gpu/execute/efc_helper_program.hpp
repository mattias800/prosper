// efc_helper_program.hpp -- recognise AGC's ELIMINATE_FAST_CLEAR rectangle (#1588).
//
// CB_COLOR_CONTROL.MODE = 2 (ELIMINATE_FAST_CLEAR) is a colour-block METADATA operation: the
// hardware expands a fast-cleared surface's compressed clear state into explicit pixels and does
// not write the pixel shader's export. prosper stores render targets uncompressed, so the expansion
// has nothing left to do; running the pass as an ordinary draw instead paints whatever pixel shader
// is still bound over the target. That is Dragon Quest VII Reimagined's (PPSA17942) white flashes
// and its blue title-menu frames: the operation's pixel shader is inherited from the previous draw,
// so the target receives an unrelated shader's output.
//
// MODE alone does not identify the operation. Astro Bot (PPSA21564) programs MODE 0, 2 and 6 and
// never 1, so its ordinary shaded draws carry a latched MODE=2 and must keep writing (#1588, #1706,
// #1724). What identifies the operation is the draw AGC builds for it: a three-vertex RectList
// whose vertex program is a fixed procedural rectangle -- NGG primitive export, then positions
// derived from the vertex id alone, with no vertex resources. Two compiles of that rectangle are
// observed (different arithmetic; the second also exports the layer); both are below, matched as
// EXACT code prefixes through s_endpgm. The words after s_endpgm in a live program block are AGC
// shader metadata and are not compared.
//
// Two observed compiles are listed EXACTLY (CONFIDENCE: HIGH; live census of every MODE=2 vertex
// program: Dragon Quest VII runs exactly one, the first entry; Astro Bot runs the second plus five
// ordinary vertex programs with 2-23 vertex resources each, which this deliberately does not
// match). Beyond them, the rectangle is recognised by its FAMILY (is_agc_rect_family_program).
// This was "exact programs only" until #4610. Once BOOL64 predication runs the eliminate pass when
// its "needs it" word says so, that pass lands after the frame's composite, and an unlisted compile
// of the same rectangle paints the inherited pixel shader over the whole scanout. The Oregon Trail
// (PPSA19244) and Kena (PPSA01802) each bind a third and a fourth compile, and both went black.
//
// THE RESIDUAL RISK, stated plainly. The family's core is the textbook vertex-id full-screen quad,
// pos = float4(float2(id & 1, id >> 1) * 2 - 1, 0, 1), and its prologue is NGG toolchain
// boilerplate. Oregon's prologue equals Astro Bot's entry, and Kena's is a reordering of the first
// entry. So an ORDINARY title pass whose vertex shader is that quad and nothing else (no memory
// read, no output beyond position) can match. If it is drawn under a stale MODE=2, or a stale MODE=0
// with DB_RENDER_CONTROL's compress-disable bits, its colour is dropped. Both of those latched states
// are measured on ordinary draws (#1724; DQ7 latches 0x60). The protections are:
//   * the MODE / DB_RENDER_CONTROL gate of the two operations below;
//   * no memory instruction before the core, which rejects any vertex shader that reads a resource
//     (Astro Bot's ordinary MODE=2 programs read 2-23);
//   * the 32-dword bound and the NGG primitive export;
//   * the draw's own shape, applied to the family branch only: AGC draws every helper as a 3-vertex
//     non-indexed RectList (DrawIndexAuto(3), VGT_PRIMITIVE_TYPE 7), in DQ7's, Oregon's and Kena's
//     dumped segments alike. A full-screen pass drawn as a triangle list or strip cannot match; one
//     drawn as a 3-vertex RectList with this exact shader still can.
// Each program the family matches is reported once ([agc] helper rectangle recognised by family),
// so a false positive is visible in an ordinary run log, not only in the dropped-draw census.
// CONFIDENCE: MED for the family branch. The trade is deliberate: an unrecognised helper is a black
// frame, while a false positive needs a position-only RectList(3) quad under a latched helper MODE.
//
// The rectangle itself is shared: Astro Bot draws it under MODE 2 and 6 with one pixel program, and
// Dragon Quest VII also draws it under MODE 0 for AGC's "Decompress Htile" helper (below; an earlier
// revision read those as ordinary copies), so the vertex program alone is not the operation either:
// both halves are needed.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <unordered_set>

#include "gpu/pm4/pm4_registers.hpp"

namespace prosper::gpu {

// Dragon Quest VII Reimagined (PPSA17942).
inline constexpr uint32_t kAgcEfcRectVertexA[] = {
    0xbfa00001u, 0x93ebff03u, 0x00080008u, 0x8700ff03u, 0x000000ffu, 0x8f6a8c6bu,
    0x887c6a00u, 0xbf800000u, 0xbf900009u, 0x81ea6bc0u, 0x90fe6ac1u, 0xf8000941u,
    0x00000000u, 0x81ea00c0u, 0xbf8cff0fu, 0x90fe6ac1u, 0x36040a81u, 0x2c060a81u,
    0x7e000280u, 0x7e0202f2u, 0x7e040d02u, 0x7e060d03u, 0xd5410002u, 0x03ce04f4u,
    0xd5410003u, 0x03ce06f4u, 0xf80008cfu, 0x01000302u, 0xbf810000u,
};
// Astro Bot (PPSA21564); also exports the layer (pos1.z = v8) for array targets.
inline constexpr uint32_t kAgcEfcRectVertexB[] = {
    0xbfa00001u, 0x93eaff03u, 0x00080008u, 0x876bff03u, 0x000000ffu, 0x8f6a8c6au, 0x887c6a6bu,
    0xbf800000u, 0xbf900009u, 0x906a8803u, 0x81ea6a80u, 0x90fe6ac1u, 0xf8000941u, 0x00000000u,
    0x81ea0380u, 0xbf8cff0fu, 0x90fe6ac1u, 0x34040a81u, 0x36060ac2u, 0x7e000280u, 0x7e0202f2u,
    0x36040482u, 0x4a0606c1u, 0x4a0404c1u, 0x7e060b03u, 0x7e040b02u, 0xf80000d4u, 0x00080000u,
    0xf80008cfu, 0x01000302u, 0xbf810000u,
};

// The rectangle's own arithmetic, from v_and_b32 v2, 1, v5 (vertex-id bit 0) through the position
// export (v2, v3, v0 = 0, v1 = 1.0) and s_endpgm. It is identical, word for word, in the first
// entry above, in The Oregon Trail's compile and in Kena's. Those compiles differ only in their
// NGG prologue (wave-id decode, s_nop placement, the primitive export's register).
inline constexpr uint32_t kAgcRectPositionCore[] = {
    0x36040a81u, 0x2c060a81u, 0x7e000280u, 0x7e0202f2u, 0x7e040d02u, 0x7e060d03u, 0xd5410002u,
    0x03ce04f4u, 0xd5410003u, 0x03ce06f4u, 0xf80008cfu, 0x01000302u, 0xbf810000u,
};

// True when `code` is a compile of AGC's procedural rectangle that is not one of the exact entries:
// at most 32 dwords through its s_endpgm, ending in kAgcRectPositionCore, with an NGG primitive
// export (EXP prim, 0xf8000941) and no memory instruction (SMEM, buffer, image, flat, LDS) before
// the core. The scan is over raw dwords. A literal in the prologue that happens to look like a
// memory encoding can only make this refuse, never match. This alone also matches an ordinary
// position-only full-screen quad: callers gate it on the helper's draw shape (see the header).
// CONFIDENCE: MED.
inline bool is_agc_rect_family_program(const uint32_t* code, size_t max_dwords) {
    constexpr size_t kMaxProgramDwords = 32;
    constexpr size_t kCore = std::size(kAgcRectPositionCore);
    if (!code) return false;
    const size_t window = std::min(max_dwords, kMaxProgramDwords);
    size_t end = window;
    for (size_t i = 0; i < window; ++i)
        if (code[i] == 0xbf810000u) { end = i; break; }
    if (end == window || end + 1 < kCore) return false;
    const size_t core_at = end + 1 - kCore;
    if (!std::equal(std::begin(kAgcRectPositionCore), std::end(kAgcRectPositionCore),
                    code + core_at))
        return false;
    bool prim_export = false;
    for (size_t i = 0; i < core_at; ++i) {
        const uint32_t top6 = code[i] >> 26;
        // SMEM 0x3d, MUBUF 0x38, MTBUF 0x3a, MIMG 0x3c, FLAT/GLOBAL/SCRATCH 0x37, DS 0x36.
        if (top6 == 0x3du || top6 == 0x38u || top6 == 0x3au || top6 == 0x3cu || top6 == 0x37u ||
            top6 == 0x36u)
            return false;
        if (code[i] == 0xf8000941u) prim_export = true;
    }
    return prim_export;
}

struct AgcEfcRectVertexProgram {
    const uint32_t* words;
    size_t dwords;
};
inline constexpr AgcEfcRectVertexProgram kAgcEfcRectVertexPrograms[] = {
    {kAgcEfcRectVertexA, std::size(kAgcEfcRectVertexA)},
    {kAgcEfcRectVertexB, std::size(kAgcEfcRectVertexB)},
};

// True when `code` begins with one of the exact observed compiles.
inline bool is_agc_efc_rect_exact_program(const uint32_t* code, size_t max_dwords) {
    if (!code) return false;
    return std::any_of(std::begin(kAgcEfcRectVertexPrograms), std::end(kAgcEfcRectVertexPrograms),
                       [&](const AgcEfcRectVertexProgram& helper) {
                           return max_dwords >= helper.dwords &&
                                  std::equal(helper.words, helper.words + helper.dwords, code);
                       });
}

// The draw's shape, as AGC issues every helper: DrawIndexAuto(3), non-indexed, RectList
// (VGT_PRIMITIVE_TYPE 7). It gates only the family branch; the exact entries keep matching without it.
struct AgcHelperDrawShape {
    uint32_t prim_type = 7;
    uint32_t vertices = 3;
    bool indexed = false;
};
inline constexpr AgcHelperDrawShape kAgcHelperRectDraw{};
inline bool is_agc_helper_draw_shape(const AgcHelperDrawShape& shape) {
    return shape.prim_type == 7u && shape.vertices == 3u && !shape.indexed;
}

// True when `code` (at least `max_dwords` readable) is AGC's helper rectangle: one of the exact
// observed compiles, or a compile of the same family drawn in the helper's shape.
inline bool is_agc_efc_rect_vertex_program(const uint32_t* code, size_t max_dwords,
                                           const AgcHelperDrawShape& shape = kAgcHelperRectDraw) {
    if (!code) return false;
    return is_agc_efc_rect_exact_program(code, max_dwords) ||
           (is_agc_helper_draw_shape(shape) && is_agc_rect_family_program(code, max_dwords));
}

// One line per distinct program the FAMILY branch identified as a helper (bounded), so a false
// positive is visible in an ordinary run log.
inline void report_agc_rect_family_match(uint64_t vs_addr, size_t dwords, uint32_t cb_color_control,
                                         const char* operation) {
    static std::mutex mu;
    static std::unordered_set<uint64_t> seen;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (seen.size() >= 64 || !seen.insert(vs_addr).second) return;
    }
    namespace P = prosper::agc::Pm4;
    std::fprintf(stderr,
                 "[agc] helper rectangle recognised by family: vs=0x%llx dwords=%zu mode=%u op=%s "
                 "(not an exact entry; a colour-writing pass here would be a false positive, #4610)\n",
                 (unsigned long long)vs_addr, dwords,
                 (cb_color_control >> P::CB_COLOR_CONTROL_MODE_SHIFT) & P::CB_COLOR_CONTROL_MODE_MASK,
                 operation);
}

// The draw IS the colour-block eliminate-fast-clear operation: its decoded MODE says so AND it is
// AGC's own rectangle for that operation. Such a draw writes no colour on this backend.
inline bool is_agc_eliminate_fast_clear_operation(uint32_t cb_color_control,
                                                  const uint32_t* vertex_code,
                                                  size_t vertex_dwords,
                                                  const AgcHelperDrawShape& shape = kAgcHelperRectDraw) {
    namespace P = prosper::agc::Pm4;
    const uint32_t mode =
        (cb_color_control >> P::CB_COLOR_CONTROL_MODE_SHIFT) & P::CB_COLOR_CONTROL_MODE_MASK;
    return mode == P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR &&
           is_agc_efc_rect_vertex_program(vertex_code, vertex_dwords, shape);
}

// The draw is AGC's "Decompress Htile" helper: AGC's own rectangle under CB_COLOR_CONTROL.MODE = 0
// (DISABLE) with DB_RENDER_CONTROL's depth and/or stencil compress-disable bits set, the register
// signature of a depth-metadata decompress. Dragon Quest VII's helper segments carry that label
// and program exactly this: CB_COLOR_CONTROL 0x00cc0000, DB_RENDER_CONTROL 0x60 (both compress-
// disable bits), and no colour state of their own (no CB_COLOR0_BASE, no target mask). So the
// colour target and masks are whatever the parent stream left bound, and running the rectangle as
// an ordinary draw paints the parent's pixel shader over the parent's target. The operation is a
// metadata expansion that prosper's uncompressed depth does not need, and it writes no colour.
//
// Keyed on all three halves, never on MODE alone. A decoded MODE = 0 is not by itself a reliable
// "colour disabled" signal on this platform: #1724 measured Astro Bot draws under a latched
// MODE = 0 that must write. The exact rectangle and the decompress bits are what identify the
// helper. Limit: Dragon Quest VII never rewrites DB_RENDER_CONTROL after these helpers, so 0x60 is
// latched on its ordinary draws too. The bits narrow the rule for other titles, not for that one.
// The "Eliminate Fast Clear" helper uses the same rectangle under MODE = 2 and is the operation above.
// CONFIDENCE: HIGH that the decompress writes no colour; MED that the signature never marks a
// colour-writing use of the rectangle (every observed match is inside a labelled AGC helper segment).
inline bool is_agc_decompress_htile_operation(uint32_t cb_color_control, uint32_t db_render_control,
                                              const uint32_t* vertex_code, size_t vertex_dwords,
                                              const AgcHelperDrawShape& shape = kAgcHelperRectDraw) {
    namespace P = prosper::agc::Pm4;
    const uint32_t mode =
        (cb_color_control >> P::CB_COLOR_CONTROL_MODE_SHIFT) & P::CB_COLOR_CONTROL_MODE_MASK;
    const uint32_t compress_disable =
        (P::DB_RENDER_CONTROL_DEPTH_COMPRESS_DISABLE_MASK
         << P::DB_RENDER_CONTROL_DEPTH_COMPRESS_DISABLE_SHIFT) |
        (P::DB_RENDER_CONTROL_STENCIL_COMPRESS_DISABLE_MASK
         << P::DB_RENDER_CONTROL_STENCIL_COMPRESS_DISABLE_SHIFT);
    return mode == P::CB_COLOR_CONTROL_MODE_DISABLE && (db_render_control & compress_disable) != 0 &&
           is_agc_efc_rect_vertex_program(vertex_code, vertex_dwords, shape);
}

}   // namespace prosper::gpu
