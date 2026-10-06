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
// EXACT programs, not a structural family, for the same reason as dcc_helper_program.hpp: an
// ordinary full-screen pass can also be a vertex-id rectangle and must still draw under a stale
// MODE. A new helper variant is added from observation, with its words. CONFIDENCE: HIGH for both
// entries (live census of every MODE=2 vertex program: Dragon Quest VII runs exactly one, the first
// entry; Astro Bot runs the second plus five ordinary vertex programs with 2-23 vertex resources
// each, which this deliberately does not match). The rectangle itself is shared: Astro Bot draws it
// under MODE 2 and 6 with one pixel program, and Dragon Quest VII also draws it under MODE 0 for
// AGC's "Decompress Htile" helper (below; an earlier revision read those as ordinary copies), so the
// vertex program alone is not the operation either -- both halves are needed.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>

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

struct AgcEfcRectVertexProgram {
    const uint32_t* words;
    size_t dwords;
};
inline constexpr AgcEfcRectVertexProgram kAgcEfcRectVertexPrograms[] = {
    {kAgcEfcRectVertexA, std::size(kAgcEfcRectVertexA)},
    {kAgcEfcRectVertexB, std::size(kAgcEfcRectVertexB)},
};

// True when `code` (at least `max_dwords` readable) begins with one of the observed AGC
// eliminate-fast-clear rectangle vertex programs.
inline bool is_agc_efc_rect_vertex_program(const uint32_t* code, size_t max_dwords) {
    if (!code) return false;
    return std::any_of(std::begin(kAgcEfcRectVertexPrograms), std::end(kAgcEfcRectVertexPrograms),
                       [&](const AgcEfcRectVertexProgram& helper) {
                           return max_dwords >= helper.dwords &&
                                  std::equal(helper.words, helper.words + helper.dwords, code);
                       });
}

// The draw IS the colour-block eliminate-fast-clear operation: its decoded MODE says so AND it is
// AGC's own rectangle for that operation. Such a draw writes no colour on this backend.
inline bool is_agc_eliminate_fast_clear_operation(uint32_t cb_color_control,
                                                  const uint32_t* vertex_code,
                                                  size_t vertex_dwords) {
    namespace P = prosper::agc::Pm4;
    const uint32_t mode =
        (cb_color_control >> P::CB_COLOR_CONTROL_MODE_SHIFT) & P::CB_COLOR_CONTROL_MODE_MASK;
    return mode == P::CB_COLOR_CONTROL_MODE_ELIMINATE_FAST_CLEAR &&
           is_agc_efc_rect_vertex_program(vertex_code, vertex_dwords);
}

// The draw is an AGC helper operation run with the colour block DISABLED: AGC's own rectangle under
// CB_COLOR_CONTROL.MODE = 0. AGC draws the same rectangle for its depth-metadata helpers -- Dragon
// Quest VII's command segments are labelled "Decompress Htile" -- which program MODE = DISABLE,
// DB_RENDER_CONTROL's compress-disable bits and no colour state of their own, so the colour target
// and masks are whatever the parent stream left bound. Hardware writes no colour under CB_DISABLE;
// running the rectangle as an ordinary draw paints the parent's pixel shader over the parent's
// target. Like the eliminate pass this is keyed on the helper's vertex program, never on MODE alone:
// MODE = 0 latched onto an ordinary title draw must keep writing (#1724, Astro Bot).
// CONFIDENCE: HIGH that the hardware operation writes no colour; MED that every MODE = 0 use of
// the rectangle is a metadata helper (every one observed is inside a labelled AGC helper segment).
inline bool is_agc_colour_disabled_helper_operation(uint32_t cb_color_control,
                                                    const uint32_t* vertex_code,
                                                    size_t vertex_dwords) {
    namespace P = prosper::agc::Pm4;
    const uint32_t mode =
        (cb_color_control >> P::CB_COLOR_CONTROL_MODE_SHIFT) & P::CB_COLOR_CONTROL_MODE_MASK;
    return mode == P::CB_COLOR_CONTROL_MODE_DISABLE &&
           is_agc_efc_rect_vertex_program(vertex_code, vertex_dwords);
}

}   // namespace prosper::gpu
