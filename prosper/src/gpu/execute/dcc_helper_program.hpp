// dcc_helper_program.hpp -- recognise AGC's descriptor-free colour-block utility pixel programs.
//
// A draw with CB_COLOR_CONTROL.MODE = DCC_DECOMPRESS is a colour-block METADATA operation: the
// hardware expands the bound target's DCC state and ignores the pixel shader's export. AGC binds a
// tiny pixel program for it that exports a constant. Two such programs are observed:
//
//   7e000280 f8001803 00000000 bf810000   v_mov_b32 v0, 0 ; exp mrt0 v0, v0 (R,G) done vm ;
//                                          s_endpgm
//   7e0002ff 3c003c00 f8001c0f 00000000 bf810000
//                                          v_mov_b32 v0, 0x3c003c00 (two fp16 1.0) ;
//                                          exp mrt0 v0, v0 compr (RGBA = 1,1,1,1) done vm ;
//                                          s_endpgm
//
// The second is Hollow Knight: Silksong's (PPSA12544): it precedes every stencil-mask pass over the
// title's scanout. It was not recognised, so it ran as an ordinary draw and exported its 1.0 --
// the boot sequence (autosave notice, logos) was a solid white screen.
//
// EXACT programs, not a structural family, and deliberately so. MODE alone is not a discriminator
// (stale operation bits can stay folded into a later submit; treating MODE alone as "no colour"
// replaced Astro Bot's post-process and scanout shaders), and "a few constant moves then one MRT0
// export" is not one either: an ordinary solid-colour fill shader has exactly that shape
// (test_gpu_execute's green kPs does), and must still draw under a stale MODE. A new AGC helper
// variant is therefore added here from observation, with its words, not inferred.
// CONFIDENCE: HIGH for both entries (live captures; the second from PPSA12544's mode-6 draws).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>

namespace prosper::gpu {

inline constexpr uint32_t kAgcDccHelperClearRg[] = {
    0x7e000280u, 0xf8001803u, 0x00000000u, 0xbf810000u,
};
inline constexpr uint32_t kAgcDccHelperFp16One[] = {
    0x7e0002ffu, 0x3c003c00u, 0xf8001c0fu, 0x00000000u, 0xbf810000u,
};

struct AgcDccHelperProgram {
    const uint32_t* words;
    size_t dwords;
};
// The table, for callers that compare against owned shader bytes rather than a raw pointer (the
// executor's retained shader analysis -- see shader_analysis_has_prefix).
inline constexpr AgcDccHelperProgram kAgcDccHelperPrograms[] = {
    {kAgcDccHelperClearRg, std::size(kAgcDccHelperClearRg)},
    {kAgcDccHelperFp16One, std::size(kAgcDccHelperFp16One)},
};

// True when `code` (at least `max_dwords` readable) begins with one of the observed AGC helpers.
inline bool is_agc_dcc_helper_program(const uint32_t* code, size_t max_dwords) {
    if (!code) return false;
    return std::any_of(std::begin(kAgcDccHelperPrograms), std::end(kAgcDccHelperPrograms),
                       [&](const AgcDccHelperProgram& helper) {
                           return max_dwords >= helper.dwords &&
                                  std::equal(helper.words, helper.words + helper.dwords, code);
                       });
}

} // namespace prosper::gpu
