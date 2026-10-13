// Raw instruction words for fail-visible recompiler diagnostics.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace prosper::gpu {

// #2319: render exactly the dwords the instruction HAS, not a fixed two.
//
// `words[]` is `uint32_t words[5] = {0,...}`, so a ONE-dword instruction printed as
// `words=bf860051,00000000` is indistinguishable from a two-dword instruction whose second dword
// is genuinely zero. That defeats the purpose of printing raw words (#2312), which is that a
// reader can name the instruction with one command:
//
//     echo '0x0e,0x16,0xea,0xbe' | llvm-mc -arch=amdgcn -mcpu=gfx1010 -disassemble
//
// A reader who takes `words=x,00000000` at face value feeds EIGHT bytes, and llvm-mc decodes a
// second, entirely fictitious instruction from the zero dword -- output that looks exactly as
// authoritative as the correct answer. On the CrossWorlds census pc=101 needed four bytes and
// pc=2038 needed eight, and nothing in either line said which.
//
// Clamped to the array bound as well as to len_dwords: len_dwords comes from the decoder, and a
// decode that failed badly enough to be rejected here is not a source to trust for indexing.
inline std::string reject_words_text(const Rdna2Inst& in) {
    const uint32_t count = std::min<uint32_t>(in.len_dwords ? in.len_dwords : 1u,
                                              (uint32_t)(sizeof(in.words) / sizeof(in.words[0])));
    std::string out;
    char buf[16];
    for (uint32_t i = 0; i < count; ++i) {
        std::snprintf(buf, sizeof buf, "%08x", in.words[i]);
        if (i) out += ',';
        out += buf;
    }
    return out;
}

} // namespace prosper::gpu
