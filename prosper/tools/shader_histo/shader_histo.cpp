// shader_histo — data-driven recompiler-coverage report. Scans a module (eboot) for the RDNA2
// shader ELFs embedded in its rodata (e_machine == EM_AMDGPU 0xE0; section ".shader_text"), walks
// each with the RDNA2 decoder (rdna2_decode), and prints a per-(format,opcode) instruction histogram
// plus how many instructions the decoder could not classify. This tells us exactly which RDNA2
// instructions the REAL game shaders use, so the RDNA2->SPIR-V recompiler is extended against real
// data instead of guesses.
//
//   shader_histo <eboot.bin> [dump_biggest.bin]
//
// Optional 2nd arg: write the largest shader's .shader_text to a file (for offline disassembly).
#include "../../src/self/module.hpp"
#include "../../src/gpu/rdna2_decode.hpp"
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <eboot.bin> [dump_biggest.bin]\n", argv[0]); return 1; }
    std::string err;
    auto m = Module::load(argv[1], &err);
    if (!m) { fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    const std::vector<uint8_t>& f = m->file;

    std::map<std::pair<int, uint32_t>, uint32_t> histo;
    int shaders = 0; size_t total = 0, unknown = 0, maxsz = 0;
    for (size_t i = 0; i + 64 < f.size(); i++) {
        if (!(f[i] == 0x7f && f[i+1] == 'E' && f[i+2] == 'L' && f[i+3] == 'F')) continue;
        auto inside = [&](uint64_t off, uint64_t len) {
            uint64_t avail = (uint64_t)(f.size() - i);
            return off <= avail && len <= avail - off;
        };
        uint16_t machine; memcpy(&machine, &f[i+18], 2);
        if (machine != 0xE0) continue;                       // EM_AMDGPU
        const uint8_t* e = &f[i];
        uint64_t shoff; memcpy(&shoff, e+0x28, 8);
        uint16_t shentsize; memcpy(&shentsize, e+0x3A, 2);
        uint16_t shnum; memcpy(&shnum, e+0x3C, 2);
        uint16_t shstrndx; memcpy(&shstrndx, e+0x3E, 2);
        if (!shoff || !shnum || shstrndx >= shnum || shentsize < 0x40 ||
            !inside(shoff, (uint64_t)shnum * shentsize)) continue;
        const uint8_t* shstr = e + shoff + (size_t)shstrndx * shentsize;
        uint64_t shstr_off, shstr_size; memcpy(&shstr_off, shstr + 0x18, 8); memcpy(&shstr_size, shstr + 0x20, 8);
        if (!inside(shstr_off, shstr_size)) continue;
        for (int s = 0; s < shnum; s++) {
            const uint8_t* sh = e + shoff + (size_t)s*shentsize;
            uint32_t noff; memcpy(&noff, sh, 4);
            if ((uint64_t)noff + 12 > shstr_size) continue;
            const char* name = (const char*)(e + shstr_off + noff);
            if (strncmp(name, ".shader_text", 12) != 0) continue;
            uint64_t so, sz; memcpy(&so, sh+0x18, 8); memcpy(&sz, sh+0x20, 8);
            if (sz < 4 || !inside(so, sz)) continue;
            const uint32_t* code = (const uint32_t*)(e + so);
            if (argc > 2 && sz > maxsz) { maxsz = sz; if (FILE* d = fopen(argv[2], "wb")) { fwrite(code, 1, sz, d); fclose(d); } }
            std::vector<Rdna2Inst> ins; rdna2_walk(code, sz/4, ins);
            shaders++;
            for (auto& in : ins) { histo[{(int)in.fmt, in.opcode}]++; total++;
                                   if (in.fmt == Rdna2Format::Unknown) unknown++; }
        }
    }
    printf("shaders=%d  total_insts=%zu  distinct(fmt,op)=%zu  unknown=%zu\n", shaders, total, histo.size(), unknown);
    // Memory/interp/export formats show op=0x0 (their sub-opcodes are not decoded yet — format count only).
    const char* fn[] = {"SOP2","SOP1","SOPK","SOPC","SOPP","SMEM","VOP2","VOP1","VOPC","VOP3",
                        "VINTRP","DS","MUBUF","MTBUF","MIMG","FLAT","EXP","UNK"};
    for (auto& [k, c] : histo) printf("  %-6s op=0x%-4x x%u\n", fn[k.first], k.second, c);
    return 0;
}
