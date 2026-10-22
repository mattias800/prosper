// Existing embedded-table shapes and admission; private to the recompiler.
#pragma once
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {
struct Rdna2Inst;

// PC-relative EMBEDDED-TABLE detection (#273). Compilers put small constant lookup tables (a 4x4
// ordered-dither matrix, etc.) directly after the shader code and address them with an inline-built
// V#: `s_getpc_b64 s[a:a+1]; s_add_u32 sa, <off>, sa; s_addc_u32 sa+1, 0, sa+1; s_mov_b32 sa+2,
// <num_records>; s_mov_b32 sa+3, <cfg>; buffer_load_dwordx4 v, v, s[a:a+3], 0 offen`. The table
// BYTES are inside the code blob we are recompiling, so the load can be folded to a compile-time
// constant lookup. This pass does forward constant propagation over the LINEAR stream (facts are
// killed on any other write; a fact is only used when no branch target enters between its def and
// the load) and returns: mubuf pc -> the table's dwords copied out of the blob.
// CONFIDENCE: MED-HIGH — exact for the compiler idiom (verified against DOLL's dither PS); every
// guard failure falls back to the old reject.
struct PcrelTables {
    std::unordered_map<uint32_t, std::vector<uint32_t>> mubuf;
    std::unordered_map<uint32_t, std::vector<uint32_t>> smem;
    // TYPED consumer of the same idiom (#2859). Sonic Frontiers' three Cyber Space scene kernels
    // build the identical getpc V# and read it with `tbuffer_load_format_x ..., s[0:3], 0 offen`,
    // twice each. Kept in its own map rather than merged into `mubuf` because the two are admitted
    // under different predicates: an MTBUF site is only foldable when its BUF_FMT performs NO
    // conversion (see the case below), and a shared map would let the untyped rules speak for a
    // typed site.
    std::unordered_map<uint32_t, std::vector<uint32_t>> mtbuf;
};

bool rdna2_buffer_format_is_raw_dwords(uint32_t format, uint32_t components);
bool rdna2_buffer_dst_sel_is_identity(uint32_t descriptor_word3, uint32_t components);
PcrelTables detect_pcrel_tables(const std::vector<Rdna2Inst>& ins, const uint32_t* code,
                                size_t dwords, size_t* required_dwords = nullptr);
}   // namespace prosper::gpu
