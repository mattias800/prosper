// Hand-built SAVEEXEC/M0 lifetime programs. The loop uses only Bool masks; an old opaque
// scalar value must neither block that fresh lifetime nor become a fabricated numeric word.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace prosper::test::saveexec_m0 {
inline std::vector<uint32_t> loop(uint32_t saved_word = 14) {
    std::vector<uint32_t> code{
        0xbe80037cu | (saved_word << 16), // s_mov_b32 sN,m0
        0xb07c0000u,                     // s_movk_i32 m0,0
        0xbefc0300u | saved_word,         // restore entry M0
        0xbe8e247eu,                     // s_and_saveexec_b64 s[14:15],exec
        0xbefe040eu,                     // restore full EXEC
        0xbe810380u,                     // s_mov_b32 s1,0
        0xbf0d8401u,                     // header: s_cmp_lg_u32 s1,4
        0xbf840003u,                     // s_cbranch_scc0 -> end
        0xbe8e047eu,                     // body: s_mov_b64 s[14:15],exec
        0x80018101u,                     // s_add_u32 s1,s1,1
        0xbf82fffbu,                     // s_branch -> header
        0xbf810000u,
    };
    if (saved_word == 15) {
        // Make the high word a separate loop-carried scalar destination, then replace
        // the complete pair before its mask view is reused. Neither old M0 half survives.
        code.insert(code.begin() + 8, 0xbe8f0380u); // s_mov_b32 s15,0
        code[7] = 0xbf840004u;
        code[11] = 0xbf82fffau;
    }
    return code;
}

inline uint64_t address() {
    static std::atomic<uint64_t> next{0x4899000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}
inline std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t diagnostic,
                                     uint32_t native_width = 0) {
    prosper::gpu::ComputeShaderConfig config;
    config.wave_size = 64;
    config.local_x = 64;
    config.native_subgroup_size = native_width;
    return prosper::gpu::recompile_compute(
        code.data(), code.size(), nullptr, config,
        {prosper::gpu::RecompileDiagnosticStage::Compute, diagnostic});
}
} // namespace prosper::test::saveexec_m0
