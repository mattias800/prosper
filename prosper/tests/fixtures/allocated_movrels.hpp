// Owned launch contract: 24 private VGPRs, no shared VGPRs; s4 is a runtime index.
#pragma once
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <vector>
namespace prosper::test::allocated_movrels {
inline gpu::ComputeShaderConfig config(uint32_t offset, uint32_t wave = 64) {
    auto c = arbitrary_placeholder::config();
    c.wave_size = c.local_x = c.native_subgroup_size = wave;
    complete_compute_launch(c);
    c.user_sgprs[4] = offset;
    c.allocated_vgprs = 24;
    c.compute_pgm_rsrc1 = (c.compute_pgm_rsrc1 & ~63u) | (wave == 64 ? 5u : 2u);
    return c;
}
inline std::vector<uint32_t> program(bool overwrite_zero = false, bool relative_write = false) {
    std::vector<uint32_t> words;
    for (uint32_t r = 1; r <= 7; ++r) {
        words.push_back(0x7e0002ffu | (r << 17));
        words.push_back(100 + r);
    }
    words.push_back(0x7e120300u); // v9 = original v0, output index
    if (overwrite_zero) words.insert(words.end(), {0x7e0002ffu, 0x13579bdfu});
    words.insert(words.end(),
                 {
                     0x7e28023cu,
                     0x7d842914u, // independent unloaded s60 predicate forces the guarded retry
                     0xbefc0304u,   // M0 = supplied s4; no compile-time specialization
                 });
    if (relative_write)
        words.insert(words.end(),
                     {
                         0x7e028505u,   // VGPR[1 + M0] = v5, including allocated v21/v23 padding
                         0xbf069404u,   // s_cmp_eq_u32 s4,20; keep distinct dispatcher cases
                         0xbf840001u,   // s_cbranch_scc0 over the nop to the relative read
                         0xbf800000u,
                     });
    words.insert(words.end(), {
                                  0x7e0e8701u,   // v7 = VGPR[1 + M0], reads old v7 when offset is 6
                                  0xe0702000u,
                                  0x80020709u,   // store v7 indexed by original lane v9
                                  0xbf810000u,
                              });
    return words;
}
inline std::vector<uint32_t> compile(uint32_t offset, uint32_t wave = 64,
                                     bool overwrite_zero = false, bool relative_write = false) {
    auto c = config(offset, wave);
    auto words = program(overwrite_zero, relative_write);
    auto resources = arbitrary_placeholder::table();
    return gpu::recompile_compute(words.data(), words.size(), &resources, c);
}
inline std::vector<uint32_t> expected(uint32_t offset, uint32_t wave, bool overwrite_zero = false) {
    std::vector<uint32_t> out(wave);
    for (uint32_t lane = 0; lane < wave; ++lane) {
        out[lane] = offset < 7 ? 101 + offset : 0;
        if (offset >= 23) out[lane] = overwrite_zero ? 0x13579bdfu : lane;
        if (offset == 8) out[lane] = lane;   // v9 retains the original index
    }
    return out;
}
}   // namespace prosper::test::allocated_movrels
