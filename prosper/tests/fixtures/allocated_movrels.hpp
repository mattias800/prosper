// Owned launch contract: 24 private VGPRs, no shared VGPRs; s4 is a runtime index.
#pragma once
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
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
inline std::vector<uint32_t> strict_spill_source(bool fallback, bool alias, bool spill) {
    const uint32_t source = fallback ? 0u : 6u;
    const uint32_t destination = alias ? source : 1u;
    std::vector<uint32_t> words{0x7e120300u, 0x7e000287u | (source << 17u)};
    if (spill) words.insert(words.end(), {0xd7610000u | source, 0x00010000u});
    if (fallback)
        words.insert(words.end(), {0xbefc03ffu, UINT32_MAX});
    else
        words.push_back(0xbefc0380u);
    words.insert(words.end(), {0x7e008706u | (destination << 17u), 0xe0702000u,
                               0x80020009u | (destination << 8u), 0xbf810000u});
    return words;   // No unrelated arbitrary predicate: the original strict attempt must reject.
}
inline std::vector<uint32_t> terminating_arms(uint32_t else_register,
                                              bool specialization_prefix = false) {
    std::vector<uint32_t> words{0x7e120300u,
                                0xbf068004u,
                                0xbf840004u,
                                0x7e020287u,
                                0xe0702000u,
                                0x80020109u,
                                0xbf810000u,
                                0x7e00028du | (else_register << 17u),
                                0x7e020300u | else_register,
                                0xe0702000u,
                                0x80020109u,
                                0xbf810000u};
    if (specialization_prefix) words.insert(words.begin(), {0xbf068080u, 0xbf840001u, 0xbf800000u});
    return words;
}
inline gpu::ShaderResourceTable ds_table(bool wide) {
    auto resources = arbitrary_placeholder::table();
    resources.resources[0].stride = wide ? 8 : 4;
    resources.resources[0].size = 64 * resources.resources[0].stride;
    return resources;
}
inline std::vector<uint32_t> ds_boundary(bool wide, bool outside_source = false,
                                         bool outside_destination = false) {
    const uint32_t address = wide ? 5u : 23u;
    const uint32_t data = wide ? (outside_source ? 23u : 22u) : 1u;
    std::vector<uint32_t> words{0x34000083u | (address << 17u),   // private LDS cell: lane * 8
                                0x7e000287u | (data << 17u)};
    if (data + 1 < 24) words.push_back(0x7e00028du | ((data + 1) << 17u));
    words.insert(words.end(), {0xd9340000u, address | (data << 8u),   // ds_write_b64
                               wide ? 0xd9d80000u : 0xd8d80000u,
                               ((outside_destination ? 23u : 1u) << 24u) | address,
                               wide ? 0xe0742000u : 0xe0702000u, 0x80020100u, 0xbf810000u});
    return words;
}
inline std::vector<uint32_t> outside_relative(bool write, bool retry, uint32_t base = 255) {
    std::vector<uint32_t> words{0x7e120300u, 0x7e020287u, 0x7e0a0287u};   // index v9, v1=v5=7
    if (write)
        words.insert(words.end(), {0xd7610005u, 0x00010000u});   // lost v5 is not read by a NOP
    else
        words.push_back(0x7e00028du);   // current VGPR0 = 13
    words.push_back(0xbefc0380u);
    words.push_back(write ? (0x7e008505u | (base << 17u)) : (0x7e028700u | base));
    if (retry) words.insert(words.end(), {0x7e28023cu, 0x7d842914u});
    words.insert(words.end(), {0xe0702000u, 0x80020109u, 0xbf810000u});
    return words;
}
}   // namespace prosper::test::allocated_movrels
