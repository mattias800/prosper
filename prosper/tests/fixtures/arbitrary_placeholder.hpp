// Synthetic integer predicates observe the existing zero placeholder consistently.
#pragma once
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <vector>

namespace prosper::test::arbitrary_placeholder {
using Words = std::vector<uint32_t>;
inline Words with_missing_spill_root(const Words& original, uint32_t scalar) {
    // Define slot0 on one genuine CFG arm, then read the possibly missing spill after the join.
    // Every original instruction byte and relative branch displacement remains unchanged.
    Words result{0xbf068004u,         0xbf840002u,          0xd76100ffu,
                 128u | (128u << 9u), 0xd7600000u | scalar, 511u | (128u << 9u)};
    result.insert(result.end(), original.begin(), original.end());
    return result;
}
inline gpu::ComputeShaderConfig config() {
    gpu::ComputeShaderConfig result;
    result.local_x = result.wave_size = result.native_subgroup_size = 64;
    result.user_sgprs.resize(12);
    complete_compute_launch(result);
    return result;
}
inline gpu::ShaderResourceTable table(uint64_t address = 0) {
    gpu::ShaderResource output;
    output.cls = gpu::ResourceClass::ConstantBuffer;
    output.format = gpu::DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    output.gpu_addr = address;
    output.size = 64 * sizeof(uint32_t);
    gpu::ShaderResourceTable result;
    result.resources.push_back(output);
    return result;
}
inline void append_output(Words& words, uint32_t mask = 106) {
    const Words suffix{
        0xd1010001u, 0x96u | (0x8bu << 9u) | (mask << 18u),   // v_cndmask v1,22,11,mask
        0xe0702000u, 0x80020100u,   // buffer_store_dword v1,v0,s[8:11] idxen
        0xbf810000u,
    };
    words.insert(words.end(), suffix.begin(), suffix.end());
}
inline Words scalar_relay() {
    Words result{0x7e0c023cu, 0x7d840d06u};   // v6=s60; v_cmp_eq_u32 vcc,v6,v6
    append_output(result);
    return result;
}
inline gpu::ShaderResourceTable scalar_input_table() {
    auto result = table();
    auto input = result.resources.front();
    input.binding = 2;
    input.sgpr_base = 4;
    input.size = sizeof(uint32_t);
    result.resources.push_back(input);
    return result;
}
inline Words loaded_scalar_relay() {
    return {
        0xf4200502u, 0xfa000000u,   // s20 = actual scalar-buffer input at binding2[0]
        0x7e0c023cu,   // v6 = unloaded s60
        0x4a0c0c14u,   // v6 = s20 + v6
        0x7d840c14u,   // VCC = (s20 == v6)
        0xd1010001u, 0x01aa0c96u,   // v1 = VCC ? v6 : 22
        0xe0702000u, 0x80020100u, 0xbf810000u,
    };
}
inline Words floating_predicate() {
    Words result{0x7e0c023cu, 0x7c040c80u};   // v6=s60; v_cmp_eq_f32 vcc,0,v6
    append_output(result);
    return result;
}
inline Words cmpx_predicate() {
    return {
        0x7e020296u,   // v1=22 in every lane before EXEC changes
        0x7e0c023cu,   // v6=s60, the existing zero placeholder
        0xbe94047eu,   // save full EXEC in s[20:21]
        0x7da40106u,   // v_cmpx_eq_u32 exec,v6,v0: only logical lane0 remains
        0x7e02028bu,   // v1=11 only in that lane
        0xbefe0414u,   // restore full EXEC
        0xe0702000u, 0x80020100u, 0xbf810000u,
    };
}
inline Words cmpx_expected() {
    Words result(64, 22);
    result[0] = 11;
    return result;
}
inline Words narrowed_writer() {
    Words result{
        0x7e0c023cu,   // v_mov_b32 v6,s60: explicitly unloaded scalar relay
        0xbe94047eu,   // s_mov_b64 s[20:21],exec
        0xd4c1006au, 0x100u | (0xa0u << 9u),   // v_cmp_lt_u32 vcc,v0,32
        0xbe96246au,   // s_and_saveexec_b64 s[22:23],vcc
        0x7e0c0287u,   // v_mov_b32 v6,7 (only the low half)
        0xbefe0414u,   // s_mov_b64 exec,s[20:21]
        0x7d840c87u,   // v_cmp_eq_u32 vcc,7,v6; high-half old v6 is unloaded
    };
    append_output(result);
    return result;
}
inline Words sdwa_predicate() {
    Words result{0x7e0c023cu, 0x7d840cf9u, 0x06869406u};
    // v6=s60; v_cmp_eq_u32_sdwa s[20:21],v6,v6 with DWORD selects.
    append_output(result, 20);
    return result;
}
inline Words loop_relay() {
    Words result{
        0xbe9e0380u,   // s_mov_b32 s30,0; keep the counter outside output descriptor s[8:11]
        0x7d840d06u,   // loop: v_cmp_eq_u32 vcc,v6,v6
        0x7e0c023cu,   // v_mov_b32 v6,s60
        0x801e811eu,   // s_add_u32 s30,s30,1
        0xbf0a831eu,   // s_cmp_lt_u32 s30,3
        0xbf85fffbu,   // s_cbranch_scc1 loop
    };
    append_output(result);
    return result;
}
inline Words scalar_loop_relay() {
    return {
        0xbe9003c1u, 0xbe9103c1u,   // initial s[16:17] = all lanes
        0xbe9e0380u,   // s30 = counter0, outside the output descriptor
        0xbeea0410u,   // loop: VCC = s[16:17]
        0xd1010001u, 0x01a91696u,   // v1 = VCC ? 11 : 22
        0xbe90043cu,   // s[16:17] = unloaded s[60:61], consistently zero
        0x801e811eu, 0xbf0a831eu, 0xbf85fff9u,   // three trips, back to pc3
        0xe0702000u, 0x80020100u, 0xbf810000u,
    };
}
inline Words expected(bool narrowed) {
    Words result(64, 11u);
    if (narrowed)
        for (unsigned lane = 32; lane < 64; ++lane) result[lane] = 22u;
    return result;
}
}   // namespace prosper::test::arbitrary_placeholder
