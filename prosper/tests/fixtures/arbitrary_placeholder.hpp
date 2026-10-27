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
inline Words readfirst_loop_relay() {
    return {
        0x7e0c0283u, 0xbe9e0380u,   // genuine preheader v6=3, scalar counter=0
        0x7e280506u,   // header2: s20 = readfirstlane(v6)
        0x7d840d06u,   // fresh VCC observes the carried v6
        0x7e0c023cu,   // v6 = unloaded s60, consistently zero
        0x4a0c0106u, 0x4a0c0c87u,   // v6 = v6 + local lane + 7
        0x801e811eu, 0xbf0a831eu, 0xbf85fff8u,   // three trips back to header2
        0x7e020214u,   // v1 = actual final scalar publication, 7 from lane0
        0xe0702000u, 0x80020100u, 0xbf810000u,
    };
}
inline Words readfirst_straight_relay() {
    Words result{0x7e0c023cu, 0x4a0c0106u};   // v6 = unloaded s60 + logical lane
    result.insert(result.end(), {0x4a0c0c87u, 0x7e280506u, 0x7d840c14u});
    // v6 += 7; s20 = readfirstlane(v6); VCC = (s20 == v6)
    append_output(result);
    return result;
}
inline Words agreeing_readfirst_relay() {
    Words result{0x7e0c0287u, 0x7e280506u,   // v6 = 7; s20 = readfirstlane(v6)
                 0x7e0e023cu, 0x7d840f07u,   // v7 = unloaded s60; VCC = (v7 == v7)
                 0x7d840c14u};   // VCC = (s20 == v6)
    append_output(result);
    return result;
}
inline Words readfirst_independent_relay(bool after) {
    const Words predicate{0x7e0c023cu, 0x7d840d06u};   // v6 = s60; VCC = (v6 == v6)
    const Words publication{0x4a0c0087u, 0x7e280506u};   // v6 = lane + 7; s20 = first(v6)
    Words result = after ? predicate : publication;
    const auto& second = after ? publication : predicate;
    result.insert(result.end(), second.begin(), second.end());
    result.insert(result.end(), {0x7e020214u, 0xe0702000u, 0x80020100u, 0xbf810000u});
    return result;   // store s20 directly, independently of the admitted predicate
}
inline Words data_after_predicate(bool unknown = false, bool before = false) {
    const uint32_t data = 0x7e020200u | (unknown ? 12u : 60u);   // v1 = real-unknown / unloaded
    Words result{0x7e0c023cu, 0x7d840d06u};   // unloaded-only fresh predicate
    result.insert(before ? result.begin() : result.end(), data);
    result.insert(result.end(), {0xe0702000u, 0x80020100u, 0xbf810000u});
    return result;
}
inline Words readlane_relay() {
    Words result{0x7e0c0302u, 0xd7600014u, 0x00010106u,   // v6=v2; s20=readlane(v6,0)
                 0x7d840c14u};   // VCC=(s20==v6), a scalar relay of the unloaded word
    append_output(result);
    return result;
}
inline Words pixel_export_after_predicate(bool unknown) {
    return {0x7e0c0302u, 0xd7600014u,
            0x00010106u, 0x7d840c14u,
            0xd1010006u, 0x01a91696u,   // v6 = VCC ? 11 : 22
            0xf8000803u, unknown ? 0x00000006u : 0x00000206u,   // R=v6, G=launch v0 / unloaded v2
            0xbf810000u};
}
inline Words implicit_compare_after_predicate(bool unknown = false) {
    return {0x7e020296u,
            0x7e0c023cu,
            0x7d840d06u,
            0xb4800000u | ((unknown ? 12u : 60u) << 16u),   // s_cmpk_eq_u32 s12/s60,0
            0xbf850001u,
            0x7e02028bu,   // keep22 if zero, otherwise write11
            0xe0702000u,
            0x80020100u,
            0xbf810000u};
}
inline Words pixel_store_after_predicate(bool unknown) {
    return {0x7e0c0302u, 0xd7600014u,
            0x00010106u, 0x7d840c14u,
            0xd1010007u, 0x01a90081u,   // v7 = VCC ? 0 : 1, the owned output index
            0xe0702000u, unknown ? 0x80020007u : 0x80020207u,   // store launch v0 / unloaded v2
            0xf800080fu, 0x06060606u,
            0xbf810000u};
}
inline Words dropped_definition_after_join() {
    Words result{0xbf068004u, 0xbf850001u, 0xbea80387u,   // if (s4 != 0) s40=7
                 0x7e0c0228u, 0x7d840c87u};   // v6=s40; VCC=(v6==7)
    append_output(result);
    return result;
}
inline Words untracked_spill_in_loop() {
    return {0xbf068004u, 0xbf840002u, 0xd76100ffu, 0x00010000u,   // one path writes slot0 from s0
            0x7e0c023cu, 0x7d840d06u, 0xbe940380u, 0xbe9e0380u,   // unloaded predicate, s20=s30=0
            0x7e020214u, 0xe0702000u, 0x80020100u,   // loop: output previous s20
            0xd7600014u, 0x000101ffu,   // s20 = possibly missing spill slot0
            0x801e811eu, 0xbf0a831eu, 0xbf85fff8u, 0xbf810000u};
}
inline Words written_unloaded_spill() {
    return {0x7e0c023cu, 0x7d840d06u,   // unloaded-only predicate
            0xd76100ffu, 0x0001003cu,   // actually write slot0 from unloaded s60
            0xd76100ffu, 0x0001023cu,   // preserve slot0 while also writing slot1
            0xd7600014u, 0x000101ffu,   // read that provided slot into s20
            0x7e020214u, 0xe0702000u, 0x80020100u, 0xbf810000u};
}
inline Words addtid_after_writelane(bool write_lane = true) {
    Words result{0x7e0c0287u};   // v6 = 7
    if (write_lane) result.insert(result.end(), {0xd7610006u, 0x00010000u});   // lane0 = s0
    result.insert(result.end(),
                  {0xb07c0000u, 0xdac00000u, 0x00000600u, 0xbf8cc07fu,   // M0=0; spill v6
                   0xdac40000u, 0x01000000u, 0xbf8cc07fu,   // reload v1
                   0x7e0e023cu, 0x7d840f07u,   // unloaded v7; VCC=(v7==v7)
                   0xd1010001u, 0x01aa0296u,   // v1 = VCC ? v1 : 22
                   0xe0702000u, 0x80020100u, 0xbf810000u});
    return result;
}
inline Words partial_old_after_writelane(bool write_lane = true) {
    Words result{0x7e0c0287u};   // v6 = 7
    if (write_lane) result.insert(result.end(), {0xd7610006u, 0x00010000u});
    result.insert(result.end(),
                  {0x7da40080u, 0x7e0c0296u,   // only lane0 overwrites v6 with 22
                   0xbefe04c1u, 0x7e0e023cu, 0x7d840f07u,   // full EXEC; unloaded equality
                   0xd1010001u, 0x01aa0c96u,   // select v6, including its old inactive lanes
                   0xe0702000u, 0x80020100u, 0xbf810000u});
    return result;
}
inline Words partial_old_expected() {
    Words result(64, 7);
    result[0] = 22;
    return result;
}

inline Words relative_after_writelane(bool write_lane = true, bool provide_word = true,
                                      bool alias_destination = false) {
    Words result{0x7e0a0285u};   // v5=5; M0 can also select explicitly unloaded v7/v8
    if (provide_word) result.push_back(0x7e0c0287u);   // v6=7
    if (write_lane) {
        result.push_back(0xd7610006u);   // v_writelane v6,s0,0 retires ordinary v6 storage
        result.push_back(0x00010000u);
    }
    const Words suffix{
        0x877c8304u,   // M0=s4&3: runtime index, supplied s4=1 selects v6
        alias_destination ? 0x7e0c8705u : 0x7e028705u,   // VDST=VGPR[5+M0]
        0x7e12023cu,
        0x7d841309u,   // independent unloaded s60 reflexive predicate
        0xe0702000u,
        alias_destination ? 0x80020600u : 0x80020100u,
        0xbf810000u,
    };
    result.insert(result.end(), suffix.begin(), suffix.end());
    return result;
}
inline Words bfe_scc_exec(bool unloaded_pair = false) {
    Words result{0x7e0c023cu, 0x7d840d06u, 0xd1010001u, 0x01a91696u};
    const Words suffix{
        0x9494ff00u | (unloaded_pair ? 60u : 12u),
        0x00010020u,   // extract bit32 into s[20:21]
        0x85fe80c1u,   // SCC selects full EXEC or zero EXEC
        0x7e020287u,
        0xbefe04c1u,   // v1=7 under EXEC, then restore full EXEC
        0xe0702000u,
        0x80020100u,
        0xbf810000u,
    };
    result.insert(result.end(), suffix.begin(), suffix.end());
    return result;
}
inline Words implicit_scc_exec(unsigned producer = 0) {
    Words result{0x7e0c023cu, 0x7d840d06u, 0xd1010001u, 0x01a91696u};
    if (producer == 1) result.push_back(0xbf068004u);   // SCC=(supplied s4==0)
    if (producer == 2) result.push_back(0xbf063c3cu);   // SCC=(unloaded s60==itself)
    const Words suffix{0x85fe80c1u, 0x7e020287u, 0xbefe04c1u,
                       0xe0702000u, 0x80020100u, 0xbf810000u};
    result.insert(result.end(), suffix.begin(), suffix.end());
    return result;
}
inline Words expected(bool narrowed) {
    Words result(64, 11u);
    if (narrowed)
        for (unsigned lane = 32; lane < 64; ++lane) result[lane] = 22u;
    return result;
}
}   // namespace prosper::test::arbitrary_placeholder
