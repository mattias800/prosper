#pragma once
#include "fixtures/fragment_packet_fixture.hpp"
#include "gpu/recompiler/fragment_resource_packet.hpp"
#include <bit>

namespace prosper::test::fragment_resource_packet {
using namespace prosper::gpu;
inline uint32_t bits(float f) { return std::bit_cast<uint32_t>(f); }
inline void valu(std::vector<uint32_t>& p, uint32_t op, uint32_t dst, uint32_t a, uint32_t vsrc1) {
    p.push_back((op << 25) | (dst << 17) | (vsrc1 << 9) | a);
}
inline void interp(std::vector<uint32_t>& p, uint32_t phase, uint32_t dst, uint32_t channel, uint32_t src) {
    p.push_back(0xc8000000u | (dst << 18) | (phase << 16) | (channel << 8) | src);
}
inline FragmentResourcePacket base() {
    FragmentResourcePacket p;
    auto& g = p.invocation;
    g.slots_available.fill(true); g.export_enabled.fill(1);
    g.mask_state_available = true; g.exec_mask = UINT64_MAX;
    g.quad_topology = FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    g.float_mode = {true, 0x30}; // RNE, preserve F32 input/output denorms
    g.float_flags = {true, true, false};
    // CPU interpreter witness ONLY. GPU companion replaces it after actual feature enablement.
    p.device = {0x4192, true, true};
    for (uint32_t reg = 0; reg <= 24; ++reg) {
        FragmentPacketVgpr column; column.reg = reg;
        for (uint32_t lane = 0; lane < 64; ++lane)
            column.words[lane] = reg == 0 ? bits(float(lane & 1)) :
                                 reg == 1 ? bits(float((lane >> 1) & 1)) : fragment_packet::poison_sentinel;
        g.vgprs.push_back(column);
    }
    return p;
}
inline FragmentResourcePacket chain(bool explicit_lod = false, bool inactive40 = false,
                                    uint32_t data_variant = 0) {
    auto p = base(); auto& g = p.invocation;
    if (inactive40) g.exec_mask &= ~(uint64_t{1} << 40);
    auto& cache = p.parameter_cache;
    cache.available = true; cache.m0 = 0x7fff0000u;
    for (uint32_t quad = 0; quad < 16; ++quad) {
        cache.quad_primitive[quad] = 100 + quad;
        cache.parameters.push_back({100 + quad, 0, 0, bits(explicit_lod ? 0.25f : 0.125f),
                                    bits(0.5f), bits(explicit_lod ? 0.0f : 0.25f)});
        cache.parameters.push_back({100 + quad, 0, 1, bits(explicit_lod ? 0.25f : 0.125f),
                                    bits(explicit_lod ? 0.0f : 0.25f), bits(0.5f)});
    }
    g.sgprs.emplace_back(16, cache.m0); // actual original MOV M0 BEFORE SMEM overwrites s16
    auto& code = g.guest_code;
    code.push_back(0xbefc0310u); // S_MOV_B32 M0,s16
    FragmentPacketBufferRead buffer;
    buffer.pc = static_cast<uint32_t>(code.size());
    buffer.descriptor = {0x1000, 0, 16, 0};
    buffer.words = {bits(0.5f), bits(data_variant ? 0.5f : 0.25f), bits(explicit_lod ? 1.0f : 0.0f), 0};
    code.insert(code.end(), {0xf4000000u | (10u << 18) | (16u << 6), 0xfa000000u});
    code.push_back(0xbf8c0000u); // original full WAITCNT, NOT admission of an unresolved async read
    p.buffers.push_back(buffer);
    for (uint32_t c = 0; c < 4; ++c) g.sgprs.emplace_back(c, buffer.descriptor[c]);
    interp(code, 0, 10, 0, 0); interp(code, 1, 10, 0, 1);
    interp(code, 0, 11, 1, 0); interp(code, 1, 11, 1, 1);
    fragment_packet::vmov(code, 12, 16);
    valu(code, 8, 13, 256 + 12, 12); // actual V_MUL_F32, .5*.5
    valu(code, 3, 14, 17, 13);       // actual V_ADD_F32, loaded s17 + actual v13
    fragment_packet::exp(code, 15, 0x0e0d0b0au); // u,v,mul,add are LIVE numeric raw sinks
    fragment_packet::vmov(code, 12, 18); // actual SMEM-supplied LOD, not baked into expected sample
    FragmentPacketImageRead image;
    image.pc = static_cast<uint32_t>(code.size());
    image.descriptor = {0x2000, (77u << 20) | (3u << 30), 3u << 14,
                        0x90010facu, 0, 0x10, 0, 0}; // 4x4 RGBA32F, base0,last1
    image.sampler = {0xdb, 256u << 12, 1u << 26, 0}; // nearest XY AND nearest mip
    for (uint32_t level = 0; level < 2; ++level) {
        FragmentPacketImageMip mip; mip.width = mip.height = 4u >> level;
        for (uint32_t texel = 0; texel < mip.width * mip.height; ++texel) {
            const float value = float(1 + texel + level * 32 + data_variant * 100);
            mip.texels.push_back({bits(value), bits(value + 1), bits(value + 2), bits(value + 3)});
        }
        image.mips.push_back(std::move(mip));
    }
    for (uint32_t c = 0; c < 8; ++c) g.sgprs.emplace_back(4 + c, image.descriptor[c]);
    for (uint32_t c = 0; c < 4; ++c) g.sgprs.emplace_back(12 + c, image.sampler[c]);
    code.insert(code.end(), {0xf0000000u | ((explicit_lod ? 0x24u : 0x27u) << 18) | 0xf00u | 8u,
                           10u | (20u << 8) | (1u << 16) | (3u << 21)});
    code.push_back(0xbf8c0000u);
    fragment_packet::exp(code, 15, 0x17161514u);
    code.insert(code.end(), {0xd760003cu, 276u | (168u << 9)}); // READLANE s60,v20,40 ignores EXEC
    fragment_packet::vmov(code, 24, 60);
    fragment_packet::exp(code, 1, 24);
    code.push_back(0xbf810000u);
    p.images.push_back(std::move(image));
    return p;
}
inline std::vector<uint32_t> expected_chain(const FragmentResourcePacket& p, bool explicit_lod,
                                           uint32_t data_variant = 0) {
    std::vector<uint32_t> words(64 * 36, 0);
    const bool source_active = ((p.invocation.exec_mask >> 40) & 1) != 0;
    const uint32_t peer = source_active ? bits(float(1 + (explicit_lod ? 32 : 0) + data_variant * 100))
                                        : fragment_packet::poison_sentinel;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool active = (p.invocation.exec_mask >> lane) & 1;
        for (uint32_t exp = 0; exp < 3; ++exp) {
            const uint32_t offset = lane * 36 + exp * 12;
            const uint32_t header[]{1, uint32_t(active), p.invocation.export_enabled[lane], 0,
                                    exp == 2 ? 1u : 15u, 0, 1, 1};
            std::copy(std::begin(header), std::end(header), words.begin() + offset);
        }
        const uint32_t i = lane & 1, j = (lane >> 1) & 1;
        const float u = explicit_lod ? 0.25f + 0.5f * float(i) : 0.125f + 0.5f * float(i) + 0.25f * float(j);
        const float v = explicit_lod ? 0.25f + 0.5f * float(j) : 0.125f + 0.25f * float(i) + 0.5f * float(j);
        const uint32_t n[]{bits(u), bits(v), bits(0.25f), bits(data_variant ? 0.75f : 0.5f)};
        for (uint32_t c = 0; c < 4; ++c) words[lane * 36 + 8 + c] = active ? n[c] : fragment_packet::poison_sentinel;
        // Independent texel census: mip0 x=2*i+j,y=i+2*j; mip1 x=i,y=j.
        const uint32_t texel = explicit_lod ? i + j * 2 : (2 * i + j) + 4 * (i + 2 * j);
        for (uint32_t c = 0; c < 4; ++c)
            words[lane * 36 + 20 + c] = active ? bits(float(1 + texel + (explicit_lod ? 32 : 0) + data_variant * 100 + c))
                                              : fragment_packet::poison_sentinel;
        words[lane * 36 + 32] = active ? peer : fragment_packet::poison_sentinel;
    }
    return words;
}
inline FragmentResourcePacket arithmetic(uint32_t opcode, uint32_t a, uint32_t c, uint8_t mode) {
    auto p = base(); p.invocation.float_mode.value = mode;
    p.invocation.vgprs[0].words.fill(a); p.invocation.vgprs[1].words.fill(c);
    valu(p.invocation.guest_code, opcode, 2, 256, 1);
    fragment_packet::exp(p.invocation.guest_code, 1, 2);
    p.invocation.guest_code.push_back(0xbf810000u);
    return p;
}
} // namespace prosper::test::fragment_resource_packet
