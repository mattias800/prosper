#pragma once
// Original SMEM/M0/P1/P2/READLANE/EXP program with hand-derived independent raw sinks. Wave
// ordinal is supplied ownership, not a raster append/subgroup/scheduling reconstruction.
#include "fragment_resource_packet_fixture.hpp"
#include "gpu/recompiler/fragment_packet_wave_data.hpp"

namespace prosper::test::fragment_packet_wave {
using namespace prosper::gpu;
namespace resource = fragment_resource_packet;
inline FragmentResourcePacket packet(uint32_t variant = 0) {
    auto p = resource::base();
    auto& g = p.invocation;
    g.vgprs.clear();
    g.exec_mask = variant == 1   ? UINT64_MAX ^ (uint64_t{1} << 40)
                  : variant == 2 ? (uint64_t{1} << 63)
                                 : UINT64_MAX;
    g.export_enabled[7] = 0;
    for (uint32_t reg : {0u, 1u, 8u, 10u, 12u, 13u, 14u}) {
        FragmentPacketVgpr column;
        column.reg = reg;
        for (uint32_t lane = 0; lane < 64; ++lane)
            column.words[lane] = reg == 0   ? resource::bits(float(lane & 1))
                                 : reg == 1 ? resource::bits(float((lane >> 1) & 1))
                                 : reg == 8 ? 0xa5000000u + variant * 0x10000 + lane * 17
                                            : fragment_packet::poison_sentinel;
        if (reg >= 10) column.available_mask = ~g.exec_mask; // only genuine inactive old words
        g.vgprs.push_back(column);
    }
    auto& cache = p.parameter_cache;
    cache.available = true;
    for (uint32_t quad = 0; quad < 16; ++quad) {
        const auto primitive = 1000 + variant * 100 + quad / (variant + 1);
        cache.quad_primitive[quad] = primitive;
        if (quad && primitive != cache.quad_primitive[quad - 1]) cache.m0 |= 1u << (15 + quad);
        if (!quad || primitive != cache.quad_primitive[quad - 1])
            cache.parameters.push_back(
                {primitive, 0, 0, resource::bits(float(1 + variant * 32 + quad / (variant + 1))),
                 resource::bits(2.0f), resource::bits(4.0f)});
    }
    // Hand-owned scalar words, NOT a claim about hardware PS system-SGPR placement.
    g.sgprs = {{16, cache.m0}, {17, 0xbc000000u + variant}};
    FragmentPacketBufferRead buffer;
    buffer.pc = 1;
    buffer.descriptor = {0x1000 + variant * 0x1000, 0, 16, 0};
    buffer.words = {0x91000000u + variant * 0x100, 2, 3, 4};
    for (uint32_t word = 0; word < 4; ++word) g.sgprs.emplace_back(word, buffer.descriptor[word]);
    p.buffers.push_back(buffer);
    g.guest_code = {0xbefc0310u, 0xf4000000u | (10u << 18) | (20u << 6), 0xfa000000u, 0xbf8c0000u};
    resource::interp(g.guest_code, 0, 10, 0, 0);
    resource::interp(g.guest_code, 1, 10, 0, 1);
    fragment_packet::vmov(g.guest_code, 12, 20);
    g.guest_code.insert(g.guest_code.end(), {0xd760003cu, 264u | (168u << 9)});   // s60=v8[lane40]
    fragment_packet::vmov(g.guest_code, 13, 60);
    fragment_packet::vmov(g.guest_code, 14, 17);
    fragment_packet::exp(g.guest_code, 15, 0x0e0d0c0au);
    g.guest_code.push_back(0xbf810000u);
    return p;
}
inline std::vector<uint32_t> expected(uint32_t variant) {
    std::vector<uint32_t> words(64 * 12, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const bool active = variant == 0 || (variant == 1 ? lane != 40 : lane == 63);
        const uint32_t header[]{1, uint32_t(active), uint32_t(lane != 7), 0, 15, 0, 1, 1};
        std::copy(std::begin(header), std::end(header), words.begin() + lane * 12);
        const uint32_t raw[]{resource::bits(float(1 + variant * 32 + (lane / 4) / (variant + 1) +
                                                  2 * (lane & 1) + 4 * ((lane >> 1) & 1))),
                             0x91000000u + variant * 0x100,
                             0xa5000000u + variant * 0x10000 + 40 * 17, 0xbc000000u + variant};
        for (uint32_t channel = 0; channel < 4; ++channel)
            words[lane * 12 + 8 + channel] =
                active ? raw[channel] : fragment_packet::poison_sentinel;
    }
    return words;
}
inline FragmentResourcePacket entry_m0_packet(uint32_t variant) {
    auto p = packet(variant);
    // Same interpolation sinks, but no guest MOV: the genuinely supplied initial M0 is now
    // load-bearing. PCs retain the actual shifted original instruction positions.
    p.invocation.guest_code.erase(p.invocation.guest_code.begin());
    --p.buffers[0].pc;
    p.parameter_cache.entry_m0_available = true;
    p.parameter_cache.entry_m0 = p.parameter_cache.m0;
    return p;
}
inline FragmentResourcePacket scalar_exec_packet(uint32_t variant) {
    auto p = packet(variant);
    const auto mask = p.invocation.exec_mask;
    p.invocation.sgprs.emplace_back(24, static_cast<uint32_t>(mask));
    p.invocation.sgprs.emplace_back(25, static_cast<uint32_t>(mask >> 32));
    p.invocation.exec_mask = 0;   // original S_MOV_B64 must restore both genuine dynamic halves
    p.invocation.guest_code.insert(p.invocation.guest_code.begin(), 0xbefe0418u);   // EXEC=s24:25
    ++p.buffers[0].pc;
    return p;
}
inline std::vector<FragmentPacketWavePlacement> placements(const FragmentPacketKernel& kernel,
                                                           uint32_t count) {
    std::vector<FragmentPacketWavePlacement> places;
    for (uint32_t wave = 0; wave < count; ++wave)
        places.push_back(
            {4 + count * 2 + 19 + (count - 1 - wave) * (kernel.layout.input_words + 25),
             11 + wave * (kernel.layout.output_words + kPacketWaveOutputPrefix + 17)});
    return places;
}
inline std::vector<FragmentPacketWavePlacement>
padding_placements(const FragmentPacketKernel& kernel) {
    auto places = placements(kernel, 3);
    const auto span = kernel.layout.output_words + kPacketWaveOutputPrefix;
    places[0].output_base = 11;
    places[1].output_base = 11 + 2 * span + 17;
    places[2].output_base = 11 + 3 * span + 34;
    // One complete disjoint unowned output span [11+span,11+2*span), allocated by the normal
    // packer. A routing-only fault may write here safely if the ownership guard is omitted;
    // unlike an alias fault, that discriminator has no inter-workgroup overlapping writes.
    return places;
}
}   // namespace prosper::test::fragment_packet_wave
