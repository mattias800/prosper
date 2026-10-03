// Original registered guest programs for the first full attachment transaction. No interpolation,
// system input, resource load, lane-id or helper value is fabricated to make the first recipe pass.
#pragma once
#include "fixtures/ps_pull_model_fixture.hpp"
#include "gpu/execute/fragment_draw_plan.hpp"
#include <bit>

namespace prosper::test::fragment_draw {
namespace g = prosper::gpu;
namespace p = prosper::agc::Pm4;
inline constexpr uint32_t width = 16, height = 12;
inline constexpr uint32_t ieee_rsrc1 = 1u << p::SPI_SHADER_PGM_RSRC1_PS_IEEE_MODE_SHIFT;
inline constexpr std::array<float, 4> color_a{.25f, .5f, .75f, .5f};
inline constexpr std::array<float, 4> color_b{.75f, .25f, .5f, .5f};

inline std::vector<uint32_t> vertex_words() {
    // Original index arithmetic emits the triangle (-1,-1), (3,-1), (-1,3). Every
    // framebuffer pixel center is strictly interior; even dimensions make genuine full quads.
    return {0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
            0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
            0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u};
}
inline std::vector<uint32_t> fragment_words() {
    // A genuine dominating original EXEC=-1. Its initial guest value is NOT guessed from
    // collected coverage, and neither VCC nor SCC is consumed. Read only actual PS user words.
    return {0xbefe04c1u, 0x7e000200u, 0x7e020201u, 0x7e040202u,
            0x7e060203u, 0xf800180fu, 0x03020100u, 0xbf810000u};
}
inline std::vector<uint32_t> distinct_fragment_words(uint32_t variant) {
    auto words = fragment_words();
    // Distinct ORIGINAL S_NOP immediate, before the genuine EXEC writer. This is not a change
    // to a dynamic scalar, a host specialization, or a replacement shader after realization.
    words.insert(words.begin(), 0xbf800000u | (variant & 0xffffu));
    return words;
}
struct Owners {
    ps_pull::Program vs, ps;
};

inline bool realize(g::DrawItem& draw, const std::array<float, 4>& color = color_a,
                    const std::vector<uint32_t>& ps_words = fragment_words(),
                    uint32_t ps_rsrc1 = ieee_rsrc1) {
    prosper::register_builtin_hle();
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    uint64_t address = 0;
    constexpr uint64_t bytes = 0x10000;
    static_assert(sizeof(Owners) <= bytes);
    if (!map ||
        map(reinterpret_cast<uint64_t>(&address), bytes, 2, 0,
            reinterpret_cast<uint64_t>("fragment-draw-programs"), 0) != 0 ||
        address < 0x100000000ull || address % alignof(Owners))
        return false;
    // AGC owns registered addresses until process exit. Each rail owns a fresh mapping, never
    // rewrites registered words, and never treats released VA as another producer generation.
    auto& owner = *std::construct_at(reinterpret_cast<Owners*>(address));
    if (!ps_pull::register_program(owner.vs, true, vertex_words()) ||
        !ps_pull::register_program(owner.ps, false, ps_words))
        return false;
    g::GpuState state;
    for (const auto* program : {&owner.vs, &owner.ps})
        for (const auto& reg : program->registers) state.sh[reg.offset] = reg.value;
    state.uc[p::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[p::CB_TARGET_MASK] = state.cx[p::CB_SHADER_MASK] = 15;
    state.cx[p::SPI_PS_IN_CONTROL] = 0;   // actual Wave64
    state.cx[p::SPI_BARYC_CNTL] = 0;
    state.cx[p::SPI_PS_INPUT_ENA] = state.cx[p::SPI_PS_INPUT_ADDR] = 0;   // actual input-free ABI
    // Supply the actual launch register BEFORE realization. The packet compiler currently
    // requires IEEE mode; a physically supplied mode0 peer remains a named negative.
    state.sh[p::SPI_SHADER_PGM_RSRC1_PS] = ps_rsrc1;
    state.sh[p::SPI_SHADER_PGM_RSRC2_PS] = 4u << p::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT;
    for (uint32_t word = 0; word < color.size(); ++word)
        state.sh[p::SPI_SHADER_USER_DATA_PS_0 + word] = std::bit_cast<uint32_t>(color[word]);
    g::GpuState::Draw packet;
    packet.index_count = 3;
    packet.instance_count = 1;
    packet.command_order = 1;
    state.draws.push_back(packet);
    return g::realize_draw_item(state, &state.draws[0], 3, 64, false, draw, nullptr, true);
}
inline std::shared_ptr<const g::FragmentPacketPreparation> prepare(const g::DrawItem& draw) {
    return g::prepare_fragment_packet_inputs(
        draw.fragment_draw_inputs, draw.fragment_draw_inputs &&
                                       draw.fragment_draw_inputs->source_vs &&
                                       draw.fragment_draw_inputs->source_fs &&
                                       *draw.fragment_draw_inputs->source_vs == draw.vs_words() &&
                                       *draw.fragment_draw_inputs->source_fs == draw.fs_words());
}
}   // namespace prosper::test::fragment_draw
