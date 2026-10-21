// Project-owned service-selection witnesses for the eleven empty #5454 census rows.
// These compile profiles assert dispatcher observations, not arithmetic or raster results.
#pragma once

#include "gpu/recompiler/float32_rte_support.hpp"
#include "gpu/recompiler/float64_support.hpp"
#include "gpu/recompiler/ir/cfg/service_nodes.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <vector>

namespace prosper::test::cfg_services {
using gpu::ir::cfg::ServiceEvent;
using gpu::ir::cfg::ServiceKind;
enum class Profile { Compute, Fragment, Packet };
struct Witness {
    const char* name;
    Profile profile;
    ServiceKind target;
    std::vector<uint32_t> code;
    std::vector<ServiceEvent> expected;
};

inline const std::vector<Witness>& witnesses() {
    static const std::vector<Witness> values = [] {
        // Save full EXEC, initialize LDS[0] in lane zero, then restore before both float atomics.
        const std::vector<uint32_t> lds = {0xbe862481u, 0x7e000280u, 0x7e0202ffu, 0x7f800000u,
                                           0xd8340000u, 0x00000100u, 0xbefe0406u, 0x7e000280u,
                                           0x7e0402f2u, 0xd8480000u, 0x00000200u, 0xd84c0000u,
                                           0x00000200u, 0xbf810000u};
        auto store = lds;
        store.erase(store.begin() + 11, store.begin() + 13);   // One MIN consumer needs the store.
        std::vector<Witness> out = {
            // M0 owns the append address. Initial counter contents are not a value oracle.
            {"Append",
             Profile::Compute,
             ServiceKind::Append,
             {0xbefc0380u, 0xd8f80008u, 0x03000000u, 0xbf810000u},
             {{1, ServiceKind::Append}}},
            {"LdsStore",
             Profile::Compute,
             ServiceKind::LdsStore,
             store,
             {{4, ServiceKind::LdsStore}, {9, ServiceKind::LdsFminmax}}},
            {"LdsFminmax",
             Profile::Compute,
             ServiceKind::LdsFminmax,
             lds,
             {{4, ServiceKind::LdsStore},
              {9, ServiceKind::LdsFminmax},
              {11, ServiceKind::LdsFminmax}}},
            {"Bpermute",
             Profile::Compute,
             ServiceKind::Bpermute,
             {0x7e020281u, 0x7e040280u, 0xdacc0000u, 0x03000102u, 0xbf810000u},
             {{2, ServiceKind::Bpermute}}},
            // Ordinary compute inputs are declared local-ID v0; all other sources have producers.
            {"ComputeDppRowRor8",
             Profile::Compute,
             ServiceKind::ComputeDppRowRor8,
             {0x7e0402f2u, 0x7e0602f4u, 0x060206fau, 0xff092802u, 0xbf810000u},
             {{2, ServiceKind::ComputeDppRowRor8}}},
            {"MaskFfbh",
             Profile::Compute,
             ServiceKind::MaskFfbh,
             {0x7d8800f9u, 0x06868891u, 0x7e147209u, 0xbf810000u},
             {{2, ServiceKind::MaskFfbh}}},
            {"MaskReduction",
             Profile::Compute,
             ServiceKind::MaskReduction,
             {0x7d8800f9u, 0x06868891u, 0xbe841008u, 0xbe851408u, 0xbf810000u},
             {{2, ServiceKind::MaskReduction}, {3, ServiceKind::MaskReduction}}},
            {"SavedMaskPairCompare",
             Profile::Compute,
             ServiceKind::SavedMaskPairCompare,
             {0x7d8800f9u, 0x06868691u, 0x7d8800f9u, 0x0686889fu, 0xbf130806u, 0xbf810000u},
             {{4, ServiceKind::SavedMaskPairCompare}}},
            {"VopcMaskZeroCompare",
             Profile::Compute,
             ServiceKind::VopcMaskZeroCompare,
             {0x7d840100u, 0xd4e4006au, 0x0001006au, 0xbf810000u},
             {{1, ServiceKind::VopcMaskZeroCompare}}},
            // Only masks/topology are entry inputs: V_MOV defines v2 before its live EXP.
            {"PacketWqm",
             Profile::Packet,
             ServiceKind::PacketWqm,
             {0xbefe0a7eu, 0x7e040280u, 0xf8001801u, 0x00000002u, 0xbf810000u},
             {{0, ServiceKind::PacketWqm}}},
            {"FragmentDppMinRowShr",
             Profile::Fragment,
             ServiceKind::FragmentDppMinRowShr,
             {0x7e060287u, 0x260606fau, 0xff011103u},
             {{1, ServiceKind::FragmentDppMinRowShr}}}};
        // Crossing scalar regions select the existing fragment dispatcher. Both exports read
        // explicitly initialized vectors; no interpolation or fabricated register input is needed.
        const std::vector<uint32_t> tail = {0x7e040280u, 0x7e060280u, 0x7e080280u, 0x7e0a0280u,
                                            0xbf068080u, 0xbf840003u, 0xbf068080u, 0xbf840002u,
                                            0x7e040281u, 0x7e060281u, 0xbf068080u, 0xbf840003u,
                                            0xf800180fu, 0x05040302u, 0xbf820003u, 0xf800180fu,
                                            0x05040302u, 0xbf810000u, 0xbf810000u};
        out.back().code.insert(out.back().code.end(), tail.begin(), tail.end());
        return out;
    }();
    return values;
}

inline void publish_support() {
    gpu::publish_float64_support(true, true, true);
    gpu::publish_float32_rte_support(true, true, true);
}

inline gpu::ComputeShaderConfig compute_config() {
    gpu::ComputeShaderConfig config;
    config.local_x = config.wave_size = 64;
    config.force_exchange_dispatcher = true;
    return config;
}

inline gpu::FragmentInvocationPacket packet(const Witness& w) {
    gpu::FragmentInvocationPacket packet;
    packet.guest_code = w.code;
    packet.slots_available.fill(true);
    packet.export_enabled.fill(1);
    packet.mask_state_available = true;
    packet.exec_mask = uint64_t(1) << 63;
    packet.vcc_mask = uint64_t(1) << 40;
    packet.quad_topology = gpu::FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
    return packet;
}

inline std::vector<uint32_t> compile(const Witness& w) {
    if (w.profile == Profile::Compute)
        return gpu::recompile_compute(w.code.data(), w.code.size(), nullptr, compute_config());
    if (w.profile == Profile::Fragment) {
        const gpu::FragmentInterpolationLayout interpolation;
        return gpu::recompile_fragment(w.code.data(), w.code.size(), nullptr, nullptr, UINT32_MAX,
                                       &interpolation);
    }
    if (w.profile != Profile::Packet) return {};
    return gpu::recompile_fragment_packet(packet(w)).spirv;
}
}   // namespace prosper::test::cfg_services
