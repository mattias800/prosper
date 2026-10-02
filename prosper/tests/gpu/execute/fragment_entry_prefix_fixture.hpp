#pragma once
#include "gpu/recompiler/raster_quad_collector.hpp"
#include <algorithm>
#include <iterator>

namespace prosper::test::entry_prefix {
// Project-owned raw MOV -> EXP. This is not a guest raster-packing or live admission oracle.
inline std::vector<uint32_t> guest(uint32_t source) {
    return {0x7e000200u | source, 0xf8001801u, 0x03020100u, 0xbf810000u};
}
inline std::shared_ptr<gpu::RasterQuadInputs> inputs(uint32_t count, uint32_t source = 0) {
    auto in = std::make_shared<gpu::RasterQuadInputs>();
    in->raw_code = std::make_shared<const std::vector<uint32_t>>(guest(source));
    // Only a nonempty selected-owner token is needed by this preflight fixture. Actual module
    // equality/realization is independently covered by test_raster_quad_collection.
    in->source_fs = std::make_shared<const std::vector<uint32_t>>(std::initializer_list<uint32_t>{0x07230203u});
    in->raw_matches_producing_source = true;
    in->entry.observed = true;
    in->entry.rsrc2_available = true;
    // Specify the hardware encoding independently of the production PM4 extraction constants.
    in->entry.rsrc2 = ((count & 31u) << 1) | ((count >> 5) << 27);
    in->entry.user_data_available = UINT32_MAX;
    for (uint32_t reg = 0; reg < 32; ++reg) in->entry.user_data[reg] = 0x42090000u + reg * 17;
    in->entry.user_data[0] = 0; // present zero, never an absent-word default
    return in;
}
inline bool gap(const gpu::FragmentPacketPreparation& p, const char* reason) {
    return std::find(p.unmet.begin(), p.unmet.end(), reason) != p.unmet.end();
}
inline gpu::FragmentInvocationPacket packet(const gpu::FragmentPacketPreparation& prepared) {
    gpu::FragmentInvocationPacket p;
    p.guest_code = *prepared.inputs->raw_code;
    p.sgprs = prepared.initial_user_sgprs; // the shipping preparation's owned authority, not raw UD
    p.slots_available.fill(true);
    p.export_enabled.fill(1);
    p.mask_state_available = true;
    p.exec_mask = UINT64_MAX;
    // The packet is supplied explicitly. Neither initialization nor these 64 slots is inferred
    // from the raster collector; EXP's complete raw payload has four genuinely available words.
    for (uint32_t reg = 0; reg < 4; ++reg) {
        gpu::FragmentPacketVgpr v;
        v.reg = reg;
        v.words.fill(0xdeadbeefu);
        p.vgprs.push_back(v);
    }
    return p;
}
inline std::vector<uint32_t> expected(uint32_t value) {
    std::vector<uint32_t> out(64 * gpu::kFragmentPacketExportWords, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const size_t base = lane * gpu::kFragmentPacketExportWords;
        const uint32_t header[] = {1, 1, 1, 0, 1, 0, 1, 1};
        std::copy(std::begin(header), std::end(header), out.begin() + base);
        out[base + 8] = value;
    }
    return out;
}
} // namespace prosper::test::entry_prefix
