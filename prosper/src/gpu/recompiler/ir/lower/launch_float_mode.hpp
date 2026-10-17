// launch_float_mode.hpp — the producing launch's FLOAT_MODE and IEEE_MODE for the stage being
// lowered (fragment RSRC1_PS, compute RSRC1, or the vertex/NGG producer). MODE cannot change
// inside an admitted program: s_setreg of MODE, s_round_mode and s_denorm_mode all refuse, so the
// launch value is the value every instruction sees. Missing launch state is unknown, never zero.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>

namespace prosper::gpu::ir {

struct LaunchFloatMode {
    bool mode_available = false;    // FLOAT_MODE[7:0] is the producing launch's
    bool flags_available = false;   // IEEE_MODE is the producing launch's
    uint8_t mode = 0;
    bool ieee = false;
};

inline LaunchFloatMode launch_float_mode(const SpirvCompute& b) {
    LaunchFloatMode m;
    if (b.is_fragment || b.is_fragment_packet()) {
        m.mode_available = b.fragment_float_mode.available;
        m.mode = b.fragment_float_mode.value;
        m.flags_available = b.fragment_float_flags.available;
        m.ieee = b.fragment_float_flags.ieee_mode;
    } else if (b.is_compute && !b.ngg_workgroup_shell) {
        m.mode_available = m.flags_available = b.compute_pgm_rsrc1 != UINT32_MAX;
        m.mode = (b.compute_pgm_rsrc1 >> 12) & 0xff;
        m.ieee = (b.compute_pgm_rsrc1 & (1u << 23)) != 0;
    } else {
        m.mode_available = m.flags_available = b.vertex_float_mode.available;
        m.mode = b.vertex_float_mode.value;
        m.ieee = b.vertex_float_mode.ieee_mode;
    }
    return m;
}

} // namespace prosper::gpu::ir
