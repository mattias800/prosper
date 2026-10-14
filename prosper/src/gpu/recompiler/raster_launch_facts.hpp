#pragma once
#include <cstdint>

namespace prosper::gpu {
// Raw producing launch facts. Availability is not inferred from a zero value, module metadata,
// or a host subgroup. Retaining these does not assert a PS5 wave-composition policy.
struct RasterLaunchFacts {
    bool ps_in_control_available = false, baryc_cntl_available = false;
    uint32_t ps_in_control = 0, baryc_cntl = 0;
    bool input_ena_available = false, input_addr_available = false;
    uint32_t input_ena = 0, input_addr = 0;
    // Complete producing context words, not decoded scheduling/helper authority. SC controls
    // sample iteration, coverage and wave-break/collision inputs; DB controls depth/PS execution.
    // A consumer must prove the domain it uses. An unwritten register is not a hardware reset value.
    bool sc_shader_control_available = false;
    uint32_t sc_shader_control = 0;
    bool sc_mode_cntl_0_available = false, sc_mode_cntl_1_available = false;
    uint32_t sc_mode_cntl_0 = 0, sc_mode_cntl_1 = 0;
    bool sc_aa_config_available = false, db_shader_control_available = false;
    uint32_t sc_aa_config = 0, db_shader_control = 0;
    bool canonical() const {
        return (ps_in_control_available || !ps_in_control) &&
               (baryc_cntl_available || !baryc_cntl) && (input_ena_available || !input_ena) &&
               (input_addr_available || !input_addr) &&
               (sc_shader_control_available || !sc_shader_control) &&
               (sc_mode_cntl_0_available || !sc_mode_cntl_0) &&
               (sc_mode_cntl_1_available || !sc_mode_cntl_1) &&
               (sc_aa_config_available || !sc_aa_config) &&
               (db_shader_control_available || !db_shader_control);
    }
    bool operator==(const RasterLaunchFacts&) const = default;
};
} // namespace prosper::gpu
