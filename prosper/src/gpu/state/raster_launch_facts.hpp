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
    bool canonical() const {
        return (ps_in_control_available || !ps_in_control) &&
            (baryc_cntl_available || !baryc_cntl) &&
            (input_ena_available || !input_ena) && (input_addr_available || !input_addr);
    }
    bool operator==(const RasterLaunchFacts&) const = default;
};
} // namespace prosper::gpu
