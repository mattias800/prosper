#pragma once
#include <array>
#include <cstdint>

namespace prosper::gpu {
// Observed physical PS USER_DATA register window, not launched SGPRs. RSRC2's guest ABI/count
// and register mapping need separate proof. Present zero is evidence; an absent word is not zero.
struct FragmentEntryFacts {
    bool observed = false;
    std::array<uint32_t, 32> user_data{};
    uint32_t user_data_available = 0;
    bool rsrc2_available = false;
    uint32_t rsrc2 = 0;
    bool operator==(const FragmentEntryFacts&) const = default;
    bool canonical() const {
        if ((!observed && (user_data_available || rsrc2_available)) ||
            (!rsrc2_available && rsrc2)) return false;
        for (uint32_t i = 0; i < user_data.size(); ++i)
            if (!(user_data_available & (uint32_t(1) << i)) && user_data[i]) return false;
        return true;
    }
};
} // namespace prosper::gpu
