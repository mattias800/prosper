// Same-TU fixture: the logical draw batch's compact-resource-order preflight, moved verbatim
// out of render_runner.h (which includes it at the same point) to keep that file under its
// size ratchet.
#pragma once

// Validate frontend-owned split-resource metadata for the whole logical draw batch. This must run
// before depth-feedback splitting: validating each physical segment independently can let an early
// segment acquire/submit Vulkan work before a malformed later segment is discovered. Compact
// resources always require explicit order metadata; an empty order retains its historical meaning
// that every resource is in R.
inline bool backend_compact_resource_orders_valid(std::span<const BackendDraw> draws) {
    for (const BackendDraw& draw : draws) {
        if (draw.resource_order.empty()) {
            if (draw.B.empty()) continue;
            std::fprintf(stderr, "prosper: compact resources require explicit order metadata\n");
            return false;
        }
        if (draw.resource_order.size() != draw.R.size() + draw.B.size()) {
            std::fprintf(stderr, "prosper: malformed compact resource order\n");
            return false;
        }
        uint32_t next_full = 0;
        uint32_t next_compact = 0;
        for (uint32_t token : draw.resource_order) {
            const bool compact = (token & 0x80000000u) != 0;
            const uint32_t index = token & 0x7fffffffu;
            uint32_t& next = compact ? next_compact : next_full;
            const size_t count = compact ? draw.B.size() : draw.R.size();
            if (index != next || index >= count) {
                std::fprintf(stderr, "prosper: compact resource order is not a permutation\n");
                return false;
            }
            ++next;
        }
    }
    return true;
}
