// Supply the host topology interface from the authoritative kernel-memory mapping generation.
#include "host/memory/guest_memory_topology.hpp"
#include "hle/memory/renderer_tracked_mapping_cache.hpp"

namespace prosper {
uint64_t guest_memory_mapping_generation(const GuestMappingLease&) {
    return detail::renderer_tracked_mapping_generation().load();
}
} // namespace prosper
