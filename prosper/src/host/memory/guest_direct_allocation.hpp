#pragma once
#include <cstdint>

namespace prosper {
// Host-facing immutable allocation observation, not capture/currentness or mapping authority.
// HLE owns its producer and topology APIs; retaining this value does not keep a VA mapped.
struct GuestDirectAllocation {
    uint64_t address = 0, minimum_bytes = 0;
    uint64_t physical_begin = 0, physical_end = 0, identity = 0;
};
}   // namespace prosper
