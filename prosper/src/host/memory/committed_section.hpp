#pragma once

#include <cstdint>

namespace prosper::host {

// Native commitment/protection observation ONLY; the caller must first authenticate the exact
// section view and keep HLE-managed topology stable. Untracked native remap/protection mutations
// are outside that contract. No reads, commits, currentness, ordering or physical alias authority.
bool committed_section_range(uint64_t address, uint64_t bytes, uint64_t view_base);

} // namespace prosper::host
