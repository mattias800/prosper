#pragma once
#include "gpu/state/fragment_entry_facts.hpp"

namespace prosper::gpu {
struct GpuState;
// Observe the same physical producing registers; caller initializes the facts. This does not
// certify source/command-order association, launched SGPRs or resource backing.
void observe_fragment_entry(const GpuState& ds, bool has_source, FragmentEntryFacts& entry);
}   // namespace prosper::gpu
