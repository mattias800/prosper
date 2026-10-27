// Output-only compiler observer port. Neither call supplies facts or returns admission authority.
#pragma once
#include <cstdint>

namespace prosper::gpu {
void observe_arbitrary_placeholder_check();
void observe_arbitrary_placeholder_admission(uint64_t program, uint64_t fresh, uint64_t loop);
} // namespace prosper::gpu
