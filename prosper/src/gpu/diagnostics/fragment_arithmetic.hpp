#pragma once
#include "gpu/recompiler/fragment_arithmetic_observation.hpp"

namespace prosper::gpu {
// Exactly one call per public direct or cached fragment compiler request. A nested direct compile
// that returns its observation to a cache does not announce/count again. Refused requests count
// when ADD/MUL lowering was actually reached; neither this nor module production proves execution.
void observe_fragment_arithmetic(const FragmentArithmeticObservation& observation,
                                 uint64_t lookup_program, bool module_produced);
} // namespace prosper::gpu
