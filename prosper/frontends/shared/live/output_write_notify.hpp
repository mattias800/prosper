// Common publication boundary for compute outputs, shared by the hosted-content observer.
#pragma once
#include <cstdint>
namespace prosper::frontend {
void notify_output_write(uint64_t advertised, const void* destination, uint64_t bytes,
                         bool preserving_bytes = false);
} // namespace prosper::frontend
