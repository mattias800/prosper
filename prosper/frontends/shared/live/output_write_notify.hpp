// Common publication boundary for compute outputs, shared by the hosted-content observer.
#pragma once
#include <cstdint>
namespace prosper::gpu {
struct ShaderResource;
}
namespace prosper::frontend {
void notify_output_write(uint64_t advertised, const void* destination, uint64_t bytes,
                         bool preserving_bytes = false, uint64_t readable_host_bytes = 0);
void notify_output_write(const prosper::gpu::ShaderResource*, const void* destination,
                         uint64_t bytes, bool preserving_bytes = false);
void notify_metadata_output_write(const prosper::gpu::ShaderResource*, const void* destination,
                                  uint64_t bytes);
} // namespace prosper::frontend
