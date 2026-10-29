// No pointer escapes the synchronous observer; that observer retains only owned content facts.
#include "shared/live/output_write_content.hpp"
#include <cstdint>
#include <span>
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <cstddef>

namespace prosper::frontend {
namespace {
thread_local const OutputWriteContent* current = nullptr;
}
OutputWriteContent::OutputWriteContent(uint64_t advertised, const void* destination, uint64_t bytes,
                                       uint64_t readable_host_bytes)
    : previous_(current), advertised_(advertised),
      effective_(reinterpret_cast<uintptr_t>(destination)) {
    const auto available = std::min(bytes, readable_host_bytes);
    if (destination && available <= SIZE_MAX)
        bytes_ = {static_cast<const uint8_t*>(destination), static_cast<size_t>(available)};
    current = this;
}
OutputWriteContent::~OutputWriteContent() {
    current = previous_;
}
std::span<const uint8_t> OutputWriteContent::for_write(uint64_t address) {
    if (current && (address == current->advertised_ || address == current->effective_))
        return current->bytes_;
    return {};
}
uint64_t hosted_output_bytes(const prosper::gpu::ShaderResource* resource,
                             const void* destination) {
    return resource && destination == resource->host_data ? resource->host_data_size : 0;
}
} // namespace prosper::frontend
