#include "shared/live/output_write_notify.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "shared/live/output_write_content.hpp"

namespace prosper::frontend {
// Hosted/replay resources can write a different live allocation than their advertised range.
// Both identities matter to renderer aliases and to the ordered journal, including cached views
// absent from this dispatch. Do not duplicate the normal guest-backed notification.
void notify_output_write(uint64_t advertised, const void* destination, uint64_t bytes,
                         bool preserving_bytes, uint64_t readable_host_bytes) {
    const OutputWriteContent content(advertised, destination, bytes, readable_host_bytes);
    // Zero-address backing is GPU-internal state (for example GDS), not a guest output.
    if (!advertised) return;
    const auto notify = [&](uint64_t address) {
        if (!address || !bytes) return;
        if (preserving_bytes)
            prosper::gpu::notify_guest_gpu_write_preserving_bytes(address, bytes);
        else
            prosper::gpu::notify_guest_gpu_write(address, bytes);
    };
    notify(advertised);
    const uint64_t effective = reinterpret_cast<uintptr_t>(destination);
    if (effective != advertised) notify(effective);
}

void notify_output_write(const prosper::gpu::ShaderResource* resource, const void* destination,
                         uint64_t bytes, bool preserving_bytes) {
    notify_output_write(resource->gpu_addr, destination, bytes, preserving_bytes,
                        hosted_output_bytes(resource, destination));
}
void notify_metadata_output_write(const prosper::gpu::ShaderResource* resource,
                                  const void* destination, uint64_t bytes) {
    notify_output_write(resource->metadata_addr, destination, bytes, false,
                        destination == resource->dcc_metadata_host_data
                            ? resource->dcc_metadata_host_data_size
                            : 0);
}
} // namespace prosper::frontend
