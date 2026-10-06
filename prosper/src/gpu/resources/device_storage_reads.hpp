// device_storage_reads.hpp -- one fact about the live render device, for code that builds a
// draw's buffer bindings before the backend sees them.
#pragma once

namespace prosper::gpu {

// True once the render device was created with robustBufferAccess2 and word-granular ranges (the
// backend's "[vk] deterministic storage word reads" line): a load past the end of a buffer binding
// then reads zero. Without it such a load may read zero or any value inside the same VkBuffer,
// which for an arena slice can be another binding's bytes. False until a device exists, so a
// caller that would lean on the guarantee does not lean on it early.
inline bool& device_reads_zero_past_a_binding() {
    static bool certified = false;
    return certified;
}

}   // namespace prosper::gpu
