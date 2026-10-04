#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace prosper::gpu {
struct BufferSourceReadOutcome {
    size_t requested = 0, copied = 0;
    // Exact copy extent is not mapping/currentness or producer authority. In particular a
    // caller's zero-initialized tail is not part of the bytes this service actually read.
    bool complete() const { return requested && copied == requested; }
};
template <typename ReadablePrefix>
BufferSourceReadOutcome read_buffer_source(uint8_t* destination, uint64_t address, size_t bytes,
                                           ReadablePrefix&& readable_prefix) {
    BufferSourceReadOutcome outcome{bytes, 0};
    if (!destination || !bytes || address > std::numeric_limits<uint64_t>::max() - bytes)
        return outcome;
    const size_t readable = std::forward<ReadablePrefix>(readable_prefix)(address, bytes);
    if (readable > bytes) return outcome;   // malformed policy never grants an oversized host copy
    if (readable) std::memcpy(destination, reinterpret_cast<const void*>(address), readable);
    outcome.copied = readable;
    return outcome;
}
// Common byte-copy primitive. The supplied prefix policy proves the borrowed source range;
// this function supplies no mapping, producer, content or shader-admission authority.
// A short read leaves the destination tail untouched.
template <typename ReadablePrefix>
size_t copy_buffer_source(uint8_t* destination, uint64_t address, size_t bytes,
                          ReadablePrefix&& readable_prefix) {
    return read_buffer_source(destination, address, bytes,
                              std::forward<ReadablePrefix>(readable_prefix))
        .copied;
}
}   // namespace prosper::gpu
