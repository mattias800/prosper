#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace prosper::gpu {
// Common byte-copy primitive. The supplied prefix policy proves the borrowed source range;
// this function supplies no mapping, producer, content or shader-admission authority.
// A short read leaves the destination tail untouched.
template <typename ReadablePrefix>
size_t copy_buffer_source(uint8_t* destination, uint64_t address, size_t bytes,
                          ReadablePrefix&& readable_prefix) {
    const size_t readable = std::forward<ReadablePrefix>(readable_prefix)(address, bytes);
    if (readable) std::memcpy(destination, reinterpret_cast<const void*>(address), readable);
    return readable;
}
} // namespace prosper::gpu
