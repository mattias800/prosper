#pragma once

#include <cstddef>

namespace prosper::libc {

// Share the registered handler's ownership rule with tests whose allocator records releases.
// A portable test cannot require a host allocator to reuse a freed address. The allocator retains
// its own zero-size policy; a null result at size zero may already
// have released the original block.
template <typename Resize, typename Release>
void* realloc_free_on_failure(void* original, std::size_t size, Resize resize, Release release) {
    void* result = resize(original, size);
    if (!result && original && size) release(original);
    return result;
}

} // namespace prosper::libc
