#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace prosper::frontend {

// Consume an already-read snapshot only AFTER cache admission. The old entry's allocation
// becomes scratch for the next read. No guest read, cache accounting or pixel ownership belongs
// here. prefix_size is the actual readable prefix, which may be shorter than scratch.size().
struct TextureSourceSnapshot {
    std::vector<uint8_t> bytes;
    bool transferred;
};

inline TextureSourceSnapshot take_texture_source_snapshot(
    std::vector<uint8_t>& scratch, std::vector<uint8_t>& inherited,
    size_t prefix_size, bool allow_transfer)
{
    assert(&scratch != &inherited && prefix_size <= scratch.size());
    // Global scratch may hold a much larger preceding texture. Do not spread that allocation
    // into many tiny cache entries whose ledger charges only their readable prefix. Moving may
    // retain no more capacity than the outgoing entry or a fresh exact-prefix allocation.
    const bool transfer = allow_transfer &&
        scratch.capacity() <= std::max(prefix_size, inherited.capacity());
    if (transfer) {
        scratch.swap(inherited);
        inherited.erase(inherited.begin() + prefix_size, inherited.end());
    } else {
        inherited.resize(prefix_size);
        if (prefix_size) std::memcpy(inherited.data(), scratch.data(), prefix_size);
    }
    return {std::move(inherited), transfer};
}

} // namespace prosper::frontend
