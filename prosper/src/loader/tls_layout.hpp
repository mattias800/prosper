// tls_layout.hpp — pure x86-64 Variant-II static-TLS layout shared by the linker and allocator.
#pragma once

#include "../hle/dispatch.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper {

struct StaticTlsLayout {
    // Distance from the thread pointer to each module's block start. Index 0 is reserved.
    std::vector<uint64_t> module_below;
    uint64_t total_below = 0;
};

inline uint64_t static_tls_align_up(uint64_t value, uint64_t align) {
    return align ? (value + align - 1) & ~(align - 1) : value;
}

// x86-64 Variant II (TLS_TCB_AT_TP): modules are packed below TP in module-id order. A symbol at
// in-block byte S therefore has a thread-pointer-relative offset S - module_below[modid]. Keeping
// this calculation shared is correctness-critical: the linker-baked TPOFF64 value and the runtime
// allocator must describe the same byte.
inline StaticTlsLayout make_static_tls_layout(const TlsModuleDesc* descs, size_t count) {
    StaticTlsLayout layout;
    layout.module_below.assign(count, 0);
    if (!descs || count == 0) return layout;

    uint64_t offset = 0;
    uint64_t max_align = 16;
    for (size_t i = 1; i < count; i++) {
        const uint64_t align = descs[i].align ? descs[i].align : 16;
        if (align > max_align) max_align = align;
        offset = static_tls_align_up(offset + descs[i].memsz, align);
        layout.module_below[i] = offset;
    }
    layout.total_below = static_tls_align_up(offset, max_align < 64 ? 64 : max_align);
    return layout;
}

} // namespace prosper
