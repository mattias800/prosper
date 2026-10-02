#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace prosper::detail {

template<size_t Capacity>
struct StagedStub {
    std::array<uint8_t, Capacity> bytes{};
    size_t length = 0;
};

// Callers bound the slot count/address geometry before staging. Every emitter runs against its
// actual scratch capacity, and the entire suffix must fit before any backing or live bytes change.
template<size_t Capacity, class Emit>
bool stage_stub_suffix(std::vector<StagedStub<Capacity>>& staged, size_t first_new, size_t total,
                       uint64_t stride, Emit emit, const char* capacity_error,
                       const char* stride_error, std::string* err) {
    auto fail = [&](const char* text) { if (err) *err = text; return false; };
    const size_t added = total - first_new;
    if (added > staged.max_size()) return fail("import stub staging allocation failed");
    try {
        staged.resize(added);
    } catch (const std::bad_alloc&) {
        return fail("import stub staging allocation failed");
    }
    for (size_t j = 0; j < added; j++) {
        auto& stub = staged[j];
        stub.length = emit(stub.bytes, first_new + j);
        if (stub.length > stub.bytes.size()) return fail(capacity_error);
        if (stub.length > stride) return fail(stride_error);
    }
    return true;
}

// After backing succeeds, publication copies only validated lengths: no allocation or refusal.
template<size_t Capacity>
void publish_stub_suffix(const std::vector<StagedStub<Capacity>>& staged, uint64_t base,
                         uint64_t stride, size_t first_new) {
    for (size_t j = 0; j < staged.size(); j++) {
        const uint64_t i = first_new + j;
        memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(base + i * stride)),
               staged[j].bytes.data(), staged[j].length);
    }
}

} // namespace prosper::detail
