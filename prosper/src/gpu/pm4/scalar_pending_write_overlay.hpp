#pragma once
// The existing pending-queue scalar overlay, separated from queue ownership and diagnostics.
// Exact-address matching and partial low-word writes intentionally retain their current behavior.
#include "gpu/pm4/pm4_decode.hpp"
#include <algorithm>
#include <cstring>

namespace prosper::gpu {
template <class ImmediateSource>
inline bool overlay_pending_scalar_write(const Pm4Command& c, uint64_t addr, uint64_t* value,
                                         ImmediateSource&& immediate_source) {
    uint64_t v = *value;
    bool touched = false;
    using K = Pm4Command::Kind;
    if (c.kind == K::ReleaseMem && c.rel_addr == addr && c.rel_value_valid) {
        if (c.rel_data_sel == 1) {
            uint32_t lo = (uint32_t)c.rel_value;
            memcpy(&v, &lo, sizeof lo);
            touched = true;
        } else if (c.rel_data_sel == 2) {
            v = c.rel_value;
            touched = true;
        }
    } else if (c.kind == K::WriteData && c.wd_valid && c.wd_addr == addr && c.wd_data && c.wd_num) {
        const size_t n = std::min<size_t>((size_t)c.wd_num * 4, sizeof v);
        memcpy(&v, c.wd_data, n);
        touched = true;
    } else if (c.kind == K::DmaData && c.dd_dst == addr && c.dd_valid && immediate_source(c)) {
        const uint32_t word = (uint32_t)c.dd_src;
        const size_t n = std::min<size_t>(c.dd_bytes, sizeof v);
        for (size_t off = 0; off < n; off += sizeof word)
            memcpy((uint8_t*)&v + off, &word, std::min(sizeof word, n - off));
        touched = true;
    }
    if (touched) *value = v;
    return touched;
}
} // namespace prosper::gpu
