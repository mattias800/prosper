// cond_indirect_buffer.cpp -- see cond_indirect_buffer.hpp.
#include "gpu/pm4/cond_indirect_buffer.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>

namespace prosper::gpu {

bool guest_readable(uint64_t addr, uint32_t bytes);   // gpu_executor.cpp

namespace {
bool report_now(std::atomic<uint64_t>& counter, uint64_t& ordinal) {
    ordinal = counter.fetch_add(1) + 1;
    return ordinal <= 8 || (ordinal & (ordinal - 1)) == 0;
}
}  // namespace

Pm4Command cond_indirect_buffer_jump(const Pm4Command& c) {
    bool take_then = true;
    if (c.cib_func != 0) {
        uint64_t value = 0;
        const bool readable = c.cib_compare_addr && guest_readable(c.cib_compare_addr, 8);
        if (readable) std::memcpy(&value, (const void*)(uintptr_t)c.cib_compare_addr, 8);
        const uint64_t lhs = value & c.cib_mask, rhs = c.cib_reference;
        switch (c.cib_func) {
            case 1: take_then = lhs < rhs; break;
            case 2: take_then = lhs <= rhs; break;
            case 3: take_then = lhs == rhs; break;
            case 4: take_then = lhs != rhs; break;
            case 5: take_then = lhs >= rhs; break;
            case 6: take_then = lhs > rhs; break;
            default: take_then = true; break;
        }
        static std::atomic<uint64_t> reported{0};
        uint64_t k = 0;
        if (report_now(reported, k))
            std::fprintf(stderr,
                         "[agc] conditional branch #%llu uses an unverified condition (mode=%u "
                         "func=%u addr=0x%llx readable=%d mask=0x%llx ref=0x%llx -> %s) (#4540)\n",
                         (unsigned long long)k, c.cib_mode, c.cib_func,
                         (unsigned long long)c.cib_compare_addr, readable ? 1 : 0,
                         (unsigned long long)c.cib_mask, (unsigned long long)c.cib_reference,
                         take_then ? "then" : "else");
    }
    Pm4Command j = c;
    j.kind = Pm4Command::Kind::Jump;
    j.jump_addr = take_then ? c.cib_then_addr : c.cib_else_addr;
    j.jump_dwords = take_then ? c.cib_then_dwords : c.cib_else_dwords;
    j.jump_pred = 0;
    j.jump_valid = true;
    return j;
}

bool jump_segment_within_limits(uint64_t addr, uint32_t dwords, uint32_t depth) {
    constexpr uint32_t kMaxJumpDwords = 0x40000;
    constexpr uint32_t kMaxJumpDepth = 8;
    if (dwords <= kMaxJumpDwords && depth < kMaxJumpDepth) return true;
    static std::atomic<uint64_t> refused{0};
    uint64_t k = 0;
    if (report_now(refused, k))
        std::fprintf(stderr,
                     "[agc] jump segment REFUSED #%llu: target=0x%llx dwords=%u (max %u) "
                     "depth=%u (max %u)\n",
                     (unsigned long long)k, (unsigned long long)addr, dwords, kMaxJumpDwords, depth,
                     kMaxJumpDepth);
    return false;
}

bool jump_segment_readable(uint64_t addr, uint32_t dwords) {
    if (guest_readable(addr, dwords * 4)) return true;
    static std::atomic<uint64_t> unreadable{0};
    uint64_t k = 0;
    if (report_now(unreadable, k))
        std::fprintf(stderr, "[agc] jump segment UNREADABLE #%llu: target=0x%llx dwords=%u\n",
                     (unsigned long long)k, (unsigned long long)addr, dwords);
    return false;
}

}  // namespace prosper::gpu
