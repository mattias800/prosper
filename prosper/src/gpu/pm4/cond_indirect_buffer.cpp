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
    Pm4Command j = c;
    j.kind = Pm4Command::Kind::Jump;
    j.jump_addr = 0;
    j.jump_dwords = 0;
    j.jump_pred = 0;
    j.jump_valid = true;
    if (c.cib_mode != kCibModeIfThen && c.cib_mode != kCibModeIfThenElse) {
        static std::atomic<uint64_t> reserved{0};
        uint64_t k = 0;
        if (report_now(reserved, k))
            std::fprintf(stderr,
                         "[agc] conditional branch REFUSED #%llu: reserved mode=%u func=%u "
                         "then=0x%llx/%u else=0x%llx/%u -- neither target runs (#4540)\n",
                         (unsigned long long)k, c.cib_mode, c.cib_func,
                         (unsigned long long)c.cib_then_addr, c.cib_then_dwords,
                         (unsigned long long)c.cib_else_addr, c.cib_else_dwords);
        return j;
    }
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
    if (take_then) {
        j.jump_addr = c.cib_then_addr;
        j.jump_dwords = c.cib_then_dwords;
    } else if (c.cib_mode == kCibModeIfThenElse) {
        j.jump_addr = c.cib_else_addr;
        j.jump_dwords = c.cib_else_dwords;
    }   // if-then with a false condition runs nothing: the else fields are not a target
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

// Whether a packet-predicated Jump is skipped, given the enclosing SetPredication window's op and
// the 64-bit condition word read at fold time.
//
// PRED_OP 3 is the hardware BOOL64 predicate. sceAgcDcbSetPredication(dcb, 1, 3, 1, addr) opens it
// with DRAW_VISIBLE semantics: the guarded work runs when the condition is NON-ZERO and is
// discarded when it is zero (the AMD contract for a boolean predicate; Vulkan's non-inverted
// conditional rendering is built on the same packet).
//
// Evidence on Dragon Quest VII Reimagined (PPSA17942), whose predicated segments are AGC's own
// labelled helpers ("Decompress Htile", "Eliminate Fast Clear"), each guarded by a per-surface
// "needs it" word. Per title frame: the main 3840x2160 depth surface's decompress reads 1 and the
// decompress of a 1x1 scratch depth surface reads 0. The previous rule ("skip when non-zero", #319)
// ran exactly the wrong one: it skipped the real decompress and ran the 1x1 one, whose 1x1
// viewport and VPORT scissor then stayed in the context registers -- the parent stream never
// re-programs them, because on hardware that helper does not run -- and every later draw of the
// frame (sky clouds, water, scene-colour copy, fog) rasterised to one pixel. #319 pinned the old
// rule on segments it read as the backbuffer composite; on the current route the only segments
// with full-screen colour state are AGC's "Eliminate Fast Clear" rectangles, which write no colour
// here (#1588), and the title still composites to scanout under this rule.
//
// Other ops (ZPASS/PRIMCOUNT read an occlusion or stream-out result, not a boolean) have no title
// evidence and keep the previous rule, reported once so the gap is not silent.
// CONFIDENCE: MED-HIGH for op 3 (hardware contract + the per-surface pattern above); LOW for others.
bool predicated_jump_skips(uint32_t pred_op, uint64_t cond) {
    constexpr uint32_t kPredOpBool64 = 3;
    if (pred_op == kPredOpBool64) return cond == 0;
    static std::atomic<bool> reported{false};
    if (!reported.exchange(true))
        std::fprintf(stderr,
                     "[agc] predicated Jump under SetPredication op=%u: only BOOL64 (3) has a "
                     "verified polarity; skipping on a non-zero condition as before\n",
                     pred_op);
    return cond != 0;
}

}  // namespace prosper::gpu
