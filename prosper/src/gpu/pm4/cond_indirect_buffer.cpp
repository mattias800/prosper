// cond_indirect_buffer.cpp -- see cond_indirect_buffer.hpp.
#include "gpu/pm4/cond_indirect_buffer.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_set>

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

namespace {

// Once per distinct value of `key` in `seen`, so a second, different shape later in the run is not
// silenced by the first (a single process-wide flag would be).
bool first_time(std::unordered_set<uint64_t>& seen, uint64_t key) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (seen.size() >= 64) return false;   // bounded: a run cannot flood the log
    return seen.insert(key).second;
}

}  // namespace

// Whether a packet-predicated Jump is skipped.
//
// SET_PREDICATION keeps the polarity in PRED_BOOL, separate from PRED_OP. For the boolean ops,
// DRAW_VISIBLE discards the guarded work when the word is zero and runs it otherwise;
// DRAW_NOT_VISIBLE is the inverse. Vulkan conditional rendering uses one op (BOOL64) for both its
// plain and inverted forms and flips only that bit. So the rule here is keyed on the flag
// arguments, and the op only selects which family of rules applies.
//
// DRAW_VISIBLE is INFERRED from the observed call shape: Dragon Quest VII Reimagined (PPSA17942)
// opens every window with sceAgcDcbSetPredication(dcb, 1, 3, 1, addr), and whichever of the two
// flag arguments is PRED_BOOL, it is 1. The functional evidence is that title's per-surface words.
// Its predicated segments are AGC's own labelled helpers ("Decompress Htile", "Eliminate Fast
// Clear"). On every title frame the main 3840x2160 depth surface's decompress reads 1 and the
// decompress of a 1x1 scratch depth surface reads 0. The previous rule ("skip when non-zero", #319)
// ran the 1x1 helper. Its 1x1 viewport and VPORT scissor then stayed in the context registers: the
// parent stream never re-programs them, because on hardware that helper does not run. Every later
// draw of the frame (sky clouds, water, scene-colour copy, fog) rasterised to one pixel (#4610).
// #319 pinned the old rule on segments it read as the backbuffer composite. They are AGC's
// eliminate-fast-clear rectangles on the scanout buffers, which write no colour here (#1588), and
// the title still composites to scanout under this rule.
//
//   op 3, flags (1, 1): run on non-zero. CONFIDENCE: HIGH (observed shape + the evidence above).
//   op 3, flags (0, 0): run on ZERO. Both candidate PRED_BOOL slots say DRAW_NOT_VISIBLE.
//                       CONFIDENCE: MED (hardware contract; no title has been seen to use it).
//   op 3, flags differ: PRED_BOOL is not identified, so this keeps the observed DRAW_VISIBLE.
//                       CONFIDENCE: LOW.
//   op 3, no flags:     a packet from a capture recorded before the flags were kept. Every such
//                       capture is of the observed shape, so run on non-zero.
//   any other op:       no title evidence. BOOL32 (4 in the hardware enum) would take the same
//                       PRED_BOOL rule over a 32-bit word, but AGC's op numbering is only known to
//                       match the hardware enum at 3, and the fold reads 64 bits. ZPASS/PRIMCOUNT
//                       compare an occlusion or stream-out result, not a boolean. So these keep the
//                       previous rule (skip on non-zero) rather than guessing.
// Every shape except op 3 with flags (1, 1) is reported once per distinct shape.
bool predicated_jump_skips(uint32_t control, uint64_t cond) {
    constexpr uint32_t kPredOpBool64 = 3;
    const uint32_t op = control & 0xffu;
    const bool flags_present = (control & kSetPredicationFlagsPresent) != 0;
    const uint32_t a1 = (control >> 8) & 0xffu, a3 = (control >> 16) & 0xffu;
    const bool observed = op == kPredOpBool64 && flags_present && a1 == 1 && a3 == 1;
    bool run_on_non_zero = true;
    const char* rule = "run on non-zero (observed DRAW_VISIBLE)";
    if (op != kPredOpBool64) {
        run_on_non_zero = false;
        rule = "op has no verified polarity; skip on non-zero as before";
    } else if (!flags_present) {
        rule = "packet predates flag capture; run on non-zero (the only recorded shape)";
    } else if (a1 == 0 && a3 == 0) {
        run_on_non_zero = false;
        rule = "both flag slots clear: DRAW_NOT_VISIBLE, run on zero (CONFIDENCE: MED)";
    } else if ((a1 == 0) != (a3 == 0)) {
        rule = "flag slots disagree: PRED_BOOL unidentified, run on non-zero (CONFIDENCE: LOW)";
    }
    static std::unordered_set<uint64_t> reported_shapes;
    if (!observed && first_time(reported_shapes, control))
        std::fprintf(stderr,
                     "[agc] predicated Jump under SetPredication op=%u a1=%u a3=%u flags=%s: "
                     "unobserved shape -- %s\n",
                     op, a1, a3, flags_present ? "kept" : "absent", rule);
    return run_on_non_zero ? cond == 0 : cond != 0;
}

void report_unreadable_predicate_condition(uint64_t addr) {
    static std::unordered_set<uint64_t> reported_addresses;
    if (first_time(reported_addresses, addr))
        std::fprintf(stderr,
                     "[agc] predicated Jump condition at 0x%llx is misaligned or unreadable: the "
                     "segment runs without a word to decide on (fail-open)\n",
                     (unsigned long long)addr);
}

}  // namespace prosper::gpu
