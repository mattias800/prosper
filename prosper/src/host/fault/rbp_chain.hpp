// rbp_chain.hpp — walk a guest frame-pointer chain into a caller list.
//
// Guest PS5 code is SysV x86-64 built with frame pointers on the paths this project traces, and the
// only durable way to answer "who called this?" at an arbitrary instruction is to follow saved-rbp
// links. That walk already existed once, inline in the fault path (`run_entry`'s post-fault
// reporting). It is extracted here because a second consumer needs it — `PROSPER_HWBP_STACK`, which
// reports the caller chain at a hardware breakpoint — and because two hand-written copies of a loop
// that dereferences guest-controlled pointers is exactly the kind of duplication that grows a
// divergence nobody notices until one of them faults inside a signal handler.
//
// Two properties the callers depend on:
//
//   * **It never dereferences without asking.** The caller supplies `readable`, so the fault path can
//     use its sigsetjmp guard and a signal handler can use a non-faulting probe. There is no
//     "probably mapped" heuristic in here.
//   * **It is allocation-free and lock-free**, so it is usable from a signal handler alongside
//     `raw_write`.
//
// What it deliberately does NOT do is validate that a recovered return address is guest code. A
// caller that wants only guest frames should filter with `guest_va_in_module_code`; a caller
// diagnosing a corrupted stack usually wants to see the junk. Note the standing caveat that applies
// to every frame-pointer walk over optimised code (instrument trap 114 and 217): a frame whose
// function omitted the prologue is skipped silently, so the chain can be missing a level rather than
// wrong. Treat it as a lead, and confirm a link by disassembling the named call site — every link in
// a chain can be checked that way, because a return address must point immediately after a `call` to
// the frame below it.
#pragma once
#include <cstdint>

namespace prosper::host {

// Follow the saved-rbp chain starting at `rbp`, writing up to `max` return addresses into `out`.
// `readable(a)` must answer whether the 8 bytes at `a` can be dereferenced.
//
// The walk stops at the first of:
//   * `max` frames written,
//   * a frame whose saved-rbp slot or return-address slot is not readable,
//   * a frame pointer at or below 0x10000 (the guest never maps the low pages as a stack),
//   * a next-frame pointer that does not strictly increase — a cycle or a corrupted frame. x86-64
//     stacks grow down, so an honest caller frame is always at a HIGHER address.
// A null return-address slot is not written but does not stop the walk: it is the ordinary terminator
// of an outermost frame, and prosper enters the guest with rbp = 0, so the last link legitimately
// reads zero (this is the same slot PPSA27640's unwinder reads, see BENEATH_STATUS.md).
//
// Returns the number of addresses written.
inline int walk_rbp_chain(uint64_t rbp, uint64_t* out, int max,
                          bool (*readable)(uint64_t)) {
    if (!out || max <= 0 || !readable) return 0;
    int n = 0;
    uint64_t bp = rbp;
    while (n < max && bp > 0x10000) {
        if (!readable(bp) || !readable(bp + 8)) break;
        const uint64_t ret = *(const uint64_t*)(bp + 8);
        if (ret) out[n++] = ret;
        const uint64_t next = *(const uint64_t*)bp;
        if (next <= bp) break;
        bp = next;
    }
    return n;
}

}  // namespace prosper::host
