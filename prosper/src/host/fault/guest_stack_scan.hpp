// guest_stack_scan.hpp — recover guest return addresses by SCANNING a stack, for code compiled
// without frame pointers.
//
// The sibling of rbp_chain.hpp, and the reason both exist is that which one works is a property of
// how the GUEST was compiled, not something the host can know. A chain walk is precise and is often
// EMPTY on an optimised title — FINAL FANTASY TACTICS' fatal raise recovers exactly one frame, the
// module entry point, because every guest frame between omitted its prologue. A scan has no such
// dependency: a return address pushed by `call` is on the stack whether or not the callee kept a
// frame pointer.
//
// The cost of a scan is false positives, so each candidate is CONFIRMED at the call site: a real
// return address has a `call` instruction ending immediately before it. A stale value left in a dead
// stack slot almost never satisfies that, which is what makes the output readable rather than a wall
// of plausible-looking numbers. What it still cannot do is tell a live frame from a dead one that
// happens to validate, so the result is stack order, not proven call order.
//
// This lives in one header rather than once per platform because the two platforms differ only in
// their `readable` probe. An earlier revision had a copy in each `exec_image_*.cpp`, which is
// exactly the divergence host/fault/ exists to prevent — and the claim that they "cannot drift" was
// made in a PR while two copies were in the diff.
//
// Every dereference is behind the caller's probe. That is not defensive style: this runs on paths
// that are already terminating the process, where a nested SIGSEGV would replace the diagnostic with
// a confusing fault report about the diagnostic itself.
#pragma once
#include <cstdint>

namespace prosper::host {

// Does a `call` instruction end at `ret`? A return address is always the byte after the call that
// pushed it, so this is the strongest cheap test that a stack word is a real frame and not residue.
//
// Encodings accepted, both forms a compiler emits for a direct or indirect call:
//   E8 rel32                      -- 5 bytes, `call rel32`
//   FF /2 with any ModRM/SIB/disp -- `call r/m64`, 2 to 7 bytes, optionally REX-prefixed
// The `FF` search admits a byte sequence that merely looks like one; `E8` is near-conclusive. Both
// are far better than accepting every readable guest address, which is the alternative.
//
// PROBE COVERAGE IS THE SUBTLE PART. `readable(a)` answers for `[a, a+8)` on the POSIX probe, so
// probing only `ret - 16` would validate `[ret-16, ret-8)` while the bytes actually read are
// `[ret-7, ret)` -- two windows that do not overlap at all. A page boundary landing between them
// (`ret % 4096` in the low single digits) would then fault on a read the guard believed it had
// covered. The module filter cannot save it either: guest_module_name classifies by fixed aperture,
// so an unmapped address inside a module's range is still labelled with that module. Probe all
// three of `ret-16`, `ret-8` and `ret-1` so the union covers every byte read below.
inline bool call_precedes(uint64_t ret, bool (*readable)(uint64_t)) {
    if (ret < 16 || !readable) return false;
    if (!readable(ret - 16) || !readable(ret - 8) || !readable(ret - 1)) return false;
    const uint8_t* p = (const uint8_t*)(uintptr_t)ret;
    if (p[-5] == 0xE8) return true;                     // call rel32
    for (int len = 2; len <= 7; ++len) {
        const uint8_t* c = p - len;
        int i = 0;
        if ((c[0] & 0xF0) == 0x40) i = 1;               // REX prefix
        if (c[i] != 0xFF) continue;
        if (i + 1 >= len) continue;
        if (((c[i + 1] >> 3) & 0x7) == 2) return true;  // /2 == call r/m
    }
    return false;
}

// Scan [rsp_seed, limit) for call-site-validated return addresses that `keep` accepts, newest first.
//
// The walk STOPS at the first unreadable slot rather than skipping it: running off the mapped stack
// is a fact about the stack, and continuing past it would report frames from unrelated memory.
// Returns how many addresses were written.
inline int scan_stack_for_returns(uint64_t rsp_seed, uint64_t limit, uint64_t* out, int max,
                                  bool (*readable)(uint64_t), bool (*keep)(uint64_t)) {
    if (!out || max <= 0 || !readable || !keep || rsp_seed <= 0x10000) return 0;
    int kept = 0;
    for (uint64_t a = rsp_seed & ~7ull; a + 8 <= limit && kept < max; a += 8) {
        if (!readable(a)) break;
        const uint64_t candidate = *(const uint64_t*)(uintptr_t)a;
        if (candidate <= 0x10000) continue;
        if (!keep(candidate)) continue;
        if (!call_precedes(candidate, readable)) continue;
        bool duplicate = false;
        for (int j = 0; j < kept; ++j) duplicate |= out[j] == candidate;
        if (duplicate) continue;
        out[kept++] = candidate;
    }
    return kept;
}

}  // namespace prosper::host
