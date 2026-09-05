// walk_rbp_chain: the frame-pointer walk shared by the fault reporter and PROSPER_HWBP_STACK.
//
// It dereferences guest-controlled pointers from inside a signal handler, so every stop condition is
// a safety property, not a nicety. These arms drive the real function over a synthetic stack whose
// readable window is known exactly, so "it stopped" and "it faulted" are distinguishable — which they
// would not be if the test used real addresses.
//
// The synthetic stack is built the way x86-64 lays one out: each frame is {saved_rbp, return_addr},
// and an honest caller frame sits at a HIGHER address than its callee's.

#include "host/fault/rbp_chain.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper::host;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

// A readable window the predicate below reports on. Everything outside it is "unmapped", so a walk
// that stepped past the end of the stack would be caught here instead of segfaulting the test.
static const uint64_t* g_lo = nullptr;
static const uint64_t* g_hi = nullptr;
static bool in_window(uint64_t a) {
    // probe_readable's real contract: [a, a+8) must be readable.
    return a >= (uint64_t)(uintptr_t)g_lo && a + 8 <= (uint64_t)(uintptr_t)g_hi;
}
static bool never_readable(uint64_t) { return false; }

int main() {
    std::printf("== test_rbp_chain ==\n");

    // frame[i] = { saved_rbp -> frame[i+1], return address }
    // Laid out in one array so the addresses strictly increase, as real frames do.
    uint64_t st[8] = {};
    const uint64_t f0 = (uint64_t)(uintptr_t)&st[0];
    const uint64_t f1 = (uint64_t)(uintptr_t)&st[2];
    const uint64_t f2 = (uint64_t)(uintptr_t)&st[4];
    st[0] = f1;  st[1] = 0x1562753;      // innermost frame returns to 0x1562753
    st[2] = f2;  st[3] = 0x156234d;
    st[4] = 0;   st[5] = 0x1603bfd;      // outermost: saved rbp is 0, the SysV entry convention
    g_lo = &st[0]; g_hi = &st[8];

    {
        uint64_t out[8] = {};
        const int n = walk_rbp_chain(f0, out, 8, &in_window);
        CHECK(n == 3 && out[0] == 0x1562753 && out[1] == 0x156234d && out[2] == 0x1603bfd,
              "a three-deep chain yields its three return addresses, innermost first");
    }
    {
        // The depth cap must truncate rather than over-read, and must not write past `max`.
        uint64_t out[4] = { 0, 0, 0xdeadbeef, 0xdeadbeef };
        const int n = walk_rbp_chain(f0, out, 2, &in_window);
        CHECK(n == 2 && out[0] == 0x1562753 && out[1] == 0x156234d && out[2] == 0xdeadbeef,
              "max caps the number of frames and nothing is written past it");
    }
    {
        // A cycle -- frame -> itself, or any non-increasing link -- must terminate. Without the
        // strictly-increasing rule this loops forever inside a signal handler.
        uint64_t cyc[2] = { (uint64_t)(uintptr_t)&cyc[0], 0xAAAA };
        g_lo = &cyc[0]; g_hi = &cyc[2];
        uint64_t out[8] = {};
        const int n = walk_rbp_chain((uint64_t)(uintptr_t)&cyc[0], out, 8, &in_window);
        CHECK(n == 1 && out[0] == 0xAAAA, "a self-referential frame yields one address and stops");
        g_lo = &st[0]; g_hi = &st[8];
    }
    {
        // An unreadable frame stops the walk instead of faulting. This is the arm that separates
        // "the walker respects the predicate" from "the predicate happened not to matter".
        uint64_t out[8] = {};
        const int n = walk_rbp_chain(f0, out, 8, &never_readable);
        CHECK(n == 0, "a predicate that refuses every address yields an empty chain, not a fault");
    }
    {
        // Only the OUTERMOST frame is readable: the walk must stop at the window edge rather than
        // following st[4]'s saved rbp of 0 into the low pages.
        g_lo = &st[4]; g_hi = &st[6];
        uint64_t out[8] = {};
        const int n = walk_rbp_chain(f2, out, 8, &in_window);
        CHECK(n == 1 && out[0] == 0x1603bfd,
              "a null saved-rbp terminates the walk after recording its own return address");
        g_lo = &st[0]; g_hi = &st[8];
    }
    {
        // Degenerate inputs a caller can reach: rbp in the low pages (prosper enters the guest with
        // rbp = 0), a null out buffer, a zero cap, a null predicate.
        uint64_t out[4] = {};
        CHECK(walk_rbp_chain(0, out, 4, &in_window) == 0, "rbp = 0 yields an empty chain");
        CHECK(walk_rbp_chain(0x1000, out, 4, &in_window) == 0,
              "an rbp in the low pages is refused without dereferencing it");
        CHECK(walk_rbp_chain(f0, nullptr, 4, &in_window) == 0, "a null out buffer yields 0");
        CHECK(walk_rbp_chain(f0, out, 0, &in_window) == 0, "a zero cap yields 0");
        CHECK(walk_rbp_chain(f0, out, 4, nullptr) == 0, "a null predicate yields 0, not a blind read");
    }
    {
        // A null return-address slot is skipped, not treated as the end of the world: the frame's
        // saved rbp is still followed. This is the shape prosper's own guest entry leaves.
        uint64_t z[6] = {};
        z[0] = (uint64_t)(uintptr_t)&z[2]; z[1] = 0;          // no return address recorded here
        z[2] = (uint64_t)(uintptr_t)&z[4]; z[3] = 0xBBBB;
        z[4] = 0;                          z[5] = 0xCCCC;
        g_lo = &z[0]; g_hi = &z[6];
        uint64_t out[8] = {};
        const int n = walk_rbp_chain((uint64_t)(uintptr_t)&z[0], out, 8, &in_window);
        CHECK(n == 2 && out[0] == 0xBBBB && out[1] == 0xCCCC,
              "a null return-address slot is skipped and the walk continues through it");
        g_lo = &st[0]; g_hi = &st[8];
    }

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
