#include "hle/memory/ampr_command_buffer_internal.hpp"
#include "hle/kernel/apr_submission.hpp"
#include <mutex>
#include <unordered_map>

namespace prosper {
// clang-format off: preserve the byte-identical moved definitions.

// The APR read chain a command buffer carries (which file its …ReadFileGatherScatter segments read
// from) belongs to the APR FILE layer, src/hle/fs/hle_file.cpp, which owns file ids and host paths.
// This file owns the command-buffer object, so it is the one that has to tell that layer when a
// buffer is rewound or destroyed.
extern "C" void prosper_apr_chain_reset(uint64_t cb);
namespace {
struct AmprCbState {
    uint64_t capacity = 0, offset = 0;
    // The storage sceAmprCommandBufferSetBuffer attached to this command buffer, recorded by both
    // platform arms so sceAmprCommandBufferGetBufferBaseAddress can answer it. The AMM flow is what
    // needs it: the SDK's inline Submit wrapper reads the base and the byte cursor off the object
    // and hands the kernel that raw span, so a zero base is a null command stream.
    uint64_t buffer_base = 0, buffer_size = 0;
    bool tracks_offset = false;
};
std::mutex g_ampr_cb_state_mx;
std::unordered_map<uint64_t, AmprCbState> g_ampr_cb_state;
} // namespace

// Does the constructor's `a2` say this command buffer keeps a byte cursor?
//
// Two shapes answer yes, and the second is #2384.
//
//  PLAIN   a2 IS the capacity -- Pathless's PS5 3.20 IoStore batch passes 0x560. Unchanged.
//  PACKED  a2 is (count << 32) | capacity -- Grand Theft Auto V (PPSA04263). Its ten constructor
//          calls in one boot carry a2 = 0x6400001370, 0x5f00001610, 0x5e00001680, 0x5d000016f0,
//          0x5c00001760, 0x9c000022b0, 0xe7000008a0, 0x7500000c70, 0x6700001290, 0x5700001990.
//
// The four consecutive calls at 0x5f/0x5e/0x5d/0x5c are why this is a decomposition and not a curve
// fit: as the high field falls 95, 94, 93, 92 the low field rises by exactly 0x70 each step -- a
// pool carved 112 bytes at a time, with the remaining-entry count in the high half. Monotone in both
// fields at a fixed stride is not something an arbitrary 64-bit value does.
//
// This predicate is STRICTLY ADDITIVE, which is the property that matters for the titles already
// working -- but the reason is the ORDER, not the `count != 0` guard, and I had that backwards until
// mutation testing corrected it. The plain arm runs first and returns true for every a2 it accepts,
// so no currently-accepted value ever reaches the packed arm at all. DOLL's a2 = 0 and a2 = 1 are
// rejected by both arms. tests/hle/memory/test_ampr_tracks_offset.cpp asserts this over a SWEEP rather than
// over examples, because examples cannot establish a disjointness claim.
//
// `count != 0` is therefore REDUNDANT and deliberately kept: when count is zero, a2 IS cap, so the
// plain arm has already decided. Deleting it is an equivalent mutation and no test can kill it --
// which is exactly why there is no arm claiming to. It stays as a cheap statement of intent, on the
// same footing as the `bc.eq` term in apr_submit_common. The `count <= 0xffff` bound is NOT
// redundant and does have a discriminating arm.
//
// KNOWN FALSE-POSITIVE CLASS, raised by Marlow in review and recorded because it cannot be fixed by
// range-checking: a GUEST POINTER can satisfy the packed arm. The guest VA range is
// 0x2000000000-0x9fc0000000, so any pointer whose LOW 32 BITS land in [0x20, 0x100000] -- the first
// MiB of every 4 GiB-aligned window -- decomposes into a plausible count and capacity. That defeats
// the "pointer-valued arguments are outside this conservative bound" guard in ampr_cb_capacity_arg
// directly above, because taking the low 32 bits discards exactly the high bits that made a pointer
// recognisable.
//
// It is not separable by value, and that is the part worth knowing before anyone tries: GTA V's own
// packed a2 values (0x6400001370, 0x5f00001610, 0x9c000022b0) are themselves numerically inside the
// guest VA range, so a "reject anything that looks like a pointer" test would reject the very calls
// this exists to accept. Distinguishing them needs a signal other than the 64-bit value -- the
// calling NID, or a constructor shape captured from a title that passes a pointer here.
//
// Accepted deliberately: no title has been observed to pass that shape, the change is additive for
// every title that works today, and the consequence of a false positive is a cursor tracked for a
// buffer that does not want one -- not a wrong answer to the guest. Revisit if a title regresses in
// AMPR submit timing.
//
// CONFIDENCE: MED -- the shape is inferred from one title's live capture, with no disassembly of the
// guest-side constructor yet. What would raise it: a second title with the same packing, or the
// constructor's own code showing the two fields written separately.
bool ampr_a2_is_capacity(uint64_t a2) { return a2 >= 0x20 && a2 <= 0x100000; }

bool ampr_cb_tracks_offset_arg(uint64_t a2) {
    if (ampr_a2_is_capacity(a2)) return true;                  // PLAIN
    const uint64_t count = a2 >> 32, cap = a2 & 0xffffffffull; // PACKED
    return count != 0 && count <= 0xffff && ampr_a2_is_capacity(cap);
}

uint64_t ampr_cb_capacity_arg(uint64_t a1, uint64_t a2, uint64_t a5) {
    // Observed constructor shapes place capacity in a1 (APR request), a2 (Pathless IoStore batch),
    // or a5 (DOLL IoStore batch). Pointer-valued arguments are outside this conservative bound.
    if (a1 && a1 <= 0x100000) return a1;
    // DOLL uses a2=1 as a mode flag and carries capacity in a5. Pathless's PS5 3.20 shape instead
    // supplies the actual command-buffer size in a2, so accept only a plausible packet capacity.
    if (a2 >= 0x20 && a2 <= 0x100000) return a2;
    if (a5 && a5 <= 0x100000) return a5;
    return 0;
}
void ampr_cb_construct(uint64_t cb, uint64_t capacity, bool tracks_offset) {
    if (!cb) return;
    // Constructing over a command buffer rewinds it (`state.offset = 0` below), so it closes any
    // APR read chain recorded against that address for exactly the reason ampr_cb_reset does: the
    // chain names the file the commands in this buffer read from, and those commands are gone. It
    // matters most here, because this is the path a RECYCLED guest VA arrives on — without it a new
    // buffer at an old address inherits the old file, and the gather/scatter range guard only
    // catches that when the newly named file happens to be smaller (review of #2924).
    prosper_apr_chain_reset(cb);
    apr_execution_reset(cb);
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    if (g_ampr_cb_state.size() >= 4096 && !g_ampr_cb_state.count(cb))
        g_ampr_cb_state.erase(g_ampr_cb_state.begin());
    auto& state = g_ampr_cb_state[cb];
    // Several SDK flows refresh a live object through a constructor-shaped call whose size slots
    // are all zero/pointers. The previous capacity map updated only when it decoded a nonzero size;
    // preserve that contract instead of erasing a valid capacity on the refresh.
    if (capacity) state.capacity = capacity;
    state.offset = 0;
    state.tracks_offset = tracks_offset;
}

void ampr_cb_reset(uint64_t cb) {
    if (!cb) return;
    // Rewinding a command buffer also closes the APR read chain recorded against it: the file a
    // …ReadFileGatherScatter segment reads from is the one the plain ReadFile named, and that
    // record dies with the commands it belonged to. Yakuza Kiwami's dispatcher at eboot+0xdb5fb0
    // calls Reset immediately before the ReadFile that opens each chain, which is consistent with
    // this -- but note it does NOT establish it, since the ReadFile that follows re-opens the chain
    // either way.
    //
    // This helper has more callers than the public Reset NID -- SUBMIT is one of them, because it
    // rewinds the same command cursor -- so closing here closes on all of them. That width is
    // deliberate and its justification (which way the undecided lifetime should fail) is written
    // out once, on f_apr_read_gather_scatter in src/hle/fs/hle_file.cpp, which owns APR file
    // identity. Read it before narrowing this; tests/hle/test_apr_gather_scatter.cpp pins the set.
    prosper_apr_chain_reset(cb);
    apr_execution_reset(cb);
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    auto it = g_ampr_cb_state.find(cb);
    if (it != g_ampr_cb_state.end()) it->second.offset = 0;
}

void ampr_cb_destroy_320(uint64_t cb) {
    if (!cb) return;
    prosper_apr_chain_reset(cb);
    apr_execution_reset(cb);   // the chain cannot outlive the command buffer it is recorded on
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    auto it = g_ampr_cb_state.find(cb);
    if (it != g_ampr_cb_state.end() && it->second.tracks_offset)
        g_ampr_cb_state.erase(it);
}


uint64_t ampr_cb_capacity(uint64_t cb) {
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    auto it = g_ampr_cb_state.find(cb);
    return it == g_ampr_cb_state.end() ? 0 : it->second.capacity;
}

uint64_t ampr_cb_offset(uint64_t cb) {
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    auto it = g_ampr_cb_state.find(cb);
    return it == g_ampr_cb_state.end() ? 0 : it->second.offset;
}

// sceAmprCommandBufferSetBuffer(cb, base, size) attaches storage to a command buffer. Record it
// for BOTH platform arms, and independently of what either arm does with the memory: the record is
// a fact about the guest's object, not about the host mapping. It is deliberately NOT cleared by
// ampr_cb_construct, for the same reason `capacity` is not -- several SDK flows refresh a live
// object through a constructor-shaped call.
void ampr_cb_set_buffer(uint64_t cb, uint64_t base, uint64_t size) {
    if (!cb) return;
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    // #2878: this eviction picks an ARBITRARY victim, and adding buffer_base to the record made
    // that newly consequential — an evicted victim's GetBufferBaseAddress answers 0, which on the
    // AMM submit path is a null command stream. Latent (no title comes near 4096 live command
    // buffers; PPSA31334 constructs two) and left as-is here rather than swapped for a different
    // arbitrary rule without evidence.
    if (g_ampr_cb_state.size() >= 4096 && !g_ampr_cb_state.count(cb))
        g_ampr_cb_state.erase(g_ampr_cb_state.begin());
    auto& state = g_ampr_cb_state[cb];
    state.buffer_base = base;
    state.buffer_size = size;
}

uint64_t ampr_cb_buffer_base(uint64_t cb) {
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    auto it = g_ampr_cb_state.find(cb);
    return it == g_ampr_cb_state.end() ? 0 : it->second.buffer_base;
}

// The ReadFile builder lives in hle_file.cpp; expose the one state transition it shares with the
// other AMPR builders without exposing the map itself.
void prosper_ampr_advance(uint64_t cb, uint64_t bytes) {
    if (!cb || !bytes) return;
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    auto it = g_ampr_cb_state.find(cb);
    if (it != g_ampr_cb_state.end() && it->second.tracks_offset)
        it->second.offset += bytes;
}
void ampr_cb_erase(uint64_t a0) {
    std::lock_guard<std::mutex> lock(g_ampr_cb_state_mx);
    g_ampr_cb_state.erase(a0);
}
// clang-format on

} // namespace prosper
