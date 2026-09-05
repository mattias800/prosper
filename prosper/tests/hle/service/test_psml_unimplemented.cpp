// test_psml_unimplemented — libScePsml must report FAILURE, because the dispatcher's default for an
// unregistered NID is `return 0`, and for a query with out-parameters `return 0` is a SUCCESS CLAIM
// about data prosper never wrote.
//
// The shape has now cost two titles. `CLAUDE.md` records the first: Metaphor: ReFantazio's
// `sceFontRenderCharGlyphImage` was unregistered, answered SCE_OK, wrote neither out-parameter, and
// the title read the untouched stack as a glyph size and divided by the resulting zero (#2951).
// PPSA20447 (The First Berserker: Khazan) is the second, and it multiplies instead of dividing.
//
// Its `eboot+0xe74b20`, transcribed from the dump:
//
//     e74b51  mov $0x137,%edi ; call <sceSysmoduleLoadModule>
//     e74b74  call <libScePsml::3WVD91e12ZQ>      ; -> r15d
//     e74b8e  lea -0x48(%rbp),%rdi                ; an OUT struct. NOT initialised by the guest --
//     e74b96  vmovups %ymm0,0x18(%rbx)            ;   the zeroing here targets a DIFFERENT buffer
//     e74b9e  call <libScePsml::+2KpvixvL6E>      ; (&out, buf) -> eax
//     e74ba3  or %r15d,%eax
//     e74ba6  jne 0xe74dc2                        ; EITHER non-zero -> clean early return
//     e74bac  mov -0x38(%rbp),%rdi                ; N := out[+0x10]
//     e74bb5  shl $0x4,%rdi                       ; size = N * 16
//     e74bb9  call <FMemory::Malloc>              ; measured at ~244 TiB, and the title asserts
//
// Two facts justify an error return rather than a guess, and this project has a recorded trap in the
// OPPOSITE direction — an error sentinel returned from a VALUE-returning contract, read by the guest
// as data, once produced a 2 GiB allocation. So both were checked against the guest's bytes, not
// assumed:
//
//   * The return value is a STATUS, not data. The guest ORs the two results and branches on
//     non-zero; it never uses either as a quantity. `CONFIDENCE: HIGH`.
//   * The non-zero branch at `eboot+0xe74dc2` is a stack-cookie check, the epilogue and `ret`. It
//     allocates nothing, so a failure return is the guest's own "this feature is unavailable" path.
//     `CONFIDENCE: HIGH`.
//
// `CONFIDENCE: LOW` on the specific error VALUE. libScePsml is absent from the PS5 3.20 reference set
// (this dump requires 12.70), so its names, signatures and error space are not derivable. The guest
// tests only zero versus non-zero. These assertions therefore pin the PROPERTY the guest reads —
// non-zero — and deliberately do not pin the constant, so recovering the real error space later is a
// one-line change rather than a test rewrite.
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

// The two NIDs PPSA20447 reaches during boot. libScePsml is not in the reference set, so the raw
// NIDs are the only identity these functions have.
static constexpr const char* kPsmlInit  = "3WVD91e12ZQ";
static constexpr const char* kPsmlQuery = "+2KpvixvL6E";

// The guest's decision, transcribed: `or %r15d,%eax ; jne <clean return>`. Returns true when the
// guest would go on to read its uninitialised out-struct and allocate from it.
static bool guest_takes_allocation_branch(uint64_t init_rc, uint64_t query_rc) {
    return ((uint32_t)init_rc | (uint32_t)query_rc) == 0;
}

int main() {
    std::printf("== test_psml_unimplemented ==\n");
    register_service_hle();

    HleFn init  = Hle::lookup(kPsmlInit);
    HleFn query = Hle::lookup(kPsmlQuery);
    CHECK(init != nullptr,  "libScePsml::3WVD91e12ZQ is registered (unregistered == answering SCE_OK)");
    CHECK(query != nullptr, "libScePsml::+2KpvixvL6E is registered");
    if (!init || !query) { std::printf("== FAIL: %d ==\n", ++fails); return 1; }

    const uint64_t init_rc  = init(0, 0, 0, 0, 0, 0);
    // The real call is (&out_struct, &buffer). Pass plausible pointers so a handler that dereferenced
    // them would be exercised rather than dodged; this one must not touch either.
    uint64_t out_struct[8] = { 0 }, buffer[4] = { 0 };
    const uint64_t query_rc = query((uint64_t)(uintptr_t)out_struct, (uint64_t)(uintptr_t)buffer,
                                    0, 0, 0, 0);

    CHECK(init_rc != 0,  "the init call reports failure, not SCE_OK");
    CHECK(query_rc != 0, "the query call reports failure, not SCE_OK");

    // The property that actually matters, stated as the guest states it.
    CHECK(!guest_takes_allocation_branch(init_rc, query_rc),
          "PPSA20447's `or ; jne` sends the guest to its clean early return, not to the allocation");

    // The counter-arm. Without it the assertion above is satisfied by any implementation at all, and
    // this is the exact pre-fix behaviour: the dispatcher's `return 0` for an unregistered NID.
    CHECK(guest_takes_allocation_branch(0, 0),
          "with both calls answering 0 the same guest code DOES reach the allocation -- so the "
          "assertion above is discriminating, not vacuous");
    // ...and one non-zero is enough, which is why registering either NID alone would also work and
    // why registering BOTH is the honest answer rather than the minimal one.
    CHECK(!guest_takes_allocation_branch(init_rc, 0) && !guest_takes_allocation_branch(0, query_rc),
          "either call reporting failure is enough for the guest to skip the allocation");

    // The size the guest would compute is `out[+0x10] * 16` from memory prosper never wrote. Its
    // magnitude is not a fixed number -- it is stack residue, measured as 0xf484c0000000 and
    // 0xff599e000000 on two runs of the same build -- so nothing here asserts a size. What is
    // assertable is that the handler leaves the out-struct alone: a handler that scribbled a
    // guessed layout into an unknown struct would be a different and worse defect.
    bool untouched = true;
    for (uint64_t v : out_struct) if (v != 0) untouched = false;
    for (uint64_t v : buffer)     if (v != 0) untouched = false;
    CHECK(untouched,
          "the handler writes NOTHING through either pointer -- the struct layout is unknown, so "
          "guessing at it would replace one fabrication with another");

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
