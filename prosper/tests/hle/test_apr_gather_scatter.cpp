// test_apr_gather_scatter — the libSceAmpr APR read builders, on the two ways they can report a
// success the guest cannot distinguish from a successful read of zeros (#2926, #2928).
//
// Three contracts, each with the arm that would fail without the fix named beside it:
//
//   1. sceAmprAprCommandBufferReadFileGather (mZSbNJVJpV8) and …ReadFileScatter (Jg-AgkdJHkk) are
//      REGISTERED and REFUSE. Unregistered, the dispatcher's return-0 default answers SCE_OK on a
//      contract where 0 means "queued, will deliver" — while nothing is written. (#2926)
//   2. …ReadFileGatherScatter (BVmR1H8l+XI) reports failure when the bytes were read but did not
//      reach the guest's own destination. `r.ok` alone is a host-file fact, not a delivery. (#2928)
//   3. The chain closer set is what the code actually does, not what its comment used to claim.
//      Submit closes a chain; so do both constructors and the destructor. (#2928)
//
// Contract 3's arms are only meaningful next to their own positive control: the SAME chain, the
// SAME segment, with NO closer in between, must SUCCEED. Without that, "the segment was refused"
// is equally explained by a fixture that never opened a chain at all, and every arm passes for the
// wrong reason. The GetSize arm is the other half — a non-closer must leave the chain open.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/mman.h>
#endif

#include "fixtures/test_scratch.h"
#include "host/image/exec_image.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

namespace prosper {
uint32_t prosper_apr_register(const std::string& path, uint64_t size);
void prosper_apr_reset_for_test();
}
using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { std::printf("  [ok]   %s\n", msg); } } while (0)

// The error every APR read builder returns for a file it cannot name or serve.
constexpr uint64_t kAprRefused = 0x80020016ull;

// The plain ReadFile takes its file offset in a STACK slot, so it must be entered the way the guest
// enters it — through the generated import stub, with the guest SysV ABI — rather than through the
// HleFn pointer. sysv_abi is a no-op on Linux and forces the guest convention on MinGW.
using GuestReadFile = uint64_t(__attribute__((sysv_abi)) *)(uint64_t, uint64_t, uint64_t, uint64_t,
                                                            uint64_t, uint64_t, uint64_t, uint64_t,
                                                            uint64_t);

int main() {
    std::printf("== test_apr_gather_scatter ==\n");
    register_builtin_hle();

    HleFn gather_scatter = Hle::lookup("BVmR1H8l+XI");
    HleFn gather         = Hle::lookup("mZSbNJVJpV8");
    HleFn scatter        = Hle::lookup("Jg-AgkdJHkk");

    // ---- Contract 1: the two underived builders refuse instead of answering SCE_OK ------------
    // Without the fix both lookups are null, because nothing registers the NIDs — which is exactly
    // the defect: an unregistered NID is answered by the dispatcher's `return 0`.
    CHECK(gather != nullptr,
          "sceAmprAprCommandBufferReadFileGather is registered (not left on the return-0 default)");
    CHECK(scatter != nullptr,
          "sceAmprAprCommandBufferReadFileScatter is registered (not left on the return-0 default)");
    if (gather)
        CHECK(gather(0x1000, 0, 0, 0x2000, 64, 0) == kAprRefused,
              "...ReadFileGather refuses: its argument layout has never been derived");
    if (scatter)
        CHECK(scatter(0x1000, 0, 0, 0x2000, 64, 0) == kAprRefused,
              "...ReadFileScatter refuses: its argument layout has never been derived");

    CHECK(gather_scatter != nullptr, "...ReadFileGatherScatter is registered");
    if (!gather_scatter) {
        std::printf("== FAIL ==\n");
        return 1;
    }

    // ---- Fixture: a real file, a real APR id, and a real chain-opening ReadFile ----------------
    const std::string fixture_path_storage =
        prosper_test::test_scratch_file("prosper-test-apr-gather-scatter.tmp");
    const char* fixture_path = fixture_path_storage.c_str();
    std::array<uint8_t, 512> expected{};
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = (uint8_t)(i * 37u + 3u);
    bool fixture_written = false;
    if (FILE* file = std::fopen(fixture_path, "wb")) {
        const size_t written = std::fwrite(expected.data(), 1, expected.size(), file);
        fixture_written = written == expected.size() && std::fclose(file) == 0;
    }
    CHECK(fixture_written, "wrote the APR container fixture");

    std::string stub_error;
    const std::vector<ImportSlot> slots = {{"libSceAmpr", "mQ16-QdKv7k"}};
    CHECK(install_stubs(slots, 0x720000000ull, 96, &stub_error) && stub_error.empty(),
          "generated the executable AMPR ReadFile import stub");
    auto read_file_guest = reinterpret_cast<GuestReadFile>(static_cast<uintptr_t>(stub_addr(0)));

    prosper_apr_reset_for_test();
    const uint32_t fixture_id = prosper_apr_register(fixture_path, expected.size());
    CHECK(fixture_id != 0, "registered the fixture in the APR container registry");

    // One command buffer for the whole run. `request` is the command buffer the chain is keyed on;
    // `completion` is the caller-supplied record out-pointer (kept distinct from cb+0x20 so the
    // byte-count qword is written too).
    std::array<uint8_t, 0x48> request{};
    const uint64_t cb = (uint64_t)(uintptr_t)request.data();
    std::array<uint64_t, 3> completion{};
    const uint64_t record = (uint64_t)(uintptr_t)completion.data();

    constexpr size_t open_offset = 16, open_size = 32;
    constexpr size_t seg_offset = 96, seg_size = 73;
    std::array<uint8_t, 0x90> open_dst{};
    std::array<uint8_t, 0x90> seg_dst{};

    // Re-open the chain and clear the segment destination. Every arm below starts from here, so the
    // only difference between the control and a closer arm is the closer call itself.
    const auto open_chain = [&]() -> bool {
        open_dst.fill(0);
        return read_file_guest(cb, 0, record, fixture_id, (uint64_t)(uintptr_t)open_dst.data(),
                               open_size, open_offset, 0, 0) == 0;
    };
    const auto append_segment = [&]() -> uint64_t {
        seg_dst.fill(0xEE);   // sentinel: a segment that never wrote leaves this intact
        return gather_scatter(cb, 0, record, (uint64_t)(uintptr_t)seg_dst.data(), seg_size,
                              seg_offset);
    };

    CHECK(open_chain(), "the plain ReadFile that opens the chain succeeds");
    CHECK(std::memcmp(open_dst.data(), expected.data() + open_offset, open_size) == 0,
          "the chain-opening ReadFile filled its own destination");

    // ---- Positive control for every arm below: an open chain serves the segment ----------------
    // This is the arm that makes the refusals mean something. It also proves the segment really is
    // delivered rather than merely reported: the destination is pre-poisoned with 0xEE, so a
    // callee that wrote nothing leaves a value the fixture bytes cannot produce at these offsets.
    CHECK(append_segment() == 0, "a gather/scatter segment on an OPEN chain succeeds");
    CHECK(std::memcmp(seg_dst.data(), expected.data() + seg_offset, seg_size) == 0,
          "the segment overwrote the 0xEE poison with the fixture's own bytes");

    // ---- Contract 3: the closer set --------------------------------------------------------
    // Each entry re-opens the chain, calls one closer, and asserts the next segment is refused.
    struct Closer { const char* nid; const char* name; bool required; };
    const Closer closers[] = {
        { "baQO9ez2gL4", "sceAmprCommandBufferReset",                    true  },
        { "ULvXMDz56po", "sceAmprCommandBufferClearBuffer",              true  },
        { "8aI7R7WaOlc", "sceAmprCommandBufferConstructor",              true  },
        { "GuchCTefuZw", "sceAmprCommandBufferDestructor",               true  },
        { "Qs1xtplKo0U", "sceAmprAprCommandBufferDestructor",            true  },
        { "YPxkUDhgoNI", "sceAmprAprCommandBufferResetGatherScatterState", true },
        // Submit is the closer the handler's comment used to deny. It closes because the submit
        // path rewinds the command cursor through the same helper; the choice to keep it is
        // documented on f_apr_read_gather_scatter.
        { "eE4Szl8sil8", "sceKernelAprSubmitCommandBuffer",              true  },
        { "ASoW5WE-UPo", "sceKernelAprSubmitCommandBufferAndGetResult",  true  },
        // POSIX-only registration (#1970 left the Windows arm without it).
        { "EDq5bqCqYpA", "sceAmprAmmCommandBufferConstructor",           false },
    };
    for (const Closer& c : closers) {
        HleFn fn = Hle::lookup(c.nid);
        char msg[256];
        if (!fn) {
            if (c.required) {
                std::snprintf(msg, sizeof msg, "%s (%s) is registered", c.name, c.nid);
                CHECK(false, msg);
            } else {
                std::snprintf(msg, sizeof msg,
                              "%s (%s) is not registered on this host -- arm not applicable",
                              c.name, c.nid);
                std::printf("  [--]   %s\n", msg);
            }
            continue;
        }
        if (!open_chain()) {
            std::snprintf(msg, sizeof msg, "re-opened the chain before the %s arm", c.name);
            CHECK(false, msg);
            continue;
        }
        // The constructors take capacity arguments; the rest ignore everything past a0. Passing a
        // plausible capacity keeps the constructor arms on their real path rather than an early out.
        fn(cb, 0x560, 0, 0, 0, 0);
        const uint64_t after = append_segment();
        std::snprintf(msg, sizeof msg, "%s closes the read chain -- the next segment is refused",
                      c.name);
        CHECK(after == kAprRefused, msg);
        std::snprintf(msg, sizeof msg, "%s: the refused segment left its destination untouched",
                      c.name);
        bool untouched = true;
        for (size_t i = 0; i < seg_size; ++i) if (seg_dst[i] != 0xEE) untouched = false;
        CHECK(untouched, msg);
    }

    // ---- The other half: a NON-closer must leave the chain open ------------------------------
    // Without this, "every NID I called refused the next segment" would also be satisfied by a
    // chain that closed itself, or by a segment path that refuses unconditionally.
    if (HleFn get_size = Hle::lookup("tZDDEo2tE5k")) {
        CHECK(open_chain(), "re-opened the chain before the non-closer arm");
        get_size(cb, 0, 0, 0, 0, 0);
        CHECK(append_segment() == 0,
              "sceAmprCommandBufferGetSize is NOT a closer -- the chain survives it");
        CHECK(std::memcmp(seg_dst.data(), expected.data() + seg_offset, seg_size) == 0,
              "the segment after a non-closer still delivered the fixture's bytes");
    } else {
        CHECK(false, "sceAmprCommandBufferGetSize is registered");
    }

    // A chain on ANOTHER command buffer is not this one's: the refusal is keyed on cb, not global.
    {
        std::array<uint8_t, 0x48> other_request{};
        const uint64_t other_cb = (uint64_t)(uintptr_t)other_request.data();
        CHECK(open_chain(), "re-opened the chain before the per-cb arm");
        seg_dst.fill(0xEE);
        CHECK(gather_scatter(other_cb, 0, record, (uint64_t)(uintptr_t)seg_dst.data(), seg_size,
                             seg_offset) == kAprRefused,
              "a segment on a command buffer with no chain is refused");
    }

    // ---- Contract 2: read but not delivered is a FAILURE --------------------------------------
    // POSIX-only: the arm needs a destination the host can read a file into but cannot write to the
    // guest, and PROT_NONE is how that is expressed portably enough to be deterministic. The
    // Windows destination writer has no equivalent primitive in this file's toolkit.
#ifndef _WIN32
    {
        void* unwritable = mmap(nullptr, 0x2000, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(unwritable != MAP_FAILED, "reserved an unwritable guest destination for the arm");
        if (unwritable != MAP_FAILED) {
            CHECK(open_chain(), "re-opened the chain before the undelivered arm");
            // The range is inside the fixture and the chain is open, so every refusal reason the
            // handler has BEFORE the read is excluded: this can only fail on the destination.
            const uint64_t r = gather_scatter(cb, 0, record, (uint64_t)(uintptr_t)unwritable,
                                              seg_size, seg_offset);
            CHECK(r == kAprRefused,
                  "a segment whose bytes never reached the guest destination reports FAILURE");
            // Prove the arm is discriminating rather than refusing everything: the identical
            // segment against a writable destination on the same open chain still succeeds.
            CHECK(append_segment() == 0,
                  "the same segment to a writable destination still succeeds (arm discriminates)");
            munmap(unwritable, 0x2000);
        }
    }
#endif

    prosper_apr_reset_for_test();
    std::remove(fixture_path);
    std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
