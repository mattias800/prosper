// #3615: the import stub stashes the CALLING host thread's guest thread pointer on the guest stack
// and restores %fs from it in its epilogue. A fiber suspended inside such a call can be resumed by a
// DIFFERENT host thread, which then returns through that frame and installs a foreign guest TCB --
// after which every guest initial-exec TLS read on it resolves to another thread's storage.
//
// These arms cover the two pure pieces of the repair: locating the stashed slot in a stub frame, and
// re-pointing it at the resuming thread. Both are exercised against a hand-built frame rather than a
// real stub, so a change to the stub's frame layout is meant to redden `kStubGuestFsSlotOffset` here.
#include "hle/dispatch/callback_fs.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

// A stand-in for one of prosper's guest TCBs: 16-byte aligned, self-pointer at +0, "PROS" at +0x108.
struct FakeTcb {
    std::vector<uint8_t> mem;
    uint64_t tp = 0;
    explicit FakeTcb(bool valid_magic = true, bool valid_self = true) : mem(0x400, 0) {
        tp = ((uint64_t)(uintptr_t)mem.data() + 0xf) & ~0xfull;
        *(uint64_t*)(uintptr_t)tp = valid_self ? tp : tp + 8;
        *(uint32_t*)(uintptr_t)(tp + 0x108) = valid_magic ? 0x50524F53u : 0x11223344u;
    }
};

// The stub frame as an HLE entry sees it: [+0]=return-into-stub, [+0x28]=stashed guest %fs.
struct FakeFrame {
    uint64_t slots[16] = {};
    uint64_t entry_rsp() { return (uint64_t)(uintptr_t)slots; }
    void build(uint64_t shim_ret, uint64_t stashed_fs) {
        std::memset(slots, 0, sizeof slots);
        slots[0] = shim_ret;
        *(uint64_t*)(uintptr_t)(entry_rsp() + kStubGuestFsSlotOffset) = stashed_fs;
    }
};


// ---- a fiber that suspends on one host thread and is resumed by another --------------------------
// Linux/macOS only: the guest body is entered with the SysV ABI, which on Windows needs the assembly
// shim test_fiber.cpp carries, and the %fs stash this file is about is a Linux mechanism anyway.
#ifndef _WIN32
static uint64_t g_run_arg[2];
static void*    g_self_after_resume;
static int      g_stack_survived;

extern "C" PROSPER_SYSV_ABI void fiber_body_migrate(uint64_t initialize_arg, uint64_t run_arg) {
    volatile uint64_t cookie = 0xC0FFEE0000ull + initialize_arg;   // a genuine guest-stack local
    g_run_arg[0] = run_arg;
    auto yield = Hle::lookup("B0ZX2hx9DMw");
    auto self  = Hle::lookup("p+zLIOg27zU");
    uint64_t next = 0;
    yield(0x3333, (uint64_t)(uintptr_t)&next, 0, 0, 0, 0);
    // Resumed -- on a different host thread than the one that suspended us.
    g_run_arg[1] = next;
    g_stack_survived = (cookie == 0xC0FFEE0000ull + initialize_arg) ? 1 : 0;
    void* who = nullptr;
    self((uint64_t)(uintptr_t)&who, 0, 0, 0, 0, 0);
    g_self_after_resume = who;
    yield(0x4444, (uint64_t)(uintptr_t)&next, 0, 0, 0, 0);
}

static void migration_arms() {
    register_builtin_hle();
    using InitFn = uint64_t (*)(void*, const char*, void*, uint64_t, void*, uint64_t,
                                const void*, uint32_t);
    auto init = reinterpret_cast<InitFn>(Hle::lookup("hVYD7Ou2pCQ"));
    auto run  = Hle::lookup("a0LLrZWac0M");
    alignas(16) static uint8_t fiber[256]{};
    alignas(16) static uint8_t stack[64 * 1024]{};
    if (!init || !run) { std::printf("  [FAIL] fiber NIDs not registered\n"); ++fails; return; }

    uint64_t returned = 0;
    CHECK(init(fiber, "migrate", (void*)fiber_body_migrate, 0xabc, stack, sizeof stack,
               nullptr, 0) == 0 &&
              run((uint64_t)(uintptr_t)fiber, 0x30, (uint64_t)(uintptr_t)&returned, 0, 0, 0) == 0 &&
              returned == 0x3333 && g_run_arg[0] == 0x30,
          "CONTROL: the fiber starts and yields on THIS host thread");

    uint64_t rc = ~0ull, returned2 = 0;
    std::thread other([&] {
        rc = run((uint64_t)(uintptr_t)fiber, 0x40, (uint64_t)(uintptr_t)&returned2, 0, 0, 0);
    });
    other.join();

    CHECK(rc == 0 && returned2 == 0x4444 && g_run_arg[1] == 0x40 && g_stack_survived == 1,
          "a fiber suspended on one host thread resumes on ANOTHER, with its guest call stack and "
          "locals intact");
    CHECK(g_self_after_resume == (void*)fiber,
          "sceFiberGetSelf on the resuming host thread names the migrated fiber");
}
#else
static void migration_arms() {}
#endif

int main() {
    const uint64_t kGoodShimRet = 0x600001000ull;   // inside the emitted-stub aperture

    // ---- locating the slot -------------------------------------------------------------------
    {
        FakeTcb tcb;
        FakeFrame f; f.build(kGoodShimRet, tcb.tp);

        // CONTROL first. Every rejection arm below asserts a 0, and a fixture that simply cannot be
        // accepted would pass all of them while testing nothing.
        const uint64_t slot = callback_guest_fs_slot_from_entry_stack(f.entry_rsp());
        CHECK(slot == f.entry_rsp() + 0x28,
              "CONTROL: a well-formed stub frame IS accepted, and the slot is at entry_rsp+0x28");
        CHECK(slot && *(uint64_t*)(uintptr_t)slot == tcb.tp &&
                  callback_guest_fs_from_entry_stack(f.entry_rsp()) == tcb.tp,
              "CONTROL: the value read through the slot is the stashed guest TP");

        f.build(0x500001000ull, tcb.tp);
        CHECK(callback_guest_fs_slot_from_entry_stack(f.entry_rsp()) == 0,
              "a frame whose return address is outside the stub aperture is refused");

        f.build(kGoodShimRet, tcb.tp + 4);
        CHECK(callback_guest_fs_slot_from_entry_stack(f.entry_rsp()) == 0,
              "a misaligned stashed pointer is refused");

        f.build(kGoodShimRet, 0x40);
        CHECK(callback_guest_fs_slot_from_entry_stack(f.entry_rsp()) == 0,
              "a small integer where the thread pointer should be is refused");

        FakeTcb no_magic(/*valid_magic=*/false);
        f.build(kGoodShimRet, no_magic.tp);
        CHECK(callback_guest_fs_slot_from_entry_stack(f.entry_rsp()) == 0,
              "a pointer that is not one of OUR guest TCBs (no PROS magic) is refused");

        FakeTcb no_self(/*valid_magic=*/true, /*valid_self=*/false);
        f.build(kGoodShimRet, no_self.tp);
        CHECK(callback_guest_fs_slot_from_entry_stack(f.entry_rsp()) == 0,
              "a TCB whose self-pointer does not point at itself is refused");

        CHECK(callback_guest_fs_slot_from_entry_stack(0) == 0, "a null entry_rsp is refused");
    }

    // ---- repairing the slot ------------------------------------------------------------------
    {
        FakeTcb suspender, resumer;
        FakeFrame f; f.build(kGoodShimRet, suspender.tp);
        const uint64_t slot = callback_guest_fs_slot_from_entry_stack(f.entry_rsp());
        CHECK(slot != 0, "CONTROL: the repair fixture's frame is accepted");

        CHECK(callback_repair_guest_fs_slot(slot, resumer.tp) &&
                  *(uint64_t*)(uintptr_t)slot == resumer.tp,
              "a fiber resumed on another host thread re-points the stashed TP at THAT thread");

        CHECK(!callback_repair_guest_fs_slot(slot, resumer.tp) &&
                  *(uint64_t*)(uintptr_t)slot == resumer.tp,
              "resuming on the thread that suspended is a no-op, not a redundant write");

        f.build(kGoodShimRet, suspender.tp);
        CHECK(!callback_repair_guest_fs_slot(slot, 0) &&
                  *(uint64_t*)(uintptr_t)slot == suspender.tp,
              "a thread with no guest TCB of its own leaves the frame ALONE -- writing 0 here would "
              "make the stub epilogue wrfsbase 0 and strand the guest with no TLS at all");

        CHECK(!callback_repair_guest_fs_slot(0, resumer.tp),
              "a call with no stub frame does nothing");
    }

    migration_arms();

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
