#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#endif

using namespace prosper;

static uint64_t seen_initialize, seen_run[2], resumed[2];
static int entries;
#ifdef _WIN32
static bool fiber_bounds_valid[2];
static bool root_bounds_valid[2];

static bool current_stack_contains(const void* address) {
    NT_TIB* tib = (NT_TIB*)NtCurrentTeb();
    return (uintptr_t)address >= (uintptr_t)tib->StackLimit &&
           (uintptr_t)address < (uintptr_t)tib->StackBase;
}
#endif

extern "C" void fiber_body_ms(uint64_t initialize_arg, uint64_t run_arg) {
    uint8_t fiber_local = 0;
#ifdef _WIN32
    fiber_bounds_valid[0] = current_stack_contains(&fiber_local);
#endif
    seen_initialize = initialize_arg;
    seen_run[0] = run_arg;
    auto yield = Hle::lookup("B0ZX2hx9DMw");
    uint64_t next = 0;
    yield(0x1111, (uint64_t)(uintptr_t)&next, 0, 0, 0, 0);
    resumed[0] = next;
#ifdef _WIN32
    fiber_bounds_valid[1] = current_stack_contains(&fiber_local);
#endif
    seen_run[1] = next;
    yield(0x2222, (uint64_t)(uintptr_t)&next, 0, 0, 0, 0);
    resumed[1] = next;
    ++entries;
}

#ifdef _WIN32
// The production fiber entry is guest SysV code.  Keep this test's body as ordinary Microsoft-x64
// C++ (so MinGW can emit valid SEH metadata) and use a tiny assembly guest entry to marshal its two
// arguments.  Applying sysv_abi directly to a MinGW function with EH state produces invalid .seh data.
extern "C" void fiber_body_guest();
__asm__(
    ".text\n"
    ".p2align 4\n"
    ".globl fiber_body_guest\n"
    "fiber_body_guest:\n"
    "  movq %rdi,%rcx\n"
    "  movq %rsi,%rdx\n"
    "  subq $40,%rsp\n"
    "  call fiber_body_ms\n"
    "  addq $40,%rsp\n"
    "  ret\n"
);
#else
#define fiber_body_guest fiber_body_ms
#endif

static int fails;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    register_builtin_hle();
    using InitFn = uint64_t (*)(void*, const char*, void*, uint64_t, void*, uint64_t,
                                const void*, uint32_t);
    auto init = reinterpret_cast<InitFn>(Hle::lookup("hVYD7Ou2pCQ"));
    auto run = Hle::lookup("a0LLrZWac0M");
    alignas(16) uint8_t fiber[256]{};
    alignas(16) uint8_t stack[64 * 1024]{};
    CHECK(init && run && init(fiber, "test", (void*)fiber_body_guest, 0xabc, stack, sizeof(stack),
                              nullptr, 0) == 0,
          "fiber initializes with a guest-provided stack");
    uint64_t returned = 0;
    CHECK(run((uint64_t)(uintptr_t)fiber, 0x10, (uint64_t)(uintptr_t)&returned, 0, 0, 0) == 0 &&
          returned == 0x1111 && seen_initialize == 0xabc && seen_run[0] == 0x10,
          "first run enters the guest stack and yields to its thread");
#ifdef _WIN32
    uint8_t root_local = 0;
    root_bounds_valid[0] = current_stack_contains(&root_local);
#endif
    CHECK(run((uint64_t)(uintptr_t)fiber, 0x20, (uint64_t)(uintptr_t)&returned, 0, 0, 0) == 0 &&
          returned == 0x2222 && resumed[0] == 0x20 && seen_run[1] == 0x20,
          "second run resumes the suspended guest call stack");
#ifdef _WIN32
    root_bounds_valid[1] = current_stack_contains(&root_local);
    CHECK(fiber_bounds_valid[0] && fiber_bounds_valid[1],
          "Windows TEB describes the Sony fiber stack on entry and resume");
    CHECK(root_bounds_valid[0] && root_bounds_valid[1],
          "Windows TEB restores the native root stack after each fiber yield");
#endif
    // ---- sceFiberRename: a real implementation, not the dispatcher's return-0 default ----------
    // The failure this pins is the quiet one. An unregistered NID answers SCE_OK and changes
    // nothing, so a guest that renames a fiber is told it worked and then reads back the old name --
    // the same "answered OK for work not done" divergence as #2951. PPSA05684 imports this NID.
    {
        // The name field's offset is asserted rather than hardcoded blind: `init` above stored
        // "test" there, so if the layout ever moves, THIS control fails instead of the arms below
        // silently reading the wrong bytes and passing.
        const char* fiber_name = reinterpret_cast<const char*>(fiber + 40);
        CHECK(std::strcmp(fiber_name, "test") == 0,
              "CONTROL: GuestFiber::name really is at this offset -- initialize stored \"test\" in it");

        using RenameFn = uint64_t (*)(void*, const char*);
        auto rename_fn = reinterpret_cast<RenameFn>(Hle::lookup("JzyT91ucGDc"));
        CHECK(rename_fn != nullptr,
              "sceFiberRename is REGISTERED -- an unregistered NID falls to the return-0 default, "
              "which reports success and renames nothing");
        if (rename_fn) {
            CHECK(rename_fn(fiber, "renamed") == 0 && std::strcmp(fiber_name, "renamed") == 0,
                  "sceFiberRename actually renames the fiber, rather than answering SCE_OK and "
                  "leaving the old name in place");
            char long_name[64];
            std::memset(long_name, 'x', sizeof long_name - 1);
            long_name[sizeof long_name - 1] = '\0';
            CHECK(rename_fn(fiber, long_name) == 0 && std::strlen(fiber_name) == 31 &&
                      fiber_name[31] == '\0',
                  "...and an over-long name is truncated INTO the 32-byte field and NUL-terminated, "
                  "rather than running past it");
            CHECK(rename_fn(nullptr, "x") != 0, "a null fiber is refused");
            CHECK(rename_fn(fiber, nullptr) != 0, "a null name is refused");
        }
    }

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
