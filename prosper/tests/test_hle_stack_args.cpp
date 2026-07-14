// test_hle_stack_args - the generated import stub must preserve the guest's complete fixed-arity
// SysV call across the host ABI boundary. Windows needs a SysV->Microsoft-x64 conversion; Linux's
// guest-FS path interposes a call while swapping FS. Both paths must forward stack args 7-9.
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace prosper;

namespace {

constexpr uint64_t kArgs[9] = {
    0x1111111111111111ull, 0x2222222222222222ull, 0x3333333333333333ull,
    0x4444444444444444ull, 0x5555555555555555ull, 0x6666666666666666ull,
    0x7777777777777777ull, 0x8888888888888888ull, 0x9999999999999999ull,
};
constexpr uint64_t kReturn = 0xabcddcba01234567ull;
uint64_t g_seen[9] = {};
extern "C" {
uint64_t prosper_test_hle_entry_rsp_mod16 = ~uint64_t{0};
}

extern "C" uint64_t prosper_test_hle9_handler(
    uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
    uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8) {
    const uint64_t args[9] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
    for (int i = 0; i < 9; ++i) g_seen[i] = args[i];
    return kReturn;
}

extern "C" void prosper_test_hle9_entry();

#define PROSPER_STRINGIFY_INNER(x) #x
#define PROSPER_STRINGIFY(x) PROSPER_STRINGIFY_INNER(x)
#ifdef __APPLE__
#define PROSPER_ASM_SYMBOL(x) "_" PROSPER_STRINGIFY(x)
#else
#define PROSPER_ASM_SYMBOL(x) PROSPER_STRINGIFY(x)
#endif

// Record RSP before any compiler prologue, then tail-jump to the real host-ABI handler. A valid
// SysV or Microsoft-x64 callee enters with RSP%16==8.
__asm__(
    ".text\n"
    ".p2align 4\n"
    ".globl " PROSPER_ASM_SYMBOL(prosper_test_hle9_entry) "\n"
    PROSPER_ASM_SYMBOL(prosper_test_hle9_entry) ":\n"
    "    movq %rsp, %r10\n"
    "    andq $15, %r10\n"
    "    movq %r10, " PROSPER_ASM_SYMBOL(prosper_test_hle_entry_rsp_mod16) "(%rip)\n"
    "    jmp " PROSPER_ASM_SYMBOL(prosper_test_hle9_handler) "\n"
);

#ifdef _WIN32
using GuestHle9 = uint64_t (__attribute__((sysv_abi)) *)(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t);
#else
using GuestHle9 = uint64_t (*)(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t);
#endif

int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

} // namespace

int main(int argc, char** argv) {
    std::printf("== test_hle_stack_args ==\n");

#if defined(__linux__)
    const bool guest_fs = argc > 1 && std::string(argv[1]) == "--guest-fs";
    if (guest_fs) {
        setenv("PROSPER_GUEST_FS", "1", 1);
        TlsModuleDesc empty_module{};
        guest_tls_set_templates(&empty_module, 1);
        CHECK(guest_tls_enabled(), "Linux guest-FS import path enabled");
    }
#else
    (void)argc;
    (void)argv;
    const bool guest_fs = false;
#endif

    constexpr const char* kNid = "test.hle.stack.args";
    Hle::register_fn(kNid, reinterpret_cast<HleFn>(&prosper_test_hle9_entry),
                     "prosper_test_hle9_entry");
    const std::vector<ImportSlot> slots = {{"test", kNid}};
    std::string err;
    CHECK(install_stubs(slots, 0x710000000ull, 96, &err), "generated one executable import stub");
    if (fails) {
        if (!err.empty()) std::printf("  install error: %s\n", err.c_str());
        return 1;
    }

    bool guest_fs_active = false;
#if defined(__linux__)
    // Do not call libc between activation and the import: the test thread is deliberately running
    // with a guest TCB until the generated stub and the explicit restore below return it to host FS.
    if (guest_fs) guest_fs_active = guest_tls_activate_thread() != 0;
#endif

    auto call = reinterpret_cast<GuestHle9>(static_cast<uintptr_t>(stub_addr(0)));
    const uint64_t result = call(kArgs[0], kArgs[1], kArgs[2], kArgs[3], kArgs[4],
                                 kArgs[5], kArgs[6], kArgs[7], kArgs[8]);

#if defined(__linux__)
    if (guest_fs_active) guest_fs_enter_host_for_signal();
#endif

    CHECK(!guest_fs || guest_fs_active, "activated a guest FS base for the call");
    CHECK(result == kReturn, "return value crossed the import ABI bridge");
    CHECK(prosper_test_hle_entry_rsp_mod16 == 8, "host handler entered with RSP%16 == 8");
    for (int i = 0; i < 9; ++i) {
        char what[96];
        std::snprintf(what, sizeof what, "argument %d preserved (0x%016llx)", i + 1,
                      static_cast<unsigned long long>(kArgs[i]));
        CHECK(g_seen[i] == kArgs[i], what);
    }

    if (fails) {
        std::printf("== FAIL: %d ==\n", fails);
        return 1;
    }
    std::printf("== PASS ==\n");
    return 0;
}
