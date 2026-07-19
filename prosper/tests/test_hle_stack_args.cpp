// test_hle_stack_args - the generated import stub must preserve the guest's complete fixed-arity
// SysV call across the host ABI boundary. Windows needs a SysV->Microsoft-x64 conversion; Linux's
// guest-FS path interposes a call while swapping FS. Both paths must forward stack args 7-10.
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper;

namespace {

constexpr uint64_t kArgs[10] = {
    0x1111111111111111ull, 0x2222222222222222ull, 0x3333333333333333ull,
    0x4444444444444444ull, 0x5555555555555555ull, 0x6666666666666666ull,
    0x7777777777777777ull, 0x8888888888888888ull, 0x9999999999999999ull,
    0xaaaaaaaaaaaaaaaaull,
};
constexpr uint64_t kReturn = 0xabcddcba01234567ull;
uint64_t g_seen[10] = {};
extern "C" {
uint64_t prosper_test_hle_entry_rsp_mod16 = ~uint64_t{0};
uint64_t prosper_test_hle_entry_rsp = 0;
}
uint64_t g_immediate_return = 0;
uint64_t g_guest_return = 0;

extern "C" uint64_t prosper_test_hle10_handler(
    uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
    uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8, uint64_t a9) {
    const uint64_t args[10] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9};
    for (int i = 0; i < 10; ++i) g_seen[i] = args[i];
    g_immediate_return = *(const uint64_t*)(uintptr_t)prosper_test_hle_entry_rsp;
    g_guest_return = hle_guest_return_address(prosper_test_hle_entry_rsp);
    return kReturn;
}

extern "C" void prosper_test_hle10_entry();

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
    ".globl " PROSPER_ASM_SYMBOL(prosper_test_hle10_entry) "\n"
    PROSPER_ASM_SYMBOL(prosper_test_hle10_entry) ":\n"
    "    movq %rsp, " PROSPER_ASM_SYMBOL(prosper_test_hle_entry_rsp) "(%rip)\n"
    "    movq %rsp, %r10\n"
    "    andq $15, %r10\n"
    "    movq %r10, " PROSPER_ASM_SYMBOL(prosper_test_hle_entry_rsp_mod16) "(%rip)\n"
    "    jmp " PROSPER_ASM_SYMBOL(prosper_test_hle10_handler) "\n"
);

#ifdef _WIN32
using GuestHle10 = uint64_t (__attribute__((sysv_abi)) *)(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using GuestHle8 = uint64_t (__attribute__((sysv_abi)) *)(
    uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t);
#else
using GuestHle10 = uint64_t (*)(
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
    uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using GuestHle8 = uint64_t (*)(
    uint64_t, uint64_t, uint64_t, uint64_t,
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
    register_builtin_hle();
    Hle::register_fn(kNid, reinterpret_cast<HleFn>(&prosper_test_hle10_entry),
                     "prosper_test_hle10_entry");
    // Keep an unresolved import immediately before the implemented one. The largest Linux guest-FS
    // variant is the unresolved stub; it must fit its fixed 96-byte slot without corrupting its
    // neighbour.
    const std::vector<ImportSlot> slots = {
        {"test", "test.hle.unimplemented"}, {"test", kNid},
        {"libSceVideoOut", "PjS5uASwcV8"},
    };
    std::string err;
    dispatch_init(&slots, nullptr);
    CHECK(install_stubs(slots, 0x710000000ull, 96, &err), "generated adjacent unresolved and executable import stubs");
#if defined(__linux__)
    if (guest_fs) {
        const auto* second = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(stub_addr(1)));
        CHECK(second[0] == 0x49 && second[1] == 0xBA,
              "unresolved guest-FS stub stayed within its 96-byte slot");
    }
#endif
    if (fails) {
        if (!err.empty()) std::printf("  install error: %s\n", err.c_str());
        return 1;
    }

#ifdef _WIN32
    _putenv_s("PROSPER_UNIMPL_STACK", "1");
#endif
    auto unresolved = reinterpret_cast<GuestHle10>(static_cast<uintptr_t>(stub_addr(0)));
    CHECK(unresolved(kArgs[0], kArgs[1], kArgs[2], kArgs[3], kArgs[4],
                     kArgs[5], kArgs[6], kArgs[7], kArgs[8], kArgs[9]) == 0,
          "unresolved import stub returned the generic result");
    CHECK(call_order().size() == 1 && call_order()[0] == 0,
          "unresolved import stub preserved its dispatch index");

    bool guest_fs_active = false;
#if defined(__linux__)
    // Do not call libc between activation and the import: the test thread is deliberately running
    // with a guest TCB until the generated stub and the explicit restore below return it to host FS.
    if (guest_fs) guest_fs_active = guest_tls_activate_thread() != 0;
#endif

    auto call = reinterpret_cast<GuestHle10>(static_cast<uintptr_t>(stub_addr(1)));
    const uint64_t callsite_begin = (uint64_t)(uintptr_t)&&before_hle_call;
    const uint64_t callsite_end = (uint64_t)(uintptr_t)&&after_hle_call;
before_hle_call:
    asm volatile("" ::: "memory");
    const uint64_t result = call(kArgs[0], kArgs[1], kArgs[2], kArgs[3], kArgs[4],
                                 kArgs[5], kArgs[6], kArgs[7], kArgs[8], kArgs[9]);
    asm volatile("" ::: "memory");
after_hle_call:

#if defined(__linux__)
    if (guest_fs_active) guest_fs_enter_host_for_signal();
#endif

    CHECK(!guest_fs || guest_fs_active, "activated a guest FS base for the call");
    CHECK(result == kReturn, "return value crossed the import ABI bridge");
    CHECK(prosper_test_hle_entry_rsp_mod16 == 8, "host handler entered with RSP%16 == 8");
    const bool called_bridge =
#if defined(_WIN32)
        true;
#elif defined(__linux__)
        guest_fs;
#else
        false;
#endif
    const uint64_t stub_begin = stub_addr(1);
    const uint64_t stub_end = stub_begin + 96;
    if (called_bridge) {
        CHECK(g_immediate_return >= stub_begin && g_immediate_return < stub_end,
              "bridge handler's immediate return points into its generated stub");
    } else {
        CHECK(g_immediate_return == g_guest_return,
              "tail-jump handler uses its immediate return as the guest callsite");
    }
    const uint64_t callsite_lo = callsite_begin < callsite_end ? callsite_begin : callsite_end;
    const uint64_t callsite_hi = callsite_begin < callsite_end ? callsite_end : callsite_begin;
    CHECK(g_guest_return >= callsite_lo && g_guest_return <= callsite_hi,
          "recovered HLE caller points to the real import callsite, not the generated stub");
    for (int i = 0; i < 10; ++i) {
        char what[96];
        std::snprintf(what, sizeof what, "argument %d preserved (0x%016llx)", i + 1,
                      static_cast<unsigned long long>(kArgs[i]));
        CHECK(g_seen[i] == kArgs[i], what);
    }

    // Exercise a real eight-argument HLE through the same generated import path. This catches an
    // accidental six-argument registration even if the generic ten-argument bridge still passes.
    if (!guest_fs) {
        uint8_t attr[0x58];
        std::memset(attr, 0xee, sizeof attr);
        constexpr uint64_t kDccClear = 0x8877665544332211ull;
        constexpr uint32_t kDccControl = 0xa1b2c3d4u;
        auto set_attribute2 = reinterpret_cast<GuestHle8>(
            static_cast<uintptr_t>(stub_addr(2)));
        CHECK(set_attribute2(reinterpret_cast<uint64_t>(attr), 0x8000000000000000ull,
                             0, 1920, 1080, 0, kDccControl, kDccClear) == 0,
              "real eight-argument VideoOut import returned success");
        CHECK(*reinterpret_cast<const uint64_t*>(attr + 0x28) == kDccClear &&
              *reinterpret_cast<const uint32_t*>(attr + 0x30) == kDccControl,
              "real VideoOut import preserved stack arguments 7 and 8");
        bool canary_untouched = true;
        for (size_t i = 0x50; i < sizeof attr; ++i) canary_untouched &= attr[i] == 0xee;
        CHECK(canary_untouched, "real VideoOut import retained the 0x50-byte write bound");
    }

    if (fails) {
        std::printf("== FAIL: %d ==\n", fails);
        return 1;
    }
    std::printf("== PASS ==\n");
    return 0;
}
