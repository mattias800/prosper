// test_module_entry_abi -- a module entry runs with System V arguments and leaves the host's
// callee-saved registers intact.
//
// THE DEFECT. Deferred module init (and runtime module load) called the guest entry through a plain C
// function pointer. Guest code is System V; on Windows the host ABI is Microsoft x64, so the plain call
// delivered `args`/`argp` in RCX/RDX instead of RDI/RSI and let a SysV guest clobber RBX/RSI/RDI/R12-R15
// that Microsoft x64 requires preserved. The platform bridge must translate both. DT_INIT also
// consumes an optional third alternate-startup callback: the old Windows two-arg bridge left argc
// there, making a descriptor-sized argc (0x10) a jump target. The descriptor itself must carry two
// 32-bit fields then one 64-bit callback, not three 64-bit fields (#4283).
//
// WHAT EACH TEST KILLS:
//   DeliversSysvArguments        a plain-pointer call: the stub would read RDI/RSI as garbage on Windows
//   PreservesHostCalleeSavedRegs the stub clobbers RSI/RDI/XMM6/7; a missing save/restore corrupts the loop
#include "host/image/runtime_module_load.hpp"
#include "host/image/exec_image.hpp"
#include "host/image/module_start_params.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

struct Slots {
    uint64_t arg0, arg1, alternate_start;
};
static_assert(offsetof(Slots, alternate_start) == 16);

void* make_exec(const uint8_t* code, size_t n, void* requested = nullptr) {
    if (n > 4096) return nullptr;
#ifdef _WIN32
    void* p = VirtualAlloc(requested, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p = mmap(requested, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    if (p == MAP_FAILED) p = nullptr;
#endif
    if (p) {
        std::memcpy(p, code, n);
#ifdef _WIN32
        FlushInstructionCache(GetCurrentProcess(), p, n);
#endif
    }
    return p;
}

// System V stub: store rdi/rsi to *slots, clobber RSI/RDI/XMM6/XMM7 (volatile in SysV, nonvolatile in Microsoft x64), return 0x77.
std::vector<uint8_t> stub_for(Slots* slots) {
    std::vector<uint8_t> c = {0x48, 0xB8};   // mov rax, imm64
    const uint64_t a = reinterpret_cast<uint64_t>(slots);
    for (int i = 0; i < 8; ++i) c.push_back(static_cast<uint8_t>(a >> (8 * i)));
    const uint8_t rest[] = {
        0x48, 0x89, 0x38,   // mov [rax], rdi
        0x48, 0x89, 0x70, 0x08,   // mov [rax+8], rsi
        0x48, 0x89, 0x50, 0x10,   // mov [rax+16], rdx
        0x66, 0x0F, 0x76, 0xF6,   // pcmpeqd xmm6, xmm6 (all ones)
        0x66, 0x0F, 0x76, 0xFF,   // pcmpeqd xmm7, xmm7
        0x48, 0xC7, 0xC6, 0xFF, 0xFF, 0xFF, 0xFF,   // mov rsi, -1
        0x48, 0xC7, 0xC7, 0xFF, 0xFF, 0xFF, 0xFF,   // mov rdi, -1
        0xB8, 0x77, 0x00, 0x00, 0x00,   // mov eax, 0x77
        0xC3,   // ret
    };
    c.insert(c.end(), rest, rest + sizeof rest);
    return c;
}

struct InitWitness {
    uint64_t argc, argp, alternate_start;
    uint32_t size, version;
    uint64_t callback, calls;
};
static_assert(offsetof(InitWitness, argc) == 0);
static_assert(offsetof(InitWitness, argp) == 8);
static_assert(offsetof(InitWitness, alternate_start) == 16);
static_assert(offsetof(InitWitness, size) == 24);
static_assert(offsetof(InitWitness, version) == 28);
static_assert(offsetof(InitWitness, callback) == 32);
static_assert(offsetof(InitWitness, calls) == 40);

// Independently authored SysV leaf. It witnesses physical argument registers and consumes literal
// descriptor byte offsets; it shares neither production field names nor descriptor parsing code.
std::vector<uint8_t> init_stub_for(InitWitness* witness, bool reads_descriptor) {
    std::vector<uint8_t> code{0x48, 0xb8};   // mov rax, witness
    const uint64_t address = reinterpret_cast<uint64_t>(witness);
    for (int i = 0; i < 8; ++i) code.push_back(static_cast<uint8_t>(address >> (8 * i)));
    const uint8_t registers[] = {
        0x48, 0x89, 0x38,   // mov [rax], rdi
        0x48, 0x89, 0x70, 0x08,   // mov [rax+8], rsi
        0x48, 0x89, 0x50, 0x10,   // mov [rax+16], rdx
        0x48, 0xff, 0x40, 0x28,   // inc qword [rax+40]
    };
    code.insert(code.end(), std::begin(registers), std::end(registers));
    if (reads_descriptor) {
        const uint8_t fields[] = {
            0x8b, 0x0e,   // mov ecx, [rsi]
            0x89, 0x48, 0x18,   // mov [rax+24], ecx
            0x8b, 0x4e, 0x04,   // mov ecx, [rsi+4]
            0x89, 0x48, 0x1c,   // mov [rax+28], ecx
            0x48, 0x8b, 0x4e, 0x08,   // mov rcx, [rsi+8]
            0x48, 0x89, 0x48, 0x20,   // mov [rax+32], rcx
        };
        code.insert(code.end(), std::begin(fields), std::end(fields));
    }
    const uint8_t finish[] = {0xb8, 0x77, 0, 0, 0, 0xc3};   // mov eax, 0x77; ret
    code.insert(code.end(), std::begin(finish), std::end(finish));
    return code;
}

class GuestEntry {
public:
    explicit GuestEntry(const std::vector<uint8_t>& code, void* requested = nullptr)
        : mapping_(make_exec(code.data(), code.size(), requested)) {}
    ~GuestEntry() {
        if (!mapping_) return;
#ifdef _WIN32
        VirtualFree(mapping_, 0, MEM_RELEASE);
#else
        munmap(mapping_, 4096);
#endif
    }
    GuestEntry(const GuestEntry&) = delete;
    GuestEntry& operator=(const GuestEntry&) = delete;
    uint64_t address() const { return reinterpret_cast<uint64_t>(mapping_); }

private:
    void* mapping_;
};

}   // namespace

TEST(ModuleEntryAbi, DeliversSysvArguments) {
    Slots slots{};
    const auto code = stub_for(&slots);
    void* fn = make_exec(code.data(), code.size());
    ASSERT_NE(fn, nullptr);
    const uint64_t r = prosper::call_guest_module_entry(
        reinterpret_cast<uint64_t>(fn), 0x1122334455667788ull, 0x99AABBCCDDEEFF00ull, 0);
    EXPECT_EQ(r, 0x77u);
    EXPECT_EQ(slots.arg0, 0x1122334455667788ull) << "args must arrive in RDI";
    EXPECT_EQ(slots.arg1, 0x99AABBCCDDEEFF00ull) << "argp must arrive in RSI";
    EXPECT_EQ(slots.alternate_start, 0u) << "DT_INIT must not inherit argc as a startup callback";
}

TEST(ModuleEntryAbi, PreservesHostCalleeSavedRegs) {
    Slots slots{};
    const auto code = stub_for(&slots);
    void* fn = make_exec(code.data(), code.size());
    ASSERT_NE(fn, nullptr);
    // Values live across the calls in registers the stub clobbers; a missing restore changes them.
    volatile uint64_t a = 1, b = 2, c = 3, d = 4, e = 5, f = 6;
    uint64_t sum = 0;
    double x = 1.0, y = 2.0;   // the compiler keeps these in XMM6+ across the call
    for (uint64_t i = 0; i < 64; ++i) {
        x = x * 1.25 + 0.5;
        y = y * 0.5 + 1.0;
        sum += prosper::call_guest_module_entry(reinterpret_cast<uint64_t>(fn), i, i + 1, 0);
        a = a + 1;
        b = b + 2;
        c = c + 3;
        d = d + 4;
        e = e + 5;
        f = f + 6;
    }
    EXPECT_EQ(sum, 64u * 0x77u);
    EXPECT_EQ(a, 65u);
    EXPECT_EQ(b, 130u);
    EXPECT_EQ(c, 195u);
    double ex = 1.0, ey = 2.0;
    for (int i = 0; i < 64; ++i) {
        ex = ex * 1.25 + 0.5;
        ey = ey * 0.5 + 1.0;
    }
    EXPECT_EQ(x, ex);
    EXPECT_EQ(y, ey);
    EXPECT_EQ(d, 260u);
    EXPECT_EQ(e, 325u);
    EXPECT_EQ(f, 390u);
}

TEST(ModuleEntryAbi, ConsumesDescriptorWithNullAlternateStartup) {
    InitWitness witness{};
    auto descriptor = prosper::kModuleStartDescriptor;
    descriptor.callback = 0x12345678abcdef01ull;   // witness all 64 callback bits, not only zero
    GuestEntry entry(init_stub_for(&witness, true));
    ASSERT_NE(entry.address(), 0u);
    const auto argp = reinterpret_cast<uint64_t>(&descriptor);
    EXPECT_EQ(prosper::call_guest_module_entry(entry.address(), 0x10, argp, 0), 0x77u);
    EXPECT_EQ(witness.argc, 0x10u);
    EXPECT_EQ(witness.argp, argp);
    EXPECT_EQ(witness.alternate_start, 0u);
    EXPECT_EQ(witness.size, 0x10u);
    EXPECT_EQ(witness.version, 0x200u);
    EXPECT_EQ(witness.callback, 0x12345678abcdef01ull);
    EXPECT_EQ(witness.calls, 1u);
}

#ifdef _WIN32
TEST(ModuleEntryAbi, PreloadedInitConsumesActualDescriptorAndDefaultArguments) {
    InitWitness descriptor_witness{}, default_witness{};
    auto code = init_stub_for(&descriptor_witness, true);
    code.insert(code.begin(), 16, 0xcc);   // entry at the selected module's ordinary base+0x10
    GuestEntry descriptor_entry(code, reinterpret_cast<void*>(prosper::BOOT_PSN));
    ASSERT_EQ(descriptor_entry.address(), prosper::BOOT_PSN) << "never replace an occupied mapping";
    GuestEntry default_entry(init_stub_for(&default_witness, false));
    ASSERT_NE(default_entry.address(), 0u);
    ASSERT_FALSE(prosper::module_start_wants_param_descriptor(default_entry.address()));
    struct ParamRangesScope {
        ParamRangesScope() {
            prosper::set_module_start_param_ranges(prosper::module_start_param_ranges());
        }
        ~ParamRangesScope() { prosper::set_module_start_param_ranges({}); }
    } ranges;
    prosper::install_trap_handler();
    const std::vector<uint64_t> inits{default_entry.address(), descriptor_entry.address() + 16,
                                      default_entry.address()};
    EXPECT_EQ(prosper::run_guest_inits(inits), 3u) << "all genuine initializers execute normally";
    EXPECT_EQ(descriptor_witness.argc, 0x10u);
    EXPECT_NE(descriptor_witness.argp, 0u);
    EXPECT_EQ(descriptor_witness.alternate_start, 0u);
    EXPECT_EQ(descriptor_witness.size, 0x10u);
    EXPECT_EQ(descriptor_witness.version, 0x200u);
    EXPECT_EQ(descriptor_witness.callback, 0u);
    EXPECT_EQ(descriptor_witness.calls, 1u);
    EXPECT_EQ(default_witness.argc, 0u);
    EXPECT_EQ(default_witness.argp, 0u);
    EXPECT_EQ(default_witness.alternate_start, 0u);
    EXPECT_EQ(default_witness.calls, 2u);
}
#endif
