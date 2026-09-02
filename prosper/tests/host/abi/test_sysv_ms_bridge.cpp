// test_sysv_ms_bridge - the guest(System V AMD64) -> host(Microsoft x64) import bridge (#2955).
//
// The PS5 guest always places arguments by the System V convention. On Windows the HLE handler reads
// them by the Microsoft one, and the two do not merely name different registers: System V runs
// SEPARATE counters for integer and SSE arguments, Microsoft runs ONE positional counter and picks
// the register file from the type. So a single float in the middle of a signature displaces every
// argument AFTER it as well, and a bridge that remaps integer registers only delivers garbage to the
// float and to everything behind it.
//
// Two independent things are checked here, and the split matters because only one of them needs a
// Windows host:
//   (1) the PLACEMENT TABLES - a pure function of the signature, asserted directly on any platform;
//   (2) the EMITTED MACHINE CODE - executed, on any x86-64 host, by calling the real trampoline
//       bytes with a System V call and landing in a Microsoft-ABI handler. `__attribute__((ms_abi))`
//       makes that boundary available on Linux, so the exact bytes a Windows build installs are run
//       and every argument is asserted to arrive intact.
// What NEITHER can check is a live guest calling a real Sony import on Windows. Nothing here claims
// to; the bridge is verified, the Windows boot is not.
#include "host/abi/sysv_ms_bridge.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#define PROSPER_TEST_CAN_EXECUTE 1
#if defined(_WIN32)
#include <windows.h>
// The host ABI already IS Microsoft x64, so the GUEST side is the one needing an attribute.
#define PROSPER_TEST_GUEST_ABI __attribute__((sysv_abi))
#define PROSPER_TEST_HOST_ABI
#else
#include <sys/mman.h>
// The host ABI already IS System V, so the HOST side is the one needing an attribute.
#define PROSPER_TEST_GUEST_ABI
#define PROSPER_TEST_HOST_ABI __attribute__((ms_abi))
#endif
#else
#define PROSPER_TEST_CAN_EXECUTE 0
#endif

using namespace prosper;
using namespace prosper::abi;
using Kind = ArgLocation::Kind;

namespace {
int g_fail = 0;

// Compile-time coverage for the deducer, checked by every build on every platform. The argument is
// only ever a TYPE to signature_of, so a null pointer is enough and no symbol is referenced.
template <class F> constexpr CallSignature deduced() { return signature_of(static_cast<F>(nullptr)); }
static_assert(deduced<void (*)(float, int*, int*)>().count == 3);
static_assert(deduced<void (*)(float, int*, int*)>().sse_mask == 0b001);
static_assert(!deduced<void (*)(float, int*, int*)>().sse_return);
static_assert(deduced<void (*)(float, int*, int*)>().needs_conversion());
static_assert(deduced<float (*)(const char*, char**)>().sse_mask == 0);
static_assert(deduced<float (*)(const char*, char**)>().sse_return);
static_assert(deduced<float (*)(const char*, char**)>().needs_conversion());
static_assert(deduced<int32_t (*)(void*, uint32_t, void*, float, float, void*, void*)>().count == 7);
static_assert(deduced<int32_t (*)(void*, uint32_t, void*, float, float, void*, void*)>().sse_mask
              == 0b0011000);
static_assert(!deduced<int32_t (*)(void*, uint32_t, void*, float, float, void*, void*)>().sse_return);
// The one shape that must NOT be recorded: no float anywhere, so the historical bridge stands.
static_assert(deduced<uint64_t (*)(void*, uint64_t, void*)>().sse_mask == 0);
static_assert(!deduced<uint64_t (*)(void*, uint64_t, void*)>().needs_conversion());
// A void return is not an SSE return -- and it is the shape that caught a portability defect in the
// deducer, because `sizeof(void)` is a GCC extension that clang rightly rejects.
static_assert(!deduced<void (*)()>().sse_return);
static_assert(!deduced<void (*)()>().needs_conversion());

void fail(const char* what, const char* detail) {
    fprintf(stderr, "FAIL %-46s: %s\n", what, detail);
    ++g_fail;
}

// ---------------------------------------------------------------------------------------------
// (1) Placement tables.
// ---------------------------------------------------------------------------------------------

CallSignature sig_of(std::initializer_list<ArgClass> args, bool sse_return = false) {
    CallSignature s{};
    s.declared = true;
    s.count = (uint8_t)args.size();
    unsigned i = 0;
    for (ArgClass c : args) { if (c == ArgClass::Sse) s.sse_mask |= (uint16_t)(1u << i); ++i; }
    s.sse_return = sse_return;
    return s;
}
constexpr ArgClass I = ArgClass::Integer;
constexpr ArgClass F = ArgClass::Sse;

std::string show(const ArgLocation& l) {
    static const char* gpr[] = { "rax","rcx","rdx","rbx","rsp","rbp","rsi","rdi",
                                 "r8","r9","r10","r11" };
    char buf[32];
    switch (l.kind) {
    case Kind::IntReg: return l.reg < 12 ? gpr[l.reg] : "gpr?";
    case Kind::SseReg: snprintf(buf, sizeof buf, "xmm%u", (unsigned)l.reg); return buf;
    case Kind::Stack:  snprintf(buf, sizeof buf, "stack%u", (unsigned)l.slot); return buf;
    }
    return "?";
}
ArgLocation int_reg(uint8_t r) { return { Kind::IntReg, r, 0 }; }
ArgLocation sse_reg(uint8_t r) { return { Kind::SseReg, r, 0 }; }
ArgLocation stack(uint8_t s)   { return { Kind::Stack, 0, s }; }

void expect_places(const char* name, const CallSignature& sig,
                   std::initializer_list<ArgLocation> sysv,
                   std::initializer_list<ArgLocation> ms) {
    unsigned i = 0;
    for (const ArgLocation& want : sysv) {
        const ArgLocation got = sysv_arg_location(sig, i);
        if (!(got == want)) {
            char d[160];
            snprintf(d, sizeof d, "SysV arg %u is %s, expected %s", i,
                     show(got).c_str(), show(want).c_str());
            fail(name, d);
        }
        ++i;
    }
    i = 0;
    for (const ArgLocation& want : ms) {
        const ArgLocation got = ms_arg_location(sig, i);
        if (!(got == want)) {
            char d[160];
            snprintf(d, sizeof d, "MS arg %u is %s, expected %s", i,
                     show(got).c_str(), show(want).c_str());
            fail(name, d);
        }
        ++i;
    }
}

void check_tables() {
    // The signature from #2955 itself: sceFontRenderCharGlyphImage. Everything from argument 4 on is
    // displaced, which is the part a "just add xmm moves" fix still gets wrong -- the two POINTERS
    // after the floats land on the Microsoft stack, not in r9 and the first stack slot.
    expect_places("render_char_glyph_image (ptr,u32,ptr,f,f,ptr,ptr)",
                  sig_of({ I, I, I, F, F, I, I }),
                  { int_reg(kRdi), int_reg(kRsi), int_reg(kRdx),
                    sse_reg(0), sse_reg(1), int_reg(kRcx), int_reg(kR8) },
                  { int_reg(kRcx), int_reg(kRdx), int_reg(kR8),
                    sse_reg(3), stack(4), stack(5), stack(6) });

    // No float: the two conventions place identically for the first four and then both spill. This
    // is why ~700 integer/pointer handlers were never affected, and why the bridge keeps its
    // historical bytes for them.
    expect_places("all integer (6)",
                  sig_of({ I, I, I, I, I, I }),
                  { int_reg(kRdi), int_reg(kRsi), int_reg(kRdx),
                    int_reg(kRcx), int_reg(kR8), int_reg(kR9) },
                  { int_reg(kRcx), int_reg(kRdx), int_reg(kR8), int_reg(kR9),
                    stack(4), stack(5) });

    // All float: also identical, because with no integer argument the SysV SSE counter and the
    // Microsoft positional counter advance together. powf/fmod/atan2 live here -- their ARGUMENTS
    // always arrived correctly, and it was the RETURN that did not survive the checkpoint call.
    expect_places("all float (4)",
                  sig_of({ F, F, F, F }, /*sse_return=*/true),
                  { sse_reg(0), sse_reg(1), sse_reg(2), sse_reg(3) },
                  { sse_reg(0), sse_reg(1), sse_reg(2), sse_reg(3) });

    // ldexpf(float, int): one float in FRONT of an integer is enough. SysV puts the int in rdi (its
    // first integer register); Microsoft puts it in rdx, because the float already consumed
    // position 0. The old bridge moved rdi to rcx.
    expect_places("ldexpf (f,i)",
                  sig_of({ F, I }, /*sse_return=*/true),
                  { sse_reg(0), int_reg(kRdi) },
                  { sse_reg(0), int_reg(kRdx) });

    // sincosf(float, float*, float*): both out-pointers displaced by one position. The old bridge
    // delivered guest rsi (the SECOND pointer) as the first, and left the second reading r8.
    expect_places("sincosf (f,ptr,ptr)",
                  sig_of({ F, I, I }),
                  { sse_reg(0), int_reg(kRdi), int_reg(kRsi) },
                  { sse_reg(0), int_reg(kRdx), int_reg(kR8) });

    // sceFontSetScalePixel(handle, float, float): floats BEHIND an integer, so the SSE index and the
    // position differ by one -- the guest's xmm0/xmm1 must become xmm1/xmm2.
    expect_places("set_scale (ptr,f,f)",
                  sig_of({ I, F, F }),
                  { int_reg(kRdi), sse_reg(0), sse_reg(1) },
                  { int_reg(kRcx), sse_reg(1), sse_reg(2) });

    // Spilling, both ways. SysV packs its overflow (arg 6 is spill slot 0); Microsoft's slot number
    // is the ARGUMENT position, because args 0..3 own home slots they are not passed in. A bridge
    // that copied SysV slot j to Microsoft slot j would be wrong by four.
    expect_places("all integer (8) - spill numbering",
                  sig_of({ I, I, I, I, I, I, I, I }),
                  { int_reg(kRdi), int_reg(kRsi), int_reg(kRdx), int_reg(kRcx),
                    int_reg(kR8), int_reg(kR9), stack(0), stack(1) },
                  { int_reg(kRcx), int_reg(kRdx), int_reg(kR8), int_reg(kR9),
                    stack(4), stack(5), stack(6), stack(7) });

    // Ten floats: SysV has eight SSE registers, Microsoft four. Args 4..7 move from xmm4..xmm7 to
    // the stack, and args 8..9 from the SysV overflow area to different stack slots.
    expect_places("all float (10)",
                  sig_of({ F, F, F, F, F, F, F, F, F, F }),
                  { sse_reg(0), sse_reg(1), sse_reg(2), sse_reg(3), sse_reg(4),
                    sse_reg(5), sse_reg(6), sse_reg(7), stack(0), stack(1) },
                  { sse_reg(0), sse_reg(1), sse_reg(2), sse_reg(3), stack(4),
                    stack(5), stack(6), stack(7), stack(8), stack(9) });

    // A signature with no declaration places nowhere; the bridge falls back to its fixed shuffle.
    if (sig_of({ I, I }).needs_conversion())
        fail("integer signature", "needs_conversion() is true for an all-integer signature");
    if (!sig_of({ I, F }).needs_conversion())
        fail("float signature", "needs_conversion() is false for a signature carrying a float");
    if (!sig_of({ I, I }, /*sse_return=*/true).needs_conversion())
        fail("float return", "needs_conversion() is false for a float-returning signature");
    if (CallSignature{}.needs_conversion())
        fail("undeclared signature", "needs_conversion() is true with nothing declared");
}

// ---------------------------------------------------------------------------------------------
// (2) The emitted bytes, executed.
// ---------------------------------------------------------------------------------------------
#if PROSPER_TEST_CAN_EXECUTE

constexpr uint64_t kClobber = 0x0BADF00DDEADBEEFull;
uint64_t g_checkpoint_calls = 0;
uint64_t g_hook_calls = 0;

// Stands in for dispatch_pending_guest_exception: an ordinary host call, and therefore free to
// destroy every volatile register -- which on Microsoft x64 includes xmm0, where a float-returning
// handler has just left its result. Clobbering deliberately is what makes the save/restore arm real.
PROSPER_TEST_HOST_ABI void test_checkpoint() {
    ++g_checkpoint_calls;
    __asm__ volatile("movq %0, %%xmm0" : : "r"(kClobber) : "xmm0");
    __asm__ volatile("" : : : "rax");
}
PROSPER_TEST_HOST_ABI void test_return_hook() {
    ++g_hook_calls;
    __asm__ volatile("movq %0, %%xmm0" : : "r"(kClobber) : "xmm0");
}

uint8_t* alloc_exec(size_t n) {
#if defined(_WIN32)
    return (uint8_t*)VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : (uint8_t*)p;
#endif
}

// Emit one bridge into executable memory. Returns nullptr (and records a failure) if it does not fit
// the real stub slot, since that is what install_stubs would refuse.
uint8_t* build_bridge(const char* name, uint64_t handler, const CallSignature& sig,
                      uint64_t return_hook = 0) {
    uint8_t staged[kMaxBridgeBytes];
    BridgeParams params;
    params.handler = handler;
    params.checkpoint = (uint64_t)(uintptr_t)&test_checkpoint;
    params.return_hook = return_hook;
    params.signature = sig;
    const size_t n = emit_sysv_to_ms_bridge(staged, params);
    if (n > kMaxBridgeBytes) { fail(name, "bridge overran the staging buffer"); return nullptr; }
    uint8_t* code = alloc_exec(4096);
    if (!code) { fail(name, "could not allocate executable memory"); return nullptr; }
    memcpy(code, staged, n);
    return code;
}

// --- recorded arguments ------------------------------------------------------------------------
struct Recorded {
    uint64_t u[12] = {};
    double   d[12] = {};
    unsigned n = 0;
} g_rec;

// The exact shape from #2955.
PROSPER_TEST_HOST_ABI int32_t h_glyph(void* handle, uint32_t code, void* surface,
                                      float x, float y, void* metrics, void* result) {
    g_rec = {};
    g_rec.u[0] = (uint64_t)(uintptr_t)handle; g_rec.u[1] = code;
    g_rec.u[2] = (uint64_t)(uintptr_t)surface;
    g_rec.d[3] = x; g_rec.d[4] = y;
    g_rec.u[5] = (uint64_t)(uintptr_t)metrics; g_rec.u[6] = (uint64_t)(uintptr_t)result;
    g_rec.n = 7;
    return 0x5A5A;
}
PROSPER_TEST_HOST_ABI int32_t h_set_scale(void* handle, float w, float h) {
    g_rec = {};
    g_rec.u[0] = (uint64_t)(uintptr_t)handle; g_rec.d[1] = w; g_rec.d[2] = h; g_rec.n = 3;
    return 0;
}
PROSPER_TEST_HOST_ABI float h_ldexpf(float x, int32_t n) {
    g_rec = {}; g_rec.d[0] = x; g_rec.u[1] = (uint64_t)(uint32_t)n; g_rec.n = 2;
    return x * (float)(1 << n);
}
PROSPER_TEST_HOST_ABI void h_sincosf(float x, void* s, void* c) {
    g_rec = {};
    g_rec.d[0] = x; g_rec.u[1] = (uint64_t)(uintptr_t)s; g_rec.u[2] = (uint64_t)(uintptr_t)c;
    g_rec.n = 3;
}
PROSPER_TEST_HOST_ABI double h_pow(double a, double b) {
    g_rec = {}; g_rec.d[0] = a; g_rec.d[1] = b; g_rec.n = 2;
    return a * 1000.0 + b;
}
PROSPER_TEST_HOST_ABI float h_strtof(const char* s, char** end) {
    g_rec = {};
    g_rec.u[0] = (uint64_t)(uintptr_t)s; g_rec.u[1] = (uint64_t)(uintptr_t)end; g_rec.n = 2;
    return 1234.5f;
}
// Every placement case at once: integer register, SSE register, integer stack, SSE stack, and a
// guest-spilled word copied into a Microsoft stack slot.
PROSPER_TEST_HOST_ABI uint64_t h_wide(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                      float a4, float a5, uint64_t a6, uint64_t a7,
                                      float a8, uint64_t a9) {
    g_rec = {};
    g_rec.u[0]=a0; g_rec.u[1]=a1; g_rec.u[2]=a2; g_rec.u[3]=a3;
    g_rec.d[4]=a4; g_rec.d[5]=a5; g_rec.u[6]=a6; g_rec.u[7]=a7;
    g_rec.d[8]=a8; g_rec.u[9]=a9; g_rec.n = 10;
    return 0xFEEDFACEull;
}
// The historical integer-only path: guest a0..a9, all integers, no declared signature.
PROSPER_TEST_HOST_ABI uint64_t h_ten(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                                     uint64_t a5, uint64_t a6, uint64_t a7, uint64_t a8,
                                     uint64_t a9) {
    g_rec = {};
    const uint64_t v[10] = { a0,a1,a2,a3,a4,a5,a6,a7,a8,a9 };
    for (unsigned i = 0; i < 10; ++i) g_rec.u[i] = v[i];
    g_rec.n = 10;
    return a0 ^ a9;
}

void expect_u(const char* name, unsigned i, uint64_t got, uint64_t want) {
    if (got == want) return;
    char d[160];
    snprintf(d, sizeof d, "argument %u arrived as 0x%llx, expected 0x%llx", i,
             (unsigned long long)got, (unsigned long long)want);
    fail(name, d);
}
void expect_d(const char* name, unsigned i, double got, double want) {
    if (got == want) return;
    char d[160];
    snprintf(d, sizeof d, "argument %u arrived as %g, expected %g", i, got, want);
    fail(name, d);
}

void check_executed() {
    char sink[8] = {};
    void* const p1 = (void*)sink;
    void* const p2 = (void*)(sink + 1);
    void* const p3 = (void*)(sink + 2);
    void* const p4 = (void*)(sink + 3);

    // --- the #2955 signature -------------------------------------------------------------------
    {
        using Guest = PROSPER_TEST_GUEST_ABI int32_t (*)(void*, uint32_t, void*, float, float,
                                                         void*, void*);
        const CallSignature sig = sig_of({ I, I, I, F, F, I, I });
        uint8_t* code = build_bridge("glyph bridge", (uint64_t)(uintptr_t)&h_glyph, sig);
        if (code) {
            const int32_t r = ((Guest)code)(p1, 0x41424344u, p2, 1.5f, -2.25f, p3, p4);
            const char* n = "executed (ptr,u32,ptr,f,f,ptr,ptr)";
            if (r != 0x5A5A) fail(n, "return value did not survive the bridge");
            expect_u(n, 0, g_rec.u[0], (uint64_t)(uintptr_t)p1);
            expect_u(n, 1, g_rec.u[1], 0x41424344u);
            expect_u(n, 2, g_rec.u[2], (uint64_t)(uintptr_t)p2);
            expect_d(n, 3, g_rec.d[3], 1.5);
            expect_d(n, 4, g_rec.d[4], -2.25);
            expect_u(n, 5, g_rec.u[5], (uint64_t)(uintptr_t)p3);   // the arbitrary-write argument
            expect_u(n, 6, g_rec.u[6], (uint64_t)(uintptr_t)p4);   // and the second one
        }

        // DISCRIMINATOR. The same call through a bridge with NO declared signature -- exactly what
        // master emitted for this handler -- must NOT deliver these arguments. Without this arm the
        // arm above could pass on a host where the two conventions happened to coincide, and the
        // whole test would be asserting nothing.
        uint8_t* blind = build_bridge("glyph bridge (undeclared)",
                                      (uint64_t)(uintptr_t)&h_glyph, CallSignature{});
        if (blind) {
            g_rec = {};
            ((Guest)blind)(p1, 0x41424344u, p2, 1.5f, -2.25f, p3, p4);
            const bool floats_ok = g_rec.d[3] == 1.5 && g_rec.d[4] == -2.25;
            const bool ptrs_ok = g_rec.u[5] == (uint64_t)(uintptr_t)p3 &&
                                 g_rec.u[6] == (uint64_t)(uintptr_t)p4;
            if (floats_ok && ptrs_ok)
                fail("signature-blind bridge",
                     "delivered a float signature correctly, so this test cannot see the defect");
        }
    }

    // --- floats BEHIND an integer (sceFontSetScalePixel) ---------------------------------------
    {
        using Guest = PROSPER_TEST_GUEST_ABI int32_t (*)(void*, float, float);
        uint8_t* code = build_bridge("set_scale bridge", (uint64_t)(uintptr_t)&h_set_scale,
                                     sig_of({ I, F, F }));
        if (code) {
            ((Guest)code)(p1, 24.0f, 32.0f);
            const char* n = "executed (ptr,f,f)";
            expect_u(n, 0, g_rec.u[0], (uint64_t)(uintptr_t)p1);
            expect_d(n, 1, g_rec.d[1], 24.0);
            expect_d(n, 2, g_rec.d[2], 32.0);
        }
    }

    // --- an integer AFTER a float, with a float return ------------------------------------------
    {
        using Guest = PROSPER_TEST_GUEST_ABI float (*)(float, int32_t);
        uint8_t* code = build_bridge("ldexpf bridge", (uint64_t)(uintptr_t)&h_ldexpf,
                                     sig_of({ F, I }, /*sse_return=*/true));
        if (code) {
            const uint64_t before = g_checkpoint_calls;
            const float r = ((Guest)code)(3.5f, 4);
            const char* n = "executed (f,i)->f";
            expect_d(n, 0, g_rec.d[0], 3.5);
            expect_u(n, 1, g_rec.u[1], 4);
            if (r != 56.0f) fail(n, "float return did not survive the checkpoint call");
            if (g_checkpoint_calls != before + 1) fail(n, "the checkpoint was not called");
        }
    }

    // --- two out-pointers behind a float (sincosf), plus a return hook --------------------------
    {
        using Guest = PROSPER_TEST_GUEST_ABI void (*)(float, void*, void*);
        uint8_t* code = build_bridge("sincosf bridge", (uint64_t)(uintptr_t)&h_sincosf,
                                     sig_of({ F, I, I }),
                                     (uint64_t)(uintptr_t)&test_return_hook);
        if (code) {
            const uint64_t before = g_hook_calls;
            ((Guest)code)(0.75f, p2, p3);
            const char* n = "executed (f,ptr,ptr)";
            expect_d(n, 0, g_rec.d[0], 0.75);
            expect_u(n, 1, g_rec.u[1], (uint64_t)(uintptr_t)p2);
            expect_u(n, 2, g_rec.u[2], (uint64_t)(uintptr_t)p3);
            if (g_hook_calls != before + 1) fail(n, "the return hook was not called");
        }
    }

    // --- all-float arguments with a double return ----------------------------------------------
    // The ARGUMENTS here always placed identically under both conventions; the RETURN is the part
    // that a host call after the handler destroys unless the bridge knows to save xmm0.
    {
        using Guest = PROSPER_TEST_GUEST_ABI double (*)(double, double);
        uint8_t* code = build_bridge("pow bridge", (uint64_t)(uintptr_t)&h_pow,
                                     sig_of({ F, F }, /*sse_return=*/true));
        if (code) {
            const double r = ((Guest)code)(2.5, 7.25);
            const char* n = "executed (d,d)->d";
            expect_d(n, 0, g_rec.d[0], 2.5);
            expect_d(n, 1, g_rec.d[1], 7.25);
            if (r != 2507.25) fail(n, "double return did not survive the checkpoint call");
        }
    }

    // --- integer arguments, float return (strtof) -----------------------------------------------
    {
        using Guest = PROSPER_TEST_GUEST_ABI float (*)(const char*, char**);
        uint8_t* code = build_bridge("strtof bridge", (uint64_t)(uintptr_t)&h_strtof,
                                     sig_of({ I, I }, /*sse_return=*/true));
        if (code) {
            char* endp = nullptr;
            const float r = ((Guest)code)("12", &endp);
            const char* n = "executed (ptr,ptr)->f";
            if (r != 1234.5f) fail(n, "float return did not survive the checkpoint call");
        }

        // DISCRIMINATOR for the return-preservation arms: without a declared signature the bridge
        // takes its historical path, which saves rax and nothing else -- so the float return must be
        // destroyed. If it survives, the checkpoint is not clobbering xmm0 and the arms above prove
        // nothing about the save.
        uint8_t* blind = build_bridge("strtof bridge (undeclared)",
                                      (uint64_t)(uintptr_t)&h_strtof, CallSignature{});
        if (blind) {
            char* endp = nullptr;
            const float r = ((Guest)blind)("12", &endp);
            if (r == 1234.5f)
                fail("signature-blind float return",
                     "survived a clobbering checkpoint, so the save arm is untested");
        }
    }

    // --- every placement kind in one call --------------------------------------------------------
    {
        using Guest = PROSPER_TEST_GUEST_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                                          float, float, uint64_t, uint64_t,
                                                          float, uint64_t);
        const CallSignature sig = sig_of({ I, I, I, I, F, F, I, I, F, I });
        uint8_t* code = build_bridge("wide bridge", (uint64_t)(uintptr_t)&h_wide, sig);
        if (code) {
            const uint64_t r = ((Guest)code)(0x1001, 0x1002, 0x1003, 0x1004, 5.5f, 6.5f,
                                             0x1007, 0x1008, 9.5f, 0x100A);
            const char* n = "executed (i,i,i,i,f,f,i,i,f,i)";
            if (r != 0xFEEDFACEull) fail(n, "return value did not survive the bridge");
            expect_u(n, 0, g_rec.u[0], 0x1001); expect_u(n, 1, g_rec.u[1], 0x1002);
            expect_u(n, 2, g_rec.u[2], 0x1003); expect_u(n, 3, g_rec.u[3], 0x1004);
            expect_d(n, 4, g_rec.d[4], 5.5);    expect_d(n, 5, g_rec.d[5], 6.5);
            expect_u(n, 6, g_rec.u[6], 0x1007); expect_u(n, 7, g_rec.u[7], 0x1008);
            expect_d(n, 8, g_rec.d[8], 9.5);    expect_u(n, 9, g_rec.u[9], 0x100A);
        }
    }

    // --- the historical integer path still delivers a0..a9 --------------------------------------
    // This is the no-op half of the change: with no declared signature the bridge emits exactly what
    // it always did, and the ~700 handlers registered by cast must be unaffected.
    {
        using Guest = PROSPER_TEST_GUEST_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                                                          uint64_t, uint64_t, uint64_t, uint64_t,
                                                          uint64_t, uint64_t);
        uint8_t* code = build_bridge("ten-integer bridge", (uint64_t)(uintptr_t)&h_ten,
                                     CallSignature{});
        if (code) {
            const uint64_t r = ((Guest)code)(0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
                                             0x2005, 0x2006, 0x2007, 0x2008, 0x2009);
            const char* n = "executed 10 integers (undeclared)";
            for (unsigned i = 0; i < 10; ++i) expect_u(n, i, g_rec.u[i], 0x2000ull + i);
            if (r != (0x2000ull ^ 0x2009ull)) fail(n, "return value did not survive the bridge");
        }
    }
}
#endif // PROSPER_TEST_CAN_EXECUTE

// ---------------------------------------------------------------------------------------------
// (3) A declared signature must still fit the stub slot, and the undeclared path must be unchanged.
// ---------------------------------------------------------------------------------------------

// loader/linker.hpp's LinkedProgram::stub_size default, which every caller of install_stubs uses.
constexpr size_t kStubSlotBytes = 96;

size_t bridge_size(const CallSignature& sig, bool with_hook) {
    uint8_t staged[kMaxBridgeBytes];
    BridgeParams params;
    params.handler = 0x1122334455667788ull;
    params.checkpoint = 0x99aabbccddeeff00ull;
    params.return_hook = with_hook ? 0x0102030405060708ull : 0;
    params.signature = sig;
    return emit_sysv_to_ms_bridge(staged, params);
}

void check_sizes_and_no_op() {
    // The undeclared path and a DECLARED but integer-only path must emit identical bytes. That is
    // the property that bounds this change: converting a registration to the typed form cannot move
    // a handler that has no float.
    uint8_t a[kMaxBridgeBytes], b[kMaxBridgeBytes];
    BridgeParams pa;
    pa.handler = 0x1122334455667788ull;
    pa.checkpoint = 0x99aabbccddeeff00ull;
    const size_t na = emit_sysv_to_ms_bridge(a, pa);
    BridgeParams pb = pa;
    pb.signature = sig_of({ I, I, I, I, I, I, I });
    const size_t nb = emit_sysv_to_ms_bridge(b, pb);
    if (na != nb || memcmp(a, b, na) != 0)
        fail("integer signature is a no-op", "declared integer-only bytes differ from undeclared");
    if (na > kStubSlotBytes) fail("undeclared bridge size", "the historical path no longer fits");

    // The historical prologue's exact bytes, written out independently of the emitter. A change to
    // the fixed shuffle has to be deliberate.
    static const uint8_t kLegacyPrologue[] = {
        0x50,
        0xFF,0x74,0x24,0x28, 0xFF,0x74,0x24,0x28, 0xFF,0x74,0x24,0x28, 0xFF,0x74,0x24,0x28,
        0x48,0x83,0xEC,0x30,
        0x4C,0x89,0x44,0x24,0x20, 0x4C,0x89,0x4C,0x24,0x28,
        0x49,0x89,0xC9, 0x49,0x89,0xD0, 0x48,0x89,0xF2, 0x48,0x89,0xF9,
    };
    uint8_t prologue[kMaxBridgeBytes];
    const size_t np = emit_legacy_integer_prologue(prologue);
    if (np != sizeof kLegacyPrologue || memcmp(prologue, kLegacyPrologue, np) != 0)
        fail("legacy prologue bytes", "the fixed integer shuffle changed");
    if (memcmp(a, kLegacyPrologue, sizeof kLegacyPrologue) != 0)
        fail("undeclared bridge", "no longer begins with the fixed integer shuffle");

    // Every signature the live registry actually declares has to fit a stub slot, with a return hook
    // and without -- install_stubs refuses one that does not, and on Windows that is a boot failure.
    register_builtin_hle();
    unsigned declared = 0;
    for (const RegisteredFn& r : Hle::registrations()) {
        if (!r.signature.declared) continue;
        ++declared;
        for (bool hook : { false, true }) {
            const size_t n = bridge_size(r.signature, hook);
            if (n <= kStubSlotBytes) continue;
            char d[200];
            snprintf(d, sizeof d, "%s (%s) needs %zu bytes%s, slot is %zu",
                     r.name.c_str(), r.nid.c_str(), n, hook ? " with a return hook" : "",
                     kStubSlotBytes);
            fail("declared signature exceeds the stub slot", d);
        }
    }
    // Vacuous otherwise: a registry with no declared signature would pass the loop by doing nothing,
    // and would also mean the typed registrations are not reaching the registry at all.
    if (declared == 0)
        fail("registry", "no handler declares a floating-point signature, so nothing was measured");
    else
        printf("test_sysv_ms_bridge: %u registered handlers declare a float signature\n", declared);
}

} // namespace

int main() {
    check_tables();
    check_sizes_and_no_op();
#if PROSPER_TEST_CAN_EXECUTE
    check_executed();
#else
    printf("test_sysv_ms_bridge: not x86-64; the emitted bytes were checked but not executed\n");
#endif
    if (g_fail) { fprintf(stderr, "test_sysv_ms_bridge: %d failure(s)\n", g_fail); return 1; }
    printf("test_sysv_ms_bridge: all cases passed\n");
    return 0;
}
