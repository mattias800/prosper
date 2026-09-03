// test_guest_varargs — a guest System V variadic call, re-expressed for a Microsoft x64 host (#3246).
//
// #2955 made the import bridge signature-driven, and #3246 is the one shape a signature cannot
// describe: a real C variadic, whose argument list is whatever the format string says at run time.
// Three independent things have to hold, and only the last needs a Windows host:
//
//   (1) READING the guest's list. `sysv_va_arg` walks the System V register save area and overflow
//       area. Checked against a REAL `va_start` frame produced by the compiler — a positive instance
//       built outside the machinery under test, including arguments that spill past both files.
//   (2) CLASSIFYING the arguments. `plan_format` decides, per conversion, which System V file the
//       argument came from. A table, asserted directly, on any platform.
//   (3) WRITING the Microsoft list. The packed slots are consumed by the COMPILER's own Microsoft
//       va_arg (`__builtin_ms_va_list`), which is available on Linux too — so the layout claim is
//       checked against the toolchain rather than against this file's own belief about it. On
//       Windows the real CRT reads the same bytes, and the formatted string is asserted as well.
//
// Plus the two structural facts the wiring depends on: a guest-ABI handler's stub is a bare
// tail-jump, and the printf family really is registered that way.
//
// What NONE of this checks is a live guest calling printf on a Windows host. Nothing here pretends to.
#include "host/abi/guest_varargs.hpp"
#include "host/abi/sysv_ms_bridge.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

using namespace prosper;
using namespace prosper::abi;

#if defined(__x86_64__) || defined(_M_X64)
#define PROSPER_TEST_X86_64 1
#else
#define PROSPER_TEST_X86_64 0
#endif

#if PROSPER_TEST_X86_64
// The producer side must build a SYSTEM V frame. On Linux that is what an ordinary variadic function
// already does; on Windows it needs the same tag the real handlers carry.
#if defined(_WIN32)
#define TEST_GUEST_ABI      __attribute__((sysv_abi))
#define TEST_GUEST_VA_LIST  __builtin_sysv_va_list
#define TEST_GUEST_VA_START __builtin_sysv_va_start
#define TEST_GUEST_VA_END   __builtin_sysv_va_end
#define TEST_MS_VA_LIST     va_list
#else
#define TEST_GUEST_ABI
#define TEST_GUEST_VA_LIST  va_list
#define TEST_GUEST_VA_START va_start
#define TEST_GUEST_VA_END   va_end
#define TEST_MS_VA_LIST     __builtin_ms_va_list
#endif
#endif

namespace {
// PROSPER_GUEST_ABI's whole value is that it is part of the FUNCTION TYPE: that is what makes
// `Hle::register_guest_abi` reject an untagged handler at compile time on Windows, and what makes
// the tag a no-op everywhere else. Both halves are checked by the compiler, here, rather than
// asserted in a PR body — and the Windows half is only ever evaluated by the Windows MinGW job,
// which is the point. If the tag ever silently stopped changing the type, the registration guard
// would still compile and would be closing nothing.
using GuestAbiVariadic = PROSPER_GUEST_ABI int (*)(const char*, ...);
using HostAbiVariadic  = int (*)(const char*, ...);
#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
static_assert(!std::is_same_v<GuestAbiVariadic, HostAbiVariadic>,
              "PROSPER_GUEST_ABI does not change the function type on Windows, so "
              "register_guest_abi's compile-time guard cannot refuse an untagged handler");
#else
static_assert(std::is_same_v<GuestAbiVariadic, HostAbiVariadic>,
              "PROSPER_GUEST_ABI is not empty off Windows, so the change is not the no-op it claims");
#endif

int g_fail = 0;
void fail(const char* what, const std::string& detail) {
    fprintf(stderr, "FAIL %-44s: %s\n", what, detail.c_str());
    ++g_fail;
}
void expect(const char* what, bool ok, const std::string& detail) {
    if (!ok) fail(what, detail);
}

// ---------------------------------------------------------------------------------------------
// (2) The format plan.
// ---------------------------------------------------------------------------------------------

std::string show(const FormatPlan& p) {
    std::string s;
    for (unsigned i = 0; i < p.count; ++i) s += (p.cls[i] == VarargClass::Sse) ? 'f' : 'i';
    if (!p.complete) s += "!";
    return s;
}

void check_plan(const char* fmt, const char* want, FormatGrammar grammar = FormatGrammar::Printf) {
    const FormatPlan plan = plan_format(fmt, grammar);
    const std::string got = show(plan);
    if (got == want) return;
    char d[400];
    snprintf(d, sizeof d, "\"%s\" planned as \"%s\", expected \"%s\"", fmt, got.c_str(), want);
    fail("format plan", d);
}

void check_plans() {
    // The shape #3246 names: one float in the middle, an integer behind it. Under System V the float
    // is in xmm0 and the two strings in rsi/rdx — different FILES, so a bridge that shuffled integer
    // registers positionally delivered the float's slot to the second string.
    check_plan("%s %d %f %s", "iifi");
    check_plan("%d", "i");
    check_plan("%f", "f");
    check_plan("no conversions at all", "");
    check_plan("100%% done", "");
    check_plan("%%%d%%", "i");
    // Length modifiers never change the SLOT count: every one of these is eight bytes under both
    // conventions, and Microsoft's `long double` is a double.
    check_plan("%lld %zu %ju %td %hhd %hd %ld %p %c %s", "iiiiiiiiii");
    check_plan("%lf %le %Lg", "ff!");             // ...except x87 long double, which is refused
    // The accepted set is a SUBSET of what the host CRT consumes an argument for (guest_varargs.cpp,
    // "THE RULE THIS WHOLE FILE OBEYS"). `q` is the BSD long-long modifier: measured under wine, the
    // MinGW CRT prints "%qd" literally and consumes NOTHING, so accepting it would shift every
    // argument behind it. It must refuse, not classify.
    check_plan("%d %qd", "i!");
    check_plan("%qu", "!");
    // `'` is the SUSv2 thousands-grouping flag, and it goes the same way for the same reason: real
    // ucrtbase.dll prints "'d" and consumes nothing. It was KEPT at first on a measurement taken
    // against a DIFFERENT CRT than the one that ships -- see guest_varargs.cpp's subset rule.
    check_plan("%d %'d", "i!");
    check_plan("%'f", "!");
    check_plan("%.2f %10.4e %+g %#a", "ffff");
    // A `*` width or precision consumes an int of its own, ahead of the conversion's own argument.
    check_plan("%*d", "ii");
    check_plan("%.*f", "if");
    check_plan("%*.*f", "iif");
    // Refusals. Each keeps the arguments of the prefix it accepted and drops the rest, so a caller
    // formats less than the guest asked rather than formatting it from the wrong slot.
    check_plan("%d %1$s", "i!");                  // POSIX positional
    check_plan("%d %y", "i!");                    // unknown conversion
    check_plan("%d %", "i!");                     // truncated conversion
    check_plan("%d %*", "i!");                    // ...and one that already consumed a `*` argument
    // scanf: assignment suppression consumes nothing, and every assignment is a pointer.
    check_plan("read %d", "i", FormatGrammar::Scanf);
    check_plan("%d %d", "ii", FormatGrammar::Scanf);
    check_plan("%*d %d", "i", FormatGrammar::Scanf);
    check_plan("%f %s", "ii", FormatGrammar::Scanf);   // a %f target is a float*, an INTEGER slot
    check_plan("%10s %[^\n] %[]] %[^]]", "iiii", FormatGrammar::Scanf);
    check_plan("%d %[abc", "i!", FormatGrammar::Scanf);

    const FormatPlan nul = plan_format(nullptr, FormatGrammar::Printf);
    expect("null format", !nul.complete && nul.count == 0, "a null format was not refused");

    // The prefix a refusal keeps must be the bytes BEFORE the offending conversion, since that is
    // what a caller formats.
    const FormatPlan cut = plan_format("a=%d b=%1$s", FormatGrammar::Printf);
    expect("refusal prefix", cut.modelled_bytes == 7,
           "kept " + std::to_string(cut.modelled_bytes) + " bytes, expected 7 (\"a=%d b=\")");
}

// ---------------------------------------------------------------------------------------------
// (5) The stub a guest-ABI handler gets, and the registry that asks for one.
// ---------------------------------------------------------------------------------------------

void check_stub_and_registry() {
    uint8_t bytes[kMaxBridgeBytes];
    BridgeParams params;
    params.handler = 0x1122334455667788ull;
    params.checkpoint = 0x99aabbccddeeff00ull;
    params.guest_abi = true;
    const size_t n = emit_sysv_to_ms_bridge(bytes, params);
    static const uint8_t kTailJump[] = { 0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
                                         0xFF, 0xE0 };
    expect("guest-abi stub length", n == sizeof kTailJump,
           "emitted " + std::to_string(n) + " bytes, expected 12");
    expect("guest-abi stub bytes", n == sizeof kTailJump && memcmp(bytes, kTailJump, n) == 0,
           "the tail-jump is not `movabs rax, handler ; jmp rax`");

    // DISCRIMINATOR. Without the flag the same params emit the converting bridge, which is many times
    // longer. If they agreed, the arm above would be asserting nothing about the flag.
    BridgeParams converting = params;
    converting.guest_abi = false;
    expect("guest-abi flag moves the emitter", emit_sysv_to_ms_bridge(bytes, converting) != n,
           "the converting bridge emitted the same length as the tail-jump");

    // A tail-jump returns straight to the guest, so it cannot run a return hook afterwards. The
    // combination is unreachable through the registry today, and the emitter must REFUSE it rather
    // than emit a stub that silently drops the hook -- the one outcome nothing downstream notices.
    BridgeParams hooked = params;
    hooked.return_hook = 0x0102030405060708ull;
    expect("guest-abi + return hook is refused",
           emit_sysv_to_ms_bridge(bytes, hooked) > kMaxBridgeBytes,
           "a guest-ABI stub with a return hook was emitted, dropping the hook");

    // The registry really carries the flag for the printf family, and only where it is meant to.
    register_builtin_hle();
    // ...and the registry keeps the flag and a return hook apart, which is what makes the emitter
    // case above a guard against a future edit rather than a live path. NOTE THE PLACEMENT: read
    // before register_builtin_hle() this loop passes vacuously against an empty registry, which is
    // exactly how it was first written here.
    for (const char* name : { "printf", "sprintf", "snprintf", "sprintf_s", "snprintf_s" })
        expect("guest-abi NIDs carry no return hook",
               Hle::return_hook_of(nid_hash(name)) == nullptr,
               std::string(name) + " has a return hook a tail-jump stub could not run");
    unsigned marked = 0;
    for (const char* name : { "printf", "sprintf", "snprintf", "sprintf_s", "snprintf_s" }) {
        if (Hle::guest_abi_nid(nid_hash(name))) { ++marked; continue; }
        fail("guest-abi registration", std::string(name) + " is not registered as a guest-ABI handler");
    }
    expect("guest-abi registry is not vacuous", marked > 0,
           "no printf-family NID is marked, so this arm measured nothing");
    // An ordinary handler must NOT be marked: the flag suppresses argument conversion entirely, so a
    // stray one would hand a Microsoft-ABI handler the guest's raw System V frame.
    expect("memcpy is not guest-abi", !Hle::guest_abi_nid(nid_hash("memcpy")),
           "an ordinary integer handler is marked guest-ABI");
    // sscanf stays on the converting path deliberately (its variadic arguments are all pointers).
    expect("sscanf is not guest-abi", !Hle::guest_abi_nid(nid_hash("sscanf")),
           "sscanf was routed through the guest-ABI path; #3246 records why it should not be");
}

// #3272 -- the CALL side of the guest-ABI boundary. #3246 closed registration at compile time; the
// three accessors below are what close retrieval, and each of these arms is a property of the
// accessor rather than of the registry it reads.
//
// The reason this needs asserting at all is that the failure is INVISIBLE on the platform it is
// written on: PROSPER_GUEST_ABI expands to nothing outside Windows, so a mis-typed pointer compiles
// and behaves identically here, and only the MinGW job dereferences the wild pointer. So these arms
// check the shape of the API surface -- what each accessor refuses -- not the eventual crash.
void check_lookup_accessors() {
    // EXPECTED STDERR: arm (1) deliberately drives the refusal path, so this test prints five
    // "[prosper] Hle::lookup refused ..." lines. They are the fix working, not a failure -- and
    // their absence would mean the refusal went quiet, which is the outcome the loud path exists to
    // prevent. Only the "test_guest_varargs: ..." lines report the verdict.
    register_builtin_hle();
    static const char* const kGuestAbi[] = { "printf", "sprintf", "snprintf",
                                             "sprintf_s", "snprintf_s" };

    // (1) `lookup` refuses every guest-ABI handler. This is the defect: it used to hand back a bare
    //     HleFn, and nothing stopped a caller invoking a System V variadic through it.
    unsigned refused = 0;
    for (const char* name : kGuestAbi) {
        if (Hle::lookup(nid_hash(name)) == nullptr) { ++refused; continue; }
        fail("lookup refuses guest-ABI handlers",
             std::string(name) + ": Hle::lookup returned a bare HleFn for a guest-ABI handler");
    }
    expect("lookup-refusal arm is not vacuous", refused == 5,
           "expected all five printf-family NIDs to be refused, got " + std::to_string(refused));

    // (2) DISCRIMINATOR for (1). Refusing everything would satisfy (1) just as well, and would be a
    //     far worse bug than the one being fixed -- every import in the tree resolves through here.
    expect("lookup still returns ordinary handlers", Hle::lookup(nid_hash("memcpy")) != nullptr,
           "Hle::lookup refused an ordinary host-ABI handler; the refusal is not selective");

    // (3) THE REGRESSION GUARD THAT MATTERS. Both import-stub emitters resolve every slot through
    //     an accessor, and if the guest-ABI NIDs stopped resolving there, `printf` would be emitted
    //     as an UNIMPLEMENTED stub for every title on both platforms -- a silent, total loss of
    //     guest stdout that no other arm here would notice. `lookup_address` is the accessor they
    //     use, and it must answer for a guest-ABI handler exactly as it does for any other.
    unsigned addressable = 0;
    for (const char* name : kGuestAbi) {
        if (Hle::lookup_address(nid_hash(name)) != nullptr) { ++addressable; continue; }
        fail("guest-ABI handlers remain addressable",
             std::string(name) + ": lookup_address returned nullptr, so the import stub for it "
             "would be emitted as unimplemented");
    }
    expect("addressability arm is not vacuous", addressable == 5,
           "expected all five printf-family NIDs to be addressable, got "
           + std::to_string(addressable));
    expect("lookup_address agrees with lookup for ordinary handlers",
           Hle::lookup_address(nid_hash("memcpy"))
               == reinterpret_cast<const void*>(Hle::lookup(nid_hash("memcpy"))),
           "the two accessors disagree about an ordinary handler's address");
    expect("lookup_address reports an unregistered NID as missing",
           Hle::lookup_address("no-such-nid-xyz") == nullptr,
           "lookup_address invented an address for a NID that was never registered");

    // (4) `lookup_guest_abi` hands back the SAME handler, through a type it constructs itself.
    auto printf_fn = Hle::lookup_guest_abi<int, const char*>(nid_hash("printf"));
    expect("lookup_guest_abi resolves a guest-ABI handler", printf_fn != nullptr,
           "lookup_guest_abi returned nullptr for printf");
    expect("lookup_guest_abi returns the registered address",
           reinterpret_cast<const void*>(printf_fn) == Hle::lookup_address(nid_hash("printf")),
           "lookup_guest_abi returned an address the registry does not hold");

    // (5) The category error in the OTHER direction. Asking for the guest convention on an ordinary
    //     host handler must not quietly succeed either -- that mis-types the pointer just as badly,
    //     and it is the mistake a caller makes when they copy the line above to a different NID.
    expect("lookup_guest_abi refuses an ordinary handler",
           Hle::lookup_guest_abi<int, const char*>(nid_hash("memcpy")) == nullptr,
           "lookup_guest_abi handed back a guest-convention pointer to a host-ABI handler");

    // (6) The pointer is USABLE, not merely non-null -- an address that resolves but cannot be
    //     called through would satisfy every arm above. snprintf is the one whose result can be
    //     asserted without writing to stdout, and the arguments deliberately spill past both
    //     register files so the call exercises the overflow area rather than registers alone.
    auto snprintf_fn =
        Hle::lookup_guest_abi<int, char*, size_t, const char*>(nid_hash("snprintf"));
    expect("lookup_guest_abi resolves snprintf", snprintf_fn != nullptr,
           "lookup_guest_abi returned nullptr for snprintf");
    if (snprintf_fn) {
        char buf[128] = {};
        const int n = snprintf_fn(buf, sizeof buf, "%s|%d|%d|%d|%.2f", "s", 1, 2, 3, 4.5);
        expect("a handler called through lookup_guest_abi formats correctly",
               n == 12 && strcmp(buf, "s|1|2|3|4.50") == 0,
               std::string("formatted \"") + buf + "\" (n=" + std::to_string(n)
                   + "), expected \"s|1|2|3|4.50\" (n=12)");
    }
}

#if PROSPER_TEST_X86_64
// ---------------------------------------------------------------------------------------------
// (1) + (3) Executed: a real System V variadic frame, read, packed, and consumed as Microsoft x64.
// ---------------------------------------------------------------------------------------------
//
// The two TEST_GUEST_ABI functions below hold no C++ object and call nothing that owns one. That is
// not tidiness: on Windows they carry `sysv_abi`, and MinGW cannot emit SEH unwind data for such a
// frame, so a `std::string` temporary anywhere inside would make the file fail to ASSEMBLE. They
// record; every comparison happens afterwards, in ordinary host frames.

VarargClass g_cls[kMaxFormatArgs]{};      // classes the reference reader should use
unsigned    g_ref_n = 0;
uint64_t    g_ref_u[kMaxFormatArgs]{};
double      g_ref_d[kMaxFormatArgs]{};
FormatPlan  g_plan{};
uint64_t    g_slots[kMaxFormatArgs]{};
uint64_t    g_blind_slots[kMaxFormatArgs]{};
bool        g_fallback_complete = true;
const char* g_fallback_reject = nullptr;
char        g_fallback_format[512]{};

// Everything the capture is used for happens HERE, while the guest frame is still live — which is
// also why this mirrors production rather than being a convenience. A System V va_list points into
// the caller's outgoing argument area and into the callee's own register save area, so BOTH die when
// the variadic function returns. Deferring the pack until after `guest_call` returned read a frame
// the next ordinary call had already overwritten, and the first version of this test did exactly
// that: arguments 14 and 16-19, the ones that came from the overflow area, arrived as stack litter.
// A PROSPER_GUEST_ABI handler has the same obligation and meets it the same way (hle_libc.cpp).
__attribute__((noinline))
void capture_and_pack(const char* fmt, const SysvVaList& ap) {
    g_plan = plan_format(fmt, FormatGrammar::Printf);
    pack_ms_va_slots(g_plan, ap, g_slots);
    // What the pre-#3246 Windows path effectively did: every argument read from the integer file.
    FormatPlan blind = g_plan;
    for (unsigned i = 0; i < blind.count; ++i) blind.cls[i] = VarargClass::Integer;
    pack_ms_va_slots(blind, ap, g_blind_slots);
    // ...and the truncating fallback, from the same frame.
    MsVarargCall call(fmt, ap, FormatGrammar::Printf);
    g_fallback_complete = call.complete();
    g_fallback_reject = call.reject();
    snprintf(g_fallback_format, sizeof g_fallback_format, "%s", call.format());
}

// The guest side: an ordinary C variadic call, placed by System V. What it captures is exactly what
// h_printf captures.
TEST_GUEST_ABI void guest_call(const char* fmt, ...) {
    TEST_GUEST_VA_LIST ap;
    TEST_GUEST_VA_START(ap, fmt);
    SysvVaList captured;
    memcpy(&captured, &ap, sizeof captured);
    TEST_GUEST_VA_END(ap);
    capture_and_pack(fmt, captured);
}

// The same frame, read by the COMPILER's own System V va_arg instead of by sysv_va_arg — the
// independently produced positive instance that stops arm (1) being checked against itself.
TEST_GUEST_ABI void guest_call_reference(const char* fmt, ...) {
    TEST_GUEST_VA_LIST ap;
    TEST_GUEST_VA_START(ap, fmt);
    for (unsigned i = 0; i < g_ref_n; ++i) {
        if (g_cls[i] == VarargClass::Sse) g_ref_d[i] = __builtin_va_arg(ap, double);
        else                              g_ref_u[i] = __builtin_va_arg(ap, uint64_t);
    }
    TEST_GUEST_VA_END(ap);
}

// Consume a packed Microsoft va_list image with the COMPILER's own Microsoft va_arg. This is what
// makes the layout claim — "a Microsoft va_list is a flat array of 8-byte slots" — checkable without
// a Windows host: were it wrong, GCC's own reader would disagree with this file's writer.
__attribute__((noinline))
void consume_ms(TEST_MS_VA_LIST ap, unsigned n, const char* name) {
    for (unsigned i = 0; i < n; ++i) {
        if (g_cls[i] == VarargClass::Sse) {
            const double got = __builtin_va_arg(ap, double);
            if (got == g_ref_d[i]) continue;
            char d[160];
            snprintf(d, sizeof d, "argument %u arrived as %g, expected %g", i, got, g_ref_d[i]);
            fail(name, d);
        } else {
            const uint64_t got = __builtin_va_arg(ap, uint64_t);
            if (got == g_ref_u[i]) continue;
            char d[160];
            snprintf(d, sizeof d, "argument %u arrived as 0x%llx, expected 0x%llx", i,
                     (unsigned long long)got, (unsigned long long)g_ref_u[i]);
            fail(name, d);
        }
    }
}

// Plan `fmt`, publish its classes for the reference reader, and clear the recorded values.
FormatPlan prepare(const char* fmt) {
    const FormatPlan plan = plan_format(fmt, FormatGrammar::Printf);
    g_ref_n = plan.count;
    for (unsigned i = 0; i < plan.count; ++i) {
        g_cls[i] = plan.cls[i];
        g_ref_u[i] = 0;
        g_ref_d[i] = 0;
    }
    return plan;
}

void check_executed() {
    // --- the shape #3246 names -------------------------------------------------------------------
    // "%s %d %f %s": System V puts the two strings and the int in rsi/rdx/rcx and the double in xmm0
    // — different FILES — so an integer-register shuffle delivers the second string where the double
    // belongs and displaces everything behind it. Distinctive values, so a shifted read cannot match.
    const char* const s1 = "alpha";
    const char* const s2 = "omega";
    {
        const char* fmt = "%s %d %f %s";
        const FormatPlan plan = prepare(fmt);
        expect("plan of %s %d %f %s", plan.count == 4 && plan.complete, "plan is wrong");
        guest_call_reference(fmt, s1, (uint64_t)0x0BADC0DEull, -2.5, s2);
        guest_call(fmt, s1, (uint64_t)0x0BADC0DEull, -2.5, s2);
        expect("reference read %s %d %f %s",
               g_ref_u[0] == (uint64_t)(uintptr_t)s1 && g_ref_u[1] == 0x0BADC0DEull &&
               g_ref_d[2] == -2.5 && g_ref_u[3] == (uint64_t)(uintptr_t)s2,
               "the compiler's own va_arg did not read the values that were passed");
        consume_ms((TEST_MS_VA_LIST)(char*)g_slots, g_plan.count, "executed %s %d %f %s");

        // DISCRIMINATOR. Reading the double's slot as an integer argument is exactly what the
        // pre-#3246 Windows path did, and it must NOT produce the double. Without this arm the one
        // above could pass on a host where the two files happened to coincide. `capture_and_pack`
        // built it from the same frame in the same call, for the lifetime reason recorded there.
        double as_double = 0;
        memcpy(&as_double, &g_blind_slots[2], 8);
        if (as_double == -2.5)
            fail("class-blind plan",
                 "an all-integer plan still delivered the double, so this test cannot see the defect");
    }

    // --- past BOTH System V register files --------------------------------------------------------
    // Ten integer-class arguments (rdi holds the format, so rsi..r9 take five and four spill) and ten
    // doubles (xmm0..xmm7 take eight, two spill). The overflow area then interleaves the two files in
    // declaration order, which no register-only model can express at all.
    {
        const char* fmt = "%d%f%d%f%d%f%d%f%d%f%d%f%d%f%d%f%d%f%d%f";
        const FormatPlan plan = prepare(fmt);
        expect("plan of the 20-argument call", plan.count == 20 && plan.complete, "plan is wrong");
        guest_call_reference(fmt,
            (uint64_t)0x5000, 1.5, (uint64_t)0x5002, 3.5, (uint64_t)0x5004, 5.5,
            (uint64_t)0x5006, 7.5, (uint64_t)0x5008, 9.5, (uint64_t)0x500A, 11.5,
            (uint64_t)0x500C, 13.5, (uint64_t)0x500E, 15.5, (uint64_t)0x5010, 17.5,
            (uint64_t)0x5012, 19.5);
        guest_call(fmt,
            (uint64_t)0x5000, 1.5, (uint64_t)0x5002, 3.5, (uint64_t)0x5004, 5.5,
            (uint64_t)0x5006, 7.5, (uint64_t)0x5008, 9.5, (uint64_t)0x500A, 11.5,
            (uint64_t)0x500C, 13.5, (uint64_t)0x500E, 15.5, (uint64_t)0x5010, 17.5,
            (uint64_t)0x5012, 19.5);
        bool reference_ok = true;
        for (unsigned i = 0; i < 20; ++i)
            reference_ok &= (i % 2) ? (g_ref_d[i] == 0.5 + (double)i)
                                    : (g_ref_u[i] == 0x5000ull + i);
        expect("reference read of the 20-argument call", reference_ok,
               "the compiler's own va_arg did not read the values that were passed");
        consume_ms((TEST_MS_VA_LIST)(char*)g_slots, g_plan.count, "executed 10 ints + 10 doubles");
    }

    // --- a `*` width, which consumes arguments the conversion character does not name --------------
    {
        const char* fmt = "%*.*f|%s";
        const FormatPlan plan = prepare(fmt);
        expect("plan of %*.*f|%s", plan.count == 4 && plan.complete, "plan is wrong");
        guest_call_reference(fmt, (uint64_t)12, (uint64_t)3, 1.25, s2);
        guest_call(fmt, (uint64_t)12, (uint64_t)3, 1.25, s2);
        consume_ms((TEST_MS_VA_LIST)(char*)g_slots, g_plan.count, "executed %*.*f|%s");
    }

    // --- the truncating fallback, executed ---------------------------------------------------------
    // A format carrying something the model refuses still has to be SAFE: MsVarargCall must hand back
    // a prefix whose conversions match the slots it packed.
    {
        const char* fmt = "a=%d b=%f c=%1$s";
        prepare(fmt);
        guest_call(fmt, (uint64_t)7, 6.5, s1);
        expect("truncating fallback refuses", !g_fallback_complete, "a positional format was accepted");
        expect("truncating fallback names a reason", g_fallback_reject != nullptr,
               "a refusal carried no reason");
        // The prefix must be exactly the conversions the packed slots match — two of them here, so
        // the guest sees less than it asked for and never sees a value read from the wrong file.
        expect("truncating fallback keeps a safe prefix",
               strcmp(g_fallback_format, "a=%d b=%f c=") == 0,
               std::string("kept \"") + g_fallback_format + "\"");
    }

    // --- the prefix cap must not slice a conversion in half -----------------------------------------
    // MsVarargCall retains at most 511 bytes of a refused format, and that limit is NOT a conversion
    // boundary. Build one where the cut lands inside a `%...` spec and require the retained string to
    // end on a boundary anyway -- otherwise the CRT is handed a dangling conversion and the packed
    // slot count no longer describes what is being formatted.
    {
        // Getting the cap to bite at all took two tries, and both failures were the arm proving
        // nothing rather than the code being wrong:
        //   * 504 bytes of "ab%d" never reached the 511-byte limit, and the arm passed against a
        //     build with the fix REMOVED;
        //   * 600 bytes of "ab%d" reached the ARGUMENT limit first (kMaxFormatArgs = 64 conversions,
        //     i.e. 258 bytes in), so the byte cap was still never exercised.
        // Hence mostly literal filler with ONE conversion, positioned so the 511-byte cut falls
        // inside it: 510 bytes of 'x', then "%d" occupying offsets 510-511. Cutting at 511 leaves a
        // dangling '%', which is exactly the slice this fix exists to prevent.
        std::string long_fmt(510, 'x');
        long_fmt += "%d";        // straddles the cut
        long_fmt += "z";
        long_fmt += "%1$s";      // ...and something the model must refuse, so the prefix path runs
        prepare(long_fmt.c_str());
        guest_call(long_fmt.c_str(), (uint64_t)1);
        const size_t kept = strlen(g_fallback_format);
        expect("the prefix cap actually bit", kept == 510,
               "kept " + std::to_string(kept) + " bytes; 510 means the cut landed inside the `%d` and "
               "was walked back, anything else means this arm is vacuous");
        expect("prefix cap stays under the limit", kept < 512,
               "kept " + std::to_string(kept) + " bytes");
        expect("prefix cap does not end inside a conversion",
               kept == 0 || g_fallback_format[kept - 1] != '%',
               std::string("prefix ends with a dangling '%'"));
        // The decisive check: re-planning what was kept must agree with it exactly, which is what
        // "the packed count describes the formatted string" means.
        const FormatPlan of_kept = plan_format(g_fallback_format, FormatGrammar::Printf);
        expect("prefix is fully modelled by its own plan",
               of_kept.complete && of_kept.modelled_bytes == kept,
               "the retained prefix does not re-plan cleanly");
    }

#if defined(_WIN32)
    // --- the subset rule, asked of the CRT THIS BUILD ACTUALLY LINKS -------------------------------
    // The rule guest_varargs.cpp obeys is that plan_format must never consume an argument the CRT
    // does not. That was a judgment call backed by a measurement on one toolchain, and the toolchain
    // was the wrong one: the first version of this change accepted `%'d` because a Fedora cross-mingw
    // probe (ANSI stdio, __mingw_vsnprintf) consumed it, while CI and the shipped build are MSYS2
    // UCRT64 (no ANSI stdio, UCRT's own vsnprintf) where it consumes nothing.
    //
    // So stop asserting it and ASK. For each specifier, hand the CRT a call with a trailing sentinel:
    // if the spec consumed its own argument the sentinel prints, and if it consumed nothing the spec
    // ate the sentinel instead. Then require the implication that actually matters —
    //
    //     we consume  =>  the CRT consumes
    //
    // — rather than equality, because refusing something the CRT does support is merely a truncated
    // line, while accepting something it does not is the argument-shifting corruption this file
    // exists to prevent. Whichever CRT a future build resolves to, this arm answers for it.
    {
        static const struct { const char* spec; const char* fmt; bool fp; } kSpecs[] = {
            { "%'d",  "%'d|%d",  false },   // the one that was got wrong
            { "%qd",  "%qd|%d",  false },
            { "%zu",  "%zu|%d",  false },
            { "%jd",  "%jd|%d",  false },
            { "%td",  "%td|%d",  false },
            { "%hhd", "%hhd|%d", false },
            { "%lld", "%lld|%d", false },
            { "%d",   "%d|%d",   false },
            { "%a",   "%a|%d",   true  },
            { "%.2f", "%.2f|%d", true  },
            { "%s",   "%s|%d",   false },
        };
        unsigned probed = 0, crt_rejects = 0;
        for (const auto& c : kSpecs) {
            // Ask the CRT. The sentinel is 77; the leading argument is distinctive either way.
            char rendered[160];
            if (c.fp)                       snprintf(rendered, sizeof rendered, c.fmt, 1.5, 77);
            else if (strcmp(c.spec, "%s") == 0)
                                            snprintf(rendered, sizeof rendered, c.fmt, "str", 77);
            else                            snprintf(rendered, sizeof rendered, c.fmt,
                                                     (long long)1234, 77);
            const bool crt_consumes = strstr(rendered, "|77") != nullptr;

            // Ask the walker.
            char just_spec[16];
            snprintf(just_spec, sizeof just_spec, "%s", c.spec);
            const FormatPlan plan = plan_format(just_spec, FormatGrammar::Printf);
            const bool we_consume = plan.complete && plan.count == 1;

            ++probed;
            if (!crt_consumes) ++crt_rejects;
            char d[220];
            snprintf(d, sizeof d,
                     "%s: the walker consumes an argument but this CRT does not (it rendered "
                     "\"%s\"), so every argument behind it would shift", c.spec, rendered);
            expect("accepted set is a subset of the CRT's", !we_consume || crt_consumes, d);
        }
        // Vacuity guards, and the second one is the load-bearing half. The implication passes trivially
        // for any specifier the CRT consumes, so if the sentinel technique could not DETECT a
        // non-consuming specifier -- if it silently reported "consumes" for everything -- the whole
        // loop would be green and would be measuring nothing. The table therefore has to contain at
        // least one specifier this CRT rejects, and `%qd` is rejected by both the UCRT and the ANSI
        // stdio implementations, so the guard holds whichever one a build resolves to.
        // The literal, NOT `sizeof kSpecs / sizeof *kSpecs` (#3273): comparing the count against the
        // array it was counted from is a tautology, and one in the very file whose subject is arms
        // that cannot fail. Pinned, an edit to kSpecs reddens this instead of silently changing what
        // the arm claims to have probed.
        expect("CRT subset probe ran", probed == 11,
               "the specifier table changed but the pinned count did not -- update both, and check "
               "the table still contains a specifier this CRT REJECTS (the guard below)");
        expect("CRT subset probe can see a refusal", crt_rejects > 0,
               "this CRT reports that it consumes an argument for EVERY specifier probed, including "
               "%qd -- the sentinel detector is broken, so the subset rule was never tested");
    }

    // --- the host CRT, on the platform this exists for ---------------------------------------------
    // The layout is checked above against the compiler's own reader; here the real vsnprintf reads
    // the same bytes and the formatted text itself is asserted.
    {
        const char* fmt = "%s|%d|%.2f|%s|%.2f|%lld";
        prepare(fmt);
        guest_call(fmt, "A", (uint64_t)1, 1.5, "B", 2.5, (uint64_t)7);
        char buf[128];
        const int n = vsnprintf(buf, sizeof buf, fmt, (va_list)(char*)g_slots);
        const char* want = "A|1|1.50|B|2.50|7";
        if (n < 0 || strcmp(buf, want) != 0)
            fail("Windows CRT round trip",
                 std::string("formatted \"") + buf + "\", expected \"" + want + "\"");
    }

    // ...and the same round trip for a format carrying `%'d`, driven through the REAL CRT. This is
    // the case the first version of this change got wrong, so it is pinned end to end rather than
    // only in the plan table: the walker must refuse at the `%'d`, and what the CRT is then handed
    // must be the safe prefix with its own arguments intact — never the full format against a slot
    // array the CRT will read differently.
    {
        const char* fmt = "n=%d t=%'d";
        prepare(fmt);
        guest_call(fmt, (uint64_t)42, (uint64_t)1234567);
        char buf[128];
        const int n = vsnprintf(buf, sizeof buf, g_fallback_format, (va_list)(char*)g_slots);
        if (n < 0 || strcmp(buf, "n=42 t=") != 0)
            fail("Windows CRT round trip with %'d",
                 std::string("formatted \"") + buf + "\", expected \"n=42 t=\"");
        if (g_fallback_complete)
            fail("Windows CRT round trip with %'d", "the walker accepted %'d");
    }
#endif
}
#endif  // PROSPER_TEST_X86_64

} // namespace

int main() {
    check_plans();
    check_stub_and_registry();
    check_lookup_accessors();
#if PROSPER_TEST_X86_64
    check_executed();
#else
    printf("test_guest_varargs: not x86-64; the plan and the stub were checked but not executed\n");
#endif
    if (g_fail) { fprintf(stderr, "test_guest_varargs: %d failure(s)\n", g_fail); return 1; }
    printf("test_guest_varargs: all cases passed\n");
    return 0;
}
