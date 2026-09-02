// guest_varargs.hpp — a guest System V variadic argument list, and its Microsoft x64 re-expression.
//
// `call_signature.hpp` describes an argument list the TYPE SYSTEM knows. This file exists for the
// one shape it cannot: a real C variadic, whose argument list is whatever the format string says at
// run time (#3246). Two conventions disagree about that list in a way no compile-time signature can
// bridge:
//
//   System V:  the callee's va_start writes a REGISTER SAVE AREA (rdi,rsi,rdx,rcx,r8,r9 then
//              xmm0..xmm7) plus a pointer to the caller's overflow words, and va_arg walks the two
//              files with two independent cursors.
//   Microsoft: there is one file. A variadic callee spills rcx,rdx,r8,r9 into its home area and
//              va_list is a bare `char*` walking 8-byte slots from there into the caller's stack —
//              which is why the convention ALSO requires a floating-point variadic argument to be
//              duplicated into the corresponding integer register: `va_arg(ap, double)` reads the
//              memory slot the integer register was spilled to, and never looks at xmm.
//
// So the conversion is not a register shuffle at all. It is: read the guest's SysV list by the SysV
// rules, learn each argument's class from the format string, and write a flat array of 8-byte slots
// — which IS a Microsoft va_list, and can be handed straight to the host CRT's `v*` function.
//
// Everything here is a pure function of (format string, argument list) and is compiled and tested on
// every platform; only the final `(va_list)MsVarargCall::va_list_image()` cast is Windows-specific,
// because only there is `va_list` the flat array this builds.
#pragma once
#include <cstddef>
#include <cstdint>

namespace prosper::abi {

// The System V AMD64 va_list, exactly as a guest's own `va_start` builds it. The guest is always
// SysV, so this layout is fixed regardless of what the host's `va_list` happens to be — reading it
// as the host's type is the defect this replaces, and on Windows that is an 8-byte `char*` read of a
// 24-byte structure.
struct SysvVaList {
    uint32_t gp_offset;          // next integer argument, as a byte offset into reg_save_area
    uint32_t fp_offset;          // next SSE argument, likewise
    uint64_t overflow_arg_area;  // the caller's spilled words
    uint64_t reg_save_area;      // 6 integer registers then 8 xmm registers
};
static_assert(sizeof(SysvVaList) == 24, "the guest's va_list is 24 bytes on System V AMD64");

// The register save area: 6 integer registers of 8 bytes, then 8 SSE registers of 16. `gp_offset`
// having reached the first and `fp_offset` the second means that file is exhausted and the argument
// comes from the overflow area instead.
inline constexpr uint32_t kSysvGpSaveEnd  = 6 * 8;
inline constexpr uint32_t kSysvRegSaveEnd = kSysvGpSaveEnd + 8 * 16;

enum class VarargClass : uint8_t { Integer = 0, Sse = 1 };

// One argument, advancing `ap` exactly as `va_arg` would. Every scalar a printf-family conversion can
// consume occupies one 8-byte slot under both conventions (Microsoft's `long double` is a double),
// so a class is all the caller needs to know.
uint64_t sysv_va_arg(SysvVaList& ap, VarargClass cls);

// The grammar a format string is read against. They differ in what consumes an argument, not in the
// register files: every scanf conversion takes a POINTER, so a scanf argument list is all-integer and
// places identically under both conventions — which is why `sscanf` is not part of #3246's defect.
enum class FormatGrammar : uint8_t { Printf = 0, Scanf = 1 };

// A format string's argument list, as far as this model can express it. `complete == false` means the
// format contains a conversion the model refuses to guess at (a POSIX positional `%2$d`, an x87
// `long double`, an unknown conversion character): `count` and `cls` then describe only the prefix
// `modelled_bytes` long, and a caller must not format past it — consuming an argument by the wrong
// class desynchronizes every argument after it, which is exactly the silent corruption this file
// exists to remove.
inline constexpr unsigned kMaxFormatArgs = 64;

struct FormatPlan {
    unsigned    count = 0;
    VarargClass cls[kMaxFormatArgs]{};
    size_t      modelled_bytes = 0;
    bool        complete = true;
    const char* reject = nullptr;   // why, when !complete
};

FormatPlan plan_format(const char* fmt, FormatGrammar grammar);

// Fill `slots` (at least `plan.count` entries) with the Microsoft x64 va_list image of the arguments
// `plan` describes, drawn from `ap`. Takes the list BY VALUE: the caller's cursor is left alone, so a
// fallback path can re-read the same arguments.
void pack_ms_va_slots(const FormatPlan& plan, SysvVaList ap, uint64_t* slots);

// One printf- or scanf-family call, re-expressed for a host whose variadic convention is Microsoft
// x64. Holds its own storage and has no destructor, no allocation and no failure mode: a format the
// model cannot express yields a TRUNCATED format string covering only the part it can, which formats
// less than the guest asked for but never formats it from the wrong argument.
//
// The no-destructor property is load-bearing on Windows and not merely tidy — see PROSPER_GUEST_ABI
// in hle/dispatch/dispatch.hpp: a guest-ABI shim cannot carry SEH unwind data, so an object needing
// cleanup must not live in one. Constructing this in an ordinary host frame keeps that decision in
// one place.
class MsVarargCall {
public:
    MsVarargCall(const char* fmt, const SysvVaList& ap, FormatGrammar grammar);

    // The format string to actually pass to the host CRT: `fmt` itself, or the prefix of it this
    // call's arguments are known to match.
    const char* format() const { return format_; }
    // The Microsoft x64 va_list image. On Windows this is castable to `va_list` directly.
    void*       va_list_image() { return slots_; }
    unsigned    count() const { return plan_.count; }
    bool        complete() const { return plan_.complete; }
    const char* reject() const { return plan_.reject; }

private:
    // Longest format prefix retained when the tail cannot be modelled. A format longer than this that
    // also carries an unmodellable conversion loses more of its tail, which is the safe direction.
    static constexpr size_t kMaxPrefixBytes = 512;

    FormatPlan plan_{};
    uint64_t   slots_[kMaxFormatArgs]{};
    char       prefix_[kMaxPrefixBytes]{};
    const char* format_ = nullptr;
};

} // namespace prosper::abi
