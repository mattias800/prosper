// call_signature.hpp — the declared calling-convention signature of one HLE handler.
//
// The PS5 guest is always System V AMD64. On Linux/macOS the host is too, so a handler is a plain
// C function the guest calls directly and nothing here is consulted. On Windows the host is
// Microsoft x64, and the emitted import stub has to REPLACE the guest's argument placement with the
// host's — which is impossible from the stub alone, because the two conventions disagree about
// *where* an argument lives in a way that depends on the handler's TYPES:
//
//   System V:  integer args in rdi,rsi,rdx,rcx,r8,r9 and SSE args in xmm0..xmm7, counted in TWO
//              INDEPENDENT sequences; whatever does not fit spills to the stack in declaration order.
//   Microsoft: ONE positional sequence. Argument i (i<4) is in {rcx,rdx,r8,r9}[i] if it is an
//              integer and in xmm[i] if it is floating-point; argument i>=4 is at [rsp+8*i], with
//              args 0..3 owning home slots they are not passed in.
//
// So a signature with no floating-point argument places identically under both counters and needs
// only the integer shuffle. One float in the middle desynchronizes the counters and displaces every
// argument AFTER it as well (#2955) — which is why this is a type-level fact the registry has to
// carry, not something the trampoline can infer.
//
// This header is the declaration only, so `hle/dispatch` can store it on every platform (a Linux
// build registers the same signatures, which is what makes the mapping testable off Windows).
// host/abi/sysv_ms_bridge.hpp turns one of these into machine code.
#pragma once
#include <cstdint>
#include <type_traits>

namespace prosper::abi {

// Most arguments a declared signature may carry. The bridge's stack displacements stay in one-byte
// form up to this, and no Sony entry point prosper implements comes close.
inline constexpr unsigned kMaxArgs = 12;

enum class ArgClass : uint8_t { Integer = 0, Sse = 1 };

// A handler's argument/return classes, as the DECLARATION states them.
//
// `declared == false` is the default and means "no signature was recorded": the Windows bridge then
// falls back to its historical fixed integer shuffle, which is correct for every integer/pointer
// handler. Recording a signature is therefore additive — it can only ever affect a handler whose
// own declaration says it takes or returns a float or a double.
struct CallSignature {
    bool     declared   = false;
    uint8_t  count      = 0;   // number of declared arguments
    uint16_t sse_mask   = 0;   // bit i set => argument i is float/double
    bool     sse_return = false;

    constexpr ArgClass arg_class(unsigned i) const {
        return ((sse_mask >> i) & 1u) ? ArgClass::Sse : ArgClass::Integer;
    }
    // Does this signature place differently under Microsoft x64 than under System V — i.e. does the
    // bridge have to do anything the historical integer shuffle does not already do?
    constexpr bool needs_conversion() const { return declared && (sse_mask != 0 || sse_return); }
};

// --- deducing a CallSignature from a real function type -------------------------------------
//
// The signature is derived from the handler's own C++ declaration rather than hand-written beside
// the registration, so it cannot drift away from the function it describes. A handler shape the
// SysV classifier here does not model (an aggregate by value, long double, __int128) is a
// compile-time error rather than a silently wrong placement.

template <class T>
inline constexpr bool is_sse_arg_v =
    std::is_same_v<std::remove_cv_t<T>, float> || std::is_same_v<std::remove_cv_t<T>, double>;

namespace detail {
// A class template with an explicit `void` specialization, NOT a `||`-guarded variable template: a
// variable template's initializer is instantiated whole, so `sizeof(T)` is ill-formed for `void` no
// matter what precedes it in the expression. GCC hides this — it accepts `sizeof(void)` as an
// extension worth 1 — and clang does not, so the void-returning handlers (`sincosf`, `sincos`)
// broke the macOS build and nothing else.
template <class T>
struct AbiScalar : std::bool_constant<
    (std::is_pointer_v<T> || std::is_arithmetic_v<T> || std::is_enum_v<T>) &&
    !std::is_same_v<T, long double> && sizeof(T) <= 8> {};
template <> struct AbiScalar<void> : std::false_type {};
} // namespace detail

template <class T>
inline constexpr bool is_abi_scalar_v = detail::AbiScalar<std::remove_cv_t<T>>::value;

template <class R, class... A>
constexpr CallSignature signature_of(R (*)(A...)) {
    static_assert(sizeof...(A) <= kMaxArgs,
                  "HLE handler declares more arguments than the guest->host bridge models");
    static_assert((is_abi_scalar_v<A> && ...),
                  "HLE handler argument must be a scalar of at most 8 bytes: aggregates passed by "
                  "value, long double and __int128 have SysV classifications the bridge does not model");
    static_assert(std::is_void_v<R> || is_abi_scalar_v<R>,
                  "HLE handler return must be void or a scalar of at most 8 bytes");
    CallSignature s{};
    s.declared = true;
    s.count = static_cast<uint8_t>(sizeof...(A));
    unsigned i = 0;
    ((s.sse_mask |= static_cast<uint16_t>(static_cast<unsigned>(is_sse_arg_v<A>) << i), ++i), ...);
    s.sse_return = is_sse_arg_v<R>;   // strips cv, exactly as the argument classes do
    return s;
}

} // namespace prosper::abi
