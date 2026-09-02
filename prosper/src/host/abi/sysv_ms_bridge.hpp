// sysv_ms_bridge.hpp — the guest(System V AMD64) -> host(Microsoft x64) import bridge.
//
// On Windows the guest cannot call an HLE handler directly: the guest places arguments by the SysV
// convention and the handler reads them by the Microsoft one. Each import stub is therefore a
// TRAMPOLINE that re-places the arguments, calls the handler, runs the pending-guest-exception
// checkpoint, and returns. This header is where those bytes are chosen.
//
// It is compiled on EVERY platform even though only exec_image_win.cpp installs the result, so that
// the mapping it implements can be asserted — and executed — from a Linux test (see
// tests/host/abi/test_sysv_ms_bridge.cpp). The placement rules are a pure function of the handler's
// signature, so they do not need a Windows host to be checked; what a Linux host cannot check is a
// live guest call, and nothing here pretends otherwise.
#pragma once
#include "host/abi/call_signature.hpp"
#include <cstddef>
#include <cstdint>

namespace prosper::abi {

// x86-64 GPR encodings (the ModRM reg field), so an ArgLocation can be handed straight to an encoder.
enum : uint8_t {
    kRax = 0, kRcx = 1, kRdx = 2, kRbx = 3, kRsp = 4, kRbp = 5, kRsi = 6, kRdi = 7,
    kR8  = 8, kR9  = 9, kR10 = 10, kR11 = 11,
};

// Where one argument lives under one convention. `None` is the default and the answer to a query
// past the end of a signature: an out-of-range argument must not come back looking like a plausible
// stack slot, which is the shape a future caller would act on without noticing.
struct ArgLocation {
    enum class Kind : uint8_t { None, IntReg, SseReg, Stack };
    Kind    kind = Kind::None;
    uint8_t reg  = 0;   // IntReg: GPR encoding above. SseReg: xmm ordinal.
    uint8_t slot = 0;   // Stack: 8-byte slot ordinal within that convention's own argument area.

    constexpr bool operator==(const ArgLocation& o) const {
        return kind == o.kind && (kind == Kind::Stack ? slot == o.slot : reg == o.reg);
    }
};

// System V AMD64: integer args consume rdi,rsi,rdx,rcx,r8,r9 and SSE args consume xmm0..xmm7, from
// TWO INDEPENDENT counters; the rest spill in declaration order, `slot` counting only the spilled.
// (The caller's spilled area starts at [entry_rsp + 8], past the return address the call pushed.)
ArgLocation sysv_arg_location(const CallSignature& sig, unsigned arg);

// Microsoft x64: ONE positional counter. Argument i<4 is in {rcx,rdx,r8,r9}[i] when it is an integer
// and in xmm[i] when it is floating-point — the OTHER register file's slot is consumed and wasted,
// which is the whole difference. Argument i>=4 is on the stack at `slot == i`, because args 0..3 own
// home slots they are not passed in.
ArgLocation ms_arg_location(const CallSignature& sig, unsigned arg);

// Everything one import stub needs. `checkpoint` and `return_hook` are host-ABI functions taking no
// arguments; `return_hook` is optional.
struct BridgeParams {
    uint64_t      handler     = 0;
    uint64_t      checkpoint  = 0;
    uint64_t      return_hook = 0;
    CallSignature signature{};
};

// Upper bound on the bytes emit_sysv_to_ms_bridge writes, for a caller's staging buffer. Measured
// worst case over every signature the type system permits (kMaxArgs arguments, every class mask,
// both return classes) is ~170 bytes, so 256 leaves real headroom.
//
// The STUB SLOT is much smaller — 96 bytes (loader/linker.hpp's LinkedProgram::stub_size), which the
// historical integer path exactly fills when a return hook is present. The practical ceiling for a
// float-bearing signature is therefore around eight arguments; the largest one registered today is
// the seven-argument sceFontRenderCharGlyphImage at 85 bytes with a hook. A signature that does not
// fit is refused visibly by install_stubs rather than silently truncated, and
// tests/host/abi/test_sysv_ms_bridge.cpp sweeps every signature the live registry declares — but a
// boot failure naming "generated Windows ABI bridge exceeds stub_size" means the slot, not a bug.
inline constexpr size_t kMaxBridgeBytes = 256;

// Emit the bridge for one import. Returns the number of bytes written (<= kMaxBridgeBytes).
//
// With no declared signature — or one whose placement is identical under both conventions, i.e. no
// floating-point argument and no floating-point return — this emits the historical fixed integer
// shuffle, byte for byte. That is what keeps the change a no-op for every handler that does not
// declare a float.
size_t emit_sysv_to_ms_bridge(uint8_t* out, const BridgeParams& params);

// The historical fixed prologue on its own: guest a0..a9 from SysV integer registers and the guest
// stack into the Microsoft integer registers and outgoing stack. Exposed so a test can pin the
// no-op property by byte comparison rather than by inspection.
size_t emit_legacy_integer_prologue(uint8_t* out);

} // namespace prosper::abi
