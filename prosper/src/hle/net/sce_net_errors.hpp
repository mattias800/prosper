#pragma once
// sce_net_errors.hpp — the libSceNet error facility, spelled out once (#3545).
//
// libSceNet reports a failed socket-layer operation as `0x80410100 | <FreeBSD errno>`: the low byte
// IS the errno. That encoding is pinned by libSceHttp2 special-casing 0x80410124 (errno 36,
// EINPROGRESS) as *not* a failure at its sceNetConnect site (#2894), and corroborated by a title's
// own classifier naming 0x8041013d (61, ECONNREFUSED) and 0x80410140 (64, EHOSTDOWN).
//
// It belongs to NEITHER HTTP library, which is why it lives here rather than in hle_http.hpp or
// hle_http2.hpp: both propagate the raw libSceNet error out of a failed connect instead of wrapping
// it in their own facility. Their HTTP-facility constants (v1 `0x8043____`, v2 `0x817b____`) are
// genuinely different and stay in their own headers. The NetCtl sub-facility (`0x804121__`,
// SCE_NET_CTL_ERROR_*) is not this rule either: its low byte is a NetCtl code, not an errno.
//
// The low byte is a FreeBSD errno, never the host's -- the same trap sce_errno.hpp documents for the
// libkernel facility, which is why this takes that header's FreeBsdErrno rather than an int.
//
// Only the errors prosper actually returns get a named constant. Add one here, next to the rule,
// rather than hand-writing a `0x8041____` literal at a call site.
#include "hle/kernel/sce_errno.hpp"

#include <cstdint>

namespace prosper::net {

using hle::FreeBsdErrno;

constexpr uint32_t kNetErrorFacilityBase = 0x80410100u;

// The libSceNet error for a FreeBSD errno. Every FreeBSD errno fits the low byte.
constexpr uint32_t net_error(FreeBsdErrno e) {
    return kNetErrorFacilityBase | (static_cast<uint32_t>(e) & 0xffu);
}

// What prosper answers for a connect it cannot make: a machine with no network has no route, which
// agrees with what NetCtl already reports (DISCONNECTED / NOT_CONNECTED). Used by both HTTP
// libraries' send paths. CONFIDENCE: HIGH on the facility; MED on this exact errno.
constexpr FreeBsdErrno kOfflineConnectErrno = FreeBsdErrno::ENetUnreach;
constexpr uint32_t kNetErrorNetUnreach = net_error(FreeBsdErrno::ENetUnreach);   // 0x80410133

// sceNetPoolCreate's refusal of a null name or a non-positive size. The value 0x80410118 predates
// this header and is unchanged; note it is errno 24, EMFILE, although its call site used to label it
// ENFILE (errno 23, which would be 0x80410117). CONFIDENCE: LOW on which errno the real library
// answers here -- the constant is named for what it encodes, not asserted to be the library's choice.
constexpr uint32_t kNetErrorMFile = net_error(FreeBsdErrno::EMFile);   // 0x80410118

// The encoding pinned against the values the evidence above names, so a change to the rule reddens
// the build rather than quietly re-numbering every caller.
static_assert(net_error(FreeBsdErrno::EInProgress) == 0x80410124u, "EINPROGRESS (libSceHttp2)");
static_assert(net_error(FreeBsdErrno::EConnRefused) == 0x8041013du, "ECONNREFUSED (title classifier)");
static_assert(net_error(FreeBsdErrno::EHostDown) == 0x80410140u, "EHOSTDOWN (title classifier)");
static_assert(kNetErrorNetUnreach == 0x80410133u, "ENETUNREACH");
static_assert(kNetErrorMFile == 0x80410118u, "EMFILE, sceNetPoolCreate's historic value");

}  // namespace prosper::net
