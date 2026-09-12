#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::http {

// PS4 and PS5 use the same public URI element ABI. Strings point into the caller-provided pool.
struct SceHttpUriElement {
    bool opaque;
    uint8_t align[7];
    char* scheme;
    char* username;
    char* password;
    char* hostname;
    char* path;
    char* query;
    char* fragment;
    uint16_t port;
    uint8_t reserved[10];
};

static_assert(sizeof(SceHttpUriElement) == 80);
static_assert(offsetof(SceHttpUriElement, scheme) == 8);
static_assert(offsetof(SceHttpUriElement, hostname) == 32);

constexpr uint32_t kErrorOutOfMemory = 0x80431022u;
constexpr uint32_t kErrorInvalidValue = 0x804311feu;
constexpr uint32_t kErrorInvalidUrl = 0x80433060u;

// The library's own id validator answers exactly this for an out-of-range or unallocated
// library context id. CONFIDENCE: HIGH -- read off the validator itself rather than inferred
// from an error-name table (see hle_http.cpp's note on where these came from).
constexpr uint32_t kErrorInvalidId = 0x80431100u;

// Called out of order -- in practice, a response read from a request that was never sent. The most
// frequent error-shaped immediate in the shipped libSceHttp (x114) and the one hle_http.cpp's
// lifecycle note already names BEFORE_INIT. CONFIDENCE: MED on the exact semantic; HIGH that it is
// an error constant of this facility.
constexpr uint32_t kErrorBeforeInit = 0x80431001u;

// What the request path answers offline (#2930). This is NOT an HTTP error: the libSceHttp family
// propagates the raw libSceNet error out of a failed connect, encoded `0x80410100 | <FreeBSD
// errno>`, so the honest answer for a machine with no route is ENETUNREACH -- errno 51 = 0x33.
// It agrees with what prosper already tells the same guest through NetCtl, which reports the link
// DISCONNECTED / NOT_CONNECTED.
//
// CONFIDENCE: HIGH that a non-zero error is required (SCE_OK for a request nobody sent is the
// defect); MED on the exact errno; **LOW that v1 propagates the net error the same way v2 does** --
// that propagation was read off libSceHttp2's connect site and is carried across here, not read off
// libSceHttp itself. The value is deliberately easy to revise: it is produced in one place.
//
// Deliberately duplicated from libSceHttp2's identical constant rather than shared, because the two
// libraries are independent and their HTTP-facility constants are NOT interchangeable (v1 is
// `0x8043____`, v2 is `0x817b____`). Hoisting the libSceNet encoding into one header is #3545.
constexpr uint32_t kNetErrorNetUnreach = 0x80410133u;
constexpr int32_t kNetErrnoNetUnreach = 51;  // what sceHttpGetLastErrno reports for the above

// sceHttpUriBuild component selectors. Each bit gates one SceHttpUriElement field; the caller
// passes the union of the parts it wants emitted.
constexpr uint32_t kUriBuildScheme = 0x01u;
constexpr uint32_t kUriBuildHostname = 0x02u;
constexpr uint32_t kUriBuildPort = 0x04u;
constexpr uint32_t kUriBuildPath = 0x08u;
constexpr uint32_t kUriBuildUsername = 0x10u;
constexpr uint32_t kUriBuildPassword = 0x20u;
constexpr uint32_t kUriBuildQuery = 0x40u;
constexpr uint32_t kUriBuildFragment = 0x80u;
constexpr uint32_t kUriBuildAll = 0xffu;

} // namespace prosper::http
