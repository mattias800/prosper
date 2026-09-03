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
