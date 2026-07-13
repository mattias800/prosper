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

} // namespace prosper::http
