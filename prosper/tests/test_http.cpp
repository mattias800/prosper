#include "../src/hle/dispatch.hpp"
#include "../src/hle/hle_http.hpp"

#include <array>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_http ==\n");
    register_builtin_hle();
    HleFn parse = Hle::lookup("IWalAn-guFs");
    CHECK(parse && std::strcmp(Hle::name_of("IWalAn-guFs"), "sceHttpUriParse") == 0,
          "sceHttpUriParse registered by its PS5 NID");
    if (!parse) return 1;

    const char* url = "https://events.backtrace.io/api/unique-events/submit?token=abc&universe=nimble";
    uint64_t required = 0;
    CHECK(parse(0, (uint64_t)url, 0, (uint64_t)&required, 0, 0) == 0 && required > 32,
          "size-query mode reports a nonzero caller-pool requirement");

    std::array<char, 512> pool{};
    http::SceHttpUriElement uri{};
    CHECK(parse((uint64_t)&uri, (uint64_t)url, (uint64_t)pool.data(), 0, required, 0) == 0,
          "fill mode succeeds without requiring a second size output");
    CHECK(!uri.opaque && std::strcmp(uri.scheme, "https") == 0 && uri.port == 443,
          "HTTPS scheme and default port parsed");
    CHECK(std::strcmp(uri.hostname, "events.backtrace.io") == 0,
          "Blasphemous 2 telemetry hostname parsed");
    CHECK(std::strcmp(uri.path, "/api/unique-events/submit") == 0 &&
          std::strcmp(uri.query, "?token=abc&universe=nimble") == 0,
          "path and leading-question-mark query parsed");
    CHECK(uri.scheme >= pool.data() && uri.fragment < pool.data() + required,
          "all URI strings live in the caller-provided pool");

    const char* complex = "http://alice:secret@[2001:db8::1]:8080/x#frag";
    required = 0;
    parse(0, (uint64_t)complex, 0, (uint64_t)&required, 0, 0);
    std::memset(&uri, 0, sizeof(uri));
    CHECK(parse((uint64_t)&uri, (uint64_t)complex, (uint64_t)pool.data(),
                (uint64_t)&required, pool.size(), 0) == 0,
          "userinfo, IPv6, explicit-port URI parsed");
    CHECK(std::strcmp(uri.username, "alice") == 0 && std::strcmp(uri.password, "secret") == 0 &&
          std::strcmp(uri.hostname, "2001:db8::1") == 0 && uri.port == 8080 &&
          std::strcmp(uri.fragment, "#frag") == 0,
          "userinfo, bracketless IPv6 output, port, and fragment match ABI");

    uint64_t need = 0;
    parse(0, (uint64_t)url, 0, (uint64_t)&need, 0, 0);
    CHECK(parse((uint64_t)&uri, (uint64_t)url, (uint64_t)pool.data(),
                (uint64_t)&required, need - 1, 0) == http::kErrorOutOfMemory,
          "undersized caller pool returns SCE_HTTP_ERROR_OUT_OF_MEMORY");
    CHECK(parse(0, 0, 0, (uint64_t)&required, 0, 0) == http::kErrorInvalidUrl,
          "null source returns SCE_HTTP_ERROR_INVALID_URL");
    CHECK(parse(0, (uint64_t)url, 0, 0, 0, 0) == http::kErrorInvalidValue,
          "size query without required-size output returns INVALID_VALUE");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
