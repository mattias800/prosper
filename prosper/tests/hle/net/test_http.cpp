#include "hle/dispatch/dispatch.hpp"
#include "hle/net/hle_http.hpp"

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

    // ---- sceHttpUriBuild (#2930) --------------------------------------------------------
    // Every assertion below is chosen so the dispatcher's unregistered default CANNOT satisfy
    // it: a return of 0 with the caller's buffer untouched fails the content checks, and the
    // error paths all answer non-zero.
    std::printf("-- sceHttpUriBuild --\n");
    HleFn build = Hle::lookup("5LZA+KPISVA");
    CHECK(build && std::strcmp(Hle::name_of("5LZA+KPISVA"), "sceHttpUriBuild") == 0,
          "sceHttpUriBuild registered by its PS5 NID");
    if (!build) { std::printf("== FAIL: %d ==\n", fails + 1); return 1; }

    // Round-trip the telemetry URL through parse -> build. The default port must NOT reappear.
    std::memset(&uri, 0, sizeof(uri));
    std::memset(pool.data(), 0, pool.size());
    parse((uint64_t)&uri, (uint64_t)url, (uint64_t)pool.data(), (uint64_t)&required, pool.size(), 0);
    std::array<char, 512> built{};
    uint64_t build_need = 0;
    CHECK(build(0, (uint64_t)&build_need, 0, (uint64_t)&uri, http::kUriBuildAll, 0) == 0 &&
          build_need == std::strlen(url) + 1,
          "size query reports the assembled length including the NUL");
    CHECK(build((uint64_t)built.data(), (uint64_t)&build_need, built.size(), (uint64_t)&uri,
                http::kUriBuildAll, 0) == 0 &&
          std::strcmp(built.data(), url) == 0,
          "parse -> build round-trips the URL, and the default 443 stays suppressed");

    // A port that is NOT the scheme default must be emitted.
    uri.port = 8443;
    std::memset(built.data(), 0, built.size());
    build((uint64_t)built.data(), (uint64_t)&build_need, built.size(), (uint64_t)&uri,
          http::kUriBuildAll, 0);
    CHECK(std::strstr(built.data(), "events.backtrace.io:8443/api") != nullptr,
          "a non-default port is emitted as \":8443\" after the host");
    uri.port = 443;

    // Flag gating: ask for scheme + host only and nothing else may appear.
    std::memset(built.data(), 0, built.size());
    CHECK(build((uint64_t)built.data(), (uint64_t)&build_need, built.size(), (uint64_t)&uri,
                http::kUriBuildScheme | http::kUriBuildHostname, 0) == 0 &&
          std::strcmp(built.data(), "https://events.backtrace.io") == 0,
          "component flags gate emission -- path and query are absent when unselected");

    // The library emits element->hostname verbatim, so an IPv6 literal that the PARSER stripped
    // of its brackets is rebuilt without them. Asserted rather than "fixed": matching the
    // shipped library is the contract, and a guest that re-parses the result depends on it.
    std::memset(&uri, 0, sizeof(uri));
    std::memset(pool.data(), 0, pool.size());
    parse((uint64_t)&uri, (uint64_t)complex, (uint64_t)pool.data(), (uint64_t)&required,
          pool.size(), 0);
    std::memset(built.data(), 0, built.size());
    CHECK(build((uint64_t)built.data(), (uint64_t)&build_need, built.size(), (uint64_t)&uri,
                http::kUriBuildAll, 0) == 0 &&
          std::strcmp(built.data(), "http://alice:secret@2001:db8::1:8080/x#frag") == 0,
          "userinfo, explicit port, path and fragment are assembled in the library's order");

    // A guest-supplied element need not have come from our own parser, so build one BY HAND to
    // reach the library's PREFIX-compare default-port lookup: "httpx" takes HTTP's 80 and the
    // port is therefore suppressed. An exact-match lookup would emit ":80" here.
    char hand_scheme[] = "httpx";
    char hand_host[] = "example.test";
    char hand_empty[] = "";
    http::SceHttpUriElement hand{};
    hand.opaque = false;
    hand.scheme = hand_scheme;
    hand.username = hand_empty;
    hand.password = hand_empty;
    hand.hostname = hand_host;
    hand.path = hand_empty;
    hand.query = hand_empty;
    hand.fragment = hand_empty;
    hand.port = 80;
    std::memset(built.data(), 0, built.size());
    CHECK(build((uint64_t)built.data(), (uint64_t)&build_need, built.size(), (uint64_t)&hand,
                http::kUriBuildAll, 0) == 0 &&
          std::strcmp(built.data(), "httpx://example.test") == 0,
          "a prefix-matching scheme takes HTTP's default port, so \":80\" is suppressed");
    // The contrasting arm: without this a lookup that suppressed every port would also pass above.
    char hand_ftp[] = "ftp";
    hand.scheme = hand_ftp;
    std::memset(built.data(), 0, built.size());
    build((uint64_t)built.data(), (uint64_t)&build_need, built.size(), (uint64_t)&hand,
          http::kUriBuildAll, 0);
    CHECK(std::strcmp(built.data(), "ftp://example.test:80") == 0,
          "a scheme with no default port emits \":80\" rather than suppressing it");

    // Error paths. None of these values can come from the dispatcher default.
    CHECK(build((uint64_t)built.data(), (uint64_t)&build_need, built.size(), 0,
                http::kUriBuildAll, 0) == http::kErrorInvalidUrl,
          "a null element returns SCE_HTTP_ERROR_INVALID_URL");
    CHECK(build(0, 0, 0, (uint64_t)&uri, http::kUriBuildAll, 0) == http::kErrorInvalidValue,
          "neither an output buffer nor a size output returns INVALID_VALUE");
    build_need = 0;
    CHECK(build((uint64_t)built.data(), (uint64_t)&build_need, 4, (uint64_t)&uri,
                http::kUriBuildAll, 0) == http::kErrorOutOfMemory,
          "an undersized pool returns SCE_HTTP_ERROR_OUT_OF_MEMORY");
    // The library stores the requirement BEFORE checking the pool, so a caller that was just
    // refused still learns the size it should retry with. A build that reordered those two
    // steps would leave this zero.
    CHECK(build_need == std::strlen("http://alice:secret@2001:db8::1:8080/x#frag") + 1,
          "the required size is reported even on the OUT_OF_MEMORY path");

    // ---- library context lifecycle (#2930) ----------------------------------------------
    std::printf("-- sceHttpTerm / context validation --\n");
    HleFn init = Hle::lookup("A9cVMUtEp4Y");
    HleFn create_tmpl = Hle::lookup("0gYjPTR-6cY");
    HleFn term = Hle::lookup("Ik-KpLTlf7Q");
    CHECK(term && std::strcmp(Hle::name_of("Ik-KpLTlf7Q"), "sceHttpTerm") == 0,
          "sceHttpTerm registered by its PS5 NID");
    if (init && create_tmpl && term) {
        uint64_t ctx = init(0, 0, 0, 0, 0, 0);
        CHECK((int64_t)ctx > 0, "sceHttpInit hands out a positive library context id");
        CHECK((int64_t)create_tmpl(ctx, 0, 0, 0, 0, 0) > 0,
              "a template is created against a live context");
        // An id nobody handed out is not a context. The encoding is the library's own.
        CHECK((int32_t)create_tmpl(0, 0, 0, 0, 0, 0) == (int32_t)http::kErrorInvalidId,
              "sceHttpCreateTemplate rejects context id 0 with INVALID_ID");
        CHECK((int64_t)create_tmpl(999, 0, 0, 0, 0, 0) < 0,
              "sceHttpCreateTemplate rejects an out-of-range context, sign-extended negative");
        CHECK(term(ctx, 0, 0, 0, 0, 0) == 0, "sceHttpTerm releases a live context");
        // Kills a term whose body is just `return 0`: the released id must stop working.
        CHECK((int32_t)create_tmpl(ctx, 0, 0, 0, 0, 0) == (int32_t)http::kErrorInvalidId,
              "a terminated context no longer creates templates");
        CHECK(term(ctx, 0, 0, 0, 0, 0) == http::kErrorInvalidId,
              "terminating an already-released context returns INVALID_ID");
        // Kills a Term that releases the context slot but leaves its templates allocated. Fill
        // the template table from one context, terminate it, and a fresh context must be able to
        // fill it to the same depth -- which it cannot if the templates outlived their owner.
        //
        // The arm this replaces ran 16 init/term cycles that created NO templates, so it could
        // not express the case it claimed to guard: a template leak was structurally invisible
        // to it. Same vacuity class as #3288.
        uint64_t owner_ctx = init(0, 0, 0, 0, 0, 0);
        int first_fill = 0;
        while ((int64_t)create_tmpl(owner_ctx, 0, 0, 0, 0, 0) > 0) first_fill++;
        CHECK(first_fill > 0, "the template table fills from a live context");
        CHECK(term(owner_ctx, 0, 0, 0, 0, 0) == 0,
              "sceHttpTerm releases a context that still owns templates");
        uint64_t next_ctx = init(0, 0, 0, 0, 0, 0);
        int second_fill = 0;
        while ((int64_t)create_tmpl(next_ctx, 0, 0, 0, 0, 0) > 0) second_fill++;
        CHECK(second_fill == first_fill,
              "sceHttpTerm reclaims the templates its context owned");
        CHECK(term(next_ctx, 0, 0, 0, 0, 0) == 0, "the second context releases cleanly");

        // Context slots are reusable too: more init/term cycles than the table is deep. One
        // CHECK site, evaluated once, so the executed count and the source count agree.
        bool ctx_cycles_ok = true;
        for (int i = 0; i < 16 && ctx_cycles_ok; i++) {
            uint64_t again = init(0, 0, 0, 0, 0, 0);
            ctx_cycles_ok = (int64_t)again > 0 && term(again, 0, 0, 0, 0, 0) == 0;
        }
        CHECK(ctx_cycles_ok, "16 init/term cycles do not leak context slots");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
