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
    // The third literal the library compares is a string-merge artifact, "TTP", and it maps to 80
    // like "HTTP". Without this arm that branch is dead under test -- deleting it from the source
    // would leave every other check green, which is a poor outcome for the one line N1 was about.
    char hand_ttp[] = "ttp";
    hand.scheme = hand_ttp;
    std::memset(built.data(), 0, built.size());
    build((uint64_t)built.data(), (uint64_t)&build_need, built.size(), (uint64_t)&hand,
          http::kUriBuildAll, 0);
    CHECK(std::strcmp(built.data(), "ttp://example.test") == 0,
          "the library's \"TTP\" literal also defaults to 80, so its \":80\" is suppressed too");

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

        // The arm above proves templates die with a Term; it does NOT prove they die SELECTIVELY.
        // A Term that cleared every slot regardless of owner satisfies it completely, which would
        // make the owner field itself untested -- the same vacuity one level in. So: two live
        // contexts, terminate one, and require the other's template to survive.
        //
        // Counted rather than probed, because there is no getter for a template. Exactly one slot
        // must come back: the depth left for ctx_b is first_fill - 1 if only ctx_a's template was
        // reclaimed, and first_fill if Term cleared the table indiscriminately.
        uint64_t ctx_a = init(0, 0, 0, 0, 0, 0);
        uint64_t ctx_b = init(0, 0, 0, 0, 0, 0);
        uint64_t tmpl_a = create_tmpl(ctx_a, 0, 0, 0, 0, 0);
        uint64_t tmpl_b = create_tmpl(ctx_b, 0, 0, 0, 0, 0);
        CHECK((int64_t)ctx_a > 0 && (int64_t)ctx_b > 0 && ctx_a != ctx_b &&
              (int64_t)tmpl_a > 0 && (int64_t)tmpl_b > 0 && tmpl_a != tmpl_b,
              "two live contexts hold distinct templates");
        CHECK(term(ctx_a, 0, 0, 0, 0, 0) == 0, "one of two live contexts terminates");
        int remaining = 0;
        while ((int64_t)create_tmpl(ctx_b, 0, 0, 0, 0, 0) > 0) remaining++;
        CHECK(remaining == first_fill - 1,
              "sceHttpTerm reclaims ONLY the terminated context's templates");
        term(ctx_b, 0, 0, 0, 0, 0);
    }

    // ---- the request path (#2930) --------------------------------------------------------
    // Everything from sceHttpCreateConnection onwards used to fall to the dispatcher's `return 0`.
    // Two different wrongs come out of that one default, and the arms below are written to kill
    // each separately: an id-returning call answering 0 hands the guest a handle nobody allocated,
    // and a response getter answering SCE_OK with its out-parameters untouched hands the guest
    // back whatever was already in its buffer. So every id arm demands a POSITIVE id for a good
    // call and a NEGATIVE one for a bad call -- 0 satisfies neither -- and every response arm
    // pre-fills its out-parameters with a sentinel and requires the sentinel to SURVIVE next to a
    // non-zero return.
    std::printf("-- request path --\n");
    {
        // init / create_tmpl / term are the handles the context block above already looked up.
        HleFn create_conn = Hle::lookup("Kiwv9r4IZCc");
        HleFn create_conn_url = Hle::lookup("qgxDBjorUxs");
        HleFn del_conn = Hle::lookup("P6A3ytpsiYc");
        HleFn create_req = Hle::lookup("tsGVru3hCe8");
        HleFn create_req2 = Hle::lookup("rGNm+FjIXKk");
        HleFn del_req = Hle::lookup("qe7oZ+v4PWA");
        HleFn add_header = Hle::lookup("EY28T2bkN7k");
        HleFn send = Hle::lookup("1e2BNwI-XzE");
        HleFn read_data = Hle::lookup("P5pdoykPYTk");
        HleFn get_headers = Hle::lookup("aCYPMSUIaP8");
        HleFn get_status = Hle::lookup("0a2TBNfE3BU");
        HleFn get_length = Hle::lookup("yuO2H2Uvnos");
        HleFn last_errno = Hle::lookup("0onIrKx9NIE");

        // Kills: the whole bug for this half of the library -- these NIDs were unregistered, so
        // the dispatcher answered every one of them with 0 = SCE_OK.
        CHECK(create_conn && create_conn_url && del_conn && create_req && create_req2 && del_req &&
              add_header && send && read_data && get_headers && get_status && get_length &&
              last_errno,
              "the libSceHttp request path is registered");
        // A registration under the wrong name is a NID typo, which at run time is indistinguishable
        // from leaving the function unregistered.
        CHECK(Hle::name_of("1e2BNwI-XzE") &&
              std::strcmp(Hle::name_of("1e2BNwI-XzE"), "sceHttpSendRequest") == 0 &&
              Hle::name_of("aCYPMSUIaP8") &&
              std::strcmp(Hle::name_of("aCYPMSUIaP8"), "sceHttpGetAllResponseHeaders") == 0,
              "the send path and the header getter carry their PS5 3.20 names");

        // Guarded only so a missing registration cannot segfault this process before it prints the
        // failure above: every handle dereferenced below is in the condition, and the arm that
        // actually reports the regression is the registration CHECK, not this `if`.
        if (init && create_tmpl && term && create_conn && create_conn_url && del_conn &&
            create_req && create_req2 && del_req && add_header && send && read_data &&
            get_headers && get_status && get_length && last_errno) {
            const uint64_t ctx = init(0, 0, 0, 0, 0, 0);
            const uint64_t tmpl = create_tmpl(ctx, 0, 0, 0, 0, 0);
            const char* host = "example.invalid";
            const char* url = "https://example.invalid/api/submit";

            // Kills: accepting any id at all. Id 0 is exactly what the OLD default handed the
            // guest, so a create path that accepts it takes the bug's own output as valid input.
            CHECK((int32_t)create_conn(0, (uint64_t)(uintptr_t)host, 0, 443, 1, 0) ==
                      (int32_t)http::kErrorInvalidId,
                  "CreateConnection on id 0 answers INVALID_ID, not a connection id");
            CHECK((int64_t)create_conn(999, (uint64_t)(uintptr_t)host, 0, 443, 1, 0) < 0,
                  "CreateConnection on an id nobody allocated answers a negative error");

            const uint64_t conn = create_conn(tmpl, (uint64_t)(uintptr_t)host, 0, 443, 1, 0);
            CHECK((int64_t)conn > 0 && conn != tmpl && conn != ctx,
                  "CreateConnection returns a positive id, distinct from its template and context");
            const uint64_t conn2 = create_conn_url(tmpl, (uint64_t)(uintptr_t)url, 1, 0, 0, 0);
            CHECK((int64_t)conn2 > 0 && conn2 != conn,
                  "CreateConnectionWithURL returns a second, distinct connection id");

            CHECK((int64_t)create_req(0, 0, (uint64_t)(uintptr_t)url, 0, 0, 0) < 0,
                  "CreateRequest on id 0 answers a negative error");
            CHECK((int64_t)create_req(conn, 0, 0, 0, 0, 0) < 0,
                  "CreateRequest with a null path answers a negative error");
            const uint64_t req = create_req(conn, 1, (uint64_t)(uintptr_t)url, 0, 0, 0);
            CHECK((int64_t)req > 0 && req != conn,
                  "CreateRequest returns a positive id, distinct from its connection");
            const uint64_t req2 =
                create_req2(conn, (uint64_t)(uintptr_t)"POST", (uint64_t)(uintptr_t)url, 7, 0, 0);
            CHECK((int64_t)req2 > 0 && req2 != req,
                  "CreateRequest2 (string method) returns a further distinct request id");

            // Kills: "return SCE_OK unconditionally", which is what the old default did for every
            // setter too. This arm cannot be satisfied by that default, because it demands the
            // BAD id be refused.
            CHECK(add_header(req, (uint64_t)(uintptr_t)"X-Prosper", (uint64_t)(uintptr_t)"1", 0,
                             0, 0) == 0,
                  "AddRequestHeader on a live request succeeds");
            CHECK(add_header(0, (uint64_t)(uintptr_t)"X-Prosper", (uint64_t)(uintptr_t)"1", 0,
                             0, 0) == http::kErrorInvalidId,
                  "AddRequestHeader on id 0 answers INVALID_ID");

            // Before any send there is still no response, and the getter must say so in a way that
            // writes nothing.
            uint64_t out_ptr = 0xDEADBEEFCAFEF00Dull, out_len = 0xDEADBEEFCAFEF00Dull;
            CHECK(get_headers(req, (uint64_t)(uintptr_t)&out_ptr, (uint64_t)(uintptr_t)&out_len,
                              0, 0, 0) != 0,
                  "GetAllResponseHeaders fails before anything was sent");
            CHECK(out_ptr == 0xDEADBEEFCAFEF00Dull && out_len == 0xDEADBEEFCAFEF00Dull,
                  "...and writes neither out-parameter");

            const uint64_t sent = send(req, 0, 0, 0, 0, 0);
            // Kills: THE BUG. The dispatcher answered 0 = SCE_OK for a request never sent.
            CHECK(sent != 0,
                  "SendRequest does NOT report success for a request prosper never sent");
            // Kills: an arbitrary non-zero. The offline answer is the libSceNet error an
            // unreachable network produces, which is the family this library propagates.
            CHECK(sent == http::kNetErrorNetUnreach,
                  "SendRequest reports SCE_NET_ERROR_ENETUNREACH (0x80410133)");
            CHECK((sent & 0x80000000u) != 0, "...and it is error-shaped, with the top bit set");
            CHECK(send(0, 0, 0, 0, 0, 0) == http::kErrorInvalidId,
                  "SendRequest on id 0 answers INVALID_ID, not the network error");

            out_ptr = out_len = 0xDEADBEEFCAFEF00Dull;
            CHECK(get_headers(req, (uint64_t)(uintptr_t)&out_ptr, (uint64_t)(uintptr_t)&out_len,
                              0, 0, 0) == sent,
                  "GetAllResponseHeaders reports the same failure the send did");
            CHECK(out_ptr == 0xDEADBEEFCAFEF00Dull && out_len == 0xDEADBEEFCAFEF00Dull,
                  "...and leaves the caller's header pointer and length untouched");

            int32_t status = (int32_t)0xA5A5A5A5;
            CHECK(get_status(req, (uint64_t)(uintptr_t)&status, 0, 0, 0, 0) != 0 &&
                      status == (int32_t)0xA5A5A5A5,
                  "GetStatusCode fails and writes no status code");
            uint64_t length = 0xDEADBEEFCAFEF00Dull;
            CHECK(get_length(req, (uint64_t)(uintptr_t)&length, 0, 0, 0, 0) != 0 &&
                      length == 0xDEADBEEFCAFEF00Dull,
                  "GetResponseContentLength fails and writes no length");
            unsigned char body[16];
            std::memset(body, 0x5A, sizeof(body));
            bool body_untouched = true;
            const uint64_t read_rc = read_data(req, (uint64_t)(uintptr_t)body, sizeof(body), 0, 0, 0);
            for (unsigned char c : body) body_untouched &= (c == 0x5A);
            CHECK(read_rc != 0 && body_untouched, "ReadData fails and writes no body bytes");

            // The one getter that IS answerable from local state: prosper's own send recorded why
            // it failed. Kills a "fail everything" implementation as well as the old silent zero,
            // because it demands the specific errno the send reported AND a fresh request reporting
            // none.
            int32_t probe[3] = {(int32_t)0xA5A5A5A5, (int32_t)0xA5A5A5A5, (int32_t)0xA5A5A5A5};
            CHECK(last_errno(req, (uint64_t)(uintptr_t)&probe[1], 0, 0, 0, 0) == 0 &&
                      probe[1] == http::kNetErrnoNetUnreach,
                  "GetLastErrno reports the errno the failed send recorded");
            CHECK(probe[0] == (int32_t)0xA5A5A5A5 && probe[2] == (int32_t)0xA5A5A5A5,
                  "...writing exactly its own 32-bit slot, not its neighbours");
            probe[1] = (int32_t)0xA5A5A5A5;
            CHECK(last_errno(req2, (uint64_t)(uintptr_t)&probe[1], 0, 0, 0, 0) == 0 &&
                      probe[1] == 0,
                  "a request that was never sent reports no errno");

            // Teardown is a tree, one level deeper than the context/template arm above.
            CHECK(del_req(req, 0, 0, 0, 0, 0) == 0, "DeleteRequest accepts its own request id");
            CHECK(del_req(req, 0, 0, 0, 0, 0) == http::kErrorInvalidId,
                  "...and the request id is dead afterwards");
            CHECK(del_conn(conn, 0, 0, 0, 0, 0) == 0, "DeleteConnection accepts its connection");
            CHECK(send(req2, 0, 0, 0, 0, 0) == http::kErrorInvalidId,
                  "DeleteConnection took its outstanding request with it");

            // The leak control, at request depth. A loop that terminates contexts holding NOTHING
            // cannot express the leak it claims to guard -- the vacuity #3295 had to fix one level
            // up -- so this fills the table with a connection-and-request tree under one context
            // before terminating it, and requires a fresh context to reach the same depth.
            CHECK(term(ctx, 0, 0, 0, 0, 0) == 0, "the context holding the tree terminates");
            int depth[2] = {0, 0};
            bool init_ok = true;
            for (int pass = 0; pass < 2; pass++) {
                const uint64_t c = init(0, 0, 0, 0, 0, 0);
                if ((int64_t)c <= 0) { init_ok = false; break; }
                while (true) {
                    const uint64_t t = create_tmpl(c, 0, 0, 0, 0, 0);
                    if ((int64_t)t <= 0) break;
                    depth[pass]++;
                    const uint64_t cn = create_conn(t, (uint64_t)(uintptr_t)host, 0, 443, 1, 0);
                    if ((int64_t)cn > 0) create_req(cn, 0, (uint64_t)(uintptr_t)url, 0, 0, 0);
                }
                term(c, 0, 0, 0, 0, 0);
            }
            CHECK(init_ok && depth[0] > 0,
                  "the table really was filled with a connection/request tree");
            CHECK(depth[1] == depth[0],
                  "sceHttpTerm reclaims connections and requests too, not just templates");
        }
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
