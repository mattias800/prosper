// test_http2 — libSceHttp2 (#2894), the FALSE SUCCESS class of #2081.
//
// The library was entirely unregistered except sceHttp2Init, so every call answered the
// dispatcher's 0 — which is SCE_OK. Two distinct wrongs follow from that single default, and this
// file asserts against BOTH, because an arm that only checks "the NID is registered" or only
// checks "the call returns non-zero" can pass against an implementation that still has the bug:
//
//   * an ID-returning entry point answering 0 hands the guest a valid-looking handle it will
//     carry into later calls, so every id arm below demands a POSITIVE id for a good call and a
//     NEGATIVE (sign-extended) error for a bad one — 0 satisfies neither;
//   * a response getter answering SCE_OK while writing nothing to its out-parameters hands the
//     guest whatever was already in its buffer, which is the NULL that crashed PGA TOUR 2K25's
//     header scanner. Every such arm pre-fills the out-parameters with a sentinel and requires
//     the sentinel to SURVIVE alongside a non-zero return.
#include "hle/dispatch/dispatch.hpp"
#include "hle/net/hle_http2.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

namespace {

// Every NID this library registers, with the name the PS5 3.20 firmware dump gives it. A typo in
// a NID is indistinguishable at run time from leaving the function unregistered -- the guest's
// import simply does not resolve -- so the whole set is checked by name rather than spot-checked.
struct Export { const char* nid; const char* name; };
const Export kExports[] = {
    {"3JCe3lCbQ8A", "sceHttp2Init"},
    {"YiBUtz-pGkc", "sceHttp2Term"},
    {"+wCt7fCijgk", "sceHttp2CreateTemplate"},
    {"pDom5-078DA", "sceHttp2DeleteTemplate"},
    {"mmyOCxQMVYQ", "sceHttp2CreateRequestWithURL"},
    {"c8D9qIjo8EY", "sceHttp2DeleteRequest"},
    {"IZ-qjhRqvjk", "sceHttp2AbortRequest"},
    {"N4UfjvWJsMw", "sceHttp2CreateCookieBox"},
    {"O9ync3F-JVI", "sceHttp2DeleteCookieBox"},
    {"nrPfOE8TQu0", "sceHttp2AddRequestHeader"},
    {"jHdP0CS4ZlA", "sceHttp2RemoveRequestHeader"},
    {"FSAFOzi0FpM", "sceHttp2SetRequestContentLength"},
    {"-HIO4VT87v8", "sceHttp2SetConnectTimeOut"},
    {"izvHhqgDt44", "sceHttp2SetRecvTimeOut"},
    {"XPtW45xiLHk", "sceHttp2SetSendTimeOut"},
    {"ACjtE27aErY", "sceHttp2SetResolveTimeOut"},
    {"Gcjh+CisAZM", "sceHttp2SetResolveRetry"},
    {"n8hMLe31OPA", "sceHttp2SetConnectionWaitTimeOut"},
    {"VYMxTcBqSE0", "sceHttp2SetTimeOut"},
    {"jjFahkBPCYs", "sceHttp2SetAuthEnabled"},
    {"m-OL13q8AI8", "sceHttp2GetAuthEnabled"},
    {"b9AvoIaOuHI", "sceHttp2SetAutoRedirect"},
    {"od5QCZhZSfw", "sceHttp2GetAutoRedirect"},
    {"uRosf8GQbHQ", "sceHttp2SetInflateGZIPEnabled"},
    {"09tk+kIA1Ns", "sceHttp2SetMinSslVersion"},
    {"mPKVhQqh2Es", "sceHttp2SetCookieMaxNum"},
    {"o7+WXe4WadE", "sceHttp2SetCookieMaxNumPerDomain"},
    {"6a0N6GPD7RM", "sceHttp2SetCookieMaxSize"},
    {"EWcwMpbr5F8", "sceHttp2SslEnableOption"},
    {"B37SruheQ5Y", "sceHttp2SslDisableOption"},
    {"BJgi0CH7al4", "sceHttp2SetRedirectCallback"},
    {"YrWX+DhPHQY", "sceHttp2SetSslCallback"},
    {"Wwj6HbB2mOo", "sceHttp2SetAuthInfoCallback"},
    {"zdtXKn9X7no", "sceHttp2SetCookieRecvCallback"},
    {"McYmUpQ3-DY", "sceHttp2SetCookieSendCallback"},
    {"UL4Fviw+IAM", "sceHttp2SetPreSendCallback"},
    {"jrVHsKCXA0g", "sceHttp2SetCookieBox"},
    {"IX23slKvtQI", "sceHttp2GetCookieBox"},
    {"5VlQSzXW-SQ", "sceHttp2CookieFlush"},
    {"WeuDjj5m4YU", "sceHttp2AuthCacheFlush"},
    {"klwUy2Wg+q8", "sceHttp2RedirectCacheFlush"},
    {"rbqZig38AT8", "sceHttp2SendRequest"},
    {"A+NVAFu4eCg", "sceHttp2SendRequestAsync"},
    {"-rdXUi2XW90", "sceHttp2GetAllResponseHeaders"},
    {"9XYJwCf3lEA", "sceHttp2GetStatusCode"},
    {"o0DBQpFE13o", "sceHttp2GetResponseContentLength"},
    {"QygCNNmbGss", "sceHttp2ReadData"},
    {"bGN-6zbo7ms", "sceHttp2ReadDataAsync"},
    {"MOp-AUhdfi8", "sceHttp2WaitAsync"},
    {"flPxnowtvWY", "sceHttp2AddCookie"},
    {"GQFGj0rYX+A", "sceHttp2GetCookie"},
    {"JlFGR4v50Kw", "sceHttp2CookieExport"},
    {"B5ibZI5UlzU", "sceHttp2CookieImport"},
    {"eij7UzkUqK8", "sceHttp2GetCookieStats"},
    {"otUQuZa-mv0", "sceHttp2GetMemoryPoolStats"},
};

HleFn fn(const char* nid) { return Hle::lookup(nid); }

// An id-returning entry point's error must be negative read as int32 AND as int64 -- the guest may
// take either width, and a zero-extended 0x817b1100 is a positive 64-bit number that looks like a
// (very large) handle. This is also the arm that fails against a plain `return 0`.
bool is_id_error(uint64_t ret) {
    return static_cast<int32_t>(ret) < 0 && static_cast<int64_t>(ret) < 0;
}
bool is_positive_id(uint64_t ret) {
    return static_cast<int64_t>(ret) > 0 && static_cast<int64_t>(ret) < 0x10000;
}

} // namespace

int main() {
    std::printf("== test_http2 ==\n");
    register_builtin_hle();

    // ---- registration ---------------------------------------------------------------------
    // Kills: the whole bug. Every one of these answered the dispatcher's 0 before.
    std::printf("-- registration --\n");
    int missing = 0, misnamed = 0;
    for (const auto& e : kExports) {
        if (!fn(e.nid)) { std::printf("  [FAIL] %s (%s) unregistered\n", e.name, e.nid); ++missing; continue; }
        const char* name = Hle::name_of(e.nid);
        if (!name || std::strcmp(name, e.name) != 0) {
            std::printf("  [FAIL] %s registered under the wrong name (%s)\n", e.nid, name ? name : "(null)");
            ++misnamed;
        }
    }
    CHECK(missing == 0, "every libSceHttp2 export this file implements is registered");
    CHECK(misnamed == 0, "every registration carries its PS5 3.20 name");
    if (missing) { std::printf("== FAIL: %d ==\n", fails); return 1; }

    HleFn init = fn("3JCe3lCbQ8A"), term = fn("YiBUtz-pGkc");
    HleFn create_tmpl = fn("+wCt7fCijgk"), del_tmpl = fn("pDom5-078DA");
    HleFn create_req = fn("mmyOCxQMVYQ"), del_req = fn("c8D9qIjo8EY");
    HleFn create_box = fn("N4UfjvWJsMw");
    HleFn add_header = fn("nrPfOE8TQu0");
    HleFn send = fn("rbqZig38AT8");
    HleFn get_headers = fn("-rdXUi2XW90"), get_status = fn("9XYJwCf3lEA");
    HleFn get_length = fn("o0DBQpFE13o"), read_data = fn("QygCNNmbGss");
    HleFn set_auth = fn("jjFahkBPCYs"), get_auth = fn("m-OL13q8AI8");
    HleFn set_box = fn("jrVHsKCXA0g"), get_box = fn("IX23slKvtQI");
    HleFn add_cookie = fn("flPxnowtvWY"), cookie_flush = fn("5VlQSzXW-SQ");

    // ---- ids ------------------------------------------------------------------------------
    std::printf("-- id lifecycle --\n");
    const uint64_t ctx = init(1, 1, 0x58000, 3, 0, 0);
    // Kills: THE BUG for this entry point -- 0 is a library context id the guest carries onward.
    CHECK(is_positive_id(ctx), "sceHttp2Init returns a positive library context id");
    const uint64_t ctx2 = init(1, 1, 0x58000, 3, 0, 0);
    CHECK(is_positive_id(ctx2) && ctx2 != ctx, "a second Init returns a different context id");

    // Kills: accepting any id at all, which is what a table-less implementation does. A context
    // id of 0 is precisely what the OLD dispatcher default handed the guest, so a CreateTemplate
    // that accepts it would take the bug's own output as valid input.
    CHECK(is_id_error(create_tmpl(0, 0, 0, 0, 0, 0)),
          "CreateTemplate on context id 0 answers a negative error, not a template id");
    CHECK(is_id_error(create_tmpl(0x7fff, 0, 0, 0, 0, 0)),
          "CreateTemplate on a never-allocated context id answers a negative error");

    const uint64_t tmpl = create_tmpl(ctx, (uint64_t)(uintptr_t)"prosper-test/1.0", 0, 0, 0, 0);
    CHECK(is_positive_id(tmpl) && tmpl != ctx, "CreateTemplate returns a positive, distinct id");

    const char* url = "https://example.invalid/api/submit?token=abc";
    CHECK(is_id_error(create_req(0, 0, (uint64_t)(uintptr_t)url, 0, 0, 0)),
          "CreateRequestWithURL on template id 0 answers a negative error");
    CHECK(is_id_error(create_req(tmpl, 0, 0, 0, 0, 0)),
          "CreateRequestWithURL with a null URL answers a negative error");
    const uint64_t req = create_req(tmpl, 0, (uint64_t)(uintptr_t)url, 0, 0, 0);
    CHECK(is_positive_id(req) && req != tmpl && req != ctx,
          "CreateRequestWithURL returns a positive, distinct request id");

    // ---- setters record, and reject an id nobody allocated ---------------------------------
    std::printf("-- setters --\n");
    CHECK(add_header(req, (uint64_t)(uintptr_t)"X-Prosper", (uint64_t)(uintptr_t)"1", 0, 0, 0) == 0,
          "AddRequestHeader on a live request succeeds");
    // Kills: "return SCE_OK unconditionally", which is what the dispatcher default did. Note this
    // arm cannot be satisfied by the default, because the default also returns 0 here.
    CHECK(add_header(0, (uint64_t)(uintptr_t)"X-Prosper", (uint64_t)(uintptr_t)"1", 0, 0, 0)
              == http2::kErrorInvalidId,
          "AddRequestHeader on id 0 answers SCE_HTTP2_ERROR_INVALID_ID");

    // A setter/getter pair over the SAME field, both directions. Kills: a getter that writes a
    // constant, a getter that writes nothing, and a setter that answers SCE_OK without recording.
    // The out-parameter sits between two sentinels, so a write of the wrong width is visible.
    int32_t probe[3] = {(int32_t)0xA5A5A5A5, (int32_t)0xA5A5A5A5, (int32_t)0xA5A5A5A5};
    CHECK(set_auth(req, 0, 0, 0, 0, 0) == 0, "SetAuthEnabled(false) succeeds");
    CHECK(get_auth(req, (uint64_t)(uintptr_t)&probe[1], 0, 0, 0, 0) == 0 && probe[1] == 0,
          "GetAuthEnabled reads back the false the setter recorded");
    CHECK(set_auth(req, 1, 0, 0, 0, 0) == 0 &&
              get_auth(req, (uint64_t)(uintptr_t)&probe[1], 0, 0, 0, 0) == 0 && probe[1] == 1,
          "GetAuthEnabled reads back the true the setter recorded");
    CHECK(probe[0] == (int32_t)0xA5A5A5A5 && probe[2] == (int32_t)0xA5A5A5A5,
          "GetAuthEnabled writes exactly its own 32-bit slot, not its neighbours");

    const uint64_t box = create_box(ctx, 0, 0, 0, 0, 0);
    CHECK(is_positive_id(box), "CreateCookieBox returns a positive cookie box id");
    CHECK(set_box(tmpl, box, 0, 0, 0, 0) == 0, "SetCookieBox binds a live box to a template");
    CHECK(set_box(tmpl, 0x7fff, 0, 0, 0, 0) == http2::kErrorInvalidId,
          "SetCookieBox rejects a cookie box id nobody allocated");
    probe[1] = (int32_t)0xA5A5A5A5;
    CHECK(get_box(tmpl, (uint64_t)(uintptr_t)&probe[1], 0, 0, 0, 0) == 0 &&
              probe[1] == (int32_t)box,
          "GetCookieBox reads back the binding SetCookieBox recorded");

    // ---- the network boundary ---------------------------------------------------------------
    std::printf("-- send and response --\n");
    // Before any send there is still no response, and a getter must say so without writing.
    uint64_t out_ptr = 0xDEADBEEFCAFEF00Dull, out_len = 0xDEADBEEFCAFEF00Dull;
    CHECK(get_headers(req, (uint64_t)(uintptr_t)&out_ptr, (uint64_t)(uintptr_t)&out_len, 0, 0, 0) != 0,
          "GetAllResponseHeaders fails before anything was sent");
    CHECK(out_ptr == 0xDEADBEEFCAFEF00Dull && out_len == 0xDEADBEEFCAFEF00Dull,
          "...and writes neither out-parameter");

    const uint64_t sent = send(req, 0, 0, 0, 0, 0);
    // Kills: THE BUG. The dispatcher answered 0 = SCE_OK for a request that was never sent, and
    // that is exactly what walked PGA TOUR 2K25 past its own error handling.
    CHECK(sent != 0, "SendRequest does NOT report success for a request prosper never sent");
    // Kills: an arbitrary non-zero. The offline answer is the libSceNet error an unreachable
    // network produces, which is the family the library itself propagates from its connect site.
    CHECK(sent == http2::kNetErrorNetUnreach,
          "SendRequest reports SCE_NET_ERROR_ENETUNREACH (0x80410133)");
    CHECK((sent & 0x80000000u) != 0, "...and it is an error-shaped value with the top bit set");

    // The four response getters, each with pre-filled out-parameters. This is the crash the issue
    // is about: eboot+0x142d5e0 scans the buffer GetAllResponseHeaders was supposed to hand back.
    out_ptr = out_len = 0xDEADBEEFCAFEF00Dull;
    CHECK(get_headers(req, (uint64_t)(uintptr_t)&out_ptr, (uint64_t)(uintptr_t)&out_len, 0, 0, 0) == sent,
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
    CHECK(read_data(req, (uint64_t)(uintptr_t)body, sizeof(body), 0, 0, 0) != 0,
          "ReadData fails");
    bool body_untouched = true;
    for (unsigned char c : body) body_untouched &= (c == 0x5A);
    CHECK(body_untouched, "...and writes no body bytes");

    // A send on an id nobody allocated is a different failure from "no response".
    CHECK(send(0, 0, 0, 0, 0, 0) == http2::kErrorInvalidId,
          "SendRequest on request id 0 answers INVALID_ID, not the network error");

    // ---- what prosper deliberately does not model ------------------------------------------
    std::printf("-- not modelled / flushes --\n");
    CHECK(add_cookie(ctx, 0, 0, 0, 0, 0) != 0,
          "AddCookie fails rather than claiming a cookie store prosper does not keep");
    CHECK(cookie_flush(ctx, 0, 0, 0, 0, 0) == 0,
          "CookieFlush succeeds: an empty jar really is discardable");
    CHECK(cookie_flush(0, 0, 0, 0, 0, 0) == http2::kErrorInvalidId,
          "CookieFlush still rejects an id nobody allocated");

    // ---- teardown reclaims the whole tree ---------------------------------------------------
    std::printf("-- teardown --\n");
    CHECK(del_req(req, 0, 0, 0, 0, 0) == 0, "DeleteRequest accepts its own request id");
    CHECK(del_req(req, 0, 0, 0, 0, 0) == http2::kErrorInvalidId,
          "...and the id is dead afterwards");
    const uint64_t req2 = create_req(tmpl, 0, (uint64_t)(uintptr_t)url, 0, 0, 0);
    CHECK(is_positive_id(req2), "a fresh request can be created from the same template");
    CHECK(del_tmpl(tmpl, 0, 0, 0, 0, 0) == 0, "DeleteTemplate accepts its own template id");
    CHECK(send(req2, 0, 0, 0, 0, 0) == http2::kErrorInvalidId,
          "DeleteTemplate took its outstanding request with it");

    // The leak control. A loop that terminates contexts WITHOUT ever creating anything under them
    // cannot express the leak it is meant to catch (the vacuity #3295 had to fix in the v1
    // sibling), so this one fills the table from one context first, tears that context down, and
    // then requires a FRESH context to reach the same depth.
    CHECK(term(ctx2, 0, 0, 0, 0, 0) == 0, "Term accepts the second context");
    CHECK(term(ctx, 0, 0, 0, 0, 0) == 0, "Term accepts the first context");
    CHECK(term(ctx, 0, 0, 0, 0, 0) == http2::kErrorInvalidId, "...and the context id is dead");
    CHECK(is_id_error(create_tmpl(ctx, 0, 0, 0, 0, 0)),
          "CreateTemplate on a terminated context answers a negative error");

    int depth[2] = {0, 0};
    for (int pass = 0; pass < 2; ++pass) {
        const uint64_t c = init(1, 1, 0x58000, 3, 0, 0);
        if (!is_positive_id(c)) { std::printf("  [FAIL] Init failed on pass %d\n", pass); ++fails; break; }
        while (true) {
            const uint64_t t = create_tmpl(c, 0, 0, 0, 0, 0);
            if (!is_positive_id(t)) break;
            ++depth[pass];
            // Give each template a request, so a Term that frees only templates still leaks.
            create_req(t, 0, (uint64_t)(uintptr_t)url, 0, 0, 0);
        }
        CHECK(term(c, 0, 0, 0, 0, 0) == 0, "the filled context terminates");
    }
    CHECK(depth[0] > 0, "the table really was filled (the control can express a leak)");
    CHECK(depth[1] == depth[0],
          "a fresh context reaches the same depth: Term reclaimed the whole tree, not just the slot");

    if (fails) std::printf("== FAIL: %d ==\n", fails);
    else std::printf("== PASS ==\n");
    return fails ? 1 : 0;
}
