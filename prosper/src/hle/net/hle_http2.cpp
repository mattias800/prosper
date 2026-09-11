// libSceHttp2 (#2894). Until now the library was entirely unregistered except sceHttp2Init, so
// every one of its 55 exports fell to the dispatcher's `return 0` -- and 0 is SCE_OK. A request
// prosper never sent reported success, and sceHttp2GetAllResponseHeaders reported success while
// writing nothing to its out-parameters, so the guest parsed the pointer it already had. In
// PGA TOUR 2K25 that pointer is NULL and its response-header scanner at eboot+0x142d5e0 dies on
// `movzx edx,BYTE PTR [r11+rcx*1]` with r11 = 0.
//
// The rule this file follows is the one src/hle/net/AGENTS.md states, in both directions:
//
//   * Anything answerable from LOCAL state is implemented, not stubbed. Creating a template, a
//     request or a cookie box really does succeed offline -- it allocates a local object -- so it
//     returns a real id from a real table. Setters really do record local state, so they record it
//     and then answer SCE_OK. A getter that reads back state prosper itself recorded answers it.
//   * Anything that needs a NETWORK answer FAILS, and leaves every out-parameter untouched.
//     There is no PSN behind this, so SCE_OK is not a true answer, and SCE_OK with an unwritten
//     out-parameter is the exact shape that crashes callers.
//
// Nothing here manufactures an HTTP response: there is no synthesised status code, no fabricated
// header block and no body. The guest's own error path runs instead, which is what it is written
// to do -- the blocking title classifies the send result and returns cleanly on any non-zero,
// and it is prosper's SCE_OK that walked it past that check.
//
// WHERE THE CONTRACTS COME FROM. The NID<->name map is the PS5 3.20 firmware library dump
// (libSceHttp2.c, 55 exports + a dummy). Argument shapes for SendRequest, GetAllResponseHeaders
// and GetStatusCode are read off the shipped libSceHttp2.sprx export thunks and recorded on
// #2894; the rest follow the v1 libSceHttp family, whose published contract takes the object id
// as the first argument everywhere. That "id first" property is the only shape assumption any
// validation below depends on. CONFIDENCE: HIGH on it; the per-call argument meanings are marked
// individually where they are weaker.
//
// WHAT IS DELIBERATELY NOT MODELLED. prosper models cookie boxes as real objects with real ids,
// but does not model their CONTENTS. So create/delete/bind/flush are implemented, while the calls
// that must produce or consume cookie DATA (GetCookie, CookieExport, CookieImport, AddCookie,
// GetCookieStats, GetMemoryPoolStats) fail with out-parameters untouched. Answering SCE_OK there
// would claim a store that does not exist, and a later read would contradict it.
#include "hle/net/hle_http2.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                                        uint64_t a3, uint64_t a4, uint64_t a5)

namespace {

constexpr size_t kMaxUrlBytes = 64 * 1024;
constexpr size_t kMaxHeaderBytes = 16 * 1024;

// One unified id space for every object the library hands out. Separate per-kind tables would let
// a template id and a request id collide numerically, and then a validator could not tell a live
// id of the wrong class from a live id of the right one. Ids start at 1, so 0 -- the value the
// dispatcher used to answer -- is never a valid id of any kind.
enum class Kind : uint8_t { None, Ctx, Template, Request, CookieBox };

struct Object {
    Kind kind = Kind::None;
    int32_t owner = 0;  // ctx for a template/cookie box, template for a request; 0 for a ctx

    // Recorded request state. Kept so the setters below are honest: SCE_OK from a setter means
    // the state really is recorded, not that the call was ignored.
    std::string url;
    int32_t method = 0;
    uint64_t content_length = 0;
    std::vector<std::pair<std::string, std::string>> headers;

    int32_t cookie_box = 0;
    int32_t auth_enabled = 1;
    int32_t auto_redirect = 1;
    int32_t inflate_gzip = 0;
    uint32_t connect_timeout_us = 0;
    uint32_t recv_timeout_us = 0;
    uint32_t send_timeout_us = 0;
    uint32_t resolve_timeout_us = 0;
    uint32_t resolve_retry = 0;
    uint32_t connection_wait_timeout_us = 0;
    uint32_t ssl_options = 0;       // union of every SslEnableOption bit, minus the disabled ones
    uint32_t min_ssl_version = 0;
    uint32_t cookie_max_num = 0;
    uint32_t cookie_max_num_per_domain = 0;
    uint32_t cookie_max_size = 0;

    // Callbacks are recorded and never invoked. Nothing calls them because nothing is ever sent,
    // nothing is redirected and no TLS handshake happens -- invoking one would mean manufacturing
    // the event it reports.
    uint64_t cb_redirect = 0, cb_redirect_arg = 0;
    uint64_t cb_ssl = 0, cb_ssl_arg = 0;
    uint64_t cb_auth_info = 0, cb_auth_info_arg = 0;
    uint64_t cb_cookie_recv = 0, cb_cookie_recv_arg = 0;
    uint64_t cb_cookie_send = 0, cb_cookie_send_arg = 0;
    uint64_t cb_pre_send = 0, cb_pre_send_arg = 0;

    bool send_attempted = false;
    uint32_t send_error = 0;
    bool aborted = false;
};

constexpr int kMaxObjects = 128;

std::mutex g_mx;              // guards g_objects in its entirety
Object g_objects[kMaxObjects + 1];  // 1-based; slot 0 is never handed out

// --- id plumbing -------------------------------------------------------------------------
// Two return conventions, and they differ because the contracts do (src/hle/net/AGENTS.md):
//   id_err  -- for an id-returning entry point. Sign-extended, so a guest reading the answer as
//              int32 OR as int64 sees it as negative and never as a handle.
//   err     -- for an SCE_OK-or-error entry point. The zero-extended 32-bit form the library
//              itself returns in eax.
uint64_t err(uint32_t code) { return static_cast<uint64_t>(code); }
uint64_t id_err(uint32_t code) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(code)));
}

int32_t alloc_object(Kind kind, int32_t owner) {  // caller holds g_mx
    for (int i = 1; i <= kMaxObjects; ++i) {
        if (g_objects[i].kind != Kind::None) continue;
        g_objects[i] = Object{};
        g_objects[i].kind = kind;
        g_objects[i].owner = owner;
        return i;
    }
    return 0;
}

Object* live(uint64_t raw, Kind kind) {  // caller holds g_mx
    const int32_t id = static_cast<int32_t>(raw);
    if (id < 1 || id > kMaxObjects) return nullptr;
    Object& obj = g_objects[id];
    return obj.kind == kind ? &obj : nullptr;
}

// Every live object, whatever its class. Used by the setters, deliberately.
//
// The v1 family lets one setter take a template, connection or request id, and which classes each
// v2 setter accepts is not something this machine can check -- so policing the class here would
// risk REJECTING a call hardware accepts, which is the under-report half of the same defect. What
// matters, and what this does enforce, is that the id was actually allocated: 0 and every
// never-created id are rejected. CONFIDENCE: HIGH that id-was-allocated is the right check;
// LOW on any per-class restriction, which is why none is imposed.
Object* live_any(uint64_t raw) {  // caller holds g_mx
    const int32_t id = static_cast<int32_t>(raw);
    if (id < 1 || id > kMaxObjects) return nullptr;
    Object& obj = g_objects[id];
    return obj.kind == Kind::None ? nullptr : &obj;
}

// Release everything owned by `owner`, transitively: a context owns its templates and cookie
// boxes, and each template owns its requests. Releasing only the named slot would leak the tree
// under it, and a title that inits/creates/terms in a loop would then exhaust a table that is
// really empty -- the failure #3295 had to fix in the v1 sibling.
void release_owned(int32_t owner) {  // caller holds g_mx
    for (int i = 1; i <= kMaxObjects; ++i) {
        if (g_objects[i].kind == Kind::None || g_objects[i].owner != owner) continue;
        release_owned(i);
        g_objects[i] = Object{};
    }
}

std::string capture(uint64_t ptr, size_t cap) {
    const char* text = reinterpret_cast<const char*>(ptr);
    if (!text) return {};
    return std::string(text, strnlen(text, cap));
}

// The answer for a response getter. Nothing was received, because nothing was sent: report the
// failure the send reported, or -- if the guest is reading a response before sending at all --
// that it called out of order. Out-parameters are not touched on either path, which is the whole
// point of the issue: a getter that answers SCE_OK while writing nothing hands the caller
// whatever was already in its buffer.
uint64_t response_unavailable(const Object& req) {
    return err(req.send_attempted ? req.send_error : http2::kErrorBeforeInit);
}

// --- library context ------------------------------------------------------------------------

// sceHttp2Init(libnetMemId, libsslCtxId, poolSize, ...) -> library context id (> 0)
//
// Moved here from hle_service.cpp, where it was the library's only registered entry point and
// answered from a free-running counter. It needs a real slot now, because sceHttp2CreateTemplate
// validates the id it is handed against this table.
HLE(h_http2_init) {
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    const int32_t id = alloc_object(Kind::Ctx, 0);
    if (!id) return id_err(http2::kErrorOutOfMemory);
    return static_cast<uint64_t>(id);
}

// sceHttp2Term(libCtxId) -> SCE_OK, releasing the context and everything created under it.
HLE(h_http2_term) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    if (!live(a0, Kind::Ctx)) return err(http2::kErrorInvalidId);
    const int32_t id = static_cast<int32_t>(a0);
    release_owned(id);
    g_objects[id] = Object{};
    return 0;
}

// --- template -------------------------------------------------------------------------------

// sceHttp2CreateTemplate(libCtxId, userAgent, httpVer, isAutoProxyConf) -> template id (> 0)
HLE(h_http2_create_template) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    if (!live(a0, Kind::Ctx)) return id_err(http2::kErrorInvalidId);
    const int32_t id = alloc_object(Kind::Template, static_cast<int32_t>(a0));
    if (!id) return id_err(http2::kErrorOutOfMemory);
    g_objects[id].url = capture(a1, 256);  // user agent, recorded so the object is not empty
    return static_cast<uint64_t>(id);
}

// sceHttp2DeleteTemplate(templateId) -> SCE_OK, taking its requests with it.
HLE(h_http2_delete_template) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    if (!live(a0, Kind::Template)) return err(http2::kErrorInvalidId);
    const int32_t id = static_cast<int32_t>(a0);
    release_owned(id);
    g_objects[id] = Object{};
    return 0;
}

// --- request --------------------------------------------------------------------------------

// sceHttp2CreateRequestWithURL(templateId, method, url, contentLength) -> request id (> 0)
//
// The parent is the TEMPLATE, not a connection object: libSceHttp2 exports no CreateConnection of
// any spelling (the v1 library exports three), so the template is the only thing a request can be
// created from. CONFIDENCE: HIGH on the parent class, MED on the trailing argument meanings.
HLE(h_http2_create_request_with_url) {
    (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* tmpl = live(a0, Kind::Template);
    if (!tmpl) return id_err(http2::kErrorInvalidId);
    if (!a2) return id_err(http2::kErrorInvalidValue);
    const Object parent = *tmpl;  // copied before alloc_object can move the table under us
    const int32_t id = alloc_object(Kind::Request, static_cast<int32_t>(a0));
    if (!id) return id_err(http2::kErrorOutOfMemory);
    Object& req = g_objects[id];
    // A request inherits its template's recorded settings, the way the library's own do.
    req.cookie_box = parent.cookie_box;
    req.auth_enabled = parent.auth_enabled;
    req.auto_redirect = parent.auto_redirect;
    req.inflate_gzip = parent.inflate_gzip;
    req.connect_timeout_us = parent.connect_timeout_us;
    req.recv_timeout_us = parent.recv_timeout_us;
    req.send_timeout_us = parent.send_timeout_us;
    req.resolve_timeout_us = parent.resolve_timeout_us;
    req.resolve_retry = parent.resolve_retry;
    req.connection_wait_timeout_us = parent.connection_wait_timeout_us;
    req.ssl_options = parent.ssl_options;
    req.min_ssl_version = parent.min_ssl_version;
    req.method = static_cast<int32_t>(a1);
    req.url = capture(a2, kMaxUrlBytes);
    req.content_length = a3;
    return static_cast<uint64_t>(id);
}

// sceHttp2DeleteRequest(requestId) -> SCE_OK
HLE(h_http2_delete_request) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    if (!live(a0, Kind::Request)) return err(http2::kErrorInvalidId);
    g_objects[static_cast<int32_t>(a0)] = Object{};
    return 0;
}

// sceHttp2AbortRequest(requestId) -> SCE_OK. Aborting a request that was never in flight is a
// purely local state change, so it genuinely succeeds. CONFIDENCE: MED -- hardware may answer an
// error for a request with nothing to abort, but it has no out-parameter, so a wrong answer here
// cannot hand the guest unwritten memory.
HLE(h_http2_abort_request) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* req = live(a0, Kind::Request);
    if (!req) return err(http2::kErrorInvalidId);
    req->aborted = true;
    return 0;
}

// --- request headers and body length ---------------------------------------------------------

// sceHttp2AddRequestHeader(id, name, value, mode) -> SCE_OK once the header is recorded.
HLE(h_http2_add_request_header) {
    (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* obj = live_any(a0);
    if (!obj) return err(http2::kErrorInvalidId);
    if (!a1) return err(http2::kErrorInvalidValue);
    obj->headers.emplace_back(capture(a1, kMaxHeaderBytes), capture(a2, kMaxHeaderBytes));
    return 0;
}

// sceHttp2RemoveRequestHeader(id, name) -> SCE_OK. Removes every header recorded under that name.
HLE(h_http2_remove_request_header) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* obj = live_any(a0);
    if (!obj) return err(http2::kErrorInvalidId);
    if (!a1) return err(http2::kErrorInvalidValue);
    const std::string name = capture(a1, kMaxHeaderBytes);
    for (size_t i = obj->headers.size(); i-- > 0;)
        if (obj->headers[i].first == name) obj->headers.erase(obj->headers.begin() + static_cast<std::ptrdiff_t>(i));
    return 0;
}

// sceHttp2SetRequestContentLength(id, length) -> SCE_OK
HLE(h_http2_set_request_content_length) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* obj = live_any(a0);
    if (!obj) return err(http2::kErrorInvalidId);
    obj->content_length = a1;
    return 0;
}

// --- the network boundary ---------------------------------------------------------------------

// sceHttp2SendRequest(requestId, body, bodySize) -> the libSceNet error an unreachable network
// produces. NOT SCE_OK: prosper has no network, so there is no request to report as sent, and
// this return is what the blocking title's classifier consumes before it decides whether to read
// a response at all.
HLE(h_http2_send_request) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* req = live(a0, Kind::Request);
    if (!req) return err(http2::kErrorInvalidId);
    req->send_attempted = true;
    req->send_error = http2::kNetErrorNetUnreach;
    return err(req->send_error);
}

// sceHttp2SendRequestAsync(requestId, body, bodySize) -> the same failure, synchronously.
// An async send that is going to fail at connect can fail at submission; what it must not do is
// report acceptance and then never complete, because the guest would wait forever on a request
// that does not exist. sceHttp2WaitAsync below answers the same way for the same reason.
HLE(h_http2_send_request_async) { return h_http2_send_request(a0, a1, a2, a3, a4, a5); }

// Every response getter. There is no response, so each reports the failure and writes NOTHING to
// its out-parameters. sceHttp2GetAllResponseHeaders(requestId, char** out, size_t* len) is the
// one that crashed PGA TOUR 2K25: it used to answer SCE_OK with **out left as it was.
HLE(h_http2_get_all_response_headers) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* req = live(a0, Kind::Request);
    if (!req) return err(http2::kErrorInvalidId);
    return response_unavailable(*req);
}
HLE(h_http2_get_status_code) {  // (requestId, int32* out)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* req = live(a0, Kind::Request);
    if (!req) return err(http2::kErrorInvalidId);
    return response_unavailable(*req);
}
HLE(h_http2_get_response_content_length) {  // (requestId, int* result, uint64* length)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* req = live(a0, Kind::Request);
    if (!req) return err(http2::kErrorInvalidId);
    return response_unavailable(*req);
}
HLE(h_http2_read_data) {  // (requestId, void* buf, size_t size)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* req = live(a0, Kind::Request);
    if (!req) return err(http2::kErrorInvalidId);
    return response_unavailable(*req);
}
HLE(h_http2_read_data_async) { return h_http2_read_data(a0, a1, a2, a3, a4, a5); }
HLE(h_http2_wait_async) {  // (requestId, ...)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* req = live(a0, Kind::Request);
    if (!req) return err(http2::kErrorInvalidId);
    return response_unavailable(*req);
}

// --- setters that record local state ----------------------------------------------------------
// Each validates that the id was really allocated, records the value, and only then answers
// SCE_OK -- "success once the state is actually recorded", not success because there is nothing
// to do. The timeouts are microseconds in the v1 family. CONFIDENCE: MED on the units, HIGH that
// the value is argument 1.
#define HTTP2_SET_U32(fn, field)                                                        \
    HLE(fn) {                                                                           \
        (void)a2; (void)a3; (void)a4; (void)a5;                                         \
        std::lock_guard<std::mutex> lk(g_mx);                                           \
        Object* obj = live_any(a0);                                                     \
        if (!obj) return err(http2::kErrorInvalidId);                                   \
        obj->field = static_cast<uint32_t>(a1);                                         \
        return 0;                                                                       \
    }

HTTP2_SET_U32(h_http2_set_connect_timeout, connect_timeout_us)
HTTP2_SET_U32(h_http2_set_recv_timeout, recv_timeout_us)
HTTP2_SET_U32(h_http2_set_send_timeout, send_timeout_us)
HTTP2_SET_U32(h_http2_set_resolve_timeout, resolve_timeout_us)
HTTP2_SET_U32(h_http2_set_resolve_retry, resolve_retry)
HTTP2_SET_U32(h_http2_set_connection_wait_timeout, connection_wait_timeout_us)
HTTP2_SET_U32(h_http2_set_min_ssl_version, min_ssl_version)
HTTP2_SET_U32(h_http2_set_cookie_max_num, cookie_max_num)
HTTP2_SET_U32(h_http2_set_cookie_max_num_per_domain, cookie_max_num_per_domain)
HTTP2_SET_U32(h_http2_set_cookie_max_size, cookie_max_size)
#undef HTTP2_SET_U32

// sceHttp2SetTimeOut(id, resolveTimeout, connectTimeout, sendTimeout) -- the one call that sets
// several at once. CONFIDENCE: MED on which slot is which; all three are recorded either way.
HLE(h_http2_set_timeout) {
    (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* obj = live_any(a0);
    if (!obj) return err(http2::kErrorInvalidId);
    obj->resolve_timeout_us = static_cast<uint32_t>(a1);
    obj->connect_timeout_us = static_cast<uint32_t>(a2);
    obj->send_timeout_us = static_cast<uint32_t>(a3);
    return 0;
}

#define HTTP2_SET_FLAG(fn, field)                                                       \
    HLE(fn) {                                                                           \
        (void)a2; (void)a3; (void)a4; (void)a5;                                         \
        std::lock_guard<std::mutex> lk(g_mx);                                           \
        Object* obj = live_any(a0);                                                     \
        if (!obj) return err(http2::kErrorInvalidId);                                   \
        obj->field = static_cast<int32_t>(a1) ? 1 : 0;                                  \
        return 0;                                                                       \
    }
HTTP2_SET_FLAG(h_http2_set_auth_enabled, auth_enabled)
HTTP2_SET_FLAG(h_http2_set_auto_redirect, auto_redirect)
HTTP2_SET_FLAG(h_http2_set_inflate_gzip_enabled, inflate_gzip)
#undef HTTP2_SET_FLAG

// The two flag getters read back exactly what the setter above recorded, so they are answered
// locally rather than failed -- refusing state prosper holds would be the under-report half of
// this issue. The out-parameter is written as an int32: the v1 family declares these `SceBool*`,
// which is a 32-bit int, not a byte. CONFIDENCE: MED on the width; it is the one place in this
// file where being wrong writes memory rather than merely returning a wrong value.
#define HTTP2_GET_FLAG(fn, field)                                                       \
    HLE(fn) {                                                                           \
        (void)a2; (void)a3; (void)a4; (void)a5;                                         \
        std::lock_guard<std::mutex> lk(g_mx);                                           \
        Object* obj = live_any(a0);                                                     \
        if (!obj) return err(http2::kErrorInvalidId);                                   \
        if (!a1) return err(http2::kErrorInvalidValue);                                 \
        *reinterpret_cast<int32_t*>(a1) = obj->field;                                   \
        return 0;                                                                       \
    }
HTTP2_GET_FLAG(h_http2_get_auth_enabled, auth_enabled)
HTTP2_GET_FLAG(h_http2_get_auto_redirect, auto_redirect)
#undef HTTP2_GET_FLAG

// sceHttp2SslEnableOption / sceHttp2SslDisableOption(id, optionFlag) -> SCE_OK. The recorded set
// is the union of the enables minus the disables, so the two calls really do compose.
HLE(h_http2_ssl_enable_option) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* obj = live_any(a0);
    if (!obj) return err(http2::kErrorInvalidId);
    obj->ssl_options |= static_cast<uint32_t>(a1);
    return 0;
}
HLE(h_http2_ssl_disable_option) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* obj = live_any(a0);
    if (!obj) return err(http2::kErrorInvalidId);
    obj->ssl_options &= ~static_cast<uint32_t>(a1);
    return 0;
}

// Callback registration: (id, callback, userArg). Both halves are recorded; neither is ever
// called, because nothing happens offline that any of them reports.
#define HTTP2_SET_CB(fn, field)                                                         \
    HLE(fn) {                                                                           \
        (void)a3; (void)a4; (void)a5;                                                   \
        std::lock_guard<std::mutex> lk(g_mx);                                           \
        Object* obj = live_any(a0);                                                     \
        if (!obj) return err(http2::kErrorInvalidId);                                   \
        obj->field = a1;                                                                \
        obj->field##_arg = a2;                                                          \
        return 0;                                                                       \
    }
HTTP2_SET_CB(h_http2_set_redirect_callback, cb_redirect)
HTTP2_SET_CB(h_http2_set_ssl_callback, cb_ssl)
HTTP2_SET_CB(h_http2_set_auth_info_callback, cb_auth_info)
HTTP2_SET_CB(h_http2_set_cookie_recv_callback, cb_cookie_recv)
HTTP2_SET_CB(h_http2_set_cookie_send_callback, cb_cookie_send)
HTTP2_SET_CB(h_http2_set_pre_send_callback, cb_pre_send)
#undef HTTP2_SET_CB

// --- cookie boxes ------------------------------------------------------------------------------

// sceHttp2CreateCookieBox(libCtxId, ...) -> cookie box id (> 0)
HLE(h_http2_create_cookie_box) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    if (!live(a0, Kind::Ctx)) return id_err(http2::kErrorInvalidId);
    const int32_t id = alloc_object(Kind::CookieBox, static_cast<int32_t>(a0));
    if (!id) return id_err(http2::kErrorOutOfMemory);
    return static_cast<uint64_t>(id);
}

HLE(h_http2_delete_cookie_box) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    if (!live(a0, Kind::CookieBox)) return err(http2::kErrorInvalidId);
    const int32_t id = static_cast<int32_t>(a0);
    // Anything still pointing at this box loses the binding rather than keeping a dangling id.
    for (int i = 1; i <= kMaxObjects; ++i)
        if (g_objects[i].cookie_box == id) g_objects[i].cookie_box = 0;
    g_objects[id] = Object{};
    return 0;
}

// sceHttp2SetCookieBox(id, cookieBoxId) -> SCE_OK once the binding is recorded.
HLE(h_http2_set_cookie_box) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* obj = live_any(a0);
    if (!obj) return err(http2::kErrorInvalidId);
    if (a1 != 0 && !live(a1, Kind::CookieBox)) return err(http2::kErrorInvalidId);
    obj->cookie_box = static_cast<int32_t>(a1);
    return 0;
}

// sceHttp2GetCookieBox(id, int32* out) -> the binding prosper recorded. Local state, so it is
// answered rather than failed. CONFIDENCE: MED on the out-parameter width (int, per the v1 ids).
HLE(h_http2_get_cookie_box) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    Object* obj = live_any(a0);
    if (!obj) return err(http2::kErrorInvalidId);
    if (!a1) return err(http2::kErrorInvalidValue);
    *reinterpret_cast<int32_t*>(a1) = obj->cookie_box;
    return 0;
}

// Flushes. The jar, the auth cache and the redirect cache are all empty and stay empty offline,
// so discarding their contents is a no-op that genuinely succeeded. None has an out-parameter.
HLE(h_http2_flush) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    if (!live_any(a0)) return err(http2::kErrorInvalidId);
    return 0;
}

// The cookie/statistics calls prosper cannot answer without inventing data. Each fails with its
// out-parameters untouched: AddCookie and CookieImport would have to claim a store that does not
// exist, and GetCookie, CookieExport, GetCookieStats and GetMemoryPoolStats would have to fill a
// structure out of nothing. Reporting SCE_OK for any of them is the defect this file removes, not
// a smaller version of it. CONFIDENCE: HIGH that failing is right; the constant is the facility's
// invalid-value code because no "unsupported" code is attested in the module.
HLE(h_http2_not_modelled) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_mx);
    if (!live_any(a0)) return err(http2::kErrorInvalidId);
    return err(http2::kErrorInvalidValue);
}

} // namespace

void register_http2_hle() {
    // Context lifecycle. sceHttp2Init used to live in hle_service.cpp as the library's only
    // registration; it is here now because CreateTemplate validates against the same table.
    Hle::register_fn("3JCe3lCbQ8A", (HleFn)h_http2_init, "sceHttp2Init");
    Hle::register_fn("YiBUtz-pGkc", (HleFn)h_http2_term, "sceHttp2Term");

    // Local objects: real ids from a real table, never the dispatcher's 0.
    Hle::register_fn("+wCt7fCijgk", (HleFn)h_http2_create_template, "sceHttp2CreateTemplate");
    Hle::register_fn("pDom5-078DA", (HleFn)h_http2_delete_template, "sceHttp2DeleteTemplate");
    Hle::register_fn("mmyOCxQMVYQ", (HleFn)h_http2_create_request_with_url,
                     "sceHttp2CreateRequestWithURL");
    Hle::register_fn("c8D9qIjo8EY", (HleFn)h_http2_delete_request, "sceHttp2DeleteRequest");
    Hle::register_fn("IZ-qjhRqvjk", (HleFn)h_http2_abort_request, "sceHttp2AbortRequest");
    Hle::register_fn("N4UfjvWJsMw", (HleFn)h_http2_create_cookie_box, "sceHttp2CreateCookieBox");
    Hle::register_fn("O9ync3F-JVI", (HleFn)h_http2_delete_cookie_box, "sceHttp2DeleteCookieBox");

    // Setters: record, then answer SCE_OK.
    Hle::register_fn("nrPfOE8TQu0", (HleFn)h_http2_add_request_header, "sceHttp2AddRequestHeader");
    Hle::register_fn("jHdP0CS4ZlA", (HleFn)h_http2_remove_request_header,
                     "sceHttp2RemoveRequestHeader");
    Hle::register_fn("FSAFOzi0FpM", (HleFn)h_http2_set_request_content_length,
                     "sceHttp2SetRequestContentLength");
    Hle::register_fn("-HIO4VT87v8", (HleFn)h_http2_set_connect_timeout, "sceHttp2SetConnectTimeOut");
    Hle::register_fn("izvHhqgDt44", (HleFn)h_http2_set_recv_timeout, "sceHttp2SetRecvTimeOut");
    Hle::register_fn("XPtW45xiLHk", (HleFn)h_http2_set_send_timeout, "sceHttp2SetSendTimeOut");
    Hle::register_fn("ACjtE27aErY", (HleFn)h_http2_set_resolve_timeout, "sceHttp2SetResolveTimeOut");
    Hle::register_fn("Gcjh+CisAZM", (HleFn)h_http2_set_resolve_retry, "sceHttp2SetResolveRetry");
    Hle::register_fn("n8hMLe31OPA", (HleFn)h_http2_set_connection_wait_timeout,
                     "sceHttp2SetConnectionWaitTimeOut");
    Hle::register_fn("VYMxTcBqSE0", (HleFn)h_http2_set_timeout, "sceHttp2SetTimeOut");
    Hle::register_fn("jjFahkBPCYs", (HleFn)h_http2_set_auth_enabled, "sceHttp2SetAuthEnabled");
    Hle::register_fn("m-OL13q8AI8", (HleFn)h_http2_get_auth_enabled, "sceHttp2GetAuthEnabled");
    Hle::register_fn("b9AvoIaOuHI", (HleFn)h_http2_set_auto_redirect, "sceHttp2SetAutoRedirect");
    Hle::register_fn("od5QCZhZSfw", (HleFn)h_http2_get_auto_redirect, "sceHttp2GetAutoRedirect");
    Hle::register_fn("uRosf8GQbHQ", (HleFn)h_http2_set_inflate_gzip_enabled,
                     "sceHttp2SetInflateGZIPEnabled");
    Hle::register_fn("09tk+kIA1Ns", (HleFn)h_http2_set_min_ssl_version, "sceHttp2SetMinSslVersion");
    Hle::register_fn("mPKVhQqh2Es", (HleFn)h_http2_set_cookie_max_num, "sceHttp2SetCookieMaxNum");
    Hle::register_fn("o7+WXe4WadE", (HleFn)h_http2_set_cookie_max_num_per_domain,
                     "sceHttp2SetCookieMaxNumPerDomain");
    Hle::register_fn("6a0N6GPD7RM", (HleFn)h_http2_set_cookie_max_size, "sceHttp2SetCookieMaxSize");
    Hle::register_fn("EWcwMpbr5F8", (HleFn)h_http2_ssl_enable_option, "sceHttp2SslEnableOption");
    Hle::register_fn("B37SruheQ5Y", (HleFn)h_http2_ssl_disable_option, "sceHttp2SslDisableOption");
    Hle::register_fn("BJgi0CH7al4", (HleFn)h_http2_set_redirect_callback,
                     "sceHttp2SetRedirectCallback");
    Hle::register_fn("YrWX+DhPHQY", (HleFn)h_http2_set_ssl_callback, "sceHttp2SetSslCallback");
    Hle::register_fn("Wwj6HbB2mOo", (HleFn)h_http2_set_auth_info_callback,
                     "sceHttp2SetAuthInfoCallback");
    Hle::register_fn("zdtXKn9X7no", (HleFn)h_http2_set_cookie_recv_callback,
                     "sceHttp2SetCookieRecvCallback");
    Hle::register_fn("McYmUpQ3-DY", (HleFn)h_http2_set_cookie_send_callback,
                     "sceHttp2SetCookieSendCallback");
    Hle::register_fn("UL4Fviw+IAM", (HleFn)h_http2_set_pre_send_callback,
                     "sceHttp2SetPreSendCallback");
    Hle::register_fn("jrVHsKCXA0g", (HleFn)h_http2_set_cookie_box, "sceHttp2SetCookieBox");
    Hle::register_fn("IX23slKvtQI", (HleFn)h_http2_get_cookie_box, "sceHttp2GetCookieBox");

    // Empty caches: flushing them really does succeed, and none writes an out-parameter.
    Hle::register_fn("5VlQSzXW-SQ", (HleFn)h_http2_flush, "sceHttp2CookieFlush");
    Hle::register_fn("WeuDjj5m4YU", (HleFn)h_http2_flush, "sceHttp2AuthCacheFlush");
    Hle::register_fn("klwUy2Wg+q8", (HleFn)h_http2_flush, "sceHttp2RedirectCacheFlush");

    // The network boundary: fail, and leave every out-parameter untouched.
    Hle::register_fn("rbqZig38AT8", (HleFn)h_http2_send_request, "sceHttp2SendRequest");
    Hle::register_fn("A+NVAFu4eCg", (HleFn)h_http2_send_request_async, "sceHttp2SendRequestAsync");
    Hle::register_fn("-rdXUi2XW90", (HleFn)h_http2_get_all_response_headers,
                     "sceHttp2GetAllResponseHeaders");
    Hle::register_fn("9XYJwCf3lEA", (HleFn)h_http2_get_status_code, "sceHttp2GetStatusCode");
    Hle::register_fn("o0DBQpFE13o", (HleFn)h_http2_get_response_content_length,
                     "sceHttp2GetResponseContentLength");
    Hle::register_fn("QygCNNmbGss", (HleFn)h_http2_read_data, "sceHttp2ReadData");
    Hle::register_fn("bGN-6zbo7ms", (HleFn)h_http2_read_data_async, "sceHttp2ReadDataAsync");
    Hle::register_fn("MOp-AUhdfi8", (HleFn)h_http2_wait_async, "sceHttp2WaitAsync");

    // Cookie data and statistics prosper does not model: fail rather than fabricate.
    Hle::register_fn("flPxnowtvWY", (HleFn)h_http2_not_modelled, "sceHttp2AddCookie");
    Hle::register_fn("GQFGj0rYX+A", (HleFn)h_http2_not_modelled, "sceHttp2GetCookie");
    Hle::register_fn("JlFGR4v50Kw", (HleFn)h_http2_not_modelled, "sceHttp2CookieExport");
    Hle::register_fn("B5ibZI5UlzU", (HleFn)h_http2_not_modelled, "sceHttp2CookieImport");
    Hle::register_fn("eij7UzkUqK8", (HleFn)h_http2_not_modelled, "sceHttp2GetCookieStats");
    Hle::register_fn("otUQuZa-mv0", (HleFn)h_http2_not_modelled, "sceHttp2GetMemoryPoolStats");
}

} // namespace prosper
