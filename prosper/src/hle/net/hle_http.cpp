// libSceHttp: the URI helpers, the local object graph (library contexts, templates, connections,
// requests) and the request path. Network requests remain offline, but parsing is local and
// deterministic, and id-returning entry points hand out real tracked ids: returning success
// without filling this structure makes callers dereference stale pointer fields, and answering 0
// for an id hands the guest a valid-looking handle that was never allocated (#2930).
//
// The request path splits the same way, in both directions. Creating a connection or a request
// really does succeed offline -- it allocates a local object -- so it returns a real id, and the
// header/content-length setters record what they are given before answering SCE_OK. Everything
// that would need a network answer FAILS and leaves its out-parameters untouched: sceHttpSendRequest
// reports the libSceNet error an unreachable network produces, and every response getter reports
// that same failure. Nothing here manufactures a status code, a header block or a body.
#include "hle/net/hle_http.hpp"
#include "hle/net/sce_net_errors.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prosper {

#define HLE(name) static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                                        uint64_t a3, uint64_t a4, uint64_t a5)

namespace {

using http::SceHttpUriElement;
constexpr size_t kMaxUriBytes = 64 * 1024;

struct ParsedUri {
    bool opaque = true;
    std::string scheme, username, password, hostname, path, query, fragment;
    uint16_t port = 0;
};

bool ascii_ieq(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        auto ca = static_cast<unsigned char>(a[i]);
        auto cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) return false;
    }
    return true;
}

bool valid_scheme(std::string_view value) {
    if (value.empty() || !std::isalpha(static_cast<unsigned char>(value[0]))) return false;
    return std::all_of(value.begin() + 1, value.end(), [](char c) {
        auto u = static_cast<unsigned char>(c);
        return std::isalnum(u) || c == '+' || c == '-' || c == '.';
    });
}

bool parse_port(std::string_view text, uint16_t& out) {
    if (text.empty() || text.size() > 5) return false;
    uint32_t value = 0;
    for (char c : text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
        value = value * 10u + static_cast<uint32_t>(c - '0');
    }
    if (value > 65535u) return false;
    out = static_cast<uint16_t>(value);
    return true;
}

bool parse_uri(std::string_view input, ParsedUri& out) {
    size_t pos = 0;
    const size_t colon = input.find(':');
    const size_t first_delim = input.find_first_of("/?#");
    if (colon != std::string_view::npos &&
        (first_delim == std::string_view::npos || colon < first_delim)) {
        if (!valid_scheme(input.substr(0, colon))) return false;
        out.scheme.assign(input.substr(0, colon));
        pos = colon + 1;
    }

    const bool has_authority = input.substr(pos, 2) == "//";
    out.opaque = !has_authority;
    if (has_authority) {
        pos += 2;
        size_t end = input.find_first_of("/?#", pos);
        if (end == std::string_view::npos) end = input.size();
        std::string_view authority = input.substr(pos, end - pos);
        pos = end;

        const size_t at = authority.rfind('@');
        if (at != std::string_view::npos) {
            std::string_view userinfo = authority.substr(0, at);
            authority.remove_prefix(at + 1);
            const size_t sep = userinfo.find(':');
            out.username.assign(userinfo.substr(0, sep));
            if (sep != std::string_view::npos) out.password.assign(userinfo.substr(sep + 1));
        }

        if (!authority.empty() && authority.front() == '[') {
            const size_t close = authority.find(']');
            if (close == std::string_view::npos) return false;
            out.hostname.assign(authority.substr(1, close - 1));
            if (close + 1 < authority.size()) {
                if (authority[close + 1] != ':' ||
                    !parse_port(authority.substr(close + 2), out.port)) return false;
            }
        } else {
            const size_t port_sep = authority.rfind(':');
            if (port_sep != std::string_view::npos) {
                out.hostname.assign(authority.substr(0, port_sep));
                if (!parse_port(authority.substr(port_sep + 1), out.port)) return false;
            } else {
                out.hostname.assign(authority);
            }
        }
    }

    size_t path_end = input.find_first_of("?#", pos);
    if (path_end == std::string_view::npos) path_end = input.size();
    out.path.assign(input.substr(pos, path_end - pos));
    pos = path_end;

    if (pos < input.size() && input[pos] == '?') {
        size_t query_end = input.find('#', pos);
        if (query_end == std::string_view::npos) query_end = input.size();
        out.query.assign(input.substr(pos, query_end - pos));
        pos = query_end;
    }
    if (pos < input.size() && input[pos] == '#') out.fragment.assign(input.substr(pos));

    if (out.port == 0) {
        if (ascii_ieq(out.scheme, "http")) out.port = 80;
        else if (ascii_ieq(out.scheme, "https")) out.port = 443;
    }
    return true;
}

size_t pool_requirement(const ParsedUri& uri) {
    return uri.scheme.size() + uri.username.size() + uri.password.size() + uri.hostname.size() +
           uri.path.size() + uri.query.size() + uri.fragment.size() + 7;
}

char* copy_component(char*& cursor, const std::string& value) {
    char* result = cursor;
    std::memcpy(cursor, value.c_str(), value.size() + 1);
    cursor += value.size() + 1;
    return result;
}

HLE(h_http_uri_parse) { // (out, src_uri, pool, required_size, pool_size)
    (void)a5;
    const char* src = reinterpret_cast<const char*>(a1);
    if (!src) return http::kErrorInvalidUrl;
    const size_t length = strnlen(src, kMaxUriBytes);
    if (length == kMaxUriBytes) return http::kErrorInvalidUrl;

    ParsedUri parsed;
    if (!parse_uri(std::string_view(src, length), parsed)) return http::kErrorInvalidUrl;
    const size_t required = pool_requirement(parsed);
    auto* required_out = reinterpret_cast<uint64_t*>(a3);
    if (required_out) *required_out = required;

    auto* out = reinterpret_cast<SceHttpUriElement*>(a0);
    auto* pool = reinterpret_cast<char*>(a2);
    const bool write_output = out && pool;
    if (!write_output) return required_out ? 0 : http::kErrorInvalidValue;
    if (a4 < required) return http::kErrorOutOfMemory;

    std::memset(out, 0, sizeof(*out));
    char* cursor = pool;
    out->opaque = parsed.opaque;
    out->scheme = copy_component(cursor, parsed.scheme);
    out->username = copy_component(cursor, parsed.username);
    out->password = copy_component(cursor, parsed.password);
    out->hostname = copy_component(cursor, parsed.hostname);
    out->path = copy_component(cursor, parsed.path);
    out->query = copy_component(cursor, parsed.query);
    out->fragment = copy_component(cursor, parsed.fragment);
    out->port = parsed.port;
    return 0;
}

// --- sceHttpUriBuild (#2930) --------------------------------------------------------------
// The inverse of sceHttpUriParse, and like it a purely local, deterministic computation - no
// network is involved in turning a SceHttpUriElement back into a string. Left to the
// dispatcher it answered SCE_OK with the caller's buffer untouched, so the caller read
// whatever already happened to be in that buffer as a URI. Sifu is the surveyed caller.
//
// Contract read off the shipped PS5 3.20 libSceHttp (sceHttpUriBuild, 5LZA+KPISVA), whose URI
// helpers are ordinary local string code:
//
//   int32_t sceHttpUriBuild(char* out, size_t* required, size_t pool_size,
//                           const SceHttpUriElement* element, uint32_t flags);
//
//   element == nullptr                       -> kErrorInvalidUrl   (checked FIRST, before out)
//   out == nullptr && required == nullptr    -> kErrorInvalidValue
//   *required                                -> assembled length INCLUDING the NUL, and it is
//                                               stored BEFORE the buffer check, so a caller
//                                               given kErrorOutOfMemory still learns the size
//   out == nullptr                           -> size query only, SCE_OK
//   needed > pool_size                       -> kErrorOutOfMemory
//
// Emission order is scheme ':' "//" user ':' pass '@' host ':' port path query fragment, each
// part gated by its flag bit, with per-component strnlen caps matching the library's.
// CONFIDENCE: HIGH.
constexpr size_t kUriSchemeMax = 0x20;
constexpr size_t kUriUserMax = 0x100;
constexpr size_t kUriHostMax = 0xff;
constexpr size_t kUriTextMax = 0x3fff;

size_t capped_len(const char* text, size_t cap) { return text ? strnlen(text, cap) : 0; }

// The library upper-cases the scheme into a stack buffer and then runs three PREFIX compares --
// strncmp(upper, LIT, strlen(LIT)) -- against "HTTPS" (443, +0x21aa2), "HTTP" (80, +0x21aab) and
// a third literal that string merging left as "TTP" (+0x326ac). That third arm ends in
// `mov eax,0x50 / cmovne eax,ecx` with ecx zeroed, and cmovne fires on MISMATCH: a "TTP" prefix
// yields 80, and only a non-match yields "no default port".
//
// Reproduced rather than narrowed to exact equality, which is what this comment used to claim was
// safe. element->scheme is GUEST-supplied and need not have come from our own parser, so "httpx"
// prefix-matches "HTTP" and takes 80 on hardware; under exact matching prosper would answer 0 and
// then emit a ":80" that the library suppresses. Divergence for a scheme nobody uses, but the
// contract this file claims is fidelity to the shipped library, so match it.
uint16_t default_port_for_scheme(const char* scheme) {
    const size_t length = capped_len(scheme, kUriSchemeMax);
    if (length == 0) return 0;
    const std::string_view value(scheme, length);
    // strncmp against a shorter buffer stops at its NUL, so a prefix match needs the whole literal.
    const auto prefixed = [value](std::string_view literal) {
        return value.size() >= literal.size() &&
               ascii_ieq(value.substr(0, literal.size()), literal);
    };
    if (prefixed("https")) return 443;
    if (prefixed("http")) return 80;
    if (prefixed("ttp")) return 80;
    return 0;
}

// The library reads the leading dword and treats "opaque" as exactly 1; any other value means
// an authority follows and "//" is emitted. Read it the same way rather than through the bool.
bool element_is_opaque(const SceHttpUriElement* element) {
    uint32_t raw = 0;
    std::memcpy(&raw, element, sizeof(raw));
    return raw == 1u;
}

HLE(h_http_uri_build) { // (out, required, pool_size, element, flags)
    (void)a5;
    const auto* element = reinterpret_cast<const SceHttpUriElement*>(a3);
    if (!element) return http::kErrorInvalidUrl;
    auto* out = reinterpret_cast<char*>(a0);
    auto* required_out = reinterpret_cast<uint64_t*>(a1);
    if (!out && !required_out) return http::kErrorInvalidValue;
    const size_t pool_size = a2;
    const auto flags = static_cast<uint32_t>(a4);

    size_t needed = 0;

    size_t scheme_len = 0;
    if (flags & http::kUriBuildScheme) {
        scheme_len = capped_len(element->scheme, kUriSchemeMax);
        if (scheme_len) needed += scheme_len + 1;  // + ':'
    }
    const bool opaque = element_is_opaque(element);
    if (!opaque) needed += 2;  // "//" is emitted whenever an authority follows

    size_t user_len = 0;
    if (flags & http::kUriBuildUsername) {
        user_len = capped_len(element->username, kUriUserMax);
        if (user_len) needed += user_len + 1;  // + '@'
    }
    size_t pass_len = 0;
    if (flags & http::kUriBuildPassword) {
        pass_len = capped_len(element->password, kUriUserMax);
        if (pass_len) {
            needed += pass_len + 1;             // + ':'
            if (!user_len) needed += 1;         // ...and the '@' the username would have paid
        }
    }

    size_t host_len = 0;
    if (flags & http::kUriBuildHostname) host_len = capped_len(element->hostname, kUriHostMax);
    needed += host_len;

    // The port is decided from element->scheme whether or not the SCHEME bit is set: it is
    // omitted when it merely restates the scheme's default, and when it is zero.
    char port_text[8] = {};
    size_t port_len = 0;
    if (flags & http::kUriBuildPort) {
        bool emit = true;
        if (element->scheme) {
            const uint16_t fallback = default_port_for_scheme(element->scheme);
            if (fallback != 0) {
                emit = element->port != fallback;
            } else if (ascii_ieq(std::string_view(element->scheme,
                                                  capped_len(element->scheme, kUriSchemeMax)),
                                 "mailto")) {
                emit = element->port != 0;
            }
        }
        if (emit && element->port != 0) {
            std::snprintf(port_text, sizeof(port_text), ":%d", (int)element->port);
            port_len = strnlen(port_text, sizeof(port_text));
            needed += port_len;
        }
    }

    size_t path_len = 0, query_len = 0, fragment_len = 0;
    if (flags & http::kUriBuildPath) path_len = capped_len(element->path, kUriTextMax);
    if (flags & http::kUriBuildQuery) query_len = capped_len(element->query, kUriTextMax);
    if (flags & http::kUriBuildFragment) fragment_len = capped_len(element->fragment, kUriTextMax);
    needed += path_len + query_len + fragment_len;
    needed += 1;  // NUL

    // Stored before the size check on purpose: an under-sized pool still learns what it needs.
    if (required_out) *required_out = needed;
    if (!out) return 0;
    if (needed > pool_size) return http::kErrorOutOfMemory;

    char* cursor = out;
    auto append = [&cursor](const char* text, size_t length) {
        if (length) std::memcpy(cursor, text, length);
        cursor += length;
    };
    if (scheme_len) { append(element->scheme, scheme_len); *cursor++ = ':'; }
    if (!opaque) { *cursor++ = '/'; *cursor++ = '/'; }
    if (user_len) append(element->username, user_len);
    if (pass_len) { *cursor++ = ':'; append(element->password, pass_len); }
    if (user_len || pass_len) *cursor++ = '@';
    append(element->hostname, host_len);
    append(port_text, port_len);
    append(element->path, path_len);
    append(element->query, query_len);
    append(element->fragment, fragment_len);
    *cursor = '\0';
    return 0;
}

// --- Local object graph and request path (#2930) -----------------------------------------
// sceHttpInit, sceHttpCreateTemplate, sceHttpCreateConnection* and sceHttpCreateRequest* return
// IDS, and an id-returning contract must never answer 0: the dispatcher's unregistered default is
// a valid-looking context/template/connection/request id that six of eight surveyed titles carry
// into later calls. Offline there is no network behind these objects, but the ids are real -
// allocated here, owned, tracked, deletable.
//
// ONE UNIFIED ID SPACE holds all four classes, which is what the request path needed and the
// two separate tables here could not give. With a table per class a template id and a request id
// collide numerically, so a validator handed an id cannot tell a live object of the WRONG class
// from a live object of the right one - and this library's setters are documented to take a
// template, connection or request id interchangeably. The guest-visible contract of what was here
// before is unchanged: positive ids, 0x80431100 for an id nobody allocated, and a Term that
// reclaims what its context owned. libSceHttp2 uses the same arrangement for the same reason
// (#2894).
//
// Repeated Init hands out a further independent context, and DeleteTemplate answers SCE_OK for
// any argument - the dispatcher default it replaces was also 0, so the explicit no-op removes
// census noise without fabricating an SDK error encoding.
//
// CreateTemplate now VALIDATES its library context id rather than accepting whatever it is
// given. The encoding is not invented: sceHttpCreateTemplate (+0x107f0) calls the validator at
// +0xb070, which range-checks the id (1..0x80), cross-checks the slot in the 0xd0-stride context
// table, and answers 0x80431100 on either failure. That is the validator on this call path, not
// a same-shaped neighbour. CONFIDENCE: HIGH.
//
// Known fidelity gap, seen and not missed: the library answers 0x80431001 (BEFORE_INIT) when
// sceHttpInit was never called and only reaches the id validator once the library is up, while
// prosper answers 0x80431100 for both. Derivable locally -- no live context means never inited --
// but the guest classifies the two identically, so it is not worth a second code path yet.
//
// This comment used to say "PS5 3.20 libSceHttp exports no sceHttpTerminate: contexts live for
// the process". The name is wrong, and so was the conclusion drawn from it - the export is
// sceHttpTerm (Ik-KpLTlf7Q), it does exist, and it takes the library context id. It is
// registered below, so a context is now releasable and the four-slot table cannot be leaked
// dry by a title that inits and terminates repeatedly.

enum class Kind : uint8_t { None, Ctx, Template, Connection, Request };

struct HttpObject {
    Kind kind = Kind::None;
    int32_t owner = 0;  // the id this object was created from; 0 for a library context

    // Recorded state, so a setter's SCE_OK means the state really is recorded rather than that the
    // call was ignored. Nothing here is ever handed back as a network answer.
    std::string url;        // connection: server name or URL. request: path or URL.
    std::string method;     // request method when it was given as a string
    int32_t method_id = 0;  // request method when it was given as the library's enum
    uint64_t content_length = 0;
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::string> raw_headers;

    bool send_attempted = false;
    uint32_t send_error = 0;
    int32_t last_errno = 0;
    bool aborted = false;

    // Option state recorded by the setters below (all default zero/false), so a setter's
    // SCE_OK means the state really is recorded. Nothing here is ever handed back as a
    // network answer.
    bool nonblock = false;
    bool auto_redirect = false;
    bool auth_enabled = false;
    uint32_t resolve_timeout_us = 0;   // the option payloads are 32-bit on hardware
    uint32_t connect_timeout_us = 0;
    uint32_t send_timeout_us = 0;
    uint32_t recv_timeout_us = 0;
    int32_t resolve_retry = 0;
    uint64_t epoll_handle = 0;   // live host block, or 0 when unbound
    uint64_t epoll_arg = 0;
};

constexpr int kMaxHttpObjects = 128;
constexpr size_t kMaxHttpUrlBytes = 64 * 1024;
constexpr size_t kMaxHttpHeaderBytes = 16 * 1024;

std::mutex g_http_mx;                            // guards the object table
HttpObject g_http_objects[kMaxHttpObjects + 1];  // 1-based; slot 0 is never handed out

// Two return conventions, and they differ because the contracts do. An id-returning entry point
// sign-extends its error, so a guest that reads the answer as int32 OR as int64 sees it as
// negative and never as a handle. An SCE_OK-or-error entry point keeps the zero-extended 32-bit
// form the URI helpers already use.
uint64_t http_err(uint32_t code) { return (uint64_t)code; }
uint64_t http_id_err(uint32_t code) { return (uint64_t)(int64_t)(int32_t)code; }

int32_t http_alloc(Kind kind, int32_t owner) {  // caller holds g_http_mx
    for (int i = 1; i <= kMaxHttpObjects; i++) {
        if (g_http_objects[i].kind != Kind::None) continue;
        g_http_objects[i] = HttpObject{};
        g_http_objects[i].kind = kind;
        g_http_objects[i].owner = owner;
        return i;
    }
    return 0;
}

HttpObject* http_live(uint64_t raw, Kind kind) {  // caller holds g_http_mx
    const int32_t id = (int32_t)raw;
    if (id < 1 || id > kMaxHttpObjects) return nullptr;
    HttpObject& obj = g_http_objects[id];
    return obj.kind == kind ? &obj : nullptr;
}

// Any live object, whatever its class.
//
// Used wherever the library's own contract is permissive: a setter documented to take "a template,
// connection or request id", and the create paths, whose exact accepted parent class this machine
// cannot check against the shipped module. Policing the class there would risk REJECTING a call
// hardware accepts, which is the under-report half of this same defect. What is enforced -- and it
// is the part that matters -- is that the id was actually ALLOCATED, so 0 and every id nobody
// handed out are refused. CONFIDENCE: HIGH on that check; LOW on any per-class restriction, which
// is why none is imposed.
HttpObject* http_live_any(uint64_t raw) {  // caller holds g_http_mx
    const int32_t id = (int32_t)raw;
    if (id < 1 || id > kMaxHttpObjects) return nullptr;
    HttpObject& obj = g_http_objects[id];
    return obj.kind == Kind::None ? nullptr : &obj;
}

// Release everything owned by `owner`, transitively: a context owns its templates, a template its
// connections, a connection its requests.
void http_release_owned(int32_t owner) {  // caller holds g_http_mx
    for (int i = 1; i <= kMaxHttpObjects; i++) {
        if (g_http_objects[i].kind == Kind::None || g_http_objects[i].owner != owner) continue;
        http_release_owned(i);
        g_http_objects[i] = HttpObject{};
    }
}

// Epoll handles are opaque host blocks (SceHttpEpollHandle is a pointer type), recorded live
// with the library context they were created under, so Set/Unset/Destroy/Wait validate against
// what Create actually handed out and sceHttpTerm can release a context's handles. Caller holds
// g_http_mx throughout.
constexpr size_t kHttpEpollBlockBytes = 64;
std::unordered_map<uint64_t, int32_t>& http_epolls() {   // handle -> owning ctx id
    static std::unordered_map<uint64_t, int32_t> live;
    return live;
}
bool http_epoll_live(uint64_t handle) {
    return http_epolls().count(handle) != 0;
}
// Destroy walks the handle's bound list on hardware (+0x1b150); without the same walk a stale
// binding would look live again the moment calloc reuses the address.
void http_epoll_release(uint64_t handle) {
    for (int i = 1; i <= kMaxHttpObjects; i++) {
        if (g_http_objects[i].epoll_handle != handle) continue;
        g_http_objects[i].epoll_handle = 0;
        g_http_objects[i].epoll_arg = 0;
    }
    http_epolls().erase(handle);
    std::free((void*)(uintptr_t)handle);
}
void http_epoll_release_owned(int32_t ctx) {
    std::vector<uint64_t> owned;
    for (const auto& [handle, owner] : http_epolls())
        if (owner == ctx) owned.push_back(handle);
    for (uint64_t handle : owned) http_epoll_release(handle);
}

std::string http_capture(uint64_t ptr, size_t cap) {
    const char* text = reinterpret_cast<const char*>(ptr);
    if (!text) return {};
    return std::string(text, strnlen(text, cap));
}

// The answer for a response getter. Nothing was received, because nothing was sent: report the
// failure the send reported, or -- if the guest is reading a response before sending at all --
// that it asked out of order. NEITHER path touches an out-parameter, which is the whole point:
// a getter that answers SCE_OK while writing nothing hands the caller back whatever was already
// in its buffer, and callers then parse it.
uint64_t http_response_unavailable(const HttpObject& req) {
    return http_err(req.send_attempted ? req.send_error : http::kErrorBeforeInit);
}

HLE(h_http_init) { // sceHttpInit(libnetMemId, libsslCtxId, poolSize) -> library context id (>0)
    (void)a0; (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    const int32_t id = http_alloc(Kind::Ctx, 0);
    if (!id) return http_id_err(http::kErrorOutOfMemory);
    return (uint64_t)id;
}

HLE(h_http_term) { // sceHttpTerm(libCtxId) -> SCE_OK, releasing the context AND what it owns
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live(a0, Kind::Ctx)) return http_err(http::kErrorInvalidId);
    const int32_t id = (int32_t)a0;
    // Term does not merely drop the context. After the refcount decrement at +0x1068b the shipped
    // sceHttpTerm runs four per-context teardown helpers (+0x83f0, +0x1c8e0, +0x35b0, +0xaf90),
    // each taking the ctx id in edi; +0xaf90 re-validates the id against the same slot table the
    // create path uses. Releasing only the slot would leak every template created under it -- and
    // now every connection and request under those too -- so a title that inits, creates and terms
    // in a loop would exhaust a table that is really empty. Its epoll handles go with it.
    http_release_owned(id);
    http_epoll_release_owned(id);
    g_http_objects[id] = HttpObject{};
    return 0;
}

HLE(h_http_create_template) { // sceHttpCreateTemplate(libCtxId, ...) -> template id (>0)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live(a0, Kind::Ctx)) return http_id_err(http::kErrorInvalidId);
    const int32_t id = http_alloc(Kind::Template, (int32_t)a0);  // so sceHttpTerm can reclaim it
    if (!id) return http_id_err(http::kErrorOutOfMemory);
    return (uint64_t)id;
}

HLE(h_http_delete_template) { // sceHttpDeleteTemplate(templateId) -> SCE_OK offline
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    // Deliberately lenient about the id, as it has been since #2965: the dispatcher default this
    // replaced was also 0, so a no-op for an unknown argument removes census noise without
    // inventing an error the library may not return here. It is NOT lenient about the tree -- a
    // template that really is live takes its connections and requests with it.
    if (HttpObject* tmpl = http_live(a0, Kind::Template)) {
        (void)tmpl;
        const int32_t id = (int32_t)a0;
        http_release_owned(id);
        g_http_objects[id] = HttpObject{};
    }
    return 0;
}

// --- connections (#2930) -----------------------------------------------------------------
// A connection is a local object: creating one allocates bookkeeping and resolves nothing, so it
// genuinely succeeds offline and must return a real id. It is the FIRST thing a title does on the
// request path, and answering 0 here is what makes everything after it operate on a handle that
// was never allocated.

HLE(h_http_create_connection) { // (templateId, serverName, scheme, port, isEnableKeepalive)
    (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live_any(a0)) return http_id_err(http::kErrorInvalidId);
    if (!a1) return http_id_err(http::kErrorInvalidValue);
    const int32_t id = http_alloc(Kind::Connection, (int32_t)a0);
    if (!id) return http_id_err(http::kErrorOutOfMemory);
    g_http_objects[id].url = http_capture(a1, kMaxHttpUrlBytes);
    (void)a2; (void)a3;
    return (uint64_t)id;
}

HLE(h_http_create_connection_with_url) { // (templateId, url, isEnableKeepalive)
    (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live_any(a0)) return http_id_err(http::kErrorInvalidId);
    if (!a1) return http_id_err(http::kErrorInvalidValue);
    const int32_t id = http_alloc(Kind::Connection, (int32_t)a0);
    if (!id) return http_id_err(http::kErrorOutOfMemory);
    g_http_objects[id].url = http_capture(a1, kMaxHttpUrlBytes);
    (void)a2;
    return (uint64_t)id;
}

HLE(h_http_delete_connection) { // sceHttpDeleteConnection(connId) -> SCE_OK, with its requests
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live(a0, Kind::Connection)) return http_err(http::kErrorInvalidId);
    const int32_t id = (int32_t)a0;
    http_release_owned(id);
    g_http_objects[id] = HttpObject{};
    return 0;
}

// --- requests (#2930) ---------------------------------------------------------------------
// Four spellings, one object. The plain forms take the method as the library's enum; the "2" forms
// take it as a string. Both record what they were given -- neither interprets it, because nothing
// offline consumes it. CONFIDENCE: HIGH that argument 0 is the parent id and that the URL/path is
// argument 2; MED on the trailing content-length argument.

HLE(h_http_create_request) { // (connId, method /*enum*/, path_or_url, contentLength)
    (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live_any(a0)) return http_id_err(http::kErrorInvalidId);
    if (!a2) return http_id_err(http::kErrorInvalidValue);
    const int32_t id = http_alloc(Kind::Request, (int32_t)a0);
    if (!id) return http_id_err(http::kErrorOutOfMemory);
    HttpObject& req = g_http_objects[id];
    req.method_id = (int32_t)a1;
    req.url = http_capture(a2, kMaxHttpUrlBytes);
    req.content_length = a3;
    return (uint64_t)id;
}

HLE(h_http_create_request_str) { // (connId, method /*string*/, path_or_url, contentLength)
    (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live_any(a0)) return http_id_err(http::kErrorInvalidId);
    if (!a2) return http_id_err(http::kErrorInvalidValue);
    const int32_t id = http_alloc(Kind::Request, (int32_t)a0);
    if (!id) return http_id_err(http::kErrorOutOfMemory);
    HttpObject& req = g_http_objects[id];
    req.method = http_capture(a1, 64);
    req.url = http_capture(a2, kMaxHttpUrlBytes);
    req.content_length = a3;
    return (uint64_t)id;
}

HLE(h_http_delete_request) { // sceHttpDeleteRequest(requestId) -> SCE_OK
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live(a0, Kind::Request)) return http_err(http::kErrorInvalidId);
    g_http_objects[(int32_t)a0] = HttpObject{};
    return 0;
}

// Aborting a request that was never in flight is a purely local state change, so it genuinely
// succeeds. CONFIDENCE: MED -- hardware may answer an error when there is nothing to abort, but
// none of these has an out-parameter, so a wrong answer here cannot hand the guest unread memory.
HLE(h_http_abort_request) {
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* req = http_live(a0, Kind::Request);
    if (!req) return http_err(http::kErrorInvalidId);
    req->aborted = true;
    return 0;
}

// --- request headers and body length (#2930) ----------------------------------------------
// Local state: recorded first, SCE_OK second. "Success once the state is actually recorded", not
// success because there is nothing to do.

HLE(h_http_add_request_header) { // (id, name, value, mode)
    (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    if (!a1) return http_err(http::kErrorInvalidValue);
    obj->headers.emplace_back(http_capture(a1, kMaxHttpHeaderBytes),
                              http_capture(a2, kMaxHttpHeaderBytes));
    return 0;
}

HLE(h_http_add_request_header_raw) { // (id, rawHeaderBlock, size, mode)
    (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    if (!a1) return http_err(http::kErrorInvalidValue);
    const size_t cap = a2 && a2 < kMaxHttpHeaderBytes ? (size_t)a2 : kMaxHttpHeaderBytes;
    obj->raw_headers.push_back(http_capture(a1, cap));
    return 0;
}

HLE(h_http_remove_request_header) { // (id, name) -> removes every header recorded under that name
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    if (!a1) return http_err(http::kErrorInvalidValue);
    const std::string name = http_capture(a1, kMaxHttpHeaderBytes);
    std::vector<std::pair<std::string, std::string>> kept;
    kept.reserve(obj->headers.size());
    for (auto& header : obj->headers)
        if (header.first != name) kept.push_back(header);
    obj->headers.swap(kept);
    return 0;
}

HLE(h_http_set_request_content_length) { // (id, length)
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->content_length = a1;
    return 0;
}

// --- the network boundary (#2930) ----------------------------------------------------------
// Everything from here down would need a network answer, and prosper has none. Each reports a
// failure and leaves every out-parameter exactly as the caller left it. There is no synthesised
// status code, no fabricated header block and no invented body anywhere in this file: the guest's
// own error path runs instead, which is what it is written to do.

HLE(h_http_send_request) { // sceHttpSendRequest(requestId, postData, size)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* req = http_live(a0, Kind::Request);
    if (!req) return http_err(http::kErrorInvalidId);
    req->send_attempted = true;
    req->send_error = net::kNetErrorNetUnreach;
    req->last_errno = static_cast<int32_t>(net::kOfflineConnectErrno);   // sceHttpGetLastErrno
    return http_err(req->send_error);
}

HLE(h_http_read_data) { // sceHttpReadData(requestId, buf, size)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* req = http_live(a0, Kind::Request);
    if (!req) return http_err(http::kErrorInvalidId);
    return http_response_unavailable(*req);
}

HLE(h_http_get_all_response_headers) { // (requestId, char** header, size_t* headerSize)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* req = http_live(a0, Kind::Request);
    if (!req) return http_err(http::kErrorInvalidId);
    return http_response_unavailable(*req);
}

HLE(h_http_get_status_code) { // (requestId, int* statusCode)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* req = http_live(a0, Kind::Request);
    if (!req) return http_err(http::kErrorInvalidId);
    return http_response_unavailable(*req);
}

HLE(h_http_get_response_content_length) { // (requestId, int* result, uint64_t* contentLength)
    (void)a1; (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* req = http_live(a0, Kind::Request);
    if (!req) return http_err(http::kErrorInvalidId);
    return http_response_unavailable(*req);
}

// sceHttpGetLastErrno(requestId, int* errNum) -> SCE_OK, writing the errno prosper's own send
// failed with. This one IS derivable from local state: the send path above recorded why it failed,
// and a getter that reads back state prosper itself holds must answer it rather than refuse it.
// A request that was never sent reports 0, which is what "no error yet" means.
// CONFIDENCE: MED on the out-parameter width (int, matching the library's other id/int outputs).
HLE(h_http_get_last_errno) {
    (void)a2; (void)a3; (void)a4; (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* req = http_live(a0, Kind::Request);
    if (!req) return http_err(http::kErrorInvalidId);
    if (!a1) return http_err(http::kErrorInvalidValue);
    *reinterpret_cast<int32_t*>(a1) = req->last_errno;
    return 0;
}

// --- option setters + epoll lifecycle (firmware 3.20 set) ----------------------------------
// All take a live id first and record the value, so SCE_OK means recorded; an id nobody handed
// out fails InvalidId. NIDs from the firmware set (verified by nid_hash round-trip). Contracts
// read from the shipped libSceHttp module: the option payloads are 32-bit, and the epoll wait is
// sceHttpWaitRequest(eh, nbev, maxevents, timeout), implemented below with
// sceHttpAbortWaitRequest(eh). Create hands out a real zeroed host block so a title can bind,
// wait and tear down through the full lifecycle; nothing ever fires headless, because
// sceHttpSendRequest fails synchronously offline.
// Known gap: hardware's sceHttpSetResolveTimeOut also refuses usec < 1,000,000 (InvalidValue)
// when the title's SDK version is above 0x016fffff; prosper does not expose that version to
// the HLE layer here, so the check is not modelled.
HLE(h_http_set_nonblock) {   // (id, enable)
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->nonblock = a1 != 0;
    return 0;
}
HLE(h_http_set_auto_redirect) {   // (id, enable)
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->auto_redirect = a1 != 0;
    return 0;
}
HLE(h_http_set_auth_enabled) {   // (id, enable)
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->auth_enabled = a1 != 0;
    return 0;
}
HLE(h_http_set_resolve_timeout) {   // (id, usec)
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->resolve_timeout_us = (uint32_t)a1;
    return 0;
}
HLE(h_http_set_resolve_retry) {   // (id, retry)
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    // Hardware refuses a negative retry before it looks at the id (+0x12cd4 -> +0x12d2f).
    if ((int32_t)a1 < 0) return http_err(http::kErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->resolve_retry = (int32_t)a1;
    return 0;
}
HLE(h_http_set_connect_timeout) {   // (id, usec)
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->connect_timeout_us = (uint32_t)a1;
    return 0;
}
HLE(h_http_set_send_timeout) {   // (id, usec)
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->send_timeout_us = (uint32_t)a1;
    return 0;
}
HLE(h_http_set_recv_timeout) {   // (id, usec)
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->recv_timeout_us = (uint32_t)a1;
    return 0;
}


HLE(h_http_create_epoll) {   // (ctxId, HttpEpollHandle* out) -> SCE_OK, handle written
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    // Context first (+0xb070), then the out-pointer (+0x1aff5 -> INVALID_VALUE at +0x1b114).
    if (!http_live(a0, Kind::Ctx)) return http_err(http::kErrorInvalidId);
    if (!a1) return http_err(http::kErrorInvalidValue);
    void* block = std::calloc(1, kHttpEpollBlockBytes);
    if (!block) return http_err(http::kErrorOutOfMemory);
    const uint64_t handle = (uint64_t)(uintptr_t)block;
    http_epolls()[handle] = (int32_t)a0;
    *(uint64_t*)a1 = handle;
    return 0;
}
HLE(h_http_destroy_epoll) {   // (ctxId, handle) -> SCE_OK
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    // Context first, then a null handle is INVALID_VALUE (+0x1b15b -> +0x1b1bb). Hardware would
    // dereference a foreign handle; prosper refuses one it never handed out under this context.
    if (!http_live(a0, Kind::Ctx)) return http_err(http::kErrorInvalidId);
    if (!a1) return http_err(http::kErrorInvalidValue);
    const auto it = http_epolls().find(a1);
    if (it == http_epolls().end() || it->second != (int32_t)a0)
        return http_err(http::kErrorInvalidId);
    http_epoll_release(a1);
    return 0;
}
HLE(h_http_set_epoll) {   // (id, handle, user_arg) -> SCE_OK, binding recorded
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    // The id is validated first; the list insert then refuses a null handle (+0x1ac2e).
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    if (!a1) return http_err(http::kErrorInvalidValue);
    if (!http_epoll_live(a1)) return http_err(http::kErrorInvalidId);
    obj->epoll_handle = a1;
    obj->epoll_arg = a2;
    return 0;
}
HLE(h_http_unset_epoll) {   // (id) -> SCE_OK, binding cleared
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    HttpObject* obj = http_live_any(a0);
    if (!obj) return http_err(http::kErrorInvalidId);
    obj->epoll_handle = 0;
    obj->epoll_arg = 0;
    return 0;
}

// sceHttpWaitRequest(SceHttpEpollHandle eh, SceHttpNBEvent* nbev, int maxevents, int timeout)
// -> number of events written (0x18-byte records), or INVALID_VALUE for a null eh, a null nbev
// or maxevents <= 0 (+0x138a9, +0x138b5, +0x138c1). Offline no event can ever be pending:
// sceHttpSendRequest fails synchronously, so nothing is in flight to complete. So this answers 0
// events and writes nothing -- after sleeping for the requested timeout, bounded, outside the
// lock, so a caller polling in a loop cannot busy-spin. CONFIDENCE: LOW on the timeout's units
// (treated as microseconds) and on negative timeouts (unverified; treated as a short bounded wait
// rather than blocking forever). A zero timeout returns at once. A handle prosper never handed out is refused like a null one.
HLE(h_http_wait_epoll) {
    (void)a4;
    (void)a5;
    {
        std::lock_guard<std::mutex> lk(g_http_mx);
        if (!a0 || !a1 || (int32_t)a2 <= 0 || !http_epoll_live(a0))
            return http_id_err(http::kErrorInvalidValue);
    }
    constexpr int64_t kMaxWaitUs = 50'000;
    const int64_t timeout = (int32_t)a3;
    // The module hands the timeout straight to the libSceNet epoll wait, where 0 means "check and
    // return": a title polling with 0 every frame must not pay the bounded sleep.
    if (timeout == 0) return 0;
    const int64_t wait_us = timeout > 0 ? std::min(timeout, kMaxWaitUs) : kMaxWaitUs;
    std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
    return 0;
}

// sceHttpAbortWaitRequest(SceHttpEpollHandle eh) (+0x139c0 -> +0x1fc20): nothing is ever waiting
// headless longer than the bounded sleep above, so aborting is a validated no-op.
// CONFIDENCE: LOW on the code for a handle prosper never handed out (treated as INVALID_VALUE,
// like a null one).
HLE(h_http_abort_wait_epoll) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!a0 || !http_epoll_live(a0)) return http_err(http::kErrorInvalidValue);
    return 0;
}

// --- libSceHttps TLS setup (exports of libSceHttp itself) ----------------------------------
// sceHttpsSetSslCallback(id, callback, user_arg): hardware classifies the id (+0x15860 ->
// +0xfdc0 via +0x22920), accepts two object classes and answers INVALID_ID otherwise (+0xfe84).
// Validated here with http_live_any -- lenient on class, per this file's policy. prosper
// performs no TLS verification headless, so a callback it would never invoke is acknowledged
// rather than stored. Uncharted's one call site drops the result.
HLE(h_https_set_ssl_callback) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live_any(a0)) return http_err(http::kErrorInvalidId);
    return 0;
}

// sceHttpsDisableOption(id, sslFlags): flags outside 0x20ff are INVALID_VALUE before the id is
// looked at (+0x153e9); the id is then validated through the option path (+0x1551c). The gated
// 0x8043506b refusal hardware gives when bits 0x25 are disabled under a system condition
// (+0x1554f) is not modelled.
HLE(h_https_disable_option) {
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    if ((uint32_t)a1 & 0xffffdf00u) return http_err(http::kErrorInvalidValue);
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (!http_live_any(a0)) return http_err(http::kErrorInvalidId);
    return 0;
}
} // namespace

void register_http_hle() {
    Hle::register_fn("IWalAn-guFs", (HleFn)h_http_uri_parse, "sceHttpUriParse");
    // Id lifecycle (#2930): an id-returning contract must never answer the dispatcher's 0.
    Hle::register_fn("A9cVMUtEp4Y", (HleFn)h_http_init, "sceHttpInit");
    Hle::register_fn("0gYjPTR-6cY", (HleFn)h_http_create_template, "sceHttpCreateTemplate");
    Hle::register_fn("4I8vEpuEhZ8", (HleFn)h_http_delete_template, "sceHttpDeleteTemplate");
    Hle::register_fn("Ik-KpLTlf7Q", (HleFn)h_http_term, "sceHttpTerm");
    // Offline-computable, so there is no reason for it to answer a false success (#2930).
    Hle::register_fn("5LZA+KPISVA", (HleFn)h_http_uri_build, "sceHttpUriBuild");

    // The request path (#2930). Local objects get real ids; local state is recorded; anything
    // that would need a network answer fails with its out-parameters untouched.
    Hle::register_fn("Kiwv9r4IZCc", (HleFn)h_http_create_connection, "sceHttpCreateConnection");
    Hle::register_fn("qgxDBjorUxs", (HleFn)h_http_create_connection_with_url,
                     "sceHttpCreateConnectionWithURL");
    Hle::register_fn("P6A3ytpsiYc", (HleFn)h_http_delete_connection, "sceHttpDeleteConnection");
    Hle::register_fn("tsGVru3hCe8", (HleFn)h_http_create_request, "sceHttpCreateRequest");
    Hle::register_fn("Aeu5wVKkF9w", (HleFn)h_http_create_request, "sceHttpCreateRequestWithURL");
    Hle::register_fn("rGNm+FjIXKk", (HleFn)h_http_create_request_str, "sceHttpCreateRequest2");
    Hle::register_fn("Cnp77podkCU", (HleFn)h_http_create_request_str,
                     "sceHttpCreateRequestWithURL2");
    Hle::register_fn("qe7oZ+v4PWA", (HleFn)h_http_delete_request, "sceHttpDeleteRequest");
    Hle::register_fn("hvG6GfBMXg8", (HleFn)h_http_abort_request, "sceHttpAbortRequest");
    Hle::register_fn("JKl06ZIAl6A", (HleFn)h_http_abort_request, "sceHttpAbortRequestForce");
    Hle::register_fn("sWQiqKvYTVA", (HleFn)h_http_abort_wait_epoll, "sceHttpAbortWaitRequest");
    Hle::register_fn("EY28T2bkN7k", (HleFn)h_http_add_request_header, "sceHttpAddRequestHeader");
    Hle::register_fn("lGAjftanhFs", (HleFn)h_http_add_request_header_raw,
                     "sceHttpAddRequestHeaderRaw");
    Hle::register_fn("zNGh-zoQTD0", (HleFn)h_http_remove_request_header,
                     "sceHttpRemoveRequestHeader");
    Hle::register_fn("PTiFIUxCpJc", (HleFn)h_http_set_request_content_length,
                     "sceHttpSetRequestContentLength");
    // Option setters: validated against the live-id table, values recorded.
    Hle::register_fn("s2-NPIvz+iA", (HleFn)h_http_set_nonblock, "sceHttpSetNonblock");
    Hle::register_fn("T-mGo9f3Pu4", (HleFn)h_http_set_auto_redirect, "sceHttpSetAutoRedirect");
    Hle::register_fn("qFg2SuyTJJY", (HleFn)h_http_set_auth_enabled, "sceHttpSetAuthEnabled");
    Hle::register_fn("Tc-hAYDKtQc", (HleFn)h_http_set_resolve_timeout, "sceHttpSetResolveTimeOut");
    Hle::register_fn("K1d1LqZRQHQ", (HleFn)h_http_set_resolve_retry, "sceHttpSetResolveRetry");
    Hle::register_fn("0S9tTH0uqTU", (HleFn)h_http_set_connect_timeout, "sceHttpSetConnectTimeOut");
    Hle::register_fn("xegFfZKBVlw", (HleFn)h_http_set_send_timeout, "sceHttpSetSendTimeOut");
    Hle::register_fn("yigr4V0-HTM", (HleFn)h_http_set_recv_timeout, "sceHttpSetRecvTimeOut");
    // Epoll lifecycle: real host-block handles; nothing ever fires headless.
    Hle::register_fn("6381dWF+xsQ", (HleFn)h_http_create_epoll, "sceHttpCreateEpoll");
    Hle::register_fn("wYhXVfS2Et4", (HleFn)h_http_destroy_epoll, "sceHttpDestroyEpoll");
    Hle::register_fn("-xm7kZQNpHI", (HleFn)h_http_set_epoll, "sceHttpSetEpoll");
    Hle::register_fn("59tL1AQBb8U", (HleFn)h_http_unset_epoll, "sceHttpUnsetEpoll");
    // libSceHttps entry points, exported by libSceHttp itself.
    Hle::register_fn("htyBOoWeS58", (HleFn)h_https_set_ssl_callback, "sceHttpsSetSslCallback");
    Hle::register_fn("mSQCxzWTwVI", (HleFn)h_https_disable_option, "sceHttpsDisableOption");
    // The network boundary.
    Hle::register_fn("1e2BNwI-XzE", (HleFn)h_http_send_request, "sceHttpSendRequest");
    Hle::register_fn("P5pdoykPYTk", (HleFn)h_http_read_data, "sceHttpReadData");
    Hle::register_fn("qISjDHrxONc", (HleFn)h_http_wait_epoll, "sceHttpWaitRequest");
    Hle::register_fn("aCYPMSUIaP8", (HleFn)h_http_get_all_response_headers,
                     "sceHttpGetAllResponseHeaders");
    Hle::register_fn("4fgkfVeVsGU", (HleFn)h_http_get_all_response_headers,
                     "sceHttpRequestGetAllHeaders");
    Hle::register_fn("0a2TBNfE3BU", (HleFn)h_http_get_status_code, "sceHttpGetStatusCode");
    Hle::register_fn("yuO2H2Uvnos", (HleFn)h_http_get_response_content_length,
                     "sceHttpGetResponseContentLength");
    Hle::register_fn("0onIrKx9NIE", (HleFn)h_http_get_last_errno, "sceHttpGetLastErrno");
}

} // namespace prosper
