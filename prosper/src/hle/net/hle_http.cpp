// libSceHttp URI helpers plus the library/template id lifecycle. Network requests remain
// offline, but parsing is local and deterministic, and id-returning entry points hand out
// real tracked ids: returning success without filling this structure makes callers dereference
// stale pointer fields, and answering 0 for an id hands the guest a valid-looking handle that
// was never allocated (#2930).
#include "hle/net/hle_http.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>

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

// --- Library/template id lifecycle (#2930) ----------------------------------------------
// sceHttpInit and sceHttpCreateTemplate return IDS, and an id-returning contract must never
// answer 0: the dispatcher's unregistered default is a valid-looking context/template id that
// six of eight surveyed titles carry into later calls. Offline there is no network behind these
// objects, but the ids are real - allocated here, tracked, deletable.
//
// Deliberately not invented until a live caller supplies evidence (id positivity itself is
// CONFIDENCE: HIGH from the published contract): repeated Init hands out a further independent
// context, CreateTemplate accepts whatever context it is given, and DeleteTemplate answers
// SCE_OK for any argument - the dispatcher default it replaces was also 0, so registering the
// explicit no-op removes census noise without fabricating an SDK error encoding. PS5 3.20
// libSceHttp exports no sceHttpTerminate: contexts live for the process.

struct HttpLibCtx   { bool in_use = false; };
struct HttpTemplate { bool in_use = false; };
constexpr int kMaxHttpLibs      = 4;
constexpr int kMaxHttpTemplates = 16;

std::mutex g_http_mx;  // guards both tables below
HttpLibCtx   g_http_libs[kMaxHttpLibs];
HttpTemplate g_http_templates[kMaxHttpTemplates];

HLE(h_http_init) { // sceHttpInit() -> library context id (>0)
    std::lock_guard<std::mutex> lk(g_http_mx);
    for (int i = 0; i < kMaxHttpLibs; i++) {
        if (g_http_libs[i].in_use) continue;
        g_http_libs[i].in_use = true;
        return (uint64_t)(i + 1);
    }
    return (uint64_t)(int64_t)-1;  // table exhausted: negative, never an id-shaped answer
}

HLE(h_http_create_template) { // sceHttpCreateTemplate(libCtxId, ...) -> template id (>0)
    std::lock_guard<std::mutex> lk(g_http_mx);
    for (int i = 0; i < kMaxHttpTemplates; i++) {
        if (g_http_templates[i].in_use) continue;
        g_http_templates[i].in_use = true;
        return (uint64_t)(i + 1);
    }
    return (uint64_t)(int64_t)-1;
}

HLE(h_http_delete_template) { // sceHttpDeleteTemplate(templateId) -> SCE_OK offline
    std::lock_guard<std::mutex> lk(g_http_mx);
    if (int id = (int32_t)a0; id >= 1 && id <= kMaxHttpTemplates)
        g_http_templates[id - 1].in_use = false;
    return 0;
}
} // namespace

void register_http_hle() {
    Hle::register_fn("IWalAn-guFs", (HleFn)h_http_uri_parse, "sceHttpUriParse");
    // Id lifecycle (#2930): an id-returning contract must never answer the dispatcher's 0.
    Hle::register_fn("A9cVMUtEp4Y", (HleFn)h_http_init, "sceHttpInit");
    Hle::register_fn("0gYjPTR-6cY", (HleFn)h_http_create_template, "sceHttpCreateTemplate");
    Hle::register_fn("4I8vEpuEhZ8", (HleFn)h_http_delete_template, "sceHttpDeleteTemplate");
}

} // namespace prosper
