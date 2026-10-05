// libSceNet socket family: byte-order helpers, a local socket-id table, and an honest
// offline failure on every operation that would need a network.
//
// prosper has no network (see hle/net/AGENTS.md), and the dispatcher's unregistered default of 0
// is SCE_OK -- so an unregistered sceNetConnect reports a successful connect nobody made, and an
// unregistered sceNetRecv reports success while writing nothing to its buffer. That false success
// is strictly worse than a reported error, because the guest's own error handling never runs
// (#2894 is the HTTP instance of the same shape).
//
// The split, mirroring hle_http.cpp:
// - Offline-computable is implemented for real: htonl/htons/ntohl/ntohs/htonll/ntohll are pure
//   byte swaps (portable shifts, no host headers, so no platform seam is touched), and
//   sceNetInetPton/sceNetInetNtop convert AF_INET dotted-decimal and AF_INET6 text both ways
//   (the shipped module supports both families: pton 0x6aa0, ntop 0x63c0 / v6 at 0x6457).
// - Socket ids are a real local table (1-based; slot 0 is never handed out, exactly like the HTTP
//   object table): sceNetSocket(name, family, type, protocol) allocates -- the first argument is a
//   debug-name string, as in sceNetPoolCreate (module entry 0x7330 keeps rdi as a pointer and
//   esi/edx/ecx as the 32-bit triple) -- and sceNetSocketClose releases. An id-returning entry
//   point sign-extends its error, so a guest reading the answer as int32 OR int64 sees it as
//   negative and never as a handle.
// - Everything that would move a packet -- connect, send, recv, bind, listen, accept, socket
//   options, peer names, shutdown -- FAILS with the libSceNet facility error for an unreachable
//   network (net::kNetErrorNetUnreach, the same answer the HTTP send paths give) and leaves every
//   out-parameter untouched. A bad socket id answers EBADF instead, so use-after-close is also
//   loud. CONFIDENCE: HIGH on the facility encoding (sce_net_errors.hpp); LOW on each exact errno
//   choice, which is why every arm names the errno it encodes. In particular ENETUNREACH is not an
//   errno a real FreeBSD bind/listen produces (a bind to INADDR_ANY succeeds on an offline
//   console): it is chosen so the guest takes its error path, not observed.
// - sceNetErrnoLoc returns the address of a thread-local FreeBSD errno slot that the failing arms
//   above keep current, so a guest that reads errno after a failed call sees the failure prosper
//   reported. A successful call never writes it, matching the module (sceNetSocket's success path
//   skips the store at 0x73da; ntop's at 0x6455). CONFIDENCE: MED -- per-host-thread is the closest this in-process design gets to the
//   console's per-guest-thread slot (cf. the fiber caveat in docs/games/UNCHARTED_STATUS.md).
//
// Contracts are read off the shipped libSceNet module (offsets above are image-relative into it).
// KytyPS5 (src/libs/network.cpp), shadPS4 (src/core/libraries/network/) and sharpemu
// (SharpEmu.Libs/Network/) were also read as hypotheses; no code was copied (CONTRIBUTING.md).
#include "hle/net/hle_net.hpp"
#include "hle/net/sce_net_errors.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

namespace prosper {

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,      \
                                          uint64_t a4, uint64_t a5)

namespace {

using net::FreeBsdErrno;

// Two return conventions, matching hle_http.cpp: an SCE_OK-or-error entry point keeps the
// zero-extended 32-bit form; an id-returning one (socket, accept) sign-extends so the error can
// never be mistaken for a handle.
uint64_t net_err(uint32_t code) {
    return (uint64_t)code;
}
uint64_t net_id_err(uint32_t code) {
    return (uint64_t)(int64_t)(int32_t)code;
}

// The errno slot sceNetErrnoLoc exposes. One per host thread; every failing arm below records
// the FreeBSD errno number it encoded before returning.
thread_local int32_t g_net_errno = 0;

uint64_t net_fail(FreeBsdErrno e) {
    g_net_errno = static_cast<int32_t>(e);
    return net_err(net::net_error(e));
}

uint64_t net_id_fail(FreeBsdErrno e) {
    g_net_errno = static_cast<int32_t>(e);
    return net_id_err(net::net_error(e));
}

constexpr int kMaxNetSockets = 128;

std::mutex g_net_mx;                 // guards the socket table
bool g_net_live[kMaxNetSockets + 1];  // 1-based; slot 0 is never handed out

bool net_known_locked(int32_t fd) {
    return fd >= 1 && fd <= kMaxNetSockets && g_net_live[fd];
}

// --- byte order: pure computation, no network anywhere in a swap ------------------------------

HLE(n_socket_htonl) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    const uint32_t v = (uint32_t)a0;
    return (uint64_t)((v << 24) | ((v << 8) & 0x00ff0000u) | ((v >> 8) & 0x0000ff00u) | (v >> 24));
}

HLE(n_socket_htons) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    const uint32_t v = (uint32_t)a0 & 0xffffu;
    return (uint64_t)(((v << 8) & 0xffffu) | (v >> 8));
}

HLE(n_socket_htonll) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    uint64_t v = a0, out = 0;
    for (int i = 0; i < 8; i++) out = (out << 8) | ((v >> (i * 8)) & 0xffu);
    return out;
}

// --- AF_INET / AF_INET6 text/numeric conversion: computed locally -------------------------------
// Strict dotted-decimal: exactly four 1-3 digit parts, each 0-255, no empty parts, nothing else.

bool pton_v4(const char* text, uint8_t out[4]) {
    if (!text) return false;
    uint8_t parts[4] = {0, 0, 0, 0};
    int part = 0, digits = 0;
    unsigned acc = 0;
    for (const char* p = text;; p++) {
        const char c = *p;
        if (c >= '0' && c <= '9') {
            acc = acc * 10 + (unsigned)(c - '0');
            if (++digits > 3 || acc > 255) return false;
        } else if (c == '.' || c == '\0') {
            if (digits == 0 || part >= 4) return false;
            parts[part++] = (uint8_t)acc;
            acc = 0;
            digits = 0;
            if (c == '\0') break;
        } else {
            return false;
        }
    }
    if (part != 4) return false;
    memcpy(out, parts, 4);
    return true;
}

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// RFC 4291 text form: up to eight 1-4 digit hex groups, at most one "::", and an optional
// dotted-decimal tail filling the last 32 bits.
bool pton_v6(const char* text, uint8_t out[16]) {
    uint8_t tmp[16] = {};
    int tp = 0, colonp = -1, digits = 0;
    unsigned val = 0;
    bool saw_xdigit = false;
    const char* p = text;
    if (*p == ':' && *++p != ':') return false;   // a leading ':' must be "::"
    const char* curtok = p;
    for (char ch; (ch = *p++) != '\0';) {
        const int d = hex_digit(ch);
        if (d >= 0) {
            if (++digits > 4) return false;
            val = (val << 4) | (unsigned)d;
            saw_xdigit = true;
            continue;
        }
        if (ch == ':') {
            curtok = p;
            if (!saw_xdigit) {
                if (colonp >= 0) return false;   // a second "::"
                colonp = tp;
                continue;
            }
            if (*p == '\0' || tp + 2 > 16) return false;
            tmp[tp++] = (uint8_t)(val >> 8);
            tmp[tp++] = (uint8_t)val;
            saw_xdigit = false;
            val = 0;
            digits = 0;
            continue;
        }
        if (ch == '.' && tp + 4 <= 16 && pton_v4(curtok, tmp + tp)) {
            tp += 4;
            saw_xdigit = false;
            break;
        }
        return false;
    }
    if (saw_xdigit) {
        if (tp + 2 > 16) return false;
        tmp[tp++] = (uint8_t)(val >> 8);
        tmp[tp++] = (uint8_t)val;
    }
    if (colonp >= 0) {
        if (tp == 16) return false;   // "::" must stand for at least one group
        const int moved = tp - colonp;
        for (int i = 1; i <= moved; i++) {
            tmp[16 - i] = tmp[colonp + moved - i];
            tmp[colonp + moved - i] = 0;
        }
        tp = 16;
    }
    if (tp != 16) return false;
    memcpy(out, tmp, 16);
    return true;
}

// Formats into `out` (at least 16 bytes); returns the length without the terminator.
int ntop_v4(const uint8_t addr[4], char* out) {
    int len = 0;
    for (int i = 0; i < 4; i++) {
        const unsigned v = addr[i];
        if (v >= 100) out[len++] = (char)('0' + v / 100);
        if (v >= 10) out[len++] = (char)('0' + (v / 10) % 10);
        out[len++] = (char)('0' + v % 10);
        if (i < 3) out[len++] = '.';
    }
    out[len] = '\0';
    return len;
}

// RFC 5952 / FreeBSD inet_ntop6 form: lowercase hex, no leading zeros, the longest run of two or
// more zero groups (the first, on a tie) as "::", and an IPv4-compatible or IPv4-mapped address
// with its last 32 bits in dotted decimal. Formats into `out` (at least 46 bytes).
int ntop_v6(const uint8_t addr[16], char* out) {
    uint16_t words[8];
    for (int i = 0; i < 8; i++) words[i] = (uint16_t)((addr[2 * i] << 8) | addr[2 * i + 1]);
    int best_base = -1, best_len = 0, cur_base = -1, cur_len = 0;
    for (int i = 0; i < 8; i++) {
        if (words[i] == 0) {
            if (cur_base < 0) cur_base = i, cur_len = 1;
            else cur_len++;
        } else if (cur_base >= 0) {
            if (best_base < 0 || cur_len > best_len) best_base = cur_base, best_len = cur_len;
            cur_base = -1;
        }
    }
    if (cur_base >= 0 && (best_base < 0 || cur_len > best_len)) best_base = cur_base, best_len = cur_len;
    if (best_base >= 0 && best_len < 2) best_base = -1;
    int len = 0;
    for (int i = 0; i < 8; i++) {
        if (best_base >= 0 && i >= best_base && i < best_base + best_len) {
            if (i == best_base) out[len++] = ':';
            continue;
        }
        if (i != 0) out[len++] = ':';
        if (i == 6 && best_base == 0 &&
            (best_len == 6 || (best_len == 5 && words[5] == 0xffff))) {
            len += ntop_v4(addr + 12, out + len);
            return len;
        }
        static const char kHex[] = "0123456789abcdef";
        bool started = false;
        for (int shift = 12; shift >= 0; shift -= 4) {
            const unsigned nib = (words[i] >> shift) & 0xfu;
            if (nib == 0 && !started && shift != 0) continue;
            started = true;
            out[len++] = kHex[nib];
        }
    }
    if (best_base >= 0 && best_base + best_len == 8) out[len++] = ':';
    out[len] = '\0';
    return len;
}

constexpr size_t kInet6AddrStrLen = 46;   // "ffff:ffff:ffff:ffff:ffff:ffff:255.255.255.255\0"

// (af, src, dst) -> 1 converted, 0 bad text (dst untouched, errno untouched), or the facility code
// for a bad family / null pointer with errno recorded (module 0x6aa0: unknown family is
// EAFNOSUPPORT; a null src is EINVAL).
HLE(n_socket_inet_pton) {
    (void)a3;
    (void)a4;
    (void)a5;
    const int32_t af = (int32_t)a0;
    if (af != net::kAfInet && af != net::kAfInet6) return net_id_fail(FreeBsdErrno::EAfNoSupport);
    if (!a1 || !a2) return net_id_fail(FreeBsdErrno::EInval);
    const char* text = reinterpret_cast<const char*>(a1);
    char buf[kInet6AddrStrLen];
    size_t n = 0;
    while (n < sizeof(buf) && text[n]) n++;
    if (n == sizeof(buf)) return net_err(0);   // longer than any valid address: bad text
    memcpy(buf, text, n + 1);
    if (af == net::kAfInet) {
        uint8_t addr[4];
        if (!pton_v4(buf, addr)) return net_err(0);
        memcpy(reinterpret_cast<void*>(a2), addr, sizeof(addr));
    } else {
        uint8_t addr[16];
        if (!pton_v6(buf, addr)) return net_err(0);
        memcpy(reinterpret_cast<void*>(a2), addr, sizeof(addr));
    }
    return net_err(1);
}

// (af, src, dst, size) -> dst on success, NULL on failure with errno recorded. Module 0x63c0:
// an unknown family is EAFNOSUPPORT; a null src or dst is ENOSPC; the text is formatted into a
// local buffer and copied only when it fits, so a short buffer that still holds the text (e.g.
// "1.2.3.4" in 8 bytes) succeeds. size is a 32-bit socklen_t: its upper register bits are ignored.
HLE(n_socket_inet_ntop) {
    (void)a4;
    (void)a5;
    const int32_t af = (int32_t)a0;
    if (af != net::kAfInet && af != net::kAfInet6) {
        g_net_errno = static_cast<int32_t>(FreeBsdErrno::EAfNoSupport);
        return net_err(0);
    }
    if (!a1 || !a2) {
        g_net_errno = static_cast<int32_t>(FreeBsdErrno::ENoSpc);
        return net_err(0);
    }
    const uint32_t size = (uint32_t)a3;
    char text[kInet6AddrStrLen];
    const int len = af == net::kAfInet ? ntop_v4(reinterpret_cast<const uint8_t*>(a1), text)
                                       : ntop_v6(reinterpret_cast<const uint8_t*>(a1), text);
    if ((uint32_t)len + 1u > size) {
        g_net_errno = static_cast<int32_t>(FreeBsdErrno::ENoSpc);
        return net_err(0);   // NULL: the errno slot above carries the reason
    }
    memcpy(reinterpret_cast<void*>(a2), text, (size_t)len + 1);
    return a2;
}

// --- socket id lifecycle -----------------------------------------------------------------------

// (name, family, type, protocol) -> socket id (>0), never 0. `name` is a debug label only.
// An unknown family is EAFNOSUPPORT, as FreeBSD's socreate answers; an unmodelled type keeps
// EOPNOTSUPP (CONFIDENCE: LOW -- FreeBSD would say EPROTOTYPE/EPROTONOSUPPORT; any of them is a
// refusal, never a handle).
HLE(n_socket_create) {
    (void)a0;
    (void)a3;
    (void)a4;
    (void)a5;
    const int32_t domain = (int32_t)a1, type = (int32_t)a2;
    if (domain != net::kAfUnix && domain != net::kAfInet && domain != net::kAfInet6)
        return net_id_fail(FreeBsdErrno::EAfNoSupport);
    if (type != net::kSockStream && type != net::kSockDgram)
        return net_id_fail(FreeBsdErrno::EOpNotSupp);
    std::lock_guard<std::mutex> lk(g_net_mx);
    for (int i = 1; i <= kMaxNetSockets; i++) {
        if (g_net_live[i]) continue;
        g_net_live[i] = true;
        return (uint64_t)i;
    }
    return net_id_fail(FreeBsdErrno::EMFile);
}

HLE(n_socket_close) {   // (fd) -> SCE_OK, releasing the id
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_net_mx);
    if (!net_known_locked((int32_t)a0)) return net_fail(FreeBsdErrno::EBadF);
    g_net_live[(int32_t)a0] = false;
    return net_err(0);
}

// --- the network boundary: fail, and leave every out-parameter untouched -----------------------
// None of these writes to a guest buffer even on the failure path: a recv that answered failure
// while scribbling the caller's buffer would hand it content nobody sent.

// connect/send/sendto/recv/recvfrom/sendmsg/bind/listen. ENETUNREACH is chosen for every arm,
// including bind/listen where FreeBSD would succeed offline -- CONFIDENCE: LOW on that errno.
HLE(n_socket_offline) {
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_net_mx);
    if (!net_known_locked((int32_t)a0)) return net_fail(FreeBsdErrno::EBadF);
    return net_fail(FreeBsdErrno::ENetUnreach);
}

HLE(n_socket_accept) {   // (fd, addr, addrlen) -> socket id; no backlog can ever complete
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_net_mx);
    if (!net_known_locked((int32_t)a0)) return net_id_fail(FreeBsdErrno::EBadF);
    return net_id_fail(FreeBsdErrno::ENetUnreach);
}

HLE(n_socket_opt_unsupported) {   // setsockopt/getsockopt: prosper models no option
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_net_mx);
    if (!net_known_locked((int32_t)a0)) return net_fail(FreeBsdErrno::EBadF);
    return net_fail(FreeBsdErrno::EOpNotSupp);
}

HLE(n_socket_not_conn) {   // getpeername/getsockname/shutdown: nothing ever connected
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_net_mx);
    if (!net_known_locked((int32_t)a0)) return net_fail(FreeBsdErrno::EBadF);
    return net_fail(FreeBsdErrno::ENotConn);
}

HLE(n_socket_errno_loc) {   // () -> &errno; the slot the failing arms above keep current
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    return (uint64_t)(uintptr_t)&g_net_errno;
}

// --- epoll, introspection, pools, resolver: local lifecycle where allocation is local ---------
// Epoll and resolver ids live in their own small tables: an epoll id must never validate as a
// socket and vice versa, so sharing g_net_live would let a stale socket id arm an epoll wait.
// All of these NIDs are in the 3.20 libSceNet export set. Argument checks and return contracts
// follow the shipped libSceNet module (image-relative offsets cited per handler).
constexpr int kMaxNetEpoll = 32;
constexpr int kMaxNetResolver = 32;
bool g_net_epoll_live[kMaxNetEpoll + 1];   // 1-based; caller holds g_net_mx
bool g_net_resolver_live[kMaxNetResolver + 1];   // 1-based; caller holds g_net_mx

HLE(n_epoll_create) {   // (name, flags) -> epoll id (>0), never 0
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    // The module requires a name and zero flags (+0x809f-+0x80b8); anything else is EINVAL.
    if (!a0 || (uint32_t)a1 != 0) return net_id_fail(FreeBsdErrno::EInval);
    std::lock_guard<std::mutex> lk(g_net_mx);
    for (int i = 1; i <= kMaxNetEpoll; i++) {
        if (g_net_epoll_live[i]) continue;
        g_net_epoll_live[i] = true;
        return (uint64_t)i;
    }
    return net_id_fail(FreeBsdErrno::EMFile);
}
HLE(n_epoll_destroy) {   // (eid) -> SCE_OK, releasing the id
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_net_mx);
    const int32_t eid = (int32_t)a0;
    if (eid < 1 || eid > kMaxNetEpoll || !g_net_epoll_live[eid])
        return net_fail(FreeBsdErrno::EBadF);
    g_net_epoll_live[eid] = false;
    return net_err(0);
}
HLE(n_epoll_control) {   // (eid, op, id, event*) -> SCE_OK; nothing is ever delivered offline
    (void)a4;
    (void)a5;
    // The module validates the request before the ids: op must be ADD 1 / MOD 2 / DEL 3
    // (+0x824d-+0x8268), and ADD/MOD need a non-null event with a non-zero events mask
    // (+0x826e-+0x827c, +0x8407-+0x8415). CONFIDENCE: MED that the mask is the event's
    // leading 32-bit word.
    const int32_t op = (int32_t)a1;
    if (op < 1 || op > 3) return net_fail(FreeBsdErrno::EInval);
    if (op != 3) {
        if (!a3) return net_fail(FreeBsdErrno::EInval);
        uint32_t mask = 0;
        memcpy(&mask, reinterpret_cast<const void*>(a3), sizeof mask);
        if (mask == 0) return net_fail(FreeBsdErrno::EInval);
    }
    std::lock_guard<std::mutex> lk(g_net_mx);
    const int32_t eid = (int32_t)a0;
    if (eid < 1 || eid > kMaxNetEpoll || !g_net_epoll_live[eid])
        return net_fail(FreeBsdErrno::EBadF);
    if (!net_known_locked((int32_t)a2)) return net_fail(FreeBsdErrno::EBadF);
    return net_err(0);
}
// (eid, events*, maxevents, timeout) -> number of events. The module checks the buffer and
// count before the id (+0x87d5-+0x8846, before the kevent at +0x88be) and takes the timeout in
// MICROSECONDS (+0x880d-+0x8834); a negative timeout blocks (+0x8880, outer retry loop
// +0x4e94-+0x4f42). Offline nothing can become ready, so this answers 0 events -- after a bounded
// sleep outside the lock, so a guest looping on a blocking wait cannot busy-spin (the same model
// as sceHttpWaitRequest in hle_http.cpp). A zero timeout returns at once. The result is a count,
// so errors are sign-extended. CONFIDENCE: LOW on capping a negative (blocking) timeout at 50 ms.
HLE(n_epoll_wait) {
    (void)a4;
    (void)a5;
    if (!a1 || (int32_t)a2 <= 0) return net_id_fail(FreeBsdErrno::EInval);
    {
        std::lock_guard<std::mutex> lk(g_net_mx);
        const int32_t eid = (int32_t)a0;
        if (eid < 1 || eid > kMaxNetEpoll || !g_net_epoll_live[eid])
            return net_id_fail(FreeBsdErrno::EBadF);
    }
    constexpr int64_t kMaxWaitUs = 50'000;
    const int64_t timeout = (int32_t)a3;
    if (timeout == 0) return net_err(0);
    const int64_t wait_us = timeout > 0 ? std::min(timeout, kMaxWaitUs) : kMaxWaitUs;
    std::this_thread::sleep_for(std::chrono::microseconds(wait_us));
    return net_err(0);
}
// (ether*, str*, len) -> int status. The export (+0x3340) wraps +0x53a0: a null ether, a null
// str or len < 18 (a 64-bit compare, before any formatting) fails with EINVAL / errno 22;
// otherwise it formats "%02x:%02x:%02x:%02x:%02x:%02x" (format string at +0x393d4) and returns
// 0. CONFIDENCE: HIGH.
HLE(n_ether_ntostr) {
    (void)a3;
    (void)a4;
    (void)a5;
    if (!a0 || !a1 || a2 < 18) return net_fail(FreeBsdErrno::EInval);
    const uint8_t* m = reinterpret_cast<const uint8_t*>(a0);
    std::snprintf(reinterpret_cast<char*>(a1), (size_t)a2, "%02x:%02x:%02x:%02x:%02x:%02x", m[0],
                  m[1], m[2], m[3], m[4], m[5]);
    return net_err(0);
}
HLE(n_get_mac_address) {   // (addr*, flags): no interface to report; fail, don't fabricate
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    // A zeroed MAC would be a manufactured hardware identity (and LanDiscovery-shaped code
    // would then proceed to a connect that cannot work); EOPNOTSUPP keeps it on error paths.
    // CONFIDENCE: LOW on the errno; HIGH that failing beats a fabricated address. The module
    // first refuses a null addr or non-zero flags with EINVAL (+0x542c-+0x5438).
    if (!a0 || (uint32_t)a1 != 0) return net_fail(FreeBsdErrno::EInval);
    return net_fail(FreeBsdErrno::EOpNotSupp);
}
// (s, info*, n, flags) -> count of records. flags & 0x31000 is EINVAL before the socket is
// looked at (+0x4170-+0x417d); the result is a count (+0x7213), so errors are sign-extended.
HLE(n_get_sock_info) {   // unknown struct: fail, don't scribble
    (void)a1;
    (void)a2;
    (void)a4;
    (void)a5;
    if ((uint32_t)a3 & 0x31000u) return net_id_fail(FreeBsdErrno::EInval);
    std::lock_guard<std::mutex> lk(g_net_mx);
    if (!net_known_locked((int32_t)a0)) return net_id_fail(FreeBsdErrno::EBadF);
    return net_id_fail(FreeBsdErrno::EOpNotSupp);
}
HLE(n_pool_destroy) {   // (memid): pools are untracked counter ids (see PoolCreate) -> SCE_OK
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    return net_err(0);
}
HLE(n_resolver_create) {   // (name, memid, flags) -> resolver id (>0), never 0
    (void)a1;
    (void)a3;
    (void)a4;
    (void)a5;
    // Non-zero flags are EINVAL (+0xbae0).
    if (!a0 || (uint32_t)a2 != 0) return net_id_fail(FreeBsdErrno::EInval);
    std::lock_guard<std::mutex> lk(g_net_mx);
    for (int i = 1; i <= kMaxNetResolver; i++) {
        if (g_net_resolver_live[i]) continue;
        g_net_resolver_live[i] = true;
        return (uint64_t)i;
    }
    return net_id_fail(FreeBsdErrno::EMFile);
}
// (rid, hostname*, addr*, timeout, retry, flags) -> NetUnreach: no DNS offline. The module
// refuses a negative timeout or retry and flags & 0xfffefffe with EINVAL (+0xbb11-+0xbb2c).
// ENETUNREACH is chosen over libSceNet's resolver-specific codes (0x804101dc..0x804101ec)
// because "no route to a DNS server" is the honest offline answer and is what the socket
// boundary already reports. CONFIDENCE: LOW on that choice.
HLE(n_resolver_start_ntoa) {
    (void)a2;
    if ((int32_t)a3 < 0 || (int32_t)a4 < 0 || ((uint32_t)a5 & 0xfffefffeu))
        return net_fail(FreeBsdErrno::EInval);
    std::lock_guard<std::mutex> lk(g_net_mx);
    const int32_t rid = (int32_t)a0;
    if (rid < 1 || rid > kMaxNetResolver || !g_net_resolver_live[rid])
        return net_fail(FreeBsdErrno::EBadF);
    if (!a1) return net_fail(FreeBsdErrno::EInval);
    return net_fail(FreeBsdErrno::ENetUnreach);
}
HLE(n_resolver_destroy) {   // (rid) -> SCE_OK, freeing the slot; an unknown id is EBADF
    (void)a1;
    (void)a2;
    (void)a3;
    (void)a4;
    (void)a5;
    std::lock_guard<std::mutex> lk(g_net_mx);
    const int32_t rid = (int32_t)a0;
    if (rid < 1 || rid > kMaxNetResolver || !g_net_resolver_live[rid])
        return net_fail(FreeBsdErrno::EBadF);
    g_net_resolver_live[rid] = false;
    return net_err(0);
}

}   // namespace

void register_net_hle() {
    // Offline-computable: real answers, no network anywhere in the computation.
    Hle::register_fn("9T2pDF2Ryqg", (HleFn)n_socket_htonl, "sceNetHtonl");
    Hle::register_fn("iWQWrwiSt8A", (HleFn)n_socket_htons, "sceNetHtons");
    Hle::register_fn("pQGpHYopAIY", (HleFn)n_socket_htonl, "sceNetNtohl");
    Hle::register_fn("Rbvt+5Y2iEw", (HleFn)n_socket_htons, "sceNetNtohs");
    Hle::register_fn("3CHi1K1wsCQ", (HleFn)n_socket_htonll, "sceNetHtonll");
    Hle::register_fn("tOrRi-v3AOM", (HleFn)n_socket_htonll, "sceNetNtohll");
    Hle::register_fn("8Kcp5d-q1Uo", (HleFn)n_socket_inet_pton, "sceNetInetPton");
    Hle::register_fn("9vA2aW+CHuA", (HleFn)n_socket_inet_ntop, "sceNetInetNtop");

    // Local id lifecycle: real ids, never 0.
    Hle::register_fn("Q4qBuN-c0ZM", (HleFn)n_socket_create, "sceNetSocket");
    Hle::register_fn("45ggEzakPJQ", (HleFn)n_socket_close, "sceNetSocketClose");
    Hle::register_fn("HQOwnfMGipQ", (HleFn)n_socket_errno_loc, "sceNetErrnoLoc");

    // The network boundary: fail loudly, out-parameters untouched.
    Hle::register_fn("OXXX4mUk3uk", (HleFn)n_socket_offline, "sceNetConnect");
    Hle::register_fn("beRjXBn-z+o", (HleFn)n_socket_offline, "sceNetSend");
    Hle::register_fn("9wO9XrMsNhc", (HleFn)n_socket_offline, "sceNetRecv");
    Hle::register_fn("gvD1greCu0A", (HleFn)n_socket_offline, "sceNetSendto");
    Hle::register_fn("304ooNZxWDY", (HleFn)n_socket_offline, "sceNetRecvfrom");
    Hle::register_fn("2eKbgcboJso", (HleFn)n_socket_offline, "sceNetSendmsg");
    Hle::register_fn("bErx49PgxyY", (HleFn)n_socket_offline, "sceNetBind");
    Hle::register_fn("kOj1HiAGE54", (HleFn)n_socket_offline, "sceNetListen");
    Hle::register_fn("PIWqhn9oSxc", (HleFn)n_socket_accept, "sceNetAccept");
    Hle::register_fn("2mKX2Spso7I", (HleFn)n_socket_opt_unsupported, "sceNetSetsockopt");
    Hle::register_fn("xphrZusl78E", (HleFn)n_socket_opt_unsupported, "sceNetGetsockopt");
    Hle::register_fn("TCkRD0DWNLg", (HleFn)n_socket_not_conn, "sceNetGetpeername");
    Hle::register_fn("hoOAofhhRvE", (HleFn)n_socket_not_conn, "sceNetGetsockname");
    Hle::register_fn("TSM6whtekok", (HleFn)n_socket_not_conn, "sceNetShutdown");
    // Epoll, introspection, pools, resolver: local lifecycle where allocation is local (never
    // 0 from an id contract), loud failure where an answer needs hardware or a network.
    Hle::register_fn("SF47kB2MNTo", (HleFn)n_epoll_create, "sceNetEpollCreate");
    Hle::register_fn("Inp1lfL+Jdw", (HleFn)n_epoll_destroy, "sceNetEpollDestroy");
    Hle::register_fn("ZVw46bsasAk", (HleFn)n_epoll_control, "sceNetEpollControl");
    Hle::register_fn("drjIbDbA7UQ", (HleFn)n_epoll_wait, "sceNetEpollWait");
    Hle::register_fn("v6M4txecCuo", (HleFn)n_ether_ntostr, "sceNetEtherNtostr");
    Hle::register_fn("6Oc0bLsIYe0", (HleFn)n_get_mac_address, "sceNetGetMacAddress");
    Hle::register_fn("hLuXdjHnhiI", (HleFn)n_get_sock_info, "sceNetGetSockInfo");
    Hle::register_fn("K7RlrTkI-mw", (HleFn)n_pool_destroy, "sceNetPoolDestroy");
    Hle::register_fn("C4UgDHHPvdw", (HleFn)n_resolver_create, "sceNetResolverCreate");
    Hle::register_fn("Nd91WaWmG2w", (HleFn)n_resolver_start_ntoa, "sceNetResolverStartNtoa");
    Hle::register_fn("kJlYH5uMAWI", (HleFn)n_resolver_destroy, "sceNetResolverDestroy");
}

}   // namespace prosper
