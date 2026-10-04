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
//   sceNetInetPton/sceNetInetNtop compute AF_INET dotted-decimal both ways.
// - Socket ids are a real local table (1-based; slot 0 is never handed out, exactly like the HTTP
//   object table): sceNetSocket allocates, sceNetSocketClose releases. An id-returning entry
//   point sign-extends its error, so a guest reading the answer as int32 OR int64 sees it as
//   negative and never as a handle.
// - Everything that would move a packet -- connect, send, recv, bind, listen, accept, socket
//   options, peer names, shutdown -- FAILS with the libSceNet facility error for an unreachable
//   network (net::kNetErrorNetUnreach, the same answer the HTTP send paths give) and leaves every
//   out-parameter untouched. A bad socket id answers EBADF instead, so use-after-close is also
//   loud. CONFIDENCE: HIGH on the facility encoding (sce_net_errors.hpp); LOW on each exact errno
//   choice, which is why every arm names the errno it encodes.
// - sceNetErrnoLoc returns the address of a thread-local FreeBSD errno slot that the failing arms
//   above keep current, so a guest that reads errno after a failed call sees the failure prosper
//   reported. CONFIDENCE: MED -- per-host-thread is the closest this in-process design gets to the
//   console's per-guest-thread slot (cf. the fiber caveat in docs/games/UNCHARTED_STATUS.md).
//
// Behaviour is re-derived for prosper's own table design; KytyPS5 (src/libs/network.cpp), shadPS4
// (src/core/libraries/network/) and sharpemu (SharpEmu.Libs/Network/) were read as hypotheses
// only, never copied (CONTRIBUTING.md).
#include "hle/net/hle_net.hpp"
#include "hle/net/sce_net_errors.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstring>
#include <mutex>

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

// --- AF_INET text/numeric conversion: computed locally ------------------------------------------
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

HLE(n_socket_inet_pton) {  // (af, src, dst) -> 1 ok, 0 bad text, -1 bad af (errno recorded)
    (void)a3;
    (void)a4;
    (void)a5;
    if ((int32_t)a0 != net::kAfInet) return net_id_fail(FreeBsdErrno::EInval);
    if (!a1 || !a2) return net_id_fail(FreeBsdErrno::EInval);
    const char* text = reinterpret_cast<const char*>(a1);
    char buf[16];
    size_t n = 0;
    while (n < sizeof(buf) && text[n]) n++;
    if (n == sizeof(buf)) return net_err(0);   // unterminated within 16 bytes: bad text, no errno
    memcpy(buf, text, n + 1);
    uint8_t addr[4];
    if (!pton_v4(buf, addr)) return net_err(0);
    memcpy(reinterpret_cast<void*>(a2), addr, 4);
    g_net_errno = 0;
    return net_err(1);
}

HLE(n_socket_inet_ntop) {   // (af, src, dst, size) -> dst on ok, NULL on failure
    (void)a4;
    (void)a5;
    if ((int32_t)a0 != net::kAfInet) {
        g_net_errno = static_cast<int32_t>(FreeBsdErrno::EInval);
        return net_err(0);
    }
    if (!a1 || !a2) {
        g_net_errno = static_cast<int32_t>(FreeBsdErrno::EInval);
        return net_err(0);
    }
    if (a3 < 16) {   // INET_ADDRSTRLEN: "255.255.255.255\0"
        g_net_errno = static_cast<int32_t>(FreeBsdErrno::ENoSpc);
        return net_err(0);   // NULL: the errno slot above carries the reason
    }
    const uint8_t* addr = reinterpret_cast<const uint8_t*>(a1);
    char* dst = reinterpret_cast<char*>(a2);
    // snprintf-free: the longest part is 3 digits, the format is fixed.
    int len = 0;
    for (int i = 0; i < 4; i++) {
        unsigned v = addr[i];
        char part[4];
        int plen = 0;
        if (v >= 100) part[plen++] = (char)('0' + v / 100);
        if (v >= 10) part[plen++] = (char)('0' + (v / 10) % 10);
        part[plen++] = (char)('0' + v % 10);
        if (len + plen + (i < 3 ? 1 : 1) > 16) {
            g_net_errno = static_cast<int32_t>(FreeBsdErrno::ENoSpc);
            return net_err(0);   // NULL: the errno slot above carries the reason
        }
        memcpy(dst + len, part, (size_t)plen);
        len += plen;
        dst[len++] = (i < 3) ? '.' : '\0';
    }
    g_net_errno = 0;
    return a2;
}

// --- socket id lifecycle -----------------------------------------------------------------------

HLE(n_socket_create) {   // (domain, type, protocol) -> socket id (>0), never 0
    (void)a3;
    (void)a4;
    (void)a5;
    const int32_t domain = (int32_t)a0, type = (int32_t)a1;
    const bool dom_ok = domain == net::kAfUnix || domain == net::kAfInet || domain == net::kAfInet6;
    const bool type_ok = type == net::kSockStream || type == net::kSockDgram;
    if (!dom_ok || !type_ok) return net_id_fail(FreeBsdErrno::EOpNotSupp);
    std::lock_guard<std::mutex> lk(g_net_mx);
    for (int i = 1; i <= kMaxNetSockets; i++) {
        if (g_net_live[i]) continue;
        g_net_live[i] = true;
        g_net_errno = 0;
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
    g_net_errno = 0;
    return net_err(0);
}

// --- the network boundary: fail, and leave every out-parameter untouched -----------------------
// None of these writes to a guest buffer even on the failure path: a recv that answered failure
// while scribbling the caller's buffer would hand it content nobody sent.

HLE(n_socket_offline) {   // connect/send/sendto/recv/recvfrom/sendmsg/bind/listen
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
}

}   // namespace prosper
