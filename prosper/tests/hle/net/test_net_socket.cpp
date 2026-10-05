// test_net_socket - libSceNet socket family: real ids and byte order offline, loud failure
// on the packet path.
//
// A socket import prosper does not register returns the dispatcher's default 0, which is SCE_OK:
// the guest is told a connect/send/recv succeeded that never ran, and a recv buffer keeps
// whatever was already there. Every assertion below dies to a specific wrong implementation;
// the mutation each arm kills is named on the arm.
#include "hle/dispatch/dispatch.hpp"
#include "hle/net/hle_net.hpp"
#include "hle/net/sce_net_errors.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <gtest/gtest.h>

using namespace prosper;

namespace {

// The guest-visible NIDs, from the PS5 3.20 export tables (see tools/nid_census --self-check).
constexpr const char* kSocket = "Q4qBuN-c0ZM";
constexpr const char* kSocketClose = "45ggEzakPJQ";
constexpr const char* kConnect = "OXXX4mUk3uk";
constexpr const char* kSend = "beRjXBn-z+o";
constexpr const char* kRecv = "9wO9XrMsNhc";
constexpr const char* kSendto = "gvD1greCu0A";
constexpr const char* kRecvfrom = "304ooNZxWDY";
constexpr const char* kSendmsg = "2eKbgcboJso";
constexpr const char* kBind = "bErx49PgxyY";
constexpr const char* kListen = "kOj1HiAGE54";
constexpr const char* kAccept = "PIWqhn9oSxc";
constexpr const char* kSetsockopt = "2mKX2Spso7I";
constexpr const char* kGetsockopt = "xphrZusl78E";
constexpr const char* kHtonl = "9T2pDF2Ryqg";
constexpr const char* kHtons = "iWQWrwiSt8A";
constexpr const char* kNtohl = "pQGpHYopAIY";
constexpr const char* kNtohs = "Rbvt+5Y2iEw";
constexpr const char* kHtonll = "3CHi1K1wsCQ";
constexpr const char* kNtohll = "tOrRi-v3AOM";
constexpr const char* kInetPton = "8Kcp5d-q1Uo";
constexpr const char* kInetNtop = "9vA2aW+CHuA";
constexpr const char* kErrnoLoc = "HQOwnfMGipQ";
constexpr const char* kGetpeername = "TCkRD0DWNLg";
constexpr const char* kGetsockname = "hoOAofhhRvE";
constexpr const char* kShutdown = "TSM6whtekok";

constexpr uint64_t kUnreach = net::kNetErrorNetUnreach;  // 0x80410133
constexpr uint64_t kBadF = net::kNetErrorBadF;           // 0x80410109

uint64_t open_socket() {
    HleFn fn = Hle::lookup(kSocket);
    EXPECT_NE(fn, nullptr) << "sceNetSocket registered (kills: unregistered NID -> dispatch 0)";
    if (!fn) return 0;
    // sceNetSocket(name, family, type, protocol): the first argument is a debug-name string.
    const uint64_t fd = fn((uint64_t)"prosper-test", 2, 1, 0, 0, 0);   // AF_INET, SOCK_STREAM
    EXPECT_GT((int64_t)fd, 0) << "socket id is positive, never the dispatcher 0";
    return fd;
}

void close_socket(uint64_t fd) {
    HleFn fn = Hle::lookup(kSocketClose);
    ASSERT_NE(fn, nullptr) << "sceNetSocketClose registered";
    EXPECT_EQ(fn(fd, 0, 0, 0, 0, 0), 0u) << "close of a live socket is SCE_OK";
}

}  // namespace

TEST(NetSocket, Registration) {
    register_builtin_hle();
    const char* nids[] = {kSocket,   kSocketClose, kConnect,     kSend,        kRecv,
                          kSendto,   kRecvfrom,    kSendmsg,     kBind,        kListen,
                          kAccept,   kSetsockopt,  kGetsockopt,  kHtonl,       kHtons,
                          kNtohl,    kNtohs,       kHtonll,      kNtohll,      kInetPton,
                          kInetNtop, kErrnoLoc,    kGetpeername, kGetsockname, kShutdown};
    for (const char* nid : nids) {
        EXPECT_NE(Hle::lookup(nid), nullptr)
            << "NID " << nid << " registered (kills: false-success 0)";
    }
    EXPECT_STREQ(Hle::name_of(kSocket), "sceNetSocket") << "name table answers the owning symbol";
    EXPECT_STREQ(Hle::name_of(kHtonl), "sceNetHtonl") << "name table answers the owning symbol";
}

TEST(NetSocket, ByteOrder) {
    register_builtin_hle();
    EXPECT_EQ(Hle::lookup(kHtonl)(0x01020304, 0, 0, 0, 0, 0), 0x04030201u)
        << "htonl swaps four bytes (kills: identity stub)";
    EXPECT_EQ(Hle::lookup(kHtons)(0x0102, 0, 0, 0, 0, 0), 0x0201u) << "htons swaps two bytes";
    EXPECT_EQ(Hle::lookup(kHtonll)(0x0102030405060708ull, 0, 0, 0, 0, 0), 0x0807060504030201ull)
        << "htonll swaps eight bytes";
    EXPECT_EQ(Hle::lookup(kNtohl)(0x04030201, 0, 0, 0, 0, 0), 0x01020304u)
        << "ntohl is the same involution (kills: separate unregistered NID)";
    EXPECT_EQ(Hle::lookup(kNtohs)(0x0201, 0, 0, 0, 0, 0), 0x0102u) << "ntohs round-trips htons";
    EXPECT_EQ(Hle::lookup(kNtohll)(0x0807060504030201ull, 0, 0, 0, 0, 0), 0x0102030405060708ull)
        << "ntohll round-trips htonll";
}

TEST(NetSocket, Lifecycle) {
    register_builtin_hle();
    const uint64_t a = open_socket(), b = open_socket();
    ASSERT_GT((int64_t)a, 0);
    ASSERT_GT((int64_t)b, 0);
    EXPECT_NE(a, b) << "two sockets are distinct ids (kills: constant-1 stub)";
    close_socket(a);
    close_socket(b);
    // Kills: a close that always answers SCE_OK -- the second close must refuse.
    EXPECT_EQ(Hle::lookup(kSocketClose)(a, 0, 0, 0, 0, 0), kBadF)
        << "double close answers EBADF, not a second SCE_OK";
    EXPECT_EQ(Hle::lookup(kSocketClose)(0, 0, 0, 0, 0, 0), kBadF)
        << "close(0) is EBADF: 0 was never a socket (kills: success-on-garbage)";
    // Kills: an id-returning error that is zero-extended -- reads as a handle in int64.
    const uint64_t bad = Hle::lookup(kSocket)((uint64_t)"t", 99, 1, 0, 0, 0);
    EXPECT_LT((int64_t)bad, 0) << "bad-domain socket answers negative in int64 too";
    EXPECT_EQ((uint32_t)bad, net::kNetErrorAfNoSupport) << "bad family is EAFNOSUPPORT";
    EXPECT_NE(bad, 0u) << "an id-returning failure is never the dispatcher 0";
    EXPECT_EQ((uint32_t)Hle::lookup(kSocket)((uint64_t)"t", 2, 5, 0, 0, 0),
              net::kNetErrorOpNotSupp)
        << "unmodelled socket type is refused";
}

TEST(NetSocket, SocketTakesANameFirst) {
    // sceNetSocket(const char* name, int family, int type, int protocol) -- module entry 0x7330
    // keeps rdi as a pointer and esi/edx/ecx as the triple. Kills: reading the family from arg0.
    register_builtin_hle();
    HleFn fn = Hle::lookup(kSocket);
    ASSERT_NE(fn, nullptr);
    static const char kName[] = "unity-player-connection";
    const uint64_t fd = fn((uint64_t)kName, 2, 1, 6, 0, 0);   // AF_INET, SOCK_STREAM, TCP
    EXPECT_GT((int64_t)fd, 0) << "a real name pointer in arg0 still yields a socket id";
    close_socket(fd);
    // The old (family, type, protocol) layout puts AF_INET's 2 where the type goes: refused.
    EXPECT_LT((int64_t)fn(2, 1, 0, 0, 0, 0), 0)
        << "(family, type, protocol) without a name is not the contract";
}

TEST(NetSocket, OfflineFailure) {
    register_builtin_hle();
    const uint64_t fd = open_socket();
    ASSERT_GT((int64_t)fd, 0);
    const char* senders[] = {kConnect, kSend, kSendto, kSendmsg, kBind, kListen};
    for (const char* nid : senders) {
        EXPECT_EQ(Hle::lookup(nid)(fd, 0, 0, 0, 0, 0), kUnreach)
            << "NID " << nid << " fails ENETUNREACH offline (kills: false-success 0)";
    }
    // Recv with a sentinel buffer: failure must leave the caller's bytes untouched.
    unsigned char buf[64];
    memset(buf, 0xA5, sizeof(buf));
    EXPECT_EQ(Hle::lookup(kRecv)(fd, (uint64_t)buf, sizeof(buf), 0, 0, 0), kUnreach)
        << "recv fails ENETUNREACH (kills: SCE_OK with nothing received)";
    bool intact = true;
    for (unsigned char c : buf) intact &= (c == 0xA5);
    EXPECT_TRUE(intact) << "failed recv writes nothing (kills: success-with-stale-buffer)";
    memset(buf, 0xA5, sizeof(buf));
    EXPECT_EQ(Hle::lookup(kRecvfrom)(fd, (uint64_t)buf, sizeof(buf), 0, 0, 0), kUnreach)
        << "recvfrom fails ENETUNREACH";
    intact = true;
    for (unsigned char c : buf) intact &= (c == 0xA5);
    EXPECT_TRUE(intact) << "failed recvfrom writes nothing";
    // Unknown ids fail differently from known-but-offline ones.
    EXPECT_EQ(Hle::lookup(kConnect)(0x7fff, 0, 0, 0, 0, 0), kBadF)
        << "connect on a dead id is EBADF, not ENETUNREACH (kills: errno-blind stub)";
    // Accept is id-returning: its offline error must be sign-extended, never a handle.
    const uint64_t acc = Hle::lookup(kAccept)(fd, 0, 0, 0, 0, 0);
    EXPECT_LT((int64_t)acc, 0) << "accept offline is negative in int64 (kills: zero-extended id)";
    EXPECT_EQ((uint32_t)acc, (uint32_t)kUnreach) << "accept offline encodes ENETUNREACH";
    close_socket(fd);
}

TEST(NetSocket, SentinelDiscriminatorPositiveControl) {
    // The "buffer untouched" assertions above are only as good as their sentinel check: prove the
    // discriminator fires by mutating one byte outside the harness and confirming it reports.
    unsigned char buf[8];
    memset(buf, 0xA5, sizeof(buf));
    buf[3] = 0x00;
    bool intact = true;
    for (unsigned char c : buf) intact &= (c == 0xA5);
    EXPECT_FALSE(intact) << "positive control: the sentinel check detects a written byte";
}

TEST(NetSocket, OptionsAndNamesFailLoud) {
    register_builtin_hle();
    const uint64_t fd = open_socket();
    ASSERT_GT((int64_t)fd, 0);
    EXPECT_EQ(Hle::lookup(kSetsockopt)(fd, 0, 0, 0, 0, 0), (uint64_t)net::kNetErrorOpNotSupp)
        << "setsockopt models no option: EOPNOTSUPP, not success (kills: blind SCE_OK)";
    EXPECT_EQ(Hle::lookup(kGetsockopt)(fd, 0, 0, 0, 0, 0), (uint64_t)net::kNetErrorOpNotSupp)
        << "getsockopt models no option";
    EXPECT_EQ(Hle::lookup(kGetpeername)(fd, 0, 0, 0, 0, 0), (uint64_t)net::kNetErrorNotConn)
        << "getpeername with nothing connected is ENOTCONN";
    EXPECT_EQ(Hle::lookup(kGetsockname)(fd, 0, 0, 0, 0, 0), (uint64_t)net::kNetErrorNotConn)
        << "getsockname with nothing bound is ENOTCONN";
    EXPECT_EQ(Hle::lookup(kShutdown)(fd, 0, 0, 0, 0, 0), (uint64_t)net::kNetErrorNotConn)
        << "shutdown with nothing connected is ENOTCONN";
    EXPECT_EQ(Hle::lookup(kSetsockopt)(0x7fff, 0, 0, 0, 0, 0), kBadF)
        << "setsockopt on a dead id is EBADF";
    close_socket(fd);
}

TEST(NetSocket, ErrnoLocTracksFailures) {
    register_builtin_hle();
    HleFn loc = Hle::lookup(kErrnoLoc);
    ASSERT_NE(loc, nullptr) << "sceNetErrnoLoc registered";
    const uint64_t p = loc(0, 0, 0, 0, 0, 0);
    EXPECT_NE(p, 0u) << "errno location is never NULL (kills: zero-extended failure)";
    // A successful call never writes errno (module: the success paths skip the store).
    *reinterpret_cast<int32_t*>(p) = 77;
    const uint64_t fd = open_socket();
    ASSERT_GT((int64_t)fd, 0);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(p), 77)
        << "a successful socket() leaves errno as it was (kills: clearing errno on success)";
    Hle::lookup(kConnect)(fd, 0, 0, 0, 0, 0);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(p), 51)
        << "failed connect records ENETUNREACH (51) where the guest reads errno";
    close_socket(fd);
}

TEST(NetSocket, InetConvert) {
    register_builtin_hle();
    HleFn pton = Hle::lookup(kInetPton), ntop = Hle::lookup(kInetNtop);
    const auto err = [] {
        return *reinterpret_cast<int32_t*>(Hle::lookup(kErrnoLoc)(0, 0, 0, 0, 0, 0));
    };
    uint8_t addr[4] = {0};
    EXPECT_EQ(pton(2, (uint64_t)"192.168.1.1", (uint64_t)addr, 0, 0, 0), 1u)
        << "AF_INET dotted-decimal parses (kills: unimplemented helper)";
    EXPECT_EQ(addr[0], 192) << "pton writes network-order bytes";
    EXPECT_EQ(addr[1], 168) << "pton writes network-order bytes";
    EXPECT_EQ(addr[2], 1) << "pton writes network-order bytes";
    EXPECT_EQ(addr[3], 1) << "pton writes network-order bytes";
    EXPECT_EQ(pton(2, (uint64_t)"999.1.1.1", (uint64_t)addr, 0, 0, 0), 0u)
        << "out-of-range part is bad text, not a fill (kills: blind success)";
    EXPECT_EQ(pton(2, (uint64_t)"1.2.3", (uint64_t)addr, 0, 0, 0), 0u) << "three parts is bad text";
    EXPECT_EQ((uint32_t)pton(99, (uint64_t)"1.2.3.4", (uint64_t)addr, 0, 0, 0),
              net::kNetErrorAfNoSupport)
        << "unknown family is EAFNOSUPPORT, not EINVAL (module 0x6aa0)";
    EXPECT_EQ(err(), 47) << "pton records EAFNOSUPPORT (47)";

    char text[16];
    const uint8_t src[4] = {10, 0, 0, 1};
    EXPECT_EQ(ntop(2, (uint64_t)src, (uint64_t)text, 16, 0, 0), (uint64_t)text)
        << "ntop returns dst on success";
    EXPECT_STREQ(text, "10.0.0.1") << "ntop emits dotted-decimal";
    // Fails only when the text does not fit: "1.2.3.4" + NUL is 8 bytes.
    const uint8_t small[4] = {1, 2, 3, 4};
    char eight[8];
    EXPECT_EQ(ntop(2, (uint64_t)small, (uint64_t)eight, 8, 0, 0), (uint64_t)eight)
        << "a buffer that holds the text succeeds (kills: a blanket size < 16 refusal)";
    EXPECT_STREQ(eight, "1.2.3.4");
    memset(text, 0x5A, sizeof(text));
    EXPECT_EQ(ntop(2, (uint64_t)src, (uint64_t)text, 8, 0, 0), 0u)
        << "\"10.0.0.1\" needs 9 bytes: NULL, not a truncation (kills: silent truncation)";
    EXPECT_EQ(err(), 28) << "short buffer records ENOSPC (28) where the guest reads errno";
    EXPECT_EQ((unsigned char)text[0], 0x5Au) << "a refused ntop writes nothing";
    // size is a 32-bit socklen_t: garbage above it must not widen it.
    EXPECT_EQ(ntop(2, (uint64_t)src, (uint64_t)text, 0xFFFFFFFF00000004ull, 0, 0), 0u)
        << "upper register bits are ignored (kills: comparing the full 64-bit size)";
    EXPECT_EQ((unsigned char)text[0], 0x5Au) << "and nothing was written";
    EXPECT_EQ(ntop(2, (uint64_t)src, 0, 16, 0, 0), 0u) << "null dst is refused";
    EXPECT_EQ(err(), 28) << "null dst records ENOSPC (module 0x63c0), not EINVAL";
    EXPECT_EQ(ntop(99, (uint64_t)src, (uint64_t)text, 16, 0, 0), 0u) << "unknown family refused";
    EXPECT_EQ(err(), 47) << "ntop unknown family records EAFNOSUPPORT (47)";
}

TEST(NetSocket, Inet6Convert) {
    // The module supports AF_INET6 (28) both ways; it is offline-computable, so it is implemented.
    register_builtin_hle();
    HleFn pton = Hle::lookup(kInetPton), ntop = Hle::lookup(kInetNtop);
    const auto round_trip = [&](const char* in) {
        uint8_t a[16];
        memset(a, 0xEE, sizeof(a));
        if (pton(28, (uint64_t)in, (uint64_t)a, 0, 0, 0) != 1u) return std::string("<pton failed>");
        char out[46];
        if (ntop(28, (uint64_t)a, (uint64_t)out, sizeof(out), 0, 0) != (uint64_t)out)
            return std::string("<ntop failed>");
        return std::string(out);
    };
    uint8_t a[16];
    ASSERT_EQ(pton(28, (uint64_t)"2001:db8::1", (uint64_t)a, 0, 0, 0), 1u);
    const uint8_t want[16] = {0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    EXPECT_EQ(memcmp(a, want, 16), 0) << "\"::\" expands to the missing zero groups";
    EXPECT_EQ(round_trip("2001:db8::1"), "2001:db8::1");
    EXPECT_EQ(round_trip("::"), "::") << "all zeros";
    EXPECT_EQ(round_trip("::1"), "::1") << "loopback";
    EXPECT_EQ(round_trip("FE80:0:0:0:0:0:0:1"), "fe80::1") << "lowercase, longest zero run";
    EXPECT_EQ(round_trip("1:0:0:2:0:0:3:4"), "1::2:0:0:3:4") << "tie: the first run compresses";
    EXPECT_EQ(round_trip("1:0:2:3:4:5:6:7"), "1:0:2:3:4:5:6:7") << "a single zero group stays";
    EXPECT_EQ(round_trip("::ffff:1.2.3.4"), "::ffff:1.2.3.4") << "IPv4-mapped keeps dotted tail";
    EXPECT_EQ(pton(28, (uint64_t)"1::2::3", (uint64_t)a, 0, 0, 0), 0u) << "two \"::\" is bad text";
    EXPECT_EQ(pton(28, (uint64_t)"12345::1", (uint64_t)a, 0, 0, 0), 0u) << "5-digit group is bad";
    EXPECT_EQ(pton(28, (uint64_t)"1:2:3:4:5:6:7:8:9", (uint64_t)a, 0, 0, 0), 0u)
        << "nine groups is bad text";
    char tiny[4];
    const uint8_t lo[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    EXPECT_EQ(ntop(28, (uint64_t)lo, (uint64_t)tiny, 3, 0, 0), 0u)
        << "\"::1\" needs 4 bytes: size 3 is refused";
    EXPECT_EQ(ntop(28, (uint64_t)lo, (uint64_t)tiny, 4, 0, 0), (uint64_t)tiny) << "size 4 fits";
    EXPECT_STREQ(tiny, "::1");
}

// Epoll, introspection, pools, resolver: local lifecycle where allocation is local (never the
// dispatcher's 0 from an id contract), loud failure where an answer needs hardware or a
// network. Every arm drives the real NIDs; the mutation each arm kills is named on the arm.
namespace {

constexpr const char* kEpollCreate = "SF47kB2MNTo";
constexpr const char* kEpollDestroy = "Inp1lfL+Jdw";
constexpr const char* kEpollControl = "ZVw46bsasAk";
constexpr const char* kEpollWait = "drjIbDbA7UQ";
constexpr const char* kEtherNtostr = "v6M4txecCuo";
constexpr const char* kGetMacAddress = "6Oc0bLsIYe0";
constexpr const char* kGetSockInfo = "hLuXdjHnhiI";
constexpr const char* kPoolDestroy = "K7RlrTkI-mw";
constexpr const char* kResolverCreate = "C4UgDHHPvdw";
constexpr const char* kResolverStartNtoa = "Nd91WaWmG2w";
constexpr const char* kResolverDestroy = "kJlYH5uMAWI";
constexpr const char* kRudpEnable = "6PBNpsgyaxw";
constexpr const char* kRudpSetHandler = "SUEVes8gvmw";

constexpr uint64_t kInval = net::kNetErrorInval;          // 0x80410116
constexpr uint64_t kOpNotSupp = net::kNetErrorOpNotSupp;  // 0x8041012d
// Id-returning contracts sign-extend their errors (never a handle); status contracts keep
// the zero-extended 32-bit form. Both conventions live in hle_net.cpp.
constexpr uint64_t kIdInval = (uint64_t)(int64_t)(int32_t)net::kNetErrorInval;
constexpr uint64_t kIdBadF = (uint64_t)(int64_t)(int32_t)net::kNetErrorBadF;
constexpr uint64_t kIdOpNotSupp = (uint64_t)(int64_t)(int32_t)net::kNetErrorOpNotSupp;

int32_t net_errno() {
    HleFn loc = Hle::lookup(kErrnoLoc);
    return loc ? *reinterpret_cast<const int32_t*>(loc(0, 0, 0, 0, 0, 0)) : -1;
}

}  // namespace

TEST(NetSocket, EpollLifecycle) {
    register_builtin_hle();
    HleFn create = Hle::lookup(kEpollCreate);
    HleFn destroy = Hle::lookup(kEpollDestroy);
    HleFn control = Hle::lookup(kEpollControl);
    HleFn wait = Hle::lookup(kEpollWait);
    for (HleFn f : {create, destroy, control, wait}) ASSERT_NE(f, nullptr);
    EXPECT_EQ(create(0, 0, 0, 0, 0, 0), kIdInval) << "null name is refused";
    EXPECT_EQ(create((uint64_t)"test", 1, 0, 0, 0, 0), kIdInval) << "non-zero flags are refused";
    const uint64_t eid = create((uint64_t)"test", 0, 0, 0, 0, 0);
    ASSERT_GT((int64_t)eid, 0) << "epoll id is positive, never the dispatcher 0";
    const uint64_t fd = open_socket();
    ASSERT_GT((int64_t)fd, 0);
    uint32_t ev[4] = {1u, 0, 0, 0};   // events mask in the leading word
    uint32_t no_ev[4] = {0, 0, 0, 0};
    EXPECT_EQ(control(eid, 0, fd, (uint64_t)ev, 0, 0), kInval) << "op 0 is not ADD/MOD/DEL";
    EXPECT_EQ(control(eid, 4, fd, (uint64_t)ev, 0, 0), kInval) << "op 4 is not ADD/MOD/DEL";
    EXPECT_EQ(control(eid, 1, fd, 0, 0, 0), kInval) << "ADD needs an event";
    EXPECT_EQ(control(eid, 2, fd, (uint64_t)no_ev, 0, 0), kInval) << "MOD needs a non-empty mask";
    EXPECT_EQ(control(9999, 1, fd, (uint64_t)ev, 0, 0), kBadF) << "unknown epoll id is refused";
    EXPECT_EQ(control(eid, 1, 9999, (uint64_t)ev, 0, 0), kBadF) << "unknown socket id is refused";
    EXPECT_EQ(control(eid, 1, fd, (uint64_t)ev, 0, 0), 0u);
    EXPECT_EQ(control(eid, 3, fd, 0, 0, 0), 0u) << "DEL needs no event";
    uint8_t events[64]{};
    // The buffer and count are checked before the id, and a count's error is sign-extended.
    EXPECT_EQ(wait(9999, 0, 0, 0, 0, 0), kIdInval) << "null buffer is refused before the id";
    EXPECT_EQ(wait(eid, (uint64_t)events, 0, 0, 0, 0), kIdInval) << "maxevents 0 is refused";
    EXPECT_EQ(wait(9999, (uint64_t)events, 4, 0, 0, 0), kIdBadF) << "unknown epoll id";
    EXPECT_EQ(wait(eid, (uint64_t)events, 4, 0, 0, 0), 0u) << "zero timeout: no events, at once";
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(wait(eid, (uint64_t)events, 4, (uint64_t)(uint32_t)-1, 0, 0), 0u)
        << "a blocking wait answers 0 events after a bounded sleep";
    EXPECT_GE(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(20))
        << "a blocking (negative-timeout) wait must sleep, or a guest loop busy-spins";
    EXPECT_EQ(destroy(eid, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(destroy(eid, 0, 0, 0, 0, 0), kBadF) << "destroy frees exactly once";
    EXPECT_EQ(wait(eid, (uint64_t)events, 4, 0, 0, 0), kIdBadF) << "wait on dead id fails";
    HleFn close = Hle::lookup(kSocketClose);
    ASSERT_NE(close, nullptr);
    EXPECT_EQ(close(fd, 0, 0, 0, 0, 0), 0u);
}

TEST(NetSocket, EtherFormatting) {
    register_builtin_hle();
    HleFn ntostr = Hle::lookup(kEtherNtostr);
    ASSERT_NE(ntostr, nullptr);
    const uint8_t mac[6] = {0x00, 0x1a, 0x2b, 0x3c, 0x4d, 0x5e};
    char str[32];
    std::memset(str, 'x', sizeof str);
    EXPECT_EQ(ntostr((uint64_t)mac, (uint64_t)str, 18, 0, 0, 0), 0u)
        << "an int status, 0 on success";
    EXPECT_STREQ(str, "00:1a:2b:3c:4d:5e") << "zero-padded %02x, as the module formats it";
    char seventeen[17];
    std::memset(seventeen, 'x', sizeof seventeen);
    EXPECT_EQ(ntostr((uint64_t)mac, (uint64_t)seventeen, 17, 0, 0, 0), kInval)
        << "len < 18 fails even though the text would not need it all";
    EXPECT_EQ(seventeen[0], 'x') << "a refused call writes nothing";
    EXPECT_EQ(net_errno(), 22) << "the module sets errno EINVAL";
    EXPECT_EQ(ntostr(0, (uint64_t)str, 18, 0, 0, 0), kInval) << "null ether is refused";
    EXPECT_EQ(ntostr((uint64_t)mac, 0, 18, 0, 0, 0), kInval) << "null out is refused";
}

TEST(NetSocket, HardwareIntrospectionRefused) {
    register_builtin_hle();
    HleFn mac = Hle::lookup(kGetMacAddress);
    HleFn info = Hle::lookup(kGetSockInfo);
    ASSERT_NE(mac, nullptr);
    ASSERT_NE(info, nullptr);
    uint8_t out[32];
    std::memset(out, 0xAA, sizeof out);
    EXPECT_EQ(mac(0, 0, 0, 0, 0, 0), kInval) << "null addr is refused first";
    EXPECT_EQ(mac((uint64_t)out, 1, 0, 0, 0, 0), kInval) << "non-zero flags are refused first";
    // No interface to report: a zeroed MAC would be a manufactured hardware identity.
    EXPECT_EQ(mac((uint64_t)out, 0, 0, 0, 0, 0), kOpNotSupp);
    EXPECT_EQ(out[0], 0xAA) << "refused query writes no address";
    const uint64_t fd = open_socket();
    ASSERT_GT((int64_t)fd, 0);
    EXPECT_EQ(info(fd, (uint64_t)out, 1, 0x1000, 0, 0), kIdInval) << "flags & 0x31000 refused";
    EXPECT_EQ(info(9999, (uint64_t)out, 1, 0, 0, 0), kIdBadF) << "unknown socket is refused";
    std::memset(out, 0xAA, sizeof out);
    EXPECT_EQ(info(fd, (uint64_t)out, 1, 0, 0, 0), kIdOpNotSupp)
        << "unknown struct: fail, don't scribble (a count contract: sign-extended)";
    EXPECT_EQ(out[0], 0xAA);
    HleFn close = Hle::lookup(kSocketClose);
    ASSERT_NE(close, nullptr);
    EXPECT_EQ(close(fd, 0, 0, 0, 0, 0), 0u);
}

TEST(NetSocket, PoolDestroyAndResolver) {
    register_builtin_hle();
    HleFn pool_destroy = Hle::lookup(kPoolDestroy);
    HleFn resolver_create = Hle::lookup(kResolverCreate);
    HleFn start_ntoa = Hle::lookup(kResolverStartNtoa);
    HleFn resolver_destroy = Hle::lookup(kResolverDestroy);
    for (HleFn f : {pool_destroy, resolver_create, start_ntoa, resolver_destroy})
        ASSERT_NE(f, nullptr);
    // Pools are untracked counter ids (see PoolCreate): nothing to free, always succeeds.
    EXPECT_EQ(pool_destroy(1, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(resolver_create(0, 0, 0, 0, 0, 0), kIdInval) << "null name is refused";
    EXPECT_EQ(resolver_create((uint64_t)"test", 0, 1, 0, 0, 0), kIdInval) << "flags refused";
    const uint64_t rid = resolver_create((uint64_t)"test", 0, 0, 0, 0, 0);
    ASSERT_GT((int64_t)rid, 0) << "resolver id is positive, never the dispatcher 0";
    uint8_t addr[16];
    std::memset(addr, 0xAA, sizeof addr);
    const uint64_t host = (uint64_t)"example.com";
    EXPECT_EQ(start_ntoa(rid, host, (uint64_t)addr, (uint64_t)(uint32_t)-1, 0, 0), kInval)
        << "negative timeout is refused";
    EXPECT_EQ(start_ntoa(rid, host, (uint64_t)addr, 0, (uint64_t)(uint32_t)-1, 0), kInval)
        << "negative retry is refused";
    EXPECT_EQ(start_ntoa(rid, host, (uint64_t)addr, 0, 0, 2), kInval) << "flags outside 0x10001";
    EXPECT_EQ(start_ntoa(9999, host, (uint64_t)addr, 0, 0, 0), kBadF);
    EXPECT_EQ(start_ntoa(rid, 0, (uint64_t)addr, 0, 0, 0), kInval) << "null hostname refused";
    EXPECT_EQ(start_ntoa(rid, host, (uint64_t)addr, 0, 0, 0), kUnreach) << "no DNS offline";
    EXPECT_EQ(addr[0], 0xAA) << "failed resolve writes no address";
    EXPECT_EQ(resolver_destroy(rid, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(resolver_destroy(rid, 0, 0, 0, 0, 0), kBadF) << "destroy frees exactly once";
    EXPECT_EQ(start_ntoa(rid, host, (uint64_t)addr, 0, 0, 0), kBadF) << "dead id is refused";
}

TEST(NetSocket, ResolverSlotsAreReusedAfterDestroy) {
    // 32 slots: a create/destroy cycle that leaked would fail the 33rd create.
    register_builtin_hle();
    HleFn resolver_create = Hle::lookup(kResolverCreate);
    HleFn resolver_destroy = Hle::lookup(kResolverDestroy);
    ASSERT_NE(resolver_create, nullptr);
    ASSERT_NE(resolver_destroy, nullptr);
    for (int i = 0; i < 33; i++) {
        const uint64_t rid = resolver_create((uint64_t)"cycle", 0, 0, 0, 0, 0);
        ASSERT_GT((int64_t)rid, 0) << "cycle " << i << ": a destroyed slot is reusable";
        ASSERT_EQ(resolver_destroy(rid, 0, 0, 0, 0, 0), 0u) << "cycle " << i;
    }
}

TEST(Rudp, SetupAcknowledged) {
    register_builtin_hle();
    HleFn enable = Hle::lookup(kRudpEnable);
    HleFn set_handler = Hle::lookup(kRudpSetHandler);
    ASSERT_NE(enable, nullptr);
    ASSERT_NE(set_handler, nullptr);
    // Setup with no out-parameters: no IO thread starts, no handler is stored.
    EXPECT_EQ(enable(0x10000, 100, 0, 0, 0, 0), 0u);
    int marker = 0;
    EXPECT_EQ(set_handler((uint64_t)&marker, 0, 0, 0, 0, 0), 0u) << "a handler is accepted";
    EXPECT_EQ(set_handler(0, 0, 0, 0, 0, 0), 0x80770022u) << "a NULL handler is refused";
}
