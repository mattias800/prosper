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

#include <cstdint>
#include <cstdio>
#include <cstring>

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
    const uint64_t fd = fn(2, 1, 0, 0, 0, 0);  // AF_INET, SOCK_STREAM
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
    const uint64_t bad = Hle::lookup(kSocket)(99, 1, 0, 0, 0, 0);
    EXPECT_LT((int64_t)bad, 0) << "bad-domain socket answers negative in int64 too";
    EXPECT_EQ((uint32_t)bad, net::kNetErrorOpNotSupp) << "bad domain is EOPNOTSUPP";
    EXPECT_NE(bad, 0u) << "an id-returning failure is never the dispatcher 0";
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
    const uint64_t fd = open_socket();
    ASSERT_GT((int64_t)fd, 0);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(p), 0)
        << "fresh socket leaves errno clear (kills: stale errno slot)";
    Hle::lookup(kConnect)(fd, 0, 0, 0, 0, 0);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(p), 51)
        << "failed connect records ENETUNREACH (51) where the guest reads errno";
    close_socket(fd);
}

TEST(NetSocket, InetConvert) {
    register_builtin_hle();
    uint8_t addr[4] = {0};
    EXPECT_EQ(Hle::lookup(kInetPton)(2, (uint64_t)"192.168.1.1", (uint64_t)addr, 0, 0, 0), 1u)
        << "AF_INET dotted-decimal parses (kills: unimplemented helper)";
    EXPECT_EQ(addr[0], 192) << "pton writes network-order bytes";
    EXPECT_EQ(addr[1], 168) << "pton writes network-order bytes";
    EXPECT_EQ(addr[2], 1) << "pton writes network-order bytes";
    EXPECT_EQ(addr[3], 1) << "pton writes network-order bytes";
    EXPECT_EQ(Hle::lookup(kInetPton)(2, (uint64_t)"999.1.1.1", (uint64_t)addr, 0, 0, 0), 0u)
        << "out-of-range part is bad text, not a fill (kills: blind success)";
    EXPECT_EQ(Hle::lookup(kInetPton)(2, (uint64_t)"1.2.3", (uint64_t)addr, 0, 0, 0), 0u)
        << "three parts is bad text";
    char text[16] = {0};
    const uint8_t src[4] = {10, 0, 0, 1};
    EXPECT_EQ(Hle::lookup(kInetNtop)(2, (uint64_t)src, (uint64_t)text, 16, 0, 0), (uint64_t)text)
        << "ntop returns dst on success";
    EXPECT_STREQ(text, "10.0.0.1") << "ntop emits dotted-decimal";
    EXPECT_EQ(Hle::lookup(kInetNtop)(2, (uint64_t)src, (uint64_t)text, 8, 0, 0), 0u)
        << "short buffer is NULL, not a truncation (kills: silent truncation)";
    EXPECT_EQ(*reinterpret_cast<int32_t*>(Hle::lookup(kErrnoLoc)(0, 0, 0, 0, 0, 0)), 28)
        << "short buffer records ENOSPC (28) where the guest reads errno";
    EXPECT_EQ(Hle::lookup(kInetNtop)(28, (uint64_t)src, (uint64_t)text, 64, 0, 0), 0u)
        << "AF_INET6 is refused, not fabricated (kills: false v6 rendering)";
}
