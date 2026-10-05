#pragma once

#include <cstdint>

namespace prosper::net {

// libSceNet address families and socket types prosper's socket table models. Values are the
// FreeBSD numbers the guest passes (AF_INET=2, AF_INET6=28, AF_UNIX=1; SOCK_STREAM=1,
// SOCK_DGRAM=2). sceNetSocket refuses any other family with EAFNOSUPPORT and any other type with
// EOPNOTSUPP; nothing outside this surface is allocated. sceNetInetPton/Ntop accept AF_INET and
// AF_INET6 and answer EAFNOSUPPORT for anything else.
constexpr int32_t kAfUnix = 1;
constexpr int32_t kAfInet = 2;
constexpr int32_t kAfInet6 = 28;
constexpr int32_t kSockStream = 1;
constexpr int32_t kSockDgram = 2;

}  // namespace prosper::net
