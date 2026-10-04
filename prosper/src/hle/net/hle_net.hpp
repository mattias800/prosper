#pragma once

#include <cstdint>

namespace prosper::net {

// libSceNet address families and socket types prosper's socket table models. Values are the
// FreeBSD numbers the guest passes (AF_INET=2, AF_INET6=28, AF_UNIX=1; SOCK_STREAM=1,
// SOCK_DGRAM=2); anything else is refused with kNetErrorInval, never allocated.
constexpr int32_t kAfUnix = 1;
constexpr int32_t kAfInet = 2;
constexpr int32_t kAfInet6 = 28;
constexpr int32_t kSockStream = 1;
constexpr int32_t kSockDgram = 2;

// Address families for sceNetInetPton/sceNetInetNtop. Only AF_INET dotted-decimal is computed;
// AF_INET6 answers "not modelled" so callers take their error path instead of reading garbage.
constexpr int32_t kPtonV4Bytes = 4;

}  // namespace prosper::net
