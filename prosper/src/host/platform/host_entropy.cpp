// host_entropy.cpp — see host_entropy.hpp.
#include "host/platform/host_entropy.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>   // BCryptGenRandom (prosper_core links bcrypt on Windows)
#else
#include <sys/random.h>   // getentropy
#include <unistd.h>
#endif

namespace prosper::host {

bool host_entropy_fill(void* dst, std::size_t bytes) {
#ifdef _WIN32
    return BCryptGenRandom(nullptr, (PUCHAR)dst, (ULONG)bytes, BCRYPT_USE_SYSTEM_PREFERRED_RNG) ==
           0;
#else
    auto* p = static_cast<unsigned char*>(dst);
    while (bytes) {
        const std::size_t chunk = bytes < 256 ? bytes : 256;   // getentropy's published maximum
        if (getentropy(p, chunk) != 0) return false;
        p += chunk;
        bytes -= chunk;
    }
    return true;
#endif
}

}   // namespace prosper::host
