// host_crt.hpp — which C runtime family this host links, for code that must spell a request in
// that runtime's own dialect (a stdio mode string, say) rather than call an OS service. The `#if`
// lives here so callers in src/hle can branch on a constant instead of on the platform.
#pragma once

namespace prosper::host {

enum class CrtFamily {
    Posix,      // glibc, Darwin libc
    Microsoft,  // UCRT / msvcrt (MinGW and MSVC builds alike)
};

#if defined(_WIN32)
inline constexpr CrtFamily kCrtFamily = CrtFamily::Microsoft;
#else
inline constexpr CrtFamily kCrtFamily = CrtFamily::Posix;
#endif

} // namespace prosper::host
