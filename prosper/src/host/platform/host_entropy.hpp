// host_entropy.hpp — the host's cryptographically secure random source, for every HLE that hands
// the guest random bytes (sceRandomGetRandomNumber, the guest's /dev/urandom and /dev/random).
#pragma once
#include <cstddef>

namespace prosper::host {

// Fill `bytes` from the host CSPRNG: getentropy() on POSIX hosts, BCryptGenRandom on Windows.
// Returns false if the host cannot supply entropy -- which the caller MUST surface as an error,
// never as a zero-filled success. Deterministic zeros presented as random are the same lie in a
// different costume.
bool host_entropy_fill(void* dst, std::size_t bytes);

}   // namespace prosper::host
