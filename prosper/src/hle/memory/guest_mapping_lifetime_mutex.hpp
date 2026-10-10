#pragma once

#include <shared_mutex>

namespace prosper {

// Shared by mapping readers and writers; native initialization finishes before either may enter.
// The detached completion worker may outlive main, so the mutex has process lifetime.
std::shared_mutex& guest_mapping_lifetime_mutex();

} // namespace prosper
