#include "hle/memory/guest_mapping_lifetime_mutex.hpp"
#include <memory>
#include <mutex>

namespace prosper {

std::shared_mutex& guest_mapping_lifetime_mutex() {
    // CONFIDENCE: HIGH: MinGW's pthread rwlock can return EINVAL when first users race its
    // lazy native initialization. Complete one balanced acquisition inside the C++ static
    // initialization guard, before publishing the mutex to any reader or mapping writer.
    static std::shared_mutex* const mutex = [] {
        auto owned = std::make_unique<std::shared_mutex>();
        { const std::unique_lock initialized(*owned); }
        return owned.release();
    }();
    return *mutex;
}

} // namespace prosper
