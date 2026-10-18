#include "gpu/recompiler/float32_rte_support.hpp"

#include <atomic>

namespace prosper::gpu {
namespace {
std::atomic<bool> published{false}, supported{true};
}

void publish_float32_rte_support(bool rte32, bool independent32, bool permitted) {
    if (!(rte32 && independent32 && permitted)) supported.store(false, std::memory_order_relaxed);
    published.store(true, std::memory_order_release);
}
bool float32_rte_admitted() {
    return published.load(std::memory_order_acquire) && supported.load(std::memory_order_relaxed);
}
void reset_float32_rte_support_for_test() {
    supported.store(true, std::memory_order_relaxed);
    published.store(false, std::memory_order_release);
}
} // namespace prosper::gpu
