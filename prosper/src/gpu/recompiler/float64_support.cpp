// float64_support.cpp — the published binary64 device verdict (float64_support.hpp).
#include "gpu/recompiler/float64_support.hpp"

#include <atomic>
#include <cstdio>

namespace prosper::gpu {

namespace {
// "Nobody published" and "a device said no" stay distinct, as for the float-controls gate.
std::atomic<bool> g_float64_published{false};
std::atomic<bool> g_float64_supported{true};
} // namespace

void publish_float64_support(bool shader_float64_enabled, bool compatible_float_controls,
                             bool float_controls_permitted) {
    const bool usable =
        shader_float64_enabled && compatible_float_controls && float_controls_permitted;
    if (!usable) {
        static std::atomic<unsigned> said{0};
        g_float64_supported.store(false, std::memory_order_relaxed);
        if (said.fetch_add(1) == 0)
            std::fprintf(
                stderr,
                "[recompile] F64 DISABLED: shaderFloat64 %s, compatible F64 float controls "
                "%s, float controls %s. Guest binary64 instructions refuse "
                "(f64-device-unavailable) instead of running unflushed-input semantics "
                "this device cannot promise.\n",
                shader_float64_enabled ? "enabled" : "not enabled",
                compatible_float_controls ? "supported" : "unavailable",
                float_controls_permitted ? "permitted" : "not permitted");
    }
    // Published LAST, so a reader that observes `published` also observes this verdict.
    g_float64_published.store(true, std::memory_order_release);
}

bool float64_admitted() {
    return g_float64_published.load(std::memory_order_acquire) &&
           g_float64_supported.load(std::memory_order_relaxed);
}

void reset_float64_support_for_test() {
    g_float64_supported.store(true, std::memory_order_relaxed);
    g_float64_published.store(false, std::memory_order_release);
}

} // namespace prosper::gpu
