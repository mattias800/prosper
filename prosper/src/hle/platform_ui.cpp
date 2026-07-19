// platform_ui.cpp — the registry for the interactive-UI backend (see platform_ui.hpp). Just holds the
// single registered pointer; the frontend sets it at startup and the dialog HLE handlers read it.
#include "platform_ui.hpp"
#include <atomic>

namespace prosper {

namespace {
std::atomic<PlatformUi*> g_platform_ui{nullptr};
std::mutex g_platform_ui_mx;
}

PlatformUi* platform_ui() { return g_platform_ui.load(std::memory_order_acquire); }
PlatformUiLease platform_ui_lease(PlatformUi* expected) {
    std::unique_lock<std::mutex> lock(g_platform_ui_mx);
    PlatformUi* ui = g_platform_ui.load(std::memory_order_acquire);
    if (expected && ui != expected) ui = nullptr;
    return PlatformUiLease(std::move(lock), ui);
}
void set_platform_ui(PlatformUi* ui) {
    std::lock_guard<std::mutex> lock(g_platform_ui_mx);
    g_platform_ui.store(ui, std::memory_order_release);
}

} // namespace prosper
