// lifecycle.cpp — see lifecycle.hpp.
#include "lifecycle.hpp"
#include <atomic>

namespace prosper {

namespace { std::atomic<bool> g_stop{false}; }

void prosper_request_stop()   { g_stop.store(true, std::memory_order_relaxed); }
bool prosper_stop_requested() { return g_stop.load(std::memory_order_relaxed); }
void prosper_reset_stop()     { g_stop.store(false, std::memory_order_relaxed); }

} // namespace prosper
