#pragma once

#include <cstdint>

namespace prosper {

// Exclude excess time spent in Prosper's synchronous host GPU backend from the guest monotonic
// clock. Real hardware submits this work asynchronously; charging a multi-second host translation
// or pipeline warmup to the next guest frame makes time-based state machines skip visible states.
// `budget_ns` is the amount of that interval that may legitimately belong to completed display
// flips. A zero token means the scope was not installed (disabled or another scope is active).
uint64_t guest_clock_host_gpu_begin(uint64_t budget_ns);
void guest_clock_host_gpu_end(uint64_t token);

class HostGpuClockScope {
public:
    explicit HostGpuClockScope(uint64_t budget_ns)
        : token_(guest_clock_host_gpu_begin(budget_ns)) {}
    ~HostGpuClockScope() { guest_clock_host_gpu_end(token_); }

    HostGpuClockScope(const HostGpuClockScope&) = delete;
    HostGpuClockScope& operator=(const HostGpuClockScope&) = delete;

private:
    uint64_t token_ = 0;
};

} // namespace prosper
