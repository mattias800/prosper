#pragma once

#include <cstdint>

namespace prosper {

// Internal GPU dependency-watchdog clock, never a guest-visible timestamp. A synchronous host
// submit holds the queue mutex and prevents its producer from advancing. Exclude excess backend
// time only from that watchdog, retaining budget_ns for completed display flips. Guest clocks and
// EOP timestamps continue on one real steady timeline, including during these scopes (#3422).
uint64_t host_gpu_progress_ns();
// A zero token means the scope was not installed (disabled or another scope is active).
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
