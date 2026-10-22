// Internal parser and operation pin for CFG trip-bound settings.
#pragma once
#include <atomic>
#include <cstdint>

namespace prosper::gpu {
struct ComputeTripBoundSettings;
std::atomic<uint64_t>& trip_bound_parse_counter();
const ComputeTripBoundSettings*& trip_bound_pin();
ComputeTripBoundSettings parse_trip_bound_settings_impl();
}  // namespace prosper::gpu
