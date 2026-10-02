// device_identity.hpp -- which GPU and driver did this process actually render on?
//
// WHY THIS EXISTS. A framerate is a statement about a (title, route, build, GPU, driver, present
// path) tuple, and a figure quoted without the tuple cannot be compared with another one. The
// capture manifest already records the title, route and build; it could not record the device,
// because the device is chosen inside the live renderer, after the manifest header is written.
// This is the one place the renderer says which device it picked, so a consumer (tools/screenshot's
// `conditions` record) can read it back without owning Vulkan.
//
// NUMBERS ONLY, DELIBERATELY. Vendor id, device id, driver version, API version and device type are
// stable, comparable identifiers. The device NAME string is not recorded: it is a free-form hardware
// model string that adds nothing the ids do not decide, and manifests are quoted into a public
// tracker.
//
// Plain integers rather than Vulkan types, so a pure consumer (tests/test_capture_manifest.cpp) can
// include this without a Vulkan loader.
#pragma once

#include <cstdint>
#include <mutex>

namespace prosper::frontend {

struct DeviceIdentity {
    bool known = false;   // false until a device has been selected in this process
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t driver_version = 0;   // vendor-encoded; comparable for equality, not orderable
    uint32_t api_version = 0;
    uint32_t device_type = 0;      // VkPhysicalDeviceType
};

namespace detail {
inline std::mutex& device_identity_mutex() {
    static std::mutex m;
    return m;
}
inline DeviceIdentity& device_identity_slot() {
    static DeviceIdentity identity;
    return identity;
}
} // namespace detail

// Called once per process by the code that selects the Vulkan device. A later call replaces the
// earlier one, so the last device selected is the one reported.
inline void record_device_identity(uint32_t vendor_id, uint32_t device_id, uint32_t driver_version,
                                   uint32_t api_version, uint32_t device_type) {
    std::lock_guard<std::mutex> lock(detail::device_identity_mutex());
    detail::device_identity_slot() =
        DeviceIdentity{true, vendor_id, device_id, driver_version, api_version, device_type};
}

inline DeviceIdentity device_identity() {
    std::lock_guard<std::mutex> lock(detail::device_identity_mutex());
    return detail::device_identity_slot();
}

} // namespace prosper::frontend
