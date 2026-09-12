#pragma once

// The storage-image device contract -- what a Vulkan device must offer before prosper may execute a
// recompiled storage-image kernel on it (#3531). Deliberately Vulkan-free so the decision can be
// unit-tested with no driver, no device and no Vulkan headers, and so the compute backend's public
// header can report the capability without pulling Vulkan into every consumer.
//
// `src/gpu/recompiler/rdna2_to_spirv_internal.hpp` (image_read) issues an OpImageRead for ALL
// invocations, including EXEC-inactive lanes whose coordinate may be out of range: the loaded value
// is discarded by write-back predication at the call site. That is defined only when the device
// enables IMAGE ROBUSTNESS (`robustImageAccess`, core in Vulkan 1.3; `VK_EXT_image_robustness`
// before that), which makes an out-of-range image read return zero. Without it the kernel is
// undefined behaviour, and on RADV an out-of-range image read is the shape that raises
// GCVM_L2_PROTECTION_FAULT (CLIENT_ID: TCP, RW: 0).
//
// Every device prosper executes recompiled storage-image work on must therefore acquire the feature,
// and every device that CANNOT must refuse that work rather than run it. This is the single place
// that decision is made, so the renderer device, the compute backend's own device and the adopt path
// cannot diverge -- divergence is what #3531 was: the renderer chained the feature, the compute
// backend's own device did not, and no test could tell because every harness chained it.

namespace prosper::frontend {

// What a device offers the recompiled storage-image path. Fields describe what is ENABLED on the
// device (or will be, at creation), never what the physical device merely supports -- the consumer's
// question is "may this device execute the kernel", and a supported-but-not-requested feature
// answers it wrongly.
struct StorageImageDeviceFeatures {
    // VkPhysicalDeviceImageRobustnessFeatures::robustImageAccess. The OOB contract above.
    bool robust_image_access = false;
    bool read_without_format = false;
    bool write_without_format = false;

    // The only predicate a caller should gate recompiled storage-image dispatches on. All three
    // terms are required: the format-free capabilities to express the raw uvec4 texel model at all,
    // and image robustness to make the model's deliberate out-of-range reads defined.
    bool storage_image_capable() const {
        return robust_image_access && read_without_format && write_without_format;
    }
};

// Pure decision, so the refusal branch is testable without a device. No device available to this
// project lacks `robustImageAccess`, so a test drawn from a real device can only ever exercise the
// accepting branch -- it tests the discriminator, not the domain. Build the refusing input by hand.
constexpr StorageImageDeviceFeatures storage_image_device_features(bool robust_image_access,
                                                                   bool read_without_format,
                                                                   bool write_without_format) {
    StorageImageDeviceFeatures features;
    features.robust_image_access = robust_image_access;
    features.read_without_format = read_without_format;
    features.write_without_format = write_without_format;
    return features;
}

} // namespace prosper::frontend
