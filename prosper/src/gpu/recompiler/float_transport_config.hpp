#pragma once
#include <cstdint>
#include <cstddef>

namespace prosper::gpu {

// A producing compiler choice, independent of the guest FLOAT_MODE register.
// ExplicitNonFinite32 requires enabled shaderFloatControls2 AND the float32 SZI property.
// Unknown means no witness was retained; it is never evidence that a feature was disabled.
enum class FloatTransportProfile : uint8_t { Unknown, Implicit, ExplicitNonFinite32 };

struct FloatTransportConfig {
    FloatTransportProfile profile = FloatTransportProfile::Unknown;

    constexpr bool canonical() const {
        return static_cast<uint8_t>(profile) <=
               static_cast<uint8_t>(FloatTransportProfile::ExplicitNonFinite32);
    }
    constexpr bool available() const { return profile != FloatTransportProfile::Unknown; }
    constexpr bool explicit_nonfinite32() const {
        return profile == FloatTransportProfile::ExplicitNonFinite32;
    }
    bool operator==(const FloatTransportConfig&) const = default;
};

constexpr const char* float_transport_profile_name(FloatTransportConfig config) {
    switch (config.profile) {
        case FloatTransportProfile::Unknown: return "unknown";
        case FloatTransportProfile::Implicit: return "implicit";
        case FloatTransportProfile::ExplicitNonFinite32: return "explicit-nonfinite32";
    }
    return "invalid";
}

// A request decision, not an enabled witness. Device owners publish it only after creation.
constexpr FloatTransportConfig select_float_transport_request(
        bool core14, bool feature_supported, bool preserve_float32, bool request_feature) {
    return {core14 && feature_supported && preserve_float32 && request_feature
        ? FloatTransportProfile::ExplicitNonFinite32 : FloatTransportProfile::Implicit};
}

// Backend admission inspects actual words, including stored replay modules. Capture metadata
// and requested profiles never prove that the executing VkDevice enabled the capability.
inline bool float_transport_module_supported(const uint32_t* words, size_t count,
                                             FloatTransportConfig device) {
    if (!words || count < 5) return true; // ordinary module validation owns malformed headers
    bool required = false;
    for (size_t at = 5; at < count;) {
        const uint32_t length = words[at] >> 16;
        if (!length || length > count - at) return false;
        if ((words[at] & 0xffffu) == 17 && length == 2 && words[at + 1] == 6029)
            required = true;
        at += length;
    }
    return !required || device.explicit_nonfinite32();
}

// Device owners publish only AFTER successful device creation, using what they enabled,
// never merely advertised support. A snapshot is copied into the key and emitter once.
// Multiple device owners conservatively AND their known profiles; offline state is Unknown.
void publish_float_transport_config(FloatTransportConfig config);
FloatTransportConfig published_float_transport_config();
void reset_float_transport_config_for_test();

} // namespace prosper::gpu
