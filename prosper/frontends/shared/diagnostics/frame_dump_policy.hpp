// frame_dump_policy.hpp — opt-in policy for boot_trace's periodic BMP output.
#pragma once

#include <string>
#include <utility>

namespace prosper::frontend {

// The environment is represented by presence, matching prosper's other diagnostic switches. Keeping
// the resolver independent of getenv makes both precedence and boot_trace-to-renderer wiring testable
// without creating a Vulkan device or writing an image.
struct FrameDumpEnvironment {
    const char* frame_dir = nullptr;
    const char* frame_dumps = nullptr;
    const char* dump_content = nullptr;
    const char* dump_first = nullptr;
    const char* dump_every = nullptr;
    const char* no_frame_dumps = nullptr;
};

constexpr bool frame_dump_request_allowed(bool requested, const char* no_frame_dumps) {
    return requested && no_frame_dumps == nullptr;
}

constexpr bool frame_dumps_enabled(const FrameDumpEnvironment& env) {
    return frame_dump_request_allowed(
        env.frame_dumps != nullptr || env.dump_content != nullptr ||
            env.dump_first != nullptr || env.dump_every != nullptr,
        env.no_frame_dumps);
}

inline constexpr bool kFrameDumpsByDefault = frame_dumps_enabled({});

template <typename Lookup>
FrameDumpEnvironment read_frame_dump_environment(Lookup&& lookup) {
    return {
        lookup("PROSPER_FRAME_DIR"),
        lookup("PROSPER_FRAME_DUMPS"),
        lookup("PROSPER_DUMP_CONTENT"),
        lookup("PROSPER_FRAME_DUMP_FIRST"),
        lookup("PROSPER_FRAME_DUMP_EVERY"),
        lookup("PROSPER_NO_FRAME_DUMPS"),
    };
}

// This is boot_trace's complete policy-to-registrar path. Tests substitute a pure lookup and a fake
// registrar, while production supplies getenv and register_live_renderer.
template <typename Lookup, typename Registrar>
void register_live_renderer_from_environment(Lookup&& lookup, Registrar&& registrar) {
    const FrameDumpEnvironment env =
        read_frame_dump_environment(std::forward<Lookup>(lookup));
    std::forward<Registrar>(registrar)(env.frame_dir ? env.frame_dir : ".",
                                      frame_dumps_enabled(env));
}

} // namespace prosper::frontend
