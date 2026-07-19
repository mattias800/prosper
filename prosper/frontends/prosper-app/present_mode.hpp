#pragma once

#include <string_view>

namespace prosper::frontend {

enum class AppPresentMode {
    fifo,
    mailbox,
    immediate,
};

struct AppPresentModeSelection {
    AppPresentMode mode = AppPresentMode::fifo;
    bool fell_back = false;
};

constexpr const char* present_mode_name(AppPresentMode mode) {
    switch (mode) {
    case AppPresentMode::fifo: return "fifo";
    case AppPresentMode::mailbox: return "mailbox";
    case AppPresentMode::immediate: return "immediate";
    }
    return "fifo";
}

constexpr bool parse_present_mode(std::string_view text, AppPresentMode& mode) {
    if (text == "fifo") mode = AppPresentMode::fifo;
    else if (text == "mailbox") mode = AppPresentMode::mailbox;
    else if (text == "immediate") mode = AppPresentMode::immediate;
    else return false;
    return true;
}

// FIFO is guaranteed by Vulkan. The optional modes are never selected implicitly: mailbox keeps
// vsync with a replaceable queue, while immediate may tear and therefore requires an explicit request.
constexpr AppPresentModeSelection select_present_mode(AppPresentMode requested,
                                                       bool mailbox_supported,
                                                       bool immediate_supported) {
    if (requested == AppPresentMode::mailbox && !mailbox_supported)
        return {AppPresentMode::fifo, true};
    if (requested == AppPresentMode::immediate && !immediate_supported)
        return {AppPresentMode::fifo, true};
    return {requested, false};
}

} // namespace prosper::frontend
