#pragma once

namespace prosper::frontend {

enum class AppWindowCommand {
    none,
    quit,
    toggle_pause,
    toggle_fullscreen,
};

struct AppWindowKey {
    bool app_window = false;
    bool pressed = false;
    bool repeat = false;
    bool escape = false;
    bool pause = false;
    bool f11 = false;
    bool enter = false;
    bool alt = false;
};

class AppWindowControls {
public:
    constexpr AppWindowCommand handle_key(const AppWindowKey& key) {
        // Key-up may be delivered after focus moved to a transient SDL window. It must still end a
        // host-owned Alt+Enter chord or the guest Options button would remain suppressed forever.
        if (key.enter && !key.pressed) suppress_enter_options_ = false;
        if (!key.app_window) return AppWindowCommand::none;
        if (key.enter) {
            if (key.pressed && key.alt) suppress_enter_options_ = true;
        }
        if (!key.pressed || key.repeat) return AppWindowCommand::none;
        if (key.escape) return AppWindowCommand::quit;
        if (key.pause) return AppWindowCommand::toggle_pause;
        if (key.f11 || (key.enter && key.alt)) return AppWindowCommand::toggle_fullscreen;
        return AppWindowCommand::none;
    }

    constexpr bool guest_options_allowed() const {
        return !suppress_enter_options_;
    }

    constexpr void set_app_focus(bool focused) {
        app_focused_ = focused;
    }

    // SDL can omit the key-up event while focus is elsewhere. Reconcile against its keyboard
    // snapshot only while the app owns focus: a fullscreen transition must not expose an Enter
    // key that is still physically held as the guest Options button.
    constexpr void reconcile_enter(bool enter_down) {
        if (app_focused_ && !enter_down) suppress_enter_options_ = false;
    }

private:
    bool suppress_enter_options_ = false;
    bool app_focused_ = true;
};

} // namespace prosper::frontend
