#include "window_controls.hpp"

#include <cstdio>

using prosper::frontend::AppWindowCommand;
using prosper::frontend::AppWindowControls;
using prosper::frontend::AppWindowKey;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

int main() {
    std::printf("== test_prosper_app_window_controls ==\n");

    AppWindowControls controls;
    AppWindowKey key{};
    key.app_window = true;
    key.pressed = true;
    key.f11 = true;
    CHECK(controls.handle_key(key) == AppWindowCommand::toggle_fullscreen,
          "F11 toggles fullscreen");

    key.repeat = true;
    CHECK(controls.handle_key(key) == AppWindowCommand::none,
          "held F11 does not toggle repeatedly");

    key = {};
    key.app_window = true;
    key.pressed = true;
    key.enter = true;
    key.alt = true;
    CHECK(controls.handle_key(key) == AppWindowCommand::toggle_fullscreen,
          "Alt+Enter toggles fullscreen");
    CHECK(!controls.guest_options_allowed(),
          "Alt+Enter is not forwarded as the guest Options button");
    key.enter = false;
    key.alt = false;
    CHECK(controls.handle_key(key) == AppWindowCommand::none &&
              !controls.guest_options_allowed(),
          "releasing Alt first does not leak the held Enter chord to the guest");
    key.enter = true;
    key.pressed = false;
    key.app_window = false;
    CHECK(controls.handle_key(key) == AppWindowCommand::none &&
              controls.guest_options_allowed(),
          "releasing Enter after focus moves ends host-shortcut suppression");

    key.pressed = true;
    CHECK(controls.handle_key(key) == AppWindowCommand::none,
          "another SDL window cannot toggle the game window");

    key.app_window = true;
    key.alt = true;
    controls.handle_key(key);
    controls.release_host_shortcuts();
    CHECK(controls.guest_options_allowed(),
          "losing window focus clears host-shortcut suppression");

    key = {};
    key.app_window = true;
    key.pressed = true;
    key.escape = true;
    CHECK(controls.handle_key(key) == AppWindowCommand::quit,
          "Escape retains the quit command");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
}
