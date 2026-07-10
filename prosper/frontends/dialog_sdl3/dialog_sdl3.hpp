// dialog_sdl3.hpp — optional SDL3 dialog frontend for the interactive-UI HLE (PlatformUi, #347).
//
// Built only when -DPROSPER_APP=ON (needs SDL3 video). Installs a PlatformUi that shows real
// SDL_ShowMessageBox dialogs for the guest's MsgDialog / ErrorDialog, so a windowed session presents
// them instead of the core's headless auto-dismiss. ImeDialog text entry (which needs a custom text
// field, not a message box) is not handled yet, so it falls back to the core's headless default.
#pragma once

namespace prosper {

// Install the SDL3 PlatformUi as the active interactive-UI backend. Idempotent; call once at startup
// after register_builtin_hle(). Returns true.
bool install_sdl3_platform_ui();

// Uninstall (restore the headless default).
void shutdown_sdl3_platform_ui();

} // namespace prosper
