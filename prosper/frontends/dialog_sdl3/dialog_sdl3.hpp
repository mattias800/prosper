// dialog_sdl3.hpp — optional SDL3 dialog frontend for the interactive-UI HLE (PlatformUi, #347).
//
// Built only when -DPROSPER_APP=ON (needs SDL3 video). Installs a PlatformUi that shows real
// SDL_ShowMessageBox dialogs for the guest's MsgDialog / ErrorDialog, so a windowed session presents
// them instead of the core's headless auto-dismiss. Custom main-thread windows provide ImeDialog text
// entry, SaveData confirmations, and non-modal SaveData percentage progress.
#pragma once

namespace prosper {

// Install the SDL3 PlatformUi as the active interactive-UI backend. Idempotent; call once at startup
// after register_builtin_hle(). Returns true.
bool install_sdl3_platform_ui();

// Uninstall (restore the headless default).
void shutdown_sdl3_platform_ui();

// Pump pending interactive UI on the MAIN thread — call once per frame from the app's event/present
// loop. ImeDialog/SaveData UI (SDL windowing is main-thread only) runs here; MsgDialog/ErrorDialog use
// SDL_ShowMessageBox directly and don't need it. Safe to call when nothing is pending.
void sdl_platform_ui_pump();

} // namespace prosper
