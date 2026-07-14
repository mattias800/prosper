// pad_sdl3.hpp — optional SDL3 game-controller frontend for the libScePad HLE.
//
// Built only when -DPROSPER_PAD_SDL3=ON. Installs an SDL_Gamepad-backed PadBackend so a real
// controller drives the guest. SDL_Gamepad is cross-platform (Windows/Linux/macOS) and ships a
// controller-mapping database, so it needs no per-device button/axis guesswork. The core HLE
// (src/hle/hle_pad.cpp) has no knowledge of SDL — this frontend lives outside prosper_core and
// plugs in at runtime.
#pragma once

namespace prosper {

// Initialise SDL gamepad and install the SDL3 PadBackend as the active backend. Idempotent.
// Returns true on success; false (leaving the core's connected-neutral default active) if SDL
// gamepad could not be initialised. Call once at startup after register_builtin_hle().
bool install_sdl3_pad_backend();

// Uninstall the SDL3 backend (restores the neutral default) and shut SDL gamepad down.
void shutdown_sdl3_pad_backend();

} // namespace prosper
