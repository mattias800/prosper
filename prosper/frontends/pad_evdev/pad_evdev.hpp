// pad_evdev.hpp — optional zero-dependency Linux controller frontend for the libScePad HLE.
//
// Reads a real gamepad through evdev (/dev/input/event*) and installs itself as the active
// PadBackend (src/input/pad.hpp). Lives OUTSIDE prosper_core — the headless core stays free of any
// /dev/input code. SDL3 (frontends/pad_sdl3) is the cross-platform primary; this is a lean,
// dependency-free alternative on Linux. On non-Linux the install is a no-op returning false.
#pragma once

namespace prosper {

// Install the evdev PadBackend (Linux only). Returns true if installed. Idempotent.
bool install_evdev_pad_backend();
// Restore the core default (neutral) backend.
void shutdown_evdev_pad_backend();

} // namespace prosper
