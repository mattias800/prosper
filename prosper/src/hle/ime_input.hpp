// IME keyboard input seam (issue #1093). A frontend (or a headless route) pushes keyboard keys,
// identified by USB HID usage id, into a queue that sceImeUpdate drains — invoking the guest's
// registered event handler per event. Mirrors pad.hpp's pad_set_backend seam for libScePad.
//
// Titles that read input through sceImeKeyboardOpen + sceImeUpdate (e.g. PPSA02664) never touch
// libScePad, so their keyboard input must arrive on this path. Common HID usage ids:
//   Enter=0x28  Space=0x2c  Escape=0x29  A..Z=0x04..0x1d  Right=0x4f Left=0x50 Down=0x51 Up=0x52
#pragma once
#include <cstdint>

namespace prosper {
// Push one keyboard key transition (USB HID usage id) into the IME event queue. Thread-safe;
// drained on the next sceImeUpdate on the guest thread. `down` = key press, else release.
void ime_push_key(uint16_t hid_usage, bool down);
}
