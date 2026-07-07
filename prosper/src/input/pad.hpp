// pad.hpp — host game-controller input for the libScePad HLE.
//
// Two layers, deliberately split so the mapping is verifiable without hardware:
//   1. A PURE mapping core (pad_fill_data / pad_fill_controller_info) that turns a host-neutral
//      HostPadState into Sony's exact ScePadData / ScePadControllerInformation byte layout. This is
//      unit-tested (struct offsets + field mapping) with no device present.
//   2. A host BACKEND (pad_backend_poll) that reads a real controller. Implemented for Linux via
//      evdev (zero external deps); a no-op elsewhere. The backend degrades gracefully to
//      "no device" so the pure core and the whole boot still run on a machine with no gamepad.
//
// Layout cross-checked against the Kyty PS5 emulator (source/emulator/src/Controller.cpp) and the
// published Sony ScePadData; asserted byte-for-byte in pad.cpp. CONFIDENCE: HIGH (layout), MED
// (per-device evdev axis/button mapping — standard Linux gamepad codes, but device drivers vary).
#pragma once
#include <cstdint>

namespace prosper::input {

// Sony ScePadButtonDataOffset bitmask (matches Kyty PAD_BUTTON_* and the DS4/DS5 report).
enum ScePadButton : uint32_t {
    SCE_PAD_BUTTON_L3        = 0x00000002,
    SCE_PAD_BUTTON_R3        = 0x00000004,
    SCE_PAD_BUTTON_OPTIONS   = 0x00000008,
    SCE_PAD_BUTTON_UP        = 0x00000010,
    SCE_PAD_BUTTON_RIGHT     = 0x00000020,
    SCE_PAD_BUTTON_DOWN      = 0x00000040,
    SCE_PAD_BUTTON_LEFT      = 0x00000080,
    SCE_PAD_BUTTON_L2        = 0x00000100,
    SCE_PAD_BUTTON_R2        = 0x00000200,
    SCE_PAD_BUTTON_L1        = 0x00000400,
    SCE_PAD_BUTTON_R1        = 0x00000800,
    SCE_PAD_BUTTON_TRIANGLE  = 0x00001000,
    SCE_PAD_BUTTON_CIRCLE    = 0x00002000,
    SCE_PAD_BUTTON_CROSS     = 0x00004000,
    SCE_PAD_BUTTON_SQUARE    = 0x00008000,
    SCE_PAD_BUTTON_TOUCH_PAD = 0x00100000,
};

// Host-neutral controller snapshot. Sticks/triggers use the Sony 8-bit convention directly:
// sticks 0..255 (0x80 = centered, up/left = 0), triggers 0..255 (0 = released). A backend fills
// this; the pure mapping copies it into ScePadData. Default = a valid, centered, disconnected pad.
struct HostPadState {
    uint32_t buttons = 0;         // ScePadButton bitmask
    uint8_t  left_x  = 0x80;      // 0=left   0x80=center  0xff=right
    uint8_t  left_y  = 0x80;      // 0=up     0x80=center  0xff=down
    uint8_t  right_x = 0x80;
    uint8_t  right_y = 0x80;
    uint8_t  l2      = 0;         // L2 analog travel 0..255
    uint8_t  r2      = 0;         // R2 analog travel 0..255
    bool     connected = false;   // is a physical device present
};

// ---- Sony ABI structs (exact byte layout; asserted in pad.cpp) --------------------------------

// ScePadData — the per-poll controller report (scePadReadState / scePadRead). 120 bytes.
struct ScePadData {
    uint32_t buttons;                       // 0x00
    uint8_t  left_stick_x;                  // 0x04
    uint8_t  left_stick_y;                  // 0x05
    uint8_t  right_stick_x;                 // 0x06
    uint8_t  right_stick_y;                 // 0x07
    uint8_t  analog_buttons_l2;             // 0x08
    uint8_t  analog_buttons_r2;             // 0x09
    uint8_t  padding[2];                    // 0x0a
    float    orientation_x;                 // 0x0c  quaternion (gyro fusion)
    float    orientation_y;                 // 0x10
    float    orientation_z;                 // 0x14
    float    orientation_w;                 // 0x18
    float    acceleration_x;                // 0x1c
    float    acceleration_y;                // 0x20
    float    acceleration_z;                // 0x24
    float    angular_velocity_x;            // 0x28
    float    angular_velocity_y;            // 0x2c
    float    angular_velocity_z;            // 0x30
    uint8_t  touch_num;                     // 0x34
    uint8_t  touch_reserve[3];              // 0x35
    uint32_t touch_reserve1;                // 0x38
    uint16_t touch0_x;                      // 0x3c
    uint16_t touch0_y;                      // 0x3e
    uint8_t  touch0_id;                     // 0x40
    uint8_t  touch0_reserve[3];             // 0x41
    uint16_t touch1_x;                      // 0x44
    uint16_t touch1_y;                      // 0x46
    uint8_t  touch1_id;                     // 0x48
    uint8_t  touch1_reserve[3];             // 0x49
    uint8_t  connected;                     // 0x4c  (bool)
    uint8_t  connected_pad[3];              // 0x4d  (align timestamp to 8)
    uint64_t timestamp;                     // 0x50
    uint32_t ext_unit_id;                   // 0x58
    uint8_t  ext_unit_reserve[1];           // 0x5c
    uint8_t  ext_unit_data_length;          // 0x5d
    uint8_t  ext_unit_data[10];             // 0x5e
    uint8_t  connected_count;               // 0x68
    uint8_t  reserve[2];                    // 0x69
    uint8_t  device_unique_data_len;        // 0x6b
    uint8_t  device_unique_data[12];        // 0x6c
};                                          // 0x78 = 120 bytes

// ScePadControllerInformation — scePadGetControllerInformation. 32 bytes (trailing reserve).
struct ScePadControllerInformation {
    float    touch_pixel_density;           // 0x00
    uint16_t touch_resolution_x;            // 0x04
    uint16_t touch_resolution_y;            // 0x06
    uint8_t  stick_dead_zone_left;          // 0x08
    uint8_t  stick_dead_zone_right;         // 0x09
    uint8_t  connection_type;               // 0x0a
    uint8_t  connected_count;               // 0x0b
    uint8_t  connected;                     // 0x0c (bool)
    uint8_t  info_pad[3];                    // 0x0d (align device_class to 4)
    int32_t  device_class;                  // 0x10
    uint8_t  reserve[12];                    // 0x14
};                                          // 0x20 = 32 bytes

// ---- Pure mapping (no hardware, no globals) ---------------------------------------------------

// Fill a full ScePadData from a host snapshot. Zeroes the struct first (motion/touch reported as
// neutral: identity orientation quaternion, no touches) so no field is left uninitialized.
void pad_fill_data(ScePadData* out, const HostPadState& s, uint64_t timestamp, uint8_t connected_count);

// Fill ScePadControllerInformation for a DualSense-class pad.
void pad_fill_controller_info(ScePadControllerInformation* out, bool connected, uint8_t connected_count);

// Normalize a raw axis reading in [min,max] to the Sony 0..255 range (clamped).
uint8_t pad_axis_u8(int raw, int min, int max);

// ---- Host backend (platform-specific; see pad_evdev.cpp) --------------------------------------

// True if a real input backend is compiled in for this platform (Linux evdev). When false,
// pad_backend_poll always reports "no device".
bool pad_backend_available();

// Poll controller `index` (0-based) into `out`. Opens the device lazily on first call. Sets
// out.connected=false and leaves neutral values if no device is present. Thread-safe.
void pad_backend_poll(int index, HostPadState& out);

} // namespace prosper::input
