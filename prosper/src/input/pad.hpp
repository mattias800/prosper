// pad.hpp — host game-controller input for the libScePad HLE.
//
// Two layers, deliberately split so the mapping is verifiable without hardware:
//   1. A PURE mapping core (pad_fill_data / pad_fill_controller_info) that turns a host-neutral
//      HostPadState into Sony's exact ScePadData / ScePadControllerInformation byte layout. This is
//      unit-tested (struct offsets + field mapping) with no device present.
//   2. A pluggable PadBackend (interface below) that reads a real controller. prosper_core ships a
//      dependency-free NEUTRAL default; a host frontend under frontends/ (SDL3 gamepad — the
//      cross-platform primary — or the zero-dep Linux evdev reader) installs itself via
//      pad_set_backend() from the harness. So prosper_core stays free of host/device code and the
//      whole boot runs on a machine with no gamepad.
//
// Layout cross-checked against the Kyty PS5 emulator (source/emulator/src/Controller.cpp) and the
// published Sony ScePadData; asserted byte-for-byte in pad.cpp. CONFIDENCE: HIGH (layout).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

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
    uint8_t  reserve[8];                     // 0x14
};                                          // 0x1c = 28 bytes (Sony's real size — reserve[8], NOT [12];
                                            // a 32-byte struct overran the game's 28-byte stack buffer
                                            // into its stack canary -> __stack_chk_fail crash, #283)

// ---- Pure mapping (no hardware, no globals) ---------------------------------------------------

// Fill a full ScePadData from a host snapshot. Zeroes the struct first (motion/touch reported as
// neutral: identity orientation quaternion, no touches) so no field is left uninitialized.
void pad_fill_data(ScePadData* out, const HostPadState& s, uint64_t timestamp, uint8_t connected_count);

// Fill ScePadControllerInformation for a DualSense-class pad.
void pad_fill_controller_info(ScePadControllerInformation* out, bool connected, uint8_t connected_count);

// Normalize a raw axis reading in [min,max] to the Sony 0..255 range (clamped).
uint8_t pad_axis_u8(int raw, int min, int max);

// The digital L2/R2 button bit trips once the analog trigger passes a small rest threshold — this
// filters sensor noise / drift while staying responsive (real pads trip near the start of travel).
constexpr uint8_t kPadTriggerButtonThreshold = 0x08;   // ~3% of full travel

// Return the L2/R2 button bits for analog trigger values past the threshold (0 for both at rest).
// Shared by every backend so the analog->digital rule lives in one unit-tested place.
uint32_t pad_trigger_buttons(uint8_t l2, uint8_t r2);

// ---- Scripted input (PROSPER_PAD_SCRIPT) — pure, unit-tested -----------------------------------
// A timed button sequence lets a headless run drive menus with no host device (issue #163). These
// three helpers are pure (no env, no clock) so the parse + time-eval logic is verifiable; the HLE
// (hle_pad.cpp) supplies getenv + the wall clock and anchors t=0 to the first input poll.
// t_secs holds a wall-clock time (default) OR, when frame_anchored, a FRAME NUMBER (flips since the
// first poll). Frame-anchoring is boot-speed-invariant: `f300:cross` fires at the same game state on a
// fast GPU and slow llvmpipe, where a wall-clock `10:cross` would drift — essential for deterministic
// menu-reach across builds (bisect/regression repro). See #302.
struct PadScriptEntry {
    double t_secs;
    uint32_t button_mask;
    bool frame_anchored = false;
    double end = 0.0;  // exclusive explicit range end; 0 uses the configured default hold
};

// Map a button name ("start"/"options"/"cross"/"x"/"up"/... case as written) to its SCE_PAD_BUTTON_*
// bit; 0 if unknown. "start" and "options" both mean the PS5 Options button (the "Start" equivalent).
uint32_t pad_button_by_name(const std::string& name);

// Parse ';'- or newline-separated entries. An entry whose time token starts with 'f' is
// FRAME-anchored: "f300:cross" fires at flip 300 since the first poll. Both anchors accept explicit
// ranges ("3-4.5:cross", "f300-340:cross"); '#' starts a comment. Malformed entries and entries with
// no recognized button are dropped.
std::vector<PadScriptEntry> parse_pad_script(const std::string& spec);

// Parse an inline script, or load and parse one from disk when `source` starts with '@'. Relative
// paths use the process working directory. Returns an empty vector and describes file I/O errors.
std::vector<PadScriptEntry> load_pad_script(const std::string& source, std::string* error = nullptr);

// Buttons held now: OR of every entry whose window contains the current time. A seconds-anchored entry
// matches [t, t+hold_secs) against `elapsed_secs`; a frame-anchored entry matches [f, f+frame_hold)
// against `frame_count` (pass -1 when no frame count is available -> frame entries never fire).
// An explicit range uses its exclusive `end`; point entries use hold_secs/frame_hold.
uint32_t pad_script_buttons_at(const std::vector<PadScriptEntry>& script, double elapsed_secs, double hold_secs,
                               int64_t frame_count = -1, int64_t frame_hold = 8);

// ---- Pluggable host backend (mirrors AudioSink / audio_set_sink) -------------------------------
//
// prosper_core is HEADLESS: it ships a default backend that reports a neutral, disconnected pad, so
// the HLE contract is fully defined with no dependencies and is unit-testable. A concrete host
// backend (SDL3 gamepad, or the zero-dep Linux evdev reader) lives OUTSIDE prosper_core, under
// frontends/, and installs itself via pad_set_backend() from the harness (boot_trace). This keeps
// prosper_core free of any host-input / device code, and makes a future libretro core trivial — it
// installs its own backend that reads retro_input_state_t instead of touching /dev/input.
struct PadBackend {
    virtual ~PadBackend() = default;
    // Poll controller `index` (0-based) into `out`. Return true if a physical device is present
    // (and out.connected/state are filled); false if none (leave `out` neutral/disconnected).
    // Called on the guest's input-polling thread; implementations must be thread-safe.
    virtual bool poll(int index, HostPadState& out) = 0;
};

// Install a backend. Non-owning; pass nullptr to restore the built-in neutral/disconnected default.
void pad_set_backend(PadBackend* backend);
// The active backend (never null — the neutral default is returned when none is installed).
PadBackend* pad_backend();

} // namespace prosper::input
