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

// ScePadControllerInformation — scePadGetControllerInformation. 28 bytes (trailing reserve).
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

// ScePadExtendedControllerInformation. The standard-pad extension is zero; device-specific
// controllers interpret class_data according to capability.
struct ScePadExtendedControllerInformation {
    ScePadControllerInformation base;        // 0x00
    uint16_t pad_type1;                      // 0x1c
    uint16_t pad_type2;                      // 0x1e
    uint8_t  capability;                     // 0x20
    uint8_t  ext_pad[3];                     // 0x21 (align class-data union to 4)
    uint8_t  class_data[8];                  // 0x24
};                                           // 0x2c = 44 bytes

// ---- Pure mapping (no hardware, no globals) ---------------------------------------------------

// Fill a full ScePadData from a host snapshot. Zeroes the struct first (motion/touch reported as
// neutral: identity orientation quaternion, no touches) so no field is left uninitialized.
void pad_fill_data(ScePadData* out, const HostPadState& s, uint64_t timestamp, uint8_t connected_count);

// Fill ScePadControllerInformation for a DualSense-class pad.
void pad_fill_controller_info(ScePadControllerInformation* out, bool connected, uint8_t connected_count);

// Fill the exact extended controller-info layout for a standard DualSense-class pad.
void pad_fill_extended_controller_info(ScePadExtendedControllerInformation* out, bool connected,
                                       uint8_t connected_count);

// Normalize a raw axis reading in [min,max] to the Sony 0..255 range (clamped).
uint8_t pad_axis_u8(int raw, int min, int max);

// The digital L2/R2 button bit trips once the analog trigger passes a small rest threshold — this
// filters sensor noise / drift while staying responsive (real pads trip near the start of travel).
constexpr uint8_t kPadTriggerButtonThreshold = 0x08;   // ~3% of full travel

// Return the L2/R2 button bits for analog trigger values past the threshold (0 for both at rest).
// Shared by every backend so the analog->digital rule lives in one unit-tested place.
uint32_t pad_trigger_buttons(uint8_t l2, uint8_t r2);

// ---- Scripted input (PROSPER_PAD_SCRIPT) — pure, unit-tested -----------------------------------
// A timed controller sequence lets a headless run drive menus and gameplay with no host device
// (issue #163). These helpers are pure (no env, no clock) so parse + time evaluation is verifiable; the HLE
// (hle_pad.cpp) supplies getenv + the wall clock and anchors t=0 to the first pad poll.
// t_secs holds wall-clock seconds by default, a flip number when frame_anchored, or a pad-read number
// when read_anchored. `p300:cross` is independent of wall time and presentation rate, which keeps a
// route usable when synchronous rendering changes both. See #302.
//
// START AND HOLD ARE SEPARATE QUESTIONS, and #3449 is what happens when one unit answers both. A
// route wants its POSITION in flips, because a flip ordinal names the same moment in the guest's own
// sequence on every host. It wants its DURATION in wall-clock seconds, because guest key repeat is
// wall-clock: hold a menu Down long enough and the title walks several rows. A flip WINDOW encodes a
// duration only at the flip rate it was tuned on. `reach-performance-story.pad` was tuned at ~80
// flips/s and its menu ran at 15-33 flips/s on another host, so a press meant to last 0.15 s lasted
// 0.737 s, three Downs walked seven rows, and GTA V spent a whole measurement session in the wrong
// graphics mode while being reported as a Performance run.
//
// So an entry may anchor its start in one unit and its hold in another: `f2272+0.15:down` starts at
// flip 2272 and is held 0.15 SECONDS, whatever the flip rate turns out to be. hold_unit records
// which unit `hold_amount` is in; hold_unit == none means the entry uses `end` or the per-axis
// default, exactly as before. Flip pacing (#3379) narrows the spread but cannot close it: pacing
// only sleeps when a host is running FAST, and the rates that caused this were far below the paced
// cadence, where pacing is inert by construction.
enum class PadScriptHoldUnit : uint8_t {
    none = 0,   // no cross-unit hold: `end`, or the configured default for this entry's axis
    seconds,
    flips,
    reads,
};

struct PadScriptEntry {
    double t_secs;
    uint32_t button_mask;
    bool frame_anchored = false;
    bool read_anchored = false;
    double end = 0.0;  // exclusive explicit range end; 0 uses the configured default hold
    int8_t left_x = 0;   // -1=left, 0=neutral/unspecified, +1=right
    int8_t left_y = 0;   // -1=up,   0=neutral/unspecified, +1=down
    int8_t right_x = 0;
    int8_t right_y = 0;
    uint8_t axis_mask = 0;  // distinguishes an explicitly centered opposing pair from no axis input
    // A hold expressed in a unit OTHER than the start anchor's. Same-unit spellings ("f100+f12")
    // are folded into `end` at parse time, so these two fields are set only when the entry
    // genuinely cannot be decided from the current counters alone.
    PadScriptHoldUnit hold_unit = PadScriptHoldUnit::none;
    double hold_amount = 0.0;
};

enum PadScriptAxis : uint8_t {
    PAD_SCRIPT_LEFT_X  = 1u << 0,
    PAD_SCRIPT_LEFT_Y  = 1u << 1,
    PAD_SCRIPT_RIGHT_X = 1u << 2,
    PAD_SCRIPT_RIGHT_Y = 1u << 3,
};

struct PadScriptState {
    uint32_t button_mask = 0;
    int8_t left_x = 0;
    int8_t left_y = 0;
    int8_t right_x = 0;
    int8_t right_y = 0;
    uint8_t axis_mask = 0;
};

// Map a button name ("start"/"options"/"cross"/"x"/"up"/"touchpad"/... case as written) to its
// SCE_PAD_BUTTON_* bit; 0 if unknown. "start" and "options" both mean the PS5 Options button (the
// "Start" equivalent); "touchpad" is the DualSense touch-pad click, which titles bind as an
// ordinary button.
uint32_t pad_button_by_name(const std::string& name);

// Canonical '+'-joined names for a button mask. The output round-trips through the parser and uses a
// stable order so recorded routes have deterministic text.
std::string pad_button_names(uint32_t mask);

// ---- Recording analog sticks as DIRECTIONS ----------------------------------------------------
//
// The script vocabulary is full-deflection only ("left-stick-left"), so recording treats each stick
// as a d-pad: past the dead zone in one direction, or centered. That is a deliberate fidelity limit,
// not an oversight -- a gentle tilt records and replays as a full push, so a route that depends on
// fine analog control will not reproduce its exact path. Recording nothing at all was the previous
// behaviour and is strictly worse: a stick-driven route replayed as standing still.
//
// Bits are the PAD_SCRIPT_* axis flags paired with a sign bit, so one integer compares cheaply in
// the recorder's change detection.
enum : uint32_t {
    PAD_REC_LEFT_X_NEG = 1u << 0, PAD_REC_LEFT_X_POS = 1u << 1,
    PAD_REC_LEFT_Y_NEG = 1u << 2, PAD_REC_LEFT_Y_POS = 1u << 3,
    PAD_REC_RIGHT_X_NEG = 1u << 4, PAD_REC_RIGHT_X_POS = 1u << 5,
    PAD_REC_RIGHT_Y_NEG = 1u << 6, PAD_REC_RIGHT_Y_POS = 1u << 7,
};

// Sony axis bytes are 0=min, 0x80=center, 0xff=max. The dead zone is generous on purpose: a resting
// stick drifts, and a recorder that emits an interval every time it wobbles produces a route full of
// one-flip noise entries that replay as real input.
inline constexpr uint8_t kPadRecordDeadZone = 48;

// -1, 0 or +1 for one axis byte.
constexpr int pad_axis_direction(uint8_t value, uint8_t dead_zone = kPadRecordDeadZone) {
    const int centered = static_cast<int>(value) - 0x80;
    if (centered <= -static_cast<int>(dead_zone)) return -1;
    if (centered >= static_cast<int>(dead_zone)) return 1;
    return 0;
}

// Pack the four axes into one comparable mask.
constexpr uint32_t pad_stick_mask(uint8_t left_x, uint8_t left_y,
                                  uint8_t right_x, uint8_t right_y,
                                  uint8_t dead_zone = kPadRecordDeadZone) {
    uint32_t mask = 0;
    const int lx = pad_axis_direction(left_x, dead_zone);
    const int ly = pad_axis_direction(left_y, dead_zone);
    const int rx = pad_axis_direction(right_x, dead_zone);
    const int ry = pad_axis_direction(right_y, dead_zone);
    if (lx < 0) mask |= PAD_REC_LEFT_X_NEG; else if (lx > 0) mask |= PAD_REC_LEFT_X_POS;
    if (ly < 0) mask |= PAD_REC_LEFT_Y_NEG; else if (ly > 0) mask |= PAD_REC_LEFT_Y_POS;
    if (rx < 0) mask |= PAD_REC_RIGHT_X_NEG; else if (rx > 0) mask |= PAD_REC_RIGHT_X_POS;
    if (ry < 0) mask |= PAD_REC_RIGHT_Y_NEG; else if (ry > 0) mask |= PAD_REC_RIGHT_Y_POS;
    return mask;
}

// '+'-joined direction names, in the exact vocabulary parse_pad_script accepts, so a recorded route
// round-trips through the parser.
std::string pad_stick_names(uint32_t stick_mask);

// Recording uses the same explicit-range syntax accepted by PROSPER_PAD_SCRIPT. Keep transition
// tracking pure so interval boundaries and axis prefixes can be tested without an output file or
// a controller backend. The caller supplies either the current display-flip or successful-read
// position; observe() returns a completed interval when the button state changes.
enum class PadRecordAxis : uint8_t {
    display_flip,
    pad_read,
};

class PadRecordState {
public:
    explicit PadRecordState(PadRecordAxis axis = PadRecordAxis::display_flip) : axis_(axis) {}
    // `sticks` is a pad_stick_mask(). A change in EITHER buttons or stick direction closes the
    // current interval, so a route records "walked left, then pressed cross" as two entries.
    std::string observe(int64_t position, uint32_t buttons, uint32_t sticks = 0);

private:
    PadRecordAxis axis_;
    uint32_t previous_ = 0;
    uint32_t previous_sticks_ = 0;
    int64_t start_ = 0;
};

// Parse ';'- or newline-separated entries. Actions are '+'-joined button names and/or full-stick
// directions such as "left-stick-left" and "right-stick-up". A time token starting with 'f' is
// frame-anchored (flips since the first pad poll); one starting with 'p' is pad-read-anchored
// (successful scePadRead/scePadReadState calls, numbered from zero). All anchors accept explicit ranges
// ("3-4.5:cross", "f300-340:cross",
// "p1200-1240:cross"); '#' starts a comment. Malformed entries and entries with no recognized action
// are dropped.
//
// A '+' in the anchor gives a HOLD instead of an end, and the hold names its unit with the SAME
// rule the start anchor uses: a bare number is seconds, 'f' is flips, 'p' is pad reads. So
// "f2272+0.15:down" starts at flip 2272 and is held 0.15 s, "f2272+p12:down" holds it for 12 guest
// pad reads -- and "f100+12" is twelve SECONDS from flip 100, not twelve flips, because the rule
// does not change halfway through an entry. A hold in the START's own unit ("f100+f12") is folded
// straight into an explicit range and needs no runner state. A hold of
// zero, a non-integer flip/read count, and an anchor carrying BOTH '-' and '+' are all rejected --
// a press nothing observes is #3267's failure, and a route that half-parses is #2439's.
// The '+' before the ':' is unambiguous against the '+' that joins actions after it.
std::vector<PadScriptEntry> parse_pad_script(const std::string& spec);

// Parse an inline script, or load and parse one from disk when `source` starts with '@'. Relative
// paths use the process working directory. Returns an empty vector and describes file I/O errors.
std::vector<PadScriptEntry> load_pad_script(const std::string& source, std::string* error = nullptr);

// Diagnostic for a route that loaded WITHOUT error and still parsed to zero entries (#2439). Such a
// route is configured-but-inert, and it used to announce itself exactly as a working route does:
// parse_pad_script drops every unparseable token silently, so the only output is the frame-0 neutral
// line that a correctly-loaded route ALSO emits before its first press window opens. "Route loaded,
// first press is still 500 flips away" and "route never existed" printed identically.
//
// Returns the text to log, or "" when the route has entries and nothing needs saying. The '@' hint is
// added only when `source` looks like a file path, so an intentionally empty inline route is reported
// without being told it meant something else.
std::string pad_script_empty_route_warning(const std::string& source, bool parsed_empty);

// Full scripted controller state now. Active entries combine buttons and full-deflection stick
// directions; opposing directions on one axis cancel to an explicitly centered axis.
//
// PURE, and therefore BLIND to cross-unit holds: "hold 0.15 s from flip 2272" cannot be decided
// from the current counters, because nothing here knows when flip 2272 happened. Such entries are
// reported inactive. Use PadScriptRunner for any script that may contain them -- which is every
// script a route file can express. This overload remains for the same-unit entry kinds and for the
// tests that pin their exact window arithmetic.
PadScriptState pad_script_state_at(const std::vector<PadScriptEntry>& script, double elapsed_secs,
                                   double hold_secs, int64_t frame_count = -1, int64_t frame_hold = 8,
                                   int64_t read_count = -1, int64_t read_hold = 8);

// True if `script` contains at least one entry whose hold is in a different unit from its start, so
// a caller can tell "this needs the stateful runner" from "the pure function is enough".
bool pad_script_needs_runner(const std::vector<PadScriptEntry>& script);

// What a cross-unit entry latched when its start anchor was crossed. One per script entry.
struct PadScriptLatch {
    bool armed = false;      // the start anchor has been crossed; the counters below are valid
    bool expired = false;    // the hold has run out. Never re-arms: counters are monotone, and a
                             // press that came back to life after its window would be a new press.
    double elapsed = 0.0;
    int64_t frame = 0;
    int64_t read = 0;
};

// Evaluates a script INCLUDING cross-unit holds.
//
// Stateful for exactly one reason: to hold a press for 0.15 s starting at flip 2272 you must record
// the clock at the moment flip 2272 arrives, and the current counters do not carry that. The latch
// is the whole of the state; every other entry kind still goes through the pure evaluation above.
//
// Latches are addressed by the entry's INDEX. PROSPER_PAD_SCRIPT_RELOAD is documented as appending
// future windows to a live route, and an append leaves every existing index where it was, so the
// vector is simply resized and the common prefix keeps its latches. Reordering or deleting entries
// mid-run moves a latch onto a different press; that is outside the append contract and is called
// out here rather than guarded against, because guarding would mean identifying entries that a
// route is free to spell identically.
class PadScriptRunner {
public:
    PadScriptState state_at(const std::vector<PadScriptEntry>& script, double elapsed_secs,
                            double hold_secs, int64_t frame_count = -1, int64_t frame_hold = 8,
                            int64_t read_count = -1, int64_t read_hold = 8);

private:
    std::vector<PadScriptLatch> latches_;
};

// Overlay a scripted state onto a host snapshot. Buttons combine with the backend; only explicitly
// active script axes replace their host values.
void pad_apply_script_state(HostPadState& target, const PadScriptState& scripted);

// Buttons held now: OR of every entry whose window contains the current time. A seconds-anchored entry
// matches [t, t+hold_secs) against `elapsed_secs`; a frame-anchored entry matches [f, f+frame_hold)
// against `frame_count`; a read-anchored entry matches [p, p+read_hold) against `read_count`.
// Pass -1 when a count is unavailable so entries on that axis never fire. An explicit range uses its
// exclusive `end`; point entries use the configured hold for their axis.
uint32_t pad_script_buttons_at(const std::vector<PadScriptEntry>& script, double elapsed_secs, double hold_secs,
                               int64_t frame_count = -1, int64_t frame_hold = 8,
                               int64_t read_count = -1, int64_t read_hold = 8);

// ---- Pluggable host backend (mirrors AudioSink / audio_set_sink) -------------------------------
//
// prosper_core is HEADLESS: it ships a default backend that reports one connected neutral pad, so
// titles with a controller-presence gate can progress and the HLE contract remains dependency-free
// and unit-testable. A concrete host
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

// Install a backend. Non-owning; pass nullptr to restore the built-in connected-neutral default.
void pad_set_backend(PadBackend* backend);
// The active backend (never null — the neutral default is returned when none is installed).
PadBackend* pad_backend();

} // namespace prosper::input
