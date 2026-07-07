// pad_evdev.cpp — host controller backend.
//
// Linux: reads a real gamepad through evdev (/dev/input/event*), using the kernel's *standardized*
// gamepad button/axis codes (BTN_SOUTH.., ABS_X..). This is the same abstraction SDL's evdev
// backend sits on, with no external dependency — fitting the project's hermetic build. Axis ranges
// are read per-device (EVIOCGABS) and normalized to Sony's 0..255. The seam (pad_backend_poll) is
// platform-agnostic, so an SDL3 or Windows-XInput backend can replace this file's body later without
// touching the HLE or the pure mapping in pad.cpp.
//
// Other platforms: a no-op backend (no device) so the build stays green and boot still runs.
//
// CONFIDENCE: MED — the standard Linux gamepad mapping is correct for well-behaved drivers
// (hid-playstation, xpad, hid-generic); exotic pads may differ. Untestable in WSL (no /dev/input
// passthrough); the mapping *math* is unit-tested via the pure core.
#include "pad.hpp"

#ifdef __linux__

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <linux/input.h>

namespace prosper::input {
namespace {

constexpr int kMaxPads = 4;

inline bool test_bit(const unsigned long* arr, int bit) {
    return (arr[bit / (8 * sizeof(long))] >> (bit % (8 * sizeof(long)))) & 1UL;
}

struct AxisCal { int min = 0, max = 255; bool known = false; };

// One opened controller. Holds the live decoded state so a poll only needs to drain new events.
struct Pad {
    int      fd = -1;
    bool     tried = false;      // did we attempt to open this index yet
    uint32_t buttons = 0;
    int      lx = 0, ly = 0, rx = 0, ry = 0;   // raw stick axis values
    int      l2 = 0, r2 = 0;                    // raw trigger axis values
    int      hat_x = 0, hat_y = 0;              // d-pad hat (-1/0/+1)
    AxisCal  cal_lx, cal_ly, cal_rx, cal_ry, cal_l2, cal_r2;
};

std::mutex     g_mtx;
Pad            g_pads[kMaxPads];

// Map a Linux gamepad BTN_* code to a Sony button bit (0 = ignore).
uint32_t sony_button_for(int code) {
    switch (code) {
        case BTN_SOUTH:  return SCE_PAD_BUTTON_CROSS;
        case BTN_EAST:   return SCE_PAD_BUTTON_CIRCLE;
        case BTN_NORTH:  return SCE_PAD_BUTTON_TRIANGLE;
        case BTN_WEST:   return SCE_PAD_BUTTON_SQUARE;
        case BTN_TL:     return SCE_PAD_BUTTON_L1;
        case BTN_TR:     return SCE_PAD_BUTTON_R1;
        case BTN_TL2:    return SCE_PAD_BUTTON_L2;   // digital fallback (also derived from ABS_Z)
        case BTN_TR2:    return SCE_PAD_BUTTON_R2;
        case BTN_START:  return SCE_PAD_BUTTON_OPTIONS;
        case BTN_SELECT: return SCE_PAD_BUTTON_TOUCH_PAD;   // touchpad click on modern PS drivers
        case BTN_THUMBL: return SCE_PAD_BUTTON_L3;
        case BTN_THUMBR: return SCE_PAD_BUTTON_R3;
        case BTN_DPAD_UP:    return SCE_PAD_BUTTON_UP;
        case BTN_DPAD_DOWN:  return SCE_PAD_BUTTON_DOWN;
        case BTN_DPAD_LEFT:  return SCE_PAD_BUTTON_LEFT;
        case BTN_DPAD_RIGHT: return SCE_PAD_BUTTON_RIGHT;
        default:         return 0;
    }
}

void read_axis_cal(int fd, int code, AxisCal& c) {
    struct input_absinfo ai;
    if (ioctl(fd, EVIOCGABS(code), &ai) == 0 && ai.maximum > ai.minimum) {
        c.min = ai.minimum; c.max = ai.maximum; c.known = true;
    }
}

// Open the index-th device that advertises a gamepad (BTN_SOUTH) key. Returns fd or -1.
int open_gamepad(int index) {
    DIR* d = opendir("/dev/input");
    if (!d) return -1;
    int seen = -1, out = -1;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        if (strncmp(e->d_name, "event", 5) != 0) continue;
        char path[300];
        snprintf(path, sizeof(path), "/dev/input/%s", e->d_name);
        int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) continue;
        unsigned long keybits[(KEY_MAX / (8 * sizeof(long))) + 1];
        memset(keybits, 0, sizeof(keybits));
        bool is_pad = ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(keybits)), keybits) >= 0 &&
                      test_bit(keybits, BTN_GAMEPAD);   // BTN_GAMEPAD == BTN_SOUTH
        if (is_pad && ++seen == index) { out = fd; break; }
        close(fd);
    }
    closedir(d);
    return out;
}

void init_pad(Pad& p, int index) {
    p.tried = true;
    p.fd = open_gamepad(index);
    if (p.fd < 0) return;
    read_axis_cal(p.fd, ABS_X,  p.cal_lx);
    read_axis_cal(p.fd, ABS_Y,  p.cal_ly);
    read_axis_cal(p.fd, ABS_RX, p.cal_rx);
    read_axis_cal(p.fd, ABS_RY, p.cal_ry);
    read_axis_cal(p.fd, ABS_Z,  p.cal_l2);
    read_axis_cal(p.fd, ABS_RZ, p.cal_r2);
    // Seed stick centers so an untouched stick reads centered before its first event.
    p.lx = (p.cal_lx.min + p.cal_lx.max) / 2;
    p.ly = (p.cal_ly.min + p.cal_ly.max) / 2;
    p.rx = (p.cal_rx.min + p.cal_rx.max) / 2;
    p.ry = (p.cal_ry.min + p.cal_ry.max) / 2;
    p.l2 = p.cal_l2.min;
    p.r2 = p.cal_r2.min;
}

void drain_events(Pad& p) {
    struct input_event ev;
    ssize_t n;
    while ((n = read(p.fd, &ev, sizeof(ev))) == (ssize_t)sizeof(ev)) {
        if (ev.type == EV_KEY) {
            uint32_t bit = sony_button_for(ev.code);
            if (bit) {
                if (ev.value) p.buttons |= bit;
                else          p.buttons &= ~bit;
            }
        } else if (ev.type == EV_ABS) {
            switch (ev.code) {
                case ABS_X:  p.lx = ev.value; break;
                case ABS_Y:  p.ly = ev.value; break;
                case ABS_RX: p.rx = ev.value; break;
                case ABS_RY: p.ry = ev.value; break;
                case ABS_Z:  p.l2 = ev.value; break;
                case ABS_RZ: p.r2 = ev.value; break;
                case ABS_HAT0X: p.hat_x = (ev.value > 0) - (ev.value < 0); break;
                case ABS_HAT0Y: p.hat_y = (ev.value > 0) - (ev.value < 0); break;
                default: break;
            }
        }
    }
    if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
        // Device removed / error: close so a later poll can re-open (hot-plug).
        close(p.fd);
        p.fd = -1;
        p.tried = false;
    }
}

uint8_t norm(int raw, const AxisCal& c) {
    return c.known ? pad_axis_u8(raw, c.min, c.max) : (uint8_t)0x80;
}

} // namespace

bool pad_backend_available() { return true; }

void pad_backend_poll(int index, HostPadState& out) {
    out = HostPadState{};   // neutral, disconnected
    if (index < 0 || index >= kMaxPads) return;

    std::lock_guard<std::mutex> lk(g_mtx);
    Pad& p = g_pads[index];
    if (!p.tried) init_pad(p, index);
    if (p.fd < 0) return;   // no device

    drain_events(p);
    if (p.fd < 0) return;   // disconnected while draining

    // Merge hat d-pad into the button bitmap (controllers report the d-pad either as a hat axis or
    // as BTN_DPAD_* keys; support both without double-counting — recompute the hat bits each poll).
    uint32_t b = p.buttons & ~(SCE_PAD_BUTTON_UP | SCE_PAD_BUTTON_DOWN |
                               SCE_PAD_BUTTON_LEFT | SCE_PAD_BUTTON_RIGHT);
    // Keep any BTN_DPAD_* bits the events already set, then OR in the hat.
    b |= (p.buttons & (SCE_PAD_BUTTON_UP | SCE_PAD_BUTTON_DOWN |
                       SCE_PAD_BUTTON_LEFT | SCE_PAD_BUTTON_RIGHT));
    if (p.hat_y < 0) b |= SCE_PAD_BUTTON_UP;
    if (p.hat_y > 0) b |= SCE_PAD_BUTTON_DOWN;
    if (p.hat_x < 0) b |= SCE_PAD_BUTTON_LEFT;
    if (p.hat_x > 0) b |= SCE_PAD_BUTTON_RIGHT;

    out.left_x  = norm(p.lx, p.cal_lx);
    out.left_y  = norm(p.ly, p.cal_ly);
    out.right_x = norm(p.rx, p.cal_rx);
    out.right_y = norm(p.ry, p.cal_ry);
    out.l2      = norm(p.l2, p.cal_l2);
    out.r2      = norm(p.r2, p.cal_r2);
    // The L2/R2 digital bit trips as soon as the analog trigger leaves rest (matches real hardware).
    if (out.l2 > 0) b |= SCE_PAD_BUTTON_L2;
    if (out.r2 > 0) b |= SCE_PAD_BUTTON_R2;

    out.buttons   = b;
    out.connected = true;
}

} // namespace prosper::input

#else  // !__linux__  — no host backend on this platform yet (Windows XInput is a follow-up).

namespace prosper::input {
bool pad_backend_available() { return false; }
void pad_backend_poll(int /*index*/, HostPadState& out) { out = HostPadState{}; }
} // namespace prosper::input

#endif
