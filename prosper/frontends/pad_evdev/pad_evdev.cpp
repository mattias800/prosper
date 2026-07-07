// pad_evdev.cpp — Linux evdev PadBackend (optional, zero-dependency frontend).
//
// Reads a real gamepad through evdev using the kernel's *standardized* gamepad button/axis codes
// (BTN_SOUTH.., ABS_X..) — the same abstraction SDL's evdev backend sits on, with no external
// dependency. Per-device axis ranges are read (EVIOCGABS) and normalized to Sony's 0..255. Installs
// itself as the active PadBackend; the headless prosper_core has no knowledge of it.
//
// CONFIDENCE: MED — the standard Linux gamepad mapping is correct for well-behaved drivers
// (hid-playstation, xpad, hid-generic); exotic pads may differ. The mapping math is unit-tested via
// the pure core (test_pad); SDL3 (pad_sdl3) is the mapping-database-backed cross-platform primary.
#include "pad_evdev.hpp"
#include "../../src/input/pad.hpp"

#ifdef __linux__

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <linux/input.h>

namespace prosper {
namespace {

using namespace prosper::input;

constexpr int kMaxPads = 4;

inline bool test_bit(const unsigned long* arr, int bit) {
    return (arr[bit / (8 * sizeof(long))] >> (bit % (8 * sizeof(long)))) & 1UL;
}

// A stick rests centered (0x80); a trigger rests released (0x00). Uncalibrated axes fall back to
// their rest value — a trigger must NOT fall back to 0x80 or it reads permanently half-pressed.
struct AxisCal { int min = 0, max = 255; uint8_t rest = 0x80; bool known = false; };

struct Pad {
    int      fd = -1;
    bool     tried = false;
    uint32_t buttons = 0;                       // BTN_* -> Sony bits (d-pad hat handled separately)
    int      lx = 0, ly = 0, rx = 0, ry = 0;
    int      l2 = 0, r2 = 0;
    int      hat_x = 0, hat_y = 0;
    AxisCal  cal_lx, cal_ly, cal_rx, cal_ry, cal_l2, cal_r2;
};

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

void read_axis_cal(int fd, int code, AxisCal& c, uint8_t rest) {
    c.rest = rest;
    struct input_absinfo ai;
    if (ioctl(fd, EVIOCGABS(code), &ai) == 0 && ai.maximum > ai.minimum) {
        c.min = ai.minimum; c.max = ai.maximum; c.known = true;
    }
}

// Open the index-th device that advertises a gamepad (BTN_SOUTH). NOTE: /dev/input enumeration is
// unsorted (readdir order), so "pad index 0" may bind different physical devices across runs when
// several are attached — fine for the common single-controller case.
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
    read_axis_cal(p.fd, ABS_X,  p.cal_lx, 0x80);
    read_axis_cal(p.fd, ABS_Y,  p.cal_ly, 0x80);
    read_axis_cal(p.fd, ABS_RX, p.cal_rx, 0x80);
    read_axis_cal(p.fd, ABS_RY, p.cal_ry, 0x80);
    read_axis_cal(p.fd, ABS_Z,  p.cal_l2, 0x00);   // triggers rest at 0
    read_axis_cal(p.fd, ABS_RZ, p.cal_r2, 0x00);
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
            if (bit) { if (ev.value) p.buttons |= bit; else p.buttons &= ~bit; }
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
        close(p.fd); p.fd = -1; p.tried = false;   // hot-unplug: allow re-open on a later poll
    }
}

uint8_t norm(int raw, const AxisCal& c) {
    return c.known ? pad_axis_u8(raw, c.min, c.max) : c.rest;
}

class EvdevPadBackend : public PadBackend {
public:
    bool poll(int index, HostPadState& out) override {
        out = HostPadState{};
        if (index < 0 || index >= kMaxPads) return false;

        std::lock_guard<std::mutex> lk(mtx_);
        Pad& p = pads_[index];
        if (!p.tried) init_pad(p, index);
        if (p.fd < 0) return false;
        drain_events(p);
        if (p.fd < 0) return false;

        uint32_t b = p.buttons;   // BTN_DPAD_* (if any) already merged; add the hat below
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
        b |= pad_trigger_buttons(out.l2, out.r2);   // digital L2/R2 bit off the analog travel

        out.buttons   = b;
        out.connected = true;
        return true;
    }
private:
    std::mutex mtx_;
    Pad        pads_[kMaxPads];
};

EvdevPadBackend g_backend;
bool            g_installed = false;

} // namespace

bool install_evdev_pad_backend() {
    if (g_installed) return true;
    input::pad_set_backend(&g_backend);
    g_installed = true;
    return true;
}

void shutdown_evdev_pad_backend() {
    if (!g_installed) return;
    input::pad_set_backend(nullptr);
    g_installed = false;
}

} // namespace prosper

#else  // !__linux__ — evdev is Linux-only; use the SDL3 frontend on other platforms.

namespace prosper {
bool install_evdev_pad_backend() { return false; }
void shutdown_evdev_pad_backend() {}
} // namespace prosper

#endif
