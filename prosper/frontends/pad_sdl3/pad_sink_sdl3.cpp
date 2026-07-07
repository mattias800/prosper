// pad_sink_sdl3.cpp — SDL3 SDL_Gamepad PadBackend (optional frontend, -DPROSPER_PAD_SDL3=ON).
//
// Bridges the headless PadBackend interface (src/input/pad.hpp) to SDL3's cross-platform gamepad
// API. SDL maps any recognised controller (DualSense, Xbox, Switch Pro, …) to a canonical layout
// via its bundled mapping database, so there is no per-device button/axis guesswork here — SDL's
// canonical buttons/axes map 1:1 to the PS5 layout. One gamepad handle per pad index, opened lazily.
#include "pad_sdl3.hpp"
#include "../../src/input/pad.hpp"

#include <SDL3/SDL.h>

#include <array>
#include <mutex>

namespace prosper {
namespace {

using namespace prosper::input;

constexpr int kMaxPads = 4;

// Sony 0..255 from an SDL stick axis (-32768..32767) or trigger axis (0..32767 -> rests at 0).
uint8_t stick_u8(Sint16 v)   { return pad_axis_u8((int)v, -32768, 32767); }
uint8_t trigger_u8(Sint16 v) { return pad_axis_u8((int)v, 0, 32767); }

// SDL canonical gamepad button -> Sony bit (0 = no mapping).
uint32_t sony_button_for(SDL_GamepadButton b) {
    switch (b) {
        case SDL_GAMEPAD_BUTTON_SOUTH:          return SCE_PAD_BUTTON_CROSS;
        case SDL_GAMEPAD_BUTTON_EAST:           return SCE_PAD_BUTTON_CIRCLE;
        case SDL_GAMEPAD_BUTTON_NORTH:          return SCE_PAD_BUTTON_TRIANGLE;
        case SDL_GAMEPAD_BUTTON_WEST:           return SCE_PAD_BUTTON_SQUARE;
        case SDL_GAMEPAD_BUTTON_START:          return SCE_PAD_BUTTON_OPTIONS;
        case SDL_GAMEPAD_BUTTON_LEFT_STICK:     return SCE_PAD_BUTTON_L3;
        case SDL_GAMEPAD_BUTTON_RIGHT_STICK:    return SCE_PAD_BUTTON_R3;
        case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER:  return SCE_PAD_BUTTON_L1;
        case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return SCE_PAD_BUTTON_R1;
        case SDL_GAMEPAD_BUTTON_DPAD_UP:        return SCE_PAD_BUTTON_UP;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN:      return SCE_PAD_BUTTON_DOWN;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT:      return SCE_PAD_BUTTON_LEFT;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:     return SCE_PAD_BUTTON_RIGHT;
        case SDL_GAMEPAD_BUTTON_TOUCHPAD:       return SCE_PAD_BUTTON_TOUCH_PAD;
        default:                                return 0;   // BACK/GUIDE/paddles/misc: no PS5 bit
    }
}

class Sdl3PadBackend : public PadBackend {
public:
    bool init() {
        if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
            SDL_Log("prosper-pad: SDL_InitSubSystem(GAMEPAD) failed: %s", SDL_GetError());
            return false;
        }
        return true;
    }
    void quit() {
        std::lock_guard<std::mutex> lk(mx_);
        for (auto& g : gp_) { if (g) { SDL_CloseGamepad(g); g = nullptr; } }
        SDL_QuitSubSystem(SDL_INIT_GAMEPAD);
    }

    bool poll(int index, HostPadState& out) override {
        out = HostPadState{};
        if (index < 0 || index >= kMaxPads) return false;

        std::lock_guard<std::mutex> lk(mx_);
        SDL_UpdateGamepads();                    // refresh state (we don't run an SDL event loop)

        SDL_Gamepad*& g = gp_[index];
        if (g && !SDL_GamepadConnected(g)) { SDL_CloseGamepad(g); g = nullptr; }   // hot-unplug
        if (!g) g = open_index(index);
        if (!g) return false;                    // no device at this index

        uint32_t b = 0;
        for (int i = 0; i < SDL_GAMEPAD_BUTTON_COUNT; i++)
            if (SDL_GetGamepadButton(g, (SDL_GamepadButton)i))
                b |= sony_button_for((SDL_GamepadButton)i);

        out.left_x  = stick_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTX));
        out.left_y  = stick_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTY));
        out.right_x = stick_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTX));
        out.right_y = stick_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTY));
        out.l2      = trigger_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
        out.r2      = trigger_u8(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
        b |= pad_trigger_buttons(out.l2, out.r2);

        out.buttons   = b;
        out.connected = true;
        return true;
    }

private:
    // Open the index-th currently-connected gamepad (SDL instance-id order), or nullptr.
    SDL_Gamepad* open_index(int index) {
        int count = 0;
        SDL_JoystickID* ids = SDL_GetGamepads(&count);
        SDL_Gamepad* g = nullptr;
        if (ids && index < count) g = SDL_OpenGamepad(ids[index]);
        SDL_free(ids);
        return g;
    }
    std::mutex mx_;
    std::array<SDL_Gamepad*, kMaxPads> gp_{};
};

Sdl3PadBackend g_backend;
bool           g_installed = false;

} // namespace

bool install_sdl3_pad_backend() {
    if (g_installed) return true;
    if (!g_backend.init()) return false;
    input::pad_set_backend(&g_backend);
    g_installed = true;
    SDL_Log("prosper-pad: SDL3 gamepad backend installed");
    return true;
}

void shutdown_sdl3_pad_backend() {
    if (!g_installed) return;
    input::pad_set_backend(nullptr);
    g_backend.quit();
    g_installed = false;
}

} // namespace prosper
