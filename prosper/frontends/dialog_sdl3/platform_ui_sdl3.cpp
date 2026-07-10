// platform_ui_sdl3.cpp — the SDL3 PlatformUi (see dialog_sdl3.hpp). Shows real SDL_ShowMessageBox
// dialogs for the guest MsgDialog / ErrorDialog; the layout parsing is in dialog_helpers.cpp (pure,
// unit-tested). Built only under -DPROSPER_APP=ON. The guest drives dialogs by polling GetStatus until
// it leaves RUNNING, so Open shows the modal (blocking the guest's dialog thread — intended for a
// modal), records the result, and reports FINISHED; the next poll returns it.
#include "dialog_sdl3.hpp"
#include "dialog_helpers.hpp"
#include "hle/platform_ui.hpp"
#include <SDL3/SDL.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace prosper {
namespace {

// Show a modal message box. Returns the clicked button's id (which we set to the guest ButtonId), or
// -1 if it was dismissed without a button (window closed) or SDL failed.
int show_box(uint32_t sdl_flags, const char* title, const char* msg, const MsgButtons& b) {
    SDL_MessageBoxButtonData btns[3] = {};
    for (int i = 0; i < b.count; i++) { btns[i].buttonID = (int)b.id[i]; btns[i].text = b.label[i]; }
    if (b.count > 0) {
        btns[0].flags            |= SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT;   // first = default (Enter)
        btns[b.count - 1].flags  |= SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT;   // last  = cancel (Esc)
    }
    SDL_MessageBoxData d = {};
    d.flags = sdl_flags; d.title = title; d.message = msg ? msg : ""; d.numbuttons = b.count; d.buttons = btns;
    int clicked = -1;
    if (!SDL_ShowMessageBox(&d, &clicked)) { SDL_Log("prosper: SDL_ShowMessageBox failed: %s", SDL_GetError()); return -1; }
    return clicked;
}

// Run a modal text-entry window (MAIN thread only). Returns endStatus (0=OK, 1=CANCELED) and, on OK,
// writes the entered text (UTF-16) into the request's guest input buffer (bounded by max_text_length).
int run_ime_modal(const ImeRequest& req) {
    SDL_Window* win = nullptr; SDL_Renderer* ren = nullptr;
    const char* wtitle = req.title.empty() ? "prosper — text entry" : req.title.c_str();
    if (!SDL_CreateWindowAndRenderer(wtitle, 560, 150, 0, &win, &ren)) {
        SDL_Log("prosper: ime window failed: %s", SDL_GetError());
        return 1;   // treat as cancel
    }
    SDL_StartTextInput(win);
    std::string text = req.initial;
    int endstatus = 1 /*USER_CANCELED*/;
    bool done = false;
    while (!done) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_EVENT_QUIT) { endstatus = 1; done = true; }
            else if (ev.type == SDL_EVENT_TEXT_INPUT) { text += ev.text.text; }   // UTF-8; write bounds it
            else if (ev.type == SDL_EVENT_KEY_DOWN) {
                SDL_Keycode k = ev.key.key;
                if (k == SDLK_RETURN || k == SDLK_KP_ENTER) { endstatus = 0 /*OK*/; done = true; }
                else if (k == SDLK_ESCAPE) { endstatus = 1; done = true; }
                else if (k == SDLK_BACKSPACE && !text.empty()) {           // pop one UTF-8 code point
                    size_t n = text.size();
                    do { n--; } while (n > 0 && ((unsigned char)text[n] & 0xC0) == 0x80);
                    text.resize(n);
                }
            }
        }
        SDL_SetRenderDrawColor(ren, 28, 30, 40, 255); SDL_RenderClear(ren);
        SDL_SetRenderDrawColor(ren, 225, 225, 230, 255);
        SDL_RenderDebugText(ren, 12, 16, req.title.empty() ? "Enter text:" : req.title.c_str());
        std::string shown = text + "_";
        SDL_RenderDebugText(ren, 12, 52, shown.c_str());
        SDL_RenderDebugText(ren, 12, 118, "Enter = OK     Esc = Cancel");
        SDL_RenderPresent(ren);
        SDL_Delay(8);
    }
    SDL_StopTextInput(win);
    SDL_DestroyRenderer(ren); SDL_DestroyWindow(win);
    if (endstatus == 0) write_ime_text(req.input_buffer, req.max_text_length, text);
    return endstatus;
}

struct SdlPlatformUi : PlatformUi {
    std::atomic<int>      msg_status{0}, err_status{0};
    std::atomic<uint32_t> msg_btn{1 /*OK*/};

    // MsgDialog. Only user-message mode maps to a message box; progress/system modes fall back to the
    // core's headless handling (return false).
    bool msgDialogOpen(uint64_t param) override {
        MsgRequest r = read_msg_request(param);
        if (r.mode != ORBIS_MSG_DIALOG_MODE_USER_MSG) return false;
        MsgButtons b = msg_buttons_for(r.buttonType);
        if (b.count == 0) return false;
        int clicked = show_box(SDL_MESSAGEBOX_INFORMATION, "prosper", r.msg, b);
        msg_btn.store(clicked >= 0 ? (uint32_t)clicked : b.id[0]);
        msg_status.store(3 /*FINISHED*/);
        return true;
    }
    int  msgDialogStatus() override { return msg_status.load(); }
    int  msgDialogResult(uint64_t result) override { uint32_t id = msg_btn.load(); write_msg_result(result, id); return (int)id; }
    void msgDialogClose() override { msg_status.store(0); }

    // ErrorDialog: a single-OK box reporting the error code.
    bool errorDialogOpen(uint64_t param) override {
        char buf[96];
        std::snprintf(buf, sizeof buf, "An error occurred.\n\nError code: 0x%08X", read_error_code(param));
        show_box(SDL_MESSAGEBOX_ERROR, "prosper — Error", buf, msg_buttons_for(0 /*OK*/));
        err_status.store(3 /*FINISHED*/);
        return true;
    }
    int  errorDialogStatus() override { return err_status.load(); }
    void errorDialogClose() override { err_status.store(0); }

    // Ime keyboard presence: a windowed session runs on a host with a keyboard, so report one (#347).
    // This makes sceImeKeyboardGetResourceId/GetInfo report a connected keyboard, enabling a title's
    // keyboard-input option. (Delivering the raw key events to the guest handler is a further step.)
    int keyboardResourceIds(int32_t /*userId*/, uint32_t* outIds, int max) override {
        if (max >= 1 && outIds) { outIds[0] = 1; return 1; }
        return 0;
    }

    // ImeDialog (text entry). Cross-thread: the guest thread calls Open/Status/Result; the MAIN thread
    // runs the actual SDL window (SDL windowing/events are main-thread only) when prosper-app calls
    // sdl_platform_ui_pump() each frame. Open records the request + RUNNING; pump() shows the modal and
    // sets FINISHED; the guest polls Status and then reads its input buffer (written before FINISHED).
    std::mutex          ime_mx;
    ImeRequest          ime_req{};
    std::atomic<int>    ime_status{0 /*OrbisImeDialogStatus::None*/};
    std::atomic<bool>   ime_pending{false};
    std::atomic<int>    ime_endstatus{1 /*USER_CANCELED until confirmed*/};

    bool imeDialogOpen(uint64_t param, uint64_t /*extended*/) override {
        ImeRequest r = read_ime_request(param);
        if (!r.input_buffer) return false;   // nothing to fill -> let the core headless-handle it
        { std::lock_guard<std::mutex> lk(ime_mx); ime_req = r; }
        ime_endstatus.store(1);
        ime_status.store(1 /*Running*/);
        ime_pending.store(true);
        return true;
    }
    int  imeDialogStatus() override { return ime_status.load(); }
    int  imeDialogResult(uint64_t result) override {
        int es = ime_endstatus.load();
        if (result) { uint32_t v = (uint32_t)es; std::memcpy((void*)(uintptr_t)result, &v, 4); }  // endstatus @0
        return es;
    }
    void imeDialogClose() override { ime_pending.store(false); ime_status.store(0 /*None*/); }

    // MAIN-thread: if a text dialog is pending, run its modal to completion (writes the guest buffer,
    // then flips status to FINISHED so the guest's next poll — and its buffer read — see it).
    void pump() {
        if (!ime_pending.exchange(false)) return;
        ImeRequest r; { std::lock_guard<std::mutex> lk(ime_mx); r = ime_req; }
        int es = run_ime_modal(r);
        ime_endstatus.store(es);
        ime_status.store(2 /*Finished*/);
    }
};

SdlPlatformUi g_sdl_ui;

} // namespace

bool install_sdl3_platform_ui() {
    set_platform_ui(&g_sdl_ui);
    SDL_Log("prosper: SDL3 dialog backend installed (MsgDialog/ErrorDialog + ImeDialog text entry)");
    return true;
}
void shutdown_sdl3_platform_ui() { set_platform_ui(nullptr); }
void sdl_platform_ui_pump() { g_sdl_ui.pump(); }

} // namespace prosper
