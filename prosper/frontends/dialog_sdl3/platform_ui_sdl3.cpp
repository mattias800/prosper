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

    // imeDialog* left as the base default (returns false) -> the core's headless auto-complete, since a
    // text-entry field is not a message box (a custom SDL text UI is a separate follow-up).
};

SdlPlatformUi g_sdl_ui;

} // namespace

bool install_sdl3_platform_ui() {
    set_platform_ui(&g_sdl_ui);
    SDL_Log("prosper: SDL3 dialog backend installed (MsgDialog/ErrorDialog -> SDL_ShowMessageBox)");
    return true;
}
void shutdown_sdl3_platform_ui() { set_platform_ui(nullptr); }

} // namespace prosper
