// test_dialog_helpers — the SDL3 dialog frontend's pure struct parsing (#347). No SDL / no display, so
// it runs in normal CI: it builds synthetic guest MsgDialog structs at the shadPS4-verified offsets and
// checks read_msg_request / msg_buttons_for / write_msg_result / read_error_code.
#include "../dialog_sdl3/dialog_helpers.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_dialog_helpers ==\n");

    // Synthetic OrbisMsgDialogParam (>=72 bytes) + UserMessageParam at the verified offsets.
    uint8_t umsg[32] = {};
    const char* message = "Delete this save?";
    uint32_t btnType = 1;   // YESNO
    std::memcpy(umsg + 0, &btnType, 4);                 // buttonType @0
    uint64_t msgp = (uint64_t)(uintptr_t)message; std::memcpy(umsg + 8, &msgp, 8);  // msg @8

    uint8_t param[80] = {};
    uint32_t mode = ORBIS_MSG_DIALOG_MODE_USER_MSG; std::memcpy(param + 56, &mode, 4);     // mode @56
    uint64_t ump = (uint64_t)(uintptr_t)umsg;       std::memcpy(param + 64, &ump, 8);      // userMsgParam @64

    MsgRequest r = read_msg_request((uint64_t)(uintptr_t)param);
    CHECK(r.mode == 1, "read_msg_request: mode @56 (USER_MSG)");
    CHECK(r.buttonType == 1, "read_msg_request: buttonType @64->@0 (YESNO)");
    CHECK(r.msg && std::strcmp(r.msg, message) == 0, "read_msg_request: msg pointer @64->@8 resolves the text");

    // Button mapping.
    MsgButtons ok = msg_buttons_for(0);
    CHECK(ok.count == 1 && ok.id[0] == 1, "msg_buttons_for(OK) -> 1 button, id OK(1)");
    MsgButtons yn = msg_buttons_for(1);
    CHECK(yn.count == 2 && yn.id[0] == 1 && yn.id[1] == 2, "msg_buttons_for(YESNO) -> Yes(1)/No(2)");
    MsgButtons okc = msg_buttons_for(3);
    CHECK(okc.count == 2 && std::strcmp(okc.label[1], "Cancel") == 0, "msg_buttons_for(OK_CANCEL) -> OK/Cancel");
    CHECK(msg_buttons_for(2).count == 0, "msg_buttons_for(NONE) -> no buttons (progress/system -> headless)");
    CHECK(msg_buttons_for(5).count == 0, "msg_buttons_for(WAIT) -> no buttons");

    // Result writing: buttonId @8, result @4 (NO/Cancel=id 2 -> USER_CANCELED), mode @0. Bounded to 12 bytes.
    uint8_t res[44]; std::memset(res, 0xAB, sizeof res);
    write_msg_result((uint64_t)(uintptr_t)res, 1 /*OK/YES*/);
    uint32_t w_mode, w_res, w_btn; std::memcpy(&w_mode, res+0, 4); std::memcpy(&w_res, res+4, 4); std::memcpy(&w_btn, res+8, 4);
    CHECK(w_mode == 0 && w_res == 0 && w_btn == 1, "write_msg_result(1) -> mode 0, result OK(0), buttonId 1");
    CHECK((uint8_t)res[12] == 0xAB && (uint8_t)res[43] == 0xAB, "write_msg_result writes only 12 bytes (reserved tail untouched)");
    write_msg_result((uint64_t)(uintptr_t)res, 2 /*NO/Cancel*/);
    std::memcpy(&w_res, res+4, 4); std::memcpy(&w_btn, res+8, 4);
    CHECK(w_res == 1 && w_btn == 2, "write_msg_result(2) -> result USER_CANCELED(1), buttonId 2");

    // ErrorDialog param: errorCode @4.
    uint8_t ep[16] = {}; uint32_t code = 0x80AABBCC; std::memcpy(ep + 4, &code, 4);
    CHECK(read_error_code((uint64_t)(uintptr_t)ep) == 0x80AABBCC, "read_error_code: errorCode @4");
    CHECK(read_error_code(0) == 0, "read_error_code(NULL) -> 0 (no deref)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
