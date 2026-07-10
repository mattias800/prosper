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

    // --- ImeDialog: read_ime_request from a synthetic OrbisImeDialogParam (96 bytes) ---
    {
        char16_t title[] = u"Enter name";
        char16_t inbuf[32]; inbuf[0] = u'H'; inbuf[1] = u'i'; inbuf[2] = 0;
        uint8_t iparam[96] = {};
        uint32_t maxlen = 16;                    std::memcpy(iparam + 36, &maxlen, 4);   // max_text_length @36
        uint64_t ibp = (uint64_t)(uintptr_t)inbuf; std::memcpy(iparam + 40, &ibp, 8);    // input_text_buffer @40
        uint64_t tp  = (uint64_t)(uintptr_t)title; std::memcpy(iparam + 72, &tp, 8);     // title @72
        ImeRequest r = read_ime_request((uint64_t)(uintptr_t)iparam);
        CHECK(r.max_text_length == 16, "read_ime_request: max_text_length @36");
        CHECK(r.input_buffer == ibp, "read_ime_request: input_text_buffer @40");
        CHECK(r.title == "Enter name", "read_ime_request: title @72 (UTF-16 -> UTF-8)");
        CHECK(r.initial == "Hi", "read_ime_request: initial text read from the input buffer");
    }

    // write_ime_text: bounded, NUL-terminated, never overruns max_len (+ the NUL slot).
    {
        char16_t buf[8]; for (auto& c : buf) c = 0xAAAA;
        uint32_t n = write_ime_text((uint64_t)(uintptr_t)buf, 5, "Hello");
        CHECK(n == 5 && buf[0] == u'H' && buf[4] == u'o' && buf[5] == 0, "write_ime_text: 'Hello' -> 5 units + NUL");
        for (auto& c : buf) c = 0xAAAA;
        n = write_ime_text((uint64_t)(uintptr_t)buf, 3, "Hello");
        CHECK(n == 3 && buf[3] == 0 && buf[4] == 0xAAAA, "write_ime_text: clamps to max_len (no overrun past max_len+NUL)");
        for (auto& c : buf) c = 0xAAAA;
        n = write_ime_text((uint64_t)(uintptr_t)buf, 7, "caf\xC3\xA9");   // "café" (é = U+00E9)
        CHECK(n == 4 && buf[3] == 0x00E9 && buf[4] == 0, "write_ime_text: multibyte UTF-8 é -> one UTF-16 unit");
    }

    // UTF-16 surrogate pair round-trips (U+1F600 grinning face).
    {
        char16_t emoji[] = { 0xD83D, 0xDE00, 0 };
        CHECK(u16_to_utf8((uint64_t)(uintptr_t)emoji) == "\xF0\x9F\x98\x80", "u16_to_utf8: surrogate pair -> 4-byte UTF-8");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
