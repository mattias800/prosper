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

    // --- SaveDataDialog: supported message/confirmation/error modes use the ABI's nested pointers. ---
    {
        const char* save_message = "Overwrite this slot?";
        uint8_t user[48] = {};
        uint32_t button_type = 1 /*YESNO*/, message_type = 0 /*NORMAL*/;
        uint64_t save_message_ptr = (uint64_t)(uintptr_t)save_message;
        std::memcpy(user + 0, &button_type, 4);
        std::memcpy(user + 4, &message_type, 4);
        std::memcpy(user + 8, &save_message_ptr, 8);

        uint8_t save_param[0x98] = {};
        uint32_t save_mode = SAVE_DATA_DIALOG_MODE_USER_MSG, display_type = 1 /*SAVE*/;
        uint64_t user_ptr = (uint64_t)(uintptr_t)user, user_data = 0x1122334455667788ull;
        uint64_t base_size = 0x30;
        uint32_t param_size = 0x98;
        std::memcpy(save_param + 0x00, &base_size, 8);
        std::memcpy(save_param + 0x30, &param_size, 4);
        std::memcpy(save_param + 0x34, &save_mode, 4);
        std::memcpy(save_param + 0x38, &display_type, 4);
        std::memcpy(save_param + 0x50, &user_ptr, 8);
        std::memcpy(save_param + 0x70, &user_data, 8);
        SaveDataRequest request = read_savedata_request((uint64_t)(uintptr_t)save_param);
        CHECK(request.supported && request.mode == SAVE_DATA_DIALOG_MODE_USER_MSG &&
                  request.message == save_message && request.buttons.count == 2 &&
                  request.buttons.id[0] == 1 && request.buttons.id[1] == 2 &&
                  request.userData == user_data,
              "SaveDataDialog USER_MSG parses text, YES/NO buttons, mode, and userData");

        uint8_t system[48] = {};
        uint32_t system_type = 2 /*CONFIRM*/;
        std::memcpy(system, &system_type, 4);
        save_mode = SAVE_DATA_DIALOG_MODE_SYSTEM_MSG; display_type = 3 /*DELETE*/;
        uint64_t system_ptr = (uint64_t)(uintptr_t)system;
        std::memcpy(save_param + 0x34, &save_mode, 4);
        std::memcpy(save_param + 0x38, &display_type, 4);
        std::memcpy(save_param + 0x58, &system_ptr, 8);
        request = read_savedata_request((uint64_t)(uintptr_t)save_param);
        CHECK(request.supported && request.message.find("delete") != std::string::npos &&
                  request.buttons.count == 2 && request.buttons.id[1] == 2,
              "SaveDataDialog SYSTEM_MSG confirmation maps display type to DELETE + YES/NO");

        uint8_t option[36] = {};
        uint32_t option_back = 1 /*DISABLE*/;
        uint64_t option_ptr = (uint64_t)(uintptr_t)option;
        system_type = 10 /*CORRUPTED_AND_DELETED*/;
        std::memcpy(system, &system_type, 4);
        std::memcpy(option, &option_back, 4);
        std::memcpy(save_param + 0x78, &option_ptr, 8);
        request = read_savedata_request((uint64_t)(uintptr_t)save_param);
        CHECK(request.supported && !request.cancelable && request.buttons.count == 1 &&
                  request.buttons.id[0] == 1,
              "SaveDataDialog OptionBack::DISABLE removes the SYSTEM_MSG cancel path");
        option_back = 0 /*ENABLE*/;
        std::memcpy(option, &option_back, 4);
        request = read_savedata_request((uint64_t)(uintptr_t)save_param);
        CHECK(request.supported && request.cancelable && request.buttons.count == 2 &&
                  request.buttons.id[1] == 2,
              "SaveDataDialog OptionBack::ENABLE exposes OK + Cancel with the ABI button ids");

        option_ptr = 1; std::memcpy(save_param + 0x78, &option_ptr, 8);
        CHECK(!read_savedata_request((uint64_t)(uintptr_t)save_param).supported,
              "SaveDataDialog rejects a non-null unreadable option pointer without crashing");
        option_ptr = 0; std::memcpy(save_param + 0x78, &option_ptr, 8);

        system_type = 5 /*PROGRESS*/; std::memcpy(system, &system_type, 4);
        CHECK(!read_savedata_request((uint64_t)(uintptr_t)save_param).supported,
              "SaveDataDialog progress message declines modal ownership");

        uint8_t error[36] = {};
        uint32_t error_code = 0x809F000F;
        std::memcpy(error, &error_code, 4);
        uint64_t error_ptr = (uint64_t)(uintptr_t)error;
        save_mode = SAVE_DATA_DIALOG_MODE_ERROR_CODE;
        std::memcpy(save_param + 0x34, &save_mode, 4);
        std::memcpy(save_param + 0x60, &error_ptr, 8);
        request = read_savedata_request((uint64_t)(uintptr_t)save_param);
        CHECK(request.supported && request.error &&
                  request.message.find("0x809F000F") != std::string::npos &&
                  request.buttons.count == 1,
              "SaveDataDialog ERROR_CODE maps to an error box with bounded code text");

        CHECK(!read_savedata_request(0).supported,
              "SaveDataDialog null outer param declines ownership without dereference");
        CHECK(!read_savedata_request(1).supported,
              "SaveDataDialog unreadable outer param declines ownership without crashing");
        param_size = 0; std::memcpy(save_param + 0x30, &param_size, 4);
        CHECK(!read_savedata_request((uint64_t)(uintptr_t)save_param).supported,
              "SaveDataDialog malformed outer size declines ownership");
        param_size = 0x98; std::memcpy(save_param + 0x30, &param_size, 4);
        error_ptr = 0; std::memcpy(save_param + 0x60, &error_ptr, 8);
        CHECK(!read_savedata_request((uint64_t)(uintptr_t)save_param).supported,
              "SaveDataDialog null nested mode param declines ownership without dereference");
        error_ptr = 1; std::memcpy(save_param + 0x60, &error_ptr, 8);
        CHECK(!read_savedata_request((uint64_t)(uintptr_t)save_param).supported,
              "SaveDataDialog unreadable ERROR_CODE pointer declines without crashing");

        save_mode = SAVE_DATA_DIALOG_MODE_USER_MSG;
        user_ptr = 1;
        std::memcpy(save_param + 0x34, &save_mode, 4);
        std::memcpy(save_param + 0x50, &user_ptr, 8);
        CHECK(!read_savedata_request((uint64_t)(uintptr_t)save_param).supported,
              "SaveDataDialog unreadable USER_MSG pointer declines without crashing");
        user_ptr = (uint64_t)(uintptr_t)user;
        save_message_ptr = 1;
        std::memcpy(save_param + 0x50, &user_ptr, 8);
        std::memcpy(user + 8, &save_message_ptr, 8);
        CHECK(!read_savedata_request((uint64_t)(uintptr_t)save_param).supported,
              "SaveDataDialog unreadable nested message pointer declines without crashing");
        save_message_ptr = (uint64_t)(uintptr_t)save_message;
        std::memcpy(user + 8, &save_message_ptr, 8);

        save_mode = SAVE_DATA_DIALOG_MODE_LIST;
        std::memcpy(save_param + 0x34, &save_mode, 4);
        CHECK(!read_savedata_request((uint64_t)(uintptr_t)save_param).supported,
              "SaveDataDialog LIST declines until the virtual-slot UI is implemented");

        request.mode = SAVE_DATA_DIALOG_MODE_USER_MSG;
        request.userData = user_data;
        uint8_t save_result[0x48]; std::memset(save_result, 0xAB, sizeof save_result);
        write_savedata_result((uint64_t)(uintptr_t)save_result, request, 2 /*NO*/, false);
        uint32_t result_mode = 0, common_result = 0, result_button = 0;
        uint64_t result_user_data = 0;
        std::memcpy(&result_mode, save_result + 0x00, 4);
        std::memcpy(&common_result, save_result + 0x04, 4);
        std::memcpy(&result_button, save_result + 0x08, 4);
        std::memcpy(&result_user_data, save_result + 0x20, 8);
        CHECK(result_mode == SAVE_DATA_DIALOG_MODE_USER_MSG && common_result == 0 &&
                  result_button == 2 && result_user_data == user_data,
              "SaveDataDialog result returns mode/OK/NO/userData");
        CHECK(save_result[0x0c] == 0xAB && save_result[0x10] == 0xAB &&
                  save_result[0x18] == 0xAB && save_result[0x28] == 0xAB &&
                  save_result[0x47] == 0xAB,
              "SaveDataDialog result preserves pad, caller output pointers, and reserved tail");
        write_savedata_result((uint64_t)(uintptr_t)save_result, request, 1, true);
        std::memcpy(&common_result, save_result + 0x04, 4);
        std::memcpy(&result_button, save_result + 0x08, 4);
        CHECK(common_result == 1 && result_button == 0,
              "SaveDataDialog canceled result returns USER_CANCELED + INVALID button");
        write_savedata_result(1, request, 1, false);
        CHECK(true, "SaveDataDialog unreadable result pointer is ignored without crashing");
    }

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
