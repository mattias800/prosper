// dialog_helpers.hpp — pure (SDL-free) parsing of the guest MsgDialog / ErrorDialog structs for the
// SDL3 PlatformUi frontend. Kept separate from the SDL glue so this layout-sensitive code is unit-
// tested in normal CI (no display needed); only the SDL_ShowMessageBox call lives in the gated module.
//
// Struct offsets verified against shadPS4 src/core/libraries/system/{msgdialog_ui,commondialog}.h:
//   OrbisMsgDialogParam:  CommonDialog::BaseParam baseParam [48] ; size_t size @48 ; mode @56 ;
//                         (pad @60) ; UserMessageParam* userMsgParam @64 ; ...
//   UserMessageParam:     ButtonType buttonType @0 ; (pad @4) ; const char* msg @8 ; ...
//   DialogResult:         u32 mode @0 ; Result result @4 ; ButtonId buttonId @8 ; u8 reserved[32]
//   ErrorDialog param:    s32 size @0 ; s32 errorCode @4 ; s32 userId @8 ; s32 reserved @12
#pragma once
#include <cstdint>
#include <string>

namespace prosper {

enum { ORBIS_MSG_DIALOG_MODE_USER_MSG = 1 };   // 2=PROGRESS_BAR, 3=SYSTEM_MSG (no buttons -> headless)

// The SDL button set a MsgDialog ButtonType maps to: labels + the guest ButtonId each returns.
struct MsgButtons { int count; const char* label[3]; uint32_t id[3]; };

// What we read out of a guest OrbisMsgDialogParam* to build the message box.
struct MsgRequest { uint32_t mode; uint32_t buttonType; const char* msg; };

// Read mode + buttonType + message pointer from a guest OrbisMsgDialogParam* (guest==host address).
MsgRequest read_msg_request(uint64_t param);

// Map an OrbisMsgDialogButtonType to its button set (OK / Yes-No / OK-Cancel / ...). ButtonId per
// shadPS4: OK=1, YES=1, NO=2, BUTTON1=1, BUTTON2=2. count 0 = a no-button mode (progress/system).
MsgButtons msg_buttons_for(uint32_t buttonType);

// Write the guest OrbisMsgDialogResult*: mode@0=0, result@4 (0=OK, 1=USER_CANCELED — NO/Cancel is id 2),
// buttonId@8. Bounded to the 3 leading u32s; never touches the caller's reserved tail.
void write_msg_result(uint64_t result, uint32_t buttonId);

// Read errorCode (@4) from a guest OrbisErrorDialogParam*.
uint32_t read_error_code(uint64_t param);

// --- ImeDialog (text entry) --- offsets from shadPS4 ime/ime_common.h OrbisImeDialogParam:
//   user_id@0 type@4 ... max_text_length@36(u32) input_text_buffer@40(char16_t*) ... title@72(char16_t*)
struct ImeRequest {
    uint64_t    input_buffer;    // guest char16_t* the entered text is written into (0 if none)
    uint32_t    max_text_length; // max UTF-16 code units (excluding NUL) the buffer holds
    std::string title;           // UTF-8 of the title (for the prompt); empty if none
    std::string initial;         // UTF-8 of any text already in the input buffer
};
// Parse a guest OrbisImeDialogParam* into an ImeRequest (title/initial converted UTF-16 -> UTF-8).
ImeRequest read_ime_request(uint64_t param);

// Write `utf8` into the guest UTF-16 input buffer (at `input_buffer`), NUL-terminated, clamped to
// max_len code units (never overruns). Returns the number of code units written (excluding NUL).
uint32_t write_ime_text(uint64_t input_buffer, uint32_t max_len, const std::string& utf8);

// UTF-16 (guest, host-endian) <-> UTF-8. u16_to_utf8 reads a NUL-terminated char16_t* at a guest addr.
std::string u16_to_utf8(uint64_t u16ptr);
// Append one UTF-8-encoded Unicode code point for `cp` (used to build strings from typed input).
void utf8_append_cp(std::string& s, uint32_t cp);

} // namespace prosper
