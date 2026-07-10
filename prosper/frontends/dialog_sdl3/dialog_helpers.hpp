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

} // namespace prosper
