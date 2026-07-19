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
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

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

// --- SaveDataDialog ---
// Param layout (0x98 bytes): CommonDialog::BaseParam[0x30], size@0x30, mode@0x34,
// dispType@0x38, then nested pointers items@0x48, userMsg@0x50, systemMsg@0x58,
// errorCode@0x60, progress@0x68, userData@0x70, option@0x78. Result layout (0x48 bytes):
// mode/result/buttonId @0/4/8, caller-owned dirName/param pointers @0x10/0x18,
// userData @0x20, reserved[32] @0x28. These offsets match the PS4/PS5 inherited ABI.
enum : uint32_t {
    SAVE_DATA_DIALOG_MODE_LIST = 1,
    SAVE_DATA_DIALOG_MODE_USER_MSG = 2,
    SAVE_DATA_DIALOG_MODE_SYSTEM_MSG = 3,
    SAVE_DATA_DIALOG_MODE_ERROR_CODE = 4,
    SAVE_DATA_DIALOG_MODE_PROGRESS_BAR = 5,
};

struct SaveDataSlot {
    std::string dirName;          // guest virtual directory name; empty selects the SAVE new-item
    std::string label;            // owned display label; never a host path
};

struct SaveDataRequest {
    bool supported = false;       // false -> frontend declines ownership; core keeps headless policy
    bool error = false;           // choose the error rather than information message-box style
    bool cancelable = true;       // false when OptionBack::DISABLE forbids a back/cancel outcome
    bool progress = false;        // non-modal PROGRESS_BAR request; guest Close owns completion
    bool slotList = false;        // modal LIST request backed only by guest-provided virtual slots
    uint32_t mode = 0;
    uint32_t displayType = 0;     // 1=SAVE, 2=LOAD, 3=DELETE
    uint64_t userData = 0;
    size_t initialSlot = 0;
    std::string message;
    std::vector<SaveDataSlot> slots;
    MsgButtons buttons{0, {nullptr, nullptr, nullptr}, {0, 0, 0}};
};

// SDL-free lifecycle seam shared by the frontend and unit tests. A pending ticket is consumed once;
// Close or a newer Open invalidates every older generation, including one already visible.
struct SaveDataModalTicket {
    uint64_t generation = 0;
    SaveDataRequest request{};
    explicit operator bool() const { return generation != 0; }
};

struct SaveDataResultSnapshot {
    SaveDataRequest request{};
    uint32_t buttonId = 0;
    bool canceled = false;
    std::string dirName;
};

struct SaveDataProgressSnapshot {
    uint64_t generation = 0;
    SaveDataRequest request{};
    uint32_t value = 0;
    explicit operator bool() const { return generation != 0; }
};

class SaveDataDialogState {
public:
    void open(SaveDataRequest request);
    int status() const;
    void close();
    SaveDataModalTicket take_pending();
    bool active(uint64_t generation) const;
    void complete(uint64_t generation, int clicked);
    void complete_list(uint64_t generation, int selected);
    SaveDataResultSnapshot result_snapshot() const;
    void progress_inc(uint32_t target, uint32_t delta);
    void progress_set(uint32_t target, uint32_t value);
    void finish_progress(uint64_t generation);
    SaveDataProgressSnapshot progress_snapshot() const;

private:
    mutable std::mutex mx_;
    SaveDataRequest request_{};
    int status_ = 0;
    uint32_t button_ = 0;
    uint32_t progress_ = 0;
    bool canceled_ = false;
    std::string dir_name_;
    uint64_t generation_ = 0;
    uint64_t pending_generation_ = 0;
};

// Parse supported modes into an owned request safe to retain until the SDL main-thread pump runs.
// LIST retains only guest-provided virtual directory names/new-item text; no host path is exposed.
// USER_MSG, ordinary SYSTEM_MSG confirmations/notices, ERROR_CODE, and PROGRESS_BAR are also supported.
SaveDataRequest read_savedata_request(uint64_t param);

// Write fields owned by the service: mode/result/buttonId/userData and, for LIST, the selected virtual
// directory through the caller-provided dirName pointer. The param output pointer, ABI pad, and all
// reserved bytes remain untouched. `canceled` writes USER_CANCELED, INVALID, and an empty LIST name.
// Unreadable result/output pointers are ignored.
void write_savedata_result(uint64_t result, const SaveDataRequest& request,
                           uint32_t buttonId, bool canceled,
                           const std::string& dirName = {});

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
