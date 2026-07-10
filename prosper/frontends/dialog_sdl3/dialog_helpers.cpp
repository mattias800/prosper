// dialog_helpers.cpp — see dialog_helpers.hpp. Pure struct parsing (no SDL). memcpy is used for every
// guest read/write so unaligned guest structs and strict aliasing are both safe.
#include "dialog_helpers.hpp"
#include <cstring>

namespace prosper {

MsgRequest read_msg_request(uint64_t param) {
    MsgRequest r{0, 0, nullptr};
    if (!param) return r;
    std::memcpy(&r.mode, (const void*)(uintptr_t)(param + 56), 4);
    uint64_t ump = 0;
    std::memcpy(&ump, (const void*)(uintptr_t)(param + 64), 8);   // userMsgParam pointer
    if (ump) {
        std::memcpy(&r.buttonType, (const void*)(uintptr_t)(ump + 0), 4);
        uint64_t msgp = 0;
        std::memcpy(&msgp, (const void*)(uintptr_t)(ump + 8), 8);
        r.msg = (const char*)(uintptr_t)msgp;
    }
    return r;
}

MsgButtons msg_buttons_for(uint32_t bt) {
    // OrbisMsgDialogButtonType (shadPS4): OK=0, YESNO=1, NONE=2, OK_CANCEL=3, WAIT=5, WAIT_CANCEL=6,
    // YESNO_FOCUS_NO=7, OK_CANCEL_FOCUS_CANCEL=8, TWO_BUTTONS=9.
    switch (bt) {
        case 1: case 7: return {2, {"Yes", "No", nullptr},        {1, 2, 0}};   // YESNO[_FOCUS_NO]
        case 3: case 8: return {2, {"OK", "Cancel", nullptr},     {1, 2, 0}};   // OK_CANCEL[_FOCUS_CANCEL]
        case 9:         return {2, {"Button 1", "Button 2", nullptr}, {1, 2, 0}}; // TWO_BUTTONS (labels are custom; ids are 1/2)
        case 2:         return {0, {nullptr, nullptr, nullptr},    {0, 0, 0}};   // NONE
        case 5: case 6: return {0, {nullptr, nullptr, nullptr},    {0, 0, 0}};   // WAIT[_CANCEL] (progress; no buttons)
        case 0:
        default:        return {1, {"OK", nullptr, nullptr},       {1, 0, 0}};   // OK
    }
}

void write_msg_result(uint64_t result, uint32_t buttonId) {
    if (!result) return;
    uint32_t mode = 0;
    uint32_t res  = (buttonId == 2u) ? 1u : 0u;   // NO / Cancel (id 2) -> USER_CANCELED, else OK
    std::memcpy((void*)(uintptr_t)(result + 0), &mode,     4);
    std::memcpy((void*)(uintptr_t)(result + 4), &res,      4);
    std::memcpy((void*)(uintptr_t)(result + 8), &buttonId, 4);
}

uint32_t read_error_code(uint64_t param) {
    uint32_t code = 0;
    if (param) std::memcpy(&code, (const void*)(uintptr_t)(param + 4), 4);
    return code;
}

} // namespace prosper
