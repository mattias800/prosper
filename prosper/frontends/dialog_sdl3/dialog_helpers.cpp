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

// --- ImeDialog text entry ---

void utf8_append_cp(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F));
    }
}

std::string u16_to_utf8(uint64_t u16ptr) {
    std::string out;
    if (!u16ptr) return out;
    const uint16_t* u = (const uint16_t*)(uintptr_t)u16ptr;
    const uint32_t CAP = 4096;   // guard a non-terminated guest string from a runaway read
    for (uint32_t i = 0; i < CAP && u[i]; ) {
        uint32_t cp = u[i++];
        if (cp >= 0xD800 && cp <= 0xDBFF && i < CAP && u[i] >= 0xDC00 && u[i] <= 0xDFFF)
            cp = 0x10000 + ((cp - 0xD800) << 10) + (u[i++] - 0xDC00);   // surrogate pair
        utf8_append_cp(out, cp);
    }
    return out;
}

ImeRequest read_ime_request(uint64_t param) {
    ImeRequest r{0, 0, {}, {}};
    if (!param) return r;
    std::memcpy(&r.max_text_length, (const void*)(uintptr_t)(param + 36), 4);
    std::memcpy(&r.input_buffer,    (const void*)(uintptr_t)(param + 40), 8);
    uint64_t titlep = 0;
    std::memcpy(&titlep, (const void*)(uintptr_t)(param + 72), 8);
    r.title   = u16_to_utf8(titlep);
    r.initial = u16_to_utf8(r.input_buffer);
    return r;
}

uint32_t write_ime_text(uint64_t input_buffer, uint32_t max_len, const std::string& utf8) {
    if (!input_buffer) return 0;
    uint16_t* out = (uint16_t*)(uintptr_t)input_buffer;
    uint32_t n = 0;                                   // UTF-16 code units written (excl. NUL)
    for (size_t i = 0; i < utf8.size() && n < max_len; ) {
        unsigned char c = (unsigned char)utf8[i];
        uint32_t cp; int len;
        if (c < 0x80)              { cp = c;                 len = 1; }
        else if ((c >> 5) == 0x6)  { cp = c & 0x1F;          len = 2; }
        else if ((c >> 4) == 0xE)  { cp = c & 0x0F;          len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07;          len = 4; }
        else                       { i++;                    continue; }   // stray continuation byte
        if (i + (size_t)len > utf8.size()) break;
        for (int k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char)utf8[i + k] & 0x3F);
        i += len;
        if (cp < 0x10000) {
            out[n++] = (uint16_t)cp;
        } else {                                       // encode as a UTF-16 surrogate pair
            if (n + 2 > max_len) break;
            cp -= 0x10000;
            out[n++] = (uint16_t)(0xD800 | (cp >> 10));
            out[n++] = (uint16_t)(0xDC00 | (cp & 0x3FF));
        }
    }
    out[n] = 0;   // NUL-terminate (Sony input buffer is sized max_text_length + 1)
    return n;
}

} // namespace prosper
