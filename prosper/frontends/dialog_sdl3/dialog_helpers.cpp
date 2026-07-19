// dialog_helpers.cpp — see dialog_helpers.hpp. Pure struct parsing (no SDL). SaveData's nested guest
// pointers use fault-contained OS copies; memcpy keeps local unaligned fields alias-safe.
#include "dialog_helpers.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <utility>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include "../../src/host/posix_shim.hpp"
#include <sys/uio.h>
#include <unistd.h>
#endif

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

// --- SaveDataDialog ---

namespace {

bool guest_read_bytes(uint64_t address, void* destination, size_t size) {
    if (!address || !destination || !size || address > UINT64_MAX - (size - 1)) return false;
#ifdef _WIN32
    SIZE_T copied = 0;
    return ReadProcessMemory(GetCurrentProcess(), (const void*)(uintptr_t)address,
                             destination, size, &copied) && copied == size;
#else
    iovec local{destination, size};
    iovec remote{(void*)(uintptr_t)address, size};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)size;
#endif
}

bool guest_write_bytes(uint64_t address, const void* source, size_t size) {
    if (!address || !source || !size || address > UINT64_MAX - (size - 1)) return false;
#ifdef _WIN32
    SIZE_T copied = 0;
    return WriteProcessMemory(GetCurrentProcess(), (void*)(uintptr_t)address,
                              source, size, &copied) && copied == size;
#else
    iovec local{const_cast<void*>(source), size};
    iovec remote{(void*)(uintptr_t)address, size};
    return process_vm_writev(getpid(), &local, 1, &remote, 1, 0) == (ssize_t)size;
#endif
}

template<typename T, size_t N>
T local_field(const std::array<uint8_t, N>& bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof value);
    return value;
}

bool guest_string(uint64_t address, std::string& out) {
    out.clear();
    if (!address) return true;
    constexpr size_t CAP = 4096;
    std::array<char, 256> chunk{};
    size_t consumed = 0;
    while (consumed < CAP) {
        if (address > UINT64_MAX - consumed) return false;
        const uint64_t current = address + consumed;
        const size_t page_left = 0x1000u - (size_t)(current & 0xfffu);
        const size_t count = std::min({chunk.size(), page_left, CAP - consumed});
        if (!guest_read_bytes(current, chunk.data(), count)) return false;
        const void* nul = std::memchr(chunk.data(), 0, count);
        const size_t length = nul ? (size_t)((const char*)nul - chunk.data()) : count;
        out.append(chunk.data(), length);
        if (nul) return true;
        consumed += count;
    }
    return false;
}

bool guest_dir_name(uint64_t address, std::string& out) {
    std::array<char, 32> name{};
    if (!guest_read_bytes(address, name.data(), name.size())) return false;
    const char* end = (const char*)std::memchr(name.data(), 0, name.size());
    if (!end) return false; // the result ABI needs a bounded, NUL-terminated 32-byte name
    out.assign(name.data(), (size_t)(end - name.data()));
    return true;
}

bool read_save_back_option(const std::array<uint8_t, 0x98>& outer, bool& enabled) {
    enabled = true;
    const uint64_t option = local_field<uint64_t>(outer, 0x78);
    if (!option) return true;
    uint32_t value = 0;
    if (!guest_read_bytes(option, &value, sizeof value)) return false;
    enabled = value == 0; // OptionBack::ENABLE=0, DISABLE=1
    return true;
}

MsgButtons save_user_buttons(uint32_t type) {
    switch (type) {
    case 0: return {1, {"OK", nullptr, nullptr}, {1, 0, 0}};
    case 1: return {2, {"Yes", "No", nullptr}, {1, 2, 0}};
    case 3: return {2, {"OK", "Cancel", nullptr}, {1, 0, 0}};
    default: return {0, {nullptr, nullptr, nullptr}, {0, 0, 0}};
    }
}

const char* save_action(uint32_t display_type) {
    switch (display_type) {
    case 1: return "save";
    case 2: return "load";
    case 3: return "delete";
    default: return "continue";
    }
}

} // namespace

SaveDataRequest read_savedata_request(uint64_t param, SaveDataSlotTimeLookup slotTime) {
    SaveDataRequest request;
    if (!param) return request;
    std::array<uint8_t, 0x98> outer{};
    if (!guest_read_bytes(param, outer.data(), outer.size())) return request;
    if (local_field<uint64_t>(outer, 0x00) != 0x30 ||
        local_field<uint32_t>(outer, 0x30) != 0x98)
        return request;
    request.mode = local_field<uint32_t>(outer, 0x34);
    request.displayType = local_field<uint32_t>(outer, 0x38);
    request.userData = local_field<uint64_t>(outer, 0x70);

    if (request.mode == SAVE_DATA_DIALOG_MODE_LIST) {
        if (request.displayType < 1 || request.displayType > 3) return request;
        bool back_enabled = true;
        if (!read_save_back_option(outer, back_enabled)) return request;
        request.cancelable = back_enabled;

        const uint64_t items_address = local_field<uint64_t>(outer, 0x48);
        if (!items_address) return request;
        // SaveDialogItems owned prefix through itemStyle: dirName@0x10, count@0x18,
        // newItem@0x20, focusPos@0x28, focusPosDirName@0x30, itemStyle@0x38.
        std::array<uint8_t, 0x3c> items{};
        if (!guest_read_bytes(items_address, items.data(), items.size())) return request;
        const uint64_t names_address = local_field<uint64_t>(items, 0x10);
        const uint32_t names_count = local_field<uint32_t>(items, 0x18);
        constexpr uint32_t MAX_SLOT_NAMES = 1024;
        if (names_count > MAX_SLOT_NAMES || (names_count && !names_address)) return request;

        const uint64_t new_item_address = local_field<uint64_t>(items, 0x20);
        if (request.displayType == 1 && new_item_address) {
            uint64_t title_address = 0;
            if (!guest_read_bytes(new_item_address, &title_address, sizeof title_address))
                return request;
            SaveDataSlot slot;
            if (title_address && !guest_string(title_address, slot.label)) return request;
            if (slot.label.empty()) slot.label = "New Save";
            request.slots.push_back(std::move(slot));
        }

        for (uint32_t i = 0; i < names_count; ++i) {
            if (names_address > UINT64_MAX - (uint64_t)i * 32u) return SaveDataRequest{};
            SaveDataSlot slot;
            if (!guest_dir_name(names_address + (uint64_t)i * 32u, slot.dirName))
                return SaveDataRequest{};
            if (slot.dirName.empty()) continue;
            slot.label = slot.dirName;
            request.slots.push_back(std::move(slot));
        }

        const uint32_t focus = local_field<uint32_t>(items, 0x28);
        if (!request.slots.empty()) {
            if (focus == 1 || focus == 3) {
                request.initialSlot = request.slots.size() - 1;
            } else if (focus == 2 && request.slots.front().dirName.empty() &&
                       request.slots.size() > 1) {
                request.initialSlot = 1; // DATAHEAD skips the SAVE new-item entry
            } else if (focus == 4 || focus == 5) {
                if (!slotTime) return SaveDataRequest{};
                bool found_time = false;
                int64_t best_time = 0;
                for (size_t index = 0; index < request.slots.size(); ++index) {
                    const SaveDataSlot& slot = request.slots[index];
                    if (slot.dirName.empty()) continue;
                    int64_t modified = 0;
                    if (!slotTime(slot.dirName, modified)) continue;
                    if (!found_time || (focus == 4 ? modified > best_time : modified < best_time)) {
                        found_time = true;
                        best_time = modified;
                        request.initialSlot = index;
                    }
                }
                if (!found_time) {
                    if (!request.slots.front().dirName.empty()) return SaveDataRequest{};
                    request.initialSlot = 0; // reference UI focuses the new item when no data exists
                }
            } else if (focus == 6) {
                const uint64_t focus_name_address = local_field<uint64_t>(items, 0x30);
                std::string focus_name;
                if (!focus_name_address || !guest_dir_name(focus_name_address, focus_name))
                    return SaveDataRequest{};
                const auto found = std::find_if(request.slots.begin(), request.slots.end(),
                    [&](const SaveDataSlot& slot) { return slot.dirName == focus_name; });
                if (found != request.slots.end())
                    request.initialSlot = (size_t)(found - request.slots.begin());
            }
        } else if (!request.cancelable) {
            return request; // avoid owning a list with neither a selectable item nor a Back path
        }

        request.message = request.displayType == 1 ? "Select where to save." :
                          request.displayType == 2 ? "Select saved data to load." :
                                                     "Select saved data to delete.";
        request.slotList = true;
        request.supported = true;
        return request;
    }

    if (request.mode == SAVE_DATA_DIALOG_MODE_USER_MSG) {
        const uint64_t user = local_field<uint64_t>(outer, 0x50);
        if (!user) return request;
        std::array<uint8_t, 16> user_param{};
        if (!guest_read_bytes(user, user_param.data(), user_param.size())) return request;
        request.buttons = save_user_buttons(local_field<uint32_t>(user_param, 0x00));
        if (request.buttons.count == 0) return request;
        request.error = local_field<uint32_t>(user_param, 0x04) == 1;
        if (!guest_string(local_field<uint64_t>(user_param, 0x08), request.message))
            return request;
        request.supported = true;
        return request;
    }

    if (request.mode == SAVE_DATA_DIALOG_MODE_SYSTEM_MSG) {
        const uint64_t system = local_field<uint64_t>(outer, 0x58);
        if (!system) return request;
        uint32_t type = 0;
        if (!guest_read_bytes(system, &type, sizeof type)) return request;
        const char* action = save_action(request.displayType);
        char message[192] = {};
        switch (type) {
        case 1: // NODATA
            request.message = "There is no saved data.";
            request.buttons = save_user_buttons(0);
            break;
        case 2: // CONFIRM
            std::snprintf(message, sizeof message, "Do you want to %s this saved data?", action);
            request.message = message;
            request.buttons = save_user_buttons(1);
            break;
        case 3: // OVERWRITE
            request.message = "Do you want to overwrite the existing saved data?";
            request.buttons = save_user_buttons(1);
            break;
        case 4: case 8: // NOSPACE / NOSPACE_CONTINUABLE
            request.message = "There is not enough space for this saved data operation.";
            request.buttons = save_user_buttons(0);
            request.error = true;
            break;
        case 6: // FILE_CORRUPTED
            request.message = "The saved data is corrupted.";
            request.buttons = save_user_buttons(0);
            request.error = true;
            break;
        case 7: // FINISHED
            std::snprintf(message, sizeof message, "Saved data %s completed.", action);
            request.message = message;
            request.buttons = save_user_buttons(0);
            break;
        case 10: // CORRUPTED_AND_DELETED
            request.message = "The saved data is corrupted and will be deleted.";
            request.error = true;
            break;
        case 11: // CORRUPTED_AND_CREATED
            request.message = "The corrupted saved data will be replaced with new saved data.";
            request.error = true;
            break;
        case 13: // CORRUPTED_AND_RESTORE
            request.message = "The corrupted saved data will be restored from its backup.";
            request.error = true;
            break;
        case 14: // TOTAL_SIZE_EXCEEDED
            request.message = "Cannot create more saved data.";
            request.buttons = save_user_buttons(0);
            request.error = true;
            break;
        default:
            return request; // PROGRESS and unknown modes need a non-modal/dedicated UI
        }
        if (type == 10 || type == 11 || type == 13) {
            bool back_enabled = true; // OptionBack defaults to ENABLE when the pointer is absent.
            if (!read_save_back_option(outer, back_enabled)) return request;
            request.cancelable = back_enabled;
            request.buttons = save_user_buttons(back_enabled ? 3u : 0u);
        }
        request.supported = true;
        return request;
    }

    if (request.mode == SAVE_DATA_DIALOG_MODE_ERROR_CODE) {
        const uint64_t error = local_field<uint64_t>(outer, 0x60);
        if (!error) return request;
        uint32_t code = 0;
        if (!guest_read_bytes(error, &code, sizeof code)) return request;
        char message[96];
        std::snprintf(message, sizeof message, "An error occurred.\n\nError code: 0x%08X", code);
        request.message = message;
        request.buttons = save_user_buttons(0);
        request.error = true;
        request.supported = true;
        return request;
    }

    if (request.mode == SAVE_DATA_DIALOG_MODE_PROGRESS_BAR) {
        const uint64_t progress = local_field<uint64_t>(outer, 0x68);
        if (!progress) return request;
        // ProgressBarParam prefix: barType@0, pad@4, msg@8, sysMsgType@16. The only documented
        // bar type is PERCENTAGE=0; reading just the owned prefix avoids relying on reserved size.
        std::array<uint8_t, 20> progress_param{};
        if (!guest_read_bytes(progress, progress_param.data(), progress_param.size()) ||
            local_field<uint32_t>(progress_param, 0x00) != 0)
            return request;
        const uint64_t message = local_field<uint64_t>(progress_param, 0x08);
        if (message) {
            if (!guest_string(message, request.message)) return request;
        } else {
            switch (local_field<uint32_t>(progress_param, 0x10)) {
            case 0: // INVALID: intentionally no message
                break;
            case 1: // PROGRESS: choose the inherited action-specific system text
                request.message = request.displayType == 1 ? "Saving..." :
                                  request.displayType == 2 ? "Loading..." :
                                  request.displayType == 3 ? "Deleting..." : "";
                break;
            case 2: // RESTORE
                request.message = "Restoring saved data...";
                break;
            default:
                return request;
            }
        }
        request.progress = true;
        request.cancelable = false;
        request.supported = true;
    }
    return request;
}

void write_savedata_result(uint64_t result, const SaveDataRequest& request,
                           uint32_t buttonId, bool canceled, const std::string& dirName) {
    if (!result || result > UINT64_MAX - 0x20) return;
    const uint32_t common_result = canceled ? 1u : 0u;
    const uint32_t guest_button = canceled ? 0u : buttonId;
    (void)guest_write_bytes(result + 0x00, &request.mode, sizeof request.mode);
    (void)guest_write_bytes(result + 0x04, &common_result, sizeof common_result);
    (void)guest_write_bytes(result + 0x08, &guest_button, sizeof guest_button);
    (void)guest_write_bytes(result + 0x20, &request.userData, sizeof request.userData);
    if (request.mode == SAVE_DATA_DIALOG_MODE_LIST) {
        uint64_t output = 0;
        if (guest_read_bytes(result + 0x10, &output, sizeof output) && output) {
            std::array<char, 32> name{};
            const size_t length = std::min(dirName.size(), name.size() - 1);
            std::memcpy(name.data(), dirName.data(), length);
            (void)guest_write_bytes(output, name.data(), name.size());
        }
    }
}

void SaveDataDialogState::open(SaveDataRequest request) {
    std::lock_guard<std::mutex> lock(mx_);
    ++generation_;
    if (!generation_) ++generation_; // zero is reserved for "no ticket"
    request_ = std::move(request);
    button_ = 0;
    progress_ = 0;
    canceled_ = false;
    dir_name_.clear();
    status_ = 2 /*RUNNING*/;
    pending_generation_ = request_.progress ? 0 : generation_;
}

int SaveDataDialogState::status() const {
    std::lock_guard<std::mutex> lock(mx_);
    return status_;
}

void SaveDataDialogState::close() {
    std::lock_guard<std::mutex> lock(mx_);
    ++generation_;
    if (!generation_) ++generation_;
    pending_generation_ = 0;
    status_ = 0 /*NONE*/;
}

SaveDataModalTicket SaveDataDialogState::take_pending() {
    std::lock_guard<std::mutex> lock(mx_);
    const uint64_t pending = pending_generation_;
    pending_generation_ = 0;
    if (!pending || pending != generation_ || status_ != 2 /*RUNNING*/) return {};
    return {pending, request_};
}

bool SaveDataDialogState::active(uint64_t generation) const {
    std::lock_guard<std::mutex> lock(mx_);
    return generation && generation_ == generation && status_ == 2 /*RUNNING*/;
}

void SaveDataDialogState::complete(uint64_t generation, int clicked) {
    std::lock_guard<std::mutex> lock(mx_);
    if (!generation || generation_ != generation || status_ != 2 /*RUNNING*/) return;
    canceled_ = clicked <= 0;
    button_ = canceled_ ? 0u : (uint32_t)clicked;
    status_ = 3 /*FINISHED*/;
}

void SaveDataDialogState::complete_list(uint64_t generation, int selected) {
    std::lock_guard<std::mutex> lock(mx_);
    if (status_ != 2 || generation != generation_ || !request_.slotList) return;
    if (selected >= 0 && (size_t)selected < request_.slots.size()) {
        dir_name_ = request_.slots[(size_t)selected].dirName;
        canceled_ = false;
    } else {
        dir_name_.clear();
        canceled_ = true;
    }
    button_ = 0; // LIST selection has no button id
    status_ = 3 /*FINISHED*/;
}

SaveDataResultSnapshot SaveDataDialogState::result_snapshot() const {
    std::lock_guard<std::mutex> lock(mx_);
    return {request_, button_, canceled_, dir_name_};
}

void SaveDataDialogState::progress_inc(uint32_t target, uint32_t delta) {
    std::lock_guard<std::mutex> lock(mx_);
    if (target != 0 || status_ != 2 /*RUNNING*/ || !request_.progress) return;
    progress_ = delta > UINT32_MAX - progress_ ? UINT32_MAX : progress_ + delta;
}

void SaveDataDialogState::progress_set(uint32_t target, uint32_t value) {
    std::lock_guard<std::mutex> lock(mx_);
    if (target != 0 || status_ != 2 /*RUNNING*/ || !request_.progress) return;
    progress_ = value;
}

void SaveDataDialogState::finish_progress(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mx_);
    if (!generation || generation_ != generation || status_ != 2 /*RUNNING*/ ||
        !request_.progress)
        return;
    button_ = 0;
    canceled_ = false;
    status_ = 3 /*FINISHED: preserve the core's neutral headless fallback*/;
}

SaveDataProgressSnapshot SaveDataDialogState::progress_snapshot() const {
    std::lock_guard<std::mutex> lock(mx_);
    if (status_ != 2 /*RUNNING*/ || !request_.progress) return {};
    return {generation_, request_, progress_};
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
