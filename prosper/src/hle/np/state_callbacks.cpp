// state_callbacks.cpp — see state_callbacks.hpp.
//
// Evidence for the contracts below:
//  * Guest call sites (Dragon Quest VII Reimagined eboot, read by disassembly):
//    sceNetCtlRegisterCallback(fn, arg, &cid) once at boot, with a writable int for the id;
//    sceNetCtlCheckCallback() once per frame with no arguments;
//    sceNpRegisterStateCallbackA(fn, userdata) whose result is sign-tested (`js` -> failure path)
//    and kept as the callback id. The NpA callback reads (edi=userId, esi=state, rdx=userdata) and
//    branches on state 2 versus 1, so SIGNED_IN=2 and SIGNED_OUT=1; any other value is an error
//    branch for it.
//  * Both libraries are PS4-inherited; the PS5 3.20 library stubs export the same names and NIDs.
#include "hle/np/state_callbacks.hpp"

#include "hle/dispatch/callback_fs.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/service/service_trace.hpp"
#include "host/abi/guest_callback.hpp"

#include <cstdio>

namespace prosper::np {

int StateCallbackTable::add(uint64_t fn, uint64_t arg) {
    std::lock_guard<std::mutex> lk(mx_);
    for (int i = 0; i < kCapacity; ++i) {
        if (slots_[i].used) continue;
        slots_[i] = Slot{true, false, fn, arg};
        return i;
    }
    return -1;
}

bool StateCallbackTable::remove(int slot) {
    std::lock_guard<std::mutex> lk(mx_);
    if (slot < 0 || slot >= kCapacity || !slots_[slot].used) return false;
    slots_[slot] = Slot{};
    return true;
}

int StateCallbackTable::size() const {
    std::lock_guard<std::mutex> lk(mx_);
    int n = 0;
    for (const Slot& s : slots_) n += s.used ? 1 : 0;
    return n;
}

void StateCallbackTable::clear() {
    std::lock_guard<std::mutex> lk(mx_);
    slots_ = {};
}

namespace {

#define HLE(name)                                                                                  \
    static PROSPER_SYSV_ABI uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,      \
                                          uint64_t a4, uint64_t a5)

// SCE errors are negative int32 in eax; sign-extend so an int32 and an int64 read both see them.
constexpr uint64_t sce_err(uint32_t code) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(code)));
}

// ---- libSceNetCtl ------------------------------------------------------------------------------
// The console is network-DISCONNECTED, and that is the single answer sceNetCtlGetState,
// sceNetCtlGetInfo and the callback event below all give. Callback prototype:
// void cb(int eventType, void* arg). SCE_NET_CTL_EVENT_TYPE_DISCONNECTED = 1 and
// SCE_NET_CTL_STATE_DISCONNECTED = 0 are PS4-inherited constants. CONFIDENCE: HIGH on the
// contract, MED on the PS5 numeric values (no PS5 header; the guest's callback was not decoded).
constexpr int32_t kNetCtlStateDisconnected = 0;
constexpr uint64_t kNetCtlEventDisconnected = 1;
constexpr uint32_t kNetCtlErrCallbackMax = 0x80412103u;
constexpr uint32_t kNetCtlErrInvalidAddr = 0x80412107u;

StateCallbackTable g_netctl_callbacks;
StateCallbackTable g_np_callbacks;

// sceNetCtlRegisterCallback(func, arg, int* cid) -> SCE_OK, writing the callback id to *cid.
// The id is slot+1 (CONFIDENCE: LOW on the first id the console hands out; the guest only stores it).
// A null func or an unwritable cid is INVALID_ADDR and registers nothing; a full table is CALLBACK_MAX.
HLE(netctl_register_callback) {
    svc_log("sceNetCtlRegisterCallback", a0, a1, a2, a3, a4, a5);
    if (!svc_ptrish(a0) || !svc_ptrish(a2)) return sce_err(kNetCtlErrInvalidAddr);
    const int slot = g_netctl_callbacks.add(a0, a1);
    if (slot < 0) return sce_err(kNetCtlErrCallbackMax);
    const int32_t cid = slot + 1;
    if (!svc_write_bytes(a2, &cid, sizeof cid)) {
        g_netctl_callbacks.remove(slot);
        return sce_err(kNetCtlErrInvalidAddr);
    }
    return 0;
}

HLE(netctl_get_state) {
    svc_log("sceNetCtlGetState", a0, a1, a2, a3, a4, a5);
    if (!svc_ptrish(a0)) return sce_err(kNetCtlErrInvalidAddr);
    const int32_t state = kNetCtlStateDisconnected;
    if (!svc_write_bytes(a0, &state, sizeof state)) return sce_err(kNetCtlErrInvalidAddr);
    return 0;
}

// sceNetCtlCheckCallback(): runs on the caller's thread and delivers each registration its pending
// state event. A console with no link still owes the initial DISCONNECTED event.
extern "C" uint64_t prosper_netctl_check_callback_c(uint64_t a0, uint64_t a1, uint64_t a2,
                                                    uint64_t a3, uint64_t a4, uint64_t a5,
                                                    uint64_t entry_rsp) {
    svc_log("sceNetCtlCheckCallback", a0, a1, a2, a3, a4, a5);
    const uint64_t gfs = callback_guest_fs_from_entry_stack(entry_rsp);
    const int n = g_netctl_callbacks.pump([&](uint64_t fn, uint64_t arg) {
        call_guest_callback(fn, kNetCtlEventDisconnected, arg, 0, gfs);
    });
    // Log only after the host %fs is back: host libc reads %fs-based TLS.
    if (n) std::fprintf(stderr, "[svc] NetCtl state callback DELIVERED x%d (DISCONNECTED)\n", n);
    return 0;
}
PROSPER_HLE_ENTRY_WITH_GUEST_FS(prosper_netctl_check_callback_entry,
                                prosper_netctl_check_callback_c)

// ---- libSceNpManager ---------------------------------------------------------------------------
// Callback-A prototype: void cb(int32 userId, int32 state, void* userdata); SCE_NP_STATE_UNKNOWN=0,
// SIGNED_OUT=1, SIGNED_IN=2 (the guest above branches on exactly 1 and 2). The one signed-in user
// the console models is the initial user (id 1), and it is signed OUT. CONFIDENCE: HIGH.
constexpr uint32_t kNpErrInvalidArgument = 0x80550003u;
constexpr uint32_t kNpErrCallbackMax = 0x8055001Du;
constexpr uint64_t kNpInitialUserId = 1;
constexpr uint64_t kNpStateSignedOut = 1;

// sceNpRegisterStateCallbackA(func, userdata) -> callback id (> 0), or a negative SCE error.
HLE(np_register_state_callback_a) {
    svc_log("sceNpRegisterStateCallbackA", a0, a1, a2, a3, a4, a5);
    if (!a0) return sce_err(kNpErrInvalidArgument);
    const int slot = g_np_callbacks.add(a0, a1);
    if (slot < 0) return sce_err(kNpErrCallbackMax);
    return static_cast<uint64_t>(slot + 1);
}

extern "C" uint64_t prosper_np_check_callback_c(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                                uint64_t a4, uint64_t a5, uint64_t entry_rsp) {
    svc_log("sceNpCheckCallback", a0, a1, a2, a3, a4, a5);
    const uint64_t gfs = callback_guest_fs_from_entry_stack(entry_rsp);
    const int n = g_np_callbacks.pump([&](uint64_t fn, uint64_t arg) {
        call_guest_callback(fn, kNpInitialUserId, kNpStateSignedOut, arg, gfs);
    });
    if (n) std::fprintf(stderr, "[svc] Np state callback DELIVERED x%d (SIGNED_OUT)\n", n);
    return 0;
}
PROSPER_HLE_ENTRY_WITH_GUEST_FS(prosper_np_check_callback_entry, prosper_np_check_callback_c)

#undef HLE

}  // namespace

void register_state_callbacks_hle() {
    auto reg = [](const char* name, HleFn fn) { Hle::register_fn(nid_hash(name), fn, name); };
    reg("sceNetCtlRegisterCallback",
        reinterpret_cast<HleFn>(netctl_register_callback));  // UJ+Z7Q+4ck0
    reg("sceNetCtlCheckCallback",
        reinterpret_cast<HleFn>(prosper_netctl_check_callback_entry));   // iQw3iQPhvUQ
    reg("sceNetCtlGetState", reinterpret_cast<HleFn>(netctl_get_state));   // uBPlr0lbuiI
    reg("sceNpRegisterStateCallbackA",
        reinterpret_cast<HleFn>(np_register_state_callback_a));   // qQJfO8HAiaY
    reg("sceNpCheckCallback",
        reinterpret_cast<HleFn>(prosper_np_check_callback_entry));   // 3Zl8BePTh9Y
}

void reset_state_callbacks_for_test() {
    g_netctl_callbacks.clear();
    g_np_callbacks.clear();
}

}   // namespace prosper::np
