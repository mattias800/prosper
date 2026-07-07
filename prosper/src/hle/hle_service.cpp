// hle_service.cpp — HLE of PS5 system services (user, NP/online, pad, mouse, app content,
// dialogs). Bring-up policy: openers return a valid positive handle; queries zero their
// output struct and report a sane "not signed in / no device" state and success, so the
// game gets consistent values instead of uninitialized memory.
#include "dispatch.hpp"
#include "nid.hpp"
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <atomic>

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define PW(x) ((void*)(uintptr_t)(x))

namespace { std::atomic<uint64_t> g_handle{1}; }

// --- user service ---
HLE(s_user_initial)   { if (a0) *(int32_t*)PW(a0) = 1; return 0; }           // GetInitialUser -> userId 1
HLE(s_user_idlist)    { if (a0) { int32_t* p = (int32_t*)PW(a0); p[0] = 1; for (int i = 1; i < 4; i++) p[i] = -1; } return 0; }
// Bounded, non-padding write (strncpy would zero-pad the whole a2-byte buffer -> a
// stack-smash if a2 is large/garbage). snprintf writes only the string + NUL.
HLE(s_user_name)      { if (a1) snprintf((char*)PW(a1), a2 ? (size_t)a2 : 17, "%s", "Player"); return 0; }
HLE(s_user_int_out)   { if (a1) *(int32_t*)PW(a1) = 0; return 0; }           // accessibility getters -> 0
HLE(s_user_age)       { if (a1) *(int32_t*)PW(a1) = 18; return 0; }          // GetAgeLevel -> adult (no restriction)
// sceUserServiceGetEvent(SceUserServiceEvent* ev): the event stream. A real system delivers the initial
// user's LOGIN event once at startup, then reports "no more events" so the game's drain loop terminates.
// The previous (unimplemented) stub returned 0 = "got an event" but left the struct unfilled -> the game
// either drained a garbage event or never saw the login it waits on. SceUserServiceEvent = { int32
// eventType (0=LOGIN,1=LOGOUT); int32 userId }. NO_EVENT = 0x80960009.
HLE(s_user_getevent)  {
    static std::atomic<int> delivered{0};
    if (a0 && delivered.exchange(1) == 0) { int32_t* ev = (int32_t*)PW(a0); ev[0] = 0; ev[1] = 1; return 0; }
    return 0x80960009ull;   // SCE_USER_SERVICE_ERROR_NO_EVENT
}
HLE(s_ok)             { return 0; }

// --- NP / online (single-player: report signed-out / unreachable, success) ---
HLE(s_np_state)       { if (a1) *(int32_t*)PW(a1) = 1; return 0; }           // SCE_NP_STATE_SIGNED_OUT
HLE(s_np_reach)       { if (a1) *(int32_t*)PW(a1) = 0; return 0; }
HLE(s_np_accountid)   { if (a1) *(uint64_t*)PW(a1) = 0; return 0; }
HLE(s_np_country)     { if (a1) memset(PW(a1), 0, 4); return 0; }

// --- pad / mouse (report a device that exists but has no input) ---
HLE(s_open)           { return g_handle++; }                                 // scePadOpen/sceMouseOpen -> handle
// Conservative output sizes so we don't smash a smaller caller buffer.
HLE(s_pad_info)       { if (a1) memset(PW(a1), 0, 0x20); return 0; }          // controller info -> zeroed
// ScePadData is 0x78 bytes (cross-checked against the Kyty reference PadData: buttons @0x00,
// sticks @0x04, touch data @0x30, connected @0x48, timestamp @0x50, extension tail to 0x77).
// Zero ALL of it: a stub that claims "1 entry read" while zeroing only part of the struct hands
// the game uninitialized touch/timestamp/connected bytes. CONFIDENCE: MED (offsets from Kyty).
static void fill_pad_data(uint8_t* d) {
    memset(d, 0, 0x78);
    if (getenv("PROSPER_PAD_PRESS")) {
        *(uint32_t*)(d + 0x00) = 0x4000u;              // buttons: CROSS held
        d[0x04] = d[0x05] = d[0x06] = d[0x07] = 128;   // sticks centered
        d[0x48] = 1;                                   // connected = true
    }
}
HLE(s_pad_read)       {
    static unsigned n = 0; n++;
    if (getenv("PROSPER_PADLOG") && (n <= 5 || n % 200 == 0)) fprintf(stderr, "[pad] scePadRead call #%u\n", n);
    if (!a1) return 0;                                 // no output buffer -> no entries read
    fill_pad_data((uint8_t*)PW(a1));
    return 1;   // scePadRead returns the number of pad data entries read (>0 = data available)
}
// scePadReadState. Normally neutral (all zero). Two diagnostics: PROSPER_PADLOG counts calls (is the game
// polling input each frame, i.e. is the frame loop live?); PROSPER_PAD_PRESS injects a connected controller
// with a button held (CROSS=0x4000) so a "Press [button]" title screen can advance to a shader-drawn scene.
HLE(s_pad_readstate)  {
    static unsigned n = 0; n++;
    if (getenv("PROSPER_PADLOG") && (n <= 5 || n % 200 == 0))
        fprintf(stderr, "[pad] scePadReadState call #%u\n", n);
    if (a1) fill_pad_data((uint8_t*)PW(a1));
    return 0;
}
// sceMouseRead(handle, SceMouseData*, num) returns the number of mouse events read. SceMouseData
// (~0x18 bytes) is NOT ScePadData — sharing the pad stub returned one "valid" entry whose 0x30-byte
// memset already overran a single-entry mouse buffer, and the game consumed a phantom mouse event
// every call. No mouse attached: zero one entry defensively, report 0 events.
HLE(s_mouse_read)     { if (a1) memset(PW(a1), 0, 0x18); return 0; }

// --- app content ---
HLE(s_appcontent_int) { if (a1) *(int32_t*)PW(a1) = 0; return 0; }

// Common/message dialogs: report FINISHED immediately so the game's "wait until dismissed" loop
// exits and it proceeds (we have no interactive dialog UI yet). Status enum: NONE=0, INITIALIZED=1,
// RUNNING=2, FINISHED=3. GetResult -> zeroed struct = OK/no button pressed.
HLE(s_dialog_finished) { return 3; }
HLE(s_dialog_result)   { if (a0) memset(PW(a0), 0, 0x30); return 0; }  // SceMsgDialogResult ~0x30 (don't overflow)

void register_service_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // user service
    R("sceUserServiceGetInitialUser", s_user_initial);
    R("sceUserServiceGetEvent", s_user_getevent);   // deliver the initial-user LOGIN event once
    R("sceUserServiceGetLoginUserIdList", s_user_idlist);
    R("sceUserServiceGetUserName", s_user_name);
    R("sceUserServiceGetAccessibilityVibration", s_user_int_out);
    R("sceUserServiceGetAccessibilityPressAndHoldDelay", s_user_int_out);
    R("sceUserServiceGetAccessibilityZoomEnabled", s_user_int_out);
    // More (userId, int* out) getters the game queries at startup — same family: write a sane default
    // (accessibility off = 0; age level = adult) so the caller reads a deterministic value instead of
    // uninitialized stack. Registered by raw NID (guaranteed match; these names aren't in our NidDb).
    Hle::register_fn("woNpu+45RLk", (HleFn)s_user_age,     "sceUserServiceGetAgeLevel");
    Hle::register_fn("rnEhHqG-4xo", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityChatTranscription");
    Hle::register_fn("O6IW1-Dwm-w", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityZoomFollowFocus");
    Hle::register_fn("-3Y5GO+-i78", (HleFn)s_user_int_out, "sceUserServiceGetAccessibilityTriggerEffect");
    // NOTE: sceUserServiceGetGamePresets (-sD02mFDBh4) intentionally left unimplemented — its output is
    // a struct of unknown layout; writing a wrong-size default would risk a stack smash (cf. f_fstat).
    R("sceUserServiceInitialize", s_ok);
    R("sceUserServiceTerminate", s_ok);
    // NP
    R("sceNpGetState", s_np_state);
    R("sceNpGetNpReachabilityState", s_np_reach);
    R("sceNpGetAccountIdA", s_np_accountid);
    R("sceNpGetAccountCountryA", s_np_country);
    R("sceNpCheckCallback", s_ok);
    R("sceNpRegisterStateCallback", s_ok);
    // pad / mouse
    R("scePadInit", s_ok);
    R("scePadOpen", s_open);
    R("scePadClose", s_ok);
    R("scePadGetControllerInformation", s_pad_info);
    R("scePadDeviceClassGetExtendedInformation", s_pad_info);
    R("scePadReadState", s_pad_readstate);
    R("scePadRead", s_pad_read);
    R("sceMouseInit", s_ok);
    R("sceMouseOpen", s_open);
    R("sceMouseRead", s_mouse_read);
    // app content / dialogs
    R("sceAppContentInitialize", s_ok);
    R("sceAppContentAppParamGetInt", s_appcontent_int);
    R("sceCommonDialogInitialize", s_ok);
    R("sceSystemServiceParamGetInt", s_appcontent_int);
    // message dialog: auto-dismiss (report FINISHED) so the startup dialog flow completes
    R("sceMsgDialogInitialize", s_ok);      R("sceMsgDialogTerminate", s_ok);
    R("sceMsgDialogOpen", s_ok);            R("sceMsgDialogClose", s_ok);
    R("sceMsgDialogUpdateStatus", s_dialog_finished);
    R("sceMsgDialogGetStatus", s_dialog_finished);
    R("sceMsgDialogGetResult", s_dialog_result);
    R("sceSystemServiceHideSplashScreen", s_ok);
    R("sceSystemServiceGetStatus", s_appcontent_int);
    #undef R
}

} // namespace prosper
