// platform_ui.hpp — a host-owned interactive-UI backend the app frontend registers so the guest's
// blocking dialogs / IME are serviced by REAL UI instead of the core's headless auto-complete.
//
// This is the same core<->frontend pattern already used for the renderer (gpu::set_submit_renderer),
// audio (audio_set_sink) and controllers (pad backends): the core owns the Sony API surface and its
// headless default; a concrete frontend (SDL3, ...) installs a PlatformUi from OUTSIDE the core via
// set_platform_ui(). When none is registered, the dialog/IME HLE handlers keep their current headless
// behavior (auto-complete / no-device), so boot_trace and CI are unchanged.
//
// The core passes the guest struct POINTERS through opaquely (guest == host address). Only the frontend
// needs to know the Orbis struct layouts (OrbisImeDialogParam / Result, etc.) — the core stays
// layout-agnostic, so there's no duplicated, drifting layout knowledge in prosper_core.
//
// Status values are the shared SceCommonDialogStatus: 0=NONE, 1=INITIALIZED, 2=RUNNING, 3=FINISHED.
#pragma once
#include <cstdint>

namespace prosper {

struct PlatformUi {
    virtual ~PlatformUi() = default;

    // --- libSceImeDialog (on-screen text entry) ---
    // Begin showing a text-entry dialog. `param`/`extended` are the guest sceImeDialogInit arguments
    // (OrbisImeDialogParam* and the extended param, as guest addresses). Return true to OWN the dialog
    // — the core then routes status/result/close here; return false to let the core auto-complete it.
    virtual bool imeDialogOpen(uint64_t param, uint64_t extended) { (void)param; (void)extended; return false; }
    // The dialog's current status (OrbisImeDialogStatus: NONE=0, RUNNING=1, FINISHED=2 — the IME
    // dialog's OWN enum, NOT the 4-value common-dialog status MsgDialog uses), polled by the guest
    // until it reads FINISHED(2). Runs on the guest's polling thread; the frontend advances the real
    // UI on its own thread.
    virtual int  imeDialogStatus() { return 2 /*OrbisImeDialogStatus::Finished*/; }
    // Write the outcome into the guest OrbisImeDialogResult* `result` (the endStatus, plus any entered
    // text into the input buffer the guest supplied at Init) and return the endStatus (0=OK,1=CANCELED).
    virtual int  imeDialogResult(uint64_t result) { (void)result; return 0; }
    // Tear down / abort the dialog (guest called Term or Abort).
    virtual void imeDialogClose() {}

    // --- libSceMsgDialog (message dialog: text + buttons: OK/YES-NO/progress) ---
    // Show a message dialog. `param` is the guest sceMsgDialogOpen argument (SceMsgDialogParam*, opaque
    // guest address). Return true to OWN it; false -> the core auto-dismisses headlessly.
    virtual bool msgDialogOpen(uint64_t param) { (void)param; return false; }
    virtual int  msgDialogStatus() { return 3 /*FINISHED*/; }
    // Write the guest SceMsgDialogResult* `result` (mode / result / buttonId) and return the buttonId.
    virtual int  msgDialogResult(uint64_t result) { (void)result; return 0; }
    virtual void msgDialogClose() {}

    // --- libSceSaveDataDialog (save/load/delete confirmation and save-slot UI) ---
    // `param` is the guest SceSaveDataDialogParam*. Return true to own this Open; the core then
    // routes every status/result/progress/close call to this same backend instance. Returning false
    // preserves the headless policy: Open auto-completes with a neutral result.
    virtual bool saveDataDialogOpen(uint64_t param) { (void)param; return false; }
    virtual int  saveDataDialogStatus() { return 3 /*FINISHED*/; }
    virtual int  saveDataDialogResult(uint64_t result) { (void)result; return 0; }
    virtual void saveDataDialogProgressBarInc(uint32_t target, uint32_t delta) {
        (void)target; (void)delta;
    }
    virtual void saveDataDialogProgressBarSetValue(uint32_t target, uint32_t value) {
        (void)target; (void)value;
    }
    virtual void saveDataDialogClose() {}

    // --- libSceErrorDialog (single-button error message; no result struct) ---
    // Show an error dialog. `param` is the guest sceErrorDialogOpen argument (opaque). Return true to
    // OWN it; false -> the core auto-dismisses headlessly (and logs the errorCode).
    virtual bool errorDialogOpen(uint64_t param) { (void)param; return false; }
    virtual int  errorDialogStatus() { return 3 /*FINISHED*/; }
    virtual void errorDialogClose() {}

    // --- libSceIme keyboard device presence ---
    // Fill up to `max` connected keyboard resource ids for `userId` into `outIds`, and return the
    // count. The headless default is 0 (no keyboard); a windowed frontend with a host keyboard reports
    // its own (a resource id must be non-zero). sceImeKeyboardGetResourceId/GetInfo consult this so a
    // title enables keyboard input only when a real keyboard is present.
    virtual int keyboardResourceIds(int32_t userId, uint32_t* outIds, int max) {
        (void)userId; (void)outIds; (void)max; return 0;
    }
};

// The registered backend, or nullptr when headless. The frontend calls set_platform_ui once at startup
// (before running the guest); the getter is used by the dialog HLE handlers on the guest thread.
PlatformUi* platform_ui();
void        set_platform_ui(PlatformUi* ui);

} // namespace prosper
