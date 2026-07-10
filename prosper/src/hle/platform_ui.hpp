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
    // The dialog's current status, polled by the guest until it is no longer RUNNING(2). Runs on the
    // guest's polling thread; the frontend advances the real UI on its own thread.
    virtual int  imeDialogStatus() { return 3 /*FINISHED*/; }
    // Write the outcome into the guest OrbisImeDialogResult* `result` (the endStatus, plus any entered
    // text into the input buffer the guest supplied at Init) and return the endStatus (0=OK,1=CANCELED).
    virtual int  imeDialogResult(uint64_t result) { (void)result; return 0; }
    // Tear down / abort the dialog (guest called Term or Abort).
    virtual void imeDialogClose() {}
};

// The registered backend, or nullptr when headless. The frontend calls set_platform_ui once at startup
// (before running the guest); the getter is used by the dialog HLE handlers on the guest thread.
PlatformUi* platform_ui();
void        set_platform_ui(PlatformUi* ui);

} // namespace prosper
