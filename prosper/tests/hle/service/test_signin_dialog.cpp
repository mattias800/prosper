// test_signin_dialog — libSceSigninDialog runs its real common-dialog lifecycle and ends DISMISSED
// (#3784).
//
// Unregistered, all four NIDs Metaphor: ReFantazio imports answered the dispatcher's `return 0`.
// Its poll loop at eboot+0x863e80 is, transcribed from the dump:
//
//     863e89  call <sceSigninDialogUpdateStatus>
//     863e8e  mov  %eax,%ecx
//     863e92  cmp  $0x3,%ecx          ; FINISHED?
//     863e95  jne  <return 0>         ; no -> poll again next frame
//     863e97  call <sceSigninDialogTerminate>
//     863ea8  call *0x10(%rax)        ; the title's own completion callback
//
// so a `return 0` is NONE forever, and the title sat on a black screen after its SYSTEM page. The
// status is the RETURN value (SceCommonDialogStatus: NONE=0, INITIALIZED=1, RUNNING=2,
// FINISHED=3); a census of every local dump with tools/re/nid_gate_scan.py finds every classified
// UpdateStatus call site comparing the return with 3 or 2.
//
// This is not a sign-in shortcut: prosper reports the console SIGNED_OUT (hle/np), and the dialog
// ending dismissed is what a real user declining it produces. The title then shows its own
// "Unable to connect to the PlayStation Network" page and continues offline.
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper;

// `fails` is a local of main(); CHECK is only used there.
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

// The title's decision at eboot+0x863e92, transcribed.
static bool guest_sees_dialog_finished(uint64_t status) { return (uint32_t)status == 3u; }

int main() {
    int fails = 0;
    std::printf("== test_signin_dialog ==\n");
    register_np_hle();

    HleFn init   = Hle::lookup("mlYGfmqE3fQ");   // sceSigninDialogInitialize
    HleFn open   = Hle::lookup("JlpJVoRWv7U");   // sceSigninDialogOpen
    HleFn update = Hle::lookup("Bw31liTFT3A");   // sceSigninDialogUpdateStatus
    HleFn status = Hle::lookup("2m077aeC+PA");   // sceSigninDialogGetStatus
    HleFn close  = Hle::lookup("M3OkENHcyiU");   // sceSigninDialogClose
    HleFn term   = Hle::lookup("LXlmS6PvJdU");   // sceSigninDialogTerminate
    CHECK(init && open && update && status && close && term,
          "the six lifecycle NIDs are registered (unregistered == NONE forever)");
    if (!init || !open || !update || !status || !close || !term) {
        std::printf("== FAIL: %d ==\n", fails);
        return 1;
    }

    // Before Initialize there is no dialog: a poll must NOT read as finished, or a title that polls
    // early would run its completion callback for a dialog it never opened.
    CHECK((uint32_t)update(0, 0, 0, 0, 0, 0) == 0u, "no dialog before Initialize reads as NONE");

    CHECK(init(0, 0, 0, 0, 0, 0) == 0, "Initialize succeeds");
    CHECK((uint32_t)update(0, 0, 0, 0, 0, 0) == 1u, "after Initialize the status is INITIALIZED");
    CHECK(!guest_sees_dialog_finished(update(0, 0, 0, 0, 0, 0)),
          "an initialized-but-unopened dialog does not run the title's completion path");

    // Metaphor's Open argument, as built at eboot+0x862cae..0x862cbd.
    struct { uint32_t size; int32_t user_id; uint64_t reserved; } param{0x10, 1, 0};
    CHECK(open((uint64_t)(uintptr_t)&param, 0, 0, 0, 0, 0) == 0, "Open succeeds");
    const uint64_t polled = update(0, 0, 0, 0, 0, 0);
    CHECK(guest_sees_dialog_finished(polled),
          "after Open the title's poll sees FINISHED and leaves its wait loop (#3784)");
    CHECK((uint32_t)status(0, 0, 0, 0, 0, 0) == 3u, "GetStatus agrees with UpdateStatus");

    CHECK(term(0, 0, 0, 0, 0, 0) == 0, "Terminate succeeds");
    CHECK((uint32_t)update(0, 0, 0, 0, 0, 0) == 0u, "after Terminate the status is NONE again");

    // A null Open argument must not be dereferenced.
    init(0, 0, 0, 0, 0, 0);
    CHECK(open(0, 0, 0, 0, 0, 0) == 0 && guest_sees_dialog_finished(update(0, 0, 0, 0, 0, 0)),
          "Open with a null param still completes, and reads nothing through it");
    CHECK(close(0, 0, 0, 0, 0, 0) == 0 && (uint32_t)update(0, 0, 0, 0, 0, 0) == 3u,
          "Close leaves the dialog FINISHED");
    term(0, 0, 0, 0, 0, 0);

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
