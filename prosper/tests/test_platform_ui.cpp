// test_platform_ui (#347) — the PlatformUi hook lets the app frontend service the guest's ImeDialog
// with real UI, while the headless default (auto-complete) stays the behavior when none is registered.
// This drives the sceImeDialog* HLE handlers both ways: no backend -> headless FINISHED; a registered
// backend -> Init/Status/Result/Close route to it (a real dialog's timing + outcome).
#include "../src/hle/dispatch.hpp"
#include "../src/hle/platform_ui.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// A stand-in frontend: takes ownership of the dialog, stays RUNNING until the test advances it, and
// reports USER_CANCELED with the entered (empty) text.
struct MockUi : PlatformUi {
    int status = 2 /*RUNNING*/, opens = 0, closes = 0, results = 0;
    uint64_t last_param = 0, last_extended = 0, last_result = 0;
    bool imeDialogOpen(uint64_t param, uint64_t extended) override {
        opens++; last_param = param; last_extended = extended; status = 2; return true;
    }
    int  imeDialogStatus() override { return status; }
    int  imeDialogResult(uint64_t result) override { results++; last_result = result;
        if (result) *(int32_t*)result = 1 /*USER_CANCELED*/; return 1; }
    void imeDialogClose() override { closes++; status = 0; }
};

int main() {
    printf("== test_platform_ui ==\n");
    register_builtin_hle();

    HleFn init = Hle::lookup("NUeBrN7hzf0"), status = Hle::lookup("IADmD4tScBY"),
          result = Hle::lookup("x01jxu+vxlc"), term = Hle::lookup("gyTyVn+bXMw");
    CHECK(init && status && result && term, "ImeDialog handlers registered");
    if (!(init && status && result && term)) { printf("== FAIL ==\n"); return 1; }

    // --- No backend (headless default): Init -> FINISHED immediately; result writes OK(0). ---
    set_platform_ui(nullptr);
    term(0,0,0,0,0,0);
    init(0,0,0,0,0,0);
    CHECK(status(0,0,0,0,0,0) == 3, "headless: Init auto-completes to FINISHED(3)");
    int32_t endStatus = 0x55;
    result((uint64_t)(uintptr_t)&endStatus, 0,0,0,0,0);
    CHECK(endStatus == 0, "headless: GetResult writes endStatus OK(0)");
    term(0,0,0,0,0,0);

    // --- With a registered backend: the dialog routes to it. ---
    MockUi ui;
    set_platform_ui(&ui);
    init(0x1234 /*param*/, 0x5678 /*extended*/, 0,0,0,0);
    CHECK(ui.opens == 1 && ui.last_param == 0x1234 && ui.last_extended == 0x5678,
          "backend: Init forwards the guest param pointers to imeDialogOpen");
    CHECK(status(0,0,0,0,0,0) == 2, "backend: status reflects the frontend (RUNNING while shown)");
    ui.status = 3;   // the frontend finishes the dialog (user confirmed/cancelled)
    CHECK(status(0,0,0,0,0,0) == 3, "backend: status follows the frontend to FINISHED");
    int32_t es = 0x55;
    result((uint64_t)(uintptr_t)&es, 0,0,0,0,0);
    CHECK(ui.results == 1 && ui.last_result == (uint64_t)(uintptr_t)&es,
          "backend: GetResult forwards the guest result pointer to imeDialogResult");
    CHECK(es == 1, "backend: the frontend's outcome (USER_CANCELED) reached the guest result struct");
    term(0,0,0,0,0,0);
    CHECK(ui.closes == 1, "backend: Term routes to imeDialogClose");

    // --- Unregister -> headless again (no dangling delegation to the freed backend). ---
    set_platform_ui(nullptr);
    init(0,0,0,0,0,0);
    CHECK(status(0,0,0,0,0,0) == 3, "after unregister: back to headless auto-complete");
    CHECK(ui.opens == 1, "after unregister: the old backend is no longer consulted");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
