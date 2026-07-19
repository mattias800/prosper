// test_platform_ui (#347) — the PlatformUi hook lets the app frontend service the guest's ImeDialog
// with real UI, while the headless default (auto-complete) stays the behavior when none is registered.
// This drives the sceImeDialog* HLE handlers both ways: no backend -> headless FINISHED; a registered
// backend -> Init/Status/Result/Close route to it (a real dialog's timing + outcome).
#include "../src/hle/dispatch.hpp"
#include "../src/hle/platform_ui.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// A stand-in frontend: takes ownership of the dialog, stays RUNNING until the test advances it, and
// reports USER_CANCELED with the entered (empty) text.
struct MockUi : PlatformUi {
    int status = 1 /*OrbisImeDialogStatus::Running*/, opens = 0, closes = 0, results = 0;
    uint64_t last_param = 0, last_extended = 0, last_result = 0;
    bool imeDialogOpen(uint64_t param, uint64_t extended) override {
        opens++; last_param = param; last_extended = extended; status = 1; return true;
    }
    int  imeDialogStatus() override { return status; }
    int  imeDialogResult(uint64_t result) override { results++; last_result = result;
        if (result) *(int32_t*)result = 1 /*USER_CANCELED*/; return 1; }
    void imeDialogClose() override { closes++; status = 0; }

    // MsgDialog
    int msg_status = 2, msg_opens = 0, msg_closes = 0, msg_results = 0; uint64_t msg_last_param = 0;
    bool msgDialogOpen(uint64_t param) override { msg_opens++; msg_last_param = param; msg_status = 2; return true; }
    int  msgDialogStatus() override { return msg_status; }
    int  msgDialogResult(uint64_t result) override { msg_results++; if (result) *(uint32_t*)(result + 8) = 2 /*buttonId*/; return 2; }
    void msgDialogClose() override { msg_closes++; msg_status = 0; }

    // ErrorDialog
    int err_status = 2, err_opens = 0, err_closes = 0; uint64_t err_last_param = 0;
    bool errorDialogOpen(uint64_t param) override { err_opens++; err_last_param = param; err_status = 2; return true; }
    int  errorDialogStatus() override { return err_status; }
    void errorDialogClose() override { err_closes++; err_status = 0; }

    // SaveDataDialog
    bool save_accept = true;
    int save_status = 2, save_opens = 0, save_closes = 0, save_results = 0;
    int save_incs = 0, save_sets = 0;
    uint64_t save_last_param = 0, save_last_result = 0;
    uint32_t save_last_target = 0, save_last_value = 0;
    bool saveDataDialogOpen(uint64_t param) override {
        save_opens++; save_last_param = param; save_status = 2; return save_accept;
    }
    int saveDataDialogStatus() override { return save_status; }
    int saveDataDialogResult(uint64_t result) override {
        save_results++; save_last_result = result;
        if (result) *(uint32_t*)result = 4 /*ERROR_CODE marker*/;
        return 1;
    }
    void saveDataDialogProgressBarInc(uint32_t target, uint32_t delta) override {
        save_incs++; save_last_target = target; save_last_value = delta;
    }
    void saveDataDialogProgressBarSetValue(uint32_t target, uint32_t value) override {
        save_sets++; save_last_target = target; save_last_value = value;
    }
    void saveDataDialogClose() override { save_closes++; save_status = 0; }
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
    CHECK(status(0,0,0,0,0,0) == 2, "headless: Init auto-completes to ImeDialog Finished(2)");
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
    CHECK(status(0,0,0,0,0,0) == 1, "backend: status reflects the frontend (ImeDialog Running=1 while shown)");
    ui.status = 2;   // the frontend finishes the dialog (user confirmed/cancelled)
    CHECK(status(0,0,0,0,0,0) == 2, "backend: status follows the frontend to ImeDialog Finished(2)");
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
    CHECK(status(0,0,0,0,0,0) == 2, "after unregister: back to headless auto-complete (Finished=2)");
    CHECK(ui.opens == 1, "after unregister: the old backend is no longer consulted");

    // --- MsgDialog: headless auto-dismiss vs backend delegation. ---
    HleFn m_init = Hle::lookup(nid_hash("sceMsgDialogInitialize")), m_open = Hle::lookup(nid_hash("sceMsgDialogOpen")),
          m_status = Hle::lookup(nid_hash("sceMsgDialogGetStatus")), m_result = Hle::lookup(nid_hash("sceMsgDialogGetResult")),
          m_close = Hle::lookup(nid_hash("sceMsgDialogClose"));
    CHECK(m_init && m_open && m_status && m_result && m_close, "MsgDialog handlers registered");
    if (m_init && m_open) {
        set_platform_ui(nullptr);
        m_init(0,0,0,0,0,0); m_open(0,0,0,0,0,0);
        CHECK(m_status(0,0,0,0,0,0) == 3, "MsgDialog headless: Open auto-dismisses to FINISHED");
        m_close(0,0,0,0,0,0);
        set_platform_ui(&ui);
        m_init(0,0,0,0,0,0); m_open(0xAAAA,0,0,0,0,0);
        CHECK(ui.msg_opens == 1 && ui.msg_last_param == 0xAAAA, "MsgDialog backend: Open forwards the param");
        CHECK(m_status(0,0,0,0,0,0) == 2, "MsgDialog backend: status follows the frontend (RUNNING)");
        uint32_t mres[3] = {0,0,0};
        m_result((uint64_t)(uintptr_t)mres, 0,0,0,0,0);
        CHECK(ui.msg_results == 1 && mres[2] == 2, "MsgDialog backend: GetResult writes the frontend's buttonId");
        m_close(0,0,0,0,0,0);
        CHECK(ui.msg_closes == 1, "MsgDialog backend: Close routes to msgDialogClose");
    }

    // --- ErrorDialog: headless auto-dismiss vs backend delegation. ---
    HleFn e_init = Hle::lookup("I88KChlynSs"), e_open = Hle::lookup("M2ZF-ClLhgY"),
          e_status = Hle::lookup("t2FvHRXzgqk"), e_term = Hle::lookup("9XAxK2PMwk8");
    CHECK(e_init && e_open && e_status, "ErrorDialog handlers registered");
    if (e_init && e_open) {
        set_platform_ui(nullptr);
        e_init(0,0,0,0,0,0); e_open(0,0,0,0,0,0);
        CHECK(e_status(0,0,0,0,0,0) == 3, "ErrorDialog headless: Open auto-dismisses to FINISHED");
        set_platform_ui(&ui);
        e_init(0,0,0,0,0,0); e_open(0xBBBB,0,0,0,0,0);
        CHECK(ui.err_opens == 1 && ui.err_last_param == 0xBBBB, "ErrorDialog backend: Open forwards the param");
        CHECK(e_status(0,0,0,0,0,0) == 2, "ErrorDialog backend: status follows the frontend (RUNNING)");
        if (e_term) { e_term(0,0,0,0,0,0); CHECK(ui.err_closes == 1, "ErrorDialog backend: Term routes to errorDialogClose"); }
    }

    // --- SaveDataDialog: declined ownership keeps headless behavior; accepted ownership stays bound
    // to the exact backend that accepted Open, including progress, result, and Term. ---
    HleFn s_init = Hle::lookup("s9e3+YpRnzw"), s_open = Hle::lookup("4tPhsP6FpDI"),
          s_status = Hle::lookup("ERKzksauAJA"), s_result = Hle::lookup("yEiJ-qqr6Cg"),
          s_close = Hle::lookup("fH46Lag88XY"), s_term = Hle::lookup("YuH2FA7azqQ"),
          s_inc = Hle::lookup("V-uEeFKARJU"), s_set = Hle::lookup("hay1CfTmLyA");
    CHECK(s_init && s_open && s_status && s_result && s_close && s_term && s_inc && s_set,
          "SaveDataDialog handlers registered");
    if (s_init && s_open && s_status && s_result && s_close && s_term && s_inc && s_set) {
        uint8_t save_param[0x98] = {};
        *(uint32_t*)(save_param + 0x34) = 3 /*SYSTEM_MSG*/;
        *(uint64_t*)(save_param + 0x70) = 0x12345678;

        set_platform_ui(&ui);
        ui.save_accept = false;
        s_init(0,0,0,0,0,0);
        s_open((uint64_t)(uintptr_t)save_param,0,0,0,0,0);
        CHECK(ui.save_opens == 1 && s_status(0,0,0,0,0,0) == 3,
              "SaveDataDialog declined ownership falls back to headless FINISHED");
        s_close(0,0,0,0,0,0);
        CHECK(ui.save_closes == 0, "SaveDataDialog declined ownership is not closed through backend");

        ui.save_accept = true;
        s_init(0,0,0,0,0,0);
        s_open((uint64_t)(uintptr_t)save_param,0,0,0,0,0);
        CHECK(ui.save_opens == 2 && ui.save_last_param == (uint64_t)(uintptr_t)save_param &&
                  s_status(0,0,0,0,0,0) == 2,
              "SaveDataDialog accepted ownership forwards Open and backend RUNNING status");
        set_platform_ui(nullptr); // ownership must remain with the backend that accepted Open
        CHECK(s_status(0,0,0,0,0,0) == 2,
              "SaveDataDialog status remains bound to accepting backend after registry changes");
        s_inc(7, 4,0,0,0,0); s_set(7, 65,0,0,0,0);
        CHECK(ui.save_incs == 1 && ui.save_sets == 1 && ui.save_last_target == 7 &&
                  ui.save_last_value == 65,
              "SaveDataDialog progress callbacks route to accepting backend");
        uint8_t save_result[0x48] = {};
        s_result((uint64_t)(uintptr_t)save_result,0,0,0,0,0);
        CHECK(ui.save_results == 1 && ui.save_last_result == (uint64_t)(uintptr_t)save_result &&
                  *(uint32_t*)save_result == 4,
              "SaveDataDialog GetResult routes to accepting backend");
        s_term(0,0,0,0,0,0);
        CHECK(ui.save_closes == 1 && s_status(0,0,0,0,0,0) == 0,
              "SaveDataDialog Term closes accepting backend and returns to NONE");
    }
    set_platform_ui(nullptr);

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
