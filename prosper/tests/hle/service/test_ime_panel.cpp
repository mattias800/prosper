// test_ime_panel — the libSceIme on-screen text-entry session: open/close, text, caret,
// candidates, panel geometry and keyboard mode.
//
// These eleven exports were unregistered, so the dispatcher answered `0`: a session nobody
// opened, text nobody stored, a caret nobody kept. Every TEST drives the real NIDs: closed
// sessions refuse, double-opens report busy, oversized text is refused before copying, and the
// panel answers "none" (0x0) rather than leaving the caller's size fields untouched. A no-op
// acknowledgement would succeed in every error arm below.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }

// The file's established IME error values.
static constexpr uint64_t kBusy = 0x80BC0001ull;
static constexpr uint64_t kNotOpened = 0x80BC0002ull;
static constexpr uint64_t kInvalidParam = 0x80BC0030ull;
static constexpr uint64_t kInvalidAddr = 0x80BC0031ull;

static uint64_t call_nid(const char* nid, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                         uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
    HleFn fn = Hle::lookup(nid_hash(nid));
    EXPECT_NE(fn, nullptr) << nid << " is not registered";
    if (!fn) return ~0ull;
    return fn(a0, a1, a2, a3, a4, a5);
}

TEST(ImePanel, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceImeParamInit", "sceImeOpen", "sceImeClose", "sceImeSetText", "sceImeSetCaret",
        "sceImeSetTextGeometry", "sceImeSetCandidateIndex", "sceImeConfirmCandidate",
        "sceImeGetPanelSize", "sceImeDisableController", "sceImeKeyboardSetMode",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 11, "the 11 session exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(ImePanel, NidsResolveToStubValues) {
    // All eleven NIDs reproduce the stub table verbatim.
    EXPECT_EQ(nid_hash("sceImeParamInit"), "WmYDzdC4EHI");
    EXPECT_EQ(nid_hash("sceImeOpen"), "RPydv-Jr1bc");
    EXPECT_EQ(nid_hash("sceImeClose"), "TmVP8LzcFcY");
    EXPECT_EQ(nid_hash("sceImeSetText"), "ieCNrVrzKd4");
    EXPECT_EQ(nid_hash("sceImeSetCaret"), "WLxUN2WMim8");
    EXPECT_EQ(nid_hash("sceImeSetTextGeometry"), "TXYHFRuL8UY");
    EXPECT_EQ(nid_hash("sceImeSetCandidateIndex"), "TQaogSaqkEk");
    EXPECT_EQ(nid_hash("sceImeConfirmCandidate"), "tKLmVIUkpyM");
    EXPECT_EQ(nid_hash("sceImeGetPanelSize"), "ziPDcIjO0Vk");
    EXPECT_EQ(nid_hash("sceImeDisableController"), "E+f1n8e8DAw");
    EXPECT_EQ(nid_hash("sceImeKeyboardSetMode"), "ua+13Hk9kKs");
    EXPECT_NE(nid_hash("sceImeOpen"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(ImePanel, ParamInitZeroesAndInvalidatesUser) {
    register_builtin_hle();
    // Void function: a null pointer is a silent no-op, matching the reference shape.
    EXPECT_EQ(call_nid("sceImeParamInit", 0), 0u);
    uint8_t param[88];
    std::memset(param, 0xAB, sizeof(param));
    EXPECT_EQ(call_nid("sceImeParamInit", addr(param)), 0u);
    int32_t user = 0;
    std::memcpy(&user, param, sizeof(user));
    EXPECT_EQ(user, -1) << "ParamInit marks the user invalid";
    for (size_t i = sizeof(user); i < sizeof(param); ++i) {
        EXPECT_EQ(param[i], 0u) << "ParamInit zeroes the block at byte " << i;
    }
}

TEST(ImePanel, OpenCloseLifecycle) {
    register_builtin_hle();
    EXPECT_EQ(call_nid("sceImeClose"), kNotOpened) << "close with no session fails";
    EXPECT_EQ(call_nid("sceImeOpen", 0, 0), kInvalidAddr) << "open without a param fails";
    uint8_t param[88]{};
    EXPECT_EQ(call_nid("sceImeOpen", addr(param), 0), 0u) << "open starts a session";
    EXPECT_EQ(call_nid("sceImeOpen", addr(param), 0), kBusy) << "double open reports busy";
    EXPECT_EQ(call_nid("sceImeClose"), 0u) << "close ends the session";
    EXPECT_EQ(call_nid("sceImeClose"), kNotOpened) << "close frees exactly once";
}

TEST(ImePanel, TextAndCaretValidateAgainstTheSession) {
    register_builtin_hle();
    EXPECT_EQ(call_nid("sceImeClose"), kNotOpened);  // start closed whatever ran before
    const char16_t text[] = u"hi";
    EXPECT_EQ(call_nid("sceImeSetText", addr(text), 2), kNotOpened)
        << "set-text on a closed session fails";
    EXPECT_EQ(call_nid("sceImeSetCaret", 0), kNotOpened) << "caret on a closed session fails";

    uint8_t param[88]{};
    ASSERT_EQ(call_nid("sceImeOpen", addr(param), 0), 0u);
    EXPECT_EQ(call_nid("sceImeSetText", 0, 2), kInvalidAddr) << "null text is refused";
    EXPECT_EQ(call_nid("sceImeSetText", addr(text), 4097), kInvalidParam)
        << "oversized text is refused before copying";
    EXPECT_EQ(call_nid("sceImeSetText", addr(text), 2), 0u) << "short text is accepted";

    uint8_t caret[16]{};
    uint32_t bad_index = 5;  // past the 2 units just stored
    std::memcpy(caret + 12, &bad_index, sizeof(bad_index));
    EXPECT_EQ(call_nid("sceImeSetCaret", addr(caret)), kInvalidParam)
        << "a caret past the text is refused";
    uint32_t good_index = 1;
    std::memcpy(caret + 12, &good_index, sizeof(good_index));
    EXPECT_EQ(call_nid("sceImeSetCaret", addr(caret)), 0u) << "a caret inside the text is kept";
    EXPECT_EQ(call_nid("sceImeSetCaret", 0), kInvalidAddr) << "null caret is refused";
    EXPECT_EQ(call_nid("sceImeClose"), 0u);
    EXPECT_EQ(call_nid("sceImeSetText", addr(text), 2), kNotOpened)
        << "the closed session keeps nothing settable";
}

TEST(ImePanel, HintsCandidatesPanelAndMode) {
    register_builtin_hle();
    EXPECT_EQ(call_nid("sceImeDisableController"), 0u) << "disable acknowledges";
    EXPECT_EQ(call_nid("sceImeSetCandidateIndex", 3), kNotOpened)
        << "candidate index on a closed session fails";
    EXPECT_EQ(call_nid("sceImeConfirmCandidate", 3), kNotOpened)
        << "candidate confirm on a closed session fails";

    uint8_t param[88]{};
    ASSERT_EQ(call_nid("sceImeOpen", addr(param), 0), 0u);
    EXPECT_EQ(call_nid("sceImeSetCandidateIndex", 3), 0u);
    EXPECT_EQ(call_nid("sceImeConfirmCandidate", 3), 0u);
    EXPECT_EQ(call_nid("sceImeSetTextGeometry", 0, 0), kInvalidAddr)
        << "null geometry is refused";
    uint8_t geometry[32]{};
    EXPECT_EQ(call_nid("sceImeSetTextGeometry", 0, addr(geometry)), 0u);

    uint8_t panel[16];
    std::memset(panel, 0xAB, sizeof(panel));
    EXPECT_EQ(call_nid("sceImeGetPanelSize", 0, addr(panel), addr(panel) + 4), kInvalidAddr)
        << "null param is refused";
    uint32_t w = 0xDEADu, h = 0xDEADu;
    EXPECT_EQ(call_nid("sceImeGetPanelSize", addr(param), addr(&w), addr(&h)), 0u);
    EXPECT_EQ(w, 0u) << "no panel exists, so the width is zero, not residue";
    EXPECT_EQ(h, 0u) << "no panel exists, so the height is zero, not residue";
    EXPECT_EQ(call_nid("sceImeClose"), 0u);

    // Keyboard mode follows the keyboard pump's own open state, not the panel session.
    EXPECT_EQ(call_nid("sceImeKeyboardSetMode", 1, 0), kNotOpened)
        << "mode with no open keyboard fails";
    EXPECT_EQ(call_nid("sceImeKeyboardOpen", 1, 0), 0u);
    EXPECT_EQ(call_nid("sceImeKeyboardSetMode", 1, 0), 0u) << "mode on the open keyboard succeeds";
    EXPECT_EQ(call_nid("sceImeKeyboardSetMode", 2, 0), kNotOpened)
        << "mode for another user fails";
    EXPECT_EQ(call_nid("sceImeKeyboardClose", 1), 0u);
}
