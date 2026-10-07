// test_hle_registered — guards that the HLE functions we rely on are actually registered.
// register_builtin_hle() binds handlers by NID; a typo'd name or a forgotten registration would
// silently leave an import as an unimplemented stub (returns 0) and regress the boot. This checks
// a representative set across every HLE module (libc, math, file, kernel, time, service, graphics).
//
// Split by library family rather than left as one 120-name loop: a single failure would otherwise
// stop the sweep and hide every name after it, and each family's registration lives in its own
// register_*() entry point, so that is also the unit a failure should point at.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace prosper;

namespace {
// Each ctest case is its own process, so every one of these starts from an empty registry.
void ensure_registered() {
    register_builtin_hle();
}

// "Is this NID registered?" is a THIRD question, distinct from both "give me a handler I may call
// through HleFn" and "give me a guest-ABI handler" -- and `lookup` used to answer all three. It no
// longer answers this one for the printf family, which are guest-ABI (#3272): `lookup` refuses
// those, so asking it about `snprintf` here reported a registered handler as missing. That is the
// whole reason the accessor was split; `Hle::registered` is the one that means existence.
void expect_registered(std::string_view name) {
    ensure_registered();
    EXPECT_TRUE(Hle::registered(nid_hash(std::string(name)))) << "not registered: " << name;
}
void expect_family_registered(const std::vector<std::string_view>& names) {
    for (const std::string_view name : names) expect_registered(name);
}
}   // namespace

TEST(HleRegistered, LibcCore) {
    expect_family_registered({"memcpy", "memset", "strlen", "malloc", "free", "snprintf", "bcmp",
                              "bsearch", "__error", "setjmp", "longjmp"});
}

TEST(HleRegistered, MathHostThunks) {
    expect_family_registered(
        {"cosf", "sinf", "sqrtf", "powf", "atan2f", "sincosf", "ldexp", "fmin", "tanh", "log1pf"});
}

TEST(HleRegistered, FileAndDirectory) {
    expect_family_registered({"open", "read", "stat", "sceKernelOpen", "sceKernelMkdir",
                              "sceKernelPread", "sceKernelStat"});
}

TEST(HleRegistered, KernelThreadsSyncAndExceptions) {
    expect_family_registered({"scePthreadCreate", "scePthreadMutexLock", "scePthreadCondWait",
                              "pthread_equal", "sceKernelInstallExceptionHandler",
                              "sceKernelRaiseException", "sceKernelIsStack",
                              "scePthreadGetschedparam"});
}

TEST(HleRegistered, PosixSpellingsMustAnswerIdenticallyToTheSceOnes) {
    // #2914
    expect_family_registered({"pthread_getschedparam", "pthread_setschedparam"});
}

TEST(HleRegistered, TimeAndEventQueues) {
    expect_family_registered({"sceKernelClockGettime", "sceKernelUsleep", "sceKernelCreateEqueue",
                              "sceKernelWaitEqueue"});
    expect_registered("getpid");
}

TEST(HleRegistered, ServicesAndDialogs) {
    expect_family_registered({"sceUserServiceGetInitialUser", "scePadOpen",
                              "sceMsgDialogUpdateStatus", "sceSystemServiceHideSplashScreen",
                              "sceHttpUriParse"});
}

TEST(HleRegistered, Graphics) {
    expect_family_registered(
        {"sceVideoOutOpen", "sceVideoOutSubmitFlip", "sceVideoOutGetFlipStatus"});
}

TEST(HleRegistered, Audio) {
    expect_family_registered({"sceAudioOutInit", "sceAudioOutOpen", "sceAudioOutOutput",
                              "sceAudioOutOutputs", "sceAudioOutSetVolume", "sceAudioOutClose",
                              "sceAudioOutGetPortState", "sceAudioInInit", "sceAudioInOpen",
                              "sceAudioInInput", "sceAudioInClose"});
}

// libSceAvPlayer. These live in their own translation unit and their own entry point
// (register_avplayer_hle, src/hle/video/avplayer.cpp) since #3735, so a split that forgot
// to call it would leave the whole library unregistered -- and the dispatcher's default of
// 0 reads as a SUCCESSFUL init, an active player and a completed video. The title would
// hang on a splash it believes is playing, with nothing in any log. That failure is exactly
// what this file exists for, and it is per-library now rather than per-file.
TEST(HleRegistered, LibSceAvPlayerHasItsOwnEntryPoint) {
    expect_family_registered(
        {"sceAvPlayerInit", "sceAvPlayerInitEx", "sceAvPlayerPostInit", "sceAvPlayerSetLogCallback",
         "sceAvPlayerAddSource", "sceAvPlayerAddSourceEx", "sceAvPlayerStart", "sceAvPlayerStartEx",
         "sceAvPlayerIsActive", "sceAvPlayerGetVideoData", "sceAvPlayerGetVideoDataEx",
         "sceAvPlayerGetAudioData", "sceAvPlayerStop", "sceAvPlayerClose"});
}

// libSceVideodec2 / libSceVdecsw, same reasoning, same entry point shape
// (register_videodec_hle, src/hle/video/videodec2.cpp). libScePsml is deliberately absent:
// its NIDs have no recovered names, so they are registered by raw NID and there is no name
// to hash here. Registering under a name the library does not have would be a worse guard
// than none -- it would pass while proving nothing about the real NID.
TEST(HleRegistered, LibSceVideodec2AndVdecsw) {
    expect_family_registered({"sceVideodec2CreateDecoder", "sceVideodec2DeleteDecoder",
                              "sceVideodec2Decode", "sceVideodec2Flush", "sceVideodec2Reset",
                              "sceVideodec2GetPictureInfo", "sceVdecswQueryComputeMemoryInfo",
                              "sceVdecswAllocateComputeQueue", "sceVdecswQueryDecoderMemoryInfo",
                              "sceVdecswCreateDecoder", "sceVdecswSetDecodeInput",
                              "sceVdecswSetDecodeOutput", "sceVdecswTrySyncDecodeInput",
                              "sceVdecswTrySyncDecodeOutput"});
}

// libSceIme + libSceImeDialog (register_ime_hle, src/hle/input/ime.cpp). sceImeUpdate is
// registered from inside an #ifdef _WIN32 / #else pair -- both arms bind the same NID to a
// different handler, so this asserts the NID is bound on whichever platform is building.
TEST(HleRegistered, LibSceImeAndImeDialog) {
    expect_family_registered({"sceImeKeyboardOpen", "sceImeKeyboardClose", "sceImeUpdate",
                              "sceImeKeyboardGetInfo", "sceImeKeyboardGetResourceId",
                              "sceImeDialogInit", "sceImeDialogGetStatus", "sceImeDialogGetResult",
                              "sceImeDialogTerm", "sceImeDialogAbort"});
}

// libSceSaveData + libSceSaveDataDialog (register_savedata_hle, src/hle/fs/savedata.cpp),
// which is where save_param.* and save_paths.* already lived.
TEST(HleRegistered, LibSceSaveDataAndDialog) {
    expect_family_registered(
        {"sceSaveDataInitialize3", "sceSaveDataMount", "sceSaveDataMount2", "sceSaveDataTerminate",
         "sceSaveDataCreateTransactionResource", "sceSaveDataDeleteTransactionResource",
         "sceSaveDataSetupSaveDataMemory2", "sceSaveDataGetSaveDataMemory2",
         "sceSaveDataDialogInitialize", "sceSaveDataDialogOpen", "sceSaveDataDialogUpdateStatus"});
}

// The NP / online family (register_np_hle, src/hle/np/np.cpp). The state-callback NIDs
// (sceNpCheckCallback, sceNpRegister/UnregisterStateCallbackA and the sceNetCtl*Callback
// set) bind on every host from src/hle/np/state_callbacks.cpp.
TEST(HleRegistered, NpAndOnline) {
    expect_family_registered(
        {"sceNpGetState", "sceNpGetNpReachabilityState", "sceNpCheckNpAvailability",
         "sceNpCheckCallback", "sceNpRegisterStateCallback", "sceNpRegisterStateCallbackA",
         "sceNpUnregisterStateCallbackA", "sceNetCtlRegisterCallback",
         "sceNetCtlUnregisterCallback", "sceNetCtlCheckCallback", "sceNetCtlGetInfo",
         "sceNetCtlGetResult", "sceNpTrophy2GetGameInfo", "sceNpTrophy2CreateContext"});
}

TEST(HleRegistered, SyncOnAddressFutexBindsUnderItsRawNid) {
    ensure_registered();
    // sync_on_address futex is registered by raw NID (no symbol name) -- check it directly.
    EXPECT_TRUE(Hle::registered("Hc4CaR6JBL0")) << "sceKernelWaitOnAddress raw NID";
}

TEST(HleRegistered, CtypeMbCurMaxReturnsTheValueNotAPointer) {
    // __ctype_get_mb_cur_max returns the VALUE of MB_CUR_MAX (1 in the "C" locale we
    // present), not a pointer to it -- guest code sizes buffers as MB_CUR_MAX*n (#141).
    ensure_registered();
    const HleFn fn = Hle::lookup(nid_hash("__ctype_get_mb_cur_max"));
    ASSERT_NE(fn, nullptr) << "not registered: __ctype_get_mb_cur_max";
    EXPECT_EQ(fn(0, 0, 0, 0, 0, 0), 1u) << "__ctype_get_mb_cur_max returned a value that is not 1";
}

TEST(HleRegistered, GetpctypeUsesTheDinkumwareMaskLayout) {
    // Sony's Dinkumware inline isspace uses (_Getpctype()[c] & 0x144), not the
    // incompatible MSVCRT bit layout. Pin representative C-locale masks and EOF/case slots.
    ensure_registered();
    const HleFn fn = Hle::lookup(nid_hash("_Getpctype"));
    ASSERT_NE(fn, nullptr) << "not registered: _Getpctype";
    const auto* table = reinterpret_cast<const short*>(fn(0, 0, 0, 0, 0, 0));
    ASSERT_NE(table, nullptr) << "_Getpctype returned no table";
    ASSERT_EQ(table[-1], 0) << "_Getpctype is missing its EOF slot";
    struct Expected {
        unsigned char ch;
        short mask;
    };
    constexpr Expected expected[] = {
        {0x00, 0x080}, {'\t', 0x4c0}, {'\n', 0x0c0}, {' ', 0x004}, {'!', 0x008},
        {'0', 0x021},  {'A', 0x003},  {'G', 0x002},  {'a', 0x011}, {'g', 0x010},
        {0x7f, 0x080}, {0x80, 0x000}, {0xff, 0x000},
    };
    for (const auto& item : expected) {
        EXPECT_EQ(table[item.ch], item.mask)
            << "_Getpctype[0x" << std::hex << static_cast<unsigned>(item.ch) << "] = 0x"
            << static_cast<unsigned>(short(table[item.ch])) << ", want 0x"
            << static_cast<unsigned>(item.mask);
    }
    constexpr short dinkumware_isspace = 0x144;   // _CN|_SP|_XS
    EXPECT_NE(table[' '] & dinkumware_isspace, 0) << "Dinkumware isspace must accept space";
    EXPECT_NE(table['\t'] & dinkumware_isspace, 0) << "Dinkumware isspace must accept tab";
    EXPECT_NE(table['\n'] & dinkumware_isspace, 0) << "Dinkumware isspace must accept newline";
    EXPECT_EQ(table['A'] & dinkumware_isspace, 0) << "Dinkumware isspace must reject a letter";
}

TEST(HleRegistered, GetptolowerKeepsTheCLocaleAndEofContract) {
    ensure_registered();
    const HleFn fn = Hle::lookup(nid_hash("_Getptolower"));
    ASSERT_NE(fn, nullptr) << "not registered: _Getptolower";
    const auto* table = reinterpret_cast<const short*>(fn(0, 0, 0, 0, 0, 0));
    ASSERT_NE(table, nullptr) << "_Getptolower returned no table";
    EXPECT_EQ(table[-1], -1) << "_Getptolower EOF slot";
    EXPECT_EQ(table['A'], static_cast<short>('a')) << "_Getptolower['A']";
    EXPECT_EQ(table['a'], static_cast<short>('a')) << "_Getptolower['a']";
    EXPECT_EQ(table[0xff], static_cast<short>(0xff)) << "_Getptolower[0xff]";
}

TEST(HleRegistered, GetptoupperKeepsTheCLocaleAndEofContract) {
    ensure_registered();
    const HleFn fn = Hle::lookup(nid_hash("_Getptoupper"));
    ASSERT_NE(fn, nullptr) << "not registered: _Getptoupper";
    const auto* table = reinterpret_cast<short*>(fn(0, 0, 0, 0, 0, 0));
    ASSERT_NE(table, nullptr) << "_Getptoupper returned no table";
    EXPECT_EQ(table[-1], -1) << "_Getptoupper EOF slot";
    EXPECT_EQ(table['a'], static_cast<short>('A')) << "_Getptoupper['a']";
    EXPECT_EQ(table['A'], static_cast<short>('A')) << "_Getptoupper['A']";
    EXPECT_EQ(table[0xff], static_cast<short>(0xff)) << "_Getptoupper[0xff]";
}

TEST(HleRegistered, GetpidIsNotTheKernelSpecialPidZero) {
    ensure_registered();
    const HleFn fn = Hle::lookup(nid_hash("getpid"));
    ASSERT_NE(fn, nullptr) << "not registered: getpid";
    EXPECT_NE(fn(0, 0, 0, 0, 0, 0), 0u) << "getpid returned kernel-special pid 0";
}

TEST(HleRegistered, GetthreadidAssignsStableDistinctPositiveGuestIds) {
    ensure_registered();
    const HleFn fn = Hle::lookup(nid_hash("scePthreadGetthreadid"));
    ASSERT_NE(fn, nullptr) << "not registered: scePthreadGetthreadid";
    const uint64_t main_id = fn(0, 0, 0, 0, 0, 0);
    const uint64_t main_id_again = fn(0, 0, 0, 0, 0, 0);
    std::array<std::atomic<uint64_t>, 2> worker_id{};
    std::array<std::atomic<uint64_t>, 2> worker_id_again{};
    std::array<std::thread, 2> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, i] {
            worker_id[i].store(fn(0, 0, 0, 0, 0, 0), std::memory_order_relaxed);
            worker_id_again[i].store(fn(0, 0, 0, 0, 0, 0), std::memory_order_relaxed);
        });
    }
    for (auto& worker : workers) worker.join();

    const auto positive = [](uint64_t id) {
        return id != 0 && id <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    };
    EXPECT_TRUE(positive(main_id)) << "main-thread guest thread id is not a positive int32";
    EXPECT_EQ(main_id, main_id_again) << "main-thread guest thread id is not stable";

    const uint64_t worker0 = worker_id[0].load(std::memory_order_relaxed);
    const uint64_t worker1 = worker_id[1].load(std::memory_order_relaxed);
    EXPECT_TRUE(positive(worker0)) << "worker 0 guest thread id is not a positive int32";
    EXPECT_TRUE(positive(worker1)) << "worker 1 guest thread id is not a positive int32";
    EXPECT_EQ(worker0, worker_id_again[0].load(std::memory_order_relaxed))
        << "worker 0 guest thread id is not stable";
    EXPECT_EQ(worker1, worker_id_again[1].load(std::memory_order_relaxed))
        << "worker 1 guest thread id is not stable";
    EXPECT_NE(worker0, main_id) << "worker 0 shares the main thread's guest id";
    EXPECT_NE(worker1, main_id) << "worker 1 shares the main thread's guest id";
    EXPECT_NE(worker0, worker1) << "the two workers share a guest id";
}