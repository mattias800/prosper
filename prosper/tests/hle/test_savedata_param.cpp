// test_savedata_param (#2786) — a save's parameter block must be STORED, and read back as what was
// stored.
//
// `sceSaveDataSetParam` (NID `85zul--eGXs`) and `sceSaveDataGetParam` (`XgvSuIdnMlw`) were both
// unregistered, so both fell to the dispatcher's `return 0` — and 0 is SCE_OK for this contract. The
// guest was told its metadata had been written when nothing was written, and a read-back left the
// caller's buffer untouched while still reporting success. Observed on *Sonic Frontiers* as the only
// unimplemented NID on a route reaching `GameModeStage`.
//
// Every arm goes through the REAL NIDs, because "is it registered" is half the defect: a test that
// called an internal helper would pass against a build where the NIDs are still absent and the guest
// still gets the stub.
//
// The arms, and what each one kills:
//
//  1. round trip     — set title/sub-title/detail/user param, read each back. Fails on the stub,
//                      which wrote nothing into the read buffer at all.
//  2. MUTATION       — a DIFFERENT param block must read back differently. A stub that returned a
//                      fixed plausible block would pass arm 1 and only fails here.
//  3. field independence — setting one field must not erase the others (the API is called once per
//                      field), and TYPE_ALL must agree with the per-field reads in both directions.
//  4. persistence    — unmount, remount, read: the block must come off DISK. An implementation that
//                      kept it in a process-global map passes 1-3 and fails here, and so does the
//                      title whose save browser is empty after a restart.
//  5. isolation      — two saves keep two blocks. A single global record passes 1-4.
//  6. refusal polarity — no mount, wrong mount point, unknown type, null buffer and an undersized
//                      buffer must all be REFUSED. This is the arm that is precisely inverted on the
//                      unfixed build, where every one of them returns SCE_OK.
//  7. container      — the bytes on disk are a real param.sfo at the path prosper's own
//                      `savedata0_dir_mtime()` already looks for, and it decodes back to what was
//                      written. A truncated file must read as ABSENT, not as an empty block.
//  8. ABI offsets    — the TYPE_ALL layout constants, pinned so a future live capture that
//                      contradicts one reddens a named arm instead of silently disagreeing.
#include "hle/dispatch/dispatch.hpp"
#include "hle/fs/save_param.hpp"
#include "hle/fs/save_paths.hpp"
#include "hle/service/hle_addcontent.hpp"
#include "fixtures/test_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace prosper;
namespace fs = std::filesystem;

static int fails = 0;
static int checks = 0;
#define CHECK(c, m) do { ++checks; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// The ABI this implements. Spelled out here independently of the implementation's own constants so
// the test states the contract rather than restating whatever the code happens to do.
constexpr uint32_t TYPE_ALL = 0, TYPE_TITLE = 1, TYPE_SUB_TITLE = 2, TYPE_DETAIL = 3,
                   TYPE_USER_PARAM = 4, TYPE_MTIME = 5;
constexpr size_t TITLE_MAX = 128, SUBTITLE_MAX = 128, DETAIL_MAX = 1024, ALL_SIZE = 1328;
constexpr size_t OFF_TITLE = 0, OFF_SUBTITLE = 128, OFF_DETAIL = 256, OFF_USER_PARAM = 1280,
                 OFF_MTIME = 1288;

// SceSaveDataMountPoint { char data[16] }.
struct MountPoint { char data[16]; };
MountPoint mount_point(const char* path) {
    MountPoint mp{};
    snprintf(mp.data, sizeof mp.data, "%s", path);
    return mp;
}

std::string make_app0(const fs::path& base, const char* name, const std::string& title_id) {
    const fs::path root = base / name;
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "sce_sys", ec);
    std::ofstream p(root / "sce_sys" / "param.json", std::ios::binary);
    p << "{\"titleId\":\"" << title_id << "\"}";
    p.close();
    return root.string();
}

HleFn g_set = nullptr, g_get = nullptr;

uint64_t set_param(const MountPoint& mp, uint32_t type, const void* buf, uint64_t size) {
    return g_set((uint64_t)(uintptr_t)&mp, type, (uint64_t)(uintptr_t)buf, size, 0, 0);
}
uint64_t get_param(const MountPoint& mp, uint32_t type, void* buf, uint64_t size,
                   uint64_t* got = nullptr) {
    return g_get((uint64_t)(uintptr_t)&mp, type, (uint64_t)(uintptr_t)buf, size,
                 (uint64_t)(uintptr_t)got, 0);
}

// Set one text field through its own param type.
uint64_t set_text(const MountPoint& mp, uint32_t type, size_t capacity, const std::string& value) {
    std::vector<char> field(capacity, 0);
    memcpy(field.data(), value.data(), value.size() < capacity ? value.size() : capacity - 1);
    return set_param(mp, type, field.data(), capacity);
}
// Read a fixed-width field out of a TYPE_ALL block WITHOUT trusting it to be terminated. The
// obvious std::string(ptr + offset) walks off the end when the call left the buffer untouched,
// which is exactly the case these arms exist to catch: an arm that segfaults on the broken build
// instead of failing cleanly cannot be read.
std::string field_text(const std::vector<uint8_t>& block, size_t offset, size_t capacity) {
    size_t n = 0;
    while (n < capacity && offset + n < block.size() && block[offset + n]) n++;
    return std::string((const char*)block.data() + offset, n);
}

std::string get_text(const MountPoint& mp, uint32_t type, size_t capacity, uint64_t& rc) {
    std::vector<char> field(capacity, 0x7f);   // poisoned: an untouched buffer must not read empty
    uint64_t got = 0;
    rc = get_param(mp, type, field.data(), capacity, &got);
    if (rc != 0) return {};
    if (got != capacity) return "<wrong gotSize>";
    size_t n = 0;
    while (n < capacity && field[n]) n++;
    return std::string(field.data(), n);
}

}   // namespace

int main() {
    printf("== test_savedata_param ==\n");
    register_builtin_hle();

    g_set = Hle::lookup("85zul--eGXs");
    g_get = Hle::lookup("XgvSuIdnMlw");
    CHECK(g_set != nullptr, "sceSaveDataSetParam is registered");
    CHECK(g_get != nullptr, "sceSaveDataGetParam is registered");
    if (!g_set || !g_get) { printf("== FAIL: %d == (%d assertions executed)\n", fails, checks); return 1; }

    const fs::path scratch = prosper_test::test_scratch_dir() / "savedata-param";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(scratch, ec);
    const fs::path save_root = scratch / "save0";
    fs::create_directories(save_root, ec);
    set_env("PROSPER_SAVE0", save_root.string().c_str());
    set_app0_root(make_app0(scratch, "title", "PPSA00042"));

    const MountPoint mp = mount_point("/savedata0");

    // ------------------------------------------------------ arm 6a: refusal with nothing mounted
    // Runs FIRST, before any mount, because it is the one that is exactly inverted on the unfixed
    // build: there every call below returns 0.
    savedata0_umount();
    {
        char buf[TITLE_MAX] = {};
        CHECK(set_param(mp, TYPE_TITLE, buf, TITLE_MAX) != 0,
              "SetParam with no save mounted is refused");
        CHECK(get_param(mp, TYPE_TITLE, buf, TITLE_MAX) != 0,
              "GetParam with no save mounted is refused");
    }

    CHECK(savedata0_mount("SlotA", SaveDataMountPolicy::OpenOrCreate) ==
              SaveDataMountOutcome::Created, "a save directory is mounted");
    const fs::path slot_a = fs::path(savedata0_dir()) / "SlotA";

    // ------------------------------------------------------------------- arm 6b: argument refusal
    {
        char buf[TITLE_MAX] = {};
        const MountPoint wrong = mount_point("/savedata9");
        CHECK(set_param(wrong, TYPE_TITLE, buf, TITLE_MAX) != 0,
              "SetParam on a mount point this build does not serve is refused");
        CHECK(get_param(wrong, TYPE_TITLE, buf, TITLE_MAX) != 0,
              "GetParam on a mount point this build does not serve is refused");
        CHECK(set_param(mp, 99, buf, TITLE_MAX) != 0, "SetParam with an unknown param type is refused");
        CHECK(get_param(mp, 99, buf, TITLE_MAX) != 0, "GetParam with an unknown param type is refused");
        CHECK(set_param(mp, TYPE_TITLE, nullptr, TITLE_MAX) != 0, "SetParam with a null buffer is refused");
        CHECK(get_param(mp, TYPE_TITLE, nullptr, TITLE_MAX) != 0, "GetParam with a null buffer is refused");
        CHECK(set_param(mp, TYPE_TITLE, buf, TITLE_MAX - 1) != 0,
              "SetParam with a buffer smaller than the field is refused");
        CHECK(get_param(mp, TYPE_TITLE, buf, TITLE_MAX - 1) != 0,
              "GetParam with a buffer smaller than the field is refused");
    }

    // ------------------------------------------------------------------------ arm 1: round trip
    const std::string title_a = "Sonic Frontiers";
    const std::string sub_a = "Kronos Island";
    const std::string detail_a = "Chapter 1 - 42 rings, 3 memory tokens";
    const uint32_t user_a = 0xC0FFEE01u;
    CHECK(set_text(mp, TYPE_TITLE, TITLE_MAX, title_a) == 0, "SetParam TITLE succeeds");
    CHECK(set_text(mp, TYPE_SUB_TITLE, SUBTITLE_MAX, sub_a) == 0, "SetParam SUB_TITLE succeeds");
    CHECK(set_text(mp, TYPE_DETAIL, DETAIL_MAX, detail_a) == 0, "SetParam DETAIL succeeds");
    CHECK(set_param(mp, TYPE_USER_PARAM, &user_a, sizeof user_a) == 0, "SetParam USER_PARAM succeeds");

    uint64_t rc = 0;
    CHECK(get_text(mp, TYPE_TITLE, TITLE_MAX, rc) == title_a && rc == 0,
          "GetParam TITLE returns what was set");
    CHECK(get_text(mp, TYPE_SUB_TITLE, SUBTITLE_MAX, rc) == sub_a && rc == 0,
          "GetParam SUB_TITLE returns what was set");
    CHECK(get_text(mp, TYPE_DETAIL, DETAIL_MAX, rc) == detail_a && rc == 0,
          "GetParam DETAIL returns what was set");
    {
        uint32_t read = 0; uint64_t got = 0;
        CHECK(get_param(mp, TYPE_USER_PARAM, &read, sizeof read, &got) == 0 && read == user_a &&
                  got == sizeof(uint32_t),
              "GetParam USER_PARAM returns what was set, with its byte count");
    }

    // ---------------------------------------------------------------------------- arm 2: MUTATION
    // The arm that separates "stored and read back" from "returns a fixed plausible block". Without
    // it, an implementation that answered every GetParam with the first title it ever saw — or with
    // a constant — would pass everything above.
    const std::string title_b = "Sonic Frontiers (second file)";
    const uint32_t user_b = 0x00BADA55u;
    CHECK(set_text(mp, TYPE_TITLE, TITLE_MAX, title_b) == 0, "a DIFFERENT title is set");
    CHECK(set_param(mp, TYPE_USER_PARAM, &user_b, sizeof user_b) == 0, "a DIFFERENT user param is set");
    CHECK(get_text(mp, TYPE_TITLE, TITLE_MAX, rc) == title_b,
          "GetParam TITLE follows the change (not a fixed block)");
    {
        uint32_t read = 0;
        CHECK(get_param(mp, TYPE_USER_PARAM, &read, sizeof read) == 0 && read == user_b,
              "GetParam USER_PARAM follows the change");
    }

    // -------------------------------------------------------------- arm 3: fields are independent
    CHECK(get_text(mp, TYPE_SUB_TITLE, SUBTITLE_MAX, rc) == sub_a,
          "setting the title did not erase the sub-title");
    CHECK(get_text(mp, TYPE_DETAIL, DETAIL_MAX, rc) == detail_a,
          "setting the title did not erase the detail");
    {
        std::vector<uint8_t> all(ALL_SIZE, 0x7f);
        uint64_t got = 0;
        CHECK(get_param(mp, TYPE_ALL, all.data(), all.size(), &got) == 0 && got == ALL_SIZE,
              "GetParam ALL fills the whole 1328-byte block");
        CHECK(field_text(all, OFF_TITLE, TITLE_MAX) == title_b &&
                  field_text(all, OFF_SUBTITLE, SUBTITLE_MAX) == sub_a &&
                  field_text(all, OFF_DETAIL, DETAIL_MAX) == detail_a,
              "ALL agrees with the per-field reads");
        uint32_t user = 0; int64_t mtime = 0;
        memcpy(&user, all.data() + OFF_USER_PARAM, sizeof user);
        memcpy(&mtime, all.data() + OFF_MTIME, sizeof mtime);
        CHECK(user == user_b, "ALL carries the user param at its ABI offset");
        // Bounded on both sides, because the read buffer is poisoned with 0x7f: a call that wrote
        // nothing leaves a huge positive value there, so "> 0" alone would pass on the broken build.
        const int64_t now = (int64_t)time(nullptr);
        CHECK(mtime > 1600000000 && mtime <= now + 86400,
              "ALL carries a plausible modification time, not a zero and not an untouched buffer");

        // ...and the other direction: a whole block written at once must be readable per field.
        std::vector<uint8_t> write(ALL_SIZE, 0);
        const std::string title_c = "Written as one block";
        const std::string sub_c = "SUB-C";
        const std::string detail_c = "DETAIL-C";
        const uint32_t user_c = 0x11223344u;
        const int64_t mtime_c = 1700000000;
        memcpy(write.data() + OFF_TITLE, title_c.data(), title_c.size());
        memcpy(write.data() + OFF_SUBTITLE, sub_c.data(), sub_c.size());
        memcpy(write.data() + OFF_DETAIL, detail_c.data(), detail_c.size());
        memcpy(write.data() + OFF_USER_PARAM, &user_c, sizeof user_c);
        memcpy(write.data() + OFF_MTIME, &mtime_c, sizeof mtime_c);
        CHECK(set_param(mp, TYPE_ALL, write.data(), write.size()) == 0, "SetParam ALL succeeds");
        CHECK(get_text(mp, TYPE_TITLE, TITLE_MAX, rc) == title_c &&
                  get_text(mp, TYPE_SUB_TITLE, SUBTITLE_MAX, rc) == sub_c &&
                  get_text(mp, TYPE_DETAIL, DETAIL_MAX, rc) == detail_c,
              "every text field of a block written with ALL reads back per field");
        int64_t mtime_read = 0;
        CHECK(get_param(mp, TYPE_MTIME, &mtime_read, sizeof mtime_read) == 0 &&
                  mtime_read == mtime_c,
              "an mtime the guest supplied is the one that comes back");
    }

    // ------------------------------------------------------------------------- arm 4: persistence
    // The block must survive the mount, because the browser that shows it runs after a restart.
    CHECK(savedata0_umount(), "the save is unmounted");
    CHECK(savedata0_mount("SlotA", SaveDataMountPolicy::Open) == SaveDataMountOutcome::Opened,
          "the save is mounted again");
    CHECK(get_text(mp, TYPE_TITLE, TITLE_MAX, rc) == "Written as one block",
          "the parameter block came back from DISK across an unmount/remount");

    // --------------------------------------------------------------------------- arm 5: isolation
    CHECK(savedata0_mount("SlotB", SaveDataMountPolicy::OpenOrCreate) ==
              SaveDataMountOutcome::Created, "a second save is created and mounted");
    CHECK(get_text(mp, TYPE_TITLE, TITLE_MAX, rc).empty() && rc == 0,
          "a save with no parameter block reads back empty, and says so honestly");
    CHECK(set_text(mp, TYPE_TITLE, TITLE_MAX, "Slot B title") == 0, "the second save gets its own title");
    CHECK(get_text(mp, TYPE_TITLE, TITLE_MAX, rc) == "Slot B title", "which reads back");
    CHECK(savedata0_mount("SlotA", SaveDataMountPolicy::Open) == SaveDataMountOutcome::Opened,
          "back to the first save");
    CHECK(get_text(mp, TYPE_TITLE, TITLE_MAX, rc) == "Written as one block",
          "the first save's block is untouched by the second save's");

    // --------------------------------------------------------------------------- arm 7: container
    const fs::path param_file = fs::path(save_param_path(slot_a.string()));
    CHECK(fs::exists(param_file),
          "the block is stored at sce_sys/param.sfo inside the save, where savedata0_dir_mtime looks");
    {
        int64_t modified = 0;
        CHECK(savedata0_dir_mtime("SlotA", modified) && modified > 0,
              "savedata0_dir_mtime now finds a real param.sfo for the save dialog's ordering");
    }
    {
        std::ifstream in(param_file, std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                         std::istreambuf_iterator<char>());
        in.close();
        CHECK(bytes.size() > 20 && bytes[0] == 0x00 && bytes[1] == 0x50 && bytes[2] == 0x53 &&
                  bytes[3] == 0x46,
              "the file really is an SFO container");
        SaveDataParam decoded;
        CHECK(decode_param_sfo(bytes.data(), bytes.size(), decoded) &&
                  decoded.title == "Written as one block" && decoded.sub_title == "SUB-C" &&
                  decoded.detail == "DETAIL-C" && decoded.user_param == 0x11223344u,
              "and it decodes back to exactly what was set");

    }
    {
        // Damaged input must look ABSENT rather than like a stored-but-empty block: the two mean
        // different things to a title, and a partially applied parse is the worse of them. Built
        // from a freshly encoded block rather than from the file on disk, so this arm still runs --
        // and still means something -- on a build where nothing was written to disk at all.
        SaveDataParam source;
        source.title = "T";
        source.sub_title = "S";
        source.detail = "D";
        source.user_param = 7;
        const std::vector<uint8_t> good = encode_param_sfo(source, "PPSA00042", "SlotA");
        for (size_t truncated : {size_t{0}, size_t{4}, size_t{19}, good.size() / 2}) {
            SaveDataParam poisoned;
            poisoned.title = "untouched";
            const bool ok = decode_param_sfo(good.data(), truncated, poisoned);
            CHECK(!ok && poisoned.title == "untouched",
                  "a truncated param.sfo is rejected without partially overwriting the caller");
        }
        std::vector<uint8_t> wrong_magic = good;
        wrong_magic[0] = 0xff;
        SaveDataParam ignored;
        CHECK(!decode_param_sfo(wrong_magic.data(), wrong_magic.size(), ignored),
              "a file that is not an SFO is rejected");
    }
    {
        // Round-trip the codec on its own, over values that exercise the maxima and a full-width
        // field with no room for a terminator.
        SaveDataParam original;
        original.title = std::string(TITLE_MAX - 1, 'T');
        original.sub_title = std::string(SUBTITLE_MAX - 1, 'S');
        original.detail = std::string(DETAIL_MAX - 1, 'D');
        original.user_param = 0xFFFFFFFFu;
        const std::vector<uint8_t> encoded = encode_param_sfo(original, "PPSA00042", "SlotA");
        SaveDataParam decoded;
        CHECK(decode_param_sfo(encoded.data(), encoded.size(), decoded) &&
                  decoded.title == original.title && decoded.sub_title == original.sub_title &&
                  decoded.detail == original.detail && decoded.user_param == original.user_param,
              "the codec round-trips maximum-length fields exactly");
    }

    // ---------------------------------------------------------- arm 8: the ABI layout constants
    // Pinned here so the published PS4-inherited offsets this implementation reads are stated in one
    // falsifiable place. A live PS5 capture that contradicts one reddens this arm.
    CHECK(OFF_TITLE == 0 && OFF_SUBTITLE == 128 && OFF_DETAIL == 256 && OFF_USER_PARAM == 1280 &&
              OFF_MTIME == 1288 && ALL_SIZE == 1328,
          "SceSaveDataParam: title[128] subTitle[128] detail[1024] userParam mtime, 1328 bytes");
    {
        // An over-long string must be clamped at the field, never truncated into a neighbour's
        // bytes: TYPE_ALL is a packed struct, so an off-by-one here corrupts the next field.
        const std::string overlong(TITLE_MAX + 64, 'X');
        std::vector<char> field(TITLE_MAX + 64, 0);
        memcpy(field.data(), overlong.data(), overlong.size());
        CHECK(set_param(mp, TYPE_TITLE, field.data(), TITLE_MAX) == 0,
              "an over-long title is accepted at the field's size");
        const std::string read = get_text(mp, TYPE_TITLE, TITLE_MAX, rc);
        CHECK(read.size() == TITLE_MAX - 1 && read == std::string(TITLE_MAX - 1, 'X'),
              "and comes back clamped to the field, terminator included");
        CHECK(get_text(mp, TYPE_SUB_TITLE, SUBTITLE_MAX, rc) == "SUB-C",
              "without spilling into the next field");
    }

    savedata0_umount();
    if (fails) { printf("== FAIL: %d == (%d assertions executed)\n", fails, checks); return 1; }
    printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}
