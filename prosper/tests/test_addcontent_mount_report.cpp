// test_addcontent_mount_report — every declared add-content gets ITS OWN mount point, and the
// mount diagnostic reports exactly what the guest was handed (#1993).
//
// Why this exists, and why the property is load-bearing rather than cosmetic.
//
// Sonic Origins Plus (PPSA05325) probes `/app0/dlc3/sound/Foo.awb` before falling back to the base
// content root. The string `/app0/dlc3` appears NOWHERE in its eboot, so the only way it can reach
// the guest is through sceAppContentAddcontMount. Two readings of that probe were live at once:
// the guest deliberately searching a DLC overlay first, or a single engine content-root variable
// that a mount had CLOBBERED. They call for opposite fixes, and the second would be prosper's bug.
//
// What separated them was the mount ORDER: the title mounts DLC04, DLC01, DLC03, DLC02 — so the
// probed `/app0/dlc3` is neither the first mount nor the last, and no single overwritten slot can
// produce it. That argument is only as good as the guarantee that each label really does get its
// own declared mount point. If prosper ever regressed to answering with, say, the first mountable
// entry regardless of label, every mount would report `/app0/dlc1`, the ordering evidence would be
// meaningless, and the falsification recorded in docs/GRIS_SONIC_COBRA_BRINGUP.md would quietly
// become wrong with nothing failing. test_addcontent_unmount.cpp cannot see this: it declares a
// single entry, so label-keyed and label-ignoring lookups are indistinguishable there.
//
// The second subject is the diagnostic itself. An instrument that misreports is worse than none —
// this project keeps a list of phantom defects that came from the apparatus rather than the
// subject — so the log line is asserted against the bytes the guest actually received, not merely
// asserted to exist.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/hle_addcontent.hpp"
#include "../src/hle/nid.hpp"
#include "test_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace prosper;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {
constexpr const char* kAddcontMount   = "VANhIWcqYak";  // sceAppContentAddcontMount
constexpr const char* kAddcontUnmount = "3rHWaV-1KC4";  // sceAppContentAddcontUnmount
constexpr uint64_t kErrNotFound = 0x80D90005ull;

int file_descriptor(FILE* file) {
#ifdef _WIN32
    return _fileno(file);
#else
    return fileno(file);
#endif
}
int duplicate_descriptor(int fd) {
#ifdef _WIN32
    return _dup(fd);
#else
    return dup(fd);
#endif
}
int replace_descriptor(int from, int to) {
#ifdef _WIN32
    return _dup2(from, to);
#else
    return dup2(from, to);
#endif
}
void close_descriptor(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

template <typename Fn>
std::string capture_stderr(Fn&& fn) {
    FILE* capture = std::tmpfile();
    if (!capture) return {};
    std::fflush(stderr);
    const int saved = duplicate_descriptor(file_descriptor(stderr));
    if (saved < 0 || replace_descriptor(file_descriptor(capture), file_descriptor(stderr)) < 0) {
        if (saved >= 0) close_descriptor(saved);
        std::fclose(capture);
        return {};
    }
    fn();
    std::fflush(stderr);
    replace_descriptor(saved, file_descriptor(stderr));
    close_descriptor(saved);
    std::rewind(capture);
    std::string out;
    char buffer[512];
    while (size_t n = std::fread(buffer, 1, sizeof buffer, capture)) out.append(buffer, n);
    std::fclose(capture);
    return out;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// A 20-byte SceNpUnifiedEntitlementLabel holding a 16-character label.
struct Label {
    char bytes[20];
    explicit Label(const char* text) { std::memset(bytes, 0, sizeof bytes);
                                       std::memcpy(bytes, text, std::strlen(text)); }
    uint64_t arg() const { return (uint64_t)(uintptr_t)bytes; }
};
} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    printf("== test_addcontent_mount_report ==\n");
    register_builtin_hle();

    // Four installed, mountable PSAC entries with distinct labels and distinct mount points — the
    // shape of the title this was written for.
    const fs::path scratch = prosper_test::test_scratch_dir() / "addcont-mount-report";
    const fs::path app0 = scratch / "app0";
    std::error_code ec;
    fs::remove_all(scratch, ec);
    fs::create_directories(app0 / "sce_sys", ec);
    for (int i = 1; i <= 4; ++i) fs::create_directories(app0 / ("dlc" + std::to_string(i)), ec);
    { std::ofstream p(app0 / "sce_sys" / "param.json", std::ios::binary);
      p << R"({"titleId":"PPSA00001"})"; }
    { std::ofstream m(app0 / "dlc_emu.ini", std::ios::binary);
      for (int i = 1; i <= 4; ++i)
          m << "[PSAC]\n"
               "content_id=UP0000-PPSA00001_00-00000000000DLC0" << i << "\n"
               "download_status=INSTALLED\n"
               "mount_point=/app0/dlc" << i << "\n\n"; }
    set_app0_root(app0.string());

    HleFn mount = Hle::lookup(kAddcontMount);
    HleFn unmount = Hle::lookup(kAddcontUnmount);
    CHECK(mount != nullptr && unmount != nullptr, "AddcontMount/AddcontUnmount are registered");
    if (!mount || !unmount) { printf("== FAIL: %d ==\n", fails); return 1; }

    // ---------------------------------------------------------------------------------------
    // 1. Each label gets ITS OWN declared mount point, mounted in the scrambled order the live
    //    title uses. Kills a lookup that ignores the entitlement label (every arm would then read
    //    /app0/dlc1) and a lookup keyed on call ORDER rather than identity (every arm would then
    //    read the manifest's Nth entry: dlc1, dlc2, dlc3, dlc4 in that sequence).
    // ---------------------------------------------------------------------------------------
    const int order[4] = { 4, 1, 3, 2 };   // the live PPSA05325 mount order
    std::string mount_log;
    char written[4][16];
    for (int slot = 0; slot < 4; ++slot) {
        const int n = order[slot];
        const Label label(("00000000000DLC0" + std::to_string(n)).c_str());
        std::memset(written[slot], 0xAB, sizeof written[slot]);
        uint64_t rc = ~uint64_t{0};
        mount_log += capture_stderr([&] {
            rc = mount(0, label.arg(), (uint64_t)(uintptr_t)written[slot], 0, 0, 0);
        });
        const std::string expected = "/app0/dlc" + std::to_string(n);
        CHECK(rc == 0, ("mount " + std::to_string(slot) + " of DLC0" + std::to_string(n) +
                        " succeeds").c_str());
        CHECK(std::strncmp(written[slot], expected.c_str(), expected.size()) == 0 &&
                  written[slot][expected.size()] == '\0',
              ("DLC0" + std::to_string(n) + " receives its own declared mount point " +
               expected).c_str());
    }

    // The four buffers must hold four DIFFERENT paths. Stated separately because the per-call arms
    // above could all pass while some other path collapsed two entries onto one point.
    bool all_distinct = true;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (std::strncmp(written[i], written[j], 16) == 0) all_distinct = false;
    CHECK(all_distinct, "four labels yield four distinct mount points");

    // The probed mount point of the live title is neither the first nor the last one handed out.
    // This is the exact fact the #1993 falsification rests on, restated as an assertion so that a
    // change making mounts order-dependent breaks HERE rather than silently invalidating the doc.
    CHECK(std::strncmp(written[0], "/app0/dlc4", 10) == 0 &&
              std::strncmp(written[3], "/app0/dlc2", 10) == 0 &&
              std::strncmp(written[2], "/app0/dlc3", 10) == 0,
          "the third mount (dlc3) is neither the first nor the last: no single slot can hold it");

    // ---------------------------------------------------------------------------------------
    // 2. The diagnostic reports what the guest was handed — same label, same point, on one line.
    //    Kills removing the log entirely, and kills a log that prints a mount point not paired
    //    with the label that received it (which is how an instrument silently misleads).
    // ---------------------------------------------------------------------------------------
    for (int n = 1; n <= 4; ++n) {
        const std::string line = "[addcontent] mount label='00000000000DLC0" + std::to_string(n) +
                                 "' service=0 -> '/app0/dlc" + std::to_string(n) + "'";
        CHECK(contains(mount_log, line),
              ("mount diagnostic pairs DLC0" + std::to_string(n) + " with its own mount point")
                  .c_str());
    }
    // A cross-paired line must NOT appear: the log must not associate a label with another
    // entry's point. Without this arm a diagnostic that printed every known mount point on every
    // call would satisfy the four `contains` checks above.
    CHECK(!contains(mount_log, "label='00000000000DLC01' service=0 -> '/app0/dlc3'") &&
              !contains(mount_log, "label='00000000000DLC03' service=0 -> '/app0/dlc1'"),
          "mount diagnostic never cross-pairs a label with another entry's mount point");

    // ---------------------------------------------------------------------------------------
    // 3. Negative arm: an undeclared label is refused, and the diagnostic invents no mount point.
    //    This is the local-inventory rule at the instrument level — a log that answered an
    //    unknown label with a path would read as evidence that content is present when it is not.
    // ---------------------------------------------------------------------------------------
    {
        const Label absent("00000000000DLC09");
        char out[16]; std::memset(out, 0xAB, sizeof out);
        uint64_t rc = 0;
        const std::string log = capture_stderr([&] {
            rc = mount(0, absent.arg(), (uint64_t)(uintptr_t)out, 0, 0, 0);
        });
        CHECK(rc == kErrNotFound, "an undeclared entitlement label is refused with NOT_FOUND");
        CHECK(contains(log, "label='00000000000DLC09'") && contains(log, "NOT_FOUND"),
              "the refusal is reported with the label that was refused");
        CHECK(!contains(log, "/app0/dlc"),
              "a refused mount reports no mount point at all");
        bool untouched = true;
        for (char ch : out) if (ch != (char)0xAB) untouched = false;
        CHECK(untouched, "a refused mount does not write the guest's mount-point buffer");
    }

    // ---------------------------------------------------------------------------------------
    // 4. The diagnostic is bounded, and says so. A rate-limited instrument whose limit is invisible
    //    is how a volume gets quoted as a frequency; the suppression line names the cap so a reader
    //    can tell "the guest stopped" from "the log stopped".
    // ---------------------------------------------------------------------------------------
    {
        const Label label("00000000000DLC01");
        char point[16];
        const std::string log = capture_stderr([&] {
            // Each cycle emits at most two lines; 64 cycles is comfortably past any sane cap.
            for (int i = 0; i < 64; ++i) {
                std::memset(point, 0, sizeof point);
                std::memcpy(point, "/app0/dlc1", 10);
                unmount((uint64_t)(uintptr_t)point, 0, 0, 0, 0, 0);
                mount(0, label.arg(), (uint64_t)(uintptr_t)point, 0, 0, 0);
            }
        });
        CHECK(contains(log, "diagnostics suppressed after"),
              "the mount diagnostic family is bounded and announces its own cap");
        size_t announcements = 0;
        for (size_t at = log.find("diagnostics suppressed after"); at != std::string::npos;
             at = log.find("diagnostics suppressed after", at + 1)) ++announcements;
        CHECK(announcements == 1, "the cap announces itself exactly once, not once per suppressed line");
        CHECK(contains(log, "not the guest's call count"),
              "the cap line warns that its own volume is not the guest's call rate");
    }

    set_app0_root(".");
    fs::remove_all(scratch, ec);
    if (fails) { printf("== FAIL: %d check(s) failed ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
