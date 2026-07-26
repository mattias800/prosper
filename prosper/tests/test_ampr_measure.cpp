// Regression guard for the libSceAmpr command-size and completion NIDs used by Sonic Origins and
// Pathless. Size queries return byte counts, not status codes. A sizing-only call must remain pure,
// while DOLL's older SDK wrapper passes a registered APR id and relies on its legacy entry point to
// fill the destination.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

using namespace prosper;

namespace prosper {
uint32_t prosper_apr_register(const std::string& path, uint64_t size);
void prosper_apr_reset_for_test();
void prosper_ampr_advance(uint64_t cb, uint64_t bytes);
uint64_t prosper_eq_identity(uint64_t eq);
void prosper_eq_post_apr_token(uint64_t eq, uint64_t eq_identity,
                               int64_t id, uint64_t token);
}

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

struct KEvent {
    int64_t ident;
    int16_t filter;
    uint16_t flags;
    uint32_t fflags;
    int64_t data;
    uint64_t udata;
};
static_assert(sizeof(KEvent) == 0x20);

int main() {
    std::printf("== test_ampr_measure ==\n");
    register_builtin_hle();

    HleFn write_address = Hle::lookup("4fgtGfXDrFc");
    HleFn write_equeue  = Hle::lookup("sSAUCCU1dv4");
    HleFn write_equeue_320 = Hle::lookup("Zi3dBUjgyXI");
    HleFn append_equeue = Hle::lookup("H896Pt-yB4I");
    HleFn append_equeue_320 = Hle::lookup("o67gODLFpls");
    HleFn construct      = Hle::lookup("8aI7R7WaOlc");
    HleFn reset          = Hle::lookup("baQO9ez2gL4");
    HleFn clear          = Hle::lookup("ULvXMDz56po");
    HleFn get_size       = Hle::lookup("tZDDEo2tE5k");
    HleFn get_offset     = Hle::lookup("GnxKOHEawhk");
    HleFn submit         = Hle::lookup("ASoW5WE-UPo");
    HleFn destruct       = Hle::lookup("GuchCTefuZw");
    HleFn add_ampr_event = Hle::lookup("bBfz7kMF2Ho");
    HleFn read_file     = Hle::lookup("vWU-odnS+fU");
    CHECK(write_address && write_equeue && write_equeue_320 && append_equeue && append_equeue_320 &&
              construct && reset && clear && get_size && get_offset && submit && destruct &&
              add_ampr_event && read_file,
          "AMPR measure and completion-event NIDs registered");
    if (add_ampr_event)
        CHECK(add_ampr_event(0, 0, 0, 0, 0, 0) == 0,
              "AMPR event registration accepts a null queue as a no-op");

    if (write_address) {
        CHECK(write_address(8, 0, 0, 0, 0, 0) == 32,
              "write-address reserves the conservative 32-byte record");
        CHECK(write_address(8, 1ull << 34, 0, 0, 0, 0) == 32,
              "write-address size does not under-allocate for a wide value");
    }
    if (write_equeue)
        CHECK(write_equeue(0, 0, 0, 0, 0, 0) == 20,
              "kernel-equeue command measures 20 bytes");
    if (write_equeue_320)
        CHECK(write_equeue_320(0, 0, 0, 0, 0, 0) == 0x20,
              "PS5 3.20 kernel-equeue command measures a 0x20-byte record");
    if (construct && reset && clear && get_size && get_offset && submit && append_equeue &&
        append_equeue_320) {
        std::array<uint64_t, 8> cb_storage{};
        const uint64_t cb = (uint64_t)(uintptr_t)cb_storage.data();
        CHECK(construct(cb, 0x200000000ull, 0x560, 0, 0, 0) == 0 &&
                  get_size(cb, 0, 0, 0, 0, 0) == 0x560,
              "Pathless constructor shape records capacity from a2");
        CHECK(get_offset(cb, 0, 0, 0, 0, 0) == 0,
              "new command buffer starts at offset zero");
        prosper_ampr_advance(cb, 24);
        CHECK(append_equeue_320(cb, 0, 0, 0, 0, 0) == 0 &&
                  get_offset(cb, 0, 0, 0, 0, 0) == 56,
              "ReadFile plus PS5 3.20 completion advances by measured bytes");
        CHECK(submit(cb, 1, 0, 0, 0, 0) == 0 && get_offset(cb, 0, 0, 0, 0, 0) == 0 &&
                  get_size(cb, 0, 0, 0, 0, 0) == 0x560,
              "submit consumes commands while preserving buffer capacity");
        CHECK(clear(cb, 0, 0, 0, 0, 0) == 0 && get_offset(cb, 0, 0, 0, 0, 0) == 0,
              "clear resets the current offset");
        CHECK(append_equeue(cb, 0, 0, 0, 0, 0) == 0 &&
                  get_offset(cb, 0, 0, 0, 0, 0) == 0,
              "legacy completion writer retains eager zero-used compatibility");
        CHECK(reset(cb, 0, 0, 0, 0, 0) == 0 && get_offset(cb, 0, 0, 0, 0, 0) == 0,
              "reset returns the current offset to zero");

        std::array<uint64_t, 8> legacy_storage{};
        const uint64_t legacy_cb = (uint64_t)(uintptr_t)legacy_storage.data();
        construct(legacy_cb, 0, 0, 0, 0, 0x720);
        prosper_ampr_advance(legacy_cb, 20);
        CHECK(get_size(legacy_cb, 0, 0, 0, 0, 0) == 0x720 &&
                  get_offset(legacy_cb, 0, 0, 0, 0, 0) == 0,
              "legacy a5-capacity buffers retain eager zero-used compatibility");
        construct(legacy_cb, 0x81, 1, 0, 0, 0x5a0);
        prosper_ampr_advance(legacy_cb, 20);
        CHECK(get_size(legacy_cb, 0, 0, 0, 0, 0) == 0x81 &&
                  get_offset(legacy_cb, 0, 0, 0, 0, 0) == 0,
              "DOLL a2=1 constructor shape keeps a1 capacity and zero-used compatibility");
        construct(legacy_cb, 0, 1, 0, 0, 0x5a0);
        CHECK(get_size(legacy_cb, 0, 0, 0, 0, 0) == 0x5a0,
              "DOLL a2=1 mode flag does not shadow its a5 capacity");
    }
#ifndef _WIN32
    // Prosper executes ReadFile at append time. If a PS5 3.20 partial batch remains unsent when the
    // producer queue drains, the matching completion record must eventually release its waiter;
    // an immediate explicit submit must still produce exactly one event.
    auto create_eq = Hle::lookup(nid_hash("sceKernelCreateEqueue"));
    auto delete_eq = Hle::lookup(nid_hash("sceKernelDeleteEqueue"));
    auto wait_eq = Hle::lookup(nid_hash("sceKernelWaitEqueue"));
    auto get_count = Hle::lookup(nid_hash("sceKernelGetEventCount"));
    if (construct && append_equeue_320 && submit && destruct && create_eq && delete_eq && wait_eq &&
        get_count) {
        uint64_t eq = 0;
        create_eq((uint64_t)(uintptr_t)&eq, 0, 0, 0, 0, 0);
        std::array<uint64_t, 8> tail_storage{};
        const uint64_t tail_cb = (uint64_t)(uintptr_t)tail_storage.data();
        constexpr int64_t event_id = 0x74fe;
        constexpr uint64_t tail_tag = 0x3e8;
        construct(tail_cb, 0, 0x560, 0, 0, 0);
        append_equeue_320(tail_cb, eq, (uint64_t)event_id, tail_tag, 0, 0);
        KEvent tail_event{};
        int32_t tail_out = -1;
        uint32_t tail_timeout = 100000;
        wait_eq(eq, (uint64_t)(uintptr_t)&tail_event, 1, (uint64_t)(uintptr_t)&tail_out,
                (uint64_t)(uintptr_t)&tail_timeout, 0);
        CHECK(tail_out == 1 && tail_event.ident == event_id && tail_event.filter == -24 &&
                  (uint64_t)tail_event.data == tail_tag,
              "unsent PS5 3.20 tail receives one deferred eager completion");

        constexpr uint64_t submitted_tag = tail_tag + 1;
        append_equeue_320(tail_cb, eq, (uint64_t)event_id, submitted_tag, 0, 0);
        submit(tail_cb, 1, 0, 0, 0, 0);
        KEvent submitted_event{};
        int32_t submitted_out = -1;
        uint32_t submitted_timeout = 100000;
        wait_eq(eq, (uint64_t)(uintptr_t)&submitted_event, 1,
                (uint64_t)(uintptr_t)&submitted_out,
                (uint64_t)(uintptr_t)&submitted_timeout, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        CHECK(submitted_out == 1 && (uint64_t)submitted_event.data == submitted_tag &&
                  get_count(eq, 0, 0, 0, 0, 0) == 0,
              "explicit submit wins the fallback race without a duplicate completion");
        delete_eq(eq, 0, 0, 0, 0, 0);

        uint64_t stale_eq = 0;
        create_eq((uint64_t)(uintptr_t)&stale_eq, 0, 0, 0, 0, 0);
        const uint64_t stale_identity = prosper_eq_identity(stale_eq);
        std::array<uint64_t, 8> stale_storage{};
        const uint64_t stale_cb = (uint64_t)(uintptr_t)stale_storage.data();
        construct(stale_cb, 0, 0x560, 0, 0, 0);
        append_equeue_320(stale_cb, stale_eq, (uint64_t)event_id, tail_tag + 2, 0, 0);
        delete_eq(stale_eq, 0, 0, 0, 0, 0);
        uint64_t replacement_eq = 0;
        create_eq((uint64_t)(uintptr_t)&replacement_eq, 0, 0, 0, 0, 0);
        const uint64_t replacement_identity = prosper_eq_identity(replacement_eq);
        CHECK(stale_identity && replacement_identity && stale_identity != replacement_identity,
              "replacement equeue receives a distinct lifetime identity");
        prosper_eq_post_apr_token(replacement_eq, stale_identity,
                                  event_id, tail_tag + 3);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        CHECK(get_count(replacement_eq, 0, 0, 0, 0, 0) == 0,
              "deleted AMPR queue cannot post into a replacement lifetime");

        std::array<uint64_t, 8> destroyed_storage{};
        const uint64_t destroyed_cb = (uint64_t)(uintptr_t)destroyed_storage.data();
        construct(destroyed_cb, 0, 0x560, 0, 0, 0);
        append_equeue_320(destroyed_cb, replacement_eq, (uint64_t)event_id,
                          tail_tag + 4, 0, 0);
        destruct(destroyed_cb, 0, 0, 0, 0, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        CHECK(get_count(replacement_eq, 0, 0, 0, 0, 0) == 0,
              "destroying a PS5 3.20 command buffer cancels its pending tail");
        delete_eq(replacement_eq, 0, 0, 0, 0, 0);
    }
#endif
    if (read_file) {
        CHECK(read_file(0, 0, UINT32_MAX, 0xffffffffffull, 0, 0) == 24,
              "wide-offset read command measures 24 bytes (never EINVAL)");
        CHECK(read_file(0, 0, 0, 0xffffffffull, 0, 0) == 20,
              "32-bit-offset read command measures 20 bytes");

        const char* fixture_path = "prosper-test-ampr-measure.tmp";
        const std::array<uint8_t, 4> fixture = {0x10, 0x20, 0x30, 0x40};
        bool fixture_written = false;
        if (FILE* file = std::fopen(fixture_path, "wb")) {
            const size_t written = std::fwrite(fixture.data(), 1, fixture.size(), file);
            const bool closed = std::fclose(file) == 0;
            fixture_written = written == fixture.size() && closed;
        }
        prosper_apr_reset_for_test();
        std::array<uint8_t, 4> destination = {0xA1, 0xB2, 0xC3, 0xD4};
        const auto before = destination;
        CHECK(read_file(0, (uint64_t)(uintptr_t)destination.data(), destination.size(),
                        0, 0, 0) == 20 &&
                  destination == before,
              "unregistered read-file measurement is a pure query");
        const uint32_t id = prosper_apr_register(fixture_path, fixture.size());
        CHECK(fixture_written &&
                  read_file(id, (uint64_t)(uintptr_t)destination.data(), destination.size(),
                            0, 0, 0) == 20 &&
                  destination == fixture,
              "registered read-file measurement preserves the DOLL compatibility read");
        prosper_apr_reset_for_test();
        std::remove(fixture_path);
    }

    std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
