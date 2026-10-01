// test_pthread_sched_getters — scePthreadGetprio / scePthreadGetaffinity must FILL their out-param.
//
// Both have correct handlers in hle_kernel.cpp (k_getprio writes priority 700; k_attr_getaffinity
// writes an 8-core mask 0xff). hle_kernel_time.cpp ALSO registered them to a bare no-op (k_ok), and
// register_kernel_time_hle() runs AFTER register_kernel_hle(), so — registration being last-write-
// wins — the no-op silently shadowed the real handlers: the getter returned success while never
// writing its out-param, handing the caller uninitialized stack memory (the exact harmful-stub class
// the hle_kernel.cpp comment says was already fixed). This locks the shadowing out: a Get* that
// returns OK MUST have written its out-param. Fails if the no-op registration ever wins again.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main(int argc, char** argv) {
    const bool one_cpu = argc == 2 && !std::strcmp(argv[1], "--one-cpu");
    // The policy is cached on first use, so exercise the diagnostic in a fresh process.
#ifdef _WIN32
    _putenv_s("PROSPER_ONE_CPU", one_cpu ? "1" : "");
#else
    if (one_cpu) setenv("PROSPER_ONE_CPU", "1", 1);
    else unsetenv("PROSPER_ONE_CPU");
#endif
    printf("== test_pthread_sched_getters ==\n");
    register_builtin_hle();

    HleFn getprio     = Hle::lookup(nid_hash("scePthreadGetprio"));
    HleFn getaffinity = Hle::lookup(nid_hash("scePthreadGetaffinity"));
    CHECK(getprio != nullptr, "scePthreadGetprio is registered");
    CHECK(getaffinity != nullptr, "scePthreadGetaffinity is registered");
    if (!getprio || !getaffinity) { printf("== FAIL ==\n"); return 1; }

    // scePthreadGetprio(thread, int* prio): must write the priority (Sony default 700), not leave the
    // caller's int untouched. Pre-seed a sentinel that a working handler overwrites.
    const int32_t SENT = (int32_t)0x0BADF00D;
    int32_t prio = SENT;
    uint64_t rp = getprio(1 /*dummy thread*/, (uint64_t)(uintptr_t)&prio, 0, 0, 0, 0);
    CHECK(rp == 0, "scePthreadGetprio returns OK");
    CHECK(prio != SENT, "scePthreadGetprio WROTE the prio out-param (not the shadowing no-op)");
    CHECK(prio == 700, "scePthreadGetprio writes the Sony default priority 700");

    // scePthreadGetaffinity(thread, SceKernelCpumask* mask): must write a usable (non-empty) core mask.
    uint64_t mask = 0xEEEEEEEEEEEEEEEEull;
    uint64_t ra = getaffinity(1 /*dummy thread*/, (uint64_t)(uintptr_t)&mask, 0, 0, 0, 0);
    CHECK(ra == 0, "scePthreadGetaffinity returns OK");
    CHECK(mask != 0xEEEEEEEEEEEEEEEEull, "scePthreadGetaffinity WROTE the mask out-param (not the shadowing no-op)");
    CHECK(mask != 0 && (mask & 0x1) != 0, "scePthreadGetaffinity writes a non-empty core mask (>=1 usable CPU)");

    // Use the actual import, not only a name-derived lookup: the registry must resolve this NID
    // after every built-in library has registered. The independent name hash fixes its identity.
    CHECK(nid_hash("sceKernelGetAvailableCpumask") == "La9uyZv4Kvw",
          "available CPU mask name matches the observed import NID");
    HleFn available = Hle::lookup("La9uyZv4Kvw");
    CHECK(available != nullptr, "sceKernelGetAvailableCpumask is registered at its imported NID");
    if (available) {
        const uint64_t cpus = available(0, 0, 0, 0, 0, 0);
        CHECK(cpus == (one_cpu ? 0x01 : 0xff), "available CPU mask returns the selected guest policy by value");
        CHECK(cpus == mask, "available CPU mask agrees with the thread affinity getter");
        uint64_t attr_mask = 0xEEEEEEEEEEEEEEEEull;
        HleFn attr_affinity = Hle::lookup(nid_hash("scePthreadAttrGetaffinity"));
        CHECK(attr_affinity != nullptr, "attribute affinity getter is registered");
        if (attr_affinity) {
            CHECK(attr_affinity(0, (uint64_t)(uintptr_t)&attr_mask, 0, 0, 0, 0) == 0,
                  "attribute affinity query succeeds");
            CHECK(attr_mask == cpus, "available CPU mask agrees with the attribute affinity getter");
        }
        // The guest consumer subtracts one for its main thread, then allocates two arrays of
        // eight-byte worker entries. An empty mask wraps this count and requests 0x7fffffff8.
        const uint32_t workers = static_cast<uint32_t>(std::popcount(cpus)) - 1;
        CHECK(workers == (one_cpu ? 0u : 7u), "guest worker count does not underflow");
        CHECK(static_cast<uint64_t>(workers) * 8 == (one_cpu ? 0u : 56u),
              "guest worker array size stays within the advertised CPU budget");
    }

    if (!one_cpu) {
        const std::string command = "\"" + std::string(argv[0]) + "\" --one-cpu";
        CHECK(std::system(command.c_str()) == 0, "fresh-process one-CPU policy arm passes");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
