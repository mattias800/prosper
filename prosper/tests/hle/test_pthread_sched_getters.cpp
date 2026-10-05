// test_pthread_sched_getters — scePthreadGetprio / scePthreadGetaffinity must FILL their out-param.
//
// Both have correct handlers in hle_kernel.cpp (k_getprio writes priority 700; k_attr_getaffinity
// writes an 8-core mask 0xff). hle_kernel_time.cpp ALSO registered them to a bare no-op (k_ok), and
// register_kernel_time_hle() runs AFTER register_kernel_hle(), so — registration being last-write-
// wins — the no-op silently shadowed the real handlers: the getter returned success while never
// writing its out-param, handing the caller uninitialized stack memory (the exact harmful-stub class
// the hle_kernel.cpp comment says was already fixed). This locks the shadowing out: a Get* that
// returns OK MUST have written its out-param. Fails if the no-op registration ever wins again.
//
// The guest CPU mask is a PROCESS-cached policy (available_guest_cpumask() reads PROSPER_ONE_CPU once,
// on first use), so the two policies cannot be observed in one process. The policy-sensitive arms
// therefore take their expectation from the variable this process was actually launched under — the
// same expression the pre-GTest single-binary version used — and OneCpuPolicyHoldsInAFreshProcess
// launches a second process with the variable set to cover the other half.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace prosper;

namespace {
constexpr int32_t kPrioSentinel = static_cast<int32_t>(0x0BADF00D);
constexpr uint64_t kMaskSentinel = 0xEEEEEEEEEEEEEEEEull;
// Set by the fresh-process arm on the child it launches; unset in every other process, so the
// evidence file is written only by a process that was actually spawned by that arm.
constexpr const char* kEvidenceVariable = "PROSPER_SCHED_GETTERS_EVIDENCE";

HleFn lookup(const char* name) {
    register_builtin_hle();
    return Hle::lookup(nid_hash(name));
}
// The guest CPU policy this process booted under: 8 advertised cores by default, 1 under
// PROSPER_ONE_CPU (which sizes a Unity job-system worker pool down to the main thread).
uint64_t expected_cpumask() {
    return std::getenv("PROSPER_ONE_CPU") != nullptr ? 0x01 : 0xff;
}

void set_variable(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (!value.empty())
        setenv(name, value.c_str(), 1);
    else
        unsetenv(name);
#endif
}
std::string hex64(uint64_t value) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%llx", static_cast<unsigned long long>(value));
    return buf;
}
// The program's own path, for re-launching this binary. testing::internal::GetArgvs() is the only
// route to argv under GoogleTest here: the alternative is a hand-written main(), which
// tools/ci/check_gtest_policy.py rejects outright. It returns the vector BY VALUE, so this copies
// argv[0] out rather than handing back a pointer into a temporary.
std::string self_path() {
    const std::vector<std::string> argv = ::testing::internal::GetArgvs();
    return argv.empty() ? std::string() : argv.front();
}
}   // namespace

TEST(PthreadSchedGetters, EveryGetterAndTheAvailableMaskAreRegistered) {
    EXPECT_NE(lookup("scePthreadGetprio"), nullptr) << "scePthreadGetprio is registered";
    EXPECT_NE(lookup("scePthreadGetaffinity"), nullptr) << "scePthreadGetaffinity is registered";
    EXPECT_NE(lookup("scePthreadAttrGetaffinity"), nullptr)
        << "attribute affinity getter is registered";
    // Use the actual import, not only a name-derived lookup: the registry must resolve this NID
    // after every built-in library has registered. The independent name hash fixes its identity.
    EXPECT_EQ(nid_hash("sceKernelGetAvailableCpumask"), "La9uyZv4Kvw")
        << "available CPU mask name matches the observed import NID";
    EXPECT_NE(Hle::lookup("La9uyZv4Kvw"), nullptr)
        << "sceKernelGetAvailableCpumask is registered at its imported NID";
}

TEST(PthreadSchedGetters, GetprioWritesItsOutParam) {
    const HleFn getprio = lookup("scePthreadGetprio");
    ASSERT_NE(getprio, nullptr);
    // scePthreadGetprio(thread, int* prio): must write the priority (Sony default 700), not leave the
    // caller's int untouched. Pre-seed a sentinel that a working handler overwrites.
    int32_t prio = kPrioSentinel;
    const uint64_t rc = getprio(1 /*dummy thread*/, reinterpret_cast<uint64_t>(&prio), 0, 0, 0, 0);
    EXPECT_EQ(rc, 0u) << "scePthreadGetprio returns OK";
    EXPECT_NE(prio, kPrioSentinel)
        << "scePthreadGetprio WROTE the prio out-param (not the shadowing no-op)";
    EXPECT_EQ(prio, 700) << "scePthreadGetprio writes the Sony default priority 700";
}

TEST(PthreadSchedGetters, GetaffinityWritesANonEmptyCoreMask) {
    const HleFn getaffinity = lookup("scePthreadGetaffinity");
    ASSERT_NE(getaffinity, nullptr);
    // scePthreadGetaffinity(thread, SceKernelCpumask* mask): must write a usable (non-empty) core mask.
    uint64_t mask = kMaskSentinel;
    const uint64_t rc =
        getaffinity(1 /*dummy thread*/, reinterpret_cast<uint64_t>(&mask), 0, 0, 0, 0);
    EXPECT_EQ(rc, 0u) << "scePthreadGetaffinity returns OK";
    EXPECT_NE(mask, kMaskSentinel)
        << "scePthreadGetaffinity WROTE the mask out-param (not the shadowing no-op)";
    EXPECT_NE(mask, 0u) << "scePthreadGetaffinity writes a non-empty core mask";
    EXPECT_NE(mask & 0x1, 0u) << "scePthreadGetaffinity reports at least one usable CPU (bit 0)";
}

TEST(PthreadSchedGetters, AvailableMaskAgreesWithTheActivePolicyAndBothGetters) {
    const HleFn getaffinity = lookup("scePthreadGetaffinity");
    const HleFn attr_affinity = lookup("scePthreadAttrGetaffinity");
    const HleFn available = Hle::lookup("La9uyZv4Kvw");
    ASSERT_NE(getaffinity, nullptr);
    ASSERT_NE(attr_affinity, nullptr);
    ASSERT_NE(available, nullptr);

    const uint64_t expected = expected_cpumask();
    uint64_t mask = kMaskSentinel;
    ASSERT_EQ(getaffinity(1 /*dummy thread*/, reinterpret_cast<uint64_t>(&mask), 0, 0, 0, 0), 0u);
    const uint64_t cpus = available(0, 0, 0, 0, 0, 0);
    // Reported to whoever launched this process. The fresh-process arm below has to distinguish "the
    // child saw PROSPER_ONE_CPU" from "the child silently ran the default policy and passed anyway",
    // and an exit code cannot: only this line's content separates the two.
    const std::string observed = std::string("policy=") +
                                 (expected == 0x01 ? "one-cpu" : "default") +
                                 " available_cpumask=0x" + hex64(cpus);
    std::printf("[pthread-sched-getters] %s\n", observed.c_str());
    std::fflush(stdout);
    if (const char* evidence = std::getenv(kEvidenceVariable)) {
        if (std::FILE* out = std::fopen(evidence, "wb")) {
            std::fwrite(observed.data(), 1, observed.size(), out);
            std::fputc('\n', out);
            std::fclose(out);
        }
    }
    EXPECT_EQ(cpus, expected) << "available CPU mask returns the selected guest policy by value";
    EXPECT_EQ(cpus, mask) << "available CPU mask agrees with the thread affinity getter";

    uint64_t attr_mask = kMaskSentinel;
    EXPECT_EQ(attr_affinity(0, reinterpret_cast<uint64_t>(&attr_mask), 0, 0, 0, 0), 0u)
        << "attribute affinity query succeeds";
    EXPECT_EQ(attr_mask, cpus) << "available CPU mask agrees with the attribute affinity getter";

    // The guest consumer subtracts one for its main thread, then allocates two arrays of
    // eight-byte worker entries. An empty mask wraps this count and requests 0x7fffffff8.
    const uint32_t workers = static_cast<uint32_t>(std::popcount(cpus)) - 1;
    const uint32_t advertised = expected == 0x01 ? 0u : 7u;
    EXPECT_EQ(workers, advertised) << "guest worker count does not underflow";
    EXPECT_EQ(static_cast<uint64_t>(workers) * 8, static_cast<uint64_t>(advertised) * 8)
        << "guest worker array size stays within the advertised CPU budget";
}

TEST(PthreadSchedGetters, OneCpuPolicyHoldsInAFreshProcess) {
    // The policy is cached per process, so the one-CPU arm needs its own process. Launching this
    // binary again with a gtest filter is the fresh process; the two variables are set here rather
    // than spelled in the child's command line, because a command line carrying more than the two
    // quotes around an executable name is one cmd.exe mangles (it strips the leading quote and the
    // LAST quote of a /c string), and this arm has to work on Windows. This process never queries a
    // mask, so arming them here cannot change what it observes.
    const std::string self = self_path();
    ASSERT_FALSE(self.empty()) << "gtest must expose the program's argv to re-launch it";
    const std::string evidence = prosper_test::test_scratch_file("pthread_sched_one_cpu.txt");
    set_variable(kEvidenceVariable, evidence);
    set_variable("PROSPER_ONE_CPU", "1");
    const std::string command =
        "\"" + self +
        "\" "
        "--gtest_filter=PthreadSchedGetters.AvailableMaskAgreesWithTheActivePolicyAndBothGetters";
    const int status = std::system(command.c_str());
    set_variable("PROSPER_ONE_CPU", "");
    set_variable(kEvidenceVariable, "");
    std::ifstream in(evidence, std::ios::binary);
    std::ostringstream captured;
    captured << in.rdbuf();
    const std::string observed = captured.str();
    EXPECT_EQ(status, 0) << "fresh-process one-CPU policy arm passes";
    // The discriminator. A child that did NOT inherit the variable would pass every assertion above --
    // it would expect the default 0xff and be handed 0xff -- so a bare exit-code check is satisfied
    // by an arm that proved nothing. Only a process that really read PROSPER_ONE_CPU reports
    // `policy=one-cpu`, and only a handler that honoured it reports that mask.
    EXPECT_NE(observed.find("policy=one-cpu available_cpumask=0x1"), std::string::npos)
        << "the child did not run under the one-CPU policy, so this arm asserted nothing. It "
           "reported: "
        << observed;
}