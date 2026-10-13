// Host option names and spaces inside a guest argv entry remain guest data after replacement.
// Counted argv transport prevents them from becoming host options or being split/requoted.
#include "loadexec.hpp"
#include "host/image/guest_executable.hpp"
#include "host/platform/lifecycle.hpp"
#include "host/platform/process_replace.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
std::vector<char*> pointers(std::vector<std::string>& values) {
    std::vector<char*> result;
    for (auto& value : values) result.push_back(value.data());
    return result;
}
}   // namespace

TEST(LoadExecOptions, ExactArgumentsIncludingHostOptionNames) {
    std::vector<std::string> values{"app", "--guest-argv", "5",      "different argv0", "two words",
                                    "",    "--dump",       "ends\\", "--frames",        "1"};
    auto args = pointers(values);
    int index = 1;
    std::string error;
    ASSERT_TRUE(prosper::frontend::parse_loadexec_option(index, args.size(), args.data(), &error));
    EXPECT_EQ(index, 7);
    EXPECT_EQ(prosper::guest_entry_arguments(),
              (std::vector<std::string>{"different argv0", "two words", "", "--dump", "ends\\"}));
    EXPECT_EQ(values[index + 1], "--frames")
        << "host parsing resumes after the counted guest vector";
}

TEST(LoadExecOptions, InvalidCountPreservesArguments) {
    const std::vector<std::string> sentinel{"unchanged"};
    prosper::set_guest_entry_arguments(sentinel);
    for (const char* count : {"0", "2", "257", "1oops", "-1"}) {
        std::vector<std::string> values{"app", "--guest-argv", count, "only-one"};
        auto args = pointers(values);
        int index = 1;
        std::string error;
        EXPECT_FALSE(
            prosper::frontend::parse_loadexec_option(index, args.size(), args.data(), &error));
        EXPECT_FALSE(error.empty());
        EXPECT_EQ(prosper::guest_entry_arguments(), sentinel);
    }
}

TEST(LoadExecOptions, RelaunchDropsPreviousGuestPayloadOnly) {
    std::vector<std::string> values{
        "app",          "--dump", "dump root",   "--guest-executable", "/app0/game/main.elf",
        "--guest-argv", "3",      "guest argv0", "--frames",           "999",
        "--frames",     "1"};
    auto args = pointers(values);
    EXPECT_EQ(prosper::frontend::loadexec_host_arguments(args.size(), args.data()),
              (std::vector<std::string>{"--dump", "dump root", "--frames", "1"}));
}

TEST(LoadExecFrontend, NativePreflightFailureLeavesOldGuestRunning) {
    prosper::prosper_reset_stop();
    std::vector<std::string> values{"app"};
    auto args = pointers(values);
    auto backend = prosper::frontend::make_loadexec_frontend_backend(
        args.size(), args.data(), "prosper-nonexistent-native-executable");
    prosper::LoadExecRequest request;
    uint64_t result = 0;
    std::thread caller([&] { result = backend(request); });
    caller.join();
    EXPECT_NE(result, 0u);
    EXPECT_FALSE(prosper::prosper_stop_requested())
        << "failed launch preflight cannot stop the old guest";
}

TEST(LoadExecFrontend, BootstrapCannotParkItsOwnFrontend) {
    std::string native_executable = "/bin/sh";
#ifdef _WIN32
    char path[4096];
    const DWORD count = GetModuleFileNameA(nullptr, path, sizeof path);
    ASSERT_GT(count, 0u);
    ASSERT_LT(count, sizeof path);
    native_executable.assign(path, count);
#endif
    ASSERT_EQ(prosper::host::process_replace_preflight(native_executable), 0)
        << "the bootstrap negative must use a loadable native target";
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            prosper::prosper_reset_stop();
            std::atomic<bool> completed{false};
            uint64_t result = 0;
            std::thread frontend([&] {
                std::vector<std::string> values{"app"};
                auto args = pointers(values);
                auto backend = prosper::frontend::make_loadexec_frontend_backend(
                    args.size(), args.data(), native_executable);
                prosper::LoadExecRequest request;
                request.dump_root = "same dump";
                request.executable = "/app0/game/main.elf";
                request.argv.push_back(request.executable);
                result = backend(request);
                completed.store(true);
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!completed.load()) {
                if (std::chrono::steady_clock::now() > deadline) std::_Exit(24);
                std::this_thread::yield();
            }
            frontend.join();
            std::_Exit(result && !prosper::prosper_stop_requested() ? 23 : 25);
        },
        ::testing::ExitedWithCode(23), "bootstrap thread unsupported");
}

#ifndef _WIN32
TEST(LoadExecFrontend, PendingCallerParksThroughStopAndNativeReplacement) {
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            prosper::prosper_reset_stop();
            std::vector<std::string> values({"sh", "-c", "exit 23", "argv0"});
            auto args = pointers(values);
            auto backend = prosper::frontend::make_loadexec_frontend_backend(
                args.size(), args.data(), "/bin/sh");
            prosper::LoadExecRequest request;
            request.dump_root = "unchanged dump";
            request.executable = "/app0/game/main.elf";
            request.argv.push_back("different argv0");
            request.argv.push_back("two words");
            std::atomic<bool> returned{false};
            std::thread old_guest([&] {
                backend(request);
                returned.store(true);
            });
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!prosper::prosper_stop_requested()) {
                if (returned.load() || std::chrono::steady_clock::now() > deadline) std::_Exit(24);
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (returned.load()) std::_Exit(25);
            // Exercise the same finish entry main calls AFTER its GPU shutdown. This CPU fixture does
            // not assert a GPU drain; it proves the real queued caller never resumes before replacement.
            prosper::frontend::finish_loadexec_replacement();
            std::_Exit(26);
        },
        ::testing::ExitedWithCode(23), "replacing guest");
}
#endif
