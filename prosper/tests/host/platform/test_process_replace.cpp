// Replacement must really enter a new native program and never continue the old caller. Preflight
// errors leave this process alive; success is observed in an owned death-test child.
#include "host/platform/process_replace.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <cerrno>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <thread>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef __linux__
TEST(ProcessReplace, LateWorkerOpenCannotEnterReplacementDescriptorTable) {
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            std::atomic<int> phase{0};
            std::atomic<int> descriptor{-1};
            std::thread producer([&] {
                phase.store(1);
                while (phase.load() < 2) std::this_thread::yield();
                const int source = ::open("/dev/null", O_RDONLY);
                descriptor.store(source >= 0 ? ::fcntl(source, F_DUPFD, 100) : -1);
                if (source >= 0) ::close(source);
                phase.store(3);
                while (phase.load() < 4) std::this_thread::yield();
                if (descriptor.load() >= 0) ::close(descriptor.load());
            });
            const auto wait_phase = [&](int target) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (phase.load() < target) {
                    if (std::chrono::steady_clock::now() > deadline) std::_Exit(25);
                    std::this_thread::yield();
                }
            };
            wait_phase(1);
            if (prosper::host::process_replace_isolate_descriptors()) std::_Exit(25);
            phase.store(2);
            wait_phase(3);
            errno = 0;
            const bool isolated = descriptor.load() >= 0 &&
                                  ::fcntl(descriptor.load(), F_GETFD) == -1 && errno == EBADF;
            phase.store(4);
            producer.join();
            std::_Exit(isolated ? 23 : 24);
        },
        ::testing::ExitedWithCode(23), "");
}
#endif

TEST(ProcessReplace, BadLaunchDoesNotReplaceCaller) {
    EXPECT_EQ(prosper::host::replace_process({}), EINVAL);
    EXPECT_NE(prosper::host::process_replace_preflight(""), 0);
    EXPECT_NE(prosper::host::replace_process({"prosper-nonexistent-native-executable"}), 0);
}

#ifndef _WIN32
TEST(ProcessReplace, NativeProgramExitProvesReplacement) {
    ASSERT_EQ(prosper::host::process_replace_preflight("/bin/sh"), 0);
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            prosper::host::replace_process({"/bin/sh", "-c",
                                            "test \"$1\" = \"two words\" && exit 23; exit 25",
                                            "argv0", "two words"});
            std::_Exit(24);
        },
        ::testing::ExitedWithCode(23), "");
}

TEST(ProcessReplace, OldDescriptorsDoNotSurviveExec) {
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            const int source = ::open("/dev/null", O_RDONLY);
            const int descriptor = source >= 0 ? ::fcntl(source, F_DUPFD, 100) : -1;
            if (descriptor < 0) std::_Exit(25);
            const std::string script =
                "test ! -e /dev/fd/" + std::to_string(descriptor) + " && exit 23; exit 25";
            prosper::host::replace_process({"/bin/sh", "-c", script});
            std::_Exit(24);
        },
        ::testing::ExitedWithCode(23), "");
}
#endif
