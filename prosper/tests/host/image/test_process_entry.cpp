// The process entry vector must be guest-backed, complete and stable before any guest code runs.
// Independent layout/overflow controls prevent an argv-only helper from corrupting envp or the stack.
#include "host/image/process_entry.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

TEST(ProcessEntry, GuestBackedLayoutIncludesEnvironmentAndAlignment) {
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            const std::vector<std::string> args(
                {"/app0/bin/game.elf", "two words", "", "--frames"});
            std::string error;
            if (!prosper::prepare_process_entry(args, ";A=hello world;;B=one=two;", &error))
                std::_Exit(24);
            const auto c = prosper::process_entry_context();
            const uint64_t begin = reinterpret_cast<uint64_t>(c.stack);
            const auto* v = reinterpret_cast<const uint64_t*>(c.stack_pointer);
            if (!begin || c.stack_pointer % 16 != 8 || c.argv != c.stack_pointer + 8 ||
                c.argc != 4 || v[0] != 4 || v[5] != 0 || v[8] != 0 || v[9] != 0 || v[10] != 0)
                std::_Exit(25);
            const std::vector<std::string> strings(
                {args[0], args[1], args[2], args[3], "A=hello world", "B=one=two"});
            const auto indices = std::vector<unsigned>({1, 2, 3, 4, 6, 7});
            for (size_t n = 0; n < strings.size(); ++n) {
                const uint64_t pointer = v[indices[n]];
                if (pointer < begin || pointer >= begin + c.stack_bytes ||
                    std::strcmp(reinterpret_cast<const char*>(pointer), strings[n].c_str()) != 0)
                    std::_Exit(26);
            }
            std::_Exit(23);
        },
        ::testing::ExitedWithCode(23), "");
}

TEST(ProcessEntry, OversizedInputCannotPublishOrUnderflowTheStack) {
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            std::string error;
            if (prosper::prepare_process_entry({std::string(size_t{17} * 1024 * 1024, 'x')},
                                               nullptr, &error) ||
                error.empty())
                std::_Exit(24);
            const auto refused = prosper::process_entry_context();
            if (refused.stack || refused.argc || refused.argv) std::_Exit(25);
            if (!prosper::prepare_process_entry({"/app0/eboot.bin"}, nullptr, &error))
                std::_Exit(26);
            std::_Exit(prosper::process_entry_context().argc == 1 ? 23 : 27);
        },
        ::testing::ExitedWithCode(23), "");
}

TEST(ProcessEntry, RepeatedPreparationRetainsOriginalGuestStorage) {
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            std::vector<std::string> input({"", "two words"});
            if (!prosper::prepare_process_entry(input, nullptr)) std::_Exit(24);
            const auto first = prosper::process_entry_context();
            input[1] = "mutated host input";
            if (!prosper::prepare_process_entry({"different"}, "IGNORED=new")) std::_Exit(25);
            const auto later = prosper::process_entry_context();
            const auto* argv = reinterpret_cast<const uint64_t*>(later.argv);
            const bool stable = first.argv == later.argv && first.stack == later.stack &&
                                later.argc == 2 && argv[2] == 0 &&
                                !std::strcmp(reinterpret_cast<const char*>(argv[0]), "") &&
                                !std::strcmp(reinterpret_cast<const char*>(argv[1]), "two words");
            std::_Exit(stable ? 23 : 26);
        },
        ::testing::ExitedWithCode(23), "");
}
