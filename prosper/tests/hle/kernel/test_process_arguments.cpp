// Returning 0/NULL loses argv even when crt0 received it. Exercise the registered queries during
// real eager initialization and primary entry, with guest-backed identity and negative controls.
#include "hle/dispatch/dispatch.hpp"
#include "host/image/exec_image.hpp"
#include "host/image/guest_executable.hpp"
#include "host/image/process_entry.hpp"
#include "support/death_test.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {
prosper::HleFn argc_query = nullptr;
prosper::HleFn argv_query = nullptr;
uint64_t initializer_argc = 0;
uint64_t initializer_argv = 0;

// This is entered as guest code. On Windows it must use SysV, while the queried HLE functions
// below retain their ordinary host ABI. A host-ABI initializer can overwrite the bridge's saves.
PROSPER_GUEST_ABI uint64_t argument_consuming_initializer(uint64_t, uint64_t, uint64_t) {
    initializer_argc = argc_query(1, 2, 3, 4, 5, 6);
    initializer_argv = argv_query(6, 5, 4, 3, 2, 1);
    return 0;
}

void configure(const char* name, const char* value) {
#ifdef _WIN32
    // Boot reads getenv's CRT table, which SetEnvironmentVariableA does not update in-process.
    // An empty CRT value removes the variable, matching the nullptr case below.
    if (_putenv_s(name, value ? value : "") != 0) std::_Exit(27);
#else
    // Only the isolated death-test child configures its environment, before guest execution.
    // NOLINTBEGIN(concurrency-mt-unsafe)
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
    // NOLINTEND(concurrency-mt-unsafe)
#endif
}

void query_functions() {
    prosper::register_builtin_hle();
    const auto a = prosper::Hle::lookup("iKJMWrAumPE");
    const auto v = prosper::Hle::lookup("FJmglmTMdr4");
    argc_query = a;
    argv_query = v;
}

bool initializers(const std::vector<std::string>& expected) {
    prosper::install_trap_handler();
    if (prosper::run_guest_inits({reinterpret_cast<uint64_t>(&argument_consuming_initializer)}) !=
        1)
        return false;
    if (initializer_argc != expected.size() || !initializer_argv) return false;
    const auto context = prosper::process_entry_context();
    if (initializer_argv != context.stack_pointer + 8 || initializer_argv != context.argv ||
        initializer_argc != context.argc)
        return false;
    const auto* argv = reinterpret_cast<const uint64_t*>(initializer_argv);
    for (size_t n = 0; n < expected.size(); ++n) {
        const uint64_t begin = reinterpret_cast<uint64_t>(context.stack);
        if (argv[n] < begin || argv[n] >= begin + context.stack_bytes ||
            std::strcmp(reinterpret_cast<const char*>(argv[n]), expected[n].c_str()) != 0)
            return false;
    }
    return argv[expected.size()] == 0;
}
} // namespace

TEST(ProcessArguments, UninitializedCellsAndIgnoredOperands) {
    query_functions();
    ASSERT_NE(argc_query, nullptr);
    ASSERT_NE(argv_query, nullptr);
    // Native getters are operand-free reads of startup cells; unused values must not be dereferenced.
    EXPECT_EQ(argc_query(UINT64_MAX, 1, 2, 3, 4, 5), 0u);
    EXPECT_EQ(argv_query(1, UINT64_MAX, 2, 3, 4, 5), 0u);
}

TEST(ProcessArguments, EagerInitReceivesOrdinaryConfiguredArguments) {
    query_functions();
    ASSERT_NE(argc_query, nullptr);
    ASSERT_NE(argv_query, nullptr);
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            configure("PROSPER_GUEST_ARGS", "-force-gfx-direct two");
            configure("PROSPER_GUEST_ENV", nullptr);
            const std::vector<std::string> expected(
                {"/app0/eboot.bin", "-force-gfx-direct", "two"});
            std::_Exit(initializers(expected) ? 23 : 24);
        },
        ::testing::ExitedWithCode(23), "");
}

TEST(ProcessArguments, EagerInitReceivesExactReplacementArguments) {
    query_functions();
    ASSERT_NE(argc_query, nullptr);
    ASSERT_NE(argv_query, nullptr);
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            configure("PROSPER_GUEST_ARGS", "must not replace the exact vector");
            configure("PROSPER_GUEST_ENV", nullptr);
            const std::vector<std::string> expected(
                {"/app0/bin/game.elf", "two words", "", "--dump"});
            prosper::set_guest_entry_arguments(expected);
            std::_Exit(initializers(expected) ? 23 : 24);
        },
        ::testing::ExitedWithCode(23), "");
}

TEST(ProcessArguments, FailedPreparationSkipsInitializersAndPrimaryEntry) {
    query_functions();
    ASSERT_NE(argc_query, nullptr);
    ASSERT_NE(argv_query, nullptr);
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            configure("PROSPER_GUEST_ENV", nullptr);
            prosper::set_guest_entry_arguments({std::string(size_t{17} * 1024 * 1024, 'x')});
            initializer_argc = UINT64_MAX;
            initializer_argv = UINT64_MAX;
            if (prosper::run_guest_inits(
                    {reinterpret_cast<uint64_t>(&argument_consuming_initializer)}) != 0 ||
                initializer_argc != UINT64_MAX || initializer_argv != UINT64_MAX)
                std::_Exit(24);
            // The initializer API reports a count, not a fatal boot status. The primary boundary
            // must independently refuse the same configuration before attempting this bad entry.
            prosper::LoadedImage image;
            image.entry = 1;
            const auto result = prosper::run_entry(image);
            const auto context = prosper::process_entry_context();
            const bool refused =
                result.kind == 2 && result.detail == "guest arguments exceed the entry stack" &&
                !context.stack && !context.argv && !context.argc &&
                argc_query(0, 0, 0, 0, 0, 0) == 0 && argv_query(0, 0, 0, 0, 0, 0) == 0;
            std::_Exit(refused ? 23 : 25);
        },
        ::testing::ExitedWithCode(23), "");
}

TEST(ProcessArguments, PrimaryEntryUsesTheInitializerVector) {
    query_functions();
    ASSERT_NE(argc_query, nullptr);
    ASSERT_NE(argv_query, nullptr);
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            configure("PROSPER_GUEST_ENV", nullptr);
            const std::vector<std::string> expected({"/app0/bin/game.elf", "two words", ""});
            prosper::set_guest_entry_arguments(expected);
            if (!initializers(expected)) std::_Exit(24);
            uint64_t actual_entry_vector = 0;
            // Synthetic guest leaf: record the actual crt0 RDI, then #UD so the real recovery path
            // returns to the test's host stack. No title, display backend, or renderer is involved.
            std::vector<uint8_t> code({0x48, 0xb8});   // mov rax, witness
            const uint64_t witness = reinterpret_cast<uint64_t>(&actual_entry_vector);
            for (unsigned n = 0; n < 8; ++n) code.push_back(witness >> (8 * n));
            code.insert(code.end(), {0x48, 0x89, 0x38, 0x0f, 0x0b});
            void* mapping = nullptr;
#ifdef _WIN32
            mapping = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
#else
            mapping = mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (mapping == MAP_FAILED) mapping = nullptr;
#endif
            if (!mapping) std::_Exit(25);
            std::memcpy(mapping, code.data(), code.size());
#ifdef _WIN32
            FlushInstructionCache(GetCurrentProcess(), mapping, code.size());
#endif
            prosper::LoadedImage image;
            image.entry = reinterpret_cast<uint64_t>(mapping);
            const auto result = prosper::run_entry(image);
            const bool matches = result.kind == 3 && actual_entry_vector + 8 == initializer_argv &&
                                 argv_query(0, 0, 0, 0, 0, 0) == initializer_argv &&
                                 argc_query(0, 0, 0, 0, 0, 0) == expected.size();
            std::_Exit(matches ? 23 : 26);
        },
        ::testing::ExitedWithCode(23), "");
}
