// A launcher must hand off its selected executable and complete argv, not receive SCE_OK while
// running the old image. Synthetic guest ELFs cover the real NID, exact arguments and refusals;
// no title dump is needed. Outside mounts, PRXs, invalid pointers and symlink escapes cannot launch.
#include "hle/dispatch/dispatch.hpp"
#include "hle/service/hle_loadexec.hpp"
#include "host/image/boot_program.hpp"
#include "host/image/guest_executable.hpp"
#include "host/image/module_path_policy.hpp"
#include "fixtures/test_scratch.h"
#include "support/death_test.hpp"

#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {
using namespace prosper;
namespace fs = std::filesystem;
constexpr uint64_t kParameter = 0x80A10003;
constexpr uint64_t kInternal = 0x80A10001;

void write_guest_elf(const fs::path& path, uint16_t type = ET_SCE_DYNEXEC) {
    fs::create_directories(path.parent_path());
    std::array<uint8_t, 1024> bytes{};
    const auto put = [&](size_t offset, auto value) {
        std::memcpy(bytes.data() + offset, &value, sizeof value);
    };
    std::memcpy(bytes.data(),
                "\x7f"
                "ELF",
                4);
    bytes[4] = 2;
    bytes[5] = 1;
    bytes[6] = 1;
    bytes[7] = 9;
    put(0x10, type);
    put(0x12, uint16_t{0x3e});
    put(0x14, uint32_t{1});
    put(0x18, uint64_t{0x100});
    put(0x20, uint64_t{0x40});
    put(0x34, uint16_t{64});
    put(0x36, uint16_t{56});
    put(0x38, uint16_t{2});
    put(0x40, uint32_t{PT_LOAD});
    put(0x44, uint32_t{5});
    put(0x60, uint64_t{bytes.size()});
    put(0x68, uint64_t{bytes.size()});
    put(0x70, uint64_t{0x1000});
    put(0x78, uint32_t{PT_DYNAMIC});
    put(0x7c, uint32_t{4});
    put(0x80, uint64_t{0x180});
    put(0x88, uint64_t{0x180});
    put(0x98, uint64_t{0x80});
    put(0xa0, uint64_t{0x80});
    put(0xa8, uint64_t{8});
    // A minimal real dynamic table: one null symbol and one null string, then DT_NULL. Keeping
    // the parser's required PT_DYNAMIC/SCE tags exercises executable validation without mocks.
    put(0x180, uint64_t{0x6100003f});
    put(0x188, uint64_t{24});
    put(0x190, uint64_t{6});
    put(0x198, uint64_t{0x220});
    put(0x1a0, uint64_t{5});
    put(0x1a8, uint64_t{0x240});
    put(0x1b0, uint64_t{10});
    put(0x1b8, uint64_t{1});
    put(0x1c0, uint64_t{11});
    put(0x1c8, uint64_t{24});
    bytes[0x100] = 0xc3;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

class LoadExec : public ::testing::Test {
protected:
    fs::path root;
    HleFn call = nullptr;
    void SetUp() override {
        root = fs::path(prosper_test::test_scratch_dir()) / "loadexec";
        fs::remove_all(root);
        fs::create_directories(root);
        write_guest_elf(root / "game" / "main.elf");
        write_guest_elf(root / "eboot.bin");
        configure_loadexec_app0(root.string());
        set_loadexec_backend({});
        register_builtin_hle();
        call = Hle::lookup("JoBqSQt1yyA");
        ASSERT_NE(call, nullptr) << "the real SystemService NID must not fall through to return-0";
    }
    void TearDown() override {
        set_loadexec_backend({});
        configure_loadexec_app0({});
        fs::remove_all(root);
    }
    uint64_t invoke(const char* path, const char* const* argv = nullptr) {
        return call(reinterpret_cast<uint64_t>(path), reinterpret_cast<uint64_t>(argv), 0, 0, 0, 0);
    }
};

TEST_F(LoadExec, ExactExecutableAndArguments) {
    bool reached = false;
    set_loadexec_backend([&](const LoadExecRequest& request) {
        reached = true;
        EXPECT_EQ(request.dump_root, root.string());
        EXPECT_EQ(request.executable, "/app0/game/main.elf");
        EXPECT_EQ(request.host_executable, fs::canonical(root / "game/main.elf").string());
        EXPECT_EQ(request.argv,
                  (std::vector<std::string>{"custom argv0", "two words", "", "--dump", "x\\\"y"}));
        return uint64_t{0x80A10001};
    });
    const char* args[]{"custom argv0", "two words", "", "--dump", "x\\\"y", nullptr};
    EXPECT_EQ(invoke("/app0/game/main.elf", args), kInternal);
    EXPECT_TRUE(reached) << "a valid request must reach process replacement with byte-exact argv";
    reached = false;
    EXPECT_EQ(invoke("\\app0\\game\\main.elf", args), kInternal);
    EXPECT_TRUE(reached) << "an accepted backslash alias must preserve a usable next-boot selector";
}

TEST_F(LoadExec, SuccessfulBackendNeverResumesOldCaller) {
    set_loadexec_backend([](const LoadExecRequest& request) -> uint64_t {
        std::_Exit(request.argv == std::vector<std::string>{"/app0/game/main.elf"} ? 23 : 24);
    });
    EXPECT_EXIT(
        {
            prosper::test::suppress_coredump();
            invoke("game/main.elf");
            std::_Exit(25);
        },
        ::testing::ExitedWithCode(23), "");
}

TEST_F(LoadExec, ReturningSuccessWithoutReplacementIsAnError) {
    set_loadexec_backend([](const LoadExecRequest&) { return uint64_t{0}; });
    EXPECT_EQ(invoke("game/main.elf"), kInternal);
    set_loadexec_backend({});
    EXPECT_EQ(invoke("game/main.elf"), kInternal)
        << "a frontend without replacement support cannot claim SCE_OK";
}

TEST_F(LoadExec, InvalidPointersAndUnterminatedArgumentsNeverLaunch) {
    unsigned launches = 0;
    set_loadexec_backend([&](const LoadExecRequest&) {
        ++launches;
        return kInternal;
    });
    EXPECT_EQ(invoke(nullptr), kParameter);
    EXPECT_EQ(invoke(reinterpret_cast<const char*>(1)), kParameter);
    EXPECT_EQ(invoke("game/main.elf", reinterpret_cast<const char* const*>(1)), kParameter);
    const char* bad[]{reinterpret_cast<const char*>(1), nullptr};
    EXPECT_EQ(invoke("game/main.elf", bad), kParameter);
    std::array<const char*, 257> unterminated;
    unterminated.fill("argument");
    EXPECT_EQ(invoke("game/main.elf", unterminated.data()), kParameter);
    EXPECT_EQ(launches, 0u) << "all pointer/cap failures must occur before committing replacement";
}

TEST_F(LoadExec, OutsideMountAndMalformedExecutableNeverLaunch) {
    unsigned launches = 0;
    set_loadexec_backend([&](const LoadExecRequest&) {
        ++launches;
        return kInternal;
    });
    write_guest_elf(root / "game/library.elf", ET_SCE_DYNAMIC);
    std::ofstream(root / "game/bad.elf") << "not an ELF";
    for (const char* path : {"/savedata0/main.elf", "/app0/../main.elf", "/app0-evil/main.elf",
                             "missing.elf", "game/library.elf", "game/bad.elf"})
        EXPECT_EQ(invoke(path), kParameter) << path;
    EXPECT_EQ(invoke((root / "game/main.elf").string().c_str()), kParameter)
        << "an absolute host path is never a guest executable path";
    EXPECT_EQ(launches, 0u);
}

TEST_F(LoadExec, SelectedMainDoesNotWidenDependentPrxPolicy) {
    std::string error;
    ASSERT_FALSE(resolve_guest_executable(root.string(), "game/main.elf", &error).empty());
    const auto inputs = boot_link_inputs(root.string(), false, "game/main.elf");
    ASSERT_FALSE(inputs.empty());
    EXPECT_EQ(inputs[0].path, fs::canonical(root / "game/main.elf").string());
    EXPECT_EQ(inputs[0].base, BOOT_EBOOT);
    EXPECT_FALSE(
        classify_module_path(root.string(), (root / "game/dependency.prx").string()).permitted())
        << "explicit main selection must not admit arbitrary dependent modules";
    write_guest_elf(root / "fakelib/main.elf");
    write_guest_elf(root / "game/libSceReplacement.elf");
    EXPECT_TRUE(resolve_guest_executable(root.string(), "fakelib/main.elf", &error).empty());
    EXPECT_TRUE(
        resolve_guest_executable(root.string(), "game/libSceReplacement.elf", &error).empty());
}

#ifndef _WIN32
TEST_F(LoadExec, SymlinksCannotEscapeOrAliasPlatformReplacements) {
    const fs::path outside = root.parent_path() / "outside.elf";
    write_guest_elf(outside);
    fs::create_symlink(outside, root / "game/escape.elf");
    write_guest_elf(root / "fakelib/main.elf");
    fs::create_symlink(root / "fakelib/main.elf", root / "game/alias.elf");
    unsigned launches = 0;
    set_loadexec_backend([&](const LoadExecRequest&) {
        ++launches;
        return kInternal;
    });
    EXPECT_EQ(invoke("game/escape.elf"), kParameter);
    EXPECT_EQ(invoke("game/alias.elf"), kParameter);
    EXPECT_EQ(launches, 0u);
    fs::remove(outside);
}
#endif
}   // namespace
