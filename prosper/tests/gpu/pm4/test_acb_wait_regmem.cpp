// #5043, found during #5037: ACB WAIT_REG_MEM has eight arguments, while DCB has nine. Reusing DCB's
// handler shifts the label/reference/mask and turns a real queue dependency into an invalid wait.
// Exercise the registered APIs, decoded packet, and CPU fold; no device or title is required.
#include "gpu/pm4/command_processor.hpp"
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace {
using namespace prosper;
using namespace prosper::gpu;

using HostHle8 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t);
using HostHle9 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                              uint64_t, uint64_t);

struct CommandBuffer {
    uint32_t* bottom;
    uint32_t* top;
    uint32_t* cursor_up;
    uint32_t* cursor_down;
    void* callback = nullptr;
    void* user_data = nullptr;
    uint32_t reserved_dw = 0;
    uint32_t pad = 0;
};

// Restore the process environment even when an assertion returns from the test early.
// NOLINTBEGIN(concurrency-mt-unsafe): this isolated CPU test changes the environment between synchronous calls.
class GfxLogEnvironment {
public:
    GfxLogEnvironment() {
        if (const auto* value = std::getenv("PROSPER_GFXLOG")) original = value;
    }
    ~GfxLogEnvironment() { set(original ? original->c_str() : nullptr); }

    static void set(const char* value) {
#ifdef _WIN32
        _putenv_s("PROSPER_GFXLOG", value ? value : "");
#else
        if (value)
            setenv("PROSPER_GFXLOG", value, 1);
        else
            unsetenv("PROSPER_GFXLOG");
#endif
    }

private:
    std::optional<std::string> original;
};
// NOLINTEND(concurrency-mt-unsafe)

class AcbWaitRegMem : public testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        acb = reinterpret_cast<HostHle8>(Hle::lookup("htn36gPnBk4"));
        dcb = reinterpret_cast<HostHle9>(Hle::lookup("VmW0Tdpy420"));
        ASSERT_NE(acb, nullptr);
        ASSERT_NE(dcb, nullptr);
        words.fill(0xdeadbeefu);
    }

    uint64_t cb_address() { return reinterpret_cast<uint64_t>(&cb); }
    uint64_t label_address() { return reinterpret_cast<uint64_t>(&label); }

    std::array<uint32_t, 10> words{};
    CommandBuffer cb{words.data(), words.data() + 9, words.data(), words.data() + 9};
    alignas(8) uint64_t label = 0;
    HostHle8 acb = nullptr;
    HostHle9 dcb = nullptr;
};

TEST_F(AcbWaitRegMem, EightArgumentLayoutPreservesFullOperandsAndPacketSize) {
    constexpr uint64_t reference = 0x123456789abcdef0ull;
    constexpr uint64_t mask = 0xfedcba9876543210ull;
    constexpr uint32_t poll = 0x2468u;
    EXPECT_EQ(acb(cb_address(), 1, 3, 0, label_address(), reference, mask, poll),
              reinterpret_cast<uint64_t>(words.data()));
    ASSERT_EQ(cb.cursor_up, words.data() + 9);
    EXPECT_EQ(words[9], 0xdeadbeefu) << "the guest reserves exactly nine dwords";
    std::vector<Pm4Command> commands;
    EXPECT_EQ(decode_pm4(words.data(), 9, commands), 9u);
    ASSERT_EQ(commands.size(), 1u);
    const auto& wait = commands.front();
    EXPECT_EQ(wait.kind, Pm4Command::Kind::WaitRegMem);
    EXPECT_TRUE(wait.wm_valid);
    EXPECT_EQ(wait.wm_addr, label_address()) << "ACB address is argument five, in r8";
    EXPECT_EQ(wait.wm_ref, reference) << "reference is argument six, in r9";
    EXPECT_EQ(wait.wm_mask, mask) << "mask is the first guest stack word";
    EXPECT_EQ(wait.wm_func, 3u);
    EXPECT_EQ(words[8], poll) << "poll interval is the second guest stack word";
}

TEST_F(AcbWaitRegMem, DcbLayoutStillReadsThreeStackWords) {
    constexpr uint64_t reference = 0x123456789abcdef0ull;
    constexpr uint64_t mask = 0xfedcba9876543210ull;
    constexpr uint32_t poll = 0x2468u;
    EXPECT_EQ(dcb(cb_address(), 1, 3, 1, 0, label_address(), reference, mask, poll),
              reinterpret_cast<uint64_t>(words.data()));
    ASSERT_EQ(cb.cursor_up, words.data() + 9);
    std::vector<Pm4Command> commands;
    EXPECT_EQ(decode_pm4(words.data(), 9, commands), 9u);
    ASSERT_EQ(commands.size(), 1u);
    EXPECT_EQ(commands.front().wm_addr, label_address());
    EXPECT_EQ(commands.front().wm_ref, reference);
    EXPECT_EQ(commands.front().wm_mask, mask);
    EXPECT_EQ(commands.front().wm_func, 3u);
    EXPECT_EQ(words[8], poll);
    EXPECT_EQ(words[9], 0xdeadbeefu);
}

TEST_F(AcbWaitRegMem, EqualMatchingLabelIsSatisfiedThroughTheRealFold) {
    label = 0x123400009abcde10ull;
    ASSERT_NE(acb(cb_address(), 1, 3, 0, label_address(), label, UINT64_MAX, 400), 0u);
    testing::internal::CaptureStderr();
    GpuState state;
    const auto count = run_command_buffer(words.data(), 9, state);
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(output.find("NOT satisfied at fold time"), std::string::npos)
        << "a matching readable label must not become the shifted reference-as-address";
    EXPECT_FALSE(last_fold_deferred());
    EXPECT_EQ(label, 0x123400009abcde10ull);
}

TEST_F(AcbWaitRegMem, EqualDifferentHighWordRemainsUnsatisfied) {
    label = 0x123400009abcde10ull;
    ASSERT_NE(acb(cb_address(), 1, 3, 0, label_address(), 0x123500009abcde10ull, UINT64_MAX, 400),
              0u);
    testing::internal::CaptureStderr();
    GpuState state;
    const auto count = run_command_buffer(words.data(), 9, state);
    const auto output = testing::internal::GetCapturedStderr();
    EXPECT_EQ(count, 1u);
    EXPECT_NE(output.find("NOT satisfied at fold time"), std::string::npos);
    EXPECT_NE(output.find("decision-value-valid=1"), std::string::npos)
        << "a real false predicate is distinct from an invalid shifted address";
    EXPECT_NE(output.find("decision-effective=0x123400009abcde10"), std::string::npos);
    EXPECT_EQ(label, 0x123400009abcde10ull);
}

TEST_F(AcbWaitRegMem, MissingBufferOrInsufficientReservationDoesNotEmit) {
    EXPECT_EQ(acb(0, 1, 3, 0, label_address(), 1, UINT64_MAX, 400), 0u);
    cb.top = cb.cursor_down = words.data() + 8;
    EXPECT_EQ(acb(cb_address(), 1, 3, 0, label_address(), 1, UINT64_MAX, 400), 0u);
    EXPECT_EQ(cb.cursor_up, words.data());
    for (const auto word : words) EXPECT_EQ(word, 0xdeadbeefu);
}

// NOLINTBEGIN(concurrency-mt-unsafe): environment controls run on one thread before each builder call.
TEST_F(AcbWaitRegMem, DiagnosticArmingAndDisarmingAreVisibleBetweenCalls) {
    GfxLogEnvironment environment;
    const auto build_and_capture = [&]() {
        cb.cursor_up = words.data();
        testing::internal::CaptureStderr();
        const auto result = acb(cb_address(), 1, 3, 0, label_address(), 1, UINT64_MAX, 400);
        const auto output = testing::internal::GetCapturedStderr();
        EXPECT_EQ(result, reinterpret_cast<uint64_t>(words.data()));
        return output;
    };

    GfxLogEnvironment::set(nullptr);
    ASSERT_EQ(std::getenv("PROSPER_GFXLOG"), nullptr);
    EXPECT_EQ(build_and_capture().find("[agc] AcbWaitRegMem"), std::string::npos);
    // Presence is the contract: even "0" enables this diagnostic, like the DCB sibling.
    GfxLogEnvironment::set("0");
    ASSERT_STREQ(std::getenv("PROSPER_GFXLOG"), "0");
    EXPECT_NE(build_and_capture().find("[agc] AcbWaitRegMem"), std::string::npos)
        << "arming after the first call must not be hidden by a process-lifetime cache";
    GfxLogEnvironment::set(nullptr);
    ASSERT_EQ(std::getenv("PROSPER_GFXLOG"), nullptr);
    EXPECT_EQ(build_and_capture().find("[agc] AcbWaitRegMem"), std::string::npos)
        << "disarming must stop the diagnostic on the next call";
}
// NOLINTEND(concurrency-mt-unsafe)
}   // namespace
