// #4090: initial Linux stubs must fit before mapping, writing or replacing the dispatcher.
// A last 32-byte return-hook stub at stride 24 otherwise writes eight bytes into a neighbour.
// The neighbour here is our writable sentinel, so the old defect is observable without a fault.
#include "host/image/exec_image.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace prosper;

namespace {
constexpr size_t kPage = 4096;
constexpr uint8_t kSentinel = 0xA5;
constexpr uint64_t kResult = 0x123456789ABCDEF0ULL;
const std::string kPlain = "initial-stub-plain";
const std::string kHooked = "initial-stub-hooked";
unsigned handler_calls = 0, hook_calls = 0;
unsigned hook_handler_calls = 0;

uint64_t handler(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
    ++handler_calls;
    return kResult;
}
void hook() {
    hook_handler_calls = handler_calls;
    ++hook_calls;
}

struct MappingProbe {
    void* address = nullptr;
    size_t length = 0;
    unsigned requests = 0, successes = 0;
    bool released = false, owned = false, inject_failure = false;
    int release_result = 0;
};
thread_local MappingProbe* active_probe = nullptr;

// Keep the body reserved until the exact backend request reaches the wrapper. Only that request
// may release it; cleanup never unmaps a hole that a failed mmap left available to another owner.
class OwnedRegion {
public:
    explicit OwnedRegion(size_t body) : body_(body) {
        bytes_ = static_cast<uint8_t*>(mmap(nullptr, body + kPage, PROT_READ | PROT_WRITE,
                                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (bytes_ == MAP_FAILED) return;
        std::memset(bytes_ + body, kSentinel, kPage);
        probe = {bytes_, body, 0, 0, false, true, false, 0};
        if (mprotect(bytes_, body, PROT_NONE) != 0) {
            munmap(bytes_, body + kPage);
            bytes_ = static_cast<uint8_t*>(MAP_FAILED);
            probe.owned = false;
        }
    }
    ~OwnedRegion() {
        if (bytes_ == MAP_FAILED) return;
        if (probe.owned) munmap(bytes_, body_);
        munmap(bytes_ + body_, kPage);
    }
    OwnedRegion(const OwnedRegion&) = delete;
    OwnedRegion& operator=(const OwnedRegion&) = delete;
    bool valid() const { return bytes_ != MAP_FAILED; }
    uint64_t base() const { return reinterpret_cast<uintptr_t>(bytes_); }
    const uint8_t* bytes() const { return bytes_; }
    const uint8_t* sentinel() const { return bytes_ + body_; }
    bool sentinel_intact() const {
        return std::all_of(bytes_ + body_, bytes_ + body_ + kPage,
                           [](uint8_t byte) { return byte == kSentinel; });
    }
    MappingProbe probe;

private:
    size_t body_;
    uint8_t* bytes_ = static_cast<uint8_t*>(MAP_FAILED);
};

bool observed_install(const std::vector<ImportSlot>& slots, uint64_t base, uint64_t stride,
                      MappingProbe& probe, std::string& error) {
    struct Scope {
        explicit Scope(MappingProbe& value) { active_probe = &value; }
        ~Scope() { active_probe = nullptr; }
    } scope(probe);
    return install_stubs(slots, base, stride, &error);
}

class InitialStubInstall : public ::testing::Test {
protected:
    void SetUp() override {
        const char* original = std::getenv("PROSPER_NO_GUEST_FS");
        had_opt_out_ = original != nullptr;
        if (original) opt_out_ = original;
        ASSERT_EQ(sysconf(_SC_PAGESIZE), static_cast<long>(kPage)) << "fixture page geometry";
        ASSERT_EQ(setenv("PROSPER_NO_GUEST_FS", "1", 1), 0) << "select host stubs";
        const TlsModuleDesc empty{};
        guest_tls_set_templates(&empty, 1);
        ASSERT_FALSE(guest_tls_enabled()) << "configure host stub mode";
        Hle::register_fn(kPlain, handler, "initial fixture plain");
        Hle::register_fn(kHooked, handler, "initial fixture hooked", hook);
        ASSERT_EQ(Hle::lookup(kHooked), handler) << "real hook handler registration";
        ASSERT_EQ(Hle::return_hook_of(kHooked), hook) << "real return hook registration";
        handler_calls = hook_calls = 0;
        hook_handler_calls = 0;
        reset_call_log();
    }
    void TearDown() override {
        active_probe = nullptr;
        dispatch_grow_slots(nullptr);
        reset_call_log();
        if (had_opt_out_)
            setenv("PROSPER_NO_GUEST_FS", opt_out_.c_str(), 1);
        else
            unsetenv("PROSPER_NO_GUEST_FS");
        const TlsModuleDesc empty{};
        guest_tls_set_templates(&empty, 1);
    }

private:
    bool had_opt_out_ = false;
    std::string opt_out_;
};
} // namespace

// Target-only GNU --wrap=mmap observes all six operands. Unrelated allocations pass through.
extern "C" void* __real_mmap(void*, size_t, int, int, int, off_t);
extern "C" void* __wrap_mmap(void* address, size_t length, int prot, int flags, int fd,
                             off_t offset) {
    MappingProbe* probe = active_probe;
    if (!probe || address != probe->address || length != probe->length ||
        prot != (PROT_READ | PROT_WRITE | PROT_EXEC) ||
        flags != (MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE) || fd != -1 || offset != 0)
        return __real_mmap(address, length, prot, flags, fd, offset);
    ++probe->requests;
    if (probe->inject_failure) {
        errno = ENOMEM;
        return MAP_FAILED;
    }
    if (!probe->released && probe->owned) {
        probe->release_result = munmap(address, length);
        if (probe->release_result != 0) return MAP_FAILED;
        probe->released = true;
        probe->owned = false;
    }
    void* result = __real_mmap(address, length, prot, flags, fd, offset);
    if (result == address) {
        ++probe->successes;
        probe->owned = true;
    }
    return result;
}

TEST_F(InitialStubInstall, LateHookRefusalPreservesSentinelAndDispatcherThenRecovers) {
    OwnedRegion prefix(kPage), table(3 * kPage);
    ASSERT_TRUE(prefix.valid() && table.valid()) << "owned prefix and four-page boundary fixture";
    std::vector<ImportSlot> baseline{{"fixture", "initial-unimplemented"}};
    std::string error;
    ASSERT_TRUE(observed_install(baseline, prefix.base(), 24, prefix.probe, error)) << error;
    ASSERT_EQ(prefix.probe.successes, 1U) << "actual baseline mapping";
    ASSERT_EQ(invoke_stub(0), 0U) << "baseline unresolved stub executes";
    const std::vector<uint32_t> first_call{0};
    ASSERT_EQ(call_order(), first_call) << "baseline first-seen census";
    std::array<uint8_t, kPage> prefix_bytes{};
    std::memcpy(prefix_bytes.data(), prefix.bytes(), kPage);

    std::vector<ImportSlot> slots(512, ImportSlot{"fixture", kPlain});
    slots.back().nid = kHooked;
    EXPECT_FALSE(observed_install(slots, table.base(), 24, table.probe, error))
        << "oversized last hook is refused";
    EXPECT_EQ(error, "generated import stub exceeds stub_size") << "specific stride refusal";
    EXPECT_EQ(table.probe.requests, 0U) << "oversized initial table maps no backing";
    EXPECT_TRUE(table.sentinel_intact()) << "last hook never writes the adjacent eight bytes";
    EXPECT_TRUE(std::all_of(table.sentinel() + 8, table.sentinel() + kPage, [](uint8_t byte) {
        return byte == kSentinel;
    })) << "boundary fixture preserves the sentinel beyond the eight-byte spill";
    EXPECT_EQ(stub_addr(0), prefix.base()) << "refusal preserves the installed stub base";
    EXPECT_EQ(std::memcmp(prefix.bytes(), prefix_bytes.data(), kPage), 0)
        << "refusal preserves actual prefix bytes";
    EXPECT_EQ(invoke_stub(0), 0U) << "installed unresolved prefix remains callable";
    EXPECT_EQ(call_order(), first_call) << "refusal preserves dispatcher first-seen counts";

    // Same geometry and address; stride32 would overlap the still-owned sentinel page.
    slots.back().nid = kPlain;
    error.clear();
    const bool recovered = observed_install(slots, table.base(), 24, table.probe, error);
    EXPECT_TRUE(recovered) << "same-address fitting retry succeeds: " << error;
    EXPECT_EQ(table.probe.requests, 1U) << "only the fitting retry requests backing";
    EXPECT_EQ(table.probe.successes, 1U) << "fitting retry owns exactly three pages";
    EXPECT_TRUE(table.sentinel_intact()) << "retry preserves the adjacent owner";
    if (recovered) {
        EXPECT_EQ(stub_addr(511), table.base() + 511 * 24) << "last fitting slot is published";
        EXPECT_EQ(invoke_stub(511), kResult) << "recovered last slot executes the real handler";
    }
}

TEST_F(InitialStubInstall, PlainAndUnimplementedImportsFit24) {
    OwnedRegion table(kPage);
    ASSERT_TRUE(table.valid()) << "owned fitting table";
    const std::vector<ImportSlot> slots{{"fixture", kPlain}, {"fixture", "initial-unimplemented"}};
    std::string error;
    ASSERT_TRUE(observed_install(slots, table.base(), 24, table.probe, error)) << error;
    ASSERT_EQ(table.probe.successes, 1U) << "real fitting mapping";
    EXPECT_EQ(invoke_stub(0), kResult) << "plain stride24 handler result";
    EXPECT_EQ(invoke_stub(1), 0U) << "unimplemented stride24 handler result";
    EXPECT_EQ(handler_calls, 1U) << "plain handler executes once";
    EXPECT_EQ(hook_calls, 0U) << "plain handler has no return hook";
    EXPECT_EQ(call_order(), (std::vector<uint32_t>{1})) << "published unresolved slot index";
    EXPECT_TRUE(table.sentinel_intact()) << "fitting table preserves adjacent owner";
}

TEST_F(InitialStubInstall, ReturnHookFits32AndRunsAfterHandler) {
    OwnedRegion table(kPage);
    ASSERT_TRUE(table.valid()) << "owned hook table";
    const std::vector<ImportSlot> slots{{"fixture", kHooked}};
    std::string error;
    ASSERT_TRUE(observed_install(slots, table.base(), 32, table.probe, error)) << error;
    ASSERT_EQ(table.probe.successes, 1U) << "real hook mapping";
    EXPECT_EQ(invoke_stub(0), kResult) << "hook preserves the handler return value";
    EXPECT_EQ(handler_calls, 1U) << "hooked handler executes once";
    EXPECT_EQ(hook_calls, 1U) << "registered return hook executes once";
    EXPECT_EQ(hook_handler_calls, 1U) << "return hook observes the completed handler";
    EXPECT_EQ(table.bytes()[30], 0xFF) << "32-byte hook ends with an indirect jump";
    EXPECT_EQ(table.bytes()[31], 0xE0) << "hook's final byte fits exactly";
    EXPECT_TRUE(table.sentinel_intact()) << "hook table preserves adjacent owner";
}

TEST_F(InitialStubInstall, GuestFsEmissionFits96AndRejectsShortStrideBeforeMapping) {
    ASSERT_EQ(unsetenv("PROSPER_NO_GUEST_FS"), 0) << "select guest-FS emission";
    const TlsModuleDesc empty{};
    guest_tls_set_templates(&empty, 1);
    ASSERT_TRUE(guest_tls_enabled()) << "configure guest-FS emission";
    OwnedRegion table(kPage), rejected(kPage);
    ASSERT_TRUE(table.valid() && rejected.valid()) << "owned guest-FS emission tables";
    const std::vector<ImportSlot> slots{
        {"fixture", kPlain}, {"fixture", "initial-unimplemented"}, {"fixture", kHooked}};
    std::string error;
    ASSERT_TRUE(observed_install(slots, table.base(), 96, table.probe, error)) << error;
    ASSERT_EQ(table.probe.successes, 1U) << "real guest-FS emission mapping";
    EXPECT_EQ(table.bytes()[0], 0x49) << "guest-FS stub starts by loading the handler";
    EXPECT_EQ(table.bytes()[1], 0xBA) << "guest-FS stub uses r10";
    EXPECT_EQ(table.bytes()[192 + 31], 0xE0) << "guest-FS return-hook stub fits its slot";
    EXPECT_FALSE(observed_install(slots, rejected.base(), 24, rejected.probe, error))
        << "nonempty guest-FS table rejects stride24";
    EXPECT_EQ(error, "stub_size too small for guest-%fs swap stub (need >= 96)")
        << "specific guest-FS stride refusal";
    EXPECT_EQ(rejected.probe.requests, 0U) << "short guest-FS stride maps no backing";
    EXPECT_EQ(stub_addr(2), table.base() + 192) << "short-stride refusal preserves publication";
    EXPECT_TRUE(table.sentinel_intact() && rejected.sentinel_intact())
        << "guest-FS emission and refusal preserve adjacent owners";
    // No guest FS is activated and no FSGSBASE instruction is executed by this byte-only control.
}

TEST_F(InitialStubInstall, Empty24RemainsValidWithGuestFsButMinimumStrideStillApplies) {
    ASSERT_EQ(unsetenv("PROSPER_NO_GUEST_FS"), 0) << "select guest-FS empty-table mode";
    const TlsModuleDesc empty{};
    guest_tls_set_templates(&empty, 1);
    ASSERT_TRUE(guest_tls_enabled()) << "configure guest-FS empty-table mode";
    const std::vector<ImportSlot> slots;
    MappingProbe probe{nullptr, 0};
    std::string error;
    EXPECT_TRUE(observed_install(slots, 0, 24, probe, error)) << "empty stride24 remains accepted";
    EXPECT_EQ(probe.requests, 0U) << "empty table never requests a zero-size mmap";
    EXPECT_FALSE(observed_install(slots, 0, 23, probe, error))
        << "minimum stride applies to empty tables";
    EXPECT_EQ(error, "stub_size too small (need >= 24)") << "specific minimum-stride refusal";
}

TEST_F(InitialStubInstall, GeometryRefusalsPrecedeMapping) {
    const std::vector<ImportSlot> slots{{"fixture", kPlain}, {"fixture", kPlain}};
    MappingProbe probe{nullptr, 0};
    probe.inject_failure = true;
    std::string error;
    EXPECT_FALSE(observed_install(slots, 0, uint64_t{1} << 63, probe, error))
        << "multiplication overflow refuses a tiny two-slot input";
    EXPECT_EQ(error, "import stub table exceeds the stub aperture") << "specific aperture refusal";
    EXPECT_EQ(probe.requests, 0U) << "overflowed geometry never maps zero bytes";

    probe.address = reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - kPage + 1);
    probe.length = kPage;
    EXPECT_FALSE(
        observed_install(slots, reinterpret_cast<uintptr_t>(probe.address), 24, probe, error))
        << "rounded mapping end overflow refuses before host mapping";
    EXPECT_EQ(error, "import stub table address overflows") << "specific address-overflow refusal";
    EXPECT_EQ(probe.requests, 0U) << "address overflow never reaches the mapper";
}
