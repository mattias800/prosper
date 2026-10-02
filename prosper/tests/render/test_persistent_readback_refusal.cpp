// CPU controls for the shipping persistent-color lookup/readback refusals (#3891).
// Hand-built cache state is the oracle; no observer/classifier is called directly. The positive
// arm calls find only. Fake image bits never enter Vulkan, transfer, eviction or destroy helpers.
#include "shared/device/vulkan_runtime.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace prosper::frontend {
[[noreturn]] static bool readback_test_forbid_runtime_loader(const char*) {
    std::fputs("FAIL: CPU refusal control reached Vulkan initialization\n", stderr);
    std::abort();
}
} // namespace prosper::frontend

// Keep the actual refusal branches; abort at the first initialization boundary if one regresses.
#define require_vulkan_runtime_loader readback_test_forbid_runtime_loader
#include "fixtures/render_runner.h"
#undef require_vulkan_runtime_loader

namespace {
namespace refusal = prosper::diagnostics::readback_refusal;
using refusal::ObservedBool;
using refusal::Reason;
using prosper::test::PersistentColorTargetKey;
static_assert(std::is_trivially_copyable_v<refusal::Record>);
static_assert(std::is_trivially_copyable_v<refusal::LookupObservation>);

constexpr uint64_t kAddress = 0x38910000;
constexpr uint32_t kWidth = 16, kHeight = 8;
constexpr VkFormat kRequestedFormat = VK_FORMAT_B8G8R8A8_UNORM;
// Independent expectation: this guest format uses the backend's canonical RGBA8 key.
constexpr VkFormat kCanonicalFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr PersistentColorTargetKey kKey{kAddress, kWidth, kHeight, kCanonicalFormat, 0};

int& failure_count() {
    static int count = 0;
    return count;
}

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failure_count();
    }
}

template <typename Handle> Handle synthetic_handle() {
    if constexpr (std::is_pointer_v<Handle>) return reinterpret_cast<Handle>(uintptr_t{1});
    else return static_cast<Handle>(1);
}

void clear_synthetic_cache() {
    auto& cache = prosper::test::persistent_color_target_cache();
    for (auto& [key, target] : cache) target.image = VK_NULL_HANDLE;
    cache.clear(); // Plain values/vectors; deliberately no Vulkan destruction helper.
}

refusal::Context context() {
    return {refusal::Caller::ComputeSnapshot, refusal::SubmitKnown | refusal::ProgramKnown,
            3891, 0, 0x38911234, ObservedBool::Yes, ObservedBool::No};
}

void positive_lookup(bool armed) {
    refusal::LookupObservation observed{ObservedBool::No, ObservedBool::No};
    refusal::LookupObservation unfiltered{ObservedBool::No, ObservedBool::Yes};
    bool found = false, raw_found = false, unchanged = false;
    {
        const prosper::test::BackendPersistentResourceGuard guard;
        clear_synthetic_cache();
        auto& target = prosper::test::persistent_color_target_cache()[kKey];
        target.valid = true;
        target.image = synthetic_handle<VkImage>();
        target.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        target.last_use = 71;
        const uint64_t generation = prosper::test::persistent_color_target_generation();
        found = prosper::test::find_persistent_color_target(
            kAddress, kWidth, kHeight, kRequestedFormat, true, 0,
            armed ? &observed : nullptr) == &target;
        target.valid = false;
        raw_found = prosper::test::find_persistent_color_target(
            kAddress, kWidth, kHeight, kRequestedFormat, false, 0,
            armed ? &unfiltered : nullptr) == &target;
        unchanged = target.last_use == 71 &&
                    prosper::test::persistent_color_target_generation() == generation && !target.valid;
        clear_synthetic_cache();
    }
    check(found, "positive: actual lookup returns the exact valid canonical key");
    check(unchanged, "positive: pure lookup preserves recency");
    check(observed.key_present == (armed ? ObservedBool::Yes : ObservedBool::No) &&
          observed.valid == (armed ? ObservedBool::Yes : ObservedBool::No),
          "positive: actual lookup fills only a supplied observation");
    check(raw_found && unfiltered.key_present == (armed ? ObservedBool::Yes : ObservedBool::No) &&
          unfiltered.valid == (armed ? ObservedBool::NotObserved : ObservedBool::Yes),
          "unfiltered lookup returns an invalid entry without observing validity");
}

struct Control {
    const char* name;
    Reason expected;
    const char* reason_text;
    bool entry_present;
    PersistentColorTargetKey stored_key;
    bool valid, image;
    VkImageLayout layout;
    ObservedBool key_present, observed_valid, image_present;
    bool layout_known;
    uint64_t requested_address = kAddress;
};

void check_refusal_observation(const Control& control, const refusal::Record& record,
                               uint64_t begin_us, uint64_t end_us) {
    check(record.reason == control.expected, "actual readback records the independent refusal reason");
    check(record.source_us >= begin_us && record.source_us <= end_us && record.source_us != 0,
          "actual refusal carries a source monotonic timestamp within the call");
    check(record.address == control.requested_address && record.width == kWidth && record.height == kHeight &&
          record.volume_depth == 0 && record.requested_format == static_cast<uint32_t>(kRequestedFormat),
          "refusal retains actual request identity and raw format");
    check(record.canonical_known == (control.expected != Reason::GlobalUnproven) &&
          (!record.canonical_known || record.canonical_format == static_cast<uint32_t>(kCanonicalFormat)),
          "canonical key is observed only after the earlier unproven guard");
    check(record.lookup.key_present == control.key_present &&
          record.lookup.valid == control.observed_valid && record.image_present == control.image_present &&
          record.layout_known == control.layout_known &&
          (!record.layout_known || record.layout == static_cast<uint32_t>(control.layout)),
          "short-circuit priority retains unknown for unevaluated operands");
    check(record.context.caller == refusal::Caller::ComputeSnapshot &&
          record.context.known == (refusal::SubmitKnown | refusal::ProgramKnown) &&
          record.context.submit == 3891 && record.context.draw == 0 && record.context.program == 0x38911234 &&
          record.context.frontend_gpu_valid == ObservedBool::Yes &&
          record.context.frontend_cpu_pixels == ObservedBool::No, "backend preserves caller context");
    char text[1024]{};
    const size_t size = refusal::format_record(text, sizeof(text), record);
    check(size > 0 && size < sizeof(text), "actual refusal formatter fits the bounded control buffer");
    char key[96]{};
    std::snprintf(key, sizeof(key), "key=0x%llx/16x8/requested-format=44/",
                  static_cast<unsigned long long>(control.requested_address));
    char stamp[64]{};
    std::snprintf(stamp, sizeof(stamp), "source-us=%llu clock=steady-us",
                  static_cast<unsigned long long>(record.source_us));
    check(std::strstr(text, control.reason_text) && std::strstr(text, key) &&
          std::strstr(text, stamp) &&
          std::strstr(text, control.expected == Reason::GlobalUnproven
                            ? "canonical-format=NOT_OBSERVED" : "canonical-format=37"),
          "formatter reports the independent reason, request key and clock domain");
    check(std::strstr(text, "history=UNKNOWN") && std::strstr(text, "producer-tokens=NOT_OBSERVED") &&
          std::strstr(text, "pins=NOT_OBSERVED") && std::strstr(text, "gate=NOT_OBSERVED") &&
          std::strstr(text, "final-consumer-disposition=UNKNOWN"),
          "typed failure does not invent producer, lifecycle or final-consumer history");
}

void refused_readback(const Control& control, bool armed) {
    std::printf("control: %s\n", control.name);
    refusal::Record record{};
    record.context = context();
    // A null output pointer must leave even stale caller-owned observation storage untouched.
    if (!armed) {
        record.reason = Reason::UndefinedLayout;
        record.source_us = 123;
        record.address = 0xbad;
        record.canonical_known = true;
        record.lookup = {ObservedBool::Yes, ObservedBool::No};
        record.image_present = ObservedBool::Yes;
        record.layout_known = true;
    }
    std::array<unsigned char, sizeof(record)> before{};
    std::memcpy(before.data(), &record, sizeof(record));
    std::vector<uint8_t> output{1, 2, 3};
    std::string error = "stale error";
    bool result = true, unchanged = false;
    const uint64_t begin_us = prosper::diagnostics::diag_now_us();
    {
        const prosper::test::BackendPersistentResourceGuard guard;
        clear_synthetic_cache();
        auto& cache = prosper::test::persistent_color_target_cache();
        if (control.entry_present) {
            auto& target = cache[control.stored_key];
            target.valid = control.valid;
            target.image = control.image ? synthetic_handle<VkImage>() : VK_NULL_HANDLE;
            target.layout = control.layout;
            target.last_use = 71;
            target.pin_count = 2;
        }
        const uint64_t generation = prosper::test::persistent_color_target_generation();
        const VkDeviceSize bytes = prosper::test::persistent_color_target_bytes();
        result = prosper::test::readback_persistent_color_target(
            control.requested_address, kWidth, kHeight, kRequestedFormat, output, error, 0,
            armed ? &record : nullptr);
        unchanged = cache.size() == (control.entry_present ? 1u : 0u) &&
                    prosper::test::persistent_color_target_generation() == generation &&
                    prosper::test::persistent_color_target_bytes() == bytes;
        if (control.entry_present) {
            const auto found = cache.find(control.stored_key);
            unchanged = unchanged && found != cache.end();
            if (found != cache.end()) {
                const auto& target = found->second;
                unchanged = unchanged && target.valid == control.valid &&
                    target.image == (control.image ? synthetic_handle<VkImage>() : VK_NULL_HANDLE) &&
                    target.layout == control.layout && target.last_use == 71 && target.pin_count == 2;
            }
        }
        clear_synthetic_cache();
    }
    const uint64_t end_us = prosper::diagnostics::diag_now_us();
    check(!result && output.empty(), "original failure returns false and clears output");
    check(error == (control.expected == Reason::GlobalUnproven
                    ? "Vulkan submission completion is unproven"
                    : "persistent color target is unavailable"), "original error text preserved");
    check(unchanged, "refusal preserves cache identity, validity, layout, pins, bytes and recency");
    if (!armed) {
        std::array<unsigned char, sizeof(record)> after{};
        std::memcpy(after.data(), &record, sizeof(record));
        check(before == after,
              "disabled: null observation pointer leaves sentinel untouched");
        return;
    }
    check_refusal_observation(control, record, begin_us, end_us);
}

size_t occurrences(std::string_view text, std::string_view token) {
    size_t count = 0, position = 0;
    while ((position = text.find(token, position)) != std::string_view::npos) {
        ++count;
        position += token.size();
    }
    return count;
}

void output_cap(bool armed) {
    if (!armed) return;
    FILE* file = std::tmpfile();
    check(file != nullptr, "cap control opens a scratch stream");
    if (!file) return;
    // Instrument silence control only; this is not evidence of frontend caller wiring.
    refusal::emit(refusal::Record{}, file);
    for (unsigned i = 0; i < 65; ++i) {
        refusal::Record record{};
        record.context = context();
        std::vector<uint8_t> output{1};
        std::string error;
        bool result;
        {
            const prosper::test::BackendPersistentResourceGuard guard;
            clear_synthetic_cache();
            result = prosper::test::readback_persistent_color_target(
                kAddress, kWidth, kHeight, kRequestedFormat, output, error, 0, &record);
        }
        check(!result && output.empty() && error == "persistent color target is unavailable" &&
              record.reason == Reason::ExactKeyAbsent && record.source_us != 0,
              "cap input comes from an actual absent-key readback refusal");
        refusal::emit(record, file); // Outside the resource guard; no manufactured failure Record.
    }
    check(std::fflush(file) == 0, "cap output is readable after flushing the scratch stream");
    std::rewind(file);
    std::string text;
    char buffer[1024];
    while (const size_t bytes = std::fread(buffer, 1, sizeof(buffer), file)) text.append(buffer, bytes);
    check(!std::ferror(file), "cap scratch stream read succeeds");
    std::fclose(file);
    check(occurrences(text, "[rtt-readback] source-us=") == 64 &&
          occurrences(text, "reason=exact-key-absent") == 64 &&
          occurrences(text, "record-limit=64 further-records=suppressed") == 1,
          "65 actual refusals yield 64 typed output records and one suppression notice");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "--armed") && std::strcmp(argv[1], "--unarmed"))) {
        std::fputs("usage: test_persistent_readback_refusal --armed|--unarmed\n", stderr);
        return 2;
    }
    const bool armed = !std::strcmp(argv[1], "--armed");
    refusal::initialize(); // Parse/allocate before entering any resource guard.
    check(refusal::selected(kAddress) == armed, "fresh-process selector matches registered control");
    char quiet[16] = "sentinel";
    check(refusal::format_record(quiet, sizeof(quiet), refusal::Record{}) == 0 && quiet[0] == '\0',
          "NotRefused formatter stays silent (formatter control, not production-route evidence)");
    positive_lookup(armed);
    const Control controls[] = {
        {"absent exact key", Reason::ExactKeyAbsent, "reason=exact-key-absent", false, kKey, false, false,
         VK_IMAGE_LAYOUT_UNDEFINED, ObservedBool::No, ObservedBool::NotObserved,
         ObservedBool::NotObserved, false},
        {"invalid masks missing image/layout", Reason::PresentInvalid, "reason=present-invalid", true, kKey, false, false,
         VK_IMAGE_LAYOUT_UNDEFINED, ObservedBool::Yes, ObservedBool::No,
         ObservedBool::NotObserved, false},
        {"missing image masks undefined layout", Reason::NoImage, "reason=no-image", true, kKey, true, false,
         VK_IMAGE_LAYOUT_UNDEFINED, ObservedBool::Yes, ObservedBool::Yes, ObservedBool::No, false},
        {"undefined layout with image", Reason::UndefinedLayout, "reason=undefined-layout", true, kKey, true, true,
         VK_IMAGE_LAYOUT_UNDEFINED, ObservedBool::Yes, ObservedBool::Yes, ObservedBool::Yes, true},
        {"same address, different width", Reason::ExactKeyAbsent, "reason=exact-key-absent", true,
         {kAddress, kWidth + 1, kHeight, kCanonicalFormat, 0}, true, false,
         VK_IMAGE_LAYOUT_UNDEFINED, ObservedBool::No, ObservedBool::NotObserved,
         ObservedBool::NotObserved, false},
        {"same address, different format", Reason::ExactKeyAbsent, "reason=exact-key-absent", true,
         {kAddress, kWidth, kHeight, VK_FORMAT_R8_UNORM, 0}, true, false,
         VK_IMAGE_LAYOUT_UNDEFINED, ObservedBool::No, ObservedBool::NotObserved,
         ObservedBool::NotObserved, false},
        {"same address, different volume key", Reason::ExactKeyAbsent, "reason=exact-key-absent", true,
         {kAddress, kWidth, kHeight, kCanonicalFormat, 1}, true, false,
         VK_IMAGE_LAYOUT_UNDEFINED, ObservedBool::No, ObservedBool::NotObserved,
         ObservedBool::NotObserved, false},
        {"zero ID performs no lookup", Reason::LookupNotPerformed, "reason=lookup-not-performed", false, kKey, false, false,
         VK_IMAGE_LAYOUT_UNDEFINED, ObservedBool::NotObserved, ObservedBool::NotObserved,
         ObservedBool::NotObserved, false, 0},
    };
    for (const auto& control : controls) refused_readback(control, armed);
    output_cap(armed);
    // Poison is intentionally last and never cleared. An otherwise readable fake image catches
    // a removed/reordered unproven guard at the forbidden initialization boundary.
    prosper::test::backend_mark_unproven_submission();
    refused_readback({"earlier sticky-unproven", Reason::GlobalUnproven, "reason=global-unproven", true, kKey, true, true,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ObservedBool::NotObserved,
                     ObservedBool::NotObserved, ObservedBool::NotObserved, false}, armed);
    const int failures = failure_count();
    std::printf("%s: %d failure(s); CPU observation controls establish no GPU completion/pixels\n",
                failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
