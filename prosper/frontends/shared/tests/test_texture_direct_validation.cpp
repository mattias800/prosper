// Real mapping/helper and ledger/engine checks. Live renderer cache/pixels remain separate.
#include "shared/live/guest_source_read.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "diagnostics/perf/perf_alarms.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace frontend = prosper::frontend;
namespace perf = prosper::diagnostics::perf;
using Outcome = frontend::GuestSourceComparisonOutcome;
static unsigned& failures() { static unsigned value = 0; return value; }
#define CHECK(ok, label) do { \
    if (ok) std::printf("[ok] %s\n", label); \
    else { std::printf("[FAIL] %s\n", label); ++failures(); } \
} while (false)

static std::string contents(FILE* file) {
    std::rewind(file);
    std::string result;
    char buffer[4096];
    while (const size_t n = std::fread(buffer, 1, sizeof buffer, file)) result.append(buffer, n);
    return result;
}
static std::string summary(perf::AlarmEngine& engine) {
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::tmpfile(), &std::fclose);
    if (!file) { CHECK(false, "summary scratch opens"); return {}; }
    CHECK(engine.write_summary(file.get()), "real engine has a completed window");
    return contents(file.get());
}
static frontend::GuestSourceComparisonObservation compare(
        const uint8_t* expected, uint64_t address, size_t bytes, size_t required,
        bool& helper_equal, size_t& matched) {
    frontend::GuestSourceComparisonObservation observation;
    helper_equal = frontend::equal_guest_source_prefix(expected, address, bytes, matched,
        prosper::gpu::guest_readable, &observation);
    perf::note_texture_direct_validation(helper_equal && matched == required,
        observation.outcome == Outcome::BytesDiffer,
        observation.outcome == Outcome::ExpectedMissing,
        observation.memcmp_calls, observation.memcmp_extent_bytes);
    return observation;
}

int main(int argc, char** argv) {
    std::puts("CPU actual mapped-source/helper/ledger/windows; live cache and Vulkan UNVERIFIED");
    prosper::register_builtin_hle();
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    const auto reserve = prosper::Hle::lookup(prosper::nid_hash("sceKernelReserveVirtualRange"));
    if (!map || !unmap || !reserve) { CHECK(false, "real memory HLE entries exist"); return 1; }
    constexpr size_t chunk = 0x10000, span = 6 * chunk;
    uint64_t base = 0;
#ifdef _WIN32
    if (reserve(reinterpret_cast<uint64_t>(&base), span, 0, 0x4000, 0, 0) != 0 || !base) {
        CHECK(false, "real tracked reservation succeeds"); return 1;
    }
#endif
    const auto release = [unmap](uint8_t* pointer) {
        if (pointer) CHECK(unmap(reinterpret_cast<uint64_t>(pointer), span, 0, 0, 0, 0) == 0,
                           "real tracked mapping released");
    };
    std::unique_ptr<uint8_t, decltype(release)> owner(reinterpret_cast<uint8_t*>(base), release);
    if (map(reinterpret_cast<uint64_t>(&base), span, 0x2, 0, 0, 0) != 0 || !base) {
        CHECK(false, "real tracked readable mapping succeeds"); return 1;
    }
    if (!owner) owner.reset(reinterpret_cast<uint8_t*>(base));
    if (prosper_renderer_guest_mapped_readable_prefix(base, span) != span) {
        CHECK(false, "actual mapping covers complete source"); return 1;
    }
    CHECK(true, "actual mapping covers complete source");
    const uint64_t address = (base + chunk - 1) & ~(uint64_t(chunk) - 1);
    constexpr size_t bytes = 3 * chunk + 13;
    auto* source = reinterpret_cast<uint8_t*>(address);
    std::fill(source, source + bytes, 0x5a);
    std::vector<uint8_t> expected(bytes, 0x5a);
    bool equal = false;
    size_t matched = 0;
    auto sample = compare(expected.data(), address, bytes, bytes, equal, matched);
    CHECK(equal && matched == bytes, "full match preserves legacy equality and matched length");
    CHECK(sample.memcmp_calls == 4 && sample.memcmp_extent_bytes == bytes,
          "full match records all actual argument extents including partial final chunk");
    const size_t mismatch_indices[] = {0, chunk + 1, bytes - 1};
    const char* const policy_labels[] = {
        "first-chunk mismatch preserves legacy decision and zero matched bytes",
        "middle-chunk mismatch preserves legacy decision and matched prefix",
        "last-chunk mismatch preserves legacy decision and matched prefix"
    };
    const char* const extent_labels[] = {
        "first-chunk mismatch includes differing-chunk argument extent",
        "middle-chunk mismatch includes differing-chunk argument extent",
        "last-chunk mismatch includes differing-chunk argument extent"
    };
    for (size_t arm = 0; arm < 3; ++arm) {
        const size_t index = mismatch_indices[arm];
        source[index] ^= 1;
        sample = compare(expected.data(), address, bytes, bytes, equal, matched);
        source[index] ^= 1;
        CHECK(!equal && matched == (index / chunk) * chunk,
              policy_labels[arm]);
        CHECK(sample.outcome == Outcome::BytesDiffer &&
              sample.memcmp_calls == index / chunk + 1 &&
              sample.memcmp_extent_bytes == std::min(bytes, (index / chunk + 1) * chunk),
              extent_labels[arm]);
    }
    sample = compare(expected.data() + 7, address + 7, chunk + 17, chunk + 17, equal, matched);
    CHECK(equal && matched == chunk + 17 && sample.memcmp_calls == 2 &&
          sample.memcmp_extent_bytes == chunk + 17, "unaligned chunk boundaries preserve exact extent");
    sample = compare(nullptr, address, bytes, bytes, equal, matched);
    CHECK(!equal && matched == 0 && sample.outcome == Outcome::ExpectedMissing &&
          sample.memcmp_calls == 0 && sample.memcmp_extent_bytes == 0,
          "missing expected representation refuses before memcmp");
    std::fill(reinterpret_cast<uint8_t*>(base + span - 17), reinterpret_cast<uint8_t*>(base + span), 0x5a);
    sample = compare(expected.data(), base + span - 17, 32, 32, equal, matched);
    CHECK(equal && matched == 17 && sample.readable_prefix == 17 &&
          sample.outcome == Outcome::ReadablePrefixEqual && sample.memcmp_extent_bytes == 17,
          "short prefix keeps helper true but caller required length rejects");
    sample = compare(nullptr, 0, 32, 32, equal, matched);
    CHECK(equal && matched == 0 && sample.outcome == Outcome::ReadablePrefixEqual &&
          sample.memcmp_calls == 0, "unreadable prefix does not claim unreached null-expected refusal");
    sample = compare(nullptr, address, 0, 0, equal, matched);
    CHECK(equal && matched == 0 && sample.memcmp_calls == 0 && sample.memcmp_extent_bytes == 0,
          "zero-length obligation preserves accepted zero-work semantics");
    auto count = [](perf::Counter counter) { return perf::ledger().counters[static_cast<size_t>(counter)].load(); };
    const bool counters_enabled = perf::enabled();
    if (counters_enabled) {
        CHECK(count(perf::Counter::TextureDirectValidationAttempts) == 9,
              "actual comparison producer records nine direct attempts");
        CHECK(count(perf::Counter::TextureDirectValidationAcceptedPrefix) == 3 &&
              count(perf::Counter::TextureDirectValidationBytesDiffer) == 3 &&
              count(perf::Counter::TextureDirectValidationExpectedMissing) == 1 &&
              count(perf::Counter::TextureDirectValidationIncompletePrefix) == 2,
              "joined CPU observations classify original decisions exactly");
    } else {
        bool all_zero = true;
        for (const auto counter : perf::kTextureDirectValidationCounters) all_zero &= count(counter) == 0;
        CHECK(all_zero, "disabled counter mode preserves all helper/admission arms with no ledger events");
    }

    perf::EngineConfig config;
    config.log = nullptr; config.window_ns = 100;
    perf::Ledger empty;
    perf::AlarmEngine no_data(config);
    no_data.on_flip(1, empty, 60); no_data.on_flip(101, empty, 60);
    CHECK(summary(no_data).find("observer=texture-direct-validation completed-windows=1 snapshot=relaxed data=NO DATA attempts=0") != std::string::npos,
          "real empty completed window reports no direct data");
    const std::string json_path = argc > 1 ? argv[1] : "texture-direct-validation.jsonl";
    config.jsonl_path = json_path;
    perf::AlarmEngine observed(config);
    observed.on_flip(1, perf::ledger(), 60);
    compare(nullptr, address, 0, 0, equal, matched);
    observed.on_flip(101, perf::ledger(), 60);
    const auto completed = summary(observed);
    if (counters_enabled) {
        CHECK(completed.find("data=OBSERVED attempts=1 memcmp-calls=0 memcmp-extent-bytes=0 accepted-prefix=1") != std::string::npos,
              "real engine excludes boot baseline and observes zero calls");
    } else {
        CHECK(completed.find("data=NO DATA attempts=0 memcmp-calls=0 memcmp-extent-bytes=0 accepted-prefix=0") != std::string::npos,
              "disabled producer remains NO DATA despite unchanged real helper acceptance");
    }
    source[0] ^= 1; compare(expected.data(), address, bytes, bytes, equal, matched); source[0] ^= 1;
    observed.on_flip(150, perf::ledger(), 60);
    CHECK(summary(observed) == completed, "real engine excludes trailing partial observations");
    perf::Ledger independent;
    config.jsonl_path.clear();
    perf::AlarmEngine partial(config);
    partial.on_flip(1, independent, 60);
    independent.counters[static_cast<size_t>(perf::Counter::TextureDirectValidationMemcmpCalls)].store(1);
    partial.on_flip(101, independent, 60);
    CHECK(summary(partial).find("data=PARTIAL attempts=0 memcmp-calls=1") != std::string::npos,
          "independent snapshot with work but no attempts remains partial");
    std::unique_ptr<FILE, decltype(&std::fclose)> json(std::fopen(json_path.c_str(), "r"), &std::fclose);
    CHECK(json != nullptr, "real engine JSONL opens");
    if (json) {
        const auto text = contents(json.get());
        const char* attempts = counters_enabled ? "\"texture_direct_validation_attempts\":1"
                                                : "\"texture_direct_validation_attempts\":0";
        const char* data = counters_enabled ? "\"texture_direct_validation_data\":\"OBSERVED\""
                                           : "\"texture_direct_validation_data\":\"NO DATA\"";
        CHECK(text.find(attempts) != std::string::npos &&
              text.find("\"texture_direct_validation_memcmp_calls\":0") != std::string::npos &&
              text.find(data) != std::string::npos,
              "actual complete-window JSONL distinguishes observed zero-work and disabled no-data");
    }
    owner.reset();
    return failures() ? 1 : 0;
}
