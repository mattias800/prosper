#include "shared/texture/validation_census.hpp"

#include <cstdio>
#include <cstring>

using namespace prosper::frontend;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

int main() {
    using Outcome = TextureValidationOutcome;
    using Watch = TextureValidationWatch;
    TextureValidationCensus c;
    CHECK(TextureValidationCensus::band((1u << 20) - 1) == 0);
    CHECK(TextureValidationCensus::band(1u << 20) == 1);
    CHECK(TextureValidationCensus::band((8u << 20) - 1) == 1);
    CHECK(TextureValidationCensus::band(8u << 20) == 2);
    c.record(Outcome::Match, Watch::DisabledAfterDirty, 8u << 20, 8u << 20, 1.25, false, 3);
    c.record(Outcome::Match, Watch::DisabledAfterDirty, 8u << 20, 8u << 20, 2.5, false, 100);
    // A mismatch at the first chunk reports zero completed-prefix bytes, despite reading memory.
    c.record(Outcome::ExactFailure, Watch::BelowMinimum, 4096, 0, 0.25, false, 2);
    // This refusal invokes no exact comparison and contributes no comparison timer value.
    c.record(Outcome::WatchOnlyRefusal, Watch::Dirty, 1u << 20, 0, 0, true, 0);
    CHECK(c.calls() == 4);
    const auto& matches = c.bucket(Outcome::Match, Watch::DisabledAfterDirty, 2);
    CHECK(matches.calls == 2 && matches.source_bytes == 16u << 20);
    CHECK(matches.reported_validated_bytes == 16u << 20);
    CHECK(matches.validation_ms == 3.75 && matches.stability[3] == 2);
    uint64_t partition = 0, stability_partition = 0;
    for (size_t o = 0; o < TextureValidationCensus::kOutcomes; ++o)
        for (size_t w = 0; w < TextureValidationCensus::kWatches; ++w)
            for (size_t s = 0; s < TextureValidationCensus::kBands; ++s) {
                const auto& b = c.bucket(static_cast<Outcome>(o), static_cast<Watch>(w), s);
                partition += b.calls;
                for (auto count : b.stability) stability_partition += count;
            }
    CHECK(partition == 4 && stability_partition == 4);
    FILE* output = std::tmpfile();
    CHECK(output != nullptr);
    if (output) {
        c.report(output, 17, "timing-inactive");
        std::rewind(output);
        char text[4096]{};
        const size_t bytes = std::fread(text, 1, sizeof(text) - 1, output);
        CHECK(bytes > 0 && bytes < sizeof(text) - 1);
        CHECK(std::strstr(text, "scope=thread-cumulative thread=17 reason=timing-inactive calls=4"));
        CHECK(std::strstr(text, "outcome=watch-only-refusal watch=dirty size=1-to-8MiB calls=1"));
        CHECK(std::strstr(text, "reported_validated_bytes=16777216 validation_ms=3.750000"));
        CHECK(!std::strstr(text, "complete="));
        std::fclose(output);
    }
    c.reset();
    CHECK(c.calls() == 0);
    CHECK(c.bucket(Outcome::Match, Watch::DisabledAfterDirty, 2).calls == 0);
    return failures ? 1 : 0;
}
