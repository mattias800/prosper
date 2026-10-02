// #3745: CachedShader must MOVE, not copy, into ShaderCache::entries.
//
// CachedShader hand-writes its copy pair because std::atomic<uint64_t> last_use is not copyable.
// A user-declared copy constructor suppresses the implicit move pair, so before the fix every
// `entries.emplace(key, std::move(value))` silently bound to the copy constructor and bumped the
// SPIR-V shared_ptr's refcount instead of transferring it.
//
// The observable difference is exactly that refcount, so that is what this asserts: after a move the
// source no longer owns the words and the destination is their sole owner. Against the unfixed
// header the refcount checks and the trait checks all fail (a move silently degrades to a copy); the
// copy checks pass either way and pin that the copy pair still behaves as a copy.
#include "gpu/execute/shader_cache_internal.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using prosper::gpu::CachedShader;
using prosper::gpu::SharedShaderWords;

// `failures` is main()'s local; CHECK is only used there.
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } \
    else { std::printf("ok: %s\n", message); } \
} while (0)

static CachedShader make_entry(const SharedShaderWords& words) {
    CachedShader value;
    value.spirv = words;
    value.identity = 7;
    value.last_use.store(42);
    value.bytes = 12;
    value.writes_trip_witness = true;
    prosper::gpu::FragmentCompileCase producer;
    producer.program_address = 17; // metadata transfer only, not a complete compiler fixture
    producer.float_transport = {prosper::gpu::FloatTransportProfile::ExplicitNonFinite32};
    value.fragment_case = std::make_shared<const prosper::gpu::FragmentCompileCase>(std::move(producer));
    value.fragment_arithmetic.producing_program = 23;
    value.fragment_arithmetic.source_fingerprint = 29;
    value.fragment_arithmetic.float_mode = {true, 0x31};
    value.fragment_arithmetic.record(19, prosper::gpu::FragmentArithmeticFamily::Add);
    return value;
}

static bool same_fields(const CachedShader& e, const SharedShaderWords& words) {
    return e.spirv == words && e.identity == 7 &&
           e.last_use.load() == 42 && e.bytes == 12 &&
           e.writes_trip_witness && e.fragment_case && e.fragment_case->program_address == 17 &&
           e.fragment_case->float_transport.explicit_nonfinite32() &&
           e.fragment_arithmetic.producing_program == 23 && e.fragment_arithmetic.source_fingerprint == 29 &&
           e.fragment_arithmetic.float_mode == prosper::gpu::FragmentFloatMode{true, 0x31} &&
           e.fragment_arithmetic.families == 1 && e.fragment_arithmetic.site_count == 1 &&
           e.fragment_arithmetic.sites[0].pc == 19 &&
           e.fragment_arithmetic.sites[0].family == prosper::gpu::FragmentArithmeticFamily::Add;
}

int main() {
    int failures = 0;
    // Traits: a trait query cannot distinguish "moved" from "copied through const&" for plain
    // is_move_constructible (both are true today), so ask for noexcept, which the hand-written
    // copy constructor does not have.
    CHECK(std::is_nothrow_move_constructible_v<CachedShader>, "CachedShader is nothrow move-constructible");
    CHECK(std::is_nothrow_move_assignable_v<CachedShader>, "CachedShader is nothrow move-assignable");
    CHECK(std::is_copy_constructible_v<CachedShader> && std::is_copy_assignable_v<CachedShader>,
          "CachedShader remains copyable");

    const auto words = std::make_shared<const std::vector<uint32_t>>(std::vector<uint32_t>{1, 2, 3});

    {   // Move construction transfers ownership.
        CachedShader source = make_entry(words);
        CHECK(words.use_count() == 2, "baseline: caller + entry own the words");
        CachedShader moved(std::move(source));
        CHECK(same_fields(moved, words), "move-constructed entry carries every field");
        CHECK(!source.spirv && !source.fragment_case, "move construction leaves the source without words or producer");
        CHECK(words.use_count() == 2, "move construction does not add a refcount");
    }
    CHECK(words.use_count() == 1, "entries released their reference");

    {   // Move assignment transfers ownership and releases the destination's previous words.
        const auto other = std::make_shared<const std::vector<uint32_t>>(std::vector<uint32_t>{9});
        CachedShader source = make_entry(words);
        CachedShader dest;
        dest.spirv = other;
        dest = std::move(source);
        CHECK(same_fields(dest, words), "move-assigned entry carries every field");
        CHECK(!source.spirv && !source.fragment_case, "move assignment leaves the source without words or producer");
        CHECK(words.use_count() == 2, "move assignment does not add a refcount");
        CHECK(other.use_count() == 1, "move assignment released the destination's old words");
        CachedShader& alias = dest;
        dest = std::move(alias);   // self-move must not drop the words
        CHECK(same_fields(dest, words), "self move-assignment is a no-op");
    }

    {   // The production insertion shape, emplace(key, std::move(value)); try_emplace constructs the
        // mapped value from the same rvalue, so it exercises the same constructor.
        std::unordered_map<int, CachedShader> entries;
        CachedShader value = make_entry(words);
        entries.try_emplace(1, std::move(value));
        CHECK(!value.spirv && !value.fragment_case, "inserting std::move(value) moves words and producer into the map");
        CHECK(words.use_count() == 2, "the map entry is the words' only other owner");
        CHECK(same_fields(entries.at(1), words), "the map entry carries every field");
    }

    {   // The copy pair still copies (and snapshots the atomic).
        CachedShader source = make_entry(words);
        CachedShader copied(source);
        CHECK(same_fields(copied, words) && same_fields(source, words) && copied.fragment_case == source.fragment_case,
              "copy construction duplicates without disturbing the source");
        CachedShader assigned;
        assigned = source;
        CHECK(same_fields(assigned, words) && words.use_count() == 4 && assigned.fragment_case == source.fragment_case,
              "copy assignment duplicates and shares the words");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all CachedShader move checks passed\n");
    return 0;
}
