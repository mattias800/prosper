#include "gpu/execute/shader_cache_internal.hpp"

#include <array>
#include <cstdio>
#include <unordered_map>

using namespace prosper::gpu;

namespace {
unsigned checks = 0;
unsigned failures = 0;

void check(bool condition, const char* message, unsigned first = 0, unsigned second = 0) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "[FAIL] %s (states=%u,%u)\n", message, first, second);
    }
}
}

int main() {
    constexpr std::array<FragmentFloatFlags, 5> states{{
        {}, {true, false, false}, {true, true, false},
        {true, false, true}, {true, true, true}}};
    std::array<ShaderCompileKey, states.size()> keys;
    for (unsigned i = 0; i < states.size(); ++i) {
        auto& key = keys[i];
        key.stage = ShaderProgramStage::Fragment;
        key.fragment_float_flags = states[i];
        // Unknown FLOAT_MODE remains independent of known flag authority. Every key
        // uses equal code bytes, deliberately held in different immutable allocations.
        key.code = std::make_shared<const std::vector<uint32_t>>(
            std::initializer_list<uint32_t>{0xbf810000u});
        key.code_hash = hash_shader_code(*key.code);
        key.cached_hash = ShaderCompileKeyHash::compute(key);
        check(key.fragment_float_flags.canonical(), "canonical independent flag state", i);
    }
    for (unsigned i = 0; i < states.size(); ++i) {
        for (unsigned j = 0; j < states.size(); ++j) {
            check((keys[i] == keys[j]) == (i == j), "exact key equality includes all flags", i, j);
            const auto word_i = ShaderCompileKeyHash::compute_impl<true>(keys[i]);
            const auto word_j = ShaderCompileKeyHash::compute_impl<true>(keys[j]);
            const auto byte_i = ShaderCompileKeyHash::compute_impl<false>(keys[i]);
            const auto byte_j = ShaderCompileKeyHash::compute_impl<false>(keys[j]);
            // Specific fixture sensitivity only, not a claim that hashes never collide.
            check((word_i == word_j) == (i == j), "word hash includes this flag fixture", i, j);
            check((byte_i == byte_j) == (i == j), "byte hash includes this flag fixture", i, j);
        }
    }
    std::unordered_map<ShaderCompileKey, unsigned, ShaderCompileKeyHash> entries;
    for (unsigned i = 0; i < states.size(); ++i) entries.emplace(keys[i], i);
    check(entries.size() == states.size(), "five producing flag identities coexist");
    for (unsigned i = 0; i < states.size(); ++i) {
        auto repeated = keys[i];
        repeated.code = std::make_shared<const std::vector<uint32_t>>(*keys[i].code);
        repeated.cached_hash = ShaderCompileKeyHash::compute(repeated);
        const auto hit = entries.find(repeated);
        check(hit != entries.end() && hit->second == i, "equal byte identity reuses its own key", i);
    }
    std::printf("fragment_float_flags_key: %u checks, %u failures; five authority states\n",
                checks, failures);
    return failures ? 1 : 0;
}
