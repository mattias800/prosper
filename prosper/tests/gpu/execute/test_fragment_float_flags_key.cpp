// test_fragment_float_flags_key — the fragment shader cache key must carry every piece of
// float-mode authority as a DISTINCT producing input.
//
// A cache key that folds two of these states together does not return a wrong image: it returns the
// other image, from a compile the guest never asked for, which is why this is a key-identity test and
// not a rendering one. Unknown FLOAT_MODE stays independent of known flag authority, so the fixtures
// cover flags-known/mode-unknown and the reverse.
//
// Every key here holds equal code bytes, deliberately in different immutable allocations, so a
// "reused key" verdict can only come from the flags and never from pointer identity or code bytes.
#include "gpu/execute/shader_cache_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace prosper::gpu;

namespace {
struct FlagState {
    FragmentFloatFlags flags;
    FragmentLaunchRsrc1 raw;
};
constexpr std::array<FlagState, 8> kStates{{
    {{}, {}},
    {{true, false, false}, {}},
    {{true, true, false}, {}},
    {{true, false, true}, {}},
    {{true, true, true}, {}},
    {{}, {true, 0}},
    {{}, {true, 1u << 29}},
    {{}, {true, (1u << 29) | (1u << 22)}},
}};

std::array<ShaderCompileKey, kStates.size()> build_keys() {
    std::array<ShaderCompileKey, kStates.size()> keys;
    for (unsigned i = 0; i < kStates.size(); ++i) {
        auto& key = keys[i];
        key.stage = ShaderProgramStage::Fragment;
        key.fragment_float_flags = kStates[i].flags;
        key.fragment_launch_rsrc1 = kStates[i].raw;
        key.code = std::make_shared<const std::vector<uint32_t>>(
            std::initializer_list<uint32_t>{0xbf810000u});
        key.code_hash = hash_shader_code(*key.code);
        key.cached_hash = ShaderCompileKeyHash::compute(key);
    }
    return keys;
}
}   // namespace

TEST(FragmentFloatFlagsKey, EveryFixtureStateIsCanonical) {
    const auto keys = build_keys();
    for (unsigned i = 0; i < kStates.size(); ++i) {
        EXPECT_TRUE(keys[i].fragment_float_flags.canonical())
            << "canonical independent flag state, fixture " << i;
        EXPECT_TRUE(keys[i].fragment_launch_rsrc1.canonical())
            << "canonical independent raw evidence state, fixture " << i;
    }
}

TEST(FragmentFloatFlagsKey, KeyEqualityIncludesEveryFlagState) {
    const auto keys = build_keys();
    for (unsigned i = 0; i < kStates.size(); ++i) {
        for (unsigned j = 0; j < kStates.size(); ++j) {
            EXPECT_EQ(keys[i] == keys[j], i == j)
                << "exact key equality includes all flags, fixtures " << i << " and " << j;
            EXPECT_EQ(ShaderCompileKeyHash::compute_impl<true>(keys[i]) ==
                          ShaderCompileKeyHash::compute_impl<true>(keys[j]),
                      i == j)
                << "word hash includes this flag fixture, fixtures " << i << " and " << j;
            EXPECT_EQ(ShaderCompileKeyHash::compute_impl<false>(keys[i]) ==
                          ShaderCompileKeyHash::compute_impl<false>(keys[j]),
                      i == j)
                << "byte hash includes this flag fixture, fixtures " << i << " and " << j;
        }
    }
}

TEST(FragmentFloatFlagsKey, EightProducingIdentitiesCoexistInOneMap) {
    const auto keys = build_keys();
    std::unordered_map<ShaderCompileKey, unsigned, ShaderCompileKeyHash> entries;
    for (unsigned i = 0; i < kStates.size(); ++i) entries.emplace(keys[i], i);
    EXPECT_EQ(entries.size(), kStates.size()) << "eight flag/raw producing identities coexist";
}

TEST(FragmentFloatFlagsKey, AnEqualByteIdentityReusesItsOwnKey) {
    const auto keys = build_keys();
    std::unordered_map<ShaderCompileKey, unsigned, ShaderCompileKeyHash> entries;
    for (unsigned i = 0; i < kStates.size(); ++i) entries.emplace(keys[i], i);
    for (unsigned i = 0; i < kStates.size(); ++i) {
        // Same bytes, a DIFFERENT allocation: the point is that the reuse comes from the hash and not
        // from the shared code pointer.
        auto repeated = keys[i];
        repeated.code = std::make_shared<const std::vector<uint32_t>>(*keys[i].code);
        repeated.cached_hash = ShaderCompileKeyHash::compute(repeated);
        const auto hit = entries.find(repeated);
        ASSERT_NE(hit, entries.end()) << "equal byte identity finds its key, fixture " << i;
        EXPECT_EQ(hit->second, i) << "equal byte identity reuses its OWN key, fixture " << i;
    }
}