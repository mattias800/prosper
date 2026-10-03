// test_shader_key_dst_sel -- a resource's DST_SEL is part of the compiled-shader key exactly where
// the emitted SPIR-V reads it.
//
// Storage-image stores route VDATA through the destination T#'s DST_SEL (#3549), and MUBUF format
// fetches route returned channels through the V#'s. Two descriptors that differ only in that word
// therefore need different modules: an Alpha8 atlas store reusing a module compiled for an identity
// destination would write .x, and Hollow Knight: Silksong's menu text would be blank again with every
// execution test still green. A texture's swizzle is applied on the Vulkan view instead, so it must
// NOT split modules.
#include "gpu/execute/shader_cache_internal.hpp"

#include <gtest/gtest.h>

using namespace prosper::gpu;

namespace {

ShaderResource resource(ResourceClass cls, std::initializer_list<uint32_t> selectors) {
    ShaderResource r{};
    r.cls = cls;
    uint32_t k = 0;
    for (const uint32_t s : selectors) r.swizzle[k++] = s;
    return r;
}

ShaderCompileKey key_with(uint32_t dst_sel) {
    ShaderCompileKey key;
    key.stage = ShaderProgramStage::Compute;
    key.code = std::make_shared<const std::vector<uint32_t>>(
        std::initializer_list<uint32_t>{0xbf810000u});
    key.code_hash = hash_shader_code(*key.code);
    ShaderResourceCompileKey compiled;
    compiled.cls = static_cast<uint32_t>(ResourceClass::StorageImage);
    compiled.emitted_dst_sel = dst_sel;
    key.resources.push_back(compiled);
    key.cached_hash = ShaderCompileKeyHash::compute(key);
    return key;
}

}  // namespace

TEST(ShaderKeyDstSel, StorageAndFormatFetchSelectorsAreKeyed) {
    const uint32_t identity = compile_key_dst_sel(resource(ResourceClass::StorageImage, {4, 5, 6, 7}));
    const uint32_t alpha8 = compile_key_dst_sel(resource(ResourceClass::StorageImage, {0, 0, 0, 4}));
    EXPECT_NE(identity, alpha8);
    EXPECT_NE(compile_key_dst_sel(resource(ResourceClass::VertexBuffer, {4, 5, 6, 7})),
              compile_key_dst_sel(resource(ResourceClass::VertexBuffer, {6, 5, 4, 7})));
}

TEST(ShaderKeyDstSel, TextureSwizzleDoesNotSplitModules) {
    EXPECT_EQ(compile_key_dst_sel(resource(ResourceClass::Texture, {0, 0, 0, 4})), 0u);
    EXPECT_EQ(compile_key_dst_sel(resource(ResourceClass::Texture, {4, 5, 6, 7})), 0u);
}

TEST(ShaderKeyDstSel, KeysDifferingOnlyInDstSelAreDistinct) {
    const uint32_t identity = compile_key_dst_sel(resource(ResourceClass::StorageImage, {4, 5, 6, 7}));
    const uint32_t alpha8 = compile_key_dst_sel(resource(ResourceClass::StorageImage, {0, 0, 0, 4}));
    const ShaderCompileKey a = key_with(identity);
    const ShaderCompileKey b = key_with(alpha8);
    EXPECT_FALSE(a == b);
    EXPECT_NE(ShaderCompileKeyHash::compute_impl<true>(a),
              ShaderCompileKeyHash::compute_impl<true>(b));
    EXPECT_NE(ShaderCompileKeyHash::compute_impl<false>(a),
              ShaderCompileKeyHash::compute_impl<false>(b));
    EXPECT_TRUE(a == key_with(identity));
}
