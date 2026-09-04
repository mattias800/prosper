// test_image_descriptor_class_lookup — an image op must find the image resource at its SRSRC even
// when a buffer-class resource shares the same lookup key, and must still refuse a resource whose
// class the operation cannot use.
//
// The three provenance lookups a MIMG uses (`by_fetch_pc`, `by_srt_offset`, `by_sgpr_base`) are
// first-match-wins AND class-blind. The resolver took the first hit and then discarded it on class
// afterwards, which is not the same as searching for an acceptable one:
//
//   * an image resource sitting BEHIND a buffer-class resource at the same key was unreachable.
//     Key collisions are ordinary, not hypothetical -- a live Stray (PPSA02101) vertex stage carries
//     a ConstantBuffer and two VertexBuffers all at sgpr_base 8 (#3126);
//   * a wrong-class hit on the SRT route left `res` non-null, so the `!res` guard on the SGPR
//     fallback never fired and a route that could have resolved was never tried.
//
// Both shapes ended as `[mimg-unresolved] ... key_res=null pc_res=null`, which reads as "the
// descriptor is absent" while the descriptor was present the whole time. Same signature as #1634
// (The Oregon Trail), so this is cross-title. Pure (no Vulkan), so it runs in CI.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

// MIMG word1: [7:0]=VADDR, [15:8]=VDATA, [20:16]=SRSRC/4, [25:21]=SSAMP/4.
constexpr uint32_t mimg_word1(uint32_t vaddr, uint32_t vdata, uint32_t srsrc, uint32_t ssamp) {
    return vaddr | (vdata << 8) | ((srsrc / 4u) << 16) | ((ssamp / 4u) << 21);
}

// Word0 encodings taken verbatim from live guest streams rather than synthesized: `f0800f08` is the
// image_sample shape test_direct_descriptor_copy also uses (op 0x20, DMASK=0xf), and `f0000108` is
// the exact IMAGE_LOAD word The Oregon Trail's rejected pixel shader submits at pc=34 (#1634).
constexpr uint32_t kImageSampleDmask15 = 0xF0800F08u;   // op 0x20, DMASK=0xf -> needs a Texture
constexpr uint32_t kImageLoadDmask1    = 0xF0000108u;   // op 0x00, DMASK=0x1 -> Texture OR StorageImage
constexpr uint32_t kEndpgm             = 0xBF810000u;

ShaderResource make_texture(uint32_t sgpr_base) {
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture;
    texture.binding = 4;
    texture.img_dim = 1;
    texture.width = texture.height = 4;
    texture.sgpr_base = sgpr_base;
    texture.sampler_sgpr_base = 16;
    return texture;
}

ShaderResource make_storage_image(uint32_t sgpr_base) {
    ShaderResource image{};
    image.cls = ResourceClass::StorageImage;
    image.binding = 5;
    image.img_dim = 1;
    image.width = image.height = 4;
    image.format = DataFormat::Unorm8;
    image.num_components = 4;
    image.sgpr_base = sgpr_base;
    return image;
}

// The shadowing resource: a four-dword buffer keyed at the same SGPR the eight-dword image
// descriptor occupies. This is what the front half emits when a user-data slot is claimed as a V#.
ShaderResource make_shadowing_buffer(uint32_t sgpr_base) {
    ShaderResource buffer{};
    buffer.cls = ResourceClass::ConstantBuffer;
    buffer.binding = 2;
    buffer.gpu_addr = 0x3010000000ull;
    buffer.size = 4096;
    buffer.stride = 16;
    buffer.format = DataFormat::Float32;
    buffer.num_components = 4;
    buffer.sgpr_base = sgpr_base;
    return buffer;
}

ComputeShaderConfig direct_texture_config() {
    ComputeShaderConfig config;
    config.user_sgprs.resize(20);       // s0..s19 are entry-time user data
    return config;
}

bool recompiles(const std::vector<uint32_t>& code, const ShaderResourceTable& rt) {
    return !recompile_compute(code.data(), code.size(), &rt, direct_texture_config()).empty();
}

}  // namespace

int main() {
    printf("== test_image_descriptor_class_lookup ==\n");

    const std::vector<uint32_t> sample_s8 = {
        kImageSampleDmask15, mimg_word1(0, 0, /*srsrc=*/8, /*ssamp=*/16), kEndpgm,
    };
    const std::vector<uint32_t> load_s8 = {
        kImageLoadDmask1, mimg_word1(0, 0, /*srsrc=*/8, /*ssamp=*/16), kEndpgm,
    };

    // --- Positive control, built independently of the collision case --------------------------
    // The same op and the same descriptor with NOTHING else at s8. Without this arm a passing
    // collision arm would only show that the harness compiles something, not that it compiles the
    // case under test; and a failing one could be blamed on the encoding rather than the lookup.
    {
        ShaderResourceTable rt;
        rt.resources.push_back(make_texture(8));
        CHECK(recompiles(sample_s8, rt),
              "control: image_sample resolves a direct T# at s8 with no colliding resource");
    }

    // --- The defect: an image resource BEHIND a buffer at the same key ------------------------
    // Order matters and is the whole point: the buffer is pushed first, so a first-match lookup
    // returns it. Before the fix this recompiled to zero words and every draw using the stage was
    // discarded.
    {
        ShaderResourceTable rt;
        rt.resources.push_back(make_shadowing_buffer(8));
        rt.resources.push_back(make_texture(8));
        CHECK(rt.by_sgpr_base(8) != nullptr &&
                  rt.by_sgpr_base(8)->cls == ResourceClass::ConstantBuffer,
              "class-blind by_sgpr_base still returns the shadowing ConstantBuffer (why the filter exists)");
        CHECK(rt.image_by_sgpr_base(8, ImageResourceRequirement::SampledOnly) != nullptr &&
                  rt.image_by_sgpr_base(8, ImageResourceRequirement::SampledOnly)->cls ==
                      ResourceClass::Texture,
              "image_by_sgpr_base searches past the shadowing buffer to the Texture");
        CHECK(recompiles(sample_s8, rt),
              "image_sample resolves its T# even though a ConstantBuffer shares sgpr_base 8");
    }

    // --- MUTATION ARM: the filter must still REFUSE a class the operation cannot use ----------
    // Removing the class check entirely would satisfy every arm above. These arms fail if it is,
    // so "accept anything at the key" cannot masquerade as the fix.
    {
        ShaderResourceTable rt;
        rt.resources.push_back(make_shadowing_buffer(8));
        CHECK(rt.image_by_sgpr_base(8, ImageResourceRequirement::SampledOnly) == nullptr,
              "a lone ConstantBuffer at s8 is not offered to a sampled image op");
        CHECK(!recompiles(sample_s8, rt),
              "image_sample whose only s8 resource is a ConstantBuffer stays unresolved");
    }
    {
        ShaderResourceTable rt;
        rt.resources.push_back(make_texture(8));
        CHECK(rt.image_by_sgpr_base(8, ImageResourceRequirement::StorageOnly) == nullptr,
              "a Texture is not offered to a storage-only op (image_store / integer image atomic)");
        CHECK(rt.image_by_sgpr_base(8, ImageResourceRequirement::SampledOnly) != nullptr,
              "the same Texture is still offered to a sampled op");
    }
    {
        ShaderResourceTable rt;
        rt.resources.push_back(make_storage_image(8));
        CHECK(rt.image_by_sgpr_base(8, ImageResourceRequirement::SampledOnly) == nullptr,
              "a StorageImage is not offered to a sampled op");
        CHECK(rt.image_by_sgpr_base(8, ImageResourceRequirement::StorageOnly) != nullptr,
              "the same StorageImage is offered to a storage-only op");
    }

    // --- IMAGE_LOAD accepts either image class, and must still search past a buffer -----------
    {
        ShaderResourceTable rt;
        rt.resources.push_back(make_shadowing_buffer(8));
        rt.resources.push_back(make_storage_image(8));
        CHECK(rt.image_by_sgpr_base(8, ImageResourceRequirement::Either) != nullptr &&
                  rt.image_by_sgpr_base(8, ImageResourceRequirement::Either)->cls ==
                      ResourceClass::StorageImage,
              "IMAGE_LOAD's Either requirement reaches a StorageImage behind a buffer");
        CHECK(recompiles(load_s8, rt),
              "image_load resolves a StorageImage that shares sgpr_base 8 with a ConstantBuffer");
    }

    // --- The same collision on the other two provenance routes --------------------------------
    // by_srt_offset's version is the sharper one: a wrong-class hit there did not merely fail, it
    // left `res` non-null and so SUPPRESSED the SGPR fallback underneath it.
    {
        ShaderResourceTable rt;
        ShaderResource keyed_buffer = make_shadowing_buffer(0xFFFFFFFFu);
        keyed_buffer.srt_offset = 0x20;
        ShaderResource keyed_image = make_storage_image(0xFFFFFFFFu);
        keyed_image.srt_offset = 0x20;
        rt.resources.push_back(keyed_buffer);
        rt.resources.push_back(keyed_image);
        CHECK(rt.by_srt_offset(0x20) != nullptr &&
                  rt.by_srt_offset(0x20)->cls == ResourceClass::ConstantBuffer,
              "class-blind by_srt_offset returns the shadowing ConstantBuffer");
        CHECK(rt.image_by_srt_offset(0x20, ImageResourceRequirement::StorageOnly) != nullptr &&
                  rt.image_by_srt_offset(0x20, ImageResourceRequirement::StorageOnly)->cls ==
                      ResourceClass::StorageImage,
              "image_by_srt_offset searches past it to the StorageImage");
        CHECK(rt.image_by_srt_offset(0x20, ImageResourceRequirement::SampledOnly) == nullptr,
              "and still refuses that StorageImage for a sampled op");
    }
    {
        // fetch_pc collision is real, not invented: the const-fold's piggyback path attaches a
        // consuming instruction's pc to an EXISTING ConstantBuffer describing the same bytes, so a
        // buffer and a fold-recovered T# can carry the same pc.
        ShaderResourceTable rt;
        ShaderResource pc_buffer = make_shadowing_buffer(0xFFFFFFFFu);
        pc_buffer.fetch_pc = 50;
        ShaderResource pc_texture = make_texture(0xFFFFFFFFu);
        pc_texture.fetch_pc = 50;
        rt.resources.push_back(pc_buffer);
        rt.resources.push_back(pc_texture);
        CHECK(rt.by_fetch_pc(50) != nullptr &&
                  rt.by_fetch_pc(50)->cls == ResourceClass::ConstantBuffer,
              "class-blind by_fetch_pc returns the shadowing ConstantBuffer");
        CHECK(rt.image_by_fetch_pc(50, ImageResourceRequirement::SampledOnly) != nullptr &&
                  rt.image_by_fetch_pc(50, ImageResourceRequirement::SampledOnly)->cls ==
                      ResourceClass::Texture,
              "image_by_fetch_pc searches past it to the Texture");
        CHECK(rt.image_by_fetch_pc(51, ImageResourceRequirement::SampledOnly) == nullptr,
              "an unrelated pc still resolves nothing");
    }

    // --- The shared acceptance predicate is exactly the operation-class rule -------------------
    // Stated once so the lookup and the resolver's own validation cannot drift apart; the project
    // has already paid for that drift once (#2275).
    {
        using R = ImageResourceRequirement;
        CHECK(image_resource_class_satisfies(ResourceClass::Texture, R::SampledOnly) &&
                  !image_resource_class_satisfies(ResourceClass::StorageImage, R::SampledOnly) &&
                  !image_resource_class_satisfies(ResourceClass::ConstantBuffer, R::SampledOnly),
              "SampledOnly admits exactly Texture");
        CHECK(image_resource_class_satisfies(ResourceClass::StorageImage, R::StorageOnly) &&
                  !image_resource_class_satisfies(ResourceClass::Texture, R::StorageOnly) &&
                  !image_resource_class_satisfies(ResourceClass::VertexBuffer, R::StorageOnly),
              "StorageOnly admits exactly StorageImage");
        CHECK(image_resource_class_satisfies(ResourceClass::Texture, R::Either) &&
                  image_resource_class_satisfies(ResourceClass::StorageImage, R::Either) &&
                  !image_resource_class_satisfies(ResourceClass::ConstantBuffer, R::Either) &&
                  !image_resource_class_satisfies(ResourceClass::Sampler, R::Either),
              "Either admits both image classes and no buffer class");
    }

    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
