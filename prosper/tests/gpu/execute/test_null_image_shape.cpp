// test_null_image_shape — a NULL image descriptor must carry the shape the SHADER declares, and the
// validator must say so when it does not.
//
// Why this exists (#3577). When the guest binds an explicitly all-zero T#, prosper synthesizes a null
// `ShaderResource` — correctly, and for a measured reason (#2422: 11 of 18 terminal-MIMG fragment
// pairs on GTA V, 1,328 rejected draws). But the synthesized record set only five fields, so `img_dim`
// kept its struct default of 1 (2D), and the descriptor-interface validator SKIPPED the image branch
// entirely for null descriptors. A shader declaring `OpTypeImage ... Dim=3D` could therefore be handed
// a `VK_IMAGE_VIEW_TYPE_2D` dummy: a view type that does not match the SPIR-V image type it is bound
// to, which is undefined behaviour.
//
// The reason to care rather than shrug is recorded in #2422's `## Ruled out` row: under a comparable
// undefined descriptor contract, RADV happened to return the expected pixel. "It works here" is not
// evidence about this class of defect.
//
// THE KEY FACT, and the one that makes the failure reachable: the recompiler derives `OpTypeImage`'s
// Dim/Arrayed/MS from the CONSUMING MIMG INSTRUCTION's dim field (`rdna2_emit_alu.cpp` switches on
// `in.mimg_dim`), never from the descriptor. So the shader's declared shape is completely independent
// of the T# words — and with an all-zero T# there is nothing in the descriptor that could have agreed
// with it. That is also the fix: the null synthesis now carries `SrtUse::mimg_dim` into `img_dim`,
// which uses the same SQ_RSRC encoding.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_null_image_shape ==\n");

    // image_load v[0:3], v0, s[8:15] dim:3D dmask:0xf   (gfx1030; dim field = bits 5:3 = 2)
    // Paired with an explicitly ALL-ZERO T# at s[8:15] -- the exact shape the null-bind path exists
    // for -- so nothing in the descriptor can supply a dimension.
    const uint32_t code_3d[] = { 0xf0000f10u, 0x00000200u, 0xbf810000u };

    // The runtime table as the executor's null synthesis now produces it: gpu_addr/size zero (which is
    // how `validate_shader_resources` recognises an explicit null), and the shape taken from the
    // consuming instruction.
    auto null_texture = [](uint32_t img_dim) {
        ShaderResourceTable rt;
        ShaderResource rn{};
        rn.cls = ResourceClass::Texture;
        rn.gpu_addr = 0;
        rn.size = 0;
        rn.binding = 0;
        rn.img_dim = img_dim;
        rn.depth = 1;
        rn.fetch_pc = 0;
        rn.srt_offset = 0xFFFFFFFFu;
        rn.sgpr_base = 0xFFFFFFFFu;
        rt.resources.push_back(rn);
        return rt;
    };

    ShaderResourceTable rt_correct = null_texture(2);   // 3D, as the fix now supplies
    std::vector<uint32_t> spirv = recompile_compute(code_3d, std::size(code_3d), &rt_correct,
                                                    ComputeShaderConfig{});
    CHECK(!spirv.empty(), "a 3D image_load through an all-zero T# recompiles");
    if (spirv.empty()) { printf("== FAIL: %d ==\n", fails + 1); return 1; }

    // Establish the premise before asserting anything about it: the module really does declare Dim=3D.
    // Without this the two arms below could both pass against a module that declared 2D, which is
    // exactly the case the issue is about.
    DescriptorValidationReport probe = validate_spirv_descriptor_interface(
        spirv, &rt_correct, 0, SpirvShaderStage::Compute, false);
    const SpirvDescriptorBinding* use = find_spirv_descriptor_binding(probe, 0, 0);
    CHECK(use && use->image_dim == 2u,
          "PREMISE: the recompiled module declares Dim=3D, taken from the MIMG instruction "
          "(the all-zero descriptor cannot have supplied it)");

    // Arm 1 — the shape the fix supplies is accepted.
    CHECK(probe.ok(),
          "#3577: a null descriptor whose img_dim matches the declared 3D shape validates clean");

    // Arm 2 — the shape the OLD code left behind is reported. `img_dim = 1` is precisely the struct
    // default the synthesis site used to leave in place, so this arm reconstructs the exact defect.
    ShaderResourceTable rt_stale = null_texture(1);   // 2D -- the pre-fix struct default
    DescriptorValidationReport stale = validate_spirv_descriptor_interface(
        spirv, &rt_stale, 0, SpirvShaderStage::Compute, false);
    bool reported = false;
    for (const auto& issue : stale.issues)
        if (issue.code == DescriptorIssueCode::InvalidImageMetadata && issue.binding == 0)
            reported = true;
    CHECK(reported,
          "#3577: a null descriptor that would build a 2D view for a Dim=3D shader is REPORTED "
          "(the validator used to skip its whole image branch for null descriptors)");

    // Arm 3 — the exemption that was correct stays. A null descriptor legitimately has no extent, no
    // format and no size; narrowing the skip must not start rejecting nulls for lacking them. Both
    // tables above have width/height/format unset, so arm 1 passing already proves this -- stated
    // explicitly so a future change that re-widens the check has something to go red.
    CHECK(rt_correct.resources[0].width == 0 && rt_correct.resources[0].height == 0 && probe.ok(),
          "#3577: a null descriptor is still exempt from the extent/format/size checks");

    // Arm 4 — the synthesized record's `depth = 1` is load-bearing and had no guard. The backend
    // builds a real VK_IMAGE_TYPE_3D for a 3D null and takes `extent.depth` from this field; a zero
    // fails vkCreateImage and the draw is then silently skipped. Deleting `rn.depth = 1` at the
    // synthesis site used to break nothing in the suite.
    ShaderResourceTable rt_flat = null_texture(2);
    rt_flat.resources[0].depth = 0;
    DescriptorValidationReport flat = validate_spirv_descriptor_interface(
        spirv, &rt_flat, 0, SpirvShaderStage::Compute, false);
    bool depth_reported = false;
    for (const auto& issue : flat.issues)
        if (issue.code == DescriptorIssueCode::InvalidImageMetadata && issue.binding == 0)
            depth_reported = true;
    CHECK(depth_reported,
          "#3577: a 3D descriptor with ZERO depth is reported -- the backend would build a "
          "VK_IMAGE_TYPE_3D with extent.depth = 0 and lose the draw to a create failure");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
