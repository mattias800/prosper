// spv_validate — recompile a representative shader from every path the RDNA2->SPIR-V recompiler
// supports and write each module as a .spv to argv[1] (a directory). A wrapper then runs spirv-val on
// them: the render tests only prove llvmpipe *accepts* the modules, but llvmpipe is lenient — strict
// validation catches latent invalid SPIR-V (bad decorations, ill-formed control flow, type mismatches)
// that would break on a real driver. Pure recompile + file write; no Vulkan.
#include "../../src/gpu/rdna2_to_spirv.hpp"
#include "../../src/gpu/shader_resources.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>

using namespace prosper::gpu;

static int fails = 0;
static bool have_val = false;   // is spirv-val on PATH?

static void dump(const std::string& dir, const char* name, const std::vector<uint32_t>& spv) {
    if (spv.empty() || spv[0] != 0x07230203u) { printf("  [FAIL] %-22s did not recompile\n", name); fails++; return; }
    std::string path = dir + "/" + name + ".spv";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("  [FAIL] %-22s cannot write %s\n", name, path.c_str()); fails++; return; }
    fwrite(spv.data(), 4, spv.size(), f); fclose(f);
    if (have_val) {
        std::string cmd = "spirv-val \"" + path + "\" > /dev/null 2>&1";
        if (system(cmd.c_str()) != 0) { printf("  [FAIL] %-22s spirv-val REJECTED it\n", name); fails++; return; }
        printf("  [ok]   %-22s (%zu words) valid\n", name, spv.size());
    } else {
        printf("  wrote  %-22s (%zu words) [spirv-val absent — recompile-only]\n", name, spv.size());
    }
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    have_val = (system("spirv-val --version > /dev/null 2>&1") == 0);

    // Compute ALU (float chain).
    { const uint32_t c[] = {0x06000300u, 0x10000500u, 0xBF810000u};
      dump(dir, "compute_alu", recompile_valu(c, 3, 3, 0)); }
    // Compute + SMEM constant-buffer load (s_buffer_load_dword; routes to binding 2).
    { const uint32_t c[] = {0xf4000000u, 0xfa000004u, 0x7e000200u, 0xbf810000u};
      dump(dir, "compute_smem", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Fragment: solid green (EXP MRT0).
    { const uint32_t c[] = {0x7E000280u,0x7E0202F2u,0x7E040280u,0x7E0602F2u,0xF800180Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_color", recompile_fragment(c, sizeof(c)/4)); }
    // Vertex: fullscreen triangle from gl_VertexIndex (EXP POS0).
    { const uint32_t c[] = {0x36020081u,0x2C040081u,0x7E020D01u,0x7E040D02u,0x7E0A02F6u,0x7E0C02F2u,0x10020B01u,
                            0x08020D01u,0x10040B02u,0x08040D02u,0x7E060280u,0x7E0802F2u,0xF80008CFu,0x04030201u,0xBF810000u};
      dump(dir, "vertex_fullscreen", recompile_vertex(c, sizeof(c)/4)); }
    // Vertex fetch (buffer_load_format_xy from a V# in user-data s[8:11]).
    { const uint32_t c[] = {0x7e060280u,0x7e0802f2u,0xe0042000u,0x80020100u,0xf80008cfu,0x04030201u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=2; vb.binding=3; vb.stride=8; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "vertex_fetch", recompile_vertex(c, sizeof(c)/4, &rt)); }
    // Fragment: image_sample a texture (T# in s[8:15]).
    { const uint32_t c[] = {0x7e0002ffu,0x3e800000u,0x7e0202ffu,0x3e800000u,0xf0800f08u,0x00820000u,0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Float32;
      t.num_components=4; t.binding=4; t.img_dim=1; t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_texture", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_sample a 3D texture (3 coords) -> mrt0. (shader_028 pattern)
    { const uint32_t c[] = {0x7E0002FFu,0x3E800000u,0x7E0202FFu,0x3E800000u,0x7E0402FFu,0x3E800000u,
                            0xF0800F10u,0x00400000u,0xF800180Fu,0x03020100u,0xBF810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Float32;
      t.num_components=4; t.binding=4; t.img_dim=2; t.width=2; t.height=2; t.sgpr_base=0; rt.resources.push_back(t);
      dump(dir, "fragment_texture_3d", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: COMPR export — v_cvt_pkrtz packs f16x2, exp ... done compr unpacks to vec4. (shader_029)
    { const uint32_t c[] = {0x7E0002F0u,0x7E0202F0u,0x5E000300u,0x5E020300u,0xF8001C0Fu,0x00000100u,0xBF810000u};
      dump(dir, "fragment_compr_export", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Vertex with PARAM export (interpolated varying out).
    { const uint32_t c[] = {0x7e140d00u,0x36020081u,0x2c040081u,0x7e020d01u,0x7e040d02u,0x100202f6u,0x100404f6u,
                            0x060202f3u,0x060404f3u,0x7e060280u,0x7e0802f2u,0xf80008cfu,0x04030201u,0xf800020fu,0x0403030au,0xbf810000u};
      dump(dir, "vertex_param", recompile_vertex(c, sizeof(c)/4)); }
    // Fragment with v_interp (interpolated varying in).
    { const uint32_t c[] = {0xc8000000u,0xc8010001u,0x7e020280u,0x7e0402f2u,0xf800080fu,0x02010100u,0xbf810000u};
      dump(dir, "fragment_interp", recompile_fragment(c, sizeof(c)/4)); }
    // Compute LDS (ds_write / s_barrier / ds_read).
    { const uint32_t c[] = {0x7e020f00u,0x34040282u,0x34060281u,0x4a060681u,0xd8340000u,0x00000302u,0xbf8a0000u,
                            0x4c0a02bfu,0x340c0a82u,0xd8d80000u,0x07000006u,0x7e000d07u,0xbf810000u};
      dump(dir, "compute_lds", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Compute MUBUF store (buffer_store_format_x).
    { const uint32_t c[] = {0x7e040f00u,0x06060100u,0xe0102000u,0x80020302u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=1; vb.binding=3; vb.stride=4; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "compute_store", recompile_valu(c, sizeof(c)/4, 1, 0, &rt)); }
    // Compute EXEC-predicated store (v_cmpx + guard execz + store).
    { const uint32_t c[] = {0x7e040f00u,0x06060100u,0x7e0a0284u,0x7da20b02u,0xbf880002u,0xe0102000u,0x80020302u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=1; vb.binding=3; vb.stride=4; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "compute_pred_store", recompile_valu(c, sizeof(c)/4, 1, 0, &rt)); }

    // Compute STORAGE images (image_load -> image_store, no sampler): 1D, NSA 3D (split-address coords),
    // and 2D_ARRAY (layer coord). One shared rt: src U# in user-data s[0:7]->binding 4, dst s[8:15]->5.
    { ShaderResourceTable rt;
      ShaderResource s{}; s.cls = ResourceClass::StorageImage; s.binding = 4; s.sgpr_base = 0; rt.resources.push_back(s);
      ShaderResource d{}; d.cls = ResourceClass::StorageImage; d.binding = 5; d.sgpr_base = 8; rt.resources.push_back(d);
      const uint32_t c1d[]  = {0x7E080300u,0xF0000F00u,0x00000004u,0xBF8C3F70u,0xF0200F00u,0x00020004u,0xBF810000u};
      dump(dir, "storage_copy_1d",    recompile_valu(c1d,  sizeof(c1d)/4,  1, 0, &rt));
      const uint32_t cnsa[] = {0xF0000F12u,0x0000000Au,0x00000C0Bu,0xBF8C3F70u,0xF0200F12u,0x0002000Au,0x00000C0Bu,0xBF810000u};
      dump(dir, "storage_nsa_3d",     recompile_valu(cnsa, sizeof(cnsa)/4, 1, 0, &rt));
      const uint32_t carr[] = {0xF0000F28u,0x00000004u,0xBF8C3F70u,0xF0200F28u,0x00020004u,0xBF810000u};
      dump(dir, "storage_arrayed_2d", recompile_valu(carr, sizeof(carr)/4, 1, 0, &rt));
      const uint32_t cms[]  = {0xF0000F30u,0x00000000u,0xBF8C3F70u,0x7E000D00u,0xBF810000u};   // image_load 2D_MSAA (x,y,sample)
      dump(dir, "storage_msaa_2d",    recompile_valu(cms,  sizeof(cms)/4,  1, 0, &rt));
      // image_load 2D_MSAA_ARRAY (dim 7): coords (x,y,layer) + sample — needs Arrayed+MS + ImageMSArray cap.
      const uint32_t cmsa[] = {0xF0000F3Au,0x00000000u,0x00030201u,0xBF8C3F70u,0xBF810000u};
      dump(dir, "storage_msaa_array_2d", recompile_valu(cmsa, sizeof(cmsa)/4, 1, 0, &rt)); }
    // Compute that SAMPLES a texture and STORES to a storage image (the bloom/downsample shape, shader 006):
    // exercises the sampled-texture path inside a compute shell (needs vec4<float> declared there).
    { ShaderResourceTable rt;
      ShaderResource t{};  t.cls=ResourceClass::Texture;      t.binding=0; t.sgpr_base=0;  t.img_dim=1; rt.resources.push_back(t);
      ShaderResource s{};  s.cls=ResourceClass::StorageImage; s.binding=1; s.sgpr_base=16; rt.resources.push_back(s);
      const uint32_t c[] = {0x7E0002F0u,0x7E0202F0u,0xF09C0F08u,0x00400400u,0xBF8C3F70u,0xF0200108u,0x00040402u,0xBF810000u};
      dump(dir, "compute_sample_store", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // Compute mul_hi (high 32 bits via OpUMulExtended -> {lo,hi} struct extract).
    { const uint32_t c[] = {0x7E0202FFu,0x80000000u,0xD56A0002u,0x00020301u,0x7E000D02u,0xBF810000u};
      dump(dir, "compute_mul_hi", recompile_valu(c, sizeof(c)/4, 1, 0)); }

    // --- #273 additions (DOLL recompiler frontier) ---
    // Fragment: 3D image_load (integer LUT fetch through the combined sampler; OpImage+OpImageFetch).
    { const uint32_t c[] = {0x7e000280u,0x7e020280u,0x7e040280u,0xf0001f10u,0x00020000u,
                            0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=2;
      t.width=16; t.height=16; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_load_3d", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_sample_lz_o (packed texel offset folded into normalized coords; ImageQuery).
    { const uint32_t c[] = {0x7e0002ffu,0x00000101u,0x7e0202f0u,0x7e0402f0u,0xf0dc0f08u,0x00820000u,
                            0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=1;
      t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_sample_lz_o", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_gather4_lz_o (dynamic gather offset; ImageGatherExtended — locks the operand-ID fix).
    { const uint32_t c[] = {0x7e0002ffu,0x00000101u,0x7e0202f0u,0x7e0402f0u,0xf15c0808u,0x00820400u,
                            0xf800000fu,0x07060504u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=1;
      t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_gather4_lz_o", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Compute: if/else-if/else cascade with common-merge s_branch arms (kernel T17's stream).
    { const uint32_t c[] = {0x7e020f00u,0x7e080501u,0xbf0a8204u,0xbf840002u,0x4a02028au,0xbf820005u,
                            0xbf0a8504u,0xbf840002u,0x4a020294u,0xbf820001u,0x4a02029eu,0x7e000d01u,0xbf810000u};
      dump(dir, "compute_cascade_ifelse", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Compute: readfirstlane waterfall + v_movrels (kernel T19's stream).
    { const uint32_t c[] = {0x7e020f00u,0xbe86047eu,0x7e0402ffu,0x40a00000u,0x7e0602ffu,0x40e00000u,
                            0x7e0802ffu,0x41100000u,0x7e080501u,0x7da40204u,0xbefc0304u,0x7e0a8702u,
                            0x8a867e06u,0xbefe0406u,0xbf85fff9u,0xbefe04c1u,0x7e000305u,0xbf810000u};
      dump(dir, "compute_waterfall_movrels", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Fragment: divergent execz region (v_cmpx + s_cbranch_execz over a scalar-writing block).
    { const uint32_t c[] = {0x7e020280u,0x7e0002f2u,0x7c2200f0u,0xbf880002u,0xbe8503f2u,0x7e020205u,
                            0x7e040205u,0xf800180fu,0x01010101u,0xbf810000u};
      dump(dir, "fragment_execz_if", recompile_fragment(c, sizeof(c)/4, nullptr)); }

    if (fails) { printf("== FAIL: %d shader(s) failed recompile/validation ==\n", fails); return 1; }
    printf("== PASS%s ==\n", have_val ? " (all modules pass spirv-val)" : " (recompiled; spirv-val not found)");
    return 0;
}
