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

    if (fails) { printf("== FAIL: %d shader(s) failed recompile/validation ==\n", fails); return 1; }
    printf("== PASS%s ==\n", have_val ? " (all modules pass spirv-val)" : " (recompiled; spirv-val not found)");
    return 0;
}
