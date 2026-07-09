// test_convention_bindings — assign_convention_bindings must never place a texture / storage image on
// binding 2 or 3, which the recompiler's declare_cbufs always occupies with two hardwired storage-
// buffer cbufs. A shader whose FIRST resource is a texture used to land it on binding 2, declaring two
// descriptor types at one binding -> layout-creation failure, the draw disappears (#157). Constant/
// vertex buffers are assigned first (2/3+), matching the common cbufs-first shaders byte-for-byte.
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/shader_resources.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static ShaderResource res(ResourceClass cls) { ShaderResource r{}; r.cls = cls; return r; }

int main() {
    printf("== test_convention_bindings ==\n");
    using RC = ResourceClass;

    // TEXTURE-FIRST (#157 bug): a texture listed before any buffer must NOT get binding 2/3.
    {
        ShaderResourceTable t;
        t.resources = { res(RC::Texture), res(RC::ConstantBuffer) };
        assign_convention_bindings(t, 2);
        CHECK(t.resources[1].binding == 2, "the constant buffer gets binding 2 (buffers first)");
        CHECK(t.resources[0].binding >= 4, "the texture is pushed to binding >= 4, off the hardwired 2/3");
    }

    // COMMON cbufs-first shape: 2 cbufs + a texture -> byte-identical to the old sequential layout.
    {
        ShaderResourceTable t;
        t.resources = { res(RC::ConstantBuffer), res(RC::ConstantBuffer), res(RC::Texture) };
        assign_convention_bindings(t, 2);
        CHECK(t.resources[0].binding == 2 && t.resources[1].binding == 3,
              "two cbufs stay at 2 and 3 (matching the hardwired v_cbuf/v_cbuf1)");
        CHECK(t.resources[2].binding == 4, "texture after two cbufs stays at 4 (no regression)");
    }

    // Texture-ONLY (no buffers): still reserve 2/3.
    {
        ShaderResourceTable t;
        t.resources = { res(RC::Texture), res(RC::Texture) };
        assign_convention_bindings(t, 2);
        CHECK(t.resources[0].binding == 4 && t.resources[1].binding == 5,
              "texture-only VS: textures start at 4 (2/3 reserved for the hardwired cbufs)");
    }

    // A storage image is treated as a texture-class resource (also kept off 2/3).
    {
        ShaderResourceTable t;
        t.resources = { res(RC::StorageImage), res(RC::VertexBuffer) };
        assign_convention_bindings(t, 2);
        CHECK(t.resources[1].binding == 2, "vertex buffer takes binding 2");
        CHECK(t.resources[0].binding >= 4, "storage image kept off 2/3");
    }

    // PS range (first = 32): buffers from 32/33, textures from 34 — disjoint from the VS range and
    // still never colliding two types at one binding.
    {
        ShaderResourceTable t;
        t.resources = { res(RC::Texture), res(RC::ConstantBuffer) };
        assign_convention_bindings(t, 32);
        CHECK(t.resources[1].binding == 32, "PS constant buffer at 32");
        CHECK(t.resources[0].binding >= 34, "PS texture off the 32/33 reserved slots");
    }

    // All bindings within a table are distinct (no two resources collapse).
    {
        ShaderResourceTable t;
        t.resources = { res(RC::Texture), res(RC::ConstantBuffer), res(RC::Texture), res(RC::VertexBuffer) };
        assign_convention_bindings(t, 2);
        bool distinct = true;
        for (size_t i = 0; i < t.resources.size(); i++)
            for (size_t j = i + 1; j < t.resources.size(); j++)
                if (t.resources[i].binding == t.resources[j].binding) distinct = false;
        CHECK(distinct, "every resource gets a distinct binding");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
