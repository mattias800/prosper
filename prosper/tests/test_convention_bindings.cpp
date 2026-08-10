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

    // A separately-installed fetch prolog and main shader share one architectural register file,
    // descriptor set, and linked instruction stream. Main-program fetch provenance must move by the
    // exact executable prolog prefix while every resource receives a fresh collision-free binding.
    {
        auto prolog = std::make_shared<ShaderResourceTable>();
        auto main = std::make_shared<ShaderResourceTable>();
        ShaderResource prolog_buffer = res(RC::VertexBuffer);
        prolog_buffer.fetch_pc = 1;
        ShaderResource main_buffer = res(RC::ConstantBuffer);
        main_buffer.fetch_pc = 3;
        ShaderResource main_texture = res(RC::Texture);
        main_texture.fetch_pc = 7;
        prolog->resources.push_back(prolog_buffer);
        main->resources = {main_buffer, main_texture};

        const auto merged = merge_vertex_chain_resource_tables(prolog, main, 10);
        CHECK(merged && merged->resources.size() == 3,
              "vertex prolog/main resource contracts merge without dropping entries");
        CHECK(merged && merged->resources[0].fetch_pc == 1 &&
                  merged->resources[1].fetch_pc == 13 && merged->resources[2].fetch_pc == 17,
              "main-program fetch provenance is rebased by the linked prolog prefix");
        CHECK(merged && merged->resources[0].binding == 2 &&
                  merged->resources[1].binding == 3 && merged->resources[2].binding == 4,
              "linked vertex resources receive one collision-free descriptor layout");
        CHECK(merge_vertex_chain_resource_tables(nullptr, nullptr, 10) == nullptr,
              "an empty vertex chain retains the absent resource-table contract");
    }

    // Stage 2 of the runtime-selected-descriptor lift (#2412): TABLE-INDEXED provenance is a fifth,
    // separate shape and must be INERT until the later stages exist. These arms pin that, because the
    // failure mode of a half-finished lift is a resource carrying an array count that some layer then
    // treats as a scalar binding — which today validates SILENTLY (no DescriptorIssueCode covers an
    // array-vs-scalar mismatch), so nothing would report it.
    {
        ShaderResource r;
        CHECK(r.table_index_count == 0 && r.table_entry_stride == 0,
              "table-indexed fields default to inert, so existing resources are unaffected by construction");

        // A count is NOT a provenance key: it must not disturb the four mutually-exclusive ones, and a
        // table-indexed resource still resolves by whichever of them it carries.
        ShaderResource t;
        t.cls = ResourceClass::ConstantBuffer;
        t.srt_offset = 0x40;
        t.table_index_count = 8;
        t.table_entry_stride = 16;
        CHECK(t.srt_offset == 0x40 && t.sgpr_base == 0xFFFFFFFFu &&
                  t.fetch_pc == 0xFFFFFFFFu,
              "a table-indexed resource leaves the four origin keys exactly as set");

        // Binding assignment must be untouched by the count: `binding` names the ARRAY, and stage 2
        // changes no layout. If this ever differs, a later stage has silently repurposed the binding
        // scalar and every existing table's layout moves with it.
        auto tbl = std::make_shared<ShaderResourceTable>();
        ShaderResource a; a.cls = ResourceClass::ConstantBuffer; a.srt_offset = 0x00;
        ShaderResource b; b.cls = ResourceClass::ConstantBuffer; b.srt_offset = 0x10;
        b.table_index_count = 4; b.table_entry_stride = 16;
        tbl->resources = {a, b};
        assign_convention_bindings(*tbl, 2);
        CHECK(tbl->resources[0].binding == 2 && tbl->resources[1].binding == 3,
              "an array count does not change how bindings are assigned (stage 2 is representation only)");
        CHECK(tbl->resources[1].table_index_count == 4 &&
                  tbl->resources[1].table_entry_stride == 16,
              "binding assignment preserves the array count and stride it did not set");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
