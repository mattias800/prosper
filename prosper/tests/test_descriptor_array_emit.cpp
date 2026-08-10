// test_descriptor_array_emit — the emitter declares an indexed descriptor ARRAY for a table-indexed
// resource, with the capability set and the OpExtension that SPIR-V 1.3 requires (#2412 stage 4b).
//
// Nothing in the guest path sets `table_index_count` yet, so without this test the entire array-emission
// path is structurally inexpressible in the suite: every other test would pass whether or not the code
// existed. Same hole test_descriptor_array_render closes on the backend side.
//
// What each arm establishes:
//
//   1. A table-indexed resource emits OpExtension "SPV_EXT_descriptor_indexing" plus the three
//      capabilities. They are core only from SPIR-V 1.5 and this emitter writes 1.3 (0x00010300), so
//      the extension is mandatory, not optional.
//   2. The extension is emitted in the RIGHT PLACE. SPIR-V requires OpExtension after every
//      OpCapability and before OpExtInstImport. Getting this wrong fails spirv-val with a layout
//      complaint that mentions nothing about descriptors, so it is asserted structurally here rather
//      than left to CI to discover.
//   3. The pointee is an OpTypeArray, so the binding occupies N descriptors instead of 1.
//   4. THE CONTROL, and the one that makes the rest mean anything: an ordinary resource emits NONE of
//      it. If the capability leaked into every module, arm 1 would pass for the wrong reason and every
//      existing shader would start demanding an extension it does not use.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Minimal SPIR-V word walk: yields (opcode, word_index) for each instruction after the 5-word header.
static std::vector<std::pair<uint32_t, size_t>> walk(const std::vector<uint32_t>& m) {
    std::vector<std::pair<uint32_t, size_t>> out;
    for (size_t i = 5; i < m.size();) {
        const uint32_t len = m[i] >> 16, op = m[i] & 0xFFFFu;
        if (len == 0) break;
        out.push_back({op, i});
        i += len;
    }
    return out;
}
static bool has_op(const std::vector<uint32_t>& m, uint32_t op) {
    for (auto& e : walk(m)) if (e.first == op) return true;
    return false;
}
static bool has_capability(const std::vector<uint32_t>& m, uint32_t cap) {
    for (auto& e : walk(m)) if (e.first == 17u && m[e.second + 1] == cap) return true;   // OpCapability
    return false;
}
static std::string extension_name(const std::vector<uint32_t>& m) {
    for (auto& e : walk(m)) {
        if (e.first != 10u) continue;                                                     // OpExtension
        const char* p = reinterpret_cast<const char*>(&m[e.second + 1]);
        return std::string(p);
    }
    return {};
}
// Index of the first instruction with `op`, or SIZE_MAX.
static size_t first_index_of(const std::vector<uint32_t>& m, uint32_t op) {
    for (auto& e : walk(m)) if (e.first == op) return e.second;
    return SIZE_MAX;
}
static size_t last_index_of(const std::vector<uint32_t>& m, uint32_t op) {
    size_t found = SIZE_MAX;
    for (auto& e : walk(m)) if (e.first == op) found = e.second;
    return found;
}

int main() {
    printf("== test_descriptor_array_emit ==\n");

    // The vertex-fetch VS used across the render tests: fetches (x,y) from the buffer at binding 3.
    const uint32_t vs[] = {
        0x7e060280u, 0x7e0802f2u, 0xe0042000u, 0x80020100u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
    };

    auto table_with_arity = [](uint32_t arity) {
        ShaderResourceTable rt;
        ShaderResource vb{};
        vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32; vb.num_components = 2;
        vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8;
        rt.resources.push_back(vb);
        if (arity) {
            // Binding 4, NOT 3. Bindings 2 and 3 are pre-declared as scalar storage buffers before
            // declare_cbufs' N-buffer loop and are seeded into its `seen` set, so a table-indexed
            // resource at either is skipped and silently emitted as an ordinary descriptor. That is a
            // real limitation of this stage rather than a quirk of the test -- filed separately -- and
            // it matters because a title's constant buffers commonly land on exactly those two
            // bindings. This arm therefore proves the mechanism on a binding that can reach it.
            ShaderResource tb{};
            tb.cls = ResourceClass::ConstantBuffer; tb.format = DataFormat::Float32;
            tb.num_components = 4; tb.binding = 4; tb.stride = 16; tb.sgpr_base = 12;
            tb.table_index_count = arity; tb.table_entry_stride = 16;
            rt.resources.push_back(tb);
        }
        return rt;
    };

    // --- Arm 4 first: the CONTROL. An ordinary resource must emit none of this. -------------------
    ShaderResourceTable plain = table_with_arity(0);
    std::vector<uint32_t> m_plain = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &plain);
    CHECK(!m_plain.empty(), "control: ordinary resource still recompiles");
    if (m_plain.empty()) { printf("== FAIL ==\n"); return 1; }
    CHECK(!has_op(m_plain, 10u), "control: an ordinary resource emits NO OpExtension");
    CHECK(!has_capability(m_plain, 5301u) && !has_capability(m_plain, 5302u) &&
              !has_capability(m_plain, 5306u),
          "control: an ordinary resource declares NO descriptor-indexing capability");

    // --- Arms 1-3: a table-indexed resource ------------------------------------------------------
    ShaderResourceTable indexed = table_with_arity(8);
    std::vector<uint32_t> m_arr = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &indexed);
    CHECK(!m_arr.empty(), "a table-indexed resource recompiles");
    if (m_arr.empty()) { printf("== FAIL ==\n"); return 1; }

    CHECK(m_arr[1] == 0x00010300u, "the module is SPIR-V 1.3, which is why the extension is required");
    CHECK(extension_name(m_arr) == "SPV_EXT_descriptor_indexing",
          "OpExtension SPV_EXT_descriptor_indexing is emitted");
    CHECK(has_capability(m_arr, 5301u), "ShaderNonUniform capability declared");
    CHECK(has_capability(m_arr, 5302u), "RuntimeDescriptorArray capability declared");
    CHECK(has_capability(m_arr, 5306u), "StorageBufferArrayNonUniformIndexing capability declared");

    // Layout: every OpCapability precedes OpExtension, which precedes OpExtInstImport.
    const size_t last_cap = last_index_of(m_arr, 17u);
    const size_t ext = first_index_of(m_arr, 10u);
    const size_t extinst = first_index_of(m_arr, 11u);
    CHECK(last_cap != SIZE_MAX && ext != SIZE_MAX && last_cap < ext,
          "OpExtension follows every OpCapability (required section order)");
    CHECK(extinst == SIZE_MAX || ext < extinst,
          "OpExtension precedes OpExtInstImport (required section order)");

    CHECK(has_op(m_arr, 28u), "the binding's pointee is an OpTypeArray — it occupies N descriptors");

    // Deliberately NOT asserted: the absence of OpTypeRuntimeArray. The Block type this emitter builds
    // for every storage buffer already contains one (a runtime array of u32 IS the buffer's payload), so
    // "no runtime array in the module" is false for every module and an assertion on it would either be
    // vacuous or wrong. The fixed-vs-runtime distinction is asserted below on the arity instead.

    // --- An absurd count must not be emitted as a fixed array of that length ----------------------
    // #2463 gives an unreadable length the sentinel UINT32_MAX. Emitting OpTypeArray with that length
    // would ask a driver for a 4-billion-descriptor binding, so the emitter degrades to a runtime
    // array instead. Asserted here so the guard cannot be removed silently once #2463 merges.
    ShaderResourceTable absurd = table_with_arity(0xFFFFFFFFu);
    std::vector<uint32_t> m_absurd = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &absurd);
    CHECK(!m_absurd.empty(), "an implausible arity still recompiles");
    if (!m_absurd.empty()) {
        bool huge_literal = false;
        for (auto& e : walk(m_absurd))
            if (e.first == 43u && e.second + 3 < m_absurd.size() && m_absurd[e.second + 3] == 0xFFFFFFFFu)
                huge_literal = true;   // OpConstant with the sentinel as its value
        CHECK(!huge_literal,
              "an implausible arity is NOT emitted as a fixed array of 4294967295 descriptors");
    }

    printf(fails ? "== FAIL ==\n" : "== PASS ==\n");
    return fails ? 1 : 0;
}
