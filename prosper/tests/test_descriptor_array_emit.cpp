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
        if (arity) {
            // Binding 3 -- the binding this shader actually READS. Before #2472 bindings 2 and 3 could
            // not be arrays, so this had to sit on binding 4 where nothing loaded from it and the access
            // chain was unreachable. Now the arity goes on the read binding and the selector is testable.
            vb.table_index_count = arity;
            vb.table_entry_stride = 16;
            vb.table_index_sgpr = 6;   // the descriptor index arrives in user SGPR 6
        }
        rt.resources.push_back(vb);
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

    // --- The access chain SELECTS an entry, and the selection is decorated NonUniform ---------------
    // These two arms could not exist before #2472: the table-indexed resource had to live on a binding
    // the shader never read, so no access chain into the array was emitted and both arms failed. They
    // matter more than the declaration arms above, because this is the part that fails QUIETLY -- a
    // mis-emitted selector reads a valid descriptor from the wrong slot, so the shader renders
    // confidently wrong content instead of erroring.
    {
        // An ordinary chain into a storage buffer is {result_type, result, base, 0, idx} -> 5 operands,
        // word length 6. A table-indexed one carries a leading selector -> 6 operands, length 7.
        bool indexed_chain = false;
        for (auto& e : walk(m_arr))
            if (e.first == 65u && (m_arr[e.second] >> 16) == 7u) indexed_chain = true;
        CHECK(indexed_chain, "an access chain carries a LEADING selector (the array is really indexed)");

        // The control that makes it mean something: the ordinary module's chains are all the short form.
        bool plain_has_indexed = false;
        for (auto& e : walk(m_plain))
            if (e.first == 65u && (m_plain[e.second] >> 16) == 7u) plain_has_indexed = true;
        CHECK(!plain_has_indexed, "control: an ordinary module emits no leading-selector chain");

        auto has_nonuniform = [](const std::vector<uint32_t>& m) {
            for (auto& e : walk(m))
                if (e.first == 71u && e.second + 2 < m.size() && m[e.second + 2] == 5300u) return true;
            return false;
        };
        CHECK(has_nonuniform(m_arr), "the selection is decorated NonUniform (5300)");
        CHECK(!has_nonuniform(m_plain), "control: an ordinary module carries no NonUniform decoration");
    }

    // --- Arm 5: two resources share one binding, of DIFFERENT classes ------------------------------
    // `table_arity_for` filters by class (ConstantBuffer/VertexBuffer, the only classes whose array
    // path exists); `index_sgpr_for` did not. Nothing prevents two resources sharing a binding, so the
    // arity came from the vertex buffer while the index SGPR came from whichever resource sat at that
    // binding FIRST. Both values are individually plausible — an arity that is right and an SGPR that
    // is wrong — so the shader loads its descriptor index from the wrong register and no check fires.
    //
    // Asserted by byte-equality rather than by decoding the push-constant offset: two tables that
    // differ ONLY in the decoy's `table_index_sgpr` must emit identical modules, because the decoy is
    // not a table-indexable class and must be ignored entirely.
    {
        auto table_with_decoy = [](uint32_t decoy_sgpr) {
            ShaderResourceTable rt;
            ShaderResource decoy{};
            decoy.cls = ResourceClass::Sampler;   // NOT table-indexable — must be skipped
            decoy.binding = 3;
            decoy.table_index_count = 2; decoy.table_entry_stride = 16;
            decoy.table_index_sgpr = decoy_sgpr;
            rt.resources.push_back(decoy);        // FIRST, so a missing class filter reaches it first
            ShaderResource vb{};
            vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32;
            vb.num_components = 2;
            vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8;
            vb.table_index_count = 3; vb.table_entry_stride = 16; vb.table_index_sgpr = 6;
            rt.resources.push_back(vb);
            return rt;
        };
        ShaderResourceTable decoy_lo = table_with_decoy(9);
        ShaderResourceTable decoy_hi = table_with_decoy(11);
        std::vector<uint32_t> m_lo = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &decoy_lo);
        std::vector<uint32_t> m_hi = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &decoy_hi);
        CHECK(!m_lo.empty() && !m_hi.empty(),
              "a binding shared by two resource classes still recompiles");
        CHECK(m_lo == m_hi,
              "the index SGPR ignores a resource whose class cannot be table-indexed");

        // Positive control, and it is load-bearing. The equality above would also hold if NO index
        // SGPR reached the module at all — a table-indexed resource that silently stopped emitting a
        // selector would satisfy it just as well as the fix does. So prove the axis is live: moving
        // the VERTEX BUFFER's own index SGPR must change the module.
        ShaderResourceTable moved = table_with_decoy(9);
        moved.resources[1].table_index_sgpr = 7;
        std::vector<uint32_t> m_moved = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &moved);
        CHECK(!m_moved.empty() && m_moved != m_lo,
              "positive control: moving the VERTEX BUFFER's index SGPR does change the module, so the "
              "equality above is the decoy being ignored rather than the SGPR being unused");
    }

    printf(fails ? "== FAIL ==\n" : "== PASS ==\n");
    return fails ? 1 : 0;
}
