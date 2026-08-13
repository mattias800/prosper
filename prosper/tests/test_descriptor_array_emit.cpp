// test_descriptor_array_emit — the emitter declares an indexed descriptor ARRAY for a table-indexed
// resource, with the capability set and the OpExtension that SPIR-V 1.3 requires (#2412 stage 4b).
//
// The runtime producer remains under proof, so this synthetic resource keeps the bounded array-emission
// path directly expressible in the suite. The live compute test closes the backend side.
//
// What each arm establishes:
//
//   1. A table-indexed resource emits OpExtension "SPV_EXT_descriptor_indexing" plus the two
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

static ShaderBufferTableEntry float2_entry(uint64_t address) {
    ShaderBufferTableEntry entry;
    entry.gpu_addr = address;
    entry.size = 16;
    entry.stride = 8;
    entry.vsharp = {
        static_cast<uint32_t>(address),
        static_cast<uint32_t>(address >> 32u) | (8u << 16u),
        2u,
        (64u << 12u) | 0xfacu,
    };
    return entry;
}

int main() {
    printf("== test_descriptor_array_emit ==\n");

    // A compute fetch of (x,y) from the V# in s[8:11], which maps to binding 3 below. Compute is the
    // shell that carries guest user SGPRs through push constants; graphics deliberately rejects this
    // selector mode until it has an equivalent runtime source.
    const uint32_t cs[] = {
        0x7e060280u, 0x7e0802f2u, 0xe0300000u, 0x80020100u, 0xbf810000u,
    };
    ComputeShaderConfig config;
    config.user_sgprs.resize(12);
    config.local_x = config.local_y = config.local_z = 1;
    auto recompile = [&](const ShaderResourceTable& table) {
        return recompile_compute(cs, std::size(cs), &table, config);
    };

    auto table_with_arity = [](uint32_t arity) {
        ShaderResourceTable rt;
        ShaderResource vb{};
        vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32; vb.num_components = 2;
        vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8;
        if (arity) {
            // Binding 3 is the binding this shader actually reads, so the selector is testable.
            vb.table_index_count = arity;
            vb.table_entry_stride = 16;
            vb.table_index_sgpr = 6;   // the descriptor index arrives in user SGPR 6
            vb.table_selector_mode = BufferTableSelectorMode::UserSgprIndex;
            if (arity <= 4096u)
                for (uint32_t index = 0; index < arity; ++index)
                    vb.table_entries.push_back(float2_entry(0x200000u + index * 0x1000u));
        }
        rt.resources.push_back(vb);
        return rt;
    };

    // --- Arm 4 first: the CONTROL. An ordinary resource must emit none of this. -------------------
    ShaderResourceTable plain = table_with_arity(0);
    std::vector<uint32_t> m_plain = recompile(plain);
    CHECK(!m_plain.empty(), "control: ordinary resource still recompiles");
    if (m_plain.empty()) { printf("== FAIL ==\n"); return 1; }
    CHECK(!has_op(m_plain, 10u), "control: an ordinary resource emits NO OpExtension");
    CHECK(!has_capability(m_plain, 5301u) && !has_capability(m_plain, 5302u) &&
              !has_capability(m_plain, 5308u),
          "control: an ordinary resource declares NO descriptor-indexing capability");

    // --- Arms 1-3: a table-indexed resource ------------------------------------------------------
    ShaderResourceTable indexed = table_with_arity(8);
    std::vector<uint32_t> m_arr = recompile(indexed);
    CHECK(!m_arr.empty(), "a table-indexed resource recompiles");
    if (m_arr.empty()) { printf("== FAIL ==\n"); return 1; }

    ShaderResourceTable out_of_range_sgpr = indexed;
    out_of_range_sgpr.resources[0].table_index_sgpr =
        static_cast<uint32_t>(config.user_sgprs.size());
    CHECK(recompile(out_of_range_sgpr).empty(),
          "same selector-site mutation: an SGPR outside the compute push constants rejects");

    CHECK(m_arr[1] == 0x00010300u, "the module is SPIR-V 1.3, which is why the extension is required");
    CHECK(extension_name(m_arr) == "SPV_EXT_descriptor_indexing",
          "OpExtension SPV_EXT_descriptor_indexing is emitted");
    CHECK(has_capability(m_arr, 5301u), "ShaderNonUniform capability declared");
    CHECK(!has_capability(m_arr, 5302u),
          "a bounded fixed array does not request RuntimeDescriptorArray");
    CHECK(has_capability(m_arr, 5308u), "StorageBufferArrayNonUniformIndexing capability declared");

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

    // --- An absurd count must fail before it can become an unbounded descriptor access ------------
    // #2463 gives an unreadable length the sentinel UINT32_MAX. A runtime descriptor array would avoid
    // the impossible fixed declaration, but it cannot provide the concrete upper bound needed to keep
    // the selector in range. Reject the module instead of turning unreadable metadata into the most
    // permissive descriptor shape.
    ShaderResourceTable absurd = table_with_arity(0xFFFFFFFFu);
    std::vector<uint32_t> m_absurd = recompile(absurd);
    CHECK(m_absurd.empty(),
          "an implausible arity is rejected rather than emitted as an unbounded descriptor access");

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
            vb.table_selector_mode = BufferTableSelectorMode::UserSgprIndex;
            for (uint32_t index = 0; index < vb.table_index_count; ++index)
                vb.table_entries.push_back(float2_entry(0x300000u + index * 0x1000u));
            rt.resources.push_back(vb);
            return rt;
        };
        ShaderResourceTable decoy_lo = table_with_decoy(9);
        ShaderResourceTable decoy_hi = table_with_decoy(11);
        std::vector<uint32_t> m_lo = recompile(decoy_lo);
        std::vector<uint32_t> m_hi = recompile(decoy_hi);
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
        std::vector<uint32_t> m_moved = recompile(moved);
        CHECK(!m_moved.empty() && m_moved != m_lo,
              "positive control: moving the VERTEX BUFFER's index SGPR does change the module, so the "
              "equality above is the decoy being ignored rather than the SGPR being unused");
    }

    const uint32_t vs[] = {
        0x7e060280u, 0x7e0802f2u, 0xe0042000u, 0x80020100u,
        0xf80008cfu, 0x04030201u, 0xbf810000u,
    };
    CHECK(recompile_vertex(vs, std::size(vs), &indexed).empty(),
          "a graphics user-SGPR array rejects until that shell has a runtime selector source");

    std::vector<uint32_t> typed_array(std::begin(cs), std::end(cs));
    typed_array[2] = 0xe0042000u; // buffer_load_format_xy through the same s[8:11] descriptor
    CHECK(recompile_compute(typed_array.data(), typed_array.size(), &indexed, config).empty(),
          "a typed array access rejects until selected-entry swizzle/default-fill is modeled");

    std::vector<uint32_t> array_store(std::begin(cs), std::end(cs));
    array_store[2] = 0xe0700000u; // buffer_store_dword through the same descriptor
    CHECK(recompile_compute(array_store.data(), array_store.size(), &indexed, config).empty(),
          "an array store remains fail-closed without writeback authority");

    std::vector<uint32_t> array_atomic(std::begin(cs), std::end(cs));
    array_atomic[2] = 0xe0c80000u; // buffer_atomic_add through the same descriptor
    CHECK(recompile_compute(array_atomic.data(), array_atomic.size(), &indexed, config).empty(),
          "an array atomic remains fail-closed without writeback authority");

    printf(fails ? "== FAIL ==\n" : "== PASS ==\n");
    return fails ? 1 : 0;
}
