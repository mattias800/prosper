#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "build_revision.hpp"
#include "fixtures/test_scratch.h"
#include <bit>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <map>
#include <thread>

using namespace prosper::gpu;
static unsigned checks = 0, failures = 0, field_checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::printf("[FAIL] line %u: %s\n", __LINE__, #x); } } while (0)
static void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
template<class F> static std::string error(F&& f) {
    try { f(); return {}; } catch (const std::exception& e) { return e.what(); }
}
static const std::vector<uint32_t> green = {
    0x7e000280, 0x7e0202f2, 0x7e040280, 0x7e0602f2,
    0xf800180f, 0x03020100, 0xbf810000,
};
static std::string identity() {
    return std::string(prosper::embedded_build_revision()) + ':' + prosper::embedded_build_source_identity();
}
static FragmentCompileCase produce(std::vector<uint32_t> code, bool wave32 = false,
                                   const ShaderResourceTable* table = nullptr) {
    FragmentCompileCase c;
    c.compiler = identity(); c.program_address = 0x123400;
    c.code = std::move(code); c.wave32 = wave32;
    own_fragment_compile_case_resources(c, table);
    c.interpolation = fragment_interpolation_layout(c.code.data(), c.code.size());
    std::vector<uint32_t> source;
    { CompilerChoiceScope choices(c.choices); TerminalRejectCapture rejection;
      source = recompile_fragment(c.code.data(), c.code.size(), table, nullptr, UINT32_MAX,
          &c.interpolation, wave32, {RecompileDiagnosticStage::Fragment, c.program_address});
      const auto rejects = rejection.take();
      c.expected_reject = rejects.empty() ? std::string{} :
          rejects.back().first + " " + rejects.back().second; }
    finish_fragment_compile_case(c, std::move(source)); return c;
}
// Fail-closed typed oracle for this project's constant MRT fixture, independent of round-trip
// equality. Only constant/composite/copy values feeding the live location-0 Output store qualify.
static bool solid_green(const std::vector<uint32_t>& w) {
    if (w.size() < 5 || w[0] != 0x07230203) return false;
    std::map<uint32_t, std::vector<uint32_t>> value;
    std::map<uint32_t, uint32_t> storage, location, types;
    std::map<uint32_t, std::vector<uint32_t>> operands;
    uint32_t output = 0, stored = 0;
    for (size_t k = 5; k < w.size();) {
        const unsigned n = w[k] >> 16, op = w[k] & 65535;
        if (!n || n > w.size() - k) return false;
        if (op == 22 && n == 3 && w[k + 2] == 32) types[w[k + 1]] = 1; // float32
        if (op == 21 && n == 4 && w[k + 2] == 32 && w[k + 3] == 0) types[w[k + 1]] = 2; // uint32
        if (op == 23 && n == 4 && w[k + 3] == 4 && types[w[k + 2]] == 1) types[w[k + 1]] = 4;
        if (op == 43 && n == 4 && (types[w[k + 1]] == 1 || types[w[k + 1]] == 2)) value[w[k + 2]] = {w[k + 3]};
        if ((op == 80 || op == 83 || op == 124) && n >= 4) {
            if (types[w[k + 1]] == 4 || types[w[k + 1]] == 1)
                operands[w[k + 2]] = std::vector<uint32_t>(w.begin() + k + 3, w.begin() + k + n);
        }
        if (op == 59 && n >= 4) storage[w[k + 2]] = w[k + 3];
        if (op == 71 && n == 4 && w[k + 2] == 30) location[w[k + 1]] = w[k + 3];
        k += n;
    }
    for (auto [id, cls] : storage) if (cls == 3 && location.contains(id) && location[id] == 0) {
        if (output) return false; output = id;
    }
    for (size_t k = 5; k < w.size(); k += w[k] >> 16) if ((w[k] & 65535) == 62 && (w[k] >> 16) == 3 && w[k + 1] == output) {
        if (stored) return false; stored = w[k + 2];
    }
    for (unsigned iteration = 0; iteration < w[3] && !value.contains(stored); ++iteration) {
        bool progress = false;
        for (const auto& [id, inputs] : operands) if (!value.contains(id)) {
            std::vector<uint32_t> v; bool known = true;
            for (auto x : inputs) { if (!value.contains(x)) { known = false; break; }
                v.insert(v.end(), value[x].begin(), value[x].end()); }
            if (known) { value[id] = std::move(v); progress = true; }
        }
        if (!progress) break;
    }
    return output && stored && value.contains(stored) &&
        value[stored] == std::vector<uint32_t>({0, 0x3f800000, 0, 0x3f800000});
}
static std::vector<std::filesystem::path> files(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> result;
    if (std::filesystem::exists(dir)) for (auto& e : std::filesystem::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".prfc") result.push_back(e.path());
    return result;
}
static void rechecksum(std::vector<uint8_t>& bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (size_t k = 0; k + 8 < bytes.size(); ++k) { hash ^= bytes[k]; hash *= 1099511628211ull; }
    for (unsigned k = 0; k < 8; ++k) bytes[bytes.size() - 8 + k] = uint8_t(hash >> (k * 8));
}
static void codec_tests(const FragmentCompileCase& produced) {
    const auto wire = encode_fragment_compile_case(produced);
    CHECK(replay_fragment_compile_case(decode_fragment_compile_case(wire), true) == produced.source);
    CHECK(solid_green(produced.source));
    auto wrong_source = produced; wrong_source.source.back() ^= 1;
    CHECK(error([&] { replay_fragment_compile_case(wrong_source, true); }).find("SOURCE differs") != std::string::npos);
    auto wrong_raw = produced; wrong_raw.code[1] ^= 2;
    CHECK(!error([&] { replay_fragment_compile_case(wrong_raw, true); }).empty());
    auto identity_loss = produced; identity_loss.compiler = "unknown:unknown";
    CHECK(!error([&] { encode_fragment_compile_case(identity_loss); }).empty());
    auto different_compiler = produced; different_compiler.compiler = "other";
    CHECK(error([&] { replay_fragment_compile_case(different_compiler, true); }).find("identity mismatch") != std::string::npos);
    CHECK(replay_fragment_compile_case(different_compiler, false) == produced.source);
    // A declared-outcome unit control, NOT evidence of a captured refused producer. Candidate
    // must actually compile and cannot veto a Produced result because the old expectation refused.
    // The executed late-refusal mutation separately verifies a genuine two-compiler transition.
    auto declared_refusal = produced; declared_refusal.expected_produced = false;
    declared_refusal.source.clear(); declared_refusal.expected_reject = "unit declared refusal";
    CHECK(replay_fragment_compile_case(declared_refusal, false) == produced.source);
    CHECK(error([&] { replay_fragment_compile_case(declared_refusal, true); }).find("SOURCE differs") != std::string::npos);
    auto input_loss = produced; input_loss.code.clear();
    CHECK(!error([&] { replay_fragment_compile_case(input_loss, false); }).empty());
    auto trace_loss = produced; trace_loss.choices.reads.clear();
    CHECK(error([&] { replay_fragment_compile_case(trace_loss, false); }).find("trace diverged") != std::string::npos);
    auto extra_read = produced; extra_read.choices.reads.push_back({CompilerChoice::MimgSoft, 0});
    CHECK(error([&] { replay_fragment_compile_case(extra_read, false); }).find("not completely consumed") != std::string::npos);
    for (size_t length : {size_t(0), size_t(19), wire.size() - 1})
        CHECK(!error([&] { decode_fragment_compile_case(std::span(wire).first(length)); }).empty());
    for (size_t offset : {size_t(0), size_t(8), wire.size() - 12}) {
        auto bad = wire; bad[offset] ^= 0x80;
        CHECK(error([&] { decode_fragment_compile_case(bad); }).find("checksum") != std::string::npos);
    }
    auto schema = wire; schema[8] = 2; rechecksum(schema);
    CHECK(error([&] { decode_fragment_compile_case(schema); }).find("schema") != std::string::npos);
    auto boolean = wire; boolean[12] = 2; rechecksum(boolean);
    CHECK(error([&] { decode_fragment_compile_case(boolean); }).find("boolean") != std::string::npos);
    auto count = wire; for (size_t k = 13; k < 17; ++k) count[k] = 255; rechecksum(count);
    CHECK(error([&] { decode_fragment_compile_case(count); }).find("count bound") != std::string::npos);
    auto trailing = wire; trailing.insert(trailing.end() - 8, 0); rechecksum(trailing);
    CHECK(error([&] { decode_fragment_compile_case(trailing); }).find("trailing") != std::string::npos);
    auto refused = produce({0xbf810000});
    CHECK(refused.complete && !refused.expected_produced && refused.source.empty() && refused.choices.reads.empty());
    CHECK(refused.expected_reject.find("no supported export") != std::string::npos);
    CHECK(replay_fragment_compile_case(decode_fragment_compile_case(encode_fragment_compile_case(refused)), true).empty());
    CHECK(replay_fragment_compile_case(refused, false).empty());
    auto unrecorded = refused; unrecorded.expected_reject.clear();
    finish_fragment_compile_case(unrecorded, {});
    CHECK(!unrecorded.complete && unrecorded.reason == "compiler-refusal-diagnostic-unavailable");
    CHECK(error([&] { replay_fragment_compile_case(unrecorded, false); }).find("INCOMPLETE") != std::string::npos);
    auto oversized_reason = refused; oversized_reason.expected_reject.assign(257, 'x');
    finish_fragment_compile_case(oversized_reason, {});
    CHECK(!oversized_reason.complete && oversized_reason.reason == "compiler-refusal-diagnostic-budget");
    CHECK(!encode_fragment_compile_case(oversized_reason).empty());
    refused.expected_reject = "wrong refusal";
    CHECK(error([&] { replay_fragment_compile_case(refused, true); }).find("refusal diagnostic differs") != std::string::npos);
}
static void resource_tests(const FragmentCompileCase& produced) {
    ShaderResourceTable table;
    auto allocation = std::make_shared<std::vector<uint8_t>>(256, 0x5a);
    table.owned_host_data = {allocation}; table.owned_diagnostic_data = {allocation};
    ShaderResource r; r.cls = ResourceClass::ConstantBuffer; r.host_data = allocation->data() + 16;
    r.host_data_size = 128; r.host_data_prefix_bytes = 8; r.size = 128; r.binding = 5;
    r.table_entries.resize(1); r.table_entries[0].host_data = allocation->data() + 32;
    r.table_entries[0].host_data_size = 64;
    table.resources.push_back(r);
    auto c = produce(green, false, &table);
    CHECK(c.complete && solid_green(c.source));
    CHECK(c.blobs.size() == 2 && c.blobs[1].alias_of == 0 && c.blobs[0].bytes == c.blobs[1].bytes);
    CHECK(c.resources.owned_host_data[0] == c.resources.owned_diagnostic_data[0]);
    CHECK(c.resources.resources[0].host_data == c.blobs[0].bytes->data() + 16);
    (*allocation)[16] = 0xa5;
    CHECK((*c.blobs[0].bytes)[16] == 0x5a);
    auto decoded = decode_fragment_compile_case(encode_fragment_compile_case(c));
    CHECK(decoded.resources.resources[0].host_data_prefix_bytes == 8);
    CHECK(decoded.resources.resources[0].table_entries[0].host_data == decoded.blobs[0].bytes->data() + 32);
    // Public in-memory pointers are untrusted; replay rehydrates checked references into a local
    // table rather than compiling these deliberately invalid addresses.
    decoded.resources.resources[0].host_data = reinterpret_cast<uint8_t*>(uintptr_t(1));
    CHECK(replay_fragment_compile_case(decoded, true) == c.source);
    auto overflow = c; overflow.backing[0].host.offset = UINT64_MAX;
    CHECK(!error([&] { replay_fragment_compile_case(overflow, false); }).empty());
    auto missing = c; missing.backing[0].host.blob = CompileCaseBacking::kUnavailable;
    CHECK(!error([&] { replay_fragment_compile_case(missing, false); }).empty());
    auto alias = c; alias.blobs[1].alias_of = 1;
    CHECK(!error([&] { encode_fragment_compile_case(alias); }).empty());
    auto special = table; special.resources[0].indirect_buffer_contract_tag = 1;
    CHECK(!produce(green, false, &special).complete);
    auto too_many = table; too_many.resources.resize(kCompileCaseMaxResources + 1);
    FragmentCompileCase budget;
    CHECK(error([&] { own_fragment_compile_case_resources(budget, &too_many); }).find("count budget") != std::string::npos);
    auto huge = table; huge.owned_host_data[0] = std::make_shared<std::vector<uint8_t>>(kCompileCaseMaxPayloadBytes + 1);
    huge.owned_diagnostic_data.clear(); huge.resources[0].host_data = huge.owned_host_data[0]->data();
    huge.resources[0].host_data_prefix_bytes = 0; huge.resources[0].host_data_size = huge.owned_host_data[0]->size();
    CHECK(error([&] { FragmentCompileCase x; own_fragment_compile_case_resources(x, &huge); }).find("payload budget") != std::string::npos);
    // Metadata-only pixels can be huge and unowned; no tiny dummy allocation is substituted.
    ShaderResourceTable opaque_table; ShaderResource image; image.cls = ResourceClass::Texture;
    image.host_data = reinterpret_cast<uint8_t*>(uintptr_t(0x100000)); image.host_data_size = UINT64_C(1) << 40;
    image.dcc_metadata_host_data = image.host_data; image.dcc_metadata_host_data_size = 32;
    opaque_table.resources = {image};
    auto opaque = produce(green, false, &opaque_table);
    CHECK(opaque.complete && opaque.blobs.size() == 1 && opaque.blobs[0].opaque && !opaque.blobs[0].bytes);
    CHECK(opaque.backing[0].host.blob == opaque.backing[0].dcc.blob && opaque.resources.resources[0].host_data == nullptr);
    CHECK(replay_fragment_compile_case(opaque, true) == opaque.source);
    { const auto& input = opaque.resources.resources[0];
      CompilerResourceScope scope({{&input, true, false, {}}});
      CHECK(compiler_resource_has_host_data(input));
      CHECK(error([&] { compiler_resource_data(input, 1); }).find("opaque") != std::string::npos);
      CHECK(error([&] { compiler_resource_forbid_guest_read(); }).find("guest-memory") != std::string::npos);
      ShaderResource unknown;
      CHECK(error([&] { compiler_resource_has_host_data(unknown); }).find("unknown") != std::string::npos); }
    // The resource class does not turn a live guest pointer into a compiler byte dependency.
    // Ordinary unowned buffers are complete when the actual invocation consumes metadata only.
    for (const auto cls : {ResourceClass::ConstantBuffer, ResourceClass::VertexBuffer}) {
        auto live = opaque_table; live.resources[0].cls = cls;
        auto metadata_only = produce(green, false, &live);
        CHECK(metadata_only.complete && solid_green(metadata_only.source));
        CHECK(metadata_only.blobs[0].opaque && !metadata_only.blobs[0].bytes);
        CHECK(replay_fragment_compile_case(metadata_only, true) == metadata_only.source);
        const auto& input = metadata_only.resources.resources[0];
        CompilerResourceScope scope({{&input, true, false, {}}});
        CHECK(compiler_resource_has_host_data(input));
        CHECK(error([&] { compiler_resource_data(input, 4); }).find("opaque") != std::string::npos);
        // Exercise a real existing live-validation entry, not a synthetic dereference. It must
        // decline its guest reread before parsing any pointer packet under offline ownership.
        CHECK(error([&] { current_indirect_buffer_shadow_matches(metadata_only.resources,
            input, IndirectBufferShadowLayout{}, {}); }).find("guest-memory") != std::string::npos);
    }
    { const auto& input = c.resources.resources[0];
      CompilerResourceScope scope({{&input, true, true, {input.host_data, size_t(input.host_data_size)}}});
      CHECK(compiler_resource_data(input, 4) == input.host_data);
      CHECK(compiler_resource_data(input, 4)[0] == 0x5a);
      CHECK(error([&] { compiler_resource_data(input, input.host_data_size + 1); }).find("out-of-bounds") != std::string::npos); }
    // Independent per-field mutation coverage prevents the shared encoder/decoder walk from
    // making an omitted field invisible to a same-schema round trip.
    auto specimen = c; specimen.complete = false; specimen.reason = "schema-field-test";
    const auto original = encode_fragment_compile_case(specimen);
#define FIELD(member, v) do { auto x = specimen; x.resources.resources[0].member = (v); \
    auto bytes = encode_fragment_compile_case(x); CHECK(bytes != original); \
    auto y = decode_fragment_compile_case(bytes); CHECK(y.resources.resources[0].member == (v)); ++field_checks; } while (0)
    FIELD(cls, ResourceClass::Texture); FIELD(format, DataFormat::Uint32); FIELD(num_components, 4u);
    FIELD(binding, 9u); FIELD(gpu_addr, UINT64_C(0x888880)); FIELD(size, 96u); FIELD(stride, 16u);
    FIELD(srt_offset, 7u); FIELD(sgpr_base, 24u); FIELD(table_index_count, 2u); FIELD(table_entry_stride, 32u);
    FIELD(table_index_sgpr, 3u); FIELD(table_selector_mode, BufferTableSelectorMode::DynamicSbufferByteOffset);
    FIELD(table_load_pc, 17u); FIELD(direct_vsharp_sh_register_base, 77u); FIELD(fetch_pc, 11u);
    FIELD(nested_raw_snapshot_admitted, true); FIELD(fetch_index_mode, VertexFetchIndexMode::Instance);
    FIELD(bvh_box_grow, 8u); FIELD(bvh_sort_enabled, true); FIELD(flat_base_sgpr, 28u);
    FIELD(img_dim, 2u); FIELD(width, 64u); FIELD(height, 48u); FIELD(depth, 3u); FIELD(sample_count, 4u);
    FIELD(tile_mode, 5u); FIELD(linear_row_pitch_bytes, 512u); FIELD(declared_mip_levels, 3u);
    FIELD(mip_chain_element_width, 32u); FIELD(mip_chain_element_height, 16u); FIELD(mip_chain_bytes_per_block, 4u);
    FIELD(mip_chain_max_level, 5u); FIELD(mip_chain_base_level, 2u); FIELD(in_mip_tail, true);
    FIELD(mip_tail_offset, 64u); FIELD(mip_tail_bytes, 128u); FIELD(mip_tail_x, 2u); FIELD(mip_tail_y, 3u);
    FIELD(layer_stride_bytes, 512u); FIELD(layer_mip_offset_bytes, 128u); FIELD(proven_zero_mip, true);
    FIELD(srgb, true); FIELD(sampler_sgpr_base, 32u); FIELD(mag_filter, 0u); FIELD(min_filter, 0u);
    FIELD(mip_filter, 2u); FIELD(addr_uvw[0], 1u); FIELD(addr_uvw[1], 1u); FIELD(addr_uvw[2], 3u);
    FIELD(border_color_type, 2u); FIELD(min_lod, 1.0f); FIELD(max_lod, 2.0f); FIELD(lod_bias, 0.5f);
    FIELD(max_aniso_ratio, 3u); FIELD(depth_compare_func, 2u); FIELD(depth_compare, true); FIELD(unnormalized, true);
    FIELD(swizzle[0], 6u); FIELD(swizzle[1], 7u); FIELD(swizzle[2], 1u); FIELD(swizzle[3], 0u);
    FIELD(max_uncompressed_block_size, 1u); FIELD(max_compressed_block_size, 1u); FIELD(meta_pipe_aligned, true);
    FIELD(write_compress_enabled, true); FIELD(compression_enabled, true); FIELD(alpha_is_on_msb, true);
    FIELD(color_transform, true); FIELD(metadata_addr, UINT64_C(0x333330)); FIELD(dcc_metadata_size, 8u);
    FIELD(dcc_metadata_host_data_size, UINT64_C(8));
    FIELD(host_data_size, UINT64_C(100)); FIELD(host_data_prefix_bytes, UINT64_C(12));
    FIELD(atomic_x2_record_count, 3u); FIELD(scalar_raw_pointer_word_hi, 17u); FIELD(selected_sbuffer_soffset, 8u);
    FIELD(selected_sbuffer_words[0], 1u); FIELD(selected_sbuffer_words[1], 2u);
    FIELD(selected_sbuffer_words[2], 3u); FIELD(selected_sbuffer_words[3], 4u);
    FIELD(indirect_buffer_contract_tag, 1u); FIELD(indirect_buffer_binding_bytes, 8u); FIELD(indirect_buffer_slot_count, 3u);
    FIELD(indirect_buffer_header_bytes, 16u); FIELD(indirect_buffer_slot_bytes, 32u);
    FIELD(indirect_pointer_relocation.carrier_version, 1u); FIELD(indirect_pointer_relocation.proof_schema, 2u);
    FIELD(indirect_pointer_relocation.binding_bytes, 64u); FIELD(indirect_pointer_relocation.record_count, 2u);
    FIELD(indirect_pointer_relocation.segment_count, 3u); FIELD(indirect_pointer_relocation.segment_directory_byte_offset, 16u);
    FIELD(indirect_pointer_relocation.proof_fingerprint, UINT64_C(0x12345)); FIELD(scalar_buffer_dword_count, 16u);
    FIELD(raw_register_snapshot, true);
#undef FIELD
    auto invalid_enum = specimen; invalid_enum.resources.resources[0].cls = static_cast<ResourceClass>(999);
    CHECK(!error([&] { encode_fragment_compile_case(invalid_enum); }).empty());
    (void)produced;
}
static void cache_tests() {
    const auto root = prosper_test::test_scratch_dir();
    const auto cold_dir = root / "cold", warm_dir = root / "warm", missing_dir = root / "missing";
    env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", cold_dir.string().c_str()); clear_shader_recompile_cache();
    ShaderResourceTable table; ShaderResource r; r.gpu_addr = 0x111000; r.size = 128; table.resources.push_back(r);
    auto code = green;
    auto cold = recompile_graphics_shader_cached(ShaderProgramStage::Fragment, code.data(), code.size(), &table);
    CHECK(solid_green(cold)); CHECK(files(cold_dir).size() == 1);
    if (files(cold_dir).empty()) return;
    const auto producer = read_fragment_compile_case(files(cold_dir)[0]);
    CHECK(producer.complete && producer.code == code && producer.resources.resources[0].gpu_addr == 0x111000);
    const auto before = shader_recompile_cache_stats();
    table.resources[0].gpu_addr = 0x222000; table.resources[0].size = 256;
    env("PROSPER_FS_TAP", "0:0"); env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", warm_dir.string().c_str());
    auto warm = recompile_graphics_shader_cached(ShaderProgramStage::Fragment, code.data(), code.size(), &table);
    const auto after = shader_recompile_cache_stats();
    CHECK(warm == cold && after.hits == before.hits + 1 && after.misses == before.misses);
    CHECK(files(warm_dir).size() == 1);
    if (!files(warm_dir).empty()) {
        const auto retained = read_fragment_compile_case(files(warm_dir)[0]);
        CHECK(encode_fragment_compile_case(retained) == encode_fragment_compile_case(producer));
        CHECK(replay_fragment_compile_case(retained, true) == cold);
    }
    auto same = recompile_graphics_shader_cached(ShaderProgramStage::Fragment, code.data(), code.size(), &table);
    CHECK(same == warm && files(warm_dir).size() == 1);
    env("PROSPER_FS_TAP", nullptr);
    env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr); clear_shader_recompile_cache();
    (void)recompile_graphics_shader_cached(ShaderProgramStage::Fragment, code.data(), code.size(), &table);
    env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", missing_dir.string().c_str());
    (void)recompile_graphics_shader_cached(ShaderProgramStage::Fragment, code.data(), code.size(), &table);
    CHECK(files(missing_dir).size() == 1);
    if (!files(missing_dir).empty()) {
        const auto missing = read_fragment_compile_case(files(missing_dir)[0]);
        CHECK(!missing.complete && missing.reason == "producing-context-not-retained");
        CHECK(error([&] { replay_fragment_compile_case(missing, false); }).find("INCOMPLETE") != std::string::npos);
    }
    const auto bypass_dir = root / "bypass"; env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", bypass_dir.string().c_str());
    env("PROSPER_NO_SHADER_CACHE", "1");
    auto bypass = recompile_graphics_shader_cached(ShaderProgramStage::Fragment, code.data(), code.size(), &table);
    CHECK(bypass == cold && files(bypass_dir).size() == 1);
    env("PROSPER_NO_SHADER_CACHE", nullptr);
    // Project-owned getpc/V# fixture proves that the producing cache retains compiler-consumed
    // table bytes beyond ENDPGM, rather than capturing only the decoded instruction prefix.
    const std::vector<uint32_t> pcrel = {0xbe841f00u,0x800404b0u,0x82050580u,0xbe860390u,
        0xbe8703ffu,0x10005004u,0x7e020280u,0xe0301000u,0x80010101u,0xbf8c3f70u,
        0xf800180fu,0x01010101u,0xbf810000u,7u,11u,13u,17u};
    const auto tail_dir = root / "tail"; env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", tail_dir.string().c_str());
    ShaderResourceTable empty;
    auto with_tail = recompile_graphics_shader_cached(ShaderProgramStage::Fragment, pcrel.data(), pcrel.size(), &empty);
    CHECK(!with_tail.empty() && files(tail_dir).size() == 1);
    if (!files(tail_dir).empty()) {
        const auto retained = read_fragment_compile_case(files(tail_dir)[0]);
        CHECK(retained.complete && retained.code == pcrel && replay_fragment_compile_case(retained, true) == with_tail);
        auto lost = retained; lost.code.resize(13);
        CHECK(!error([&] { replay_fragment_compile_case(lost, true); }).empty());
    }
    env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr);
}
static void atomic_tests(const FragmentCompileCase& c) {
    const auto path = prosper_test::test_scratch_path("atomic.prfc"); write_fragment_compile_case(path, c);
    auto invalid = c; invalid.code.clear();
    CHECK(!error([&] { write_fragment_compile_case(path, invalid); }).empty());
    CHECK(encode_fragment_compile_case(read_fragment_compile_case(path)) == encode_fragment_compile_case(c));
    auto other = produce(green, true);
    CHECK(!error([&] { write_fragment_compile_case(path, other, false); }).empty());
    CHECK(encode_fragment_compile_case(read_fragment_compile_case(path)) == encode_fragment_compile_case(c));
    std::atomic<unsigned> errors{0};
    std::thread a([&] { for (unsigned k = 0; k < 8; ++k) if (!error([&] { write_fragment_compile_case(path, c); }).empty()) ++errors; });
    std::thread b([&] { for (unsigned k = 0; k < 8; ++k) if (!error([&] { write_fragment_compile_case(path, other); }).empty()) ++errors; });
    a.join(); b.join(); CHECK(errors == 0);
    const auto final = encode_fragment_compile_case(read_fragment_compile_case(path));
    CHECK(final == encode_fragment_compile_case(c) || final == encode_fragment_compile_case(other));
    for (const auto& item : std::filesystem::directory_iterator(path.parent_path()))
        CHECK(item.path().filename().string().find("atomic.prfc.tmp-") == std::string::npos);
}
int main(int argc, char** argv) {
    try {
        env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr); env("PROSPER_FS_TAP", nullptr);
        if (argc == 3 && std::string_view(argv[1]) == "--emit-attempt") {
            // Captures the real producing outcome without prescribing Produced versus Refused.
            // This supports an executed late-refusal mutation/restored-compiler transition check.
            auto attempt = produce(green); CHECK(attempt.complete);
            write_fragment_compile_case(argv[2], attempt);
        } else if (argc == 3 && std::string_view(argv[1]) == "--emit") {
            const std::filesystem::path dir = argv[2]; std::filesystem::create_directories(dir);
            for (bool wave32 : {false, true}) {
                auto c = produce(green, wave32); CHECK(c.complete && solid_green(c.source));
                write_fragment_compile_case(dir / (wave32 ? "wave32.prfc" : "wave64.prfc"), c);
            }
            auto rejected = produce({0xbf810000}); CHECK(rejected.complete && rejected.source.empty());
            write_fragment_compile_case(dir / "refused.prfc", rejected);
            auto incomplete = produce(green); incomplete.complete = false; incomplete.reason = "input-loss-control";
            write_fragment_compile_case(dir / "incomplete.prfc", incomplete);
            auto mismatch = produce(green); mismatch.compiler = "different-compiler";
            write_fragment_compile_case(dir / "different.prfc", mismatch);
            auto broken_source = produce(green); broken_source.source.back() ^= 1;
            write_fragment_compile_case(dir / "wrong-source.prfc", broken_source);
        } else {
            CHECK(argc == 1 || (argc == 3 && std::string_view(argv[1]) == "--dump"));
            auto c = produce(green); CHECK(c.complete);
            codec_tests(c); resource_tests(c); cache_tests(); atomic_tests(c);
            auto wave32 = produce(green, true); CHECK(wave32.complete && solid_green(wave32.source));
            CHECK(wave32.wave32 && !c.wave32); // width need not alter a wave-insensitive program
            auto unknown = c; unknown.float_mode_available = false; unknown.float_mode = 1;
            CHECK(!error([&] { encode_fragment_compile_case(unknown); }).empty());
            if (argc == 3) { const std::filesystem::path dir = argv[2]; std::filesystem::create_directories(dir);
                write_fragment_compile_source(dir / "wave64.spv", c.source);
                write_fragment_compile_source(dir / "wave32.spv", wave32.source); }
        }
    } catch (const std::exception& e) { ++failures; std::printf("[FAIL] unexpected: %s\n", e.what()); }
    std::printf("fragment compile case: %u checks, %u independent resource fields, %u failures\n", checks, field_checks, failures);
    return failures ? 1 : 0;
}
