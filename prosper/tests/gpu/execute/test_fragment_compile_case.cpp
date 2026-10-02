#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "diagnostics/perf/perf_ledger.hpp"
#include "build_revision.hpp"
#include "fixtures/test_scratch.h"
#include <bit>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <map>
#include <thread>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

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
template<class F> static std::string capture_stderr(F&& body) {
    const auto path = prosper_test::test_scratch_path("case-arithmetic.log");
    FILE* output = std::fopen(path.string().c_str(), "w+b");
    CHECK(output != nullptr); if (!output) return {};
    std::fflush(stderr);
#ifdef _WIN32
    const int fd = _fileno(stderr), saved = _dup(fd);
    const bool redirected = saved >= 0 && _dup2(_fileno(output), fd) == 0;
#else
    const int fd = fileno(stderr), saved = dup(fd);
    const bool redirected = saved >= 0 && dup2(fileno(output), fd) >= 0;
#endif
    CHECK(redirected);
    const auto restore = [&] {
        std::fflush(stderr);
#ifdef _WIN32
        if (saved >= 0) { CHECK(_dup2(saved, fd) == 0); _close(saved); }
#else
        if (saved >= 0) { CHECK(dup2(saved, fd) >= 0); close(saved); }
#endif
    };
    try { if (redirected) body(); } catch (...) { restore(); std::fclose(output); throw; }
    restore(); std::rewind(output);
    std::string text; char bytes[4096];
    for (size_t n; (n = std::fread(bytes, 1, sizeof bytes, output)) != 0;) text.append(bytes, n);
    std::fclose(output); std::filesystem::remove(path); return text;
}
static const std::vector<uint32_t> green = {
    0x7e000280, 0x7e0202f2, 0x7e040280, 0x7e0602f2,
    0xf800180f, 0x03020100, 0xbf810000,
};
// Defined raw integer literal -> ordered F32 zero compare -> live MRT0.r. No float Input,
// runtime memory or host FP is involved: guest preserve compares subnormal != zero; flush equals.
static const std::vector<uint32_t> mode_probe = {
    0x7e0002ffu, 1u,                    // v0 raw bits = smallest positive subnormal
    0xd402006au, 128u | (256u << 9),   // v_cmp_eq_f32_e64 VCC, 0, v0
    0xd5010000u, 128u | (242u << 9) | (106u << 18), // v0 = VCC ? 1.0 : 0.0
    0x7e0202f2u, 0x7e040280u, 0x7e0602f2u,
    0xf800180fu, 0x03020100u, 0xbf810000u,
};
static const std::vector<uint32_t> add_probe = {
    0x7e0002ffu, 0x3f800000u, 0x7e0202ffu, 0x40000000u, // v0=1, v1=2
    0x06000300u,                                        // v_add_f32 v0,v0,v1
    0x7e020280u, 0x7e040280u, 0x7e0602f2u,
    0xf800180fu, 0x03020100u, 0xbf810000u,
};
static std::string identity() {
    return std::string(prosper::embedded_build_revision()) + ':' + prosper::embedded_build_source_identity();
}
static FragmentCompileCase produce(std::vector<uint32_t> code, bool wave32 = false,
                                   const ShaderResourceTable* table = nullptr,
                                   FragmentFloatMode mode = {}, FloatTransportConfig transport = {}) {
    FragmentCompileCase c;
    c.compiler = identity(); c.program_address = 0x123400;
    c.code = std::move(code); c.wave32 = wave32; c.float_mode = mode;
    c.float_transport = transport;
    own_fragment_compile_case_resources(c, table);
    c.interpolation = fragment_interpolation_layout(c.code.data(), c.code.size());
    std::vector<uint32_t> source;
    { CompilerChoiceScope choices(c.choices); TerminalRejectCapture rejection;
      source = recompile_fragment(c.code.data(), c.code.size(), table, nullptr, UINT32_MAX,
          &c.interpolation, wave32, {RecompileDiagnosticStage::Fragment, c.program_address},
          mode, nullptr, transport);
      const auto rejects = rejection.take();
      c.expected_reject = rejects.empty() ? std::string{} :
          rejects.back().first + " " + rejects.back().second; }
    finish_fragment_compile_case(c, std::move(source)); return c;
}
// Fail-closed typed oracle for this project's constant MRT fixture, independent of round-trip
// equality. Only defined constants, a bounded typed UInt32/Boolean relation, selects and
// composites feeding the live location-0 Output qualify. Only the exact normal 1+2 ADD fixture
// is evaluated; arbitrary FP, Input and Undef dependencies fail. This is not a host FP interpreter.
static bool constant_color(const std::vector<uint32_t>& w, const std::vector<uint32_t>& expected) {
    if (w.size() < 5 || w[0] != 0x07230203) return false;
    std::map<uint32_t, std::vector<uint32_t>> value;
    std::map<uint32_t, uint32_t> storage, location, types;
    std::map<uint32_t, uint32_t> kind, operation;
    std::map<uint32_t, std::vector<uint32_t>> operands;
    uint32_t output = 0, stored = 0;
    for (size_t k = 5; k < w.size();) {
        const unsigned n = w[k] >> 16, op = w[k] & 65535;
        if (!n || n > w.size() - k) return false;
        if (op == 22 && n == 3 && w[k + 2] == 32) types[w[k + 1]] = 1; // float32
        if (op == 21 && n == 4 && w[k + 2] == 32 && w[k + 3] == 0) types[w[k + 1]] = 2; // uint32
        if (op == 20 && n == 2) types[w[k + 1]] = 3; // Boolean
        if (op == 23 && n == 4 && w[k + 3] == 4 && types[w[k + 2]] == 1) types[w[k + 1]] = 4;
        if (op == 43 && n == 4 && (types[w[k + 1]] == 1 || types[w[k + 1]] == 2)) {
            value[w[k + 2]] = {w[k + 3]}; kind[w[k + 2]] = types[w[k + 1]];
        }
        if ((op == 41 || op == 42) && n == 3 && types[w[k + 1]] == 3) {
            value[w[k + 2]] = {op == 41 ? 1u : 0u}; kind[w[k + 2]] = 3;
        }
        if ((op == 80 || op == 83 || op == 124 || op == 129 || op == 168 || op == 169 ||
             op == 170 || op == 171 || op == 174 || op == 199) && n >= 4) {
            if (types[w[k + 1]] >= 1 && types[w[k + 1]] <= 4) {
                operands[w[k + 2]] = std::vector<uint32_t>(w.begin() + k + 3, w.begin() + k + n);
                kind[w[k + 2]] = types[w[k + 1]]; operation[w[k + 2]] = op;
            }
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
            if (!known) continue;
            const auto op = operation[id], result_kind = kind[id];
            bool typed = false;
            if (op == 80) typed = result_kind == 4 && inputs.size() == 4 && v.size() == 4 &&
                std::all_of(inputs.begin(), inputs.end(), [&](auto x) { return kind[x] == 1 && value[x].size() == 1; });
            else if (op == 83 || op == 124) typed = inputs.size() == 1 &&
                (op == 83 ? kind[inputs[0]] == result_kind :
                    (result_kind == 1 && kind[inputs[0]] == 2) || (result_kind == 2 && kind[inputs[0]] == 1));
            else if (op == 129 && result_kind == 1 && inputs.size() == 2 && v.size() == 2 &&
                     kind[inputs[0]] == 1 && kind[inputs[1]] == 1 && v[0] == 0x3f800000u && v[1] == 0x40000000u) {
                typed = true; v = {0x40400000u}; // exact binary32 1+2=3, without executing host FP
            }
            else if (op == 168 && result_kind == 3 && inputs.size() == 1 && v.size() == 1 && kind[inputs[0]] == 3) {
                typed = true; v[0] = !v[0];
            } else if (op == 169 && inputs.size() == 3 && kind[inputs[0]] == 3 && value[inputs[0]].size() == 1 &&
                       kind[inputs[1]] == result_kind && kind[inputs[2]] == result_kind) {
                typed = true; v = value[inputs[value[inputs[0]][0] ? 1 : 2]];
            } else if (inputs.size() == 2 && v.size() == 2 && kind[inputs[0]] == 2 && kind[inputs[1]] == 2) {
                if (op == 199 && result_kind == 2) { typed = true; v = {v[0] & v[1]}; }
                if (result_kind == 3 && (op == 170 || op == 171 || op == 174)) {
                    typed = true; v = {uint32_t(op == 170 ? v[0] == v[1] : op == 171 ? v[0] != v[1] : v[0] >= v[1])};
                }
            }
            if (typed) { value[id] = std::move(v); progress = true; }
        }
        if (!progress) break;
    }
    return output && stored && value.contains(stored) &&
        value[stored] == expected;
}
static bool solid_green(const std::vector<uint32_t>& w) {
    return constant_color(w, {0, 0x3f800000, 0, 0x3f800000});
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
    auto schema = wire; schema[8] = 3; rechecksum(schema);
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
    std::string actual_refusal;
    CHECK(replay_fragment_compile_case(refused, false, &actual_refusal).empty() && actual_refusal == refused.expected_reject);
    CHECK(replay_fragment_compile_case(produced, false, &actual_refusal) == produced.source && actual_refusal.empty());
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
static void mode_tests() {
    const auto compare = rdna2_decode_one(mode_probe.data() + 2, 2);
    CHECK(compare.fmt == Rdna2Format::VOPC && compare.opcode == 2 && compare.dst.value == 106 &&
          compare.src[0].kind == OperandKind::InlineInt && compare.src[0].value == 0 &&
          compare.src[1].kind == OperandKind::VGPR && compare.src[1].value == 0);
    for (const bool wave32 : {false, true}) {
        const auto flush = produce(mode_probe, wave32, nullptr, {true, 0});
        const auto preserve = produce(mode_probe, wave32, nullptr, {true, 16});
        CHECK(flush.complete && preserve.complete && flush.source != preserve.source);
        CHECK(constant_color(flush.source, {0x3f800000, 0x3f800000, 0, 0x3f800000}));
        CHECK(solid_green(preserve.source));
        CHECK(replay_fragment_compile_case(decode_fragment_compile_case(encode_fragment_compile_case(flush)), true) == flush.source);
        CHECK(replay_fragment_compile_case(decode_fragment_compile_case(encode_fragment_compile_case(preserve)), true) == preserve.source);
        auto lost_mode = flush; lost_mode.float_mode = preserve.float_mode;
        CHECK(error([&] { replay_fragment_compile_case(lost_mode, true); }).find("SOURCE differs") != std::string::npos);
    }
    for (unsigned byte = 0; byte < 256; ++byte) {
        const FragmentFloatMode mode{true, uint8_t(byte)};
        const auto c = produce(green, false, nullptr, mode);
        const auto decoded = decode_fragment_compile_case(encode_fragment_compile_case(c));
        CHECK(c.complete && decoded.float_mode == mode && replay_fragment_compile_case(decoded, true) == c.source);
    }
    const auto directory = prosper_test::test_scratch_dir() / "mode";
    clear_shader_recompile_cache(); env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", directory.string().c_str());
    const auto compile = [&](FragmentFloatMode mode) {
        return recompile_graphics_shader_cached(ShaderProgramStage::Fragment, mode_probe.data(), mode_probe.size(),
            nullptr, nullptr, nullptr, nullptr, false, 0, false, {}, mode);
    };
    const auto flush = compile({true, 0}), preserve = compile({true, 16});
    const auto before = shader_recompile_cache_stats();
    CHECK(flush != preserve && solid_green(preserve));
    CHECK(compile({true, 0}) == flush && shader_recompile_cache_stats().hits == before.hits + 1);
    CHECK(files(directory).size() == 2);
    for (const auto& path : files(directory)) {
        const auto c = read_fragment_compile_case(path);
        CHECK(c.complete && c.float_mode.available && (c.float_mode.value == 0 || c.float_mode.value == 16));
        CHECK(c.source == (c.float_mode.value == 0 ? flush : preserve) && replay_fragment_compile_case(c, true) == c.source);
    }
    env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr);
}
static void transport_tests() {
    reset_float_controls_support_for_test();
    constexpr FloatTransportConfig profiles[] = {{FloatTransportProfile::Unknown},
        {FloatTransportProfile::Implicit}, {FloatTransportProfile::ExplicitNonFinite32}};
    const std::vector<uint32_t> color = {0x40400000u, 0, 0, 0x3f800000u};
    for (bool wave32 : {false, true}) for (const auto profile : profiles) {
        const auto c = produce(add_probe, wave32, nullptr, {true, 0x31}, profile);
        CHECK(c.complete && constant_color(c.source, color));
        const auto wire = encode_fragment_compile_case(c);
        CHECK(wire.size() > 21 && wire[8] == 2 && wire[wire.size() - 9] == uint8_t(profile.profile));
        const auto decoded = decode_fragment_compile_case(wire);
        CHECK(decoded.complete && decoded.float_transport == profile && decoded.code == c.code);
        CHECK(replay_fragment_compile_case(decoded, true) == c.source);
        unsigned legacy_reads = 0;
        for (const auto& read : c.choices.reads) if (read.choice == CompilerChoice::FloatControls) ++legacy_reads;
        CHECK(legacy_reads == (profile.explicit_nonfinite32() ? 0u : 1u));
        CHECK(float_transport_module_supported(c.source.data(), c.source.size(),
            {FloatTransportProfile::Implicit}) == !profile.explicit_nonfinite32());

        auto legacy = wire;
        legacy.erase(legacy.end() - 9); // genuine schema-1 payload, NOT a version-only relabel
        legacy[8] = 1; rechecksum(legacy);
        const auto old = decode_fragment_compile_case(legacy);
        CHECK(!old.complete && old.reason == "fragment-transport-config-unavailable" &&
            old.float_transport == FloatTransportConfig{} && old.source == c.source && old.code == c.code);
        CHECK(error([&] { replay_fragment_compile_case(old, false); }).find("INCOMPLETE") != std::string::npos);
        auto relabel = wire; relabel[8] = 1; rechecksum(relabel);
        CHECK(error([&] { decode_fragment_compile_case(relabel); }).find("trailing data") != std::string::npos);
        auto missing = wire; missing.erase(missing.end() - 9); rechecksum(missing);
        CHECK(error([&] { decode_fragment_compile_case(missing); }).find("truncated") != std::string::npos);
        auto invalid = wire; invalid[invalid.size() - 9] = 3; rechecksum(invalid);
        CHECK(error([&] { decode_fragment_compile_case(invalid); }).find("noncanonical float transport") != std::string::npos);
        auto invalid_input = c; invalid_input.float_transport.profile = static_cast<FloatTransportProfile>(3);
        CHECK(error([&] { encode_fragment_compile_case(invalid_input); }).find("noncanonical float transport") != std::string::npos);
        if (profile.explicit_nonfinite32()) {
            auto loss = c; loss.float_transport = {};
            CHECK(!error([&] { replay_fragment_compile_case(loss, false); }).empty());
        }
    }
    // The legacy ambient read remains captured only when consumed, and replay does not replace
    // that producing choice with a later publication. Explicit attempts are independent of it.
    const auto implicit = produce(add_probe, false, nullptr, {true, 0x31}, profiles[1]);
    const auto explicit_case = produce(add_probe, false, nullptr, {true, 0x31}, profiles[2]);
    publish_float_controls_support(true, true);
    CHECK(replay_fragment_compile_case(implicit, true) == implicit.source);
    CHECK(replay_fragment_compile_case(explicit_case, true) == explicit_case.source);
    reset_float_controls_support_for_test();

    const auto directory = prosper_test::test_scratch_dir() / "transport";
    clear_shader_recompile_cache(); env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", directory.string().c_str());
    const auto compile = [&](FloatTransportConfig profile) {
        return recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
            add_probe.data(), add_probe.size(), nullptr, nullptr, nullptr, nullptr,
            false, 0, false, {}, {true, 0x31}, profile);
    };
    std::array<SharedShaderWords, 3> outputs;
    for (size_t k = 0; k < 3; ++k) { outputs[k] = compile(profiles[k]); CHECK(outputs[k] && constant_color(*outputs[k], color)); }
    CHECK(outputs[0] != outputs[1] && outputs[1] != outputs[2] && outputs[0] != outputs[2]);
    const auto before = shader_recompile_cache_stats();
    for (size_t k = 0; k < 3; ++k) CHECK(compile(profiles[k]) == outputs[k]);
    const auto after = shader_recompile_cache_stats();
    CHECK(after.hits == before.hits + 3 && after.misses == before.misses);
    const auto captured = files(directory); CHECK(captured.size() == 3);
    unsigned seen = 0;
    for (const auto& path : captured) {
        const auto c = read_fragment_compile_case(path);
        const auto k = size_t(c.float_transport.profile);
        CHECK(c.complete && k < 3);
        if (k < 3) { seen |= 1u << k; CHECK(c.source == *outputs[k] && replay_fragment_compile_case(c, true) == c.source); }
    }
    CHECK(seen == 7);
    env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr); clear_shader_recompile_cache();
}
static std::vector<uint32_t> arithmetic_accounting_tests() {
    namespace perf = prosper::diagnostics::perf;
    const auto count = [](perf::Counter counter) { return perf::ledger().counters[size_t(counter)].load(); };
    const auto requests = [&] { return count(perf::Counter::FragmentArithmeticRequests); };
    const auto additions = [&] { return count(perf::Counter::FragmentArithmeticAddRequests); };
    const auto color = std::vector<uint32_t>{0x40400000u, 0, 0, 0x3f800000u};
    const auto decode = rdna2_decode_one(add_probe.data() + 4, 1);
    CHECK(decode.fmt == Rdna2Format::VOP2 && decode.opcode == 3 && decode.dst.value == 0);
    clear_shader_recompile_cache();
    const auto directory = prosper_test::test_scratch_dir() / "arithmetic";
    env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", directory.string().c_str());
    auto code = add_probe, alias = code;
    const auto compile = [](const std::vector<uint32_t>& input) {
        return recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
            input.data(), input.size(), nullptr, nullptr, nullptr, nullptr, false, 0, false, {}, {true, 0x31});
    };
    const auto before = requests(), add_before = additions();
    SharedShaderWords cold, warm, bypass;
    const auto cold_log = capture_stderr([&] { cold = compile(code); });
    CHECK(cold && !cold->empty() && constant_color(*cold, color));
    if (!cold || cold->empty()) { env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr); return {}; }
    CHECK(requests() == before + 1 && additions() == add_before + 1); // owned auto baseline adds zero
    CHECK(cold_log.find("family=F32-ADD") != std::string::npos && cold_log.find("pc=4 ") != std::string::npos);
    const auto recorded = files(directory);
    CHECK(recorded.size() == 1);
    if (recorded.empty()) { env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr); return {}; }
    const auto saved = read_fragment_compile_case(recorded.front());
    CHECK(saved.complete && saved.source == *cold && saved.float_mode == FragmentFloatMode({true, 0x31}));
    const auto warm_log = capture_stderr([&] { warm = compile(alias); });
    CHECK(warm == cold && requests() == before + 2 && additions() == add_before + 2);
    const auto address = [](const void* pointer) {
        char text[32]; std::snprintf(text, sizeof text, "0x%llx", (unsigned long long)reinterpret_cast<uintptr_t>(pointer));
        return std::string(text);
    };
    CHECK(warm_log.find("program=" + address(alias.data()) + " ") != std::string::npos &&
          warm_log.find("producing-program=" + address(code.data()) + " ") != std::string::npos &&
          warm_log.find("family=F32-ADD") != std::string::npos);
    env("PROSPER_NO_SHADER_CACHE", "1");
    capture_stderr([&] { bypass = compile(alias); });
    env("PROSPER_NO_SHADER_CACHE", nullptr);
    CHECK(bypass && *bypass == *cold && requests() == before + 3 && additions() == add_before + 3);
    capture_stderr([&] { CHECK(replay_fragment_compile_case(saved, true) == *cold); });
    CHECK(requests() == before + 4 && additions() == add_before + 4); // explicit replay remains visible
    std::array<std::vector<uint32_t>, 8> aliases;
    for (auto& input : aliases) input = code;
    std::atomic<bool> same{true}; std::vector<std::thread> threads;
    capture_stderr([&] {
        for (auto& input : aliases) threads.emplace_back([&, input_pointer = &input] {
            if (compile(*input_pointer) != cold) same.store(false);
        });
        for (auto& thread : threads) thread.join();
    });
    CHECK(same.load() && requests() == before + 12 && additions() == add_before + 12);
    const auto exported = files(directory);
    for (const auto& input : aliases) {
        char prefix[32]; std::snprintf(prefix, sizeof prefix, "f_%016llx_",
            (unsigned long long)reinterpret_cast<uintptr_t>(input.data()));
        unsigned matches = 0;
        for (const auto& path : exported) if (path.filename().string().starts_with(prefix)) {
            ++matches;
            const auto concurrent_case = read_fragment_compile_case(path);
            CHECK(concurrent_case.complete && concurrent_case.program_address == saved.program_address &&
                  concurrent_case.code == saved.code && concurrent_case.source == saved.source &&
                  concurrent_case.float_mode == saved.float_mode && concurrent_case.compiler == saved.compiler);
        }
        CHECK(matches == 1); // every lookup exports the same immutable producing context
    }
    env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr);
    auto wrong = *cold;
    bool changed = false;
    for (size_t k = 5; k < wrong.size(); k += wrong[k] >> 16) if ((wrong[k] & 65535u) == 129) {
        wrong[k] = (wrong[k] & 0xffff0000u) | 133u; changed = true; break;
    }
    CHECK(changed && !constant_color(wrong, color)); // same operands but MUL cannot prove the ADD output
    return *cold;
}
int main(int argc, char** argv) {
    try {
        env("PROSPER_FRAGMENT_COMPILE_CASE_DIR", nullptr); env("PROSPER_FS_TAP", nullptr);
        env("PROSPER_NO_PERF_ALARMS", nullptr);
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
            for (uint8_t mode : {uint8_t(0), uint8_t(16)}) {
                auto c = produce(mode_probe, false, nullptr, {true, mode}); CHECK(c.complete);
                CHECK(constant_color(c.source, {mode ? 0u : 0x3f800000u, 0x3f800000u, 0, 0x3f800000u}));
                write_fragment_compile_case(dir / (mode ? "preserve.prfc" : "flush.prfc"), c);
            }
            for (const auto profile : {FloatTransportProfile::Unknown, FloatTransportProfile::Implicit,
                                      FloatTransportProfile::ExplicitNonFinite32}) {
                auto c = produce(add_probe, false, nullptr, {true, 0x31}, {profile}); CHECK(c.complete);
                write_fragment_compile_case(dir / (std::string("transport-") +
                    float_transport_profile_name({profile}) + ".prfc"), c);
            }
        } else {
            CHECK(argc == 1 || (argc == 3 && std::string_view(argv[1]) == "--dump"));
            auto c = produce(green); CHECK(c.complete);
            codec_tests(c); resource_tests(c); cache_tests(); atomic_tests(c); mode_tests(); transport_tests();
            const auto add = arithmetic_accounting_tests();
            auto wave32 = produce(green, true); CHECK(wave32.complete && solid_green(wave32.source));
            CHECK(wave32.wave32 && !c.wave32); // width need not alter a wave-insensitive program
            auto unknown = c; unknown.float_mode = {false, 1};
            CHECK(!error([&] { encode_fragment_compile_case(unknown); }).empty());
            if (argc == 3) { const std::filesystem::path dir = argv[2]; std::filesystem::create_directories(dir);
                write_fragment_compile_source(dir / "wave64.spv", c.source);
                write_fragment_compile_source(dir / "wave32.spv", wave32.source);
                write_fragment_compile_source(dir / "arithmetic-add.spv", add);
                for (bool narrow : {false, true}) for (uint8_t mode : {uint8_t(0), uint8_t(16)}) {
                    auto known = produce(mode_probe, narrow, nullptr, {true, mode}); CHECK(known.complete);
                    write_fragment_compile_source(dir / (std::string("mode") + std::to_string(mode) +
                        "-wave" + (narrow ? "32" : "64") + ".spv"), known.source);
                }
                for (bool narrow : {false, true}) for (const auto profile : {
                    FloatTransportProfile::Unknown, FloatTransportProfile::Implicit,
                    FloatTransportProfile::ExplicitNonFinite32}) {
                    auto known = produce(add_probe, narrow, nullptr, {true, 0x31}, {profile}); CHECK(known.complete);
                    write_fragment_compile_source(dir / (std::string("transport-") +
                        float_transport_profile_name({profile}) + "-wave" + (narrow ? "32" : "64") + ".spv"), known.source);
                }
            }
        }
    } catch (const std::exception& e) { ++failures; std::printf("[FAIL] unexpected: %s\n", e.what()); }
    std::printf("fragment compile case: %u checks, %u independent resource fields, %u failures\n", checks, field_checks, failures);
    return failures ? 1 : 0;
}
