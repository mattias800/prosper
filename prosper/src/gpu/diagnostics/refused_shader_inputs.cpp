#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/diagnostics/refused_graphics_inputs.hpp"

#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <system_error>

namespace prosper::gpu {
namespace {

constexpr char kMagic[8] = {'P', 'R', 'R', 'S', 'I', 'N', 'P', 0};
constexpr size_t kMaxFileBytes = kCompileCaseMaxFileBytes + 64 * 1024;
constexpr size_t kMaxText = 512;
constexpr const char* kCarrierReason = "refused-shader-inputs-carrier";

void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}

uint64_t fnv(const uint8_t* bytes, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    for (size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

struct Out {
    std::vector<uint8_t> bytes;
    template <class T>
    void num(T v) {
        uint64_t n = static_cast<uint64_t>(v);
        for (size_t k = 0; k < sizeof(T); ++k) bytes.push_back(uint8_t(n >> (8 * k)));
    }
    void text(const std::string& s) {
        require(s.size() <= kMaxText, "refused-shader inputs text bound");
        num<uint32_t>(uint32_t(s.size()));
        bytes.insert(bytes.end(), s.begin(), s.end());
    }
    void blob(const std::vector<uint8_t>& b) {
        num<uint32_t>(uint32_t(b.size()));
        bytes.insert(bytes.end(), b.begin(), b.end());
    }
};

struct HashOut {
    uint64_t hash = 0xcbf29ce484222325ull;
    template <class T>
    void num(T value) {
        const uint64_t bits = static_cast<uint64_t>(value);
        for (size_t i = 0; i != sizeof(T); ++i) {
            hash ^= uint8_t(bits >> (8 * i));
            hash *= 0x100000001b3ull;
        }
    }
};

struct In {
    const std::vector<uint8_t>& bytes;
    size_t end;
    size_t at = 0;
    template <class T>
    T num() {
        require(sizeof(T) <= end - at, "truncated refused-shader inputs");
        uint64_t n = 0;
        for (size_t k = 0; k < sizeof(T); ++k) n |= uint64_t(bytes[at++]) << (8 * k);
        return static_cast<T>(n);
    }
    bool flag() {
        const uint8_t v = num<uint8_t>();
        require(v <= 1, "refused-shader inputs boolean");
        return v != 0;
    }
    std::string text() {
        const uint32_t n = num<uint32_t>();
        require(n <= kMaxText && n <= end - at, "refused-shader inputs text bound");
        std::string s(reinterpret_cast<const char*>(bytes.data() + at), n);
        at += n;
        return s;
    }
    std::vector<uint8_t> blob() {
        const uint32_t n = num<uint32_t>();
        require(n <= end - at, "truncated refused-shader inputs carrier");
        std::vector<uint8_t> b(bytes.begin() + ptrdiff_t(at), bytes.begin() + ptrdiff_t(at + n));
        at += n;
        return b;
    }
};

// Original config fields in their schema-1 order. The schema-4 launch record follows the resource
// carrier, and joins the same input-set identity. The round-trip test sets every field.
// The schema-1 config prefix is immutable. New launch words follow the resource carrier in schema4.
template <class Sink>
void write_config(Sink& o, const ComputeShaderConfig& c) {
    o.template num<uint32_t>(uint32_t(c.user_sgprs.size()));
    for (uint32_t v : c.user_sgprs) o.num(v);
    o.num(c.local_x), o.num(c.local_y), o.num(c.local_z);
    o.template num<uint8_t>(c.exact_thread_extent);
    o.num(c.threads_x), o.num(c.threads_y), o.num(c.threads_z);
    o.num(c.wave_size), o.num(c.compute_pgm_rsrc1), o.num(c.tidig_comp_cnt);
    o.template num<uint8_t>(c.tgid_x_en), o.template num<uint8_t>(c.tgid_y_en),
        o.template num<uint8_t>(c.tgid_z_en);
    o.template num<uint8_t>(c.tg_size_en);
    o.num(c.lds_bytes), o.num(c.native_subgroup_size), o.num(c.wave64_exchange_width);
    o.template num<uint8_t>(c.force_exchange_dispatcher);
    o.num(c.native_storage_format_support);
    o.template num<uint8_t>(c.storage_buffer_int64_atomics),
        o.template num<uint8_t>(c.packed_r11_storage);
    o.template num<uint8_t>(static_cast<uint8_t>(c.float_transport.profile));
}

ComputeShaderConfig read_config(In& in) {
    ComputeShaderConfig c;
    const uint32_t n = in.num<uint32_t>();
    require(n <= 64, "refused-shader inputs user-SGPR bound");
    c.user_sgprs.resize(n);
    for (uint32_t& v : c.user_sgprs) v = in.num<uint32_t>();
    c.local_x = in.num<uint32_t>(), c.local_y = in.num<uint32_t>(), c.local_z = in.num<uint32_t>();
    c.exact_thread_extent = in.flag();
    c.threads_x = in.num<uint32_t>(), c.threads_y = in.num<uint32_t>();
    c.threads_z = in.num<uint32_t>();
    c.wave_size = in.num<uint32_t>(), c.compute_pgm_rsrc1 = in.num<uint32_t>();
    c.tidig_comp_cnt = in.num<uint32_t>();
    c.tgid_x_en = in.flag(), c.tgid_y_en = in.flag(), c.tgid_z_en = in.flag();
    c.tg_size_en = in.flag();
    c.lds_bytes = in.num<uint32_t>(), c.native_subgroup_size = in.num<uint32_t>();
    c.wave64_exchange_width = in.num<uint32_t>();
    c.force_exchange_dispatcher = in.flag();
    c.native_storage_format_support = in.num<uint32_t>();
    c.storage_buffer_int64_atomics = in.flag(), c.packed_r11_storage = in.flag();
    c.float_transport.profile = static_cast<FloatTransportProfile>(in.num<uint8_t>());
    require(c.float_transport.canonical(), "refused-shader inputs float transport");
    return c;
}

template <class Sink>
void write_launch(Sink& o, const ComputeDispatchLaunchState& launch) {
    o.num(launch.known);
    o.num(launch.initiator);
    for (uint32_t word : launch.num_thread) o.num(word);
    for (uint32_t word : launch.dimensions) o.num(word);
}

ComputeDispatchLaunchState read_launch(In& in) {
    ComputeDispatchLaunchState launch;
    launch.known = in.num<uint32_t>();
    require(!(launch.known & ~ComputeDispatchLaunchState::kAll),
            "refused-shader inputs launch presence");
    launch.initiator = in.num<uint64_t>();
    for (uint32_t& word : launch.num_thread) word = in.num<uint32_t>();
    for (uint32_t& word : launch.dimensions) word = in.num<uint32_t>();
    return launch;
}

// Owns the table through the compile-case carrier, then drops buffer bytes above the cap.
FragmentCompileCase carrier_for(uint64_t address, const ShaderResourceTable* table,
                                bool& payload_omitted) {
    FragmentCompileCase c;
    c.complete = false;
    c.reason = kCarrierReason;
    c.compiler = "refused-shader-inputs";
    c.program_address = address;
    own_fragment_compile_case_resources(c, table);
    uint64_t payload = 0;
    for (const auto& blob : c.blobs)
        if (!blob.opaque && blob.alias_of == UINT32_MAX && blob.bytes)
            payload += blob.bytes->size();
    payload_omitted = payload > kRefusedShaderInputsMaxPayloadBytes;
    if (payload_omitted) {
        for (auto& blob : c.blobs) {
            blob.opaque = true;
            blob.bytes.reset();
        }
        restore_fragment_compile_case_backing(c);
    }
    return c;
}

void take_carrier(RefusedShaderInputs& inputs, FragmentCompileCase&& c) {
    inputs.has_resources = c.has_resources;
    inputs.resources = c.resources;   // host_data points into shared owned blobs, kept alive here
    inputs.backing_present.clear();
    inputs.backing_owned.clear();
    for (const auto& refs : c.backing) {
        const bool present = refs.host.blob < c.blobs.size();
        inputs.backing_present.push_back(present);
        inputs.backing_owned.push_back(present && !c.blobs[refs.host.blob].opaque);
    }
}

void mix(uint64_t& h, uint64_t v) {
    h = fnv(reinterpret_cast<const uint8_t*>(&v), sizeof v, h);
}

}   // namespace

uint64_t refused_compute_inputs_identity(const ShaderResourceTable* table,
                                         const ComputeShaderConfig& c) {
    HashOut encoded;
    write_config(encoded, c);
    write_launch(encoded, c.dispatch_launch);
    uint64_t hash = encoded.hash;
    mix(hash, compile_case_resource_metadata_identity(table));
    return hash;
}

namespace {
std::vector<uint8_t> encode(const std::string& stage, uint64_t address,
                            const std::string& live_reject, const std::string& incomplete,
                            bool payload_omitted, const ComputeShaderConfig& config,
                            const std::vector<uint8_t>& carrier) {
    require(config.user_sgprs.size() <= 64, "refused-shader inputs user-SGPR bound");
    require(!(config.dispatch_launch.known & ~ComputeDispatchLaunchState::kAll),
            "refused-shader inputs launch presence");
    Out o;
    o.bytes.insert(o.bytes.end(), std::begin(kMagic), std::end(kMagic));
    o.num<uint32_t>(kRefusedShaderInputsSchema);
    o.text(stage);
    o.num<uint64_t>(address);
    o.text(live_reject);
    o.text(incomplete);
    o.num<uint8_t>(payload_omitted);
    write_config(o, config);
    o.blob(carrier);
    write_launch(o, config.dispatch_launch);
    o.num<uint64_t>(fnv(o.bytes.data(), o.bytes.size()));
    return std::move(o.bytes);
}
}   // namespace

RefusedShaderInputs decode_refused_shader_inputs(const std::vector<uint8_t>& bytes) {
    require(bytes.size() >= sizeof kMagic + 12 && bytes.size() <= kMaxFileBytes,
            "refused-shader inputs file size");
    In trailer{bytes, bytes.size(), bytes.size() - 8};
    require(trailer.num<uint64_t>() == fnv(bytes.data(), bytes.size() - 8),
            "refused-shader inputs checksum");
    require(std::memcmp(bytes.data(), kMagic, sizeof kMagic) == 0,
            "not a refused-shader inputs file");
    In in{bytes, bytes.size() - 8, sizeof kMagic};
    const uint32_t schema = in.num<uint32_t>();
    if (schema == 2 || schema == kRefusedGraphicsInputsSchema)
        return decode_refused_graphics_inputs(bytes);
    require(schema == 1 || schema == kRefusedShaderInputsSchema, "refused-shader inputs schema");
    RefusedShaderInputs inputs;
    inputs.stage = in.text();
    inputs.program_address = in.num<uint64_t>();
    inputs.live_reject = in.text();
    inputs.incomplete = in.text();
    inputs.payload_omitted = in.flag();
    inputs.compute = read_config(in);
    const std::vector<uint8_t> carrier = in.blob();
    if (schema == kRefusedShaderInputsSchema) inputs.compute.dispatch_launch = read_launch(in);
    require(in.at == in.end, "refused-shader inputs trailing data");
    if (!carrier.empty()) {
        FragmentCompileCase c = decode_fragment_compile_case(carrier);
        require(!c.complete, "refused-shader inputs carrier must be INCOMPLETE");
        take_carrier(inputs, std::move(c));
    }
    return inputs;
}

std::filesystem::path refused_shader_inputs_path(const std::filesystem::path& program) {
    std::filesystem::path path = program;
    path.replace_extension(".inputs");
    return path;
}

RefusedShaderInputs read_refused_shader_inputs(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f), "cannot open refused-shader inputs");
    const auto size = f.tellg();
    require(size >= 0 && uint64_t(size) <= kMaxFileBytes, "refused-shader inputs file size");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(bool(f), "cannot read refused-shader inputs");
    return decode_refused_shader_inputs(bytes);
}

bool write_refused_compute_inputs(const std::filesystem::path& path, uint64_t program_address,
                                  const ShaderResourceTable* table,
                                  const ComputeShaderConfig& config,
                                  const std::string& live_reject) {
    std::vector<uint8_t> bytes;
    try {
        std::string incomplete;
        bool payload_omitted = false;
        std::vector<uint8_t> carrier;
        if (table) {
            try {
                carrier = encode_fragment_compile_case(
                    carrier_for(program_address, table, payload_omitted));
            } catch (const std::exception& e) {
                // The config is still worth keeping: a replay then says which half is missing.
                incomplete = std::string("resource table not recorded: ") + e.what();
                carrier.clear();
            }
        }
        std::string reject = live_reject;
        if (reject.size() > kMaxText) reject.resize(kMaxText);
        if (incomplete.size() > kMaxText) incomplete.resize(kMaxText);
        bytes = encode("cs", program_address, reject, incomplete, payload_omitted, config, carrier);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[refused-shader] inputs for 0x%llx not recorded: %s\n",
                     (unsigned long long)program_address, e.what());
        return false;
    }
    std::filesystem::path partial = path;
    partial += ".partial";
    bool ok = false;
    if (FILE* f = std::fopen(partial.string().c_str(), "wb")) {
        ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
        ok = (std::fclose(f) == 0) && ok;
    }
    std::error_code ec;
    if (ok) std::filesystem::rename(partial, path, ec);
    const bool installed = ok && !ec;
    if (!installed) {
        std::error_code cleanup_error;
        std::filesystem::remove(partial, cleanup_error);
    }
    return installed;
}

RefusedShaderReplay replay_refused_shader_inputs(const std::vector<uint32_t>& code,
                                                 const RefusedShaderInputs& inputs,
                                                 bool use_inputs) {
    if (inputs.graphics) return replay_refused_graphics_inputs(code, inputs, use_inputs);
    RefusedShaderReplay replay;
    if (use_inputs && inputs.live_reject.empty()) {
        replay.incomplete = "live terminal refusal was not recorded";
        return replay;
    }
    if (use_inputs && !inputs.incomplete.empty()) replay.incomplete = inputs.incomplete;
    const bool table = use_inputs && inputs.has_resources;
    std::vector<CompilerResourceAccess> access;
    if (table) {
        for (size_t k = 0; k < inputs.resources.resources.size(); ++k) {
            const ShaderResource& r = inputs.resources.resources[k];
            const bool present = k < inputs.backing_present.size() && inputs.backing_present[k];
            const bool owned = present && inputs.backing_owned[k] && r.host_data;
            access.push_back({&r, present, owned,
                              owned
                                  ? std::span<const uint8_t>(r.host_data, size_t(r.host_data_size))
                                  : std::span<const uint8_t>()});
        }
    }
    // Pinned, not inherited from this process: the live run's trip bound was disarmed by default.
    const TripBoundOperation trip(ComputeTripBoundSettings{});
    TerminalRejectCapture rejection;
    std::vector<uint32_t> spirv;
    try {
        // Only replay-on-recorded-inputs runs under a scope: the fallback is the historical
        // offline answer, which never had one.
        std::optional<CompilerResourceScope> scope;
        if (use_inputs) scope.emplace(std::move(access));
        // The live compile key holds the recompile span (to S_ENDPGM, plus a proven PC-relative
        // table), not the dump's whole readable window: its length is part of the reject text.
        const size_t dwords = rdna2_recompile_code_span(code.data(), code.size());
        spirv = recompile_compute(code.data(), dwords, table ? &inputs.resources : nullptr,
                                  use_inputs ? inputs.compute : ComputeShaderConfig{},
                                  {RecompileDiagnosticStage::Compute, inputs.program_address});
    } catch (const std::exception& e) {
        replay.incomplete = e.what();
        return replay;
    }
    replay.produced = !spirv.empty();
    const auto records = rejection.take();
    if (!replay.produced && !records.empty())
        replay.reject = records.back().first + " " + records.back().second;
    if (use_inputs && !replay.produced && replay.reject.empty() && replay.incomplete.empty())
        replay.incomplete = "recompile returned no module and no terminal refusal";
    return replay;
}

}   // namespace prosper::gpu
