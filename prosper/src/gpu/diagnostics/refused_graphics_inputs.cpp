#include "gpu/diagnostics/refused_graphics_inputs.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>

namespace prosper::gpu {
namespace {
constexpr char kMagic[8] = {'P', 'R', 'R', 'S', 'I', 'N', 'P', 0};
constexpr size_t kMaxFile = kCompileCaseMaxFileBytes + 64 * 1024;
void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
uint64_t hash(std::span<const uint8_t> bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (auto b : bytes) h = (h ^ b) * 0x100000001b3ull;
    return h;
}
struct Out {
    std::vector<uint8_t> bytes;
    template <class T>
    void num(T value) {
        const uint64_t v = static_cast<uint64_t>(value);
        for (size_t i = 0; i < sizeof(T); ++i) bytes.push_back(uint8_t(v >> (8 * i)));
    }
    void text(const std::string& s) {
        require(s.size() <= 512, "graphics-input text budget");
        num<uint32_t>(uint32_t(s.size()));
        bytes.insert(bytes.end(), s.begin(), s.end());
    }
    void words(const std::vector<uint32_t>& v) {
        require(v.size() <= kCompileCaseMaxWords, "graphics-input source budget");
        num<uint32_t>(uint32_t(v.size()));
        for (auto w : v) num(w);
    }
};
struct In {
    std::span<const uint8_t> bytes;
    size_t at = 0;
    template <class T>
    T num() {
        require(sizeof(T) <= bytes.size() - at, "truncated graphics inputs");
        uint64_t v = 0;
        for (size_t i = 0; i < sizeof(T); ++i) v |= uint64_t(bytes[at++]) << (8 * i);
        return static_cast<T>(v);
    }
    bool flag() {
        const auto v = num<uint8_t>();
        require(v <= 1, "graphics-input flag");
        return v;
    }
    std::string text() {
        const auto n = num<uint32_t>();
        require(n <= 512 && n <= bytes.size() - at, "graphics-input text budget");
        std::string s(reinterpret_cast<const char*>(bytes.data() + at), n);
        at += n;
        return s;
    }
    std::vector<uint32_t> words() {
        const auto n = num<uint32_t>();
        require(n <= kCompileCaseMaxWords && n <= (bytes.size() - at) / 4,
                "graphics-input source budget");
        std::vector<uint32_t> v(n);
        for (auto& w : v) w = num<uint32_t>();
        return v;
    }
};
void write_arguments(Out& o, const RefusedGraphicsInputs& g) {
    o.num<uint32_t>(uint32_t(g.selector));
    o.words(g.raw);
    o.words(g.chain);
    o.num<uint8_t>(g.raw_is_source_prefix);
    o.num<uint8_t>(g.capture_position);
    o.num(g.vertex_lds_dwords);
    o.num(g.shell.waves);
    o.num(g.shell.wave_lanes);
    o.num(g.shell.rsrc2_gs_lds_size);
    o.num(g.shell.user_sgprs);
    o.num<uint8_t>(g.shell.user_data_address_known);
    o.num<uint8_t>(g.shell.native_wave64);
    o.num<uint8_t>(g.launch.observed);
    for (const auto& bank : g.launch.user_data)
        for (auto v : bank) o.num(v);
    for (auto v : g.launch.available) o.num(v);
    for (auto v : g.launch.rsrc1) o.num(v);
    for (auto v : g.launch.rsrc2) o.num(v);
    o.num(g.launch.rsrc1_available);
    o.num(g.launch.rsrc2_available);
}
void read_arguments(In& in, RefusedGraphicsInputs& g) {
    const auto selector = in.num<uint32_t>();
    require(selector <= uint32_t(RefusedGraphicsCompiler::NggSubgroup),
            "graphics-input compiler selector");
    g.selector = static_cast<RefusedGraphicsCompiler>(selector);
    g.raw = in.words();
    g.chain = in.words();
    g.raw_is_source_prefix = in.flag();
    g.capture_position = in.flag();
    g.vertex_lds_dwords = in.num<uint32_t>();
    g.shell.waves = in.num<uint32_t>();
    g.shell.wave_lanes = in.num<uint32_t>();
    g.shell.rsrc2_gs_lds_size = in.num<uint32_t>();
    g.shell.user_sgprs = in.num<uint32_t>();
    g.shell.user_data_address_known = in.flag();
    g.shell.native_wave64 = in.flag();
    g.launch.observed = in.flag();
    for (auto& bank : g.launch.user_data)
        for (auto& v : bank) v = in.num<uint32_t>();
    for (auto& v : g.launch.available) v = in.num<uint32_t>();
    for (auto& v : g.launch.rsrc1) v = in.num<uint32_t>();
    for (auto& v : g.launch.rsrc2) v = in.num<uint32_t>();
    g.launch.rsrc1_available = in.num<uint8_t>();
    g.launch.rsrc2_available = in.num<uint8_t>();
    require((g.launch.rsrc1_available | g.launch.rsrc2_available) < 8,
            "graphics-input register knownness");
}
std::vector<uint8_t> encode(const std::string& stage, const RefusedGraphicsInputs& g) {
    Out o;
    o.bytes.insert(o.bytes.end(), std::begin(kMagic), std::end(kMagic));
    o.num<uint32_t>(2);
    o.text(stage);
    o.num(g.invocation.program_address);
    o.text(g.invocation.expected_reject);
    o.text(g.incomplete);
    o.num<uint8_t>(g.payload_omitted);
    write_arguments(o, g);
    FragmentCompileCase c = g.invocation;
    c.complete = false;
    c.reason = "refused-graphics-inputs-carrier";
    const auto carrier = encode_fragment_compile_case(c);
    o.num<uint32_t>(uint32_t(carrier.size()));
    o.bytes.insert(o.bytes.end(), carrier.begin(), carrier.end());
    require(o.bytes.size() <= kMaxFile - 8, "graphics-input file budget");
    o.num(hash(o.bytes));
    return std::move(o.bytes);
}
} // namespace

void own_refused_graphics_resources(RefusedGraphicsInputs& g, const ShaderResourceTable* table) {
    own_fragment_compile_case_resources(g.invocation, table, kRefusedShaderInputsMaxPayloadBytes,
                                        &g.payload_omitted);
    // Ownership failures named by the existing carrier remain unavailable, never replay authority.
    if (!g.invocation.complete) g.incomplete = g.invocation.reason;
}

bool write_refused_graphics_inputs(const std::filesystem::path& path, const std::string& stage,
                                   const RefusedGraphicsInputs& g) {
    std::filesystem::path temporary;
    try {
        const auto bytes = encode(stage, g);
        static std::atomic<uint64_t> serial{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned attempt = 0; attempt < 32; ++attempt) {
            auto candidate = path;
            candidate += ".partial-" + std::to_string(stamp) + "-" +
                         std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
            std::error_code error;
            if (std::filesystem::create_directory(candidate, error)) {
                temporary = std::move(candidate);
                break;
            }
            if (error) return false;
        }
        if (temporary.empty()) return false;
        const auto partial = temporary / "payload";
        bool ok = false;
        if (auto* f = std::fopen(partial.string().c_str(), "wb")) {
            ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
            ok = std::fclose(f) == 0 && ok;
        }
        std::error_code error;
        if (ok) std::filesystem::rename(partial, path, error);
        const bool installed = ok && !error;
        std::error_code cleanup;
        std::filesystem::remove_all(temporary, cleanup);
        return installed;
    } catch (const std::exception&) {
        std::error_code cleanup;
        if (!temporary.empty()) std::filesystem::remove_all(temporary, cleanup);
        return false;
    }
}

RefusedShaderInputs decode_refused_graphics_inputs(const std::vector<uint8_t>& bytes) {
    require(bytes.size() >= 20 && bytes.size() <= kMaxFile, "graphics-input file size");
    require(!std::memcmp(bytes.data(), kMagic, sizeof kMagic), "graphics-input magic");
    In trailer{bytes, bytes.size() - 8};
    require(trailer.num<uint64_t>() == hash(std::span(bytes).first(bytes.size() - 8)),
            "graphics-input checksum");
    In in{std::span(bytes).first(bytes.size() - 8), sizeof kMagic};
    require(in.num<uint32_t>() == 2, "graphics-input schema");
    RefusedShaderInputs result;
    result.stage = in.text();
    result.program_address = in.num<uint64_t>();
    result.live_reject = in.text();
    result.incomplete = in.text();
    result.payload_omitted = in.flag();
    auto g = std::make_shared<RefusedGraphicsInputs>();
    read_arguments(in, *g);
    const auto n = in.num<uint32_t>();
    require(n <= kCompileCaseMaxFileBytes && n <= in.bytes.size() - in.at,
            "graphics-input carrier budget");
    g->invocation = decode_fragment_compile_case(in.bytes.subspan(in.at, n));
    in.at += n;
    require(in.at == in.bytes.size() && !g->invocation.complete,
            "graphics-input trailing data or complete carrier");
    require(g->invocation.program_address == result.program_address &&
                g->invocation.expected_reject == result.live_reject,
            "graphics-input carrier association");
    g->incomplete = result.incomplete;
    g->payload_omitted = result.payload_omitted;
    result.graphics = std::move(g);
    return result;
}

uint64_t refused_graphics_inputs_identity(const RefusedGraphicsInputs& g) {
    Out o;
    write_arguments(o, g);
    // Code/chain and semantic choices participate. Buffer bytes and host pointer values do not.
    const auto& actual = g.invocation;
    FragmentCompileCase c;
    c.complete = false;
    c.reason = "identity";
    c.expected_produced = actual.expected_produced;
    c.program_address = actual.trip.bound ? actual.program_address : 0;
    c.code = actual.code;
    c.source = actual.source;
    c.has_pixel_inputs = actual.has_pixel_inputs;
    c.has_system_inputs = actual.has_system_inputs;
    c.pixel_inputs = actual.pixel_inputs;
    c.system_inputs = actual.system_inputs;
    c.interpolation = actual.interpolation;
    c.wave32 = actual.wave32;
    c.float_mode = actual.float_mode;
    c.float_transport = actual.float_transport;
    c.float_flags = actual.float_flags;
    c.launch_rsrc1 = actual.launch_rsrc1;
    c.export_formats = actual.export_formats;
    c.pcrel_target = actual.pcrel_target;
    c.trip = actual.trip;
    c.choices = actual.choices;
    // Config-only: do not copy resource metadata, backing references or shared payload handles.
    // Relocation remains diagnostic-only when the trip selector is disarmed.
    const auto config = encode_fragment_compile_case(c);
    o.bytes.insert(o.bytes.end(), config.begin(), config.end());
    o.num(compile_case_resource_metadata_identity(
        g.invocation.has_resources ? &g.invocation.resources : nullptr));
    // Opaque replay pointers are null, but actual backing presence/ranges/aliases remain inputs.
    // Hash references and blob metadata only: never allocate/copy/hash their bytes or addresses.
    o.num<uint32_t>(uint32_t(g.invocation.blobs.size()));
    for (const auto& blob : g.invocation.blobs) {
        o.num<uint32_t>(uint32_t(blob.owner));
        o.num<uint8_t>(blob.present);
        o.num<uint8_t>(blob.opaque);
        o.num(blob.size);
        o.num(blob.alias_of);
    }
    o.num<uint32_t>(uint32_t(g.invocation.backing.size()));
    const auto reference = [&o](CompileCaseBacking r) {
        o.num(r.blob);
        o.num(r.offset);
    };
    for (const auto& r : g.invocation.backing) {
        reference(r.host);
        reference(r.dcc);
        o.num<uint32_t>(uint32_t(r.entries.size()));
        for (auto entry : r.entries) reference(entry);
    }
    return hash(o.bytes);
}

RefusedShaderReplay replay_refused_graphics_inputs(const std::vector<uint32_t>& raw,
                                                   const RefusedShaderInputs& inputs,
                                                   bool use_inputs) {
    RefusedShaderReplay replay;
    const auto& g = *inputs.graphics;
    const auto& c = g.invocation;
    const bool fragment = g.selector == RefusedGraphicsCompiler::Fragment;
    if (inputs.stage != (fragment ? "ps" : "vs"))
        replay.incomplete = "recorded compiler/stage mismatch";
    else if (raw != g.raw)
        replay.incomplete = "raw dump does not match producing compiler source association";
    else if (g.selector == RefusedGraphicsCompiler::Unavailable)
        replay.incomplete = "actual compiler invocation was not recorded";
    else if (g.raw_is_source_prefix ? g.selector != RefusedGraphicsCompiler::NggSubgroup ||
                                          g.raw.empty() || g.raw.size() >= c.code.size() ||
                                          !std::equal(g.raw.begin(), g.raw.end(), c.code.begin())
                                    : c.code != g.raw)
        replay.incomplete = "compiler source does not match recorded raw association";
    else if (c.code.empty() ||
             (g.selector == RefusedGraphicsCompiler::VertexChain && g.chain.empty()))
        replay.incomplete = "producing compiler source was not recorded";
    else if (use_inputs && inputs.live_reject.empty())
        replay.incomplete = "live terminal refusal was not recorded";
    else if (use_inputs && !inputs.incomplete.empty())
        replay.incomplete = inputs.incomplete;
    if (!replay.incomplete.empty()) return replay;
    TerminalRejectCapture rejection;
    try {
        auto owned = c;
        restore_fragment_compile_case_backing(owned);
        std::vector<CompilerResourceAccess> access;
        for (size_t k = 0; k < owned.backing.size(); ++k) {
            const auto& r = owned.resources.resources[k];
            const auto ref = owned.backing[k].host;
            const bool present = ref.blob < owned.blobs.size();
            const bool readable = present && !owned.blobs[ref.blob].opaque;
            access.push_back({&r, present, readable,
                              readable
                                  ? std::span<const uint8_t>(r.host_data, size_t(r.host_data_size))
                                  : std::span<const uint8_t>()});
        }
        std::optional<CompilerResourceScope> resource_scope;
        std::optional<CompilerChoiceScope> choices;
        if (use_inputs) {
            resource_scope.emplace(std::move(access));
            choices.emplace(c.choices);
        }
        TripBoundOperation trip(use_inputs ? c.trip : ComputeTripBoundSettings{});
        const auto* table = use_inputs && c.has_resources ? &owned.resources : nullptr;
        const auto* pixel = use_inputs && c.has_pixel_inputs ? &c.pixel_inputs : nullptr;
        const auto diagnostic = RecompileDiagnosticContext{
            fragment ? RecompileDiagnosticStage::Fragment : RecompileDiagnosticStage::Vertex,
            c.program_address};
        std::vector<uint32_t> module;
        switch (g.selector) {
            case RefusedGraphicsCompiler::Vertex:
                module =
                    recompile_vertex(c.code.data(), c.code.size(), table, pixel, g.capture_position,
                                     g.vertex_lds_dwords, diagnostic, c.float_transport);
                break;
            case RefusedGraphicsCompiler::VertexChain:
                module = recompile_vertex_chain(c.code.data(), c.code.size(), g.chain.data(),
                                                g.chain.size(), table, pixel, g.capture_position,
                                                g.vertex_lds_dwords, diagnostic, c.float_transport);
                break;
            case RefusedGraphicsCompiler::Fragment:
                module = recompile_fragment(
                    c.code.data(), c.code.size(), table,
                    use_inputs && c.has_system_inputs ? &c.system_inputs : nullptr, c.pcrel_target,
                    use_inputs ? &c.interpolation : nullptr, c.wave32, diagnostic, c.float_mode,
                    nullptr, c.float_transport, c.float_flags, c.export_formats);
                break;
            case RefusedGraphicsCompiler::NggSubgroup:
                module = recompile_ngg_subgroup(c.code.data(), c.code.size(), table, g.shell,
                                                nullptr, diagnostic, &replay.reject);
                break;
            default: break;
        }
        if (choices) choices->finish_replay();
        replay.produced = !module.empty();
        const auto records = rejection.take();
        if (!replay.produced && replay.reject.empty() && !records.empty())
            replay.reject = records.back().first + " " + records.back().second;
        if (use_inputs && !replay.produced && replay.reject.empty())
            replay.incomplete = "recompile returned no module and no terminal refusal";
    } catch (const std::exception& e) { replay.incomplete = e.what(); }
    return replay;
}
}   // namespace prosper::gpu
