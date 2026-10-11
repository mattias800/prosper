#include "gpu/execute/refused_graphics_compile.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/ngg_subgroup_draw.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "build_revision.hpp"

namespace prosper::gpu {
namespace {
thread_local RefusedGraphicsLaunchScope* launch_scope = nullptr;
std::string bounded_capture_failure(const char* message) {
    size_t length = 0;
    while (length < 256 && message[length]) ++length;
    return {message, length};
}
}   // namespace
RefusedGraphicsLaunchScope::RefusedGraphicsLaunchScope(const GpuState& state)
    : state_(&state), previous_(launch_scope) {
    launch_scope = this;
}
RefusedGraphicsLaunchScope::~RefusedGraphicsLaunchScope() {
    launch_scope = previous_;
}
RefusedGraphicsLaunch refused_graphics_launch() {
    RefusedGraphicsLaunch out;
    if (!launch_scope) return out;
    out.observed = true;
    namespace P = prosper::agc::Pm4;
    const uint32_t bases[] = {P::SPI_SHADER_USER_DATA_GS_0, P::SPI_SHADER_USER_DATA_VS_0,
                              P::SPI_SHADER_USER_DATA_PS_0};
    const uint32_t rsrc1[] = {P::SPI_SHADER_PGM_RSRC1_GS, P::SPI_SHADER_PGM_RSRC1_VS,
                              P::SPI_SHADER_PGM_RSRC1_PS};
    const uint32_t rsrc2[] = {P::SPI_SHADER_PGM_RSRC2_GS, P::SPI_SHADER_PGM_RSRC2_VS,
                              P::SPI_SHADER_PGM_RSRC2_PS};
    const auto& sh = launch_scope->state_->sh;
    for (uint32_t bank = 0; bank < 3; ++bank) {
        for (uint32_t word = 0; word < 32; ++word) {
            const auto found = sh.find(bases[bank] + word);
            if (found != sh.end()) {
                out.available[bank] |= uint32_t(1) << word;
                out.user_data[bank][word] = found->second;
            }
        }
        const auto a = sh.find(rsrc1[bank]), b = sh.find(rsrc2[bank]);
        if (a != sh.end()) {
            out.rsrc1_available |= uint8_t(1 << bank);
            out.rsrc1[bank] = a->second;
        }
        if (b != sh.end()) {
            out.rsrc2_available |= uint8_t(1 << bank);
            out.rsrc2[bank] = b->second;
        }
    }
    return out;
}

std::vector<uint32_t> observe_refused_graphics_compile(const char* stage,
                                                       const RefusedShaderSource& source,
                                                       RefusedGraphicsInputs inputs,
                                                       const ShaderResourceTable* table,
                                                       const RefusedGraphicsCompile& compile,
                                                       const std::string* direct_refusal) {
    if (refused_shader_dump_full() || !source.words || source.words->empty()) return compile();
    auto& c = inputs.invocation;
    try {
        inputs.launch = refused_graphics_launch();
        c.compiler = std::string(prosper::embedded_build_revision()) + ":" +
                     prosper::embedded_build_source_identity();
        if (source.words->size() <= kCompileCaseMaxWords) {
            note_refused_graphics_source_copy(source.words->size());
            inputs.raw = *source.words;
        } else
            inputs.incomplete = "raw source exceeds diagnostic budget";
        inputs.raw_is_source_prefix =
            inputs.selector == RefusedGraphicsCompiler::NggSubgroup && !inputs.raw.empty() &&
            inputs.raw.size() < c.code.size() &&
            std::equal(inputs.raw.begin(), inputs.raw.end(), c.code.begin());
        try {
            own_refused_graphics_resources(inputs, table);
        } catch (const std::exception& e) {
            inputs.incomplete = bounded_capture_failure(e.what());
            c.resources = {};
            c.has_resources = false;
            c.blobs.clear();
            c.backing.clear();
        }
    } catch (const std::exception&) {
        // Diagnostic ownership is optional. Do not catch or repeat the actual compiler call.
        return compile();
    }
    std::vector<std::pair<std::string, std::string>> rejects;
    std::vector<uint32_t> result;
    {
        CompilerChoiceScope choices(c.choices);
        TerminalRejectCapture rejection;
        result = compile();
        rejects = rejection.take();
    }
    try {
        // The observational nested scope forwards the actual attempt's diagnostics to its caller.
        replay_terminal_reject_reasons(c.program_address, rejects);
        if (!result.empty()) return result;
        if (direct_refusal)
            c.expected_reject = *direct_refusal;
        else if (!rejects.empty())
            c.expected_reject = rejects.back().first + " " + rejects.back().second;
        if (c.expected_reject.size() > 256) {
            c.expected_reject.resize(256);
            inputs.incomplete = "live terminal refusal exceeds diagnostic budget";
        }
        if (c.expected_reject.empty())
            inputs.incomplete = "actual terminal refusal was not recorded";
        if (!c.choices.error.empty())
            inputs.incomplete = bounded_capture_failure(c.choices.error.c_str());
        struct Context {
            const char* stage;
            const RefusedGraphicsInputs* inputs;
        } context{stage, &inputs};
        const RefusedShaderInputSet callbacks{
            0,
            [p = &context](const std::string& stem) {
                return write_refused_graphics_inputs(stem + ".inputs", p->stage, *p->inputs);
            },
            [p = &context] { return refused_graphics_inputs_identity(*p->inputs); }};
        note_refused_shader_inputs(stage, c.program_address, source, "actual-compiler-inputs",
                                   callbacks);
    } catch (const std::exception&) {
        // Publication, identity hashing and refusal strings cannot turn a module/refusal into a
        // guest failure. The original call has already completed exactly once.
        return result;
    }
    return result;
}

std::vector<uint32_t> observe_graphics_shader_attempt(const ShaderCompileKey& key,
                                                      const ShaderResourceTable* table,
                                                      uint64_t address,
                                                      const RefusedGraphicsCompile& compile) {
    const char* stage = key.stage == ShaderProgramStage::Fragment ? "ps" : "vs";
    if (!refused_shader_inputs_needed(stage, {key.code, key.refused_memo})) return compile();
    RefusedGraphicsInputs g;
    try {
        g.selector = key.stage == ShaderProgramStage::Fragment ? RefusedGraphicsCompiler::Fragment
                     : key.chain_code ? RefusedGraphicsCompiler::VertexChain
                                      : RefusedGraphicsCompiler::Vertex;
        auto& c = g.invocation;
        c.program_address = address;
        c.trip = key.trip_bound;
        if (key.code->size() <= kCompileCaseMaxWords)
            c.code = *key.code;
        else
            g.incomplete = "compile source exceeds diagnostic budget";
        if (key.chain_code) {
            if (key.chain_code->size() <= kCompileCaseMaxWords)
                g.chain = *key.chain_code;
            else
                g.incomplete = "chain source exceeds diagnostic budget";
        }
        c.has_pixel_inputs = key.has_pixel_inputs;
        c.pixel_inputs = key.pixel_inputs;
        c.has_system_inputs = key.has_system_inputs;
        c.system_inputs = key.system_inputs;
        c.interpolation = fragment_interpolation_layout(
            c.code.data(), c.code.size(), c.has_system_inputs ? &c.system_inputs : nullptr,
            c.has_pixel_inputs ? &c.pixel_inputs : nullptr);
        c.wave32 = key.fragment_wave32;
        c.float_mode = key.fragment_float_mode;
        c.float_transport = key.float_transport;
        c.float_flags = key.fragment_float_flags;
        c.launch_rsrc1 = key.fragment_launch_rsrc1;
        c.export_formats = key.fragment_export_formats;
        c.pcrel_target = key.has_pcrel_dispatch ? key.pcrel_dispatch_target : UINT32_MAX;
        g.capture_position = key.capture_position;
        g.vertex_lds_dwords = key.vertex_lds_dwords;
    } catch (const std::exception&) { return compile(); }
    return observe_refused_graphics_compile(key.stage == ShaderProgramStage::Fragment ? "ps" : "vs",
                                            {key.code, key.refused_memo}, std::move(g), table,
                                            compile);
}

std::vector<uint32_t> observe_ngg_shader_attempt(const NggSubgroupDrawRequest& request,
                                                 const NggSubgroupShellConfig& shell,
                                                 const RefusedGraphicsCompile& compile,
                                                 const std::string* refusal) {
    if (refused_shader_dump_full()) return compile();
    std::shared_ptr<const ShaderCodeAnalysis> analysis;
    try {
        analysis = acquire_shader_analysis(request.linked_code, request.dwords);
    } catch (const std::exception&) { return compile(); }
    if (!analysis) return compile();
    const RefusedShaderSource source{{analysis, &analysis->code}, &analysis->refused_shader_memo};
    if (!refused_shader_inputs_needed("vs", source)) return compile();
    RefusedGraphicsInputs g;
    // Pin the actual compiler's trip settings while copying its diagnostic inputs.
    const TripBoundOperation trip;
    try {
        g.selector = RefusedGraphicsCompiler::NggSubgroup;
        g.invocation.program_address = request.diagnostic.program_address;
        g.invocation.trip = trip.settings();
        g.shell = shell;
        if (request.dwords <= kCompileCaseMaxWords)
            g.invocation.code.assign(request.linked_code, request.linked_code + request.dwords);
        else
            g.incomplete = "compile source exceeds diagnostic budget";
    } catch (const std::exception&) { return compile(); }
    // NGG's public compiler returns its named refusal directly. Do not invent terminal records.
    return observe_refused_graphics_compile("vs", source, std::move(g), request.resources, compile,
                                            refusal);
}

void note_unavailable_graphics_inputs(const char* stage, uint64_t address,
                                      const RefusedShaderSource& source,
                                      const std::string& detail) {
    // The shipping caller already probes its immutable owner. Keep this exported helper equally
    // cheap when called directly: an unavailable invocation has no input-set counter to update.
    if (refused_shader_already_noted(stage, source)) return;
    try {
        RefusedGraphicsInputs g;
        g.invocation.program_address = address;
        if (source.words && source.words->size() <= kCompileCaseMaxWords) {
            note_refused_graphics_source_copy(source.words->size());
            g.raw = *source.words;
        }
        g.incomplete = "draw refused before an actual compiler invocation was recorded";
        struct Context {
            const char* stage;
            const RefusedGraphicsInputs* inputs;
        } context{stage, &g};
        const RefusedShaderInputSet callbacks{
            0,
            [p = &context](const std::string& stem) {
                return write_refused_graphics_inputs(stem + ".inputs", p->stage, *p->inputs);
            },
            [p = &context] { return refused_graphics_inputs_identity(*p->inputs); }};
        note_refused_shader(stage, address, source, detail, RefusedShaderBudget::Refusal,
                            &callbacks);
    } catch (const std::exception&) {
        // This is already a draw refusal; an optional unavailable carrier must not throw.
        return;
    }
}
}   // namespace prosper::gpu
