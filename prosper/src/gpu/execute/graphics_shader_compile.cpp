#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/graphics_shader_compile.hpp"
#include "build_revision.hpp"
#include "gpu/execute/refused_graphics_compile.hpp"

namespace prosper::gpu {
// `program_address` is the guest address of the shader this key was built from. It is DIAGNOSTIC
// PROVENANCE ONLY and deliberately arrives as a separate argument rather than inside the key: the
// same program bytes produce the same SPIR-V at every guest address, so folding it into
// ShaderCompileKey would give each address its own cache entry and quietly defeat the cache. The
// key's own `code` pointer cannot serve here -- it points into the key's owned COPY of the words,
// not at guest memory (#3130).
static std::vector<uint32_t>
compile_graphics_shader_actual(ShaderProgramStage stage, const ShaderCompileKey& key,
                               const ShaderResourceTable* resources, uint64_t program_address,
                               FragmentArithmeticObservation* arithmetic_observation,
                               std::shared_ptr<const FragmentCompileCase>* producer) {
    const uint32_t* code = !key.code || key.code->empty() ? nullptr : key.code->data();
    const size_t code_size = key.code ? key.code->size() : 0u;
    if (stage == ShaderProgramStage::Vertex)
        // Geometry probe: decorate gl_Position only when the caller proved the VS is the last
        // pre-rasterization stage. Generated interpolation geometry stages own XFB themselves.
        // Same provenance the fragment stage received in #3130: without the guest address the
        // vertex stage's terminal reject is never RECORDED (record_terminal_reject_reason()
        // early-returns on zero), so every vertex skip printed `reason=unrecorded`.
        return key.chain_code
                   ? recompile_vertex_chain(
                         code, code_size, key.chain_code->data(), key.chain_code->size(), resources,
                         key.has_pixel_inputs ? &key.pixel_inputs : nullptr, key.capture_position,
                         key.vertex_lds_dwords, {RecompileDiagnosticStage::Vertex, program_address},
                         key.float_transport)
                   : recompile_vertex(code, code_size, resources,
                                      key.has_pixel_inputs ? &key.pixel_inputs : nullptr,
                                      key.capture_position, key.vertex_lds_dwords,
                                      {RecompileDiagnosticStage::Vertex, program_address},
                                      key.float_transport);
    if (stage == ShaderProgramStage::Fragment) {
        const FragmentInterpolationLayout interpolation = fragment_interpolation_layout(
            code, code_size, key.has_system_inputs ? &key.system_inputs : nullptr,
            key.has_pixel_inputs ? &key.pixel_inputs : nullptr);
        std::shared_ptr<FragmentCompileCase> record;
        if (producer && fragment_compile_case_recording_enabled()) {
            record = std::make_shared<FragmentCompileCase>();
            record->compiler = std::string(prosper::embedded_build_revision()) + ":" +
                               prosper::embedded_build_source_identity();
            record->program_address = program_address;
            record->has_pixel_inputs = key.has_pixel_inputs;
            record->pixel_inputs = key.pixel_inputs;
            record->has_system_inputs = key.has_system_inputs;
            record->system_inputs = key.system_inputs;
            record->interpolation = interpolation;
            record->wave32 = key.fragment_wave32;
            record->float_mode = key.fragment_float_mode;
            record->float_transport = key.float_transport;
            record->float_flags = key.fragment_float_flags;
            record->launch_rsrc1 = key.fragment_launch_rsrc1;
            record->export_formats = key.fragment_export_formats;
            record->pcrel_target = key.has_pcrel_dispatch ? key.pcrel_dispatch_target : UINT32_MAX;
            record->trip = key.trip_bound;
            try {
                if (!key.code || key.code->empty() || key.code->size() > kCompileCaseMaxWords)
                    throw std::runtime_error("raw-code-unavailable-or-budget");
                record->code = *key.code;
                own_fragment_compile_case_resources(*record, resources);
            } catch (const std::exception& e) {
                record->complete = false;
                record->reason = e.what();
                record->resources = {};
                record->backing.clear();
                record->blobs.clear();
            }
        }
        // The normal compiler still consumes its existing inputs. The case's independent owned
        // replay must reproduce the whole resulting SOURCE (or refusal) before it is COMPLETE.
        CompilerChoiceTrace trace;
        auto compile = [&] {
            return recompile_fragment(
                code, code_size, resources, key.has_system_inputs ? &key.system_inputs : nullptr,
                key.has_pcrel_dispatch ? key.pcrel_dispatch_target : UINT32_MAX, &interpolation,
                key.fragment_wave32, {RecompileDiagnosticStage::Fragment, program_address},
                key.fragment_float_mode, arithmetic_observation, key.float_transport,
                key.fragment_float_flags, key.fragment_export_formats);
        };
        std::vector<uint32_t> result;
        if (record) {
            std::vector<std::pair<std::string, std::string>> rejects;
            {
                CompilerChoiceScope reads(trace);
                TerminalRejectCapture rejection;
                result = compile();
                rejects = rejection.take();
            }
            record->expected_reject = rejects.empty()
                                          ? std::string{}
                                          : rejects.back().first + " " + rejects.back().second;
            record->choices = std::move(trace);
            finish_fragment_compile_case(*record, result);
            // Nested capture keeps the independent baseline out of the cache's producer reasons.
            // Forward only the records from the ACTUAL producing attempt to the outer cache scope.
            replay_terminal_reject_reasons(program_address, rejects);
            *producer = std::move(record);
        } else
            result = compile();
        return result;
    }
    return {};
}

std::vector<uint32_t>
compile_graphics_shader(ShaderProgramStage stage, const ShaderCompileKey& key,
                        const ShaderResourceTable* resources, uint64_t address,
                        FragmentArithmeticObservation* arithmetic,
                        std::shared_ptr<const FragmentCompileCase>* producer) {
    struct Context {
        ShaderProgramStage stage;
        const ShaderCompileKey* key;
        const ShaderResourceTable* resources;
        uint64_t address;
        FragmentArithmeticObservation* arithmetic;
        std::shared_ptr<const FragmentCompileCase>* producer;
    };
    const Context context{stage, &key, resources, address, arithmetic, producer};
    return observe_graphics_shader_attempt(key, resources, address, [p = &context] {
        return compile_graphics_shader_actual(p->stage, *p->key, p->resources, p->address,
                                              p->arithmetic, p->producer);
    });
}

}   // namespace prosper::gpu
