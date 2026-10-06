// spv_validate — emit a representative module from EVERY SPIR-V-producing entry point in the tree
// and write each as a .spv to argv[1] (a directory), then run spirv-val on it: the render tests only
// prove llvmpipe *accepts* the modules, but llvmpipe is lenient — strict validation catches latent
// invalid SPIR-V (bad decorations, ill-formed control flow, type mismatches) that would break on a
// real driver. Pure emit + file write; no Vulkan.
//
// This gate has two failure modes of its own, and #1711 is what both of them cost. That defect —
// an OpAccessChain in build_compute_compare_uvec4() whose result type disagreed with the type it
// walked — shipped to real devices through frontends/shared/live/live_compute.cpp and was found by a
// Vulkan validation LAYER, not here, because:
//
//   1. the corpus only ever walked recompile_*, so spirv_builder.cpp's hand-assembled modules were
//      never validated even though they are created at runtime exactly like recompiled shaders; and
//   2. a missing spirv-val was reported as "== PASS (recompiled; spirv-val not found) ==" with exit
//      status 0. spirv-val is on neither the GitHub runner image nor a plain Fedora host, and CI
//      never installed it — so the gate that CLAUDE.md and both READMEs describe as strict had, in
//      CI, validated nothing at all.
//
// Both are closed below: check_emitter_coverage() reads every `src/gpu/*.hpp` and fails on any
// declared emitter this run did not actually emit a validated module from, and an absent spirv-val
// is a hard failure rather than a pass.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "gpu/recompiler/raster_quad_collector.hpp"
#include "gpu/recompiler/fragment_draw_capacity.hpp"
#include "gpu/recompiler/fragment_draw_gpu.hpp"
#include "gpu/capture/fragment_compile_case.hpp"
#include "build_revision.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/spirv_builder.hpp"
#include "../../tests/fixtures/spirv_wave_width_fixtures.hpp"
#include "../../tests/fixtures/spirv_fragment_vote_fixtures.hpp"
#include "../../tests/fixtures/spirv_fragment_vote_execution.hpp"
#include "../../tests/fixtures/spirv_fragment_neutral_fixtures.hpp"
#include "../../tests/fixtures/portable_bpermute_fixture.hpp"
#include "../../tests/fixtures/dpp_row_max.hpp"
#include "../../tests/fixtures/dpp_row_fadd.hpp"
#include "../../tests/fixtures/ngg_merged_lut_fixture.hpp"
#include "../../tests/fixtures/fragment_packet_fixture.hpp"
#include "../../tests/fixtures/fragment_packet_wqm_fixture.hpp"
#include "../../tests/fixtures/fragment_packet_mbcnt_fixture.hpp"
#include "../../tests/fixtures/fragment_resource_packet_fixture.hpp"
#include "../../tests/fixtures/fragment_packet_definedness_fixture.hpp"
#include "../../tests/fixtures/fragment_packet_raw_masks_fixture.hpp"
#include "../../tests/fixtures/fragment_packet_mask_entry_fixture.hpp"
#include "../../tests/fixtures/fragment_special_f32_fixture.hpp"
#include "../../tests/fixtures/fragment_packet_wave_fixture.hpp"
#include "../../tests/fixtures/fragment_packet_exports_fixture.hpp"
#include "gpu/recompiler/spirv_fragment_vote_lowering.hpp"
#if defined(PROSPER_SPV_VALIDATE_SCALAR_BANK)
#include "../../tests/fixtures/fragment_scalar_bank_fixture.hpp"
#include "gpu/recompiler/fragment_scalar_bank_wire.hpp"
#include "shared/live/live_renderer.hpp"
#endif
#include <algorithm>
#include <bit>
#include <array>
#include <cctype>
#include <filesystem>
#include <memory>
#include <set>
#include <system_error>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>

using namespace prosper::gpu;

static int fails = 0;

static std::string read_text(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}

// spirv-val's diagnostic IS the value of this gate, so capture it rather than discarding it: a bare
// "REJECTED it" tells the next reader nothing about which instruction is wrong. Redirecting to a
// file (not /dev/null) also keeps the invocation portable — cmd.exe has no /dev/null, so the old
// probe could never find the validator on Windows even when it was installed.
static bool run_spirv_val(const std::string& path, std::string& message) {
    const std::string log = path + ".val.txt";
    // --target-env vulkan1.1: these modules are consumed by a Vulkan instance, and the universal
    // default environment does not apply the Vulkan-specific rules a driver's validation layer
    // would. Measured over the whole corpus before adopting it: identical results, so it only
    // tightens what can pass here.
    const std::string cmd =
        "spirv-val --target-env vulkan1.1 \"" + path + "\" > \"" + log + "\" 2>&1";
    const int rc = system(cmd.c_str());
    message = read_text(log);
    std::remove(log.c_str());
    return rc == 0;
}

// Emitters this run actually produced a module from. Recorded here, at the point of use, rather
// than inferred from the source text: coverage then means "this emitter ran and its output was
// validated", which is the property the gate is for. A textual check could be satisfied by a
// call-shaped string in a comment, a string literal, or a call whose result is thrown away.
static std::set<std::string> exercised_emitters;

static void dump(const std::string& dir, const char* name, const std::vector<uint32_t>& spv,
                 const char* emitter = nullptr) {
    if (emitter) exercised_emitters.insert(emitter);
    if (spv.empty() || spv[0] != 0x07230203u) { printf("  [FAIL] %-26s did not recompile\n", name); fails++; return; }
    std::string path = dir + "/" + name + ".spv";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("  [FAIL] %-26s cannot write %s\n", name, path.c_str()); fails++; return; }
    fwrite(spv.data(), 4, spv.size(), f); fclose(f);
    std::string message;
    if (!run_spirv_val(path, message)) {
        printf("  [FAIL] %-26s spirv-val REJECTED it:\n", name);
        if (message.empty()) message = "(spirv-val produced no output)\n";
        printf("%s", message.c_str());
        if (message.back() != '\n') printf("\n");
        fails++;
        return;
    }
    printf("  [ok]   %-26s (%zu words) valid\n", name, spv.size());
}

#if defined(PROSPER_SPV_VALIDATE_SCALAR_BANK)
// The bank representative is issued by the same registered original-source/ordered producer
// as the CPU and GPU fixtures. These are real metadata-only backend queries, not fabricated
// completion booleans. The submit callback captures the actual draw; it does not create a device,
// submit GPU commands, wait for completion or publish pixels.
static void validate_scalar_bank(const std::string& dir) {
    namespace fixture = prosper::test::scalar_bank;
    namespace draw_fixture = prosper::test::fragment_draw;
    namespace live = prosper::frontend;
    const auto require = [](bool condition, const char* reason) {
        if (!condition) {
            printf("  [FAIL] scalar-bank: %s\n", reason);
            ++fails;
        }
        return condition;
    };
    fixture::Scene scene;
    if (!require(scene.create(), "genuine AGC/direct-memory scene unavailable")) return;
    const auto backend = live::live_graphics_producer_status();
    if (!require(backend.known && !backend.pending, "live backend baseline unknown/pending"))
        return;
    struct Queries {
        explicit Queries(std::vector<DrawItem>& draws) {
            set_graphics_producer_status_query(live::live_graphics_producer_status);
            set_graphics_raw_source_authority(live::live_graphics_raw_source_current);
            set_graphics_raw_allocation_authority(live::live_graphics_raw_allocation_current);
            set_submit_renderer([&draws](const std::vector<DrawItem>& items, uint32_t, uint32_t) {
                draws.insert(draws.end(), items.begin(), items.end());
                return RenderedFrame{};
            });
        }
        ~Queries() {
            set_submit_renderer({});
            set_graphics_raw_allocation_authority({});
            set_graphics_raw_source_authority({});
            set_graphics_producer_status_query({});
        }
    };
    // Offline declared compiler inputs, NOT evidence of enabled Vulkan device features. Publish
    // before actual realization so its original producing capsule captures this same profile.
    publish_float_transport_config({FloatTransportProfile::ExplicitNonFinite32});
    std::vector<DrawItem> draws;
    Queries queries(draws);
    const bool presented = execute_ordered_and_present(scene.state(), draw_fixture::width,
                                                       draw_fixture::height, 4801, false);
    if (!require(!presented && draws.size() == 1, "ordered producer did not capture one draw"))
        return;
    const auto& inputs = draws[0].fragment_draw_inputs;
    if (!require(inputs && inputs->scalar_bank, "ordered producer did not seal a genuine bank"))
        return;
    const auto& bank = inputs->scalar_bank;
    if (!require(bank->matches(inputs->raw_code, inputs->vgpr_requirements, inputs->entry) &&
                     bank->original_words() &&
                     *bank->original_words() == fixture::fragment_words() && bank->requirements() &&
                     bank->read_point_identity() != 0 && bank->sites().size() == 1 &&
                     bank->sources().size() == 1 && bank->intervals().size() == 1 &&
                     bank->sites()[0].pc == 1 && bank->sites()[0].words == 4 &&
                     bank->sources()[0].descriptor == scene.descriptor &&
                     bank->payload_bytes() == sizeof(draw_fixture::color_a) &&
                     bank->intervals()[0].bytes &&
                     bank->intervals()[0].bytes->size() == sizeof(draw_fixture::color_a) &&
                     std::memcmp(bank->intervals()[0].bytes->data(), draw_fixture::color_a.data(),
                                 sizeof(draw_fixture::color_a)) == 0,
                 "complete original-PC/entry/descriptor/payload association missing"))
        return;
    const auto prepared = draw_fixture::prepare(draws[0]);
    if (!require(bool(prepared), "actual producing native modules failed preparation")) return;
    const FragmentPacketDeviceContract declared_device{0x1234, true, false};
    const auto plan = cached_fragment_draw_program(*inputs, *prepared, declared_device, 48);
    if (!require(bool(plan), "no bank program plan")) return;
    if (!require(plan->rejection_reason().empty(), plan->rejection_reason().c_str())) return;
    if (!require(plan->requires_scalar_bank(), "program plan omitted the scalar-bank schema"))
        return;
    const auto& capacity = plan->capacity_owner();
    if (!require(capacity && capacity->kernel()->layout.gpu_capacity &&
                     capacity->kernel()->guest_code == *bank->original_words() &&
                     capacity->kernel()->program.scalar_bank_sites == bank->requirements()->sites,
                 "capacity kernel does not retain the complete original bank schema"))
        return;
    const auto transaction = instantiate_fragment_draw_transaction(
        plan, inputs, *prepared, draw_fixture::width, draw_fixture::height, 1,
        declared_device.device_identity);
    if (!require(transaction.rejection().empty() && transaction.scalar_bank() == bank,
                 "actual transaction consumer did not retain the sealed bank"))
        return;
    struct Module {
        const char* name;
        const std::vector<uint32_t>& words;
        SpirvShaderStage stage;
        uint32_t set;
        uint32_t allowed_bindings;
        bool needs_bank;
    };
    // Six actual modules, not six statically-used bindings in every module. The normal compute
    // layout has bindings 0..5; only the original kernel reads SBR2 at 5. Collector uses set1/0,
    // and replay's private set1/1..5 exclude persistent GDS at 0; set1/5 is commit, NOT SBR2.
    constexpr uint32_t compute_bindings = (1u << (kFragmentScalarBankBinding + 1u)) - 1u;
    const std::array modules{Module{"scalar_bank_original_kernel",
                                    capacity->kernel()->program.packet.spirv,
                                    SpirvShaderStage::Compute, 0, compute_bindings, true},
                             Module{"scalar_bank_collector", plan->collect_words(),
                                    SpirvShaderStage::Fragment, 1, 1u, false},
                             Module{"scalar_bank_count", plan->count_words(),
                                    SpirvShaderStage::Compute, 0, compute_bindings, false},
                             Module{"scalar_bank_assembly", plan->assembly_words(),
                                    SpirvShaderStage::Compute, 0, compute_bindings, false},
                             Module{"scalar_bank_validation", plan->validation_words(),
                                    SpirvShaderStage::Compute, 0, compute_bindings, false},
                             Module{"scalar_bank_replay", plan->replay_words(),
                                    SpirvShaderStage::Fragment, 1, 0x3eu, false}};
    for (const auto& module : modules) {
        dump(dir, module.name, module.words);
        const auto report = validate_spirv_descriptor_interface(module.words, nullptr, module.set,
                                                                module.stage, false);
        if (!require(spirv_descriptor_reflection_complete(report), "incomplete module reflection"))
            continue;
        for (const auto& binding : report.descriptors) {
            printf("  [binding] %s set=%u binding=%u storage=%d read=%d write=%d\n", module.name,
                   binding.set, binding.binding, binding.kind == SpirvDescriptorKind::StorageBuffer,
                   binding.readable, binding.writable);
            require(binding.set == module.set && binding.stage == module.stage &&
                        binding.kind == SpirvDescriptorKind::StorageBuffer &&
                        binding.binding < 32u &&
                        (module.allowed_bindings & (1u << binding.binding)) != 0,
                    "module binding outside the actual WAT2 interface");
        }
        if (module.needs_bank) {
            const auto binding =
                find_spirv_descriptor_binding(report, 0, kFragmentScalarBankBinding);
            require(binding && binding->readable && !binding->writable,
                    "original kernel did not declare/read readonly SBR2 binding5");
        }
    }
}
#endif

// --- Emitter coverage -------------------------------------------------------------------------
// An entry point that emits SPIR-V but has no module here is invisible to this gate, and nothing
// reports the omission: that is exactly how #1711 survived. So read the emitter declarations out of
// the graphics headers and fail on any that this run did not actually emit a module from.
//
// Coverage is a RUNTIME fact (`exercised_emitters`, recorded inside dump()), not a grep of this
// file. A source-text check would accept a call-shaped string in a comment or a string literal, or
// a call whose result is discarded — and "a check a comment can pass" is the shape of defect this
// file exists to stop.
//
// The list of gaps is EMPTY, and that is the intended state. A gap belongs here only with the issue
// that tracks it, never as a silent omission.
// (#1715's former missing geometry Invocations passed spirv-val under both universal and vulkan1.1.
// The shared emitter now declares Invocations 1 explicitly; raster_quad_collector_contract pins
// the actual Geometry entry's count on every generated form. Pipeline rule stage-00715 still
// needs the Vulkan validation layer: strict module legality alone cannot prove its absence.)
struct KnownGap { const char* emitter; const char* reason; };
static const std::vector<KnownGap> kKnownGaps = {};

// Names the scan finds that are NOT distinct SPIR-V producers. The default is deliberately
// inverted: an unclassified name fails, so a new emitter cannot be quietly skipped — only a
// deliberate entry here can exempt one, and it has to say why.
struct NotAnEmitter { const char* name; const char* why; };
static const NotAnEmitter kNotEmitters[] = {
    {"recompile_fragment_packet_impl",
     "shared body of the public integer and resource packet entries, BOTH validated here with "
     "actual original instruction inputs; it has no independent caller/configuration route"},
    {"shader_analysis_owned_words",
     "aliases the immutable owned RAW RDNA2 analysis bytes; it neither translates instructions "
     "nor assembles a SPIR-V module"},
    {"registered_graphics_original",
     "aliases the registered immutable RAW RDNA2 analysis bytes; consuming owned packet modules "
     "are validated here, but the accessor neither translates instructions nor assembles SPIR-V"},
    {"shader_source_snapshot",
     "copies bounded original RDNA2 words and checks complete encoded instruction prefixes; "
     "it neither translates those instructions nor assembles a SPIR-V module"},
    // Two SpirvCompute members became visible to this scan when the recompiler's shared internals
    // moved into rdna2_to_spirv_internal.hpp so the emit functions could be split into their own
    // translation units. Neither is a new code path -- both were always reached through the entry
    // points validated below; they were simply inside a .cpp, and this gate reads headers.
    {"build_interpolation_geometry",
     "the body of recompile_interpolation_geometry, which IS validated here: that entry point is "
     "`SpirvCompute builder; return builder.build_interpolation_geometry(layout, "
     "capture_position);` "
     "and nothing else, so every word this gate validates for it is emitted by this member"},
    {"finish",
     "SpirvCompute's module-assembly tail, not an entry point. It is the last call of every "
     "emitter "
     "validated here, so it is exercised by all of them; a module it broke would fail spirv-val "
     "under whichever emitter produced it"},
    {"safe_execz_branches_for_test", "returns a transformed RDNA2 instruction stream, not SPIR-V"},
    {"structured_execz_branches_for_test", "returns analyzed RDNA2 branch PCs, not SPIR-V"},
    {"mask_test_branches_for_test", "returns a transformed RDNA2 instruction stream, not SPIR-V"},
    {"cselect_b64_low_only_pcs_for_test", "returns CFG-proven instruction PCs, not SPIR-V"},
    {"rdna2_proven_raw_x2_data_loads",
     "returns decoded instruction PCs, not SPIR-V; dynfetch_fold covers positive, branch, "
     "pointer-only and SGPR-lifetime cases, and rdna2_to_spirv_exec validates a consuming module"},
    {"rdna2_proven_raw_register_wide_entry_loads",
     "returns proven raw-wide load PCs (the register proof's entry stage), not SPIR-V; "
     "memory_fed_raw_wide covers it through rdna2_proven_raw_register_wide_data_loads"},
    {"rdna2_proven_raw_immediate_wide_data_loads",
     "returns decoded instruction PCs, not SPIR-V; dynfetch_fold covers admission and refusal "
     "paths, and rdna2_to_spirv_exec validates a consuming module"},
    {"rdna2_proven_raw_register_wide_data_loads",
     "returns decoded instruction PCs, not SPIR-V; raw_register_wide_data covers complete "
     "scalar-offset provenance and raw_register_wide_exec executes changed source bytes"},
    {"rdna2_owned_raw_wide_data_loads",
     "returns decoded instruction PCs, not SPIR-V; owned_raw_wide_x4 and owned_raw_wide_x8 "
     "below assert those PCs and strictly validate consuming recompile_valu modules"},
    {"rdna2_proven_raw_nested_wide_data_loads",
     "returns decoded instruction PCs, not SPIR-V; recompile_coverage covers admitted numeric "
     "children and bypass refusals, and rdna2_to_spirv_exec validates consuming modules"},
    {"rdna2_raw_nested_numeric_loads",
     "returns decoded numeric-child PCs, not SPIR-V; owned_nested_vertex_x4/x8 and "
     "owned_nested_fragment_x4/x8 below assert the census and exact owned chains, then "
     "strictly validate the consuming direct-stage modules"},
    {"rdna2_raw_wide_data_loads",
     "returns decoded instruction PCs, not SPIR-V; recompile_coverage covers numeric reads, "
     "overwrites, branches and no-effect instructions, and validates consuming modules"},
    {"rdna2_raw_wave_wide_data_loads",
     "returns original numeric-load PCs, not SPIR-V; raw-register-wide and capture tests retain "
     "the obligation after bounded admission refuses; owned64 modules validate the consumer"},
    {"recompile_graphics_shader_cached",
     "caching wrapper; ctest shader_recompile_cache asserts its words are byte-identical to the "
     "direct emitter, which is validated here"},
    {"recompile_compute_shader_cached",
     "caching wrapper; ctest shader_recompile_cache asserts its words are byte-identical to the "
     "direct emitter, which is validated here"},
    {"recompile_graphics_shader_cached_shared",
     "caching wrapper returning shared immutable words; ctest shader_recompile_cache asserts "
     "*shared == recompile_vertex(...), i.e. byte-identical to the direct emitter"},
    {"recompile_vertex_chain_cached_shared",
     "caching wrapper over recompile_vertex_chain, which IS validated here. Note the difference "
     "from its siblings: shader_recompile_cache pins its cache identity and reuse, but does NOT "
     "compare its words against the direct emitter, so this entry rests on the wrapper adding no "
     "emission of its own"},
};

// Every declaration of a function returning SPIR-V words, whitespace-tolerant and with no
// name-prefix filter — both of those were false-PASS directions: a differently named emitter, or one
// written with a leading qualifier or an extra space, would simply not be seen. Struct members and
// parameters of the same type are excluded by requiring the '(' of a parameter list.
//
// BOTH spellings, and that is not cosmetic: gpu_execute.hpp's live draw path declares entry points
// as `SharedShaderWords` (an alias for shared_ptr<const vector<uint32_t>>), so keying only on the
// literal vector type left two of them neither covered NOR classifiable — a producer written in the
// idiom the live path already uses could be added with no failure and no mention, which is exactly
// the #1711 shape one level up.
//
// Note this scan has no comment handling, deliberately. The hazard runs the safe way now: a
// declaration-shaped line inside a header comment invents a phantom REQUIRED emitter and fails
// loudly, where the previous, textual coverage check could be silently SATISFIED by a comment.
// The same fail-loud direction covers a parenthesised local in inline header code
// (`std::vector<uint32_t> words(n);` reads as a declaration of an emitter named `words`). None
// exists today; if you write one and this gate goes red, that is why — use `=` or brace init.
static std::vector<std::string> declared_emitters(const std::string& header_text) {
    // FragmentPacketProgram owns the module plus its exact raw input/output buffers. Its different
    // return shape must not remove the genuine packet compiler entry from strict emitter coverage.
    static const char* const kReturnTypes[] = {
        "std::vector<uint32_t>", "SharedShaderWords", "FragmentPacketProgram",
        "FragmentResourcePacketProgram", "FragmentPacketKernel"};
    std::vector<std::string> names;
    for (const char* ret_type : kReturnTypes) {
        const std::string kRet = ret_type;
        size_t at = 0;
        while ((at = header_text.find(kRet, at)) != std::string::npos) {
            size_t i = at + kRet.size();
            while (i < header_text.size() && std::isspace((unsigned char)header_text[i])) ++i;
            std::string name;
            while (i < header_text.size() &&
                   (std::isalnum((unsigned char)header_text[i]) || header_text[i] == '_'))
                name.push_back(header_text[i++]);
            while (i < header_text.size() && std::isspace((unsigned char)header_text[i])) ++i;
            if (!name.empty() && i < header_text.size() && header_text[i] == '(')
                names.push_back(name);
            at += kRet.size();
        }
    }
    return names;
}

static int check_emitter_coverage(const std::string& src_root) {
    // Every header under these roots, not a hardcoded pair: an emitter declared in a NEW header is
    // precisely the #1711 shape one level up, and a fixed list cannot see it. RECURSIVE, and both
    // the GPU code and the frontend that drives it, so neither a new subdirectory nor a producer
    // that grows in the frontend re-creates that blind spot at the level of LOCATION rather than
    // filename. Nothing outside these roots emits SPIR-V today. The command that SHOWS that is
    // `git grep -l 0x07230203 -- prosper/src prosper/frontends`: three files, of which
    // rdna2_to_spirv.cpp and spirv_builder.cpp emit and shader_resources.cpp only PARSES. Grepping
    // the whole tree instead returns 29 files — test fixtures, gpu_replay reading a module back,
    // vendored imgui — and demonstrates nothing. If that changes, add the root here rather than
    // discovering it the way #1711 was discovered.
    static const char* const kSearchRoots[] = {"/src/gpu", "/frontends/shared"};
    std::vector<std::string> headers;
    for (const char* root : kSearchRoots) {
        const std::string dir = src_root + root;
        std::error_code ec;
        // Advanced explicitly with an error_code: the range-for's operator++ is the THROWING
        // overload, so an error arising mid-iteration would terminate instead of being reported.
        std::filesystem::recursive_directory_iterator it(dir, ec), done;
        for (; !ec && it != done; it.increment(ec))
            if (it->path().extension() == ".hpp") headers.push_back(it->path().string());
        if (ec) {
            printf("  [FAIL] emitter coverage: cannot enumerate %s (%s)\n", dir.c_str(),
                   ec.message().c_str());
            return 1;
        }
    }
    if (headers.empty()) {
        printf("  [FAIL] emitter coverage: no headers found under %s\n", src_root.c_str());
        return 1;
    }
    std::sort(headers.begin(), headers.end());

    int problems = 0;
    std::set<std::string> declared;
    for (const std::string& header : headers)
        for (const std::string& name : declared_emitters(read_text(header))) {
            bool excluded = false;
            for (const NotAnEmitter& n : kNotEmitters) excluded = excluded || name == n.name;
            if (!excluded) declared.insert(name);
        }

    for (const std::string& name : declared) {
        const bool exercised = exercised_emitters.count(name) != 0;
        const KnownGap* gap = nullptr;
        for (const KnownGap& g : kKnownGaps)
            if (name == g.emitter) gap = &g;
        if (exercised && gap) {
            printf("  [FAIL] emitter coverage: %s is now validated, so its known-gap entry is "
                   "stale -- delete it\n", name.c_str());
            ++problems;
        } else if (!exercised && !gap) {
            printf("  [FAIL] emitter coverage: %s emits SPIR-V that this gate never validates.\n"
                   "         Add a module for it, passing \"%s\" as dump()'s emitter argument; or\n"
                   "         record it in kKnownGaps with the issue that tracks it; or, if it does\n"
                   "         not emit SPIR-V at all, name it in kNotEmitters with the reason.\n",
                   name.c_str(), name.c_str());
            ++problems;
        } else if (!exercised) {
            printf("  [gap]  %-26s not validated: %s\n", name.c_str(), gap->reason);
        }
    }
    for (const KnownGap& g : kKnownGaps)
        if (!declared.count(g.emitter)) {
            printf("  [FAIL] emitter coverage: known-gap entry %s no longer names a declared "
                   "emitter -- delete it\n", g.emitter);
            ++problems;
        }
    // An emitter exercised below but absent from every header means the scan stopped seeing it.
    // That is the silent-coverage-loss direction, and it must not read as success.
    for (const std::string& name : exercised_emitters)
        if (!declared.count(name)) {
            printf("  [FAIL] emitter coverage: a module was emitted from %s, but the header scan no "
                   "longer finds it declared -- the scan has stopped working\n", name.c_str());
            ++problems;
        }
    if (!problems) {
        // Count, rather than say "all": a kKnownGaps entry takes the [gap] branch without adding a
        // problem, so "all N were exercised" would be false the moment that list is non-empty.
        size_t gaps = 0;
        for (const KnownGap& g : kKnownGaps) gaps += declared.count(g.emitter);
        printf("  [ok]   emitter coverage: %zu of %zu declared SPIR-V emitters were exercised%s\n",
               declared.size() - gaps, declared.size(),
               fails ? " (one or more of which FAILED above)" : ", and every module validated");
    }
    return problems;
}

// #3135 P2: the merged-NGG subgroup shell. The synthetic program covers two portable Wave64 waves,
// GS_ALLOC_REQ capture and set-2 shell I/O beside a guest binding 0; Kena's captured linked LUT
// producer covers the CFG dispatcher, both as one portable wave and as two native Wave64 waves.
static void dump_ngg_subgroup_shell(const std::string& dir, const std::string& src_root) {
    const auto synthetic = prosper::test::ngg::synthetic_program();
    const auto synthetic_rt = prosper::test::ngg::synthetic_resources();
    NggSubgroupShellConfig cfg;
    cfg.waves = 2;
    cfg.user_sgprs = 4;
    dump(dir, "ngg_subgroup_synthetic",
         recompile_ngg_subgroup(synthetic.data(), synthetic.size(), &synthetic_rt, cfg),
         "recompile_ngg_subgroup");
    cfg.native_wave64 = true;
    dump(dir, "ngg_subgroup_synthetic_native",
         recompile_ngg_subgroup(synthetic.data(), synthetic.size(), &synthetic_rt, cfg),
         "recompile_ngg_subgroup");
    const auto kena =
        prosper::test::ngg::kena_linked(std::filesystem::path(src_root) / "tests" / "data");
    const auto kena_rt = prosper::test::ngg::kena_resources(4);
    NggSubgroupShellConfig kena_cfg;
    kena_cfg.rsrc2_gs_lds_size = ngg_rsrc2_gs_lds_size(prosper::test::ngg::kKenaRsrc2Gs);
    kena_cfg.user_sgprs = prosper::test::ngg::kKenaUserSgprs;
    dump(dir, "ngg_subgroup_kena_lut",
         recompile_ngg_subgroup(kena.data(), kena.size(), &kena_rt, kena_cfg),
         "recompile_ngg_subgroup");
    kena_cfg.waves = 2;
    kena_cfg.native_wave64 = true;
    dump(dir, "ngg_subgroup_kena_lut_native2",
         recompile_ngg_subgroup(kena.data(), kena.size(), &kena_rt, kena_cfg),
         "recompile_ngg_subgroup");
}

static void dump_numeric_mbcnt(const std::string& dir) {
    // Numeric MBCNT is source-word DATA, not a peer population scan. Validate both half prefixes
    // in the compact and CFG compute routes and in both guest fragment wave widths.
    for (bool hi : {false, true}) {
        const std::array compute = {
            hi ? 0xd7660000u : 0xd7650000u, 0x00010affu, 0x80000001u, 0xbf810000u};
        for (bool cfg : {false, true}) {
            const std::string name = std::string("compute_numeric_mbcnt_") +
                (hi ? "hi" : "lo") + (cfg ? "_cfg" : "_compact");
            dump(dir, name.c_str(), recompile_valu(compute.data(), compute.size(), 1, 0,
                 nullptr, 0, kDefaultComputePgmRsrc1, cfg));
            const std::array float_bits = {
                hi ? 0xd7660000u : 0xd7650000u, 0x00010af2u, 0xbf810000u};
            const std::string float_name = std::string("compute_inline_float_mbcnt_") +
                (hi ? "hi" : "lo") + (cfg ? "_cfg" : "_compact");
            dump(dir, float_name.c_str(), recompile_valu(float_bits.data(), float_bits.size(),
                 1, 0, nullptr, 0, kDefaultComputePgmRsrc1, cfg));
        }
        const std::array fragment = {
            hi ? 0xd7660000u : 0xd7650000u, 0x00010affu, 0x80000001u,
            0x7e000d00u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
            0xf800180fu, 0x03020100u, 0xbf810000u};
        for (bool wave32 : {false, true}) {
            const std::string name = std::string("fragment_numeric_mbcnt_") +
                (hi ? "hi" : "lo") + (wave32 ? "_w32" : "_w64");
            dump(dir, name.c_str(), recompile_fragment(fragment.data(), fragment.size(),
                 nullptr, nullptr, UINT32_MAX, nullptr, wave32));
        }
    }
}

static void dump_numeric_wqm(const std::string& dir) {
    for (bool wide : {false, true}) {
        const std::array compute = {
            wide ? 0xbe940affu : 0xbe9409ffu, 0x80000001u, 0x7e000214u, 0xbf810000u};
        for (bool cfg : {false, true}) {
            const std::string name = std::string("compute_numeric_wqm_") +
                (wide ? "b64" : "b32") + (cfg ? "_cfg" : "_compact");
            dump(dir, name.c_str(), recompile_valu(compute.data(), compute.size(), 1, 0,
                 nullptr, 0, kDefaultComputePgmRsrc1, cfg));
        }
        const std::array fragment = {
            wide ? 0xbe940affu : 0xbe9409ffu, 0x80000001u, 0x7e000214u, 0x7e000d00u,
            0x7e020280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu, 0x03020100u, 0xbf810000u};
        for (bool wave32 : {false, true}) {
            const std::string name = std::string("fragment_numeric_wqm_") +
                (wide ? "b64" : "b32") + (wave32 ? "_w32" : "_w64");
            dump(dir, name.c_str(), recompile_fragment(fragment.data(), fragment.size(),
                 nullptr, nullptr, UINT32_MAX, nullptr, wave32));
        }
    }
}

static void dump_numeric_wqm_sources(const std::string& dir) {
    const std::array raw = {0xbe8403ffu, 0x80000000u, 0xbe8503ffu, 0x00000001u,
                            0xbe940a04u, 0x7e000215u, 0xbf810000u};
    dump(dir, "compute_numeric_wqm_raw_b64", recompile_valu(raw.data(), raw.size(), 1, 0));
    const std::array float_bits = {0xbe9409f2u, 0x7e000214u, 0xbf810000u};
    for (bool cfg : {false, true}) {
        const std::string name = std::string("compute_numeric_wqm_float_b32") +
            (cfg ? "_cfg" : "_compact");
        dump(dir, name.c_str(), recompile_valu(float_bits.data(), float_bits.size(), 1, 0,
             nullptr, 0, kDefaultComputePgmRsrc1, cfg));
    }
}

static void dump_wqm_mask_compatibility(const std::string& dir) {
    for (uint32_t operand : {128u, 143u, 193u, 208u}) {
        for (bool cfg : {false, true}) {
            std::vector<uint32_t> code{0xbe840a00u | operand};
            if (cfg) code.insert(code.end(), {0xbf820001u, 0xbf800000u});
            code.insert(code.end(), {0xbefe0404u, 0x7e000281u, 0xbefe04c1u, 0xbf810000u});
            const std::string name = "compute_wqm_exact_mask_" + std::to_string(operand) +
                (cfg ? "_cfg" : "_compact");
            dump(dir, name.c_str(), recompile_valu(code.data(), code.size(), 1, 0,
                 nullptr, 0, kDefaultComputePgmRsrc1, cfg));
        }
        if (operand != 128 && operand != 193) continue;
        std::vector<uint32_t> graphics{
            0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
            0xbe840a00u | operand, 0xbefe0404u, 0x7e000281u, 0xbefe04c1u,
            0xf80008cfu, 0x03020100u, 0xbf810000u,
        };
        const std::string suffix = std::to_string(operand);
        dump(dir, ("vertex_wqm_exact_mask_" + suffix).c_str(),
             recompile_vertex(graphics.data(), graphics.size()));
        graphics[8] = 0xf800000fu;
        dump(dir, ("fragment_wqm_exact_mask_" + suffix).c_str(),
             recompile_fragment(graphics.data(), graphics.size()));
    }
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    // The source root is required, not optional: the coverage check is the half of this gate that
    // survives the next emitter being added, and a check that silently skips itself is the defect
    // this tool exists to stop.
    if (argc <= 2 || argc > 4 || (argc == 4 && std::strcmp(argv[3], "--scalar-bank") != 0)) {
        printf("== FAIL: usage: spv_validate <output-dir> <source-root> [--scalar-bank] ==\n"
               "  <source-root> is prosper/ -- the emitter-coverage check reads its headers.\n");
        return 1;
    }
    const std::string src_root = argv[2];
    const bool scalar_bank_only = argc == 4;
    if (scalar_bank_only) {
#if !defined(PROSPER_SPV_VALIDATE_SCALAR_BANK)
        printf("== FAIL: --scalar-bank requires the compiled normal live producer ==\n");
        return 1;
#endif
    }

    // Validate the DECLARED form of every module (#3479/#3561). The SignedZeroInfNanPreserve
    // declaration is device-gated, and this tool owns no Vulkan device, so without this line it
    // would validate only the neutral form and the declaration would have no spirv-val coverage at
    // all. That coverage is not theoretical: spirv-val is the instrument that caught the emitter
    // naming capability 4467 (RoundingModeRTE) while the execution mode named 4461 -- a mismatch no
    // assertion sharing the emitter's own constants can see. Asserted true here for the same reason
    // the rest of this tool synthesizes inputs: it is validating what the emitter can produce, not
    // what this host happens to run.
    prosper::gpu::publish_float_controls_support(/*signed_zero_inf_nan_preserve_float32=*/true,
                                                 /*declaration_permitted_by_api=*/true);

    // Ask the directory directly rather than inferring writability from whether the validator probe
    // left anything behind: a spirv-val that exists but exits non-zero while printing nothing would
    // otherwise be diagnosed as an unwritable directory, sending the reader to the wrong place.
    const std::string probe = dir + "/spv_validate-probe.txt";
    if (FILE* w = fopen(probe.c_str(), "wb")) {
        fclose(w);
    } else {
        printf("== FAIL: cannot write into the output directory %s ==\n"
               "  Every module and the spirv-val probe are written there, so this run cannot\n"
               "  proceed. Check <output-dir>; this is NOT a missing spirv-tools install.\n",
               dir.c_str());
        return 1;
    }
    const bool have_val = (system(("spirv-val --version > \"" + probe + "\" 2>&1").c_str()) == 0);
    std::remove(probe.c_str());
    if (!have_val) {
        // Previously this printed "PASS (recompiled; spirv-val not found)" and exited 0, which is
        // how a gate documented as strict validation ran in CI for its whole life without ever
        // validating a module. Install spirv-tools (Fedora/Ubuntu: spirv-tools; macOS:
        // brew install spirv-tools; MSYS2: mingw-w64-ucrt-x86_64-spirv-tools).
        printf("== FAIL: spirv-val is not on PATH ==\n"
               "  This test IS the strict SPIR-V validation gate; without the validator it proves\n"
               "  only that the emitters returned bytes. Install spirv-tools and re-run.\n");
        return 1;
    }

    if (scalar_bank_only) {
#if defined(PROSPER_SPV_VALIDATE_SCALAR_BANK)
        validate_scalar_bank(dir);
        if (fails) {
            printf("== FAIL: %d scalar-bank emission/validation/contract failure(s) ==\n", fails);
            return 1;
        }
        printf("== PASS (six actual scalar-bank modules pass spirv-val; offline compiler inputs; "
               "no GPU or full-emitter coverage claim) ==\n");
        return 0;
#endif
    }

    // Compute ALU (float chain).
    { const uint32_t c[] = {0x8f148402u, 0xf4080200u, 0x28000010u,
                           0x7e000c09u, 0xbf810000u};
      ShaderResourceTable table;
      ShaderResource source;
      source.cls = ResourceClass::ConstantBuffer;
      source.format = DataFormat::Uint32;
      source.num_components = 1u;
      source.binding = 2u;
      // Complete owned bytes satisfy the raw snapshot's logical address/range contract.
      // This sample validates emitted SPIR-V; it does not execute an upload or GPU read.
      alignas(uint32_t) uint32_t bytes[4]{1u, 2u, 3u, 4u};
      source.gpu_addr = reinterpret_cast<uint64_t>(bytes);
      source.host_data = reinterpret_cast<uint8_t*>(bytes);
      source.host_data_size = sizeof(bytes);
      source.size = sizeof(bytes);
      source.fetch_pc = 1u;
      source.raw_register_snapshot = true;
      table.resources.push_back(source);
      dump(dir, "raw_register_wide_data", recompile_valu(c, std::size(c), 1, 0, &table),
           "recompile_valu"); }
    { const uint32_t c[] = {0xf4000101u, 0xfa000000u, 0xbe820380u, 0xbe830380u,
                           0x8f6b8404u, 0x876bff6bu, 0x000001f0u,
                           0xf4080200u, 0xd6000000u, 0x7e000c0bu, 0xbf810000u};
      ShaderResourceTable table;
      ShaderResource scalar;
      scalar.cls = ResourceClass::ConstantBuffer;
      scalar.format = DataFormat::Uint32;
      scalar.num_components = 1u;
      scalar.binding = 2u;
      scalar.size = 4u;
      scalar.fetch_pc = 0u;
      uint32_t selector = 2u;
      scalar.gpu_addr = 0x20000u;
      scalar.host_data = reinterpret_cast<uint8_t*>(&selector);
      scalar.host_data_size = sizeof(selector);
      table.resources.push_back(scalar);
      ShaderResource wide = scalar;
      wide.binding = 3u;
      wide.size = 16u;
      wide.fetch_pc = 7u;
      wide.raw_register_snapshot = true;
      wide.host_data = nullptr;
      wide.host_data_size = 0u;
      table.resources.push_back(wide);
      dump(dir, "memory_fed_raw_wide", recompile_valu(c, std::size(c), 1u, 0u, &table),
           "recompile_valu"); }
    // The parent and child both consume real complete hosted bytes. The parent pointer is
    // overwritten only after its read, and its highest word selects the register-offset child.
    // Strict validation here proves module legality, not renderer upload or GPU execution.
    for (bool wide8 : {false, true}) {
      const uint32_t size = wide8 ? 32u : 16u;
      const uint32_t last = wide8 ? 7u : 3u;
      const uint32_t c[] = {
          wide8 ? 0xf40c0600u : 0xf4080600u, 0xfa000004u,
          0xbe800380u, 0xbe810380u,
          0x8f6b8400u | (24u + last), 0x876bff6bu, 0x1f0u,
          wide8 ? 0xf40c0201u : 0xf4080201u, 0xd6000000u,
          wide8 ? 0x7e00020fu : 0x7e00020bu, 0xbf810000u};
      std::vector<Rdna2Inst> decoded;
      std::vector<uint32_t> sources;
      if (rdna2_walk(c, std::size(c), decoded) != std::size(c) ||
          rdna2_owned_raw_wide_data_loads(decoded) != std::vector<uint32_t>{0u} ||
          rdna2_proven_raw_register_wide_data_loads(decoded, &sources) !=
              std::vector<uint32_t>{7u} || sources != std::vector<uint32_t>{0u}) {
          printf("  [FAIL] owned_raw_wide_%s exact parent/child proof PCs\n", wide8 ? "x8" : "x4");
          ++fails;
      }
      alignas(16) std::array<uint32_t, 9> parent{};
      alignas(16) std::array<uint32_t, 8> child{};
      parent[1u + last] = 2u;
      child[last] = 42u;
      auto owner = std::make_shared<std::vector<uint8_t>>(size);
      std::memcpy(owner->data(), parent.data() + 1u, size);
      ShaderResourceTable table;
      ShaderResource source;
      source.cls = ResourceClass::ConstantBuffer;
      source.format = DataFormat::Uint32;
      source.num_components = 1u;
      source.binding = 2u;
      source.gpu_addr = reinterpret_cast<uint64_t>(parent.data() + 1u);
      source.host_data = owner->data();
      source.host_data_size = size;
      source.size = size;
      source.fetch_pc = 0u;
      source.owned_raw_snapshot_bytes = size;
      table.resources.push_back(source);
      table.owned_host_data.push_back(owner);
      table.owned_raw_snapshot_requirements.emplace_back(0u, size);
      ShaderResource selected = source;
      selected.binding = 3u;
      selected.fetch_pc = 7u;
      selected.gpu_addr = reinterpret_cast<uint64_t>(child.data());
      selected.host_data = reinterpret_cast<uint8_t*>(child.data());
      selected.raw_register_snapshot = true;
      selected.owned_raw_snapshot_bytes = 0u;
      table.resources.push_back(selected);
      auto short_parent = table;
      short_parent.resources.front().host_data_size = size - 4u;
      if (!recompile_valu(c, std::size(c), 1u, 0u, &short_parent).empty()) {
          printf("  [FAIL] owned_raw_wide_%s missing highest parent word was admitted\n",
                 wide8 ? "x8" : "x4");
          ++fails;
      }
      dump(dir, wide8 ? "owned_raw_wide_x8" : "owned_raw_wide_x4",
           recompile_valu(c, std::size(c), 1u, 0u, &table), "recompile_valu");
    }
    // One-hop direct graphics consumes two complete owners, not a descriptor placeholder.
    // Independently pin the numeric-child census and the parent/child proof before validating
    // both widths in both actual stage emitters. This proves module legality, not guest-byte
    // visibility, renderer upload, executing-device features or numeric execution.
    for (bool wide8 : {false, true}) for (bool fragment : {false, true}) {
      const uint32_t width = wide8 ? 32u : 16u;
      const uint32_t code[] = {
          wide8 ? 0xf40c0a00u : 0xf4080a00u, 0xfa000004u,
          wide8 ? 0xf40c0b14u : 0xf4080b14u, 0xfa000010u,
          0x7e000200u | (wide8 ? 51u : 47u),
          0x7e020280u, 0x7e0402f2u, 0x7e0602f2u,
          fragment ? 0xf800180fu : 0xf80008cfu, 0x03020100u, 0xbf810000u};
      const std::string name = std::string("owned_nested_") +
          (fragment ? "fragment_" : "vertex_") + (wide8 ? "x8" : "x4");
      std::vector<Rdna2Inst> decoded;
      if (rdna2_walk(code, std::size(code), decoded) != std::size(code) ||
          rdna2_raw_nested_numeric_loads(decoded) != std::vector<uint32_t>{2u} ||
          rdna2_owned_nested_wide_chains(decoded) !=
              std::vector<RawNestedWideChain>{{0u, 2u, width, width, 4u, 16u}}) {
          printf("  [FAIL] %s exact numeric census and parent/child chain\n", name.c_str());
          ++fails;
      }
      ShaderResourceTable table;
      for (uint32_t pc : {0u, 2u}) {
          auto owner = std::make_shared<std::vector<uint8_t>>(width, 0u);
          if (pc == 0u) {
              const uint64_t child_pointer = 0x300000u;
              std::memcpy(owner->data(), &child_pointer, sizeof(child_pointer));
          } else {
              const uint32_t numeric_bits = 0x3f000000u;
              std::memcpy(owner->data() + width - sizeof(numeric_bits),
                          &numeric_bits, sizeof(numeric_bits));
          }
          ShaderResource resource;
          resource.cls = ResourceClass::ConstantBuffer;
          resource.format = DataFormat::Uint32;
          resource.num_components = 1u;
          resource.binding = pc ? 3u : 2u;
          resource.fetch_pc = pc;
          resource.gpu_addr = pc ? 0x300010u : 0x200004u;
          resource.size = resource.host_data_size = width;
          resource.host_data = owner->data();
          resource.owned_nested_snapshot_bytes = width;
          table.resources.push_back(resource);
          table.owned_host_data.push_back(std::move(owner));
          table.owned_nested_snapshot_requirements.emplace_back(pc, width);
      }
      dump(dir, name.c_str(), fragment
          ? recompile_fragment(code, std::size(code), &table)
          : recompile_vertex(code, std::size(code), &table),
          fragment ? "recompile_fragment" : "recompile_vertex");
    }
    { const uint32_t c[] = {0x06000300u, 0x10000500u, 0xBF810000u};
      dump(dir, "compute_alu", recompile_valu(c, 3, 3, 0), "recompile_valu"); }
    dump_numeric_mbcnt(dir);
    dump_numeric_wqm(dir);
    dump_numeric_wqm_sources(dir);
    dump_wqm_mask_compatibility(dir);
    // #4056: real mode-selected integer predicates feed both MRT0 and an Any-controlled live Phi.
    // Validate SOURCE plus admitted EFFECTIVE, not only an isolated hand-written predicate.
    for (uint32_t op : {2u,13u}) for (uint8_t mode : {uint8_t{0},uint8_t{16},uint8_t{32},uint8_t{48}})
        for (bool wave32 : {false,true}) {
            const uint32_t c[] = {
                0xf4200500u,0xfa000000u,0x7e000214u,0x7e1202ffu,0x40000000u,
                0xd400006au | (op<<16),128u | (256u<<9),
                0xd5010008u,128u | (242u<<9) | (106u<<18),
                0xbf870002u,0x7e1202ffu,0x40400000u,
                0xf800180fu,0x09080908u,0xbf810000u,
            };
            ShaderResourceTable table;
            ShaderResource buffer;
            buffer.cls=ResourceClass::ConstantBuffer; buffer.format=DataFormat::Uint32;
            buffer.sgpr_base=0; buffer.binding=32; buffer.size=4;
            table.resources.push_back(buffer);
            const auto source=recompile_fragment(c,std::size(c),&table,nullptr,UINT32_MAX,nullptr,
                wave32,{RecompileDiagnosticStage::Fragment,0},{true,mode});
            const auto name="fragment_float_mode_op"+std::to_string(op)+"_mode"+std::to_string(mode)+
                (wave32?"_wave32":"_wave64");
            dump(dir,name.c_str(),source,"recompile_fragment");
            if (!wave32)
                dump(dir,(name+"_effective").c_str(),lower_fragment_votes(source,true,true).words);
        }
    for (const auto& fixture : prosper::test::wave_width::fixtures()) {
        if (!fixture.strict_vulkan) continue; // unsupported-environment parser controls, not modules
        const std::string name = "wave_width_" + fixture.name;
        dump(dir, name.c_str(), fixture.words);
    }
    for (const auto& fixture : prosper::test::fragment_votes::fixtures()) {
        const std::string name = "fragment_vote_" + fixture.name;
        dump(dir, name.c_str(), fixture.words);
        if (fixture.admitted) {
            const auto lowered = lower_fragment_votes(fixture.words, fixture.immutable_storage, true);
            dump(dir, (name + "_effective").c_str(), lowered.words);
        }
    }
    namespace vote_execution = prosper::test::fragment_vote_execution;
    for (uint32_t y = 2; y < 4; ++y) for (uint32_t x = 2; x < 4; ++x) {
        const auto name = "fragment_vote_edge_vertex_" + std::to_string(x) + "_" + std::to_string(y);
        dump(dir, name.c_str(), vote_execution::edge_vertex(x, y));
    }
    dump(dir, "fragment_vote_helper_witness", vote_execution::helper_witness());
    for (bool poison : {false, true}) {
        const std::string name = poison ? "fragment_vote_edge_poison" : "fragment_vote_edge";
        const auto source = vote_execution::derivative_fragment(poison);
        dump(dir, name.c_str(), source);
        dump(dir, (name + "_effective").c_str(), lower_fragment_votes(source).words);
    }
    namespace loop_votes = prosper::test::fragment_loop_votes;
    for (const auto shape : {loop_votes::Shape::Counter, loop_votes::Shape::Nested,
                             loop_votes::Shape::CrossCarried, loop_votes::Shape::BoolToggle,
                             loop_votes::Shape::CounterWithDeadVote}) {
        const bool counter = shape == loop_votes::Shape::Counter;
        const bool changing_bool = shape == loop_votes::Shape::BoolToggle;
        for (uint32_t bound = counter ? 0 : changing_bool ? 1 : 2;
             bound < (counter || changing_bool ? 4u : 3u); ++bound) {
            const std::string name = "fragment_vote_loop_poison_" +
                std::to_string(static_cast<uint32_t>(shape)) + "_" + std::to_string(bound);
            const auto source = loop_votes::make_module(shape, bound, true);
            dump(dir, name.c_str(), source);
            dump(dir, (name + "_effective").c_str(), lower_fragment_votes(source).words);
        }
    }
    namespace neutral = prosper::test::fragment_neutral;
    {
        namespace resources = prosper::test::fragment_resource_packet;
        for (bool lod : {false, true}) {
            const auto packet = recompile_fragment_resource_packet(resources::chain(lod, true));
            dump(dir, lod ? "fragment_resource_packet_l" : "fragment_resource_packet_lz",
                 packet.packet.spirv, "recompile_fragment_resource_packet");
        }
        const auto rectangular = recompile_fragment_resource_packet(resources::rectangular_chain());
        dump(dir, "fragment_resource_packet_rectangular_l4", rectangular.packet.spirv,
             "recompile_fragment_resource_packet");
        namespace definedness = prosper::test::fragment_definedness;
        for (bool lod : {false, true})
            for (bool inactive : {false, true}) {
                const auto packet =
                    recompile_fragment_resource_packet(definedness::resource_chain(lod, inactive));
                const auto name = "fragment_resource_definedness_" + std::to_string(lod) + "_" +
                                  std::to_string(inactive);
                dump(dir, name.c_str(), packet.packet.spirv, "recompile_fragment_resource_packet");
            }
        const auto missing_lod =
            recompile_fragment_resource_packet(definedness::resource_missing_lod());
        dump(dir, "fragment_resource_definedness_third_lod_absent", missing_lod.packet.spirv,
             "recompile_fragment_resource_packet");
        namespace special = prosper::test::fragment_special_f32;
        namespace raw_masks = prosper::test::fragment_raw_masks;
        for (const auto& c :
             {raw_masks::overwritten(raw_masks::asymmetric, false, false),
              raw_masks::overwritten(raw_masks::asymmetric, true, true), raw_masks::wqm_saved(),
              raw_masks::saveexec_scc(false), raw_masks::not_scc(false)}) {
            const auto packet = recompile_fragment_packet(c.packet);
            dump(dir, ("fragment_raw_masks_" + c.name).c_str(), packet.spirv,
                 "recompile_fragment_packet");
        }
        namespace mask_entry = prosper::test::fragment_mask_entry;
        for (const auto& c : {mask_entry::exec_writer(raw_masks::asymmetric, true),
                              mask_entry::new_scc(), mask_entry::cmpx_vcc(false),
                              mask_entry::peer_before_exec(), mask_entry::wqm_numeric()}) {
            const auto packet = recompile_fragment_packet(c.packet);
            dump(dir, ("fragment_mask_entry_" + c.name).c_str(), packet.spirv,
                 "recompile_fragment_packet");
        }
        const auto mask_entry_kernel =
            recompile_fragment_packet_kernel(mask_entry::wave_input(mask_entry::cases().front()));
        dump(dir, "fragment_mask_entry_cached_kernel", mask_entry_kernel.program.packet.spirv,
             "recompile_fragment_packet_kernel");
        const auto wave_kernel =
            recompile_fragment_packet_kernel(prosper::test::fragment_packet_wave::packet());
        dump(dir, "fragment_packet_wave_kernel", wave_kernel.program.packet.spirv,
             "recompile_fragment_packet_kernel");
        const auto wave_image_kernel = recompile_fragment_packet_kernel(
            prosper::test::fragment_packet_wave::resource::chain());
        dump(dir, "fragment_packet_wave_image_kernel", wave_image_kernel.program.packet.spirv,
             "recompile_fragment_packet_kernel");
        const auto entry_m0_kernel = recompile_fragment_packet_kernel(
            prosper::test::fragment_packet_wave::entry_m0_packet(0));
        dump(dir, "fragment_packet_wave_entry_m0_kernel", entry_m0_kernel.program.packet.spirv,
             "recompile_fragment_packet_kernel");
        const auto scalar_exec_kernel = recompile_fragment_packet_kernel(
            prosper::test::fragment_packet_wave::scalar_exec_packet(0));
        dump(dir, "fragment_packet_wave_scalar_exec_kernel",
             scalar_exec_kernel.program.packet.spirv, "recompile_fragment_packet_kernel");
        // A separate original resource-free PS represents the first assembly recipe. The wave
        // fixture above genuinely consumes SMEM/M0/P1/P2 and must not have its resources stripped
        // to make this representative emit. Hand-owned user words remain distinct from WAT2
        // placement authority; this factory does not establish shipping launch or attachment data.
        auto draw_schema = resources::base();
        auto& draw_invocation = draw_schema.invocation;
        draw_invocation.export_observation = FragmentPacketExportObservation::Architectural;
        draw_invocation.mask_state_available = false;
        draw_invocation.vgprs.clear();
        draw_invocation.sgprs = {{0, resources::bits(.25f)},
                                 {1, resources::bits(.5f)},
                                 {2, resources::bits(.75f)},
                                 {3, resources::bits(1.0f)}};
        // The original EXEC=-1 dominates every vector writer/export. No guest initial mask,
        // system input, helper value, parameter coefficient or scratch VGPR is invented.
        draw_invocation.guest_code = {0xbefe04c1u, 0x7e000200u, 0x7e020201u, 0x7e040202u,
                                      0x7e060203u, 0xf800180fu, 0x03020100u, 0xbf810000u};
        auto draw_kernel = std::make_shared<const FragmentPacketKernel>(
            recompile_fragment_packet_capacity_kernel(draw_schema));
        dump(dir, "fragment_draw_capacity_kernel", draw_kernel->program.packet.spirv,
             "recompile_fragment_packet_capacity_kernel");
        RasterQuadCollector draw_collector;
        draw_collector.max_quads = 48;
        draw_collector.lane_words = kRasterQuadLaneFixedWords;
        draw_collector.record_words = 4 * draw_collector.lane_words;
        std::string draw_rejection;
        const auto draw_capacity =
            fragment_draw_capacity(draw_kernel, draw_collector, draw_rejection);
        if (!draw_capacity) {
            printf("  [FAIL] fragment draw capacity: %s\n", draw_rejection.c_str());
            ++fails;
        } else {
            dump(dir, "fragment_draw_count",
                 build_fragment_draw_count(*draw_capacity, draw_collector),
                 "build_fragment_draw_count");
            dump(dir, "fragment_draw_assembly",
                 build_fragment_draw_assembly(*draw_capacity, draw_collector),
                 "build_fragment_draw_assembly");
            dump(dir, "fragment_draw_validation",
                 build_fragment_draw_validation(*draw_capacity, draw_collector),
                 "build_fragment_draw_validation");
            dump(dir, "fragment_draw_replay",
                 build_fragment_draw_replay(*draw_capacity, draw_collector),
                 "build_fragment_draw_replay");
        }
        namespace architectural = prosper::test::fragment_packet_exports;
        for (const auto& [name, input] :
             std::vector<std::pair<const char*, FragmentResourcePacket>>{
                 {"scratch", architectural::scratch()},
                 {"multiple", architectural::multiple()},
                 {"compressed", architectural::compressed()},
                 {"p2", architectural::previous_destination()},
                 {"wait_cmpx", architectural::pending_write(3, true)},
                 {"wait_join", architectural::pending_join(true, false)},
                 {"numeric_saveexec", architectural::numeric_saveexec(3)},
                 {"wait_image", architectural::pending_image(true)}}) {
            const auto compiled = recompile_fragment_resource_packet(input);
            dump(dir, (std::string("fragment_architectural_export_") + name).c_str(),
                 compiled.packet.spirv, "recompile_fragment_resource_packet");
        }
        const auto architectural_kernel =
            recompile_fragment_packet_kernel(architectural::scratch());
        dump(dir, "fragment_architectural_export_cached", architectural_kernel.program.packet.spirv,
             "recompile_fragment_packet_kernel");
        for (uint32_t op : {0x2au, 0x2eu, 0x33u}) {
            const auto p =
                recompile_fragment_resource_packet(special::packet(op, special::rails(op)));
            dump(dir, ("fragment_special_f32_" + std::to_string(op)).c_str(), p.packet.spirv,
                 "recompile_fragment_resource_packet");
        }
    }
    {
        namespace fp = prosper::test::fragment_packet;
        uint32_t ordinal = 0;
        for (const auto& c : prosper::test::fragment_definedness::cases()) {
            const auto packet = recompile_fragment_packet(c.packet);
            const auto name = "fragment_definedness_" + c.name;
            dump(dir, name.c_str(), packet.spirv, "recompile_fragment_packet");
        }
        for (uint32_t selected : {63u, 64u}) for (bool inactive : {false, true}) {
            fp::Case c;
            c.selected_lane = selected;
            c.inactive_source = inactive;
            c.leave_source_inactive = inactive;
            c.second_export = true;
            const auto compiled = recompile_fragment_packet(fp::packet(c));
            const auto name = "fragment_packet_" + std::to_string(ordinal++);
            dump(dir, name.c_str(), compiled.spirv, "recompile_fragment_packet");
        }
        for (const auto source : {fp::wqm::Source::Exec, fp::wqm::Source::Vcc,
                                  fp::wqm::Source::ScalarPair, fp::wqm::Source::Empty}) {
            fp::wqm::Case c;
            c.source = source;
            c.destination = source == fp::wqm::Source::Vcc ? 106 : 16;
            const auto compiled = recompile_fragment_packet(fp::wqm::packet(c));
            const auto name = "fragment_packet_wqm_" + std::to_string(ordinal++);
            dump(dir, name.c_str(), compiled.spirv, "recompile_fragment_packet");
        }
        for (const auto source : {fp::mbcnt::Source::Exec, fp::mbcnt::Source::SavedVcc,
                                  fp::mbcnt::Source::Scalar, fp::mbcnt::Source::Vgpr,
                                  fp::mbcnt::Source::Literal, fp::mbcnt::Source::Full}) {
            fp::mbcnt::Case c; c.source = source; c.exec = uint64_t(1) << 63;
            const auto compiled = recompile_fragment_packet(fp::mbcnt::packet(c));
            const auto name = "fragment_packet_mbcnt_" + std::to_string(ordinal++);
            dump(dir, name.c_str(), compiled.spirv, "recompile_fragment_packet");
        }
        for (bool width8 : {false, true})
            for (uint32_t first : {63u, 64u}) {
                FragmentInvocationPacket packet;
                packet.slots_available.fill(true);
                packet.export_enabled.fill(1);
                packet.mask_state_available = true;
                packet.exec_mask = first == 64u ? 0u : uint64_t(1) << first;
                packet.sgprs = {{0u, 0x10000000u}, {1u, 0u}, {16u, UINT32_MAX}, {17u, UINT32_MAX}};
                packet.guest_code = {0x7e280500u,
                                     0x87148f14u,
                                     0x8f148414u,
                                     width8 ? 0xf40c0200u : 0xf4080200u,
                                     0x28000010u,
                                     0xbefe0410u,
                                     width8 ? 0x7e02020fu : 0x7e02020bu,
                                     0xf8001801u,
                                     1u,
                                     0xbf810000u};
                FragmentPacketVgpr column;
                column.reg = 0u;
                for (uint32_t lane = 0; lane < 64u; ++lane) column.words[lane] = lane;
                packet.vgprs.push_back(column);
                PacketRawWaveWindow owner;
                owner.load_pc = 3u;
                owner.guest_base = 0x10000000u;
                owner.guest_begin = owner.guest_base + 16u;
                owner.words.resize((240u + (width8 ? 32u : 16u)) / 4u);
                for (uint32_t word = 0; word < owner.words.size(); ++word)
                    owner.words[word] = 0x41000000u + word;
                packet.raw_windows.push_back(std::move(owner));
                const auto compiled = recompile_fragment_packet(packet);
                const auto name = "fragment_packet_owned_raw_x" + std::to_string(width8 ? 8u : 4u) +
                                  "_first_" + std::to_string(first);
                dump(dir, name.c_str(), compiled.spirv, "recompile_fragment_packet");
            }
        for (bool width8 : {false, true}) {
            FragmentInvocationPacket packet;
            packet.slots_available.fill(true);
            packet.export_enabled.fill(1);
            packet.mask_state_available = true;
            packet.exec_mask = UINT64_MAX;
            packet.sgprs = {{0u, 0x10000000u}, {1u, 0u}};
            // Two visits to the SAME event/load PC with a changed current source. A cycle with
            // no EXP is admitted; the final export is outside it. Validate this real CFG too.
            // s40 is outside both SMEM destinations s[24:27] and s[24:31].
            packet.guest_code = {0xbea80380u, 0x7e280501u,
                                 0x90149714u, 0x87148314u,
                                 0x8f148414u, width8 ? 0xf40c0600u : 0xf4080600u,
                                 0x28000000u, 0x7e000200u | (width8 ? 31u : 27u),
                                 0x80288128u, 0x060202ffu,
                                 0x40800000u, 0xbf0a8228u,
                                 0xbf85fff4u, 0xf8001801u,
                                 0u,          0xbf810000u};
            FragmentPacketVgpr column;
            column.reg = 1u;
            for (uint32_t lane = 0; lane < 64u; ++lane)
                column.words[lane] = std::bit_cast<uint32_t>(float(lane / 8u) + 0.5f);
            packet.vgprs.push_back(column);
            PacketRawWaveWindow owner;
            owner.load_pc = 5u;
            owner.guest_base = owner.guest_begin = 0x10000000u;
            owner.words.resize((48u + (width8 ? 32u : 16u)) / 4u, 0x3f400000u);
            packet.raw_windows.push_back(std::move(owner));
            const auto compiled = recompile_fragment_packet(packet);
            const auto name =
                "fragment_packet_owned_raw_repeated_x" + std::to_string(width8 ? 8u : 4u);
            dump(dir, name.c_str(), compiled.spirv, "recompile_fragment_packet");
        }
        const auto commit = build_owned_vertex_export_commit({0u, 3u}, 64u, nullptr, {});
        dump(dir, "owned_vertex_export_commit", commit, "build_owned_vertex_export_commit");
        dump(dir, "owned_fragment_export_commit", build_owned_fragment_export_commit(64u, {}),
             "build_owned_fragment_export_commit");
    }
    {
        namespace bp = prosper::test::bpermute;
        uint32_t ordinal = 0;
        for (const auto& c : bp::cases()) {
            const auto name = "portable_bpermute_" + std::to_string(ordinal++);
            dump(dir, name.c_str(), bp::compile(c), "recompile_compute");
        }
    }
    for (const auto& fixture : neutral::fixtures()) {
        if (!fixture.strict) continue;
        const std::string name = std::string("fragment_neutral_") + fixture.name;
        const auto source = neutral::make_module(fixture.shape);
        dump(dir, name.c_str(), source);
        if (fixture.admitted) dump(dir, (name + "_effective").c_str(), lower_fragment_votes(source).words);
    }
    for (const auto predicate : {neutral::Predicate::Helpers, neutral::Predicate::Visible,
                                neutral::Predicate::AllFalse, neutral::Predicate::AllTrue}) {
        for (const bool poison : {false, true}) {
            const auto name = "fragment_neutral_quad_" +
                std::to_string(static_cast<uint32_t>(predicate)) + (poison ? "_poison" : "");
            const auto source = neutral::make_module(neutral::Shape::Masked, predicate, poison);
            dump(dir, name.c_str(), source);
            dump(dir, (name + "_effective").c_str(), lower_fragment_votes(source).words);
        }
    }
    // Validate the exact newly executed GPU arms, including the helper-poison fault variants.
    for (const auto shape : {neutral::Shape::Termination, neutral::Shape::UndefinedConjunction,
                             neutral::Shape::MaskedFloat, neutral::Shape::MaskedBitcastPoison,
                             neutral::Shape::BitcastPoisonSelection, neutral::Shape::FloatPoisonSelection}) {
        for (const bool poison : {false, true}) {
            const auto name = "fragment_neutral_extra_" +
                std::to_string(static_cast<uint32_t>(shape)) + (poison ? "_poison" : "");
            const auto source = neutral::make_module(shape, neutral::Predicate::AllFalse, poison);
            dump(dir, name.c_str(), source);
            dump(dir, (name + "_effective").c_str(), lower_fragment_votes(source).words);
        }
    }
    const auto terminated = neutral::make_module(neutral::Shape::Termination, neutral::Predicate::AllTrue);
    dump(dir, "fragment_neutral_termination_true", terminated);
    dump(dir, "fragment_neutral_termination_true_effective", lower_fragment_votes(terminated).words);
    const auto buffered_neutral = neutral::make_module(neutral::Shape::BufferPredicate);
    dump(dir, "fragment_neutral_buffer_certified_effective",
         lower_fragment_votes(buffered_neutral, true, true).words);
    // GTA V's exact literal-bearing V_ALIGNBYTE_B32 packet.  Strict validation guards the
    // masked-shift lowering: SPIR-V shift operands must stay in the defined 0..31 range.
    { const uint32_t c[] = {0xd54f0006u,0x0415fe80u,0x3024240cu,0xbf810000u};
      dump(dir, "compute_alignbyte", recompile_valu(c, std::size(c), 6, 6), "recompile_valu"); }
    // GTA V exec_cs_413d1bf00 pc458: exact V_LDEXP_F32 production packet. This exercises the
    // integer-domain edge lowering under the strict Vulkan SPIR-V validator, not merely the generic
    // recompile_valu entry point above.
    { const uint32_t c[] = {0xd7620000u, 0x0002030du, 0xbf810000u};
      dump(dir, "compute_ldexp", recompile_valu(c, sizeof(c) / sizeof(c[0]), 14, 0)); }
    // #4060: one ordinary numeric scalar word; FindILsb preserves the zero sentinel and
    // produces scalar data without a guest-wave reduction or a native-subgroup requirement.
    { const uint32_t c[] = {
          0xbea403ffu, 0x00010000u, // s_mov_b32 s36,0x10000
          0xbea51324u,             // s_ff1_i32_b32 s37,s36
          0x7e000225u,             // v_mov_b32 v0,s37
          0xbf810000u};
      dump(dir, "compute_scalar_ff1_b32", recompile_valu(c, std::size(c), 0, 0)); }
    // Compute + SMEM constant-buffer load (s_buffer_load_dword; routes to binding 2).
    { const uint32_t c[] = {0xf4000000u, 0xfa000004u, 0x7e000200u, 0xbf810000u};
      dump(dir, "compute_smem", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Compute + immediate SMEM x16 descriptor bundle. Both eight-dword halves are independently
    // resolved by their consuming MIMG PCs; the load's one immediate offset is not a shared key.
    { const uint32_t c[] = {
          0xf4100300u, 0xfa000000u,
          0xf0800f08u, 0x01630000u,
          0xf0800f08u, 0x01850000u,
          0xbf810000u};
      ShaderResourceTable rt;
      for (uint32_t i = 0; i < 2; ++i) {
          ShaderResource t{};
          t.cls = ResourceClass::Texture;
          t.format = DataFormat::Float32;
          t.num_components = 4;
          t.binding = 4 + i;
          t.fetch_pc = 2 + i * 2;
          t.img_dim = 1;
          t.width = t.height = 2;
          rt.resources.push_back(t);
      }
      dump(dir, "compute_smem_x16_descriptor_bundle",
           recompile_valu(c, sizeof(c)/4, 2, 0, &rt)); }
    // Compute private spill/fill, including a signed short crossing a dword boundary.
    { const uint32_t c[] = {0x7e0002ffu,0x00008001u,0xdc684003u,0x00000000u,0x7e000280u,
                            0xdc2c4003u,0x00000000u,0x7e000b00u,0xBF810000u};
      dump(dir, "compute_private_spill", recompile_valu(c, sizeof(c)/4, 0, 0)); }
    // Fragment: solid green (EXP MRT0).
    { const uint32_t c[] = {0x7E000280u,0x7E0202F2u,0x7E040280u,0x7E0602F2u,0xF800180Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_color", recompile_fragment(c, sizeof(c)/4), "recompile_fragment"); }
    // Fragment: Astro's exact wave64 MBCNT + device-global append allocation shape.
    { const uint32_t c[] = {
          0xD7660007u,0x0001007Fu,0xBEFC0380u,0xD8FA0014u,0x06000000u,
          0xD7650000u,0x00020E7Eu,0x4A140106u,0x36001481u,0x7E000D00u,
          0x7E020280u,0x7E040280u,0x7E0602F2u,0xF800180Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_gds_append", recompile_fragment(c, sizeof(c)/4)); }
    // Same allocation with a live implicit-LOD sample: sampled alpha reaches EXP and forces WQM.
    { const uint32_t c[] = {
          0x7E0002FFu,0x3E800000u,0x7E0202FFu,0x3E800000u,
          0xF0800F08u,0x00820000u,0xD7660007u,0x0001007Fu,0xBEFC0380u,
          0xD8FA0014u,0x06000000u,0xD7650000u,0x00020E7Eu,0x4A140106u,
          0x36001481u,0x7E000D00u,0x7E020280u,0x7E040280u,
          0xF800180Fu,0x03020100u,0xBF810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture;
      t.binding=4; t.img_dim=1; t.width=2; t.height=2; t.sgpr_base=8;
      rt.resources.push_back(t);
      dump(dir, "fragment_gds_wqm", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Consume shares the same wave allocator but subtracts the non-helper population.
    { const uint32_t c[] = {
          0xD7660007u,0x0001007Fu,0xBEFC0380u,0xD8F60014u,0x06000000u,
          0xD7650000u,0x00020E7Eu,0x4A140106u,0x36001481u,0x7E000D00u,
          0x7E020280u,0x7E040280u,0x7E0602F2u,0xF800180Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_gds_consume", recompile_fragment(c, sizeof(c)/4)); }
    // Fragment private spill/fill (Function-storage declaration in the graphics shell).
    { const uint32_t c[] = {0xdc704010u,0x00000000u,0x7e000280u,0xdc304010u,0x00000000u,
                            0xf800000fu,0x00000000u,0xBF810000u};
      dump(dir, "fragment_private_spill", recompile_fragment(c, sizeof(c)/4)); }
    // Vertex: fullscreen triangle from gl_VertexIndex (EXP POS0).
    { const uint32_t c[] = {0x36020081u,0x2C040081u,0x7E020D01u,0x7E040D02u,0x7E0A02F6u,0x7E0C02F2u,0x10020B01u,
                            0x08020D01u,0x10040B02u,0x08040D02u,0x7E060280u,0x7E0802F2u,0xF80008CFu,0x04030201u,0xBF810000u};
      dump(dir, "vertex_fullscreen", recompile_vertex(c, sizeof(c)/4), "recompile_vertex");
      PixelInputMapping consumed;
      consumed.valid_mask = 0xffffffffu;
      consumed.consumed_known = true;
      consumed.consumed_mask = 1u;
      dump(dir, "vertex_unwritten_consumed_param",
           recompile_vertex(c, sizeof(c)/4, nullptr, &consumed));
      // Geometry-probe capture variant: gl_Position decorated for transform-feedback readback. Must
      // still pass spirv-val (Xfb capability + execution mode + member Offset/XfbBuffer/XfbStride).
      dump(dir, "vertex_xfb_capture", recompile_vertex(c, sizeof(c)/4, nullptr, nullptr, true)); }
    // Vertex private spill/fill (Function-storage declaration in the graphics shell).
    { const uint32_t c[] = {0xdc704010u,0x00000000u,0x7e000280u,0xdc304010u,0x00000000u,
                            0xf80008cfu,0x00000000u,0xBF810000u};
      dump(dir, "vertex_private_spill", recompile_vertex(c, sizeof(c)/4)); }
    // Vertex fetch (buffer_load_format_xy from a V# in user-data s[8:11]).
    { const uint32_t c[] = {0x7e060280u,0x7e0802f2u,0xe0042000u,0x80020100u,0xf80008cfu,0x04030201u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=2; vb.binding=3; vb.stride=8; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "vertex_fetch", recompile_vertex(c, sizeof(c)/4, &rt)); }
    // A compute fetch through a bounded, non-uniform descriptor array. This representative is kept in
    // the strict corpus because capability-number mistakes can pass the emitter's word-level tests and
    // even some drivers while spirv-val correctly rejects the module.
    { const uint32_t c[] = {0x7e060280u,0x7e0802f2u,0xe0300000u,0x80020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Uint32;
      vb.num_components=1; vb.binding=3; vb.stride=4; vb.sgpr_base=8;
      vb.table_index_count=4; vb.table_entry_stride=16; vb.table_index_sgpr=6;
      vb.table_selector_mode=BufferTableSelectorMode::UserSgprIndex;
      for (uint32_t index=0; index<vb.table_index_count; ++index) {
          ShaderBufferTableEntry entry;
          entry.gpu_addr=0x200000u+index*0x1000u; entry.size=16; entry.stride=4;
          entry.vsharp={static_cast<uint32_t>(entry.gpu_addr),
                        static_cast<uint32_t>(entry.gpu_addr>>32u)|(4u<<16u),
                        4u,(20u<<12u)|0xfacu};
          vb.table_entries.push_back(entry);
      }
      rt.resources.push_back(vb);
      ComputeShaderConfig config; config.user_sgprs.resize(12);
      config.local_x=config.local_y=config.local_z=1;
      dump(dir, "compute_fetch_descriptor_array",
           recompile_compute(c, std::size(c), &rt, config)); }
    // Fragment: image_sample a texture (T# in s[8:15]).
    { const uint32_t c[] = {0x7e0002ffu,0x3e800000u,0x7e0202ffu,0x3e800000u,0xf0800f08u,0x00820000u,0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Float32;
      t.num_components=4; t.binding=4; t.img_dim=1; t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_texture", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: Astro Bot R32_UINT image_atomic_swap with GLC return.
    { const uint32_t c[] = {0x7e000280u,0x7e020280u,0x7e1202ffu,0x3f800000u,
                            0xf03c2108u,0x00000900u,0x7e000280u,0x7e020309u,
                            0x7e040280u,0x7e0602f2u,0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource image{}; image.cls=ResourceClass::StorageImage;
      image.format=DataFormat::Uint32; image.num_components=1; image.binding=4;
      image.img_dim=1; image.width=1; image.height=1; image.depth=1; image.sgpr_base=0;
      rt.resources.push_back(image);
      dump(dir, "fragment_image_atomic", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_sample a 3D texture (3 coords) -> mrt0. (shader_028 pattern)
    { const uint32_t c[] = {0x7E0002FFu,0x3E800000u,0x7E0202FFu,0x3E800000u,0x7E0402FFu,0x3E800000u,
                            0xF0800F10u,0x00400000u,0xF800180Fu,0x03020100u,0xBF810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Float32;
      t.num_components=4; t.binding=4; t.img_dim=2; t.width=2; t.height=2; t.sgpr_base=0; rt.resources.push_back(t);
      dump(dir, "fragment_texture_3d", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: COMPR export — v_cvt_pkrtz packs f16x2, exp ... done compr unpacks to vec4. (shader_029)
    { const uint32_t c[] = {0x7E0002F0u,0x7E0202F0u,0x5E000300u,0x5E020300u,0xF8001C0Fu,0x00000100u,0xBF810000u};
      dump(dir, "fragment_compr_export", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Vertex with PARAM export (interpolated varying out).
    { const uint32_t c[] = {0x7e140d00u,0x36020081u,0x2c040081u,0x7e020d01u,0x7e040d02u,0x100202f6u,0x100404f6u,
                            0x060202f3u,0x060404f3u,0x7e060280u,0x7e0802f2u,0xf80008cfu,0x04030201u,0xf800020fu,0x0403030au,0xbf810000u};
      dump(dir, "vertex_param", recompile_vertex(c, sizeof(c)/4)); }
    // Fragment with v_interp (interpolated varying in).
    { const uint32_t c[] = {0xc8000000u,0xc8010001u,0x7e020280u,0x7e0402f2u,0xf800080fu,0x02010100u,0xbf810000u};
      dump(dir, "fragment_interp", recompile_fragment(c, sizeof(c)/4)); }
    // Compute LDS (ds_write / s_barrier / ds_read).
    { const uint32_t c[] = {0x7e020f00u,0x34040282u,0x34060281u,0x4a060681u,0xd8340000u,0x00000302u,0xbf8a0000u,
                            0x4c0a02bfu,0x340c0a82u,0xd8d80000u,0x07000006u,0x7e000d07u,0xbf810000u};
      dump(dir, "compute_lds", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // GTA V exec_cs_413ced900 pc69: exact DS_MIN_F32 fields. Strictly validate the core-SPIR-V
    // compare-exchange loop used in place of an unavailable portable float atomic min.
    { const uint32_t c[] = {0xbefe0481u, 0x7e180280u,
                            0xd9340010u, 0x0000040cu,
                            0xdb7c0000u, 0x0000000cu,
                            0xbefe04c1u, 0x7e000280u,
                            0xd8480000u, 0x00000900u,
                            0xd8480004u, 0x00000a00u,
                            0xd8480008u, 0x00000b00u,
                            0xd84c000cu, 0x00000600u,
                            0xd84c0010u, 0x00000700u,
                            0xd84c0014u, 0x00000800u,
                            0xbefe0481u,                         // pc81 exec = lane zero
                            0x7e080280u,                         // pc82 v4 = byte address zero
                            0xdbfc0000u, 0x00000004u,            // pc83 ds_read_b128 v[0:3], v4
                            0xd9d80010u, 0x04000004u,            // pc85 ds_read_b64 v[4:5], v4
                            0xbf8cc17fu, 0xbf810000u};           // wait; end (no guest barrier)
      dump(dir, "compute_ds_min_f32", recompile_valu(c, std::size(c), 2, 0)); }
    // Compute MUBUF store (buffer_store_format_x).
    { const uint32_t c[] = {0x7e040f00u,0x06060100u,0xe0102000u,0x80020302u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=1; vb.binding=3; vb.stride=4; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "compute_store", recompile_valu(c, sizeof(c)/4, 1, 0, &rt)); }
    // Compute MUBUF PACKED-WORD store (#3575). The module above stores a raw dword and therefore
    // exercises none of this: pack_ufloat alone emits a nest of OpSelect over exponent/mantissa
    // extraction and a round-to-even shift, which is exactly the shape strict validation catches and
    // llvmpipe accepts. Two arms, because the float and the normalized halves reach different
    // converters (pack_ufloat vs pack_norm) and only the format differs between them.
    { const uint32_t c[] = {0x7e140f00u, 0x7e020280u, 0x7e0402ffu, 0x3e800000u, 0x7e0602f0u,
                            0x7e0802f2u, 0xe01c2000u, 0x8002010au, 0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer;
      vb.format=DataFormat::Float10_11_11; vb.num_components=3; vb.binding=3; vb.stride=4;
      vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "compute_store_packed_10_11_11", recompile_valu(c, sizeof(c)/4, 1, 0, &rt));
      ShaderResourceTable rtn; ShaderResource vn = vb; vn.format=DataFormat::Unorm2_10_10_10;
      vn.num_components=4; rtn.resources.push_back(vn);
      dump(dir, "compute_store_packed_2_10_10_10", recompile_valu(c, sizeof(c)/4, 1, 0, &rtn)); }
    // GTA V's cross-workgroup scan publication shape: distinct descriptor variables alias one guest
    // allocation, a GLC store is released by vscnt(0), and a GLC+DLC load polls through the alias.
    // This representative module strictly validates Aliased/Coherent decorations, per-access
    // Volatile operands, and Device-scope UniformMemory release under SPIR-V 1.3 + Vulkan 1.1.
    { const uint32_t c[] = {0xe0744008u,0x80001100u,0xbf8c3f70u,0xbbfd0000u,
                            0xe0706010u,0x80001211u,0xe030e010u,0x80001e1du,
                            0xbf810000u};
      ShaderResourceTable rt;
      ShaderResource publish{}; publish.cls=ResourceClass::ConstantBuffer;
      publish.format=DataFormat::Uint32; publish.binding=4; publish.gpu_addr=0x1000;
      publish.size=180; publish.stride=20; publish.fetch_pc=0; rt.resources.push_back(publish);
      ShaderResource flag=publish; flag.binding=5; flag.fetch_pc=4; rt.resources.push_back(flag);
      ShaderResource poll=publish; poll.binding=6; poll.fetch_pc=6; rt.resources.push_back(poll);
      ComputeShaderConfig cfg; cfg.local_x=64; cfg.wave_size=64;
      dump(dir, "compute_coherent_alias", recompile_compute(c, sizeof(c)/4, &rt, cfg)); }
    // GTA V's exact qword-atomic resource uses same-binding u32/u64 aliased Block variables. Validate
    // both RMW opcodes strictly: driver acceptance alone does not prove the duplicate binding, u64
    // AccessChain, capability, or atomic result type is legal Vulkan SPIR-V.
    { const uint32_t swap[] = {0xe0302000u,0x80000013u,
                               0xe1402000u,0x80000913u,0xbf810000u};
      const uint32_t bit_or[] = {0xe0302000u,0x80000013u,
                                 0xe1686000u,0x80000913u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource atomic{};
      atomic.cls=ResourceClass::ConstantBuffer; atomic.format=DataFormat::Uint32;
      atomic.num_components=1; atomic.binding=2; atomic.gpu_addr=0x2000;
      atomic.size=200; atomic.stride=8; atomic.sgpr_base=0; atomic.fetch_pc=2;
      atomic.atomic_x2_record_count=25; rt.resources.push_back(atomic);
      ComputeShaderConfig cfg; cfg.local_x=1; cfg.storage_buffer_int64_atomics=true;
      dump(dir, "compute_atomic_swap_x2",
           recompile_compute(swap, std::size(swap), &rt, cfg));
      dump(dir, "compute_atomic_or_x2",
           recompile_compute(bit_or, std::size(bit_or), &rt, cfg)); }
    // Astro Bot exact raw buffer_store_dwordx3 packet.
    { const uint32_t c[] = {0x7e140f00u,0x7e060281u,0x7e080282u,0x7e0a0283u,
                            0xe07c2000u,0x8004030au,0xbf810000u};
      ShaderResourceTable rt; ShaderResource dst{}; dst.cls=ResourceClass::ConstantBuffer;
      dst.format=DataFormat::Uint32; dst.num_components=1; dst.binding=3;
      dst.stride=12; dst.sgpr_base=16; rt.resources.push_back(dst);
      dump(dir, "compute_store_x3", recompile_valu(c, sizeof(c)/4, 1, 0, &rt)); }
    // Astro Bot's exact RTIP 1.1 BVH instruction. The software intersection path is intentionally
    // scalar/SSBO SPIR-V so it does not require Vulkan ray-query capabilities.
    { const uint32_t c[] = {0xf1989f07u,0x00040303u,0x43440d3fu,0x46424140u,0x00004847u,
                            0xbf810000u};
      ShaderResourceTable rt; ShaderResource bvh{}; bvh.cls=ResourceClass::ConstantBuffer;
      bvh.format=DataFormat::Uint32; bvh.num_components=1; bvh.binding=4;
      bvh.size=128; bvh.fetch_pc=0; rt.resources.push_back(bvh);
      ComputeShaderConfig cfg; cfg.local_x=1;
      dump(dir, "compute_bvh_intersect", recompile_compute(c, sizeof(c)/4, &rt, cfg),
           "recompile_compute"); }
    // GTA V's exact program 0x205b5e8600 pc1476 packet uses a 64-byte BVH. This separately covers
    // the constant-false 128-byte-node bound that triangle and FP16-box allocations require.
    { constexpr uint32_t pc = 1476u;
      std::vector<uint32_t> c(pc, 0xbf800000u); // s_nop 0
      const uint32_t tail[] = {0xf1989f07u,0x00060202u,0x28292c23u,0x22262725u,0x00002a24u,
                               0xbf810000u};
      c.insert(c.end(), tail, tail + sizeof(tail)/sizeof(tail[0]));
      ShaderResourceTable rt; ShaderResource bvh{}; bvh.cls=ResourceClass::ConstantBuffer;
      bvh.format=DataFormat::Uint32; bvh.num_components=1; bvh.binding=4;
      bvh.size=64; bvh.fetch_pc=pc; rt.resources.push_back(bvh);
      ComputeShaderConfig cfg; cfg.local_x=1;
      dump(dir, "compute_bvh_intersect_64", recompile_compute(c.data(), c.size(), &rt, cfg)); }
    // Compute EXEC-predicated store (v_cmpx + guard execz + store).
    { const uint32_t c[] = {0x7e040f00u,0x06060100u,0x7e0a0284u,0x7da20b02u,0xbf880002u,0xe0102000u,0x80020302u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer; vb.format=DataFormat::Float32;
      vb.num_components=1; vb.binding=3; vb.stride=4; vb.sgpr_base=8; rt.resources.push_back(vb);
      dump(dir, "compute_pred_store", recompile_valu(c, sizeof(c)/4, 1, 0, &rt)); }

    // Compute STORAGE images (image_load -> image_store, no sampler): 1D, NSA 3D (split-address coords),
    // and 2D_ARRAY (layer coord). One shared rt: src U# in user-data s[0:7]->binding 4, dst s[8:15]->5.
    { ShaderResourceTable rt;
      ShaderResource s{}; s.cls = ResourceClass::StorageImage; s.binding = 4; s.sgpr_base = 0; rt.resources.push_back(s);
      ShaderResource d{}; d.cls = ResourceClass::StorageImage; d.binding = 5; d.sgpr_base = 8; rt.resources.push_back(d);
      const uint32_t c1d[]  = {0x7E080300u,0xF0000F00u,0x00000004u,0xBF8C3F70u,0xF0200F00u,0x00020004u,0xBF810000u};
      dump(dir, "storage_copy_1d",    recompile_valu(c1d,  sizeof(c1d)/4,  1, 0, &rt));
      const uint32_t cnsa[] = {0xF0000F12u,0x0000000Au,0x00000C0Bu,0xBF8C3F70u,0xF0200F12u,0x0002000Au,0x00000C0Bu,0xBF810000u};
      dump(dir, "storage_nsa_3d",     recompile_valu(cnsa, sizeof(cnsa)/4, 1, 0, &rt));
      const uint32_t carr[] = {0xF0000F28u,0x00000004u,0xBF8C3F70u,0xF0200F28u,0x00020004u,0xBF810000u};
      dump(dir, "storage_arrayed_2d", recompile_valu(carr, sizeof(carr)/4, 1, 0, &rt));
      const uint32_t cms[]  = {0xF0000F30u,0x00000000u,0xBF8C3F70u,0x7E000D00u,0xBF810000u};   // image_load 2D_MSAA (x,y,sample)
      dump(dir, "storage_msaa_2d",    recompile_valu(cms,  sizeof(cms)/4,  1, 0, &rt));
      // image_load 2D_MSAA_ARRAY (dim 7): coords (x,y,layer) + sample — needs Arrayed+MS + ImageMSArray cap.
      const uint32_t cmsa[] = {0xF0000F3Au,0x00000000u,0x00030201u,0xBF8C3F70u,0xBF810000u};
      dump(dir, "storage_msaa_array_2d", recompile_valu(cmsa, sizeof(cmsa)/4, 1, 0, &rt)); }
    // GTA V's audited one-level IMAGE_LOAD_MIP / IMAGE_STORE_MIP subset. Validate both sampled
    // image shapes (including the retained array layer) and the NSA storage write strictly.
    { ShaderResourceTable rt;
      ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Uint32;
      t.num_components=1; t.binding=4; t.fetch_pc=1; t.img_dim=1;
      t.width=4; t.height=4; t.depth=1; t.size=64; t.proven_zero_mip=true;
      rt.resources.push_back(t);
      const uint32_t c[] = {0x7e040207u,0xf0043f08u,0x00050000u,0xbf810000u};
      dump(dir, "compute_load_mip_zero_2d", recompile_valu(c, sizeof(c)/4, 1, 0, &rt));
      rt.resources[0].img_dim=5; rt.resources[0].depth=2; rt.resources[0].size=128;
      const uint32_t ca[] = {0x7e060206u,0xf0043128u,0x00050000u,0xbf810000u};
      dump(dir, "compute_load_mip_zero_array", recompile_valu(ca, sizeof(ca)/4, 1, 0, &rt)); }
    { ShaderResourceTable rt;
      ShaderResource s{}; s.cls=ResourceClass::StorageImage; s.format=DataFormat::Uint32;
      s.num_components=1; s.binding=4; s.fetch_pc=1; s.img_dim=1;
      s.width=4; s.height=4; s.depth=1; s.size=64; s.proven_zero_mip=true;
      rt.resources.push_back(s);
      const uint32_t c[] = {
          0x7e0a0206u,0xf024310au,0x00030004u,0x00000503u,0xbf810000u};
      dump(dir, "compute_store_mip_zero_2d", recompile_valu(c, sizeof(c)/4, 1, 0, &rt));
      rt.resources[0].format=DataFormat::Uint8; rt.resources[0].num_components=4;
      const uint32_t cf[] = {
          0x7e0c0206u,0xf0243f0au,0x00030005u,0x00000604u,0xbf810000u};
      dump(dir, "compute_store_mip_zero_xyzw_2d",
           recompile_valu(cf, sizeof(cf)/4, 1, 0, &rt)); }
    // Compute that SAMPLES a texture and STORES to a storage image (the bloom/downsample shape, shader 006):
    // exercises the sampled-texture path inside a compute shell (needs vec4<float> declared there).
    { ShaderResourceTable rt;
      ShaderResource t{};  t.cls=ResourceClass::Texture;      t.binding=0; t.sgpr_base=0;  t.img_dim=1; rt.resources.push_back(t);
      ShaderResource s{};  s.cls=ResourceClass::StorageImage; s.binding=1; s.sgpr_base=16; rt.resources.push_back(s);
      const uint32_t c[] = {0x7E0002F0u,0x7E0202F0u,0xF09C0F08u,0x00400400u,0xBF8C3F70u,0xF0200108u,0x00040402u,0xBF810000u};
      dump(dir, "compute_sample_store", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // Compute image_get_resinfo on a sampled 3D image (DOLL's volume-initializer bounds query).
    { ShaderResourceTable rt;
      ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.sgpr_base=0; t.img_dim=2; rt.resources.push_back(t);
      const uint32_t c[] = {0x7E060280u,0xF0380710u,0x00000003u,0xBF810000u};
      dump(dir, "compute_resinfo_3d", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // Compute image_get_resinfo on a SAMPLED 1D image. The storage 1D case above declares
    // Sampled1D; the sampled path did not, and nothing here exercised it -- spirv-val:
    // "Operand 3 of TypeImage requires one of these capabilities: Sampled1D". Same encoding as
    // compute_resinfo_3d with MIMG DIM (word0 bits [5:3]) set to 0 = 1D, and img_dim 0 = 1D.
    // This is Sonic Frontiers' query-only binding shape (#657, #2790).
    { ShaderResourceTable rt;
      ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.sgpr_base=0; t.img_dim=0; rt.resources.push_back(t);
      const uint32_t c[] = {0x7E060280u,0xF0380700u,0x00000003u,0xBF810000u};
      dump(dir, "compute_resinfo_1d", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // Compute integer image_load from a UINT8x4 3D texture. The sampled image's scalar type and
    // OpImageFetch result must be uint, matching the explicit v_cvt_f32_ubyte* sequence used by
    // UE4's volumetric-lightmap indirection volume (DOLL producer pc 816).
    { ShaderResourceTable rt;
      ShaderResource t{}; t.cls=ResourceClass::Texture; t.format=DataFormat::Uint8;
      t.num_components=4; t.binding=4; t.sgpr_base=0; t.img_dim=2;
      t.width=32; t.height=32; t.depth=32; rt.resources.push_back(t);
      const uint32_t c[] = {0x7E1E0280u,0x7E200280u,0x7E220280u,
                            0xF0000F10u,0x0000000Fu,0xBF8C3F70u,
                            0x7E000D00u,0x7E020D01u,0x7E040D02u,0x7E060D03u,
                            0xBF810000u};
      dump(dir, "compute_uint_load_3d", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // Compute mul_hi (high 32 bits via OpUMulExtended -> {lo,hi} struct extract).
    { const uint32_t c[] = {0x7E0202FFu,0x80000000u,0xD56A0002u,0x00020301u,0x7E000D02u,0xBF810000u};
      dump(dir, "compute_mul_hi", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Compute: saved VOPC mask combined with VCC by s_nor_b64 (UE4 visibility-mask shape).
    { const uint32_t c[] = {0x7C040CF9u,0x06869880u,0x8DEA6A18u,0xBF810000u};
      dump(dir, "compute_mask_nor", recompile_valu(c, sizeof(c)/4, 0, 0)); }
    // Compute: nested/multi-branch CFG dispatcher (varying VCC exit + inner SCC branch + back-edge).
    // Locks both structured switch-loop formation and the workgroup-scratch hardware-wave vote.
    { const uint32_t c[] = {0xBE800380u,0x7E000280u,0x7E020300u,
                            0xD7610013u,0x00014A7Eu,0xD7610013u,0x0001507Fu,
                            0xD760000Eu,0x00014B13u,0xD760000Fu,0x00015113u,0xBEFE040Eu,
                            0xE00C2000u,0x80020400u,0x7DB900F9u,0x86050007u,
                            0x7D020200u,0xBF860006u,0xBF0A8204u,0x360000FDu,0xBF840001u,
                            0x81008100u,0x81008100u,0xBF82FFF4u,
                            0xBF810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer;
      vb.binding=3; vb.sgpr_base=8; vb.stride=16; vb.format=DataFormat::Float32;
      vb.num_components=4; rt.resources.push_back(vb);
      dump(dir, "compute_cfg_dispatch", recompile_valu(c, sizeof(c)/4, 0, 0, &rt)); }
    // GTA V Wave64 survivor-mask join: one arm retains scalar EXEC words while the other computes
    // the same physical pair through S_ANDN2_B64. Validate the native subgroup ballots that make
    // the logical result scalar-readable at the exact trailing S_CMP_EQ_U64.
    { const uint32_t c[] = {
          0xBEB8037Eu,0xBEB9037Fu,0x7E400280u,0xBF068008u,0xBF840003u,
          0x7D8A40C1u,0x8AB86A38u,0xBF800000u,0xBF128038u,
          0xBE800380u,0x7E000280u,0x7E020300u,
          0xD7610013u,0x00014A7Eu,0xD7610013u,0x0001507Fu,
          0xD760000Eu,0x00014B13u,0xD760000Fu,0x00015113u,0xBEFE040Eu,
          0xE00C2000u,0x80020400u,0x7DB900F9u,0x86050007u,
          0x7D020200u,0xBF860006u,0xBF0A8204u,0x360000FDu,0xBF840001u,
          0x81008100u,0x81008100u,0xBF82FFF4u,0xBF810000u};
      ShaderResourceTable rt; ShaderResource vb{}; vb.cls=ResourceClass::VertexBuffer;
      vb.binding=3; vb.sgpr_base=8; vb.stride=16; vb.format=DataFormat::Float32;
      vb.num_components=4; rt.resources.push_back(vb);
      ComputeShaderConfig cfg; cfg.local_x=64; cfg.wave_size=64;
      cfg.native_subgroup_size=64;
      dump(dir, "compute_wave64_logical_ballot",
           recompile_compute(c, sizeof(c)/4, &rt, cfg)); }
    // Ordinary e64 integer-pair comparisons (unsigned then signed), separate from mask provenance.
    { const uint32_t c[] = {0xd4e4006au,0x00010000u,0xd4a4006au,0x00010000u,
                            0xbf810000u};
      ComputeShaderConfig cfg; cfg.local_x=128; cfg.wave_size=64;
      dump(dir, "compute_i64_compare",
           recompile_compute(c, sizeof(c)/4, nullptr, cfg)); }
    // Astro Bot Wave64: e64 mask-vs-zero normalization followed by a last-live B64 mask SCC vote.
    // Both reductions use the portable dispatcher's uniform common phase.
    { const uint32_t c[] = {0x7d840000u,0xd4e4006au,0x0001006au,0xbea0047eu,
                            0x87ea6a20u,0x7e000280u,0xbf840003u,0x7d840100u,
                            0x02020100u,0xbf860002u,0xbf060000u,0xbf850001u,
                            0x7e020280u,0xbf810000u};
      ComputeShaderConfig cfg; cfg.local_x=128; cfg.wave_size=64;
      dump(dir, "compute_wave64_mask_vote",
           recompile_compute(c, sizeof(c)/4, nullptr, cfg)); }

    // --- #273 additions (DOLL recompiler frontier) ---
    // Fragment: 3D image_load (integer LUT fetch through the combined sampler; OpImage+OpImageFetch).
    { const uint32_t c[] = {0x7e000280u,0x7e020280u,0x7e040280u,0xf0001f10u,0x00020000u,
                            0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=2;
      t.width=16; t.height=16; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_load_3d", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_sample_lz_o (packed texel offset folded into normalized coords; ImageQuery).
    { const uint32_t c[] = {0x7e0002ffu,0x00000101u,0x7e0202f0u,0x7e0402f0u,0xf0dc0f08u,0x00820000u,
                            0xf800000fu,0x03020100u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=1;
      t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_sample_lz_o", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Fragment: image_gather4_lz_o (dynamic gather offset; ImageGatherExtended — locks the operand-ID fix).
    { const uint32_t c[] = {0x7e0002ffu,0x00000101u,0x7e0202f0u,0x7e0402f0u,0xf15c0808u,0x00820400u,
                            0xf800000fu,0x07060504u,0xbf810000u};
      ShaderResourceTable rt; ShaderResource t{}; t.cls=ResourceClass::Texture; t.binding=4; t.img_dim=1;
      t.width=2; t.height=2; t.sgpr_base=8; rt.resources.push_back(t);
      dump(dir, "fragment_gather4_lz_o", recompile_fragment(c, sizeof(c)/4, &rt)); }
    // Compute: if/else-if/else cascade with common-merge s_branch arms (kernel T17's stream).
    { const uint32_t c[] = {0x7e020f00u,0x7e080501u,0xbf0a8204u,0xbf840002u,0x4a02028au,0xbf820005u,
                            0xbf0a8504u,0xbf840002u,0x4a020294u,0xbf820001u,0x4a02029eu,0x7e000d01u,0xbf810000u};
      dump(dir, "compute_cascade_ifelse", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Compute: readfirstlane waterfall + v_movrels (kernel T19's stream).
    { const uint32_t c[] = {0x7e020f00u,0xbe86047eu,0x7e0402ffu,0x40a00000u,0x7e0602ffu,0x40e00000u,
                            0x7e0802ffu,0x41100000u,0x7e080501u,0x7da40204u,0xbefc0304u,0x7e0a8702u,
                            0x8a867e06u,0xbefe0406u,0xbf85fff9u,0xbefe04c1u,0x7e000305u,0xbf810000u};
      dump(dir, "compute_waterfall_movrels", recompile_valu(c, sizeof(c)/4, 1, 0)); }
    // Fragment: divergent execz region (v_cmpx + s_cbranch_execz over a scalar-writing block).
    { const uint32_t c[] = {0x7e020280u,0x7e0002f2u,0x7c2200f0u,0xbf880002u,0xbe8503f2u,0x7e020205u,
                            0x7e040205u,0xf800180fu,0x01010101u,0xbf810000u};
      dump(dir, "fragment_execz_if", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // A saved lane mask first defined inside a forward if and consumed after its merge. The skipped
    // edge contributes false; without that phi the arm-local definition does not dominate its use.
    { const uint32_t c[] = {0x7E020280u,0x7E0002F2u,0x7C2200F0u,0xBF880001u,0xBE82047Eu,
                            0xBEFE0402u,0x7E0202F2u,0xF800180Fu,0x01010101u,0xBF810000u};
      dump(dir, "fragment_if_new_mask", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Fragment: DIVERGENT execz-exit loop with a nested execz if in the body (#273 — the DOLL
    // title post-process accumulation shape; test_recompiled_fragment executes it).
    { const uint32_t c[] = {0x7E020284u,0xBE800380u,0x7E040280u,0x7E060280u,0x7E080282u,0x7E0A02F2u,
                            0xBE82047Eu,0x7DA20200u,0xBF88000Au,0xBE84047Eu,0x7DA20800u,0xBF880002u,
                            0x060404FFu,0x3E800000u,0xBEFE0404u,0x060606FFu,0x3E800000u,0x81008100u,
                            0xBF82FFF4u,0xBEFE0402u,0xF800180Fu,0x05020302u,0xBF810000u};
      dump(dir, "fragment_divergent_loop", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Fragment: uniform VCCZ-exit light-accumulation loop (#615, Dead Cells). The compare's VGPR
    // bound is moved from an inline uniform value, making the wave-empty VCC branch structurizable.
    { const uint32_t c[] = {0xBE800380u,0x7E000280u,0x7E020284u,0x7E0602F2u,0x7D020200u,
                            0xBF860004u,0x060000FFu,0x3E800000u,0x81008100u,0xBF82FFFAu,
                            0x7E020300u,0x7E040300u,0xF800080Fu,0x03020100u,0xBF810000u};
      dump(dir, "fragment_uniform_vcc_loop", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Fragment: the same loop with VCC a live mask BEFORE it and a body that recycles vcc_lo/vcc_hi
    // as scalar scratch (#4508, Space Adventure Cobra's per-light loop). The VCC header phi closes
    // with a placeholder on the back-edge; test_fragment_loop_vcc_scratch executes the uniform form.
    // The second module takes its bound from a lane-varying register, so the exit is a wave vote.
    {
        const uint32_t c[] = {0xBE800380u, 0x7E000280u, 0x7E020284u, 0x7E0602F2u, 0x7D020200u,
                              0x7D020200u, 0xBF860006u, 0x816A8100u, 0x876B8300u, 0x060000FFu,
                              0x3E800000u, 0xBE80036Au, 0xBF82FFF8u, 0x7E020300u, 0x7E040300u,
                              0xF800080Fu, 0x03020100u, 0xBF810000u};
        dump(dir, "fragment_vcc_scratch_loop", recompile_fragment(c, sizeof(c) / 4, nullptr));
        uint32_t voted[sizeof(c) / 4];
        std::copy(std::begin(c), std::end(c), voted);
        voted[2] = 0x7E020302u;
        dump(dir, "fragment_vcc_scratch_loop_voted",
             recompile_fragment(voted, sizeof(voted) / 4, nullptr));
    }
    // Fragment: EXECNZ-back-edge loop with a mid-body vccz break (#273 — the scalar-indexed unroll).
    { const uint32_t c[] = {0xBE82047Eu,0xBE800380u,0xBE810383u,0x7E040280u,0x7E0A02F2u,0xBF880009u,
                            0xBF0A0100u,0x8584807Eu,0xBEEA0404u,0xBEFE0404u,0xBF860004u,0x060404FFu,
                            0x3E800000u,0x81008100u,0xBF89FFF6u,0xBEFE0402u,0xF800180Fu,0x05020202u,
                            0xBF810000u};
      dump(dir, "fragment_execnz_loop", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // Fragment: v_writelane/v_readlane scalar-spill slots (#273).
    { const uint32_t c[] = {0xBE8503FFu,0x3E800000u,0xBE8603F2u,0xD761000Au,0x00010605u,0xD761000Au,
                            0x00010E06u,0xD7600007u,0x0001070Au,0xD7600008u,0x00010F0Au,0x7E000207u,
                            0x7E020208u,0xF800180Fu,0x01000100u,0xBF810000u};
      dump(dir, "fragment_lane_slots", recompile_fragment(c, sizeof(c)/4, nullptr)); }
    // NESTED divergent execz-exit loops (#590/#1067 — DOLL's last post-process kernel shape): an
    // inner table loop entirely inside an outer row loop, execz exits + backward s_branch
    // back-edges, an exec save around the inner loop, post-loop s_barrier (compute). Locks the
    // nested OpLoopMerge emission under strict validation for BOTH shells; the execution twins
    // live in test_rdna2_to_spirv / test_recompiled_fragment.
    { const uint32_t c[] = {0x7E020283u,0x7E080284u,0xBE800380u,0x7E040280u,0x7E060280u,0x7E0A02F2u,
                            0xBE82047Eu,0x7DA20200u,0xBF88000Du,0xBE810380u,0xBE84047Eu,0x7DA20801u,
                            0xBF880004u,0x060606FFu,0x3D800000u,0x81018101u,0xBF82FFFAu,0xBEFE0404u,
                            0x060404FFu,0x3E800000u,0x81008100u,0xBF82FFF1u,0xBEFE0402u,0xBF8A0000u,
                            0xBF810000u};
      dump(dir, "compute_nested_exec_loops", recompile_valu(c, sizeof(c)/4, 1, 3)); }
    { const uint32_t c[] = {0x7E020283u,0x7E080284u,0xBE800380u,0x7E040280u,0x7E060280u,0x7E0A02F2u,
                            0xBE82047Eu,0x7DA20200u,0xBF88000Du,0xBE810380u,0xBE84047Eu,0x7DA20801u,
                            0xBF880004u,0x060606FFu,0x3D800000u,0x81018101u,0xBF82FFFAu,0xBEFE0404u,
                            0x060404FFu,0x3E800000u,0x81008100u,0xBF82FFF1u,0xBEFE0402u,
                            0xF800180Fu,0x05020302u,0xBF810000u};
      dump(dir, "fragment_nested_exec_loops", recompile_fragment(c, sizeof(c)/4, nullptr)); }

    // --- Emitters that are NOT the RDNA2 recompiler ---
    // spirv_builder.cpp hand-assembles compute modules that the live frontend creates at runtime
    // (frontends/shared/live/live_compute.cpp: prepare_compare_pipeline / the scale-bias probe), so they
    // reach vkCreateShaderModule exactly like a recompiled shader. #1711 lived here.
    dump(dir, "builder_scale_bias", build_compute_scale_bias(2.0f, 0.5f),
         "build_compute_scale_bias");
    dump(dir, "builder_compare_uvec4", build_compute_compare_uvec4(),
         "build_compute_compare_uvec4");
    dump(dir, "builder_retile_compare", build_compute_retile_words(RetileShaderKind::Words2D,
         RetileImageSource::LinearBuffer, false, true), "build_compute_retile_words");
    dump(dir, "builder_detile_rgba16f", build_compute_detile_float16(),
         "build_compute_detile_float16");
    dump(dir, "builder_detile_rg16f", build_compute_detile_float16(2),
         "build_compute_detile_float16");
    dump(dir, "builder_detile_raw_rg16f", build_compute_detile_float16(2, Float16DetileOutput::RawUvec4),
         "build_compute_detile_float16");
    dump(dir, "builder_detile_raw_rgba16f", build_compute_detile_float16(4, Float16DetileOutput::RawUvec4),
         "build_compute_detile_float16");
    dump(dir, "builder_retile_words", build_compute_retile_words(),
         "build_compute_retile_words");
    dump(dir, "builder_retile_packed_subword_array", build_compute_retile_words(RetileShaderKind::PackedSubwordArray),
         "build_compute_retile_words");
    dump(dir, "builder_retile_volume_words", build_compute_retile_words(RetileShaderKind::Volume3D),
         "build_compute_retile_words");
    for (auto source : {RetileImageSource::R32Uint, RetileImageSource::Rgba8Uint})
        for (bool arrayed : {false, true}) {
            const std::string name = std::string("builder_retile_image_") +
                (source == RetileImageSource::R32Uint ? "r32" : "rgba8") + (arrayed ? "_array" : "_2d");
            dump(dir, name.c_str(), build_compute_retile_words(RetileShaderKind::Words2D, source, arrayed),
                 "build_compute_retile_words");
            dump(dir, (name + "_compare").c_str(),
                 build_compute_retile_words(RetileShaderKind::Words2D, source, arrayed, true),
                 "build_compute_retile_words");
        }
    dump(dir, "builder_rgba8_to_packed10", build_compute_rgba8_to_packed10(),
         "build_compute_rgba8_to_packed10");
    dump(dir, "builder_depth_to_rgba8", build_compute_depth_to_rgba8(),
         "build_compute_depth_to_rgba8");
    dump(dir, "builder_indirect_dispatch_validate", build_compute_indirect_dispatch_validate(),
         "build_compute_indirect_dispatch_validate");   // #3656

    // --- Recompiler entry points beyond the four main stage functions ---
    // Wave32 fragment lowering (s_wqm_b32 through the low-half EXEC/VCC mask path).
    { const uint32_t c[] = {0xbe80037eu,0xbefe0900u,0xbefe097eu,
                            0x7e000280u,0x7e020280u,0x7e040280u,0x7e0602f2u,
                            0xf800000fu,0x03020100u,0xbf810000u};
      dump(dir, "fragment_wave32_wqm", recompile_fragment_wave32_for_test(c, sizeof(c)/4),
           "recompile_fragment_wave32_for_test"); }
    // Terminal NGG output gate: CMPX + EXECZ around the POS export.
    { const uint32_t c[] = {0xBF900009u,0x34040A81u,0x36060AC2u,0x7E000280u,0x7E0202F2u,
                            0x36040482u,0x4A0606C1u,0x4A0404C1u,0x7E060B03u,0x7E040B02u,
                            0x7E280281u,0x7E2A0280u,0x7C3E2B14u,0xBF880002u,
                            0xF80008CFu,0x01000302u,0xBF810000u};
      dump(dir, "vertex_ngg_terminal_gate",
           recompile_vertex_terminal_ngg_gate_for_test(c, sizeof(c)/4),
           "recompile_vertex_terminal_ngg_gate_for_test"); }
    // One-lane NGG projection: a B64 mask scan reduced against the single live guest lane.
    { const uint32_t c[] = {0xBF900009u,0xBEEA04C1u,0xBE80146Au,0x7E000C00u,
                            0x7E020280u,0x7E040280u,0x7E0602F2u,
                            0xF80008CFu,0x03020100u,0xBF810000u};
      dump(dir, "vertex_ngg_one_lane",
           recompile_vertex_ngg_one_lane_for_test(c, sizeof(c)/4),
           "recompile_vertex_ngg_one_lane_for_test"); }
    // Split no-GS NGG program: a separately installed producer plus its terminal wrapper, recompiled
    // as one register-preserving module (the chain path the loader installs for real Astro draws).
    { const uint32_t producer[] = {
          0xD765000Au,0x000100C1u,0xD766000Au,0x000214C1u,
          0x34040A81u,0x36060AC2u,0x7E000280u,0x7E0202F2u,
          0x36040482u,0x4A0606C1u,0x4A0404C1u,0x7E060B03u,0x7E040B02u,
          0x7E0C0280u,0x7E0E0280u,0x7E1002F2u,
          0xD8380100u,0x00080706u,0xD8380302u,0x00030206u,0xD8380504u,0x00010006u,
          0x7E12030Au,0xD8340018u,0x00000906u,0xBF8A0000u,0xBE802006u};
      const uint32_t wrapper[] = {
          0xBF900009u,0xF8000941u,0x00000000u,
          0xD5430000u,0x03FE249Cu,0x00000728u,0xD5430002u,0x03FE249Cu,0x00000730u,
          0xD8DC0100u,0x00000000u,0xD8DC0100u,0x02000002u,0xF80000CFu,0x03020100u,
          0xD5430000u,0x03FE249Cu,0x00000720u,0xD8DC0100u,0x00000000u,
          0xF8000203u,0x00000100u,0x1608249Cu,0xD8D80738u,0x04000004u,
          0xF8000211u,0x00000004u,0xBF810000u};
      ShaderResourceTable rt; rt.vertices_per_instance = 3;
      dump(dir, "vertex_ngg_chain",
           recompile_vertex_chain(producer, sizeof(producer)/4, wrapper, sizeof(wrapper)/4,
                                  &rt, nullptr, false, 7),
           "recompile_vertex_chain"); }
    // Portable guest-wave execution with explicit primitive, position, layer and parameter exports.
    // The target is a compute test sink, not a rasterizing graphics module.
    { const uint32_t c[] = {
          0x7e000f00u,0x4a0200c0u,0xd8340000u,0x00000100u,0xbf8a0000u,
          0x3a040084u,0xd8d80000u,0x03000002u,0x7e0802c0u,0x7e0a0280u,
          0xbe80047eu,0x7da20900u,0xd7650005u,0x0001007eu,0xd7660005u,
          0x00020a7fu,0xbefe0400u,0x4a0c0b03u,0x7e000d06u,
          0xf8000941u,0x00000006u,0xf80000cfu,0x03020100u,
          0xf80008d4u,0x00050000u,0xf800020fu,0x00050603u,0xbf810000u};
      dump(dir, "ngg_workgroup_exports",
           recompile_ngg_exports_for_test(c, std::size(c), 1),
           "recompile_ngg_exports_for_test"); }
    dump_ngg_subgroup_shell(dir, src_root);
    { const uint32_t c[] = {
          0x7e000f00u,0x7e0202ffu,160u,
          0x7d8402f9u,0x06068600u, // save one VOPC lane in s[6:7]
          0xbe841006u,0x7e0c0204u, // portable whole-wave BCNT -> v6
          0xf8000941u,0x00000006u,0xbf810000u};
      dump(dir, "ngg_workgroup_saved_mask_count",
           recompile_ngg_exports_for_test(c, std::size(c), 1),
           "recompile_ngg_exports_for_test"); }
    { const uint32_t c[] = {
          0xbf8a0000u,0x7e080502u,0xbf8a0000u,0x7e0a0204u,
          0xf80000c1u,0x00000005u,0xbf810000u};
      dump(dir, "ngg_portable_wave64_readfirstlane",
           recompile_ngg_exports_for_test(c, std::size(c), 10, 0, nullptr,
                                           4, 0, {}, true, true),
           "recompile_ngg_exports_for_test"); }
    { const uint32_t c[] = {
          0x7e080500u,0xbf068003u,0xbf840004u,0xbf8a0000u,
          0x7e040204u,0xf80000c1u,0x00000002u,0xbf810000u};
      dump(dir, "ngg_portable_readfirstlane_guard_prefix",
           recompile_ngg_exports_for_test(c, std::size(c), 2, 0, nullptr,
                                           4, 0, {}, true),
           "recompile_ngg_exports_for_test"); }
    { const uint32_t c[] = {0x7ed40500u,0x7e02026au,0xbf810000u};
      dump(dir, "compute_partial_wave64_readfirstlane_vcc",
           recompile_valu(c, std::size(c), 2, 1, nullptr, 0,
                          kDefaultComputePgmRsrc1, true, 96, 80));
      dump(dir, "compute_single_lane_readfirstlane_vcc",
           recompile_valu(c, std::size(c), 2, 1, nullptr, 0,
                          kDefaultComputePgmRsrc1, true, 1, 1)); }
    { const uint32_t c[] = {
          0x7e000f00u,0x7e140300u,0xbf8a0000u,
          0x4a1614fau,0xff09110au,
          0xf8000941u,0x0000000bu,0xbf810000u};
      dump(dir, "ngg_workgroup_bounded_row_shr",
           recompile_ngg_exports_for_test(c, std::size(c), 1),
           "recompile_ngg_exports_for_test"); }
    // Unsigned row scans need both the ordinary uniform route and event-isolated CFG routes.
    {
        using prosper::test::DppRowFaddCase;
        uint32_t index = 0;
        for (const auto shape :
             {DppRowFaddCase::Linear, DppRowFaddCase::DivergentSites,
              DppRowFaddCase::LoopAndCompletedPeer, DppRowFaddCase::LaterBarrierPhase}) {
            const auto c = prosper::test::dpp_row_fadd_program(shape);
            for (bool native : {false, true}) {
                const std::string id = std::string("compute_dpp_row_fadd_") +
                                       std::to_string(index) + (native ? "_native64" : "_portable");
                dump(dir, id.c_str(),
                     recompile_ngg_exports_for_test(c.data(), c.size(), 10, 0, nullptr, 4, 0, {},
                                                    true, true, native),
                     "recompile_ngg_exports_for_test");
            }
            ++index;
        }
    }
    {
        const auto c = prosper::test::dpp_row_max_program({1, 2, 4, 8});
        dump(dir, "compute_dpp_row_max_linear", recompile_valu(c.data(), c.size(), 3, 1));
    }
    {
        using prosper::test::DppRowCfgCase;
        const std::pair<DppRowCfgCase, const char*> cases[] = {
            {DppRowCfgCase::Mixed, "mixed"},
            {DppRowCfgCase::DivergentSites, "sites"},
            {DppRowCfgCase::LoopAndCompletedPeer, "loop"},
            {DppRowCfgCase::LaterBarrierPhase, "later_phase"}};
        for (const auto& [shape, name] : cases) {
            const auto c = prosper::test::dpp_row_cfg_export_program(shape);
            for (bool native : {false, true}) {
                const std::string id = std::string("compute_dpp_row_max_") + name +
                                       (native ? "_native64" : "_portable");
                dump(dir, id.c_str(),
                     recompile_ngg_exports_for_test(c.data(), c.size(), 10, 0, nullptr, 4, 0, {},
                                                    true, true, native),
                     "recompile_ngg_exports_for_test");
            }
        }
    }
    // Generated interpolation geometry stage: AMD's explicit-parameter form publishes P0/P10/P20
    // plus perspective-center I/J from a synthesised Geometry entry point.
    { const uint32_t ps[] = {0xc80e0000u,0xc8120001u,0xc8160002u,
                             0xd54b0003u,0x04160103u,0xd54b0003u,0x040e0304u,
                             0x7e080280u,0x7e0a0280u,0x7e0c02f2u,
                             0xf800000fu,0x06050403u,0xbf810000u};
      PixelSystemInputMapping perspective_center{1u << 1, 1u << 1};
      const FragmentInterpolationLayout layout =
          fragment_interpolation_layout(ps, sizeof(ps)/4, &perspective_center);
      dump(dir, "geometry_interpolation", recompile_interpolation_geometry(layout),
           "recompile_interpolation_geometry");
      // The rect-synthesis variant is a DIFFERENT module -- extra types, a bvec4, and a dozen selects
      // the plain form never emits -- so validating only the form above validates none of it. That
      // gap is how an OpSelect with a vec4 result and a scalar condition (legal only from SPIR-V 1.4,
      // and this emitter stamps 1.3) reached CI in #3511; spirv-val rejects it outright. One emitter,
      // two shapes, so both are dumped.
      dump(dir, "geometry_interpolation_rect",
           recompile_interpolation_geometry(layout, /*capture_position=*/false,
                                            /*synthesize_rect=*/true),
           "recompile_interpolation_geometry");
      const FloatTransportConfig profile{FloatTransportProfile::ExplicitNonFinite32};
      dump(dir, "geometry_interpolation_primitive_id",
           recompile_interpolation_geometry(layout, false, false, profile, true),
           "recompile_interpolation_geometry");
      RasterQuadInputs inputs;
      inputs.raw_code = std::make_shared<const std::vector<uint32_t>>(ps, ps + std::size(ps));
      inputs.raw_matches_producing_source = true;
      inputs.system_inputs = perspective_center; inputs.has_system_inputs = true;
      inputs.launch.input_ena_available = inputs.launch.input_addr_available = true;
      inputs.launch.input_ena = perspective_center.ena;
      inputs.launch.input_addr = perspective_center.addr;
      inputs.pixel_inputs.valid_mask = 1; inputs.has_pixel_inputs = true;
      inputs.interpolation = layout; inputs.generated_interpolation_geometry = true;
      inputs.float_transport = profile;
      inputs.source_fs = std::make_shared<const std::vector<uint32_t>>(recompile_fragment(
          ps, std::size(ps), nullptr, &perspective_center, UINT32_MAX, &layout, false, {}, {}, nullptr, profile));
      RasterQuadCollector collector;
      dump(dir, "raster_quad_interpolation", build_raster_quad_collector(inputs, 2, collector),
           "build_raster_quad_collector"); }
    { const uint32_t ps[]{0xd7600014u,256u | (168u << 9),0x7e080214u,
                         0xf800180fu,0x04040404u,0xbf810000u};
      RasterQuadInputs inputs;
      inputs.raw_code = std::make_shared<const std::vector<uint32_t>>(ps, ps + std::size(ps));
      inputs.raw_matches_producing_source = true;
      inputs.system_inputs = {0x300u, 0x300u}; inputs.has_system_inputs = true;
      inputs.launch.input_ena_available = inputs.launch.input_addr_available = true;
      inputs.launch.input_ena = inputs.launch.input_addr = 0x300u;
      inputs.float_transport = {FloatTransportProfile::ExplicitNonFinite32};
      inputs.interpolation = fragment_interpolation_layout(ps, std::size(ps), &inputs.system_inputs);
      inputs.source_fs = std::make_shared<const std::vector<uint32_t>>(recompile_fragment(
          ps, std::size(ps), nullptr, &inputs.system_inputs, UINT32_MAX, &inputs.interpolation,
          false, {}, {}, nullptr, inputs.float_transport));
      RasterQuadCollector collector;
      dump(dir, "raster_quad_builtin", build_raster_quad_collector(inputs, 2, collector),
           "build_raster_quad_collector"); }

    // The owned compiler-case adapter is itself a SPIR-V-producing entry point. Exercise its
    // actual input rehydration and full-word baseline, not merely the underlying compiler.
    for (bool wave32 : {false, true}) {
      FragmentCompileCase c;
      c.compiler = std::string(prosper::embedded_build_revision()) + ":" +
                   prosper::embedded_build_source_identity();
      c.code = {0x7e000280u,0x7e0202f2u,0x7e040280u,0x7e0602f2u,
                0xf800180fu,0x03020100u,0xbf810000u};
      c.wave32 = wave32;
      c.interpolation = fragment_interpolation_layout(c.code.data(), c.code.size());
      std::vector<uint32_t> source;
      { CompilerChoiceScope reads(c.choices);
        source = recompile_fragment(c.code.data(), c.code.size(), nullptr, nullptr,
                                    UINT32_MAX, &c.interpolation, wave32); }
      finish_fragment_compile_case(c, std::move(source));
      try {
        dump(dir, wave32 ? "fragment_compile_case_wave32" : "fragment_compile_case_wave64",
             replay_fragment_compile_case(c, true), "replay_fragment_compile_case");
      } catch (const std::exception& error) {
        printf("  [FAIL] compiler-case baseline: %s\n", error.what());
        ++fails;
      }
    }

    fails += check_emitter_coverage(src_root);

    if (fails) { printf("== FAIL: %d emitter(s) failed emission/validation/coverage ==\n", fails); return 1; }
    printf("== PASS (every declared emitter accounted for; all modules pass spirv-val) ==\n");
    return 0;
}
