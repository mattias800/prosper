#include "gpu/capture/fragment_compile_case.hpp"
#include "build_revision.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include "diagnostics/perf/perf_ledger.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <map>
#include <stdexcept>

namespace prosper::gpu {
namespace {
void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
void incomplete(FragmentCompileCase& c, const char* reason) {
    c.complete = false;
    if (c.reason.empty()) c.reason = reason;
}
// These carriers have validators whose authority may depend on live guest reads, not just owned
// bytes. Schema 1 records their full metadata but does not substitute replay ownership as proof.
bool unsupported_carrier(const ShaderResource& r) {
    return r.indirect_buffer_contract_tag || r.indirect_buffer_binding_bytes ||
        r.indirect_buffer_slot_count || r.indirect_buffer_header_bytes ||
        r.indirect_buffer_slot_bytes ||
        r.indirect_pointer_relocation != IndirectPointerRelocationBinding{} ||
        r.selected_sbuffer_soffset != UINT32_MAX || r.atomic_x2_record_count;
}
CompileCaseBacking locate(FragmentCompileCase& c, const uint8_t* pointer,
                          uint64_t bytes, uint64_t prefix,
                          std::vector<CompileCaseBlob>& original,
                          std::map<uintptr_t, CompileCaseBacking>& opaque) {
    if (!pointer) {
        if (bytes || prefix) incomplete(c, "absent-backing-with-size");
        return {};
    }
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    for (uint32_t k = 0; k < original.size(); ++k) {
        const auto& blob = original[k];
        if (!blob.present || !blob.bytes || blob.bytes->empty()) continue;
        const auto begin = reinterpret_cast<uintptr_t>(blob.bytes->data());
        if (address < begin) continue;
        const uint64_t offset = address - begin;
        if (offset <= blob.bytes->size() && prefix <= offset &&
            bytes <= blob.bytes->size() - offset) return {k, offset};
    }
    {
        require(bytes <= UINT64_MAX - prefix, "compile-case opaque range overflow");
        auto found = opaque.find(address);
        if (found != opaque.end()) {
            require(found->second.offset == prefix, "compile-case opaque alias prefix unsupported");
            auto& blob = original[found->second.blob];
            blob.size = std::max(blob.size, prefix + bytes);
            return found->second;
        }
        require(original.size() < kCompileCaseMaxBlobs, "compile-case opaque count budget");
        CompileCaseBlob blob; blob.owner = CompileCaseOwner::ExternalOpaque;
        blob.opaque = true; blob.size = prefix + bytes;
        CompileCaseBacking ref{static_cast<uint32_t>(original.size()), prefix};
        original.push_back(std::move(blob)); opaque.emplace(address, ref); return ref;
    }
}
uint8_t* resolve(const FragmentCompileCase& c, CompileCaseBacking ref,
                 uint64_t bytes, uint64_t prefix) {
    if (ref.blob == CompileCaseBacking::kAbsent ||
        ref.blob == CompileCaseBacking::kUnavailable) return nullptr;
    require(ref.blob < c.blobs.size(), "compile-case backing index");
    const auto& blob = c.blobs[ref.blob];
    require(blob.present, "compile-case backing absent");
    require(ref.offset <= blob.size && prefix <= ref.offset &&
            bytes <= blob.size - ref.offset, "compile-case backing range");
    if (blob.opaque) return nullptr;
    require(bool(blob.bytes), "compile-case backing bytes absent");
    return ref.offset ? blob.bytes->data() + ref.offset : blob.bytes->data();
}
void valid_reference(const FragmentCompileCase& c, CompileCaseBacking ref,
                     uint64_t bytes, uint64_t prefix) {
    if (ref.blob >= CompileCaseBacking::kUnavailable) {
        require(ref.offset == 0, "compile-case absent backing offset");
        if (c.complete) require(ref.blob == CompileCaseBacking::kAbsent &&
                               bytes == 0 && prefix == 0, "compile-case unavailable backing");
    } else (void)resolve(c, ref, bytes, prefix);
}
} // namespace

void own_fragment_compile_case_resources(FragmentCompileCase& c, const ShaderResourceTable* table) {
    c.has_resources = table != nullptr;
    if (!table) return;
    // Check aggregate metadata before copying attacker-sized vectors. The caller can retain an
    // explicit INCOMPLETE case if this ownership transaction fails; no partial table is compiled.
    require(table->resources.size() <= kCompileCaseMaxResources, "compile-case resource count budget");
    require(table->owned_host_data.size() <= kCompileCaseMaxBlobs &&
        table->owned_diagnostic_data.size() <= kCompileCaseMaxBlobs - table->owned_host_data.size(),
        "compile-case blob count budget");
    size_t entries = 0;
    std::map<uintptr_t, CompileCaseBacking> opaque;
    for (const auto& r : table->resources) {
        require(r.table_entries.size() <= kCompileCaseMaxEntries - entries, "compile-case aggregate entry budget");
        entries += r.table_entries.size();
    }
    std::vector<CompileCaseBlob> original;
    for (auto& owner : {&table->owned_host_data, &table->owned_diagnostic_data}) {
        const auto kind = owner == &table->owned_host_data ? CompileCaseOwner::Host :
                                                           CompileCaseOwner::Diagnostic;
        for (const auto& allocation : *owner) {
            CompileCaseBlob b; b.owner = kind; b.present = bool(allocation);
            b.size = allocation ? allocation->size() : 0; b.bytes = allocation;
            original.push_back(std::move(b));
        }
    }
    require(original.size() <= kCompileCaseMaxBlobs, "compile-case blob count budget");
    for (const auto& r : table->resources) {
        if (unsupported_carrier(r)) incomplete(c, "unsupported-live-resource-carrier");
        CompileCaseResourceBacking refs;
        refs.host = locate(c, r.host_data, r.host_data_size, r.host_data_prefix_bytes, original, opaque);
        refs.dcc = locate(c, r.dcc_metadata_host_data, r.dcc_metadata_host_data_size, 0, original, opaque);
        require(r.table_entries.size() <= kCompileCaseMaxEntries, "compile-case table count budget");
        for (const auto& entry : r.table_entries)
            refs.entries.push_back(locate(c, entry.host_data, entry.host_data_size, 0, original, opaque));
        c.backing.push_back(std::move(refs));
    }
    // Keep owned buffer words available to actual compiler byte readers. An unowned ordinary
    // buffer can still be a metadata-only input, just like image pixels: it retains opaque
    // presence/range metadata, never a substitute pointer. A subsequent byte request throws.
    std::set<uint32_t> read_blobs;
    for (size_t k = 0; k < table->resources.size(); ++k) {
        const auto& r = table->resources[k];
        if (r.cls == ResourceClass::ConstantBuffer || r.cls == ResourceClass::VertexBuffer) {
            const auto index = c.backing[k].host.blob;
            if (index < original.size() && original[index].bytes) read_blobs.insert(index);
        }
    }
    for (uint32_t k = 0; k < original.size(); ++k)
        for (uint32_t prior = 0; prior < k; ++prior)
            if (original[k].bytes && original[k].bytes == original[prior].bytes) {
                original[k].alias_of = prior;
                if (read_blobs.contains(k)) {
                    uint32_t root = prior;
                    while (original[root].alias_of != UINT32_MAX) root = original[root].alias_of;
                    read_blobs.insert(root);
                }
                break;
            }
    uint64_t payload = 0;
    for (uint32_t k = 0; k < original.size(); ++k) {
        auto b = original[k];
        b.opaque = !read_blobs.contains(k);
        if (b.alias_of != UINT32_MAX) b.opaque = original[b.alias_of].opaque;
        original[k].opaque = b.opaque;
        if (b.alias_of == UINT32_MAX && !b.opaque) {
            require(b.size <= kCompileCaseMaxPayloadBytes - payload, "compile-case payload budget");
            payload += b.size;
        }
    }
    c.resources = *table;
    for (uint32_t k = 0; k < original.size(); ++k) {
        auto b = original[k];
        if (b.alias_of != UINT32_MAX) b.bytes = c.blobs[b.alias_of].bytes;
        else b.bytes = !b.opaque && b.bytes ? std::make_shared<std::vector<uint8_t>>(*b.bytes) : nullptr;
        c.blobs.push_back(std::move(b));
    }
    restore_fragment_compile_case_backing(c);
}

void restore_fragment_compile_case_backing(FragmentCompileCase& c) {
    c.resources.owned_host_data.clear(); c.resources.owned_diagnostic_data.clear();
    for (const auto& blob : c.blobs) {
        require(uint32_t(blob.owner) <= uint32_t(CompileCaseOwner::ExternalOpaque),
                "compile-case owner tag");
        if (blob.owner == CompileCaseOwner::ExternalOpaque) continue;
        (blob.owner == CompileCaseOwner::Host ? c.resources.owned_host_data :
                                             c.resources.owned_diagnostic_data).push_back(blob.bytes);
    }
    require(c.backing.size() == c.resources.resources.size(), "compile-case resource/backing count");
    for (size_t k = 0; k < c.backing.size(); ++k) {
        auto& r = c.resources.resources[k]; const auto& refs = c.backing[k];
        r.host_data = resolve(c, refs.host, r.host_data_size, r.host_data_prefix_bytes);
        r.dcc_metadata_host_data = resolve(c, refs.dcc, r.dcc_metadata_host_data_size, 0);
        require(refs.entries.size() == r.table_entries.size(), "compile-case entry/backing count");
        for (size_t e = 0; e < refs.entries.size(); ++e)
            r.table_entries[e].host_data = resolve(c, refs.entries[e], r.table_entries[e].host_data_size, 0);
    }
}

void validate_fragment_compile_case(const FragmentCompileCase& c) {
    require(c.compiler.size() <= 128 && c.reason.size() <= 256 && c.choices.error.size() <= 256 && c.expected_reject.size() <= 256,
            "compile-case text bound");
    require(c.complete == c.reason.empty(), "compile-case completeness/reason disagreement");
    require(c.code.size() <= kCompileCaseMaxWords && c.source.size() <= kCompileCaseMaxWords,
            "compile-case word count budget");
    if (c.complete) {
        require(!c.code.empty(), "compile-case raw missing");
        require(c.expected_produced == !c.source.empty(), "compile-case expected outcome disagreement");
        require(c.expected_produced ? c.expected_reject.empty() : !c.expected_reject.empty(),
                "compile-case expected refusal diagnostic absent");
        if (c.expected_produced) require(c.source.size() >= 5 && c.source[0] == 0x07230203u,
                                         "compile-case invalid SOURCE");
        require(c.compiler.find("unknown") == std::string::npos && !c.compiler.empty(), "compile-case compiler identity unknown");
        require(c.choices.error.empty(), "compile-case choices unavailable");
    }
    require(c.float_mode.canonical(), "compile-case noncanonical float mode");
    require(c.float_transport.canonical(), "compile-case noncanonical float transport");
    require(c.float_flags.canonical(), "compile-case noncanonical float flags");
    require(c.launch_rsrc1.canonical(), "compile-case noncanonical RSRC1_PS evidence");
    require(c.resources.resources.size() <= kCompileCaseMaxResources &&
            c.blobs.size() <= kCompileCaseMaxBlobs && c.choices.reads.size() <= 65536,
            "compile-case collection budget");
    require(c.has_resources || (c.resources.resources.empty() && c.blobs.empty() &&
                               c.resources.vertices_per_instance == 0), "compile-case absent table populated");
    uint64_t payload = 0;
    for (size_t k = 0; k < c.blobs.size(); ++k) {
        const auto& blob = c.blobs[k];
        require(uint32_t(blob.owner) <= uint32_t(CompileCaseOwner::ExternalOpaque),
                "compile-case owner tag");
        if (blob.owner == CompileCaseOwner::ExternalOpaque) require(blob.opaque && blob.present,
                                                "compile-case external backing cannot own bytes");
        require(blob.opaque ? !blob.bytes : blob.present == bool(blob.bytes), "compile-case blob presence");
        require(blob.present || blob.size == 0, "compile-case absent blob size");
        if (blob.alias_of != UINT32_MAX) {
            require(blob.alias_of < k && c.blobs[blob.alias_of].present == blob.present &&
                c.blobs[blob.alias_of].size == blob.size && c.blobs[blob.alias_of].opaque == blob.opaque &&
                c.blobs[blob.alias_of].bytes == blob.bytes, "compile-case alias identity");
        } else if (blob.bytes) require(blob.bytes->size() == blob.size, "compile-case blob size");
        const uint64_t n = blob.bytes && blob.alias_of == UINT32_MAX ? blob.bytes->size() : 0;
        require(n <= kCompileCaseMaxPayloadBytes - payload, "compile-case payload budget"); payload += n;
    }
    require(c.backing.size() == c.resources.resources.size(), "compile-case resource/backing count");
    size_t entries = 0;
    for (size_t k = 0; k < c.backing.size(); ++k) {
        const auto& r = c.resources.resources[k]; const auto& refs = c.backing[k];
        require(uint32_t(r.cls) <= uint32_t(ResourceClass::StorageImage) &&
                uint32_t(r.format) <= uint32_t(DataFormat::Sscaled16) &&
                uint32_t(r.table_selector_mode) <= uint32_t(BufferTableSelectorMode::DynamicSbufferByteOffset) &&
                uint32_t(r.fetch_index_mode) <= uint32_t(VertexFetchIndexMode::Instance), "compile-case resource enum");
    if (c.complete) require(!unsupported_carrier(r), "compile-case unsupported carrier");
        require(r.table_entries.size() <= kCompileCaseMaxEntries && refs.entries.size() == r.table_entries.size(),
                "compile-case table entries");
        require(r.table_entries.size() <= kCompileCaseMaxEntries - entries, "compile-case aggregate entries");
        entries += r.table_entries.size();
        valid_reference(c, refs.host, r.host_data_size, r.host_data_prefix_bytes);
        valid_reference(c, refs.dcc, r.dcc_metadata_host_data_size, 0);
        for (size_t e = 0; e < refs.entries.size(); ++e)
            valid_reference(c, refs.entries[e], r.table_entries[e].host_data_size, 0);
    }
    for (const auto& read : c.choices.reads) {
        require(uint32_t(read.choice) <= uint32_t(CompilerChoice::NormalizeSamplerCoordinates), "compile-case choice tag");
        if (read.choice == CompilerChoice::FragmentTapPc)
            require(read.value >= 0 && uint64_t(read.value) <= UINT32_MAX, "compile-case tap PC");
        else if (read.choice == CompilerChoice::ForcedArrayLayer)
            require(read.value >= -1 && read.value <= 65535, "compile-case forced layer");
        else require(read.value == 0 || read.value == 1, "compile-case choice boolean");
    }
}

std::vector<uint32_t> replay_fragment_compile_case(const FragmentCompileCase& c, bool baseline,
                                                 std::string* actual_refusal) {
    if (actual_refusal) actual_refusal->clear();
    validate_fragment_compile_case(c);
    require(c.complete, "INCOMPLETE fragment compile case cannot replay");
    const std::string compiler = std::string(prosper::embedded_build_revision()) + ":" + prosper::embedded_build_source_identity();
    if (baseline) require(c.compiler == compiler, "baseline compiler identity mismatch");
    FragmentCompileCase owned = c;
    restore_fragment_compile_case_backing(owned);
    std::vector<CompilerResourceAccess> access;
    for (size_t k = 0; k < owned.backing.size(); ++k) {
        const auto& r = owned.resources.resources[k]; const auto ref = owned.backing[k].host;
        const bool present = ref.blob < owned.blobs.size();
        const bool readable = present && !owned.blobs[ref.blob].opaque;
        access.push_back({&r, present, readable, readable ?
            std::span<const uint8_t>(r.host_data, static_cast<size_t>(r.host_data_size)) : std::span<const uint8_t>()});
    }
    CompilerResourceScope resources(std::move(access));
    TripBoundOperation trip(c.trip);
    CompilerChoiceScope choices(c.choices);
    TerminalRejectCapture rejection;
    auto result = recompile_fragment(c.code.data(), c.code.size(), c.has_resources ? &owned.resources : nullptr,
        c.has_system_inputs ? &c.system_inputs : nullptr, c.pcrel_target,
        &c.interpolation, c.wave32, {RecompileDiagnosticStage::Fragment, c.program_address},
        c.float_mode, nullptr, c.float_transport, c.float_flags);
    choices.finish_replay();
    if (baseline) require(result == c.source, "baseline SOURCE differs (full word comparison)");
    const auto reject_records = rejection.take();
    if (result.empty()) {
        require(!reject_records.empty(), "INCOMPLETE: actual compiler refusal diagnostic absent");
        const std::string reject = reject_records.back().first + " " + reject_records.back().second;
        require(reject.size() <= 256, "INCOMPLETE: actual compiler refusal diagnostic budget");
        if (baseline) require(reject == c.expected_reject, "baseline refusal diagnostic differs");
        if (actual_refusal) *actual_refusal = reject;
    }
    return result;
}

void finish_fragment_compile_case(FragmentCompileCase& c, std::vector<uint32_t> source) {
    c.source = std::move(source);
    c.expected_produced = !c.source.empty();
    if (c.expected_produced) c.expected_reject.clear();
    else if (c.expected_reject.empty()) incomplete(c, "compiler-refusal-diagnostic-unavailable");
    else if (c.expected_reject.size() > 256) {
        c.expected_reject.clear();
        incomplete(c, "compiler-refusal-diagnostic-budget");
    }
    if (!c.choices.error.empty()) incomplete(c, "compiler-choice-budget");
    if (c.compiler.empty() || c.compiler.find("unknown") != std::string::npos) incomplete(c, "compiler-identity-unavailable");
    if (!c.complete) return;
    try {
        // This is the producer's internal fidelity check, not another observed compiler request.
        // Explicit standalone baseline/candidate replay deliberately remains visible.
        prosper::diagnostics::perf::SuppressDrawDropCounting internal_check;
        (void)replay_fragment_compile_case(c, true);
    }
    catch (const std::exception&) { incomplete(c, "producing-baseline-selfcheck-failed"); }
}
uint64_t fragment_compile_case_retained_bytes(const FragmentCompileCase& c) {
    // Conservative charge includes owned vectors and metadata. Per-case codec limits are separate.
    uint64_t n = sizeof(c) + c.code.capacity() * 4u + c.source.capacity() * 4u +
        c.choices.reads.capacity() * sizeof(CompilerChoiceRead) +
        c.resources.resources.capacity() * sizeof(ShaderResource) + c.blobs.capacity() * sizeof(CompileCaseBlob) +
        c.backing.capacity() * sizeof(CompileCaseResourceBacking) + c.compiler.capacity() +
        c.reason.capacity() + c.expected_reject.capacity() + c.choices.error.capacity();
    for (const auto& blob : c.blobs) if (blob.bytes) n += sizeof(*blob.bytes) + blob.bytes->capacity();
    for (const auto& r : c.resources.resources) n += r.table_entries.capacity() * sizeof(ShaderBufferTableEntry);
    for (const auto& refs : c.backing) n += refs.entries.capacity() * sizeof(CompileCaseBacking);
    n += (c.resources.owned_host_data.capacity() + c.resources.owned_diagnostic_data.capacity()) * sizeof(std::shared_ptr<std::vector<uint8_t>>);
    return n;
}
bool fragment_compile_case_recording_enabled() {
    const char* dir = std::getenv("PROSPER_FRAGMENT_COMPILE_CASE_DIR"); return dir && *dir;
}
void maybe_dump_fragment_compile_case(const std::shared_ptr<const FragmentCompileCase>& producer,
                                     uint64_t lookup_address, const std::vector<uint32_t>& source) {
    const char* dir = std::getenv("PROSPER_FRAGMENT_COMPILE_CASE_DIR");
    if (!dir || !*dir) return;
    try {
        FragmentCompileCase missing;
        if (!producer) {
            missing.complete = false; missing.reason = "producing-context-not-retained";
            missing.compiler = "unknown"; missing.program_address = lookup_address; missing.source = source;
            missing.expected_produced = !source.empty();
        }
        const auto& c = producer ? *producer : missing;
        auto bytes = encode_fragment_compile_case(c);
        const uint64_t hash = gpu_capture_hash(bytes.data(), bytes.size());
        static std::mutex mutex;
        struct Installed { std::filesystem::path directory; uint64_t lookup; std::vector<uint8_t> bytes; };
        static std::vector<Installed> installed;
        static uint64_t retained = 0;
        std::lock_guard lock(mutex);
        const auto directory = std::filesystem::absolute(dir).lexically_normal();
        for (const auto& prior : installed)
            if (prior.directory == directory && prior.lookup == lookup_address && prior.bytes == bytes) return;
        constexpr uint64_t memo_budget = 64 * 1024 * 1024;
        const uint64_t charge = bytes.capacity() + directory.native().size() * sizeof(std::filesystem::path::value_type) + sizeof(Installed);
        if (installed.size() >= 4096 || charge > memo_budget - retained) {
            fprintf(stderr, "[fragment-compile-case] recording-memo-budget\n"); return;
        }
        std::filesystem::create_directories(directory);
        // A hash is only a readable filename hint. Exact bytes and destination decide reuse;
        // independently reserved suffixes prevent a collision from overwriting another case.
        bool written = false;
        for (unsigned suffix = 0; suffix < 4096 && !written; ++suffix) {
            char name[80]; snprintf(name, sizeof(name), "f_%016llx_%016llx_%u.prfc",
                static_cast<unsigned long long>(lookup_address),
                static_cast<unsigned long long>(hash), suffix);
            const auto path = directory / name;
            auto reservation = path; reservation += ".reserve";
            if (!std::filesystem::create_directory(reservation)) continue;
            try {
                if (std::filesystem::exists(path)) {
                    // Existing files are never replaced by the diagnostic exporter. An unreadable
                    // or different file keeps its name; try a fresh suffix instead.
                    try { written = encode_fragment_compile_case(read_fragment_compile_case(path)) == bytes; }
                    catch (const std::exception&) { written = false; }
                } else { write_fragment_compile_case(path, c, false); written = true; }
                std::filesystem::remove(reservation);
            } catch (...) { std::error_code ignored; std::filesystem::remove(reservation, ignored); throw; }
        }
        require(written, "compile-case filename reservation budget");
        retained += charge; installed.push_back({directory, lookup_address, std::move(bytes)});
        fprintf(stderr, "[fragment-compile-case] %s reason=%s source-words=%zu\n",
            c.complete ? "COMPLETE" : "INCOMPLETE", c.reason.empty() ? "none" : c.reason.c_str(), c.source.size());
    } catch (const std::exception& e) {
        fprintf(stderr, "[fragment-compile-case] recording-failed: %s\n", e.what());
    }
}
} // namespace prosper::gpu
