#pragma once

#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <filesystem>
#include <span>

namespace prosper::gpu {

inline constexpr uint32_t kCompileCaseMaxWords = 65536;
inline constexpr uint32_t kCompileCaseMaxResources = 512;
inline constexpr uint32_t kCompileCaseMaxEntries = 4096;
inline constexpr uint32_t kCompileCaseMaxBlobs = 1024;
inline constexpr uint32_t kCompileCaseMaxPayloadBytes = 16 * 1024 * 1024;
inline constexpr uint32_t kCompileCaseMaxFileBytes = 32 * 1024 * 1024;

enum class CompileCaseOwner : uint32_t { Host, Diagnostic, ExternalOpaque };
struct CompileCaseBlob {
    CompileCaseOwner owner = CompileCaseOwner::Host;
    bool present = true;
    bool opaque = false;
    uint64_t size = 0;
    uint32_t alias_of = UINT32_MAX;
    std::shared_ptr<std::vector<uint8_t>> bytes;
};
struct CompileCaseBacking {
    static constexpr uint32_t kAbsent = UINT32_MAX, kUnavailable = UINT32_MAX - 1u;
    uint32_t blob = kAbsent;
    uint64_t offset = 0;
};
struct CompileCaseResourceBacking {
    CompileCaseBacking host, dcc;
    std::vector<CompileCaseBacking> entries;
};

// A compiler invocation, not a draw capture. Logical guest addresses are never replay pointers.
// Backing references preserve allocation offsets and ownership membership without serializing
// host addresses. INCOMPLETE cases are inspectable but cannot enter the compiler.
struct FragmentCompileCase {
    bool complete = true;
    std::string reason, compiler, expected_reject;
    bool expected_produced = false;
    uint64_t program_address = 0;
    std::vector<uint32_t> code, source;
    bool has_resources = false, has_pixel_inputs = false, has_system_inputs = false;
    PixelInputMapping pixel_inputs{};
    PixelSystemInputMapping system_inputs{};
    FragmentInterpolationLayout interpolation{};
    bool wave32 = false;
    FragmentFloatMode float_mode{}; // actual producing key authority; unavailable is never inferred
    // Schema 2 retains even an explicitly producing Unknown profile. A schema-1 file has no
    // such input and is downgraded to INCOMPLETE, never inferred from SOURCE or today's device.
    FloatTransportConfig float_transport{};
    // Schema 3 retains independent launch flags. Legacy schemas cannot reconstruct them
    // from FLOAT_MODE, SOURCE, width or a host device and remain inspectable INCOMPLETE.
    FragmentFloatFlags float_flags{};
    uint32_t pcrel_target = UINT32_MAX;
    ComputeTripBoundSettings trip{};
    ShaderResourceTable resources;
    std::vector<CompileCaseBlob> blobs;
    std::vector<CompileCaseResourceBacking> backing;
    CompilerChoiceTrace choices;
};

// Deep ownership is established before an opted-in producing compile. Unsupported live-read
// carriers or missing ownership retain metadata but explicitly decline complete replay.
void own_fragment_compile_case_resources(FragmentCompileCase&, const ShaderResourceTable*);
void restore_fragment_compile_case_backing(FragmentCompileCase&);
void validate_fragment_compile_case(const FragmentCompileCase&);
std::vector<uint32_t> replay_fragment_compile_case(const FragmentCompileCase&, bool baseline,
                                                 std::string* actual_refusal = nullptr);
void finish_fragment_compile_case(FragmentCompileCase&, std::vector<uint32_t> source);
std::vector<uint8_t> encode_fragment_compile_case(const FragmentCompileCase&);
FragmentCompileCase decode_fragment_compile_case(std::span<const uint8_t>);
// Explicit output replacement is atomic. Diagnostic exports use replace=false, which must never
// overwrite an existing case even if another writer installs it after the caller's existence check.
void write_fragment_compile_case(const std::filesystem::path&, const FragmentCompileCase&, bool replace = true);
FragmentCompileCase read_fragment_compile_case(const std::filesystem::path&);
void write_fragment_compile_source(const std::filesystem::path&, std::span<const uint32_t>);
uint64_t fragment_compile_case_retained_bytes(const FragmentCompileCase&);
bool fragment_compile_case_recording_enabled();
// Warm exports with a null producer write only explicit INCOMPLETE records; they do not read
// current code/resources/environment to invent provenance for an older cache entry.
void maybe_dump_fragment_compile_case(const std::shared_ptr<const FragmentCompileCase>& producer,
                                     uint64_t lookup_program_address,
                                     const std::vector<uint32_t>& source);

} // namespace prosper::gpu
