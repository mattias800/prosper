#pragma once

#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/diagnostics/refused_shader_dump.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"

#include <array>
#include <filesystem>
#include <functional>
#include <memory>

namespace prosper::gpu {
struct GpuState;
struct RefusedShaderInputs;
struct RefusedShaderReplay;

enum class RefusedGraphicsCompiler : uint32_t {
    Unavailable,
    Vertex,
    VertexChain,
    Fragment,
    NggSubgroup
};

// Physical producing register observations, with absence kept separate from an observed zero.
// These are diagnostic launch evidence, never inferred shader SGPRs or new admission authority.
struct RefusedGraphicsLaunch {
    bool observed = false;
    std::array<std::array<uint32_t, 32>, 3> user_data{}; // GS, VS, PS physical banks
    std::array<uint32_t, 3> available{};
    std::array<uint32_t, 3> rsrc1{}, rsrc2{};
    uint8_t rsrc1_available = 0, rsrc2_available = 0;
};

// Exact arguments at one actual compiler boundary. The chain compiler owns its own link/rewrite;
// retaining both input spans and the selector reproduces that same transformation. `raw` binds the
// separate dump to this invocation, which may be linked source rather than the guest prolog alone.
struct RefusedGraphicsInputs {
    RefusedGraphicsCompiler selector = RefusedGraphicsCompiler::Unavailable;
    FragmentCompileCase invocation;
    std::vector<uint32_t> raw, chain;
    bool raw_is_source_prefix =
        false; // producer-established NGG span relation, not replay inference
    bool capture_position = false;
    uint32_t vertex_lds_dwords = 0;
    NggSubgroupShellConfig shell;
    RefusedGraphicsLaunch launch;
    bool payload_omitted = false;
    std::string incomplete;
};

// Scope owns no register copies. Only a cold compiler attempt reads the producing state while it
// is still live; a cache hit or disarmed/full dump adds no table/register traversal.
class RefusedGraphicsLaunchScope {
public:
    explicit RefusedGraphicsLaunchScope(const GpuState&);
    ~RefusedGraphicsLaunchScope();
    RefusedGraphicsLaunchScope(const RefusedGraphicsLaunchScope&) = delete;
    RefusedGraphicsLaunchScope& operator=(const RefusedGraphicsLaunchScope&) = delete;

private:
    friend RefusedGraphicsLaunch refused_graphics_launch();
    const GpuState* state_;
    RefusedGraphicsLaunchScope* previous_;
};
RefusedGraphicsLaunch refused_graphics_launch();

void own_refused_graphics_resources(RefusedGraphicsInputs&, const ShaderResourceTable*);
bool write_refused_graphics_inputs(const std::filesystem::path&, const std::string& stage,
                                   const RefusedGraphicsInputs&);
RefusedShaderInputs decode_refused_graphics_inputs(const std::vector<uint8_t>&);
RefusedShaderReplay replay_refused_graphics_inputs(const std::vector<uint32_t>&,
                                                   const RefusedShaderInputs&, bool use_inputs);
uint64_t refused_graphics_inputs_identity(const RefusedGraphicsInputs&);

// Synchronous producing attempt. The original compiler consumes its original pointers, under an
// observation-only semantic-choice trace; only a refusal is published, with pre-attempt owned bytes.
using RefusedGraphicsCompile = std::function<std::vector<uint32_t>()>;
std::vector<uint32_t> observe_refused_graphics_compile(const char* stage,
                                                       const RefusedShaderSource&,
                                                       RefusedGraphicsInputs,
                                                       const ShaderResourceTable*,
                                                       const RefusedGraphicsCompile&,
                                                       const std::string* direct_refusal = nullptr);
void note_unavailable_graphics_inputs(const char* stage, uint64_t address,
                                      const RefusedShaderSource&, const std::string& detail);
}   // namespace prosper::gpu
