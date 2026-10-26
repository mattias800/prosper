// Diagnostic roots of fabricated words. No result from this module grants admission.
#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_launch_vgprs.hpp"
#include <compare>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace prosper::gpu {
struct RegState;
struct SpirvCompute;
struct ComputeShaderConfig;
enum class PlaceholderOrigin {
    Unloaded,
    LaunchInput,
    Spill,
    UnknownLaunch,
    LostDefinition,
    UnmodelledTransfer,
    TraceBudget,
    Unreachable
};
enum class PlaceholderFile { Scalar, Vector, Spill, Scc };
struct PlaceholderWord {
    PlaceholderFile file = PlaceholderFile::Scalar;
    int reg = 0;
    int lane = -1;
    auto operator<=>(const PlaceholderWord&) const = default;
};
struct PlaceholderRoot {
    PlaceholderOrigin origin = PlaceholderOrigin::UnknownLaunch;
    PlaceholderWord word;
    uint32_t pc = UINT32_MAX;
    auto operator<=>(const PlaceholderRoot&) const = default;
};
using PlaceholderRoots = std::set<PlaceholderRoot>;
struct PlaceholderOriginObservations {
    bool native_entry_checked = false;
    std::set<int> supplied_special_words;
    std::map<uint32_t, bool> genuine_resources;
    std::set<uint32_t> native_inputs;
    std::map<uint32_t, std::set<int>> image_addresses;
    std::map<uint32_t, std::set<int>> relative_words;
    std::set<uint32_t> unbounded_relative;
};
struct PlaceholderOriginProgram {
    std::vector<Rdna2Inst> instructions;
    std::vector<LaunchVgpr> vectors;
    std::set<int> supplied_vectors, supplied_scalars, loaded_scalars;
    bool vector_profile_known = false, scalar_profile_known = false;
    bool native_interpolation = false, entry_full_exec = false;
    uint32_t wave_size = 64;
    // Observations of existing publications, not new validity or resource facts.
    std::shared_ptr<PlaceholderOriginObservations> observations =
        std::make_shared<PlaceholderOriginObservations>();
};
void seed_placeholder_origins(RegState&, const std::vector<Rdna2Inst>&,
                              const std::vector<LaunchVgpr>*, bool native_interpolation);
void configure_compute_placeholder_origins(RegState&, const ComputeShaderConfig&);
void configure_pixel_placeholder_origins(RegState&, const std::vector<LaunchVgpr>&,
                                         bool profile_known);
void note_placeholder_resources(const SpirvCompute&, const RegState&, const Rdna2Inst&);
void note_placeholder_native_input(const SpirvCompute&, const RegState&, const Rdna2Inst&);
void note_placeholder_image_addresses(const RegState&, const Rdna2Inst&);
PlaceholderRoots placeholder_roots_before(const PlaceholderOriginProgram&, uint32_t pc,
                                          const std::vector<PlaceholderWord>&);
std::string format_placeholder_origins(const PlaceholderRoots&);
std::string fresh_placeholder_origins(const RegState&, const Rdna2Inst&,
                                      const PlaceholderOriginProgram* fallback = nullptr);
std::string loop_placeholder_origins(const RegState&, uint32_t header,
                                     const std::vector<int>& words,
                                     const PlaceholderOriginProgram* fallback = nullptr);
} // namespace prosper::gpu
