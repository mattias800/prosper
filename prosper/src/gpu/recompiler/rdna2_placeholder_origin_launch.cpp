#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_launch_vgprs.hpp"
#include <vector>
#include <utility>
#include <string>
#include <cstdint>
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_reg_state.hpp"
#include <memory>

namespace prosper::gpu {
void seed_placeholder_origins(RegState& rs, const std::vector<Rdna2Inst>& instructions,
                              const std::vector<LaunchVgpr>* launch, bool native_interpolation) {
    auto p = std::make_shared<PlaceholderOriginProgram>();
    p->instructions = instructions;
    if (launch) p->vectors = *launch;
    p->vector_profile_known = launch && launch_profile_complete(*launch);
    p->native_interpolation = native_interpolation;
    p->wave_size = rs.vector_wave_size;
    p->entry_full_exec = rs.exec && !rs.exec_narrowed;
    for (const auto& [word, value] : rs.vreg)
        if (value) p->supplied_vectors.insert(word);
    for (const auto& [word, value] : rs.sreg)
        if (value) p->supplied_scalars.insert(word);
    for (const auto& [word, value] : rs.sreg_input)
        if (value) p->supplied_scalars.insert(word);
    rs.placeholder_origin_program = std::move(p);
}

void configure_compute_placeholder_origins(RegState& rs, const ComputeShaderConfig& config) {
    if (!rs.placeholder_origin_program) return;
    // Copy on configuration: copies of an earlier RegState retain their producing inventory.
    auto p = std::make_shared<PlaceholderOriginProgram>(*rs.placeholder_origin_program);
    p->scalar_profile_known = config.dispatch_launch.known == ComputeDispatchLaunchState::kAll;
    int end = static_cast<int>(config.user_sgprs.size());
    end += config.tgid_x_en + config.tgid_y_en + config.tgid_z_en + config.tg_size_en;
    for (int word = 0; word < end; ++word) p->loaded_scalars.insert(word);
    rs.placeholder_origin_program = std::move(p);
}

void configure_pixel_placeholder_origins(RegState& rs, const std::vector<LaunchVgpr>& inventory,
                                         bool known) {
    if (!rs.placeholder_origin_program) return;
    auto p = std::make_shared<PlaceholderOriginProgram>(*rs.placeholder_origin_program);
    p->vectors = inventory;
    p->vector_profile_known = known && launch_profile_complete(inventory);
    rs.placeholder_origin_program = std::move(p);
}

std::string format_placeholder_origins(const PlaceholderRoots& roots) {
    static constexpr const char* names[] = {"unloaded",
                                            "launch-input",
                                            "spill",
                                            "unknown-launch",
                                            "lost-definition",
                                            "unmodelled-transfer",
                                            "trace-budget",
                                            "unreachable",
                                            "unproved-memory-result",
                                            "unproved-strong-read",
                                            "unmodelled-scalar-publication"};
    std::string out;
    for (const auto& root : roots) {
        if (!out.empty()) out += ';';
        out += "origin=";
        out += names[static_cast<unsigned>(root.origin)];
        out += " word=";
        switch (root.word.file) {
            case PlaceholderFile::Scalar: out += 's'; break;
            case PlaceholderFile::Vector: out += 'v'; break;
            case PlaceholderFile::Spill: out += "spill-v"; break;
            case PlaceholderFile::Scc: out += "scc"; break;
        }
        if (root.word.file != PlaceholderFile::Scc) out += std::to_string(root.word.reg);
        if (root.word.lane >= 0) out += ":" + std::to_string(root.word.lane);
        out += root.pc == UINT32_MAX ? " at=entry" : " at=" + std::to_string(root.pc);
    }
    return out;
}
} // namespace prosper::gpu
