// Retaining an existing inactive word is separate from reading its value in active lanes.
#pragma once
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_launch_vgprs.hpp"
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper::test::inactive_destination_storage {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;
enum class OldValue { Missing, Reserved, Placeholder, Supplied };
enum class Operation { Copy, Accumulate };
enum class Observation { ActiveOnly, AllLanes };
inline constexpr uint32_t sentinel = 0xdeadbeefu;
struct Compilation {
    Words module;
    std::string roots;
};
inline Words program(Operation operation, Observation observation) {
    // LLVM gfx1030 Wave64 encodings: full scratch initialization, CMPX 16>v3, active update.
    Words words{0x7e0402ffu, sentinel, 0x7da80690u,
                operation == Operation::Copy ? 0x7e000200u : 0x56000200u};
    if (observation == Observation::AllLanes) words.push_back(0xbefe04c1u);
    words.insert(words.end(), {0x7e040300u, 0xbf810000u}); // v2=v0; END
    return words;
}
inline Words emit(bool native, OldValue old, Operation operation, Observation observation,
                  std::string& roots) {
    SpirvCompute b;
    b.begin(4, nullptr, 64, 1, 1, 64, 1, true, true);
    b.native_subgroup_size = native ? 64 : 0;
    ComputeShaderConfig config;
    config.local_x = config.wave_size = 64;
    config.user_sgprs.resize(1);
    complete_compute_launch(config);
    bind_checked_compute_launch(b, config);
    RegState rs;
    rs.exec = b.btrue();
    rs.vcc = rs.scc = b.bfalse();
    rs.vector_full_exec = true;
    rs.vector_wave_size = 64;
    rs.sreg[0] = b.load_push_constant(0);
    rs.vreg[1] = b.uconst(0x3f800000u);
    rs.vreg[3] = b.guest_lane_id();
    if (old == OldValue::Supplied)
        rs.vreg[0] = operation == Operation::Copy ? b.guest_lane_id() : b.uconst(0x3e800000u);
    else if (old == OldValue::Placeholder) {
        rs.vreg[0] = b.uconst(0);
        rs.vreg_source_marks[0].placeholder = rs.vreg_source_marks[0].hard = true;
    } else if (old == OldValue::Reserved)
        rs.vreg[0] = 0; // a reserved map entry is not a stored SPIR-V value
    const auto words = program(operation, observation);
    auto origins = std::make_shared<PlaceholderOriginProgram>();
    if (rdna2_walk(words.data(), words.size(), origins->instructions) != words.size()) return {};
    origins->vector_profile_known = origins->scalar_profile_known = origins->entry_full_exec = true;
    // Compute TIDX v0 is a real launch input. This fixture deliberately omits its binding in
    // the Hard twins; it is never an unloaded word or permission for Arbitrary admission.
    origins->vectors.push_back({0, 0, 0, LaunchVgprState::Unsupplied});
    origins->supplied_vectors.insert(1);
    origins->supplied_vectors.insert(3);
    if (old == OldValue::Supplied) origins->supplied_vectors.insert(0);
    origins->supplied_scalars.insert(0);
    b.placeholder_origin_program = rs.placeholder_origin_program = origins;
    if (!emit_body(
            b, rs, origins->instructions, {}, nullptr, true, false,
            [](RegState&, const Rdna2Inst&) { return true; }, words.data(), words.size())) {
        if (b.diagnostic.placeholder_origins) roots = *b.diagnostic.placeholder_origins;
        return {};
    }
    b.store_output_word(rs.vreg.at(2), 1, 0);
    return b.finish();
}
inline Compilation compile(bool native, OldValue old = OldValue::Placeholder,
                           Operation operation = Operation::Copy,
                           Observation observation = Observation::ActiveOnly) {
    Compilation result;
    result.module = recompile_with_arbitrary_placeholders({}, [&] {
        if (!allow_scalar_merge_dependency_repair()) return Words{};
        return emit(native, old, operation, observation, result.roots);
    });
    return result;
}
inline Words expected(uint32_t input, Observation observation = Observation::ActiveOnly) {
    Words result(64, sentinel);
    for (uint32_t lane = 0; lane < 64; ++lane)
        if (lane < 16)
            result[lane] = input;
        else if (observation == Observation::AllLanes)
            result[lane] = lane;
    return result;
}
} // namespace prosper::test::inactive_destination_storage
