// Save the actual sidecar-replayed module, then execute that same module with lane-owned inputs.
#pragma once
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include <ios>
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include <filesystem>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
namespace prosper::test::ir {
inline std::vector<uint32_t> saved_integer64_shift_witness(const std::filesystem::path& dump,
                                                           uint64_t address, size_t index,
                                                           const LaneProgram& lp) {
    write_compare_witness_program(dump, address, index, lp);
    const auto stem = dump / ("cs_" + std::to_string(address) + "_" + std::to_string(index));
    const auto input = prosper::gpu::read_refused_shader_inputs(stem.string() + ".inputs");
    const auto replay = prosper::gpu::replay_refused_shader_inputs(lp.code, input);
    EXPECT_TRUE(replay.produced) << replay.reject;
    std::ofstream file(stem.string() + ".spv", std::ios::binary);
    file.write(reinterpret_cast<const char*>(replay.spirv.data()),
               static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
    file.close();
    EXPECT_TRUE(file.good());
    return replay.spirv;
}
inline LaneRun execute_integer64_shift_module(const std::vector<uint32_t>& module,
                                              const LaneInputs& in) {
    if (module.empty()) return {{}, "empty shift module"};
    std::vector<uint32_t> input;
    input.reserve(in.s0.size() + in.s1.size() + in.old_d.size() + in.active.size() + in.s2.size() +
                  in.s3.size());
    for (const auto* v : {&in.s0, &in.s1, &in.old_d})
        input.insert(input.end(), v->begin(), v->end());
    for (bool active : in.active) input.push_back(active ? 1 : 0);
    input.insert(input.end(), in.s2.begin(), in.s2.end());
    input.insert(input.end(), in.s3.begin(), in.s3.end());
    bpermute_oracle::Interpreter vm(module);
    auto out = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, input}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(in.lanes());
    return {out, {}};
}
} // namespace prosper::test::ir
