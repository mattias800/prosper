#pragma once
// Execute an actual native fragment module with explicitly owned Input words. The caller names
// a complete active subgroup for fixed READLANE and ballot witnesses. No quad packing, derivative,
// resource, partial-wave or divergent collective service is inferred from the module.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "fixtures/native_fragment_buffers.hpp"
#include <array>
#include <map>
#include <stdexcept>
#include <vector>

namespace prosper::test {
inline std::pair<bpermute_oracle::Lane, uint32_t>
native_fragment_lane(bpermute_oracle::Interpreter& vm,
                     const std::map<uint32_t, std::array<uint32_t, 4>>& locations,
                     const std::map<uint32_t, std::array<uint32_t, 4>>& builtins) {
    bpermute_oracle::Lane lane;
    uint32_t output = 0;
    for (const auto& global : vm.globals) {
        const auto& a = global.a;
        if (a.size() != 3 || (a[2] != 1 && a[2] != 3 && a[2] != 12))
            throw std::runtime_error("unsupported native fragment global");
        const auto type = vm.pointee(a[0]);
        lane.p[a[1]] = {a[1], 0, type};
        if (a[2] == 12) {
            // Declaring an unused buffer does not supply it. An actual read fails bounds/poison.
            vm.shared[a[1]] = vm.make_memory(type, 0);
            continue;
        }
        auto memory = vm.make_memory(type, vm.size(type));
        const auto location = vm.decorations.find({a[1], 30});
        if (a[2] == 1) {
            const auto builtin = vm.builtins.find(a[1]);
            const std::array<uint32_t, 4>* supplied = nullptr;
            if (builtin != vm.builtins.end() && builtins.contains(builtin->second))
                supplied = &builtins.at(builtin->second);
            else if (location != vm.decorations.end() && location->second.size() == 1 &&
                     locations.contains(location->second[0]))
                supplied = &locations.at(location->second[0]);
            // Unused declarations grant no input. Leave absent words poison so an actual
            // load fails in the shared VM, just like an unwritten Output or missing buffer.
            if (supplied) {
                if (memory.words.size() > supplied->size())
                    throw std::runtime_error("native fragment Input shape mismatch");
                std::copy_n(supplied->begin(), memory.words.size(), memory.words.begin());
                std::fill(memory.valid.begin(), memory.valid.end(), true);
            }
        } else if (location != vm.decorations.end() &&
                   location->second == std::vector<uint32_t>{0}) {
            if (output || memory.words.size() != 4)
                throw std::runtime_error("ambiguous native fragment Output");
            output = a[1];
        }
        lane.memory[a[1]] = std::move(memory);
    }
    if (!output || !vm.entry) throw std::runtime_error("missing native fragment entry/output");
    lane.pc = vm.entry - 1;
    return {std::move(lane), output};
}

inline void native_fragment_subgroup(bpermute_oracle::Interpreter& vm,
                                     std::vector<bpermute_oracle::Lane>& lanes) {
    for (size_t phase = 0; phase < 1000; ++phase) {
        for (auto& lane : lanes) vm.advance(lane);
        if (std::all_of(lanes.begin(), lanes.end(), [](const auto& lane) { return lane.done; }))
            return;
        const auto site = lanes.front().pc;
        const auto& collective = vm.ins.at(site);
        const auto& a = collective.a;
        if (collective.op ==
            339) {   // OpGroupNonUniformBallot over the explicitly supplied subgroup
            if ((lanes.size() != 32 && lanes.size() != 64) || a.size() != 4)
                throw std::runtime_error("unsupported native fragment ballot");
            const auto& type = vm.ty(a[0]);
            if (type.op != 23 || type.a.size() != 2 || type.a[1] != 4 || !vm.uint_type(type.a[0]))
                throw std::runtime_error("unsupported native fragment ballot type");
            std::vector<uint32_t> bits(4);
            for (uint32_t index = 0; index < lanes.size(); ++index) {
                auto& lane = lanes[index];
                if (lane.done || lane.pc != site || vm.word(lane, a[2]) != 3)
                    throw std::runtime_error("partial/divergent native fragment ballot");
                if (vm.boolean(lane, a[3])) bits[index / 32] |= 1u << (index % 32);
            }
            for (auto& lane : lanes) {
                vm.set(lane, a[1], a[0], bits);
                ++lane.pc;
            }
            continue;
        }
        if ((lanes.size() != 32 && lanes.size() != 64) || collective.op != 345 || a.size() != 5 ||
            !vm.uint_type(a[0]))
            throw std::runtime_error("unsupported native fragment collective");
        std::vector<bpermute_oracle::Value> sources;
        for (auto& lane : lanes) {
            if (lane.done || lane.pc != site || vm.word(lane, a[2]) != 3)
                throw std::runtime_error("partial/divergent native fragment subgroup");
            auto source = vm.value(lane, a[3]);
            if (source.type != a[0] || !source.defined || source.words.size() != 1)
                throw std::runtime_error("undefined native fragment shuffle source");
            sources.push_back(std::move(source));
        }
        for (auto& lane : lanes) {
            const auto index = vm.word(lane, a[4]);
            if (!vm.uint_type(vm.value(lane, a[4]).type) || index >= lanes.size())
                throw std::runtime_error("native fragment shuffle index outside fixture");
            vm.set(lane, a[1], a[0], sources[index].words);
            ++lane.pc;
        }
    }
    throw std::runtime_error("nonterminating native fragment subgroup");
}

inline std::array<uint32_t, 4>
native_fragment_words(const std::vector<uint32_t>& module,
                      const std::map<uint32_t, std::array<uint32_t, 4>>& locations,
                      const std::map<uint32_t, std::array<uint32_t, 4>>& builtins,
                      uint32_t width = 1, const NativeFragmentBuffers& buffers = {}) {
    if (width != 1 && width != 32 && width != 64)
        throw std::runtime_error("unsupported native fragment fixture width");
    bpermute_oracle::Interpreter vm(module, 4);
    if (!vm.error.empty()) throw std::runtime_error(vm.error);
    std::vector<bpermute_oracle::Lane> lanes;
    uint32_t output = 0;
    for (uint32_t i = 0; i < width; ++i) {
        auto supplied = builtins;
        if (width == 32 || width == 64) {
            supplied[41] = {i, 0, 0,
                            0};   // SubgroupLocalInvocationId in this complete fixture cohort
            supplied[36] = {width, 0, 0,
                            0};   // SubgroupSize, never inferred from a module declaration
        }
        auto invocation = native_fragment_lane(vm, locations, supplied);
        lanes.push_back(std::move(invocation.first));
        output = invocation.second;
    }
    native_fragment_buffers(vm, buffers);
    native_fragment_subgroup(vm, lanes);
    std::array<uint32_t, 4> expected{};
    for (uint32_t i = 0; i < width; ++i) {
        const auto& result = lanes[i].memory.at(output);
        if (!std::all_of(result.valid.begin(), result.valid.end(),
                         [](bool value) { return value; }))
            throw std::runtime_error("undefined native fragment Output");
        const std::array<uint32_t, 4> actual{result.words[0], result.words[1], result.words[2],
                                             result.words[3]};
        if (i && actual != expected) throw std::runtime_error("nonuniform fixture colour");
        expected = actual;
    }
    return expected;
}
}   // namespace prosper::test
