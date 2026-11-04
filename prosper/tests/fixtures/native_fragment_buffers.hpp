#pragma once
// Explicit fixture-owned, readonly U32 storage buffers. A declaration alone still supplies nothing.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace prosper::test {
using NativeFragmentBuffers = std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>;
inline void native_fragment_buffers(bpermute_oracle::Interpreter& vm,
                                    const NativeFragmentBuffers& supplied) {
    for (const auto& [key, words] : supplied) {
        uint32_t root = 0;
        for (const auto& global : vm.globals) {
            const auto& a = global.a;
            if (a.size() != 3 || a[2] != 12) continue;
            const auto set = vm.decorations.find({a[1], 34});
            const auto binding = vm.decorations.find({a[1], 33});
            if (set == vm.decorations.end() || binding == vm.decorations.end() ||
                set->second != std::vector<uint32_t>{key.first} ||
                binding->second != std::vector<uint32_t>{key.second})
                continue;
            if (root) throw std::runtime_error("ambiguous native fragment buffer");
            root = a[1];
        }
        if (!root) throw std::runtime_error("missing native fragment buffer declaration");
        const auto type = vm.shared.at(root).type;
        const auto& block = vm.ty(type);
        if (block.op != 30 || block.a.size() != 1)
            throw std::runtime_error("unsupported native fragment buffer block");
        const auto array = block.a[0];
        const auto& element = vm.ty(array);
        if (element.op != 29 || element.a.size() != 1 || !vm.uint_type(element.a[0]) ||
            vm.decorations[{array, 6}] != std::vector<uint32_t>{4} ||
            vm.member_decorations[{type, 0, 35}] != std::vector<uint32_t>{0})
            throw std::runtime_error("unsupported native fragment buffer layout");
        auto memory = vm.make_memory(type, words.size());
        memory.words = words;
        std::fill(memory.valid.begin(), memory.valid.end(), true);
        vm.shared[root] = std::move(memory);
        vm.readonly_storage[root] = true;
    }
}
}   // namespace prosper::test
