#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace prosper::frontend {

// depth is Z for a 3D image, or the layer count for an array image. Aliases can
// share mask_binding, but must describe the same view and dimensions. The caller
// supplies a zeroed width*height*depth uint32_t StorageBuffer for each mask.
struct StorageWriteMaskTarget {
    uint32_t descriptor_set = 0;
    uint32_t image_binding = 0;
    uint32_t mask_binding = 0;
    uint32_t width = 0;
    uint32_t height = 1;
    uint32_t depth = 1;
    bool arrayed = false;
};

struct StorageWriteMaskSpirvResult {
    std::vector<uint32_t> words;
    std::string error;
    uint32_t instrumented_writes = 0;
    explicit operator bool() const { return error.empty() && !words.empty(); }
};

// Deliberately bounded to the compiler's direct, non-MS storage-image
// loads. This is not a general SPIR-V validator. The original module must already
// be valid. Unknown store provenance, descriptor arrays, image atomics, and
// unsupported coordinate types are errors, never permission to omit tracking.
// No control-flow edges are changed: out-of-range writes atomically OR zero into
// mask[0]. In-range writes OR one at their exact texel, including same-value
// stores. This preserves loops, existing OpPhi predecessors and execution masks.
inline StorageWriteMaskSpirvResult instrument_storage_image_writes(
    std::span<const uint32_t> module,
    std::span<const StorageWriteMaskTarget> targets) {
    using Words = std::vector<uint32_t>;
    auto fail = [](const char* reason) {
        return StorageWriteMaskSpirvResult{{}, reason, 0};
    };
    if (module.size() < 5 || module[0] != 0x07230203u || module[4] != 0 ||
        module[3] == 0 || module[1] < 0x00010300u || module[1] > 0x00010600u)
        return fail("store mask requires a SPIR-V 1.3-1.6 module header");
    struct Inst { size_t at; uint32_t count, op; };
    std::vector<Inst> instructions;
    std::map<uint32_t, Inst> types, variables, loads;
    std::map<uint32_t, uint32_t> value_types, bindings, sets;
    size_t first_type = module.size(), first_function = module.size();
    uint32_t uint_type = 0, bool_type = 0, uint_pointer = 0;
    unsigned entries = 0, memory_models = 0;
    bool in_function = false;
    auto typed_result = [](uint32_t op) {
        return op == 12 || (op >= 41 && op <= 52) || op == 54 || op == 55 ||
            op == 57 || op == 59 || op == 61 || (op >= 65 && op <= 70) ||
            (op >= 77 && op <= 84) || (op >= 86 && op <= 98) ||
            (op >= 100 && op <= 107) || (op >= 109 && op <= 215) ||
            op == 227 || (op >= 229 && op <= 242) || op == 245;
    };
    for (size_t at = 5; at < module.size();) {
        const uint32_t count = module[at] >> 16, op = module[at] & 0xffffu;
        if (!count || count > module.size() - at)
            return fail("truncated or zero-length SPIR-V instruction");
        const Inst inst{at, count, op};
        instructions.push_back(inst);
        if (op == 14) {
            if (count != 3 || module[at + 1] != 0 || module[at + 2] != 1)
                return fail("store mask requires Logical GLSL450 memory model");
            ++memory_models;
        }
        if (op == 15) {
            if (count < 4 || module[at + 1] != 5)
                return fail("store mask requires one compute entry point");
            ++entries;
        }
        if (op >= 19 && op <= 39) {
            if (count < 2 || !module[at + 1] || module[at + 1] >= module[3])
                return fail("invalid SPIR-V type declaration");
            types.emplace(module[at + 1], inst);
            first_type = std::min(first_type, at);
            if (op == 21 && count == 4 && module[at + 2] == 32 && module[at + 3] == 0)
                uint_type = module[at + 1];
            if (op == 20 && count == 2) bool_type = module[at + 1];
        }
        if (op == 54) {
            if (in_function || count != 5) return fail("invalid function declaration");
            first_function = std::min(first_function, at);
            in_function = true;
        }
        if (op == 56) {
            if (!in_function || count != 1) return fail("invalid function end");
            in_function = false;
        }
        if (typed_result(op)) {
            if (count < 3 || !module[at + 2] || module[at + 2] >= module[3])
                return fail("invalid SPIR-V typed result");
            value_types.emplace(module[at + 2], module[at + 1]);
        }
        if (op == 59 && !in_function) {
            if (count < 4) return fail("invalid global variable");
            variables.emplace(module[at + 2], inst);
        }
        if (op == 61) {
            if (count < 4) return fail("invalid image load");
            loads.emplace(module[at + 2], inst);
        }
        if (op == 71 && count >= 3 && (module[at + 2] == 33 || module[at + 2] == 34)) {
            if (count != 4) return fail("invalid descriptor decoration");
            auto& table = module[at + 2] == 33 ? bindings : sets;
            if (!table.emplace(module[at + 1], module[at + 3]).second)
                return fail("duplicate descriptor decoration");
        }
        at += count;
    }
    if (in_function || entries != 1 || memory_models != 1 || first_type >= first_function ||
        first_function == module.size()) return fail("incomplete compute module");
    if (targets.empty()) return StorageWriteMaskSpirvResult{Words(module.begin(), module.end()), {}, 0};
    // Avoid collisions with existing descriptors, not only the selected images.
    using Binding = std::pair<uint32_t, uint32_t>;
    std::map<Binding, uint32_t> descriptors;
    for (const auto& [id, binding] : bindings) {
        const auto set = sets.find(id);
        if (set == sets.end()) return fail("descriptor has no direct DescriptorSet decoration");
        if (!descriptors.emplace(Binding{set->second, binding}, id).second)
            return fail("ambiguous descriptor binding");
    }
    struct TargetInfo { StorageWriteMaskTarget target; uint32_t dim, variable, image_type; };
    std::map<uint32_t, TargetInfo> selected;
    std::map<Binding, TargetInfo> masks;
    auto get_type = [&](uint32_t id) -> const Inst* {
        const auto found = types.find(id);
        return found == types.end() ? nullptr : &found->second;
    };
    for (const auto& target : targets) {
        if (!target.width || !target.height || !target.depth ||
            uint64_t(target.width) * target.height > std::numeric_limits<uint32_t>::max() /
                                                       uint64_t(target.depth))
            return fail("invalid or overflowing store-mask extent");
        const auto descriptor = descriptors.find({target.descriptor_set, target.image_binding});
        if (descriptor == descriptors.end()) return fail("target image binding is absent");
        const auto variable = variables.find(descriptor->second);
        if (variable == variables.end()) return fail("target is not a global variable");
        const Inst* pointer = get_type(module[variable->second.at + 1]);
        if (!pointer || pointer->op != 32 || pointer->count != 4 ||
            module[pointer->at + 2] != 0 || module[variable->second.at + 3] != 0)
            return fail("target is not a direct UniformConstant image");
        const uint32_t image_type = module[pointer->at + 3];
        const Inst* image = get_type(image_type);
        if (!image || image->op != 25 || image->count < 9)
            return fail("target is not a non-array descriptor image type");
        const Inst* scalar = get_type(module[image->at + 2]);
        const bool numeric32 = scalar &&
            ((scalar->op == 21 && scalar->count == 4 && module[scalar->at + 3] <= 1) ||
             (scalar->op == 22 && scalar->count == 3)) && module[scalar->at + 2] == 32;
        // Alias members can have typed float or signed views. Tracking depends
        // only on coordinates, never on the texel's numeric representation.
        if (!numeric32 || module[image->at + 6] != 0 || module[image->at + 7] != 2)
            return fail("target must be a non-MS 32-bit numeric storage image");
        const uint32_t dim = module[image->at + 3], arrayed = module[image->at + 5];
        if (dim > 2 || arrayed > 1 || bool(arrayed) != target.arrayed ||
            (dim == 2 && arrayed) || (dim == 0 && target.height != 1) ||
            (dim < 2 && !arrayed && target.depth != 1))
            return fail("unsupported or inconsistent image dimensions");
        const TargetInfo info{target, dim, descriptor->second, image_type};
        if (!selected.emplace(info.variable, info).second)
            return fail("duplicate target image");
        const Binding mask_key{target.descriptor_set, target.mask_binding};
        if (descriptors.count(mask_key)) return fail("store mask binding collides with a descriptor");
        const auto [mask, inserted] = masks.emplace(mask_key, info);
        const auto& prior = mask->second;
        if (!inserted && (prior.dim != dim || prior.target.arrayed != target.arrayed ||
            prior.target.width != target.width || prior.target.height != target.height ||
            prior.target.depth != target.depth)) return fail("aliased masks have different views");
    }
    auto direct_image_variable = [&](uint32_t value) -> uint32_t {
        const auto load = loads.find(value);
        if (load == loads.end()) return 0;
        const auto var = variables.find(module[load->second.at + 3]);
        if (var == variables.end()) return 0;
        const Inst* ptr = get_type(module[var->second.at + 1]);
        if (!ptr || ptr->op != 32 || ptr->count != 4 || module[ptr->at + 2] != 0 ||
            module[ptr->at + 3] != module[load->second.at + 1]) return 0;
        const Inst* type = get_type(module[ptr->at + 3]);
        return type && type->op == 25 ? var->first : 0;
    };
    struct Write { TargetInfo target; uint32_t coordinate, scalar_type, components; bool signed_coord; };
    std::map<size_t, Write> writes;
    for (const auto& inst : instructions) {
        // OpImageTexelPointer takes a pointer to an image variable, unlike OpImageWrite.
        // Refuse all such operations conservatively; a pointer alias could hide a target.
        if (inst.op == 60) return fail("image atomics cannot be tracked by image-store masks");
        if (inst.op != 99) continue;
        if (inst.count < 4) return fail("invalid OpImageWrite");
        const uint32_t variable = direct_image_variable(module[inst.at + 1]);
        if (!variable) return fail("unsupported image-write provenance");
        const auto target = selected.find(variable);
        if (target == selected.end()) continue;
        const uint32_t coordinate = module[inst.at + 2];
        const auto coord_type = value_types.find(coordinate);
        if (coord_type == value_types.end()) return fail("unknown image coordinate result type");
        const Inst* scalar = get_type(coord_type->second);
        uint32_t components = 1, scalar_id = coord_type->second;
        if (scalar && scalar->op == 23 && scalar->count == 4) {
            components = module[scalar->at + 3];
            scalar_id = module[scalar->at + 2];
            scalar = get_type(scalar_id);
        }
        const uint32_t needed = target->second.dim + 1 + uint32_t(target->second.target.arrayed);
        if (!scalar || scalar->op != 21 || scalar->count != 4 ||
            module[scalar->at + 2] != 32 || module[scalar->at + 3] > 1 ||
            components < needed || components > 4)
            return fail("unsupported image coordinate type");
        writes.emplace(inst.at, Write{target->second, coordinate, scalar_id, components,
                                      module[scalar->at + 3] == 1});
    }
    // Compute a conservative upper bound before allocating IDs, so no intermediate
    // fresh ID can wrap and accidentally refer to a guest shader definition.
    const uint64_t needed_ids = 16ull + masks.size() * 8ull + writes.size() * 32ull;
    if (needed_ids > uint64_t(std::numeric_limits<uint32_t>::max()) - module[3])
        return fail("SPIR-V ID bound exhausted by store masks");
    uint32_t next_id = module[3];
    auto id = [&]() { return next_id++; };
    auto put = [](Words& words, uint32_t op, std::initializer_list<uint32_t> operands) {
        words.push_back((uint32_t(operands.size() + 1) << 16) | op);
        words.insert(words.end(), operands.begin(), operands.end());
    };
    Words annotations, declarations;
    if (!uint_type) { uint_type = id(); put(declarations, 21, {uint_type, 32, 0}); }
    if (!bool_type) { bool_type = id(); put(declarations, 20, {bool_type}); }
    for (const auto& [type_id, type] : types)
        if (type.op == 32 && type.count == 4 && module[type.at + 2] == 12 &&
            module[type.at + 3] == uint_type) uint_pointer = type_id;
    if (!uint_pointer) { uint_pointer = id(); put(declarations, 32, {uint_pointer, 12, uint_type}); }
    std::map<uint32_t, uint32_t> constants;
    auto constant = [&](uint32_t value) {
        const auto [entry, inserted] = constants.emplace(value, 0);
        if (inserted) { entry->second = id(); put(declarations, 43, {uint_type, entry->second, value}); }
        return entry->second;
    };
    const uint32_t zero = constant(0), one = constant(1);
    const uint32_t array_type = id(), struct_type = id(), struct_pointer = id();
    put(declarations, 29, {array_type, uint_type});
    put(declarations, 30, {struct_type, array_type});
    put(declarations, 32, {struct_pointer, 12, struct_type});
    put(annotations, 71, {array_type, 6, 4}); // ArrayStride
    put(annotations, 71, {struct_type, 2}); // Block
    put(annotations, 72, {struct_type, 0, 35, 0}); // Offset
    std::map<Binding, uint32_t> mask_variables;
    for (const auto& [key, info] : masks) {
        const uint32_t variable = id();
        mask_variables.emplace(key, variable);
        put(declarations, 59, {struct_pointer, variable, 12});
        put(annotations, 71, {variable, 34, key.first});
        put(annotations, 71, {variable, 33, key.second});
        constant(info.target.width);
        constant(info.target.height);
        constant(info.target.depth);
    }
    std::map<size_t, Words> prefixes;
    for (const auto& [at, write] : writes) {
        Words& prefix = prefixes[at];
        auto op2 = [&](uint32_t op, uint32_t type, uint32_t a, uint32_t b) {
            const uint32_t result = id(); put(prefix, op, {type, result, a, b}); return result;
        };
        auto select = [&](uint32_t condition, uint32_t yes, uint32_t no) {
            const uint32_t result = id();
            put(prefix, 169, {uint_type, result, condition, yes, no}); return result;
        };
        auto component = [&](uint32_t index) {
            uint32_t value = write.coordinate;
            if (write.components > 1) {
                value = id(); put(prefix, 81, {write.scalar_type, value, write.coordinate, index});
            }
            if (write.signed_coord) {
                const uint32_t converted = id(); put(prefix, 124, {uint_type, converted, value});
                value = converted;
            }
            return value;
        };
        const auto& target = write.target.target;
        const uint32_t x = component(0);
        const uint32_t y = write.target.dim >= 1 ? component(1) : zero;
        const uint32_t z = write.target.dim == 2 ? component(2) :
            target.arrayed ? component(write.target.dim + 1) : zero;
        uint32_t valid = op2(176, bool_type, x, constant(target.width));
        if (write.target.dim >= 1)
            valid = op2(167, bool_type, valid, op2(176, bool_type, y, constant(target.height)));
        if (write.target.dim == 2 || target.arrayed)
            valid = op2(167, bool_type, valid, op2(176, bool_type, z, constant(target.depth)));
        const uint32_t yz = op2(128, uint_type, y, op2(132, uint_type, constant(target.height), z));
        const uint32_t index = op2(128, uint_type, x, op2(132, uint_type, constant(target.width), yz));
        const uint32_t safe_index = select(valid, index, zero);
        const uint32_t mark = select(valid, one, zero);
        const uint32_t pointer = id(), atomic_result = id();
        const uint32_t variable = mask_variables.at({target.descriptor_set, target.mask_binding});
        put(prefix, 65, {uint_pointer, pointer, variable, zero, safe_index});
        put(prefix, 241, {uint_type, atomic_result, pointer, one, zero, mark});
    }
    StorageWriteMaskSpirvResult result;
    result.words.assign(module.begin(), module.begin() + 5);
    result.words[3] = next_id;
    result.instrumented_writes = uint32_t(writes.size());
    for (const auto& inst : instructions) {
        if (inst.at == first_type)
            result.words.insert(result.words.end(), annotations.begin(), annotations.end());
        if (inst.at == first_function)
            result.words.insert(result.words.end(), declarations.begin(), declarations.end());
        if (const auto prefix = prefixes.find(inst.at); prefix != prefixes.end())
            result.words.insert(result.words.end(), prefix->second.begin(), prefix->second.end());
        if (inst.op == 15 && module[1] >= 0x00010400u) {
            if (mask_variables.size() > 0xffffu - inst.count)
                return fail("entry-point interface exceeds instruction limit");
            result.words.push_back(((inst.count + uint32_t(mask_variables.size())) << 16) | 15);
            result.words.insert(result.words.end(), module.begin() + inst.at + 1,
                                module.begin() + inst.at + inst.count);
            for (const auto& [key, variable] : mask_variables) result.words.push_back(variable);
        } else {
            result.words.insert(result.words.end(), module.begin() + inst.at,
                                module.begin() + inst.at + inst.count);
        }
    }
    return result;
}

} // namespace prosper::frontend
