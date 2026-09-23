// A resolved vertex fetch binds a V# whose known OFFSET/SOFFSET is already in its base.
// With stride zero, the shader's idxen element contributes no additional byte offset.
// Keep the otherwise identical Shader-mode fetch as a control: it binds the unshifted
// descriptor and must still add SOFFSET at execution time.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

using namespace prosper::gpu;

struct Definition {
    uint16_t opcode = 0;
    std::vector<uint32_t> operands;
};

struct FetchAddress {
    bool valid = false;
    uint32_t byte_address_id = 0;
    uint32_t vertex_index_variable = 0;
    std::map<uint32_t, Definition> definitions;
};

static FetchAddress fetch_address(const std::vector<uint32_t>& spv, uint32_t binding) {
    constexpr uint16_t OpConstant = 43, OpLoad = 61, OpAccessChain = 65;
    constexpr uint16_t OpDecorate = 71, OpBitcast = 124, OpIAdd = 128;
    constexpr uint16_t OpShiftRightLogical = 194;
    constexpr uint32_t DecorationBuiltIn = 11, DecorationBinding = 33;
    constexpr uint32_t BuiltInVertexIndex = 42;
    FetchAddress out;
    if (spv.size() < 5 || spv[0] != 0x07230203u) return out;
    uint32_t buffer_variable = 0;
    for (size_t at = 5; at < spv.size();) {
        const uint32_t count = spv[at] >> 16;
        const uint16_t opcode = static_cast<uint16_t>(spv[at]);
        if (!count || at + count > spv.size()) return {};
        if (opcode == OpDecorate && count >= 4 &&
            spv[at + 2] == DecorationBinding && spv[at + 3] == binding)
            buffer_variable = spv[at + 1];
        if (opcode == OpDecorate && count >= 4 &&
            spv[at + 2] == DecorationBuiltIn && spv[at + 3] == BuiltInVertexIndex)
            out.vertex_index_variable = spv[at + 1];
        if ((opcode == OpConstant && count == 4) ||
            ((opcode == OpLoad || opcode == OpBitcast) && count == 4) ||
            ((opcode == OpIAdd || opcode == OpShiftRightLogical) && count == 5)) {
            Definition def;
            def.opcode = opcode;
            def.operands.assign(spv.begin() + at + 3, spv.begin() + at + count);
            out.definitions[spv[at + 2]] = std::move(def);
        }
        at += count;
    }
    if (!buffer_variable) return {};
    for (size_t at = 5; at < spv.size();) {
        const uint32_t count = spv[at] >> 16;
        const uint16_t opcode = static_cast<uint16_t>(spv[at]);
        if (opcode == OpAccessChain && count >= 6 && spv[at + 3] == buffer_variable) {
            const uint32_t index_id = spv[at + count - 1];
            const auto shift = out.definitions.find(index_id);
            if (shift == out.definitions.end() || shift->second.opcode != OpShiftRightLogical ||
                shift->second.operands.size() != 2)
                return {};
            out.byte_address_id = shift->second.operands[0];
            out.valid = true;
            return out;
        }
        at += count;
    }
    return {};
}

static bool contains_literal(const FetchAddress& address, uint32_t id,
                             uint32_t literal, unsigned depth = 0) {
    constexpr uint16_t OpConstant = 43, OpIAdd = 128;
    if (depth > 8) return false;
    const auto it = address.definitions.find(id);
    if (it == address.definitions.end()) return false;
    const Definition& def = it->second;
    if (def.opcode == OpConstant)
        return def.operands.size() == 1 && def.operands[0] == literal;
    if (def.opcode != OpIAdd || def.operands.size() != 2) return false;
    return contains_literal(address, def.operands[0], literal, depth + 1) ||
           contains_literal(address, def.operands[1], literal, depth + 1);
}

static bool is_literal(const FetchAddress& address, uint32_t id, uint32_t literal) {
    constexpr uint16_t OpConstant = 43;
    const auto it = address.definitions.find(id);
    return it != address.definitions.end() && it->second.opcode == OpConstant &&
           it->second.operands.size() == 1 && it->second.operands[0] == literal;
}

static bool contains_vertex_index(const FetchAddress& address, uint32_t id,
                                  unsigned depth = 0) {
    constexpr uint16_t OpLoad = 61, OpBitcast = 124, OpIAdd = 128;
    if (depth > 8 || !address.vertex_index_variable) return false;
    const auto it = address.definitions.find(id);
    if (it == address.definitions.end()) return false;
    const Definition& def = it->second;
    if (def.opcode == OpLoad)
        return def.operands.size() == 1 && def.operands[0] == address.vertex_index_variable;
    if (def.opcode == OpBitcast && def.operands.size() == 1)
        return contains_vertex_index(address, def.operands[0], depth + 1);
    if (def.opcode == OpIAdd && def.operands.size() == 2)
        return contains_vertex_index(address, def.operands[0], depth + 1) ||
               contains_vertex_index(address, def.operands[1], depth + 1);
    return false;
}

int main() {
    // s_movk_i32 s5, 0xe00; buffer_load_format_xyzw v[1:4], v0, s[0:3], s5 idxen;
    // exp pos0 v1..v4 done; s_endpgm. Same stride-zero fetch shape as draw 281.
    constexpr std::array<uint32_t, 6> code = {
        0xb0050e00u, 0xe00c2000u, 0x05000100u,
        0xf80008cfu, 0x04030201u, 0xbf810000u,
    };
    ShaderResourceTable table;
    ShaderResource buffer{};
    buffer.cls = ResourceClass::VertexBuffer;
    buffer.format = DataFormat::Unorm8;
    buffer.num_components = 4;
    buffer.binding = 8;
    buffer.size = 4;
    buffer.stride = 0;
    buffer.fetch_pc = 1;
    buffer.fetch_index_mode = VertexFetchIndexMode::Vertex;
    table.resources.push_back(buffer);

    const auto folded_spv = recompile_vertex(code.data(), code.size(), &table);
    const auto folded = fetch_address(folded_spv, buffer.binding);
    if (!folded.valid || !is_literal(folded, folded.byte_address_id, 0) ||
        contains_literal(folded, folded.byte_address_id, 0xe00u)) {
        std::puts("[FAIL] folded stride-zero vertex fetch must address byte zero of the shifted V#");
        return 1;
    }
    std::puts("[ok] folded stride-zero vertex fetch uses the shifted V# without adding SOFFSET");

    table.resources[0].fetch_index_mode = VertexFetchIndexMode::Shader;
    const auto shader_spv = recompile_vertex(code.data(), code.size(), &table);
    const auto shader = fetch_address(shader_spv, buffer.binding);
    if (!shader.valid || !contains_literal(shader, shader.byte_address_id, 0xe00u)) {
        std::puts("[FAIL] shader-mode stride-zero fetch must retain SOFFSET on the unshifted V#");
        return 1;
    }
    std::puts("[ok] shader-mode stride-zero fetch retains SOFFSET");

    // OFFEN supplies a potentially unaligned second VADDR VGPR. Folded stride-zero
    // OFFEN is deliberately excluded from the new specialization until packed-format
    // loads handle that byte address. Guard the existing address expression in both modes.
    constexpr std::array<uint32_t, 7> offen_code = {
        0xb0050e00u,               // s_movk_i32 s5, 0xe00
        0x7e020300u,               // v_mov_b32 v1, v0 (VertexIndex)
        0xe00c3000u, 0x05000200u,  // buffer_load_format_xyzw v[2:5], v[0:1], s[0:3], s5 idxen offen
        0xf80008cfu, 0x05040302u,  // exp pos0 v2..v5 done
        0xbf810000u,
    };
    table.resources[0].fetch_pc = 2;
    table.resources[0].fetch_index_mode = VertexFetchIndexMode::Vertex;
    const auto folded_offen = fetch_address(
        recompile_vertex(offen_code.data(), offen_code.size(), &table), buffer.binding);
    if (!folded_offen.valid ||
        !contains_vertex_index(folded_offen, folded_offen.byte_address_id) ||
        !contains_literal(folded_offen, folded_offen.byte_address_id, 0xe00u)) {
        std::puts("[FAIL] folded stride-zero idxen+offen must retain its pre-existing address path");
        return 1;
    }
    std::puts("[ok] folded stride-zero idxen+offen is excluded from the new specialization");

    table.resources[0].fetch_index_mode = VertexFetchIndexMode::Shader;
    const auto shader_offen = fetch_address(
        recompile_vertex(offen_code.data(), offen_code.size(), &table), buffer.binding);
    if (!shader_offen.valid ||
        !contains_vertex_index(shader_offen, shader_offen.byte_address_id) ||
        !contains_literal(shader_offen, shader_offen.byte_address_id, 0xe00u)) {
        std::puts("[FAIL] shader-mode stride-zero idxen+offen fetch must keep VADDR[1] and SOFFSET");
        return 1;
    }
    std::puts("[ok] shader-mode stride-zero idxen+offen fetch retains VADDR[1] and SOFFSET");
    return 0;
}
