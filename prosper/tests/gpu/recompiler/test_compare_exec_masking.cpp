// #4016: architectural vector-compare predication is not strict scalar Boolean arithmetic.
// AMD RDNA2 ISA (70648), 3.9 / 12.9: inactive EXEC lanes produce zero compare-mask bits;
// gfx10 CMPX writes EXEC, not VCC. SOP2 12.1 instead specifies strict bitwise S_AND_B64.
// Primary reference: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
//
// These project-owned packets follow the gfx1030 encodings also used by test_rdna2_decode and
// test_rdna2_to_spirv. Decode checks pin each packet before checking its emitted SSA dataflow.
// A tagged cndmask reads the *actual destination*: finding an unrelated OpSelect cannot pass.
// No GPU is created. Wave64 here names the requested guest contract, not a claim about native
// fragment high-half execution or about a driver producing a particular value from poison.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace prosper::gpu;

namespace {
int failures = 0;

void check(bool condition, std::string_view label) {
    if (!condition) {
        std::printf("  [FAIL] %.*s\n", static_cast<int>(label.size()), label.data());
        ++failures;
    }
}

constexpr uint32_t kTypeBool = 20, kTypeInt = 21, kConstantTrue = 41, kConstantFalse = 42;
constexpr uint32_t kConstant = 43, kBitcast = 124, kLogicalAnd = 167, kSelect = 169;
constexpr uint32_t kIEqual = 170, kINotEqual = 171, kFOrdLessThan = 184;

struct Instruction {
    uint32_t opcode;
    std::vector<uint32_t> operands;
};

struct Module {
    std::vector<Instruction> instructions;
    bool valid = false;

    explicit Module(const std::vector<uint32_t>& words) {
        if (words.size() < 5 || words[0] != 0x07230203u) return;
        for (size_t offset = 5; offset < words.size();) {
            const size_t count = words[offset] >> 16u;
            if (!count || count > words.size() - offset) return;
            instructions.push_back({words[offset] & 0xffffu,
                {words.begin() + offset + 1, words.begin() + offset + count}});
            offset += count;
        }
        valid = true;
    }

    const Instruction* unique(uint32_t opcode) const {
        const Instruction* found = nullptr;
        for (const auto& instruction : instructions) {
            if (instruction.opcode != opcode) continue;
            if (found) return nullptr;
            found = &instruction;
        }
        return found;
    }

    // Only the typed results this oracle follows, not a guessed result position for every opcode.
    const Instruction* definition(uint32_t id) const {
        if (!id) return nullptr;
        for (const auto& instruction : instructions) {
            switch (instruction.opcode) {
                case kConstantTrue: case kConstantFalse: case kConstant: case kBitcast:
                case kLogicalAnd: case kSelect: case kIEqual: case kINotEqual:
                case kFOrdLessThan:
                    if (instruction.operands.size() >= 2 && instruction.operands[1] == id)
                        return &instruction;
                    break;
                default: break;
            }
        }
        return nullptr;
    }

    bool boolean_constant(uint32_t id, uint32_t type, bool value) const {
        const auto* instruction = definition(id);
        return instruction && instruction->opcode == (value ? kConstantTrue : kConstantFalse) &&
               instruction->operands.size() == 2 && instruction->operands[0] == type;
    }

    bool uint_constant(uint32_t id, uint32_t value) const {
        const auto* instruction = definition(id);
        return instruction && instruction->opcode == kConstant &&
               instruction->operands.size() == 3 && instruction->operands[2] == value &&
               uint_type(instruction->operands[0]);
    }

    bool uint_type(uint32_t type) const {
        for (const auto& instruction : instructions)
            if (instruction.opcode == kTypeInt && instruction.operands.size() == 3 &&
                instruction.operands[0] == type && instruction.operands[1] == 32 &&
                instruction.operands[2] == 0) return true;
        return false;
    }

    bool bitcast_of(uint32_t id, uint32_t raw) const {
        const auto* instruction = definition(id);
        return instruction && instruction->opcode == kBitcast &&
               instruction->operands.size() == 3 && instruction->operands[2] == raw;
    }

    uint32_t result(const Instruction* instruction) const {
        return instruction && instruction->operands.size() >= 2 ? instruction->operands[1] : 0;
    }

    // Exact typed Select(oldEXEC, actualCmp, false), with an independently identified compare.
    uint32_t mask(uint32_t comparison, uint32_t old_exec, bool full_exec) const {
        const auto* compare = definition(comparison);
        if (!compare || compare->operands.size() != 4) return 0;
        const uint32_t type = compare->operands[0];
        uint32_t found = 0;
        for (const auto& instruction : instructions) {
            const auto& o = instruction.operands;
            if (instruction.opcode != kSelect || o.size() != 5 || o[0] != type ||
                o[3] != comparison || !boolean_constant(o[4], type, false)) continue;
            if (full_exec ? !boolean_constant(o[2], type, true) : o[2] != old_exec) continue;
            if (found) return 0;
            found = o[1];
        }
        return found;
    }

    uint32_t tagged_condition(uint32_t true_value) const {
        uint32_t found = 0;
        for (const auto& instruction : instructions) {
            const auto& o = instruction.operands;
            if (instruction.opcode != kSelect || o.size() != 5 ||
                !uint_type(o[0]) ||
                !uint_constant(o[3], true_value) || !uint_constant(o[4], 0)) continue;
            if (found) return 0;
            found = o[2];
        }
        return found;
    }
};

enum class Stage { Compute, Fragment };
enum class Encoding { E32, E64, Sdwa };

struct Case {
    const char* name;
    Encoding encoding;
    std::array<uint32_t, 2> packet;
    size_t packet_words;
    uint32_t destination;
    bool cmpx;
    bool preserves_vcc;
    bool wave32_only;
};

constexpr std::array cases = {
    Case{"vopc_e32_vcc", Encoding::E32, {0x7c020300u, 0}, 1, 106, false, false, false},
    Case{"vopc_e64_s8", Encoding::E64, {0xd4010008u, 0x00020300u}, 2, 8, false, true, false},
    Case{"vopc_e64_vcc", Encoding::E64, {0xd401006au, 0x00020300u}, 2, 106, false, false, false},
    Case{"vopc_sdwa_s8", Encoding::Sdwa, {0x7c0202f9u, 0x06068800u}, 2, 8, false, true, false},
    Case{"vopc_sdwa_vcc", Encoding::Sdwa, {0x7c0202f9u, 0x0606ea00u}, 2, 106, false, false, false},
    Case{"vopc_sdwa_vcc_hi", Encoding::Sdwa, {0x7c0202f9u, 0x0606eb00u}, 2, 107, false, true, true},
    Case{"cmpx_e32", Encoding::E32, {0x7c220300u, 0}, 1, 106, true, true, false},
    Case{"cmpx_e64", Encoding::E64, {0xd411007eu, 0x00020300u}, 2, 126, true, true, false},
    Case{"cmpx_sdwa", Encoding::Sdwa, {0x7c2202f9u, 0x06060000u}, 2, 106, true, true, false},
};

void packet_controls() {
    for (const auto& test : cases) {
        const auto decoded = rdna2_decode_one(test.packet.data(), test.packet_words);
        const std::string label = std::string(test.name) + " decoded ISA contract";
        check(decoded.fmt == Rdna2Format::VOPC && decoded.opcode == (test.cmpx ? 0x11u : 1u) &&
              decoded.len_dwords == test.packet_words && !decoded.has_modifier &&
              decoded.n_src == 2 && decoded.src[0].kind == OperandKind::VGPR &&
              decoded.src[0].value == 0 && decoded.src[1].kind == OperandKind::VGPR &&
              decoded.src[1].value == 1 && decoded.dst.value == static_cast<int>(test.destination) &&
              decoded.dst.kind == (test.encoding == Encoding::E32 ||
                  (test.cmpx && test.encoding == Encoding::Sdwa) ? OperandKind::Special : OperandKind::SGPR) &&
              decoded.has_sdwa == (test.encoding == Encoding::Sdwa) &&
              decoded.sdwa_src0_sel == 6 && decoded.sdwa_src1_sel == 6 &&
              vopc_is_cmpx(decoded.opcode) == test.cmpx, label);
    }
    constexpr uint32_t seed[] = {0x7da40080u}; // v_cmpx_eq_u32 0, v0: narrows EXEC, preserves VCC
    const auto decoded = rdna2_decode_one(seed, 1);
    check(decoded.fmt == Rdna2Format::VOPC && decoded.opcode == 0xd2 &&
          decoded.src[0].kind == OperandKind::InlineInt && decoded.src[0].value == 0 &&
          decoded.src[1].kind == OperandKind::VGPR && decoded.src[1].value == 0,
          "independent old-EXEC producer is CMPX_EQ_U32 0,v0");
}

void append_consumer(std::vector<uint32_t>& program, uint32_t destination, uint32_t marker) {
    // v_cndmask_b32_e64 v[8 or 9], 0, marker, exact-mask-register
    const uint32_t packet[] = {0xd5010000u | (marker == 7 ? 8u : 9u),
        0x80u | ((0x80u + marker) << 9u) | (destination << 18u)};
    const auto decoded = rdna2_decode_one(packet, 2);
    check(decoded.fmt == Rdna2Format::VOP3 && decoded.opcode == 0x101 &&
          decoded.src[0].kind == OperandKind::InlineInt && decoded.src[0].value == 0 &&
          decoded.src[1].kind == OperandKind::InlineInt &&
          decoded.src[1].value == static_cast<int>(marker) &&
          decoded.src[2].value == static_cast<int>(destination), "tagged consumer reads exact mask destination");
    program.insert(program.end(), std::begin(packet), std::end(packet));
}

std::vector<uint32_t> compile(std::vector<uint32_t> program, Stage stage, uint32_t wave_size) {
    if (stage == Stage::Fragment) {
        // The actual tagged consumers feed MRT0. FragCoord.x supplies an initialized dynamic v0.
        program.insert(program.end(), {0xf800180fu, 0x09080908u}); // exp mrt0 v8,v9,v8,v9 done vm
    }
    program.push_back(0xbf810000u);
    if (stage == Stage::Compute) {
        ComputeShaderConfig config;
        config.wave_size = wave_size;
        return recompile_compute(program.data(), program.size(), nullptr, config);
    }
    PixelSystemInputMapping inputs;
    inputs.ena = inputs.addr = 1u << 8u;
    return recompile_fragment(program.data(), program.size(), nullptr, &inputs, UINT32_MAX,
                              nullptr, wave_size == 32);
}

void dump(const char* directory, const std::string& name, const std::vector<uint32_t>& words) {
    if (!directory || words.empty()) return;
    std::ofstream output(std::filesystem::path(directory) / (name + ".spv"), std::ios::binary);
    const auto bytes = std::as_bytes(std::span(words));
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    output.close();
    check(static_cast<bool>(output), name + " dump serialization");
}

std::string name_for(Stage stage, uint32_t wave_size, std::string_view arm) {
    return std::string(stage == Stage::Compute ? "compute" : "fragment") + "_wave" +
           std::to_string(wave_size) + "_" + std::string(arm);
}

void compare_control(const Case& test, Stage stage, uint32_t wave_size, bool narrowed,
                     const char* directory) {
    const std::string name = name_for(stage, wave_size,
        std::string(test.name) + (narrowed ? "_narrowed" : "_full"));
    std::vector<uint32_t> program = {
        0x7e0202f2u, // v_mov_b32 v1,1.0 before any EXEC narrowing
        0x7d8a0081u, // v_cmp_ne_u32_e32 vcc,1,v0: distinct preexisting VCC
    };
    if (narrowed) program.push_back(0x7da40080u); // independent CMPX_EQ_U32 old EXEC
    program.insert(program.end(), test.packet.begin(), test.packet.begin() + test.packet_words);
    if (test.cmpx) program.push_back(0xbe84047eu); // s_mov_b64 s[4:5],exec: observe CMPX destination
    program.push_back(0xbefe04c1u); // s_mov_b64 exec,-1: no EXEC predicate around the tagged readers
    append_consumer(program, test.cmpx ? 4u : test.destination, 7);
    append_consumer(program, 106, 9); // observe preexisting VCC after CMPX / explicit non-VCC SDST
    const auto words = compile(std::move(program), stage, wave_size);
    const Module module(words);
    check(!words.empty() && module.valid, name + " module emitted");
    if (!module.valid) return;
    dump(directory, name, words);
    const auto* compare = module.unique(kFOrdLessThan);
    const auto* old_vcc = module.unique(kINotEqual);
    const auto* bool_type = module.unique(kTypeBool);
    const uint32_t comparison = module.result(compare);
    check(compare && compare->operands.size() == 4 && bool_type &&
          bool_type->operands.size() == 1 && compare->operands[0] == bool_type->operands[0] &&
          old_vcc && old_vcc->operands.size() == 4 &&
          module.bitcast_of(compare->operands[2], old_vcc->operands[3]),
          name + " actual float comparison of initialized v0");
    if (!comparison || compare->operands.size() != 4 || !old_vcc ||
        old_vcc->operands.size() != 4) return;
    const auto* rhs = module.definition(compare->operands[3]);
    check(rhs && rhs->opcode == kBitcast && rhs->operands.size() == 3 &&
          module.uint_constant(rhs->operands[2], 0x3f800000u), name + " actual comparison rhs is v1=1.0");

    uint32_t old_exec = 0;
    if (narrowed) {
        const auto* seed = module.unique(kIEqual);
        const uint32_t seed_result = module.result(seed);
        check(seed && seed->operands.size() == 4 && module.uint_constant(seed->operands[2], 0) &&
              old_vcc && seed->operands[3] == old_vcc->operands[3], name + " old EXEC compare uses same v0");
        old_exec = module.mask(seed_result, 0, true);
        check(old_exec != 0, name + " old EXEC is exact typed Select(true,seedCmp,false)");
    }
    const uint32_t expected = narrowed || test.cmpx ? module.mask(comparison, old_exec, !narrowed) : comparison;
    check(expected != 0, name + " exact Select(oldEXEC,actualCmp,typedFalse) or full-EXEC fast path");
    check(expected != 0 && module.tagged_condition(7) == expected,
          name + " architectural destination reaches its tagged consumer");
    if (!narrowed && !test.cmpx)
        check(module.mask(comparison, 0, true) == 0, name + " full EXEC does not invent a masked compare");
    const uint32_t expected_vcc = test.preserves_vcc ? module.result(old_vcc) : expected;
    check(expected_vcc != 0 && module.tagged_condition(9) == expected_vcc,
          name + (test.preserves_vcc ? " preserves distinct preexisting VCC" : " publishes new VCC"));
}

void scalar_and_control(Stage stage, uint32_t wave_size, const char* directory) {
    const std::string name = name_for(stage, wave_size, "scalar_and_b64_strict");
    std::vector<uint32_t> program = {
        0x7e0202f2u, // v_mov_b32 v1,1.0
        0x7d8a0081u, // v_cmp_ne_u32 vcc,1,v0
        0xd4010008u, 0x00020300u, // full-EXEC v_cmp_lt_f32_e64 s[8:9],v0,v1
        0x87ea086au, // s_and_b64 vcc,vcc,s[8:9]: unpredicated scalar mask arithmetic
    };
    const auto scalar = rdna2_decode_one(program.data() + 4, 1);
    check(scalar.fmt == Rdna2Format::SOP2 && scalar.opcode == 0x0f &&
          scalar.dst.value == 106 && scalar.src[0].value == 106 && scalar.src[1].value == 8,
          name + " decoded scalar strict-AND contract");
    append_consumer(program, 106, 7);
    append_consumer(program, 106, 9);
    const auto words = compile(std::move(program), stage, wave_size);
    const Module module(words);
    check(!words.empty() && module.valid, name + " module emitted");
    if (!module.valid) return;
    dump(directory, name, words);
    const uint32_t condition = module.tagged_condition(7);
    const auto* actual = module.definition(condition);
    check(actual && actual->opcode == kLogicalAnd && actual->operands.size() == 4 &&
          actual->operands[2] == module.result(module.unique(kINotEqual)) &&
          actual->operands[3] == module.result(module.unique(kFOrdLessThan)) &&
          module.tagged_condition(9) == condition, name + " exact scalar operands remain strict OpLogicalAnd");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && !(argc == 3 && std::string_view(argv[1]) == "--dump")) return 2;
    const char* directory = argc == 3 ? argv[2] : nullptr;
    packet_controls();
    size_t count = 0;
    for (const auto stage : {Stage::Compute, Stage::Fragment}) {
        for (const uint32_t wave_size : {32u, 64u}) {
            for (const auto& test : cases) {
                if (test.wave32_only && wave_size != 32) continue;
                for (const bool narrowed : {false, true}) {
                    compare_control(test, stage, wave_size, narrowed, directory);
                    ++count;
                }
            }
            scalar_and_control(stage, wave_size, directory);
            ++count;
        }
    }
    std::printf("== compare_exec_masking: %zu modules, %d failures ==\n", count, failures);
    return failures ? 1 : 0;
}
