// The ONE definition of each scalar-ALU data op (ir/ops/scalar_alu.hpp) is evaluated by the constant
// folder and emitted as SPIR-V by the recompiler. This test makes the two answer the same inputs and
// compares: for every definition, a real RDNA2 program computes the op on many operand pairs through
// recompile_compute, the integer SPIR-V interpreter executes the module, and each stored D and SCC
// must equal eval_scalar_op() on the same operands. It also requires every IR opcode to occur in
// some definition, so no opcode's evaluation goes unchecked against SPIR-V (#2514, ADR 0012).
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/ops/scalar_alu.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <set>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {

constexpr uint32_t kA = 20, kB = 21, kD = 22, kScc = 23;   // SGPRs the program uses
constexpr uint32_t kSentinel = 0xdeadbeefu;

struct Vector {
    uint32_t a, b;   // S0 and S1 (SOPK: the SDST value and SIMM16)
    bool scc;        // incoming SCC, for the ops that read it
};

struct Program {
    std::vector<uint32_t> code;
    uint32_t slots = 0;
};

void mov_literal(std::vector<uint32_t>& p, uint32_t sgpr, uint32_t value) {
    p.insert(p.end(), {0xbe8003ffu | (sgpr << 16), value});   // s_mov_b32 sN, literal
}

// v0 = slot; v3 = sN; buffer_store_dword v3 at index v0 (binding 3, V# in s[8:11]).
void store_sgpr(Program& out, uint32_t sgpr) {
    out.code.insert(out.code.end(), {0x7e0002ffu, out.slots++,   // v_mov_b32 v0, literal
                                     0x7e060200u | sgpr,         // v_mov_b32 v3, sN
                                     0xe0702000u, 0x80020300u});
}

uint32_t encode(const ScalarOpDef& def, uint32_t simm16) {
    switch (def.fmt) {
        case Rdna2Format::SOP2: return 0x80000000u | (def.opcode << 23) | (kD << 16) | (kB << 8) | kA;
        case Rdna2Format::SOP1: return 0xbe800000u | (kD << 16) | (def.opcode << 8) | kA;
        case Rdna2Format::SOPC: return 0xbf000000u | (def.opcode << 16) | (kB << 8) | kA;
        case Rdna2Format::SOPK: return 0xb0000000u | (def.opcode << 23) | (kD << 16) | simm16;
        default: ADD_FAILURE() << "format"; return 0;
    }
}

Program program(const ScalarOpDef& def, const std::vector<Vector>& vectors) {
    Program out;
    for (const Vector& v : vectors) {
        mov_literal(out.code, kA, v.a);
        mov_literal(out.code, kB, v.b);
        // SOPK reads SDST; a compare leaves it alone, so give it a known value either way.
        mov_literal(out.code, kD, def.fmt == Rdna2Format::SOPK ? v.a : 0u);
        // Incoming SCC, set for every op: one that does not write SCC must leave this value.
        out.code.push_back(0xbf000000u | ((v.scc ? 0x06u : 0x07u) << 16) | (kA << 8) | kA);
        out.code.push_back(encode(def, v.b & 0xffffu));
        out.code.push_back(0x80000000u | (0x0au << 23) | (kScc << 16) | (0x80u << 8) | 0x81u);   // s_cselect_b32 s23, 1, 0
        store_sgpr(out, kD);
        store_sgpr(out, kScc);
    }
    out.code.push_back(0xbf810000u);   // s_endpgm
    return out;
}

std::vector<Vector> vectors() {
    const uint32_t edges[] = {0u, 1u, 2u, 15u, 16u, 31u, 32u, 33u, 0x7fffu, 0x8000u, 0xffffu,
                              0x10000u, 0x7fffffffu, 0x80000000u, 0xfffffffeu, 0xffffffffu};
    std::vector<Vector> out;
    uint32_t seed = 0x2514u;
    const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
    for (uint32_t a : edges)
        for (uint32_t b : {0u, 1u, 31u, 33u, 0x80000000u, 0xffffffffu, a})
            out.push_back({a, b, (next() & 1u) != 0});
    for (int i = 0; i < 48; ++i) {
        const uint32_t a = next(), b = next();
        // Small S1 values reach the bitfield/shift fields that random words rarely land in.
        out.push_back({a, i % 3 ? b : (b & 0x003f003fu), (next() & 1u) != 0});
    }
    return out;
}

ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.threads_x = 1;
    c.local_y = c.local_z = c.threads_y = c.threads_z = 1;
    c.wave_size = 64;
    c.dispatch_launch.known = ComputeDispatchLaunchState::kAll;
    c.dispatch_launch.initiator = 1u;
    c.dispatch_launch.num_thread = {1, 1, 1};
    c.dispatch_launch.dimensions = {1, 1, 1};
    return c;
}

ShaderResourceTable table(uint32_t slots) {
    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.binding = 3;
    r.sgpr_base = 8;
    r.stride = 4;
    r.size = slots * 4;
    ShaderResourceTable t;
    t.resources.push_back(r);
    return t;
}

uint64_t address() {
    static std::atomic<uint64_t> next{0x4931000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}

std::vector<const ScalarOpDef*> all_defs() {
    std::vector<const ScalarOpDef*> out;
    for (auto defs : {scalar_alu_defs(), scalar_compare_defs()})
        for (const ScalarOpDef& def : defs) out.push_back(&def);
    return out;
}

} // namespace

TEST(IrScalarAluExecution, EveryDefinitionEvaluatesWhatItsSpirvComputes) {
    const auto inputs = vectors();
    for (const ScalarOpDef* def : all_defs()) {
        const char* name = rdna2_instruction_effects(def->fmt, def->opcode).name;
        SCOPED_TRACE(name ? name : "?");
        const Program p = program(*def, inputs);
        const uint64_t addr = address();
        const ShaderResourceTable t = table(p.slots);
        const auto module = recompile_compute(p.code.data(), p.code.size(), &t, config(),
                                              {RecompileDiagnosticStage::Compute, addr});
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(addr);
        bpermute_oracle::Interpreter vm(module);
        const auto out = vm.run_buffers(1, {{3, std::vector<uint32_t>(p.slots, kSentinel)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), p.slots);
        for (size_t i = 0; i < inputs.size(); ++i) {
            const Vector& v = inputs[i];
            const bool sopk = def->fmt == Rdna2Format::SOPK;
            Rdna2Inst encoded{};
            encoded.simm16 = static_cast<int16_t>(v.b & 0xffffu);
            const uint32_t s1 = sopk ? scalar_op_simm16(*def, encoded) : v.b;
            const ScalarOpValue want = eval_scalar_op(*def, v.a, s1, uint32_t{v.scc});
            SCOPED_TRACE(::testing::Message() << std::hex << "a=0x" << v.a << " b=0x" << v.b
                                              << " scc=" << v.scc);
            ASSERT_TRUE(want.d || want.scc) << "the op evaluated to nothing";
            // A compare leaves its SGPR as initialized.
            EXPECT_EQ(out[2 * i], want.d ? *want.d : (sopk ? v.a : 0u));
            // SCC: the op's own result, or the incoming SCC an op without an SCC write kept.
            EXPECT_EQ(out[2 * i + 1], want.scc ? *want.scc : uint32_t{v.scc});
        }
    }
}

TEST(IrScalarAluExecution, EveryIrOpcodeIsExercised) {
    std::set<Opcode> seen;
    for (const ScalarOpDef* def : all_defs()) {
        Block block;
        Builder ir(block);
        const Value s0 = ir.input(Type::U32), s1 = ir.input(Type::U32), scc = ir.input(Type::Bool);
        def->build(ir, {s0, s1, scc});
        for (const Inst& inst : block.insts()) seen.insert(inst.op);
    }
    for (uint32_t op = 0; op <= static_cast<uint32_t>(Opcode::SGe); ++op)
        EXPECT_TRUE(seen.contains(static_cast<Opcode>(op))) << opcode_info(static_cast<Opcode>(op)).name;
}

// Which ops write SCC, and which read it, is the instruction-effects table's answer: a definition
// that disagreed would compute SCC the hardware leaves alone, or leave a stale one.
TEST(IrScalarAluExecution, DefinitionsAgreeWithTheEffectsTable) {
    for (const ScalarOpDef* def : all_defs()) {
        const auto& effects = rdna2_instruction_effects(def->fmt, def->opcode);
        SCOPED_TRACE(effects.name ? effects.name : "?");
        ASSERT_TRUE(effects.known());
        const ScalarOpValue v = eval_scalar_op(*def, 1u, 2u, 1u);
        EXPECT_EQ(v.scc.has_value(), rdna2_writes_scc(def->fmt, def->opcode) == Rdna2Answer::Yes);
        EXPECT_EQ(v.d.has_value(), effects.sdst_dwords == 1u);
    }
}
