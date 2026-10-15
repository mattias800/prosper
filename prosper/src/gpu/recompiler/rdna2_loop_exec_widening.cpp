// See rdna2_loop_exec_widening.hpp.
#include "gpu/recompiler/rdna2_loop_exec_widening.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {
namespace {

// Two independent facts per scalar word: subset of the low or high physical EXEC bound.
// A copied high word can stay high-bounded in an even SGPR; register parity is not authority.
using Words = std::bitset<256>;
size_t fact(int reg, uint32_t half) {
    return static_cast<size_t>(reg) + 128u * static_cast<size_t>(half);
}
constexpr int kExecLo = 126;
constexpr int kVccLo = 106;

Words exec_words() {
    Words words;
    words.set(fact(kExecLo, 0));
    words.set(fact(kExecLo + 1, 1));
    return words;
}

bool branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02u && in.opcode <= 0x09u &&
           in.opcode != 0x03u;
}

bool unmodelled_control(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::Unknown) return true;
    if (in.fmt == Rdna2Format::SOP1)
        return in.opcode == kSop1OpcodeSetpcB64 || in.opcode == kSop1OpcodeSwappcB64 ||
               in.opcode == kSop1OpcodeRfeB64;
    if (in.fmt == Rdna2Format::SOPK)
        return in.opcode == kSopkOpcodeCallB64 || in.opcode >= kSopkOpcodeSubvectorLoopBegin;
    return in.fmt == Rdna2Format::SOPP && !branch(in) && !sopp_is_noop(in) && !in.is_end &&
           in.opcode != 0x0au;
}

// One source word is bounded by this particular physical half, or is zero.
bool within(const Words& s, const Operand& op, uint32_t word, uint32_t half) {
    if (op.kind == OperandKind::InlineInt) return op.value == 0;
    if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return false;
    const int reg = op.value + static_cast<int>(word);
    return reg >= 0 && reg < 128 && s.test(fact(reg, half));
}

struct Effect {
    Words out;
    bool exec_ok = true;
};

// CONFIDENCE: HIGH -- each rule is the ISA 70648 definition of the instruction's result read
// bitwise: a lane outside every within-bound input is outside AND, OR, XOR, ANDN2 (of its first
// operand), CSELECT and a copy; a compare writes EXEC & result; a save copies EXEC.
Effect transfer(const Rdna2Inst& in, const Words& s) {
    Effect e{s};
    Words written;
    const auto record = [&](int base, uint32_t width) {
        for (uint32_t word = 0; word < width; ++word) {
            const int reg = base + static_cast<int>(word);
            if (reg >= 0 && reg < 128) {
                written.set(fact(reg, 0));
                written.set(fact(reg, 1));
            }
        }
    };
    for_each_scalar_write(in, record);
    if (in.fmt == Rdna2Format::SOP2) record(in.dst.value, rdna2_sop2_dest_dwords(in.opcode));
    const bool vopc = in.fmt == Rdna2Format::VOPC;
    if (vopc && !vopc_is_cmpx(in.opcode) && in.dst.kind == OperandKind::Special) record(kVccLo, 2);
    if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28u && in.opcode <= 0x2au) record(kVccLo, 2);
    // A B32 saveexec/wrexec form writes EXEC_LO only; every other implicit EXEC writer both words.
    const bool b32_exec = in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x3cu && in.opcode <= 0x47u;
    const uint32_t exec_width = b32_exec ? 1u : 2u;
    const bool exec_write =
        rdna2_instruction_may_change_exec(in) ||
        (written.test(fact(kExecLo, 0)) || written.test(fact(kExecLo, 1))) ||
        (written.test(fact(kExecLo + 1, 0)) || written.test(fact(kExecLo + 1, 1)));
    if (exec_write && !(written.test(fact(kExecLo, 0)) || written.test(fact(kExecLo, 1))) &&
        !(written.test(fact(kExecLo + 1, 0)) || written.test(fact(kExecLo + 1, 1))))
        record(kExecLo, exec_width);
    Words bounded;
    const auto mark = [&](int reg, uint32_t half, bool ok) {
        if (ok && reg >= 0 && reg < 128) bounded.set(fact(reg, half));
    };
    for (uint32_t half = 0; half < 2; ++half) {
        if (vopc) {
            // Each physical result word is bounded by the corresponding word of incoming EXEC.
            for (uint32_t word = 0; word < 2; ++word) {
                const bool active = s.test(fact(kExecLo + static_cast<int>(word), half));
                if (vopc_is_cmpx(in.opcode)) {
                    mark(kExecLo + static_cast<int>(word), half, active);
                } else {
                    for_each_scalar_write(in, [&](int base, uint32_t width) {
                        if (word < width) mark(base + static_cast<int>(word), half, active);
                    });
                    if (in.dst.kind == OperandKind::Special)
                        mark(kVccLo + static_cast<int>(word), half, active);
                }
            }
        } else if (in.fmt == Rdna2Format::SOP1 && !in.has_literal) {
            const uint32_t width = scalar_write_width(in);
            for (uint32_t word = 0; word < width; ++word) {
                const bool src = within(s, in.src[0], word, half);
                const bool exec_in = s.test(fact(kExecLo + static_cast<int>(word), half));
                std::optional<bool> exec;
                bool dst = false;
                switch (in.opcode) {
                    case kSop1OpcodeMovB32:
                    case kSop1OpcodeMovB64: dst = src; break;
                    case kSop1OpcodeCmovB32:
                    case kSop1OpcodeCmovB64: dst = src && within(s, in.dst, word, half); break;
                    case 0x24:
                    case 0x3c:   // and_saveexec: S0 & EXEC
                    case 0x37:
                    case 0x44:   // andn1_saveexec: ~S0 & EXEC
                        exec = exec_in;
                        break;
                    case 0x25:
                    case 0x3d:   // or_saveexec
                    case 0x26:
                    case 0x3e:   // xor_saveexec
                    case 0x27:
                    case 0x3f:   // andn2_saveexec: S0 & ~EXEC
                        exec = exec_in && src;
                        break;
                    case 0x28:
                    case 0x40:   // orn2_saveexec
                    case 0x29:
                    case 0x41:   // nand_saveexec
                    case 0x2a:
                    case 0x42:   // nor_saveexec
                    case 0x2b:
                    case 0x43:   // xnor_saveexec
                    case 0x38:
                    case 0x45:   // orn1_saveexec
                        exec = false;
                        break;
                    case 0x39:
                    case 0x46:   // andn1_wrexec: ~S0 & EXEC
                        exec = exec_in;
                        break;
                    case 0x3a:
                    case 0x47:   // andn2_wrexec: S0 & ~EXEC
                        exec = exec_in && src;
                        break;
                    default: break;
                }
                const bool wrexec = in.opcode == 0x39u || in.opcode == 0x3au ||
                                    in.opcode == 0x46u || in.opcode == 0x47u;
                if (exec) dst = wrexec ? *exec : exec_in;
                mark(in.dst.value + static_cast<int>(word), half, dst);
                if (exec && word < exec_width) mark(kExecLo + static_cast<int>(word), half, *exec);
            }
        } else if (in.fmt == Rdna2Format::SOP2 && !in.has_literal) {
            const uint32_t width = rdna2_sop2_dest_dwords(in.opcode);
            for (uint32_t word = 0; word < width; ++word) {
                const bool a = within(s, in.src[0], word, half);
                const bool b = within(s, in.src[1], word, half);
                bool dst = false;
                switch (in.opcode) {
                    case kSop2OpcodeAndB32:
                    case kSop2OpcodeAndB64: dst = a || b; break;
                    case kSop2OpcodeOrB32:
                    case kSop2OpcodeOrB64:
                    case kSop2OpcodeXorB32:
                    case 0x13u:   // s_xor_b64
                    case kSop2OpcodeCselectB32:
                    case kSop2OpcodeCselectB64: dst = a && b; break;
                    case kSop2OpcodeAndn2B32:
                    case 0x15u: dst = a; break;   // s_andn2_b64
                    default: break;
                }
                mark(in.dst.value + static_cast<int>(word), half, dst);
            }
        }
    }
    e.out &= ~written;
    e.out |= bounded & written;
    if (exec_write) e.exec_ok = (e.out & exec_words()) == exec_words();
    return e;
}

}   // namespace

int loop_widening_exec_pc(const std::vector<Rdna2Inst>& ins, const DivLoop& loop) {
    std::unordered_map<uint32_t, std::size_t> index;
    for (std::size_t i = 0; i < ins.size(); ++i)
        if (ins[i].pc >= loop.header_pc && ins[i].pc <= loop.backedge_pc)
            index.emplace(ins[i].pc, i);
    const auto header = index.find(loop.header_pc);
    if (header == index.end()) return static_cast<int>(loop.header_pc);
    std::vector<std::optional<Words>> state(ins.size());
    std::vector<std::size_t> pending;
    const auto flow = [&](uint32_t pc, const Words& s) {
        const auto it = index.find(pc);
        if (it == index.end()) return;   // leaves the loop
        auto& at = state[it->second];
        const Words met = at ? (*at & s) : s;
        if (at && *at == met) return;
        at = met;
        pending.push_back(it->second);
    };
    // The back edge restarts from the EXEC its last exit test left; earlier saves are not kept.
    state[header->second] = exec_words();
    pending.push_back(header->second);
    while (!pending.empty()) {
        const std::size_t i = pending.back();
        pending.pop_back();
        const auto& in = ins[i];
        if (unmodelled_control(in)) return static_cast<int>(in.pc);
        const auto& at = state[i];
        if (!at) return static_cast<int>(in.pc);
        const Effect e = transfer(in, *at);
        if (!e.exec_ok) return static_cast<int>(in.pc);
        if (in.is_end) continue;
        const uint32_t next = in.pc + in.len_dwords;
        if (!branch(in)) {
            flow(next, e.out);
            continue;
        }
        const uint32_t target = branch_target(in);
        const bool leaves = target < loop.header_pc || target > loop.backedge_pc;
        if (target != loop.header_pc) flow(target, e.out);
        if (in.opcode == 0x02u) continue;   // s_branch
        // An EXEC test that leaves the loop is where an invocation with a clear bit leaves: on the
        // fallthrough, the bound becomes this EXEC.
        const bool exit_test = leaves && (in.opcode == 0x08u || in.opcode == 0x09u);
        flow(next, exit_test ? exec_words() : e.out);
    }
    return -1;
}

}   // namespace prosper::gpu
