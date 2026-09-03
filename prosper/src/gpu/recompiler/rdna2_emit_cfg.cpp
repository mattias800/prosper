// rdna2_emit_cfg.cpp — the divergent-control-flow state machine and emit_body, split out of
// rdna2_to_spirv.cpp. Shared state lives in gpu/recompiler/rdna2_to_spirv_internal.hpp.
#include <atomic>
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/diagnostic_selectors.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_cf9200_contract.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_compute_contracts.hpp"
#include "gpu/recompiler/gta5/rdna2_gta5_packed_pointer.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_buffer_shadow.hpp"
#include "gpu/recompiler/indirect/rdna2_indirect_pointer_analysis.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"

namespace prosper::gpu {


// #2319: render exactly the dwords the instruction HAS, not a fixed two.
//
// `words[]` is `uint32_t words[5] = {0,...}`, so a ONE-dword instruction printed as
// `words=bf860051,00000000` is indistinguishable from a two-dword instruction whose second dword
// is genuinely zero. That defeats the purpose of printing raw words (#2312), which is that a
// reader can name the instruction with one command:
//
//     echo '0x0e,0x16,0xea,0xbe' | llvm-mc -arch=amdgcn -mcpu=gfx1010 -disassemble
//
// A reader who takes `words=x,00000000` at face value feeds EIGHT bytes, and llvm-mc decodes a
// second, entirely fictitious instruction from the zero dword -- output that looks exactly as
// authoritative as the correct answer. On the CrossWorlds census pc=101 needed four bytes and
// pc=2038 needed eight, and nothing in either line said which.
//
// Clamped to the array bound as well as to len_dwords: len_dwords comes from the decoder, and a
// decode that failed badly enough to be rejected here is not a source to trust for indexing.
inline std::string reject_words_text(const Rdna2Inst& in) {
    const uint32_t count = std::min<uint32_t>(in.len_dwords ? in.len_dwords : 1u,
                                              (uint32_t)(sizeof(in.words) / sizeof(in.words[0])));
    std::string out;
    char buf[16];
    for (uint32_t i = 0; i < count; ++i) {
        std::snprintf(buf, sizeof buf, "%08x", in.words[i]);
        if (i) out += ',';
        out += buf;
    }
    return out;
}

// Some GFX10 pixel shaders leave scheduled 64-bit mask operations whose VCC/SCC results feed only
// other dead mask operations and are overwritten before an observable read. Astro Bot's SSAO shader does this with
// `s_and_b64 vcc, s[0:1], vcc`, where s[0:1] is also a live T# descriptor; attempting to reinterpret
// the descriptor bits as a per-lane mask is both impossible in descriptor-backed SPIR-V and pointless.
// Elide only the mechanically proven dead form: the shader has no SCC consumer anywhere, and CFG
// liveness proves the VCC pair cannot reach a non-mask read before redefinition. This deliberately
// does not become a general scalar-pair-to-wave-mask fallback.
std::unordered_set<uint32_t> dead_wave_mask_writes(const std::vector<Rdna2Inst>& ins) {
    for (const auto& in : ins) {
        const bool reads_scc =
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x04 || in.opcode == 0x05)) ||
            (in.fmt == Rdna2Format::SOP2 &&
             (in.opcode == 0x04 || in.opcode == 0x05 ||
              in.opcode == 0x0a || in.opcode == 0x0b)) ||
            (in.fmt == Rdna2Format::SOP1 && (in.opcode == 0x05 || in.opcode == 0x06)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x02);
        if (reads_scc) return {};
    }
    auto is_mask = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP2 &&
            (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 ||
             in.opcode == 0x15 || in.opcode == 0x17 || in.opcode == 0x19 ||
             in.opcode == 0x1b || in.opcode == 0x1d) &&
            (in.dst.value == 106 || in.dst.value == 107);
    };
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i) by_pc[ins[i].pc] = i;
    std::vector<std::vector<size_t>> succ(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03;
        if (!branch || in.opcode != 0x02) {
            if (i + 1 < ins.size()) succ[i].push_back(i + 1);
        }
        if (branch) {
            const uint32_t target = in.pc + in.len_dwords +
                                    static_cast<uint32_t>(static_cast<int32_t>(in.simm16));
            auto it = by_pc.find(target);
            if (it == by_pc.end()) return {}; // malformed/unbounded CFG: make no dead-write claim
            succ[i].push_back(it->second);
        }
    }
    auto uses = [&](const Rdna2Inst& in) -> uint8_t {
        uint8_t bits = 0;
        for (uint8_t k = 0; k < in.n_src; ++k) {
            if (in.src[k].kind != OperandKind::SGPR &&
                in.src[k].kind != OperandKind::Special) continue;
            if (in.src[k].value == 106) bits |= 1;
            if (in.src[k].value == 107) bits |= 2;
        }
        if (is_mask(in) && bits) bits = 3; // every modeled mask logical reads a full B64 pair
        if (in.fmt == Rdna2Format::VOP2 &&
            (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a))) bits |= 3;
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07)) bits |= 3;
        return bits;
    };
    auto defs = [&](const Rdna2Inst& in) -> uint8_t {
        if (is_mask(in)) return 3;
        // A cmpx writes EXEC and has NO VCC destination, so it must NOT count as defining VCC —
        // the decoder gives every VOPC e32 dst = 106 (VCC_LO), so without this exclusion a cmpx
        // would satisfy both conjuncts and record a phantom definition. A private copy of the
        // windows here listed three of the six, so every v_cmpx_*_f64/_i64/_u64/_u16 was recorded
        // as defining VCC; a preceding live `s_and_b64 vcc` then looked overwritten before use,
        // was classified dead and ELIDED, leaving stale VCC at the real consumer with no
        // diagnostic. Kernel 32r13v pins it. Use the one shared predicate (#2120).
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            (in.dst.value == 106 || in.dst.value == 107)) return 3;
        if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2a)
            return 3;
        if (in.fmt == Rdna2Format::VOP3 &&
            (in.sdst.value == 106 || in.sdst.value == 107)) return 3;
        if (in.fmt == Rdna2Format::SOP1) {
            if (in.dst.value == 106) return in.opcode == 0x04 ? 3 : 1;
            if (in.dst.value == 107) return 2;
        }
        if (in.fmt == Rdna2Format::SMEM) {
            uint32_t n = 0;
            switch (in.opcode) {
                case 0x0: case 0x8: n=1; break; case 0x1: case 0x9: n=2; break;
                case 0x2: case 0xa: n=4; break; case 0x3: case 0xb: n=8; break;
                case 0x4: case 0xc: n=16; break; default: break;
            }
            uint8_t bits = 0;
            if (n && in.dst.value <= 106 && 106 < in.dst.value + static_cast<int>(n)) bits |= 1;
            if (n && in.dst.value <= 107 && 107 < in.dst.value + static_cast<int>(n)) bits |= 2;
            return bits;
        }
        return 0;
    };
    // Least-fixed-point dataflow rooted only in observable (non-candidate) VCC reads. A mask
    // candidate propagates liveness to its B64 input only when its output is itself live; this also
    // removes dead self-dependent mask chains inside loops without mistaking the cycle for a use.
    std::vector<uint8_t> live_in(ins.size(), 0), live_out(ins.size(), 0);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t ri = ins.size(); ri-- > 0;) {
            uint8_t out = 0;
            for (size_t s : succ[ri]) out |= live_in[s];
            const uint8_t def = defs(ins[ri]), use = uses(ins[ri]);
            const uint8_t propagated_use = is_mask(ins[ri]) && !(out & def) ? 0 : use;
            const uint8_t in = propagated_use | (out & static_cast<uint8_t>(~def));
            if (out != live_out[ri] || in != live_in[ri]) {
                live_out[ri] = out; live_in[ri] = in; changed = true;
            }
        }
    }
    std::unordered_set<uint32_t> dead;
    for (size_t i = 0; i < ins.size(); ++i)
        if (is_mask(ins[i]) && !(live_out[i] & 3)) dead.insert(ins[i].pc);
    return dead;
}

namespace {


// Number of consecutive scalar dwords consumed by one explicit ALU source. Operand decode names
// only the first physical register, so every CFG/liveness user must share this opcode-aware width
// rather than infer B64 from the register number. Unknown VOP3 operations stay conservative.

// S_MOV_B64 from VCC publishes both an exact Bool-domain saved mask and its two ballot words under
// native Wave64. The compact structured emitter preserves both views through its SSA/PHI machinery,
// but unlike the dispatcher it previously had no lifetime tag telling S_FF1/S_BCNT which view owns
// the pair at an exact consumer. Compute a small forward MUST analysis over the decoded scalar CFG:
// a saved-mask fact is generated by an exact VCC copy reached from a proved mask-domain VCC or by a
// SAVEEXEC destination (which receives OLD_EXEC independently of its logical source). Every
// overlapping scalar write kills it, and joins retain it only when every reachable predecessor
// agrees. Indirect PC updates have no successor, so they cannot manufacture a dominance fact beyond
// an unknown transfer.
std::unordered_set<uint32_t> proven_structured_wave64_mask_reduction_pcs(
        const std::vector<Rdna2Inst>& ins) {
    size_t count = 0;
    while (count < ins.size() && !ins[count].is_end) ++count;
    if (!count) return {};

    std::unordered_map<uint32_t, size_t> index_for_pc;
    for (size_t i = 0; i < count; ++i) index_for_pc.emplace(ins[i].pc, i);

    std::vector<std::set<int>> incoming(count);
    std::vector<bool> reachable(count, false);
    std::vector<size_t> pending{0};
    reachable[0] = true;
    while (!pending.empty()) {
        const size_t index = pending.back();
        pending.pop_back();
        std::set<int> saved = incoming[index];
        const Rdna2Inst& in = ins[index];

        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (auto it = saved.begin(); it != saved.end();) {
                const int root = *it;
                if (base < root + 2 && root < base + static_cast<int>(width))
                    it = saved.erase(it);
                else
                    ++it;
            }
        }, /*wave32_one_word_masks=*/false);
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
            in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
            in.src[0].kind == OperandKind::Special && in.src[0].value == 106 &&
            saved.contains(106))
            saved.insert(in.dst.value);
        // Every SAVEEXEC variant writes OLD_EXEC to SDST before replacing architectural EXEC with
        // its logical result. OLD_EXEC is always one complete Wave64 mask, so an ordinary pair or
        // VCC destination becomes an exact mask producer without requiring source provenance. Do
        // not seed an EXEC destination: its architectural state after the instruction is the new
        // logical result, not the saved value.
        const bool saves_old_exec = in.fmt == Rdna2Format::SOP1 &&
            ((in.opcode >= kSop1OpcodeAndSaveexecB64 &&
              in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
             in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
             in.opcode == kSop1OpcodeOrn1SaveexecB64);
        if (saves_old_exec && in.dst.kind == OperandKind::SGPR &&
            in.dst.value >= 0 && in.dst.value <= 106)
            saved.insert(in.dst.value);
        // A non-CMPX vector compare writes a fresh architectural VCC predicate. This deliberately
        // stays narrower than the dispatcher's general mask transfer: the structured override only
        // needs to certify the captured VOPC -> S_MOV_B64 -> reduction chain, and an ordinary scalar
        // VCC write above kills the fact before it can seed a saved pair.
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
            const int mask_destination =
                in.dst.kind == OperandKind::SGPR && in.dst.value <= 105
                    ? in.dst.value : 106;
            saved.insert(mask_destination);
        }

        std::vector<size_t> successors;
        auto add_successor = [&](uint32_t pc) {
            const auto found = index_for_pc.find(pc);
            if (found != index_for_pc.end() &&
                std::find(successors.begin(), successors.end(), found->second) == successors.end())
                successors.push_back(found->second);
        };
        const bool indirect_pc = in.fmt == Rdna2Format::SOP1 &&
            in.opcode >= kSop1OpcodeSetpcB64 && in.opcode <= kSop1OpcodeRfeB64;
        if (!indirect_pc && in.fmt == Rdna2Format::SOPP &&
            sopp_opcode_is_direct_branch(in.opcode)) {
            add_successor(branch_target(in));
            if (in.opcode != kSoppOpcodeBranch && index + 1 < count)
                successors.push_back(index + 1);
        } else if (!indirect_pc && in.fmt == Rdna2Format::SOPP && in.opcode == 0x12u) {
            // S_TRAP transfers control outside the decoded shader stream.
        } else if (!indirect_pc && index + 1 < count) {
            successors.push_back(index + 1);
        }

        for (size_t successor : successors) {
            if (!reachable[successor]) {
                reachable[successor] = true;
                incoming[successor] = saved;
                pending.push_back(successor);
                continue;
            }
            std::set<int> joined;
            std::set_intersection(
                incoming[successor].begin(), incoming[successor].end(),
                saved.begin(), saved.end(), std::inserter(joined, joined.end()));
            if (joined != incoming[successor]) {
                incoming[successor] = std::move(joined);
                pending.push_back(successor);
            }
        }
    }

    std::unordered_set<uint32_t> proven;
    for (size_t i = 0; i < count; ++i) {
        const Rdna2Inst& in = ins[i];
        if (!reachable[i] || in.fmt != Rdna2Format::SOP1 ||
            (in.opcode != kSop1OpcodeBcnt1I32B64 &&
             in.opcode != kSop1OpcodeFf1I32B64) ||
            in.src[0].kind != OperandKind::SGPR)
            continue;
        if (incoming[i].contains(in.src[0].value)) proven.insert(in.pc);
    }
    return proven;
}

// Prove S_LOAD_DWORDX2 descriptor-fragment shapes. The load supplies one or two live words of a
// four-dword V#; scalar code fills or replaces the other words before MUBUF, MTBUF, or S_BUFFER_LOAD
// consumes the complete live descriptor. The front half has already read the guest words and
// published the resulting resource at that exact consumer PC, so the loaded fragment is
// provenance-only in SPIR-V and can be represented by zero placeholders. Immediate descriptor-table
// offsets are admitted by this use proof rather than a title-specific offset inventory.
//
// This is deliberately a whole-CFG use proof, not opcode-wide admission. Every reachable path is
// followed until the loaded words are overwritten or execution ends. A loaded word may only be read
// through an exact-PC, key-less buffer descriptor; every ordinary scalar/address read and every
// unresolved control edge rejects the candidate. CONFIDENCE: HIGH for the admitted shape.
std::unordered_set<uint32_t> proven_smem_x2_descriptor_fragment_loads(
        const std::vector<Rdna2Inst>& ins, const ShaderResourceTable* rt,
        uint32_t wave_size) {
    std::unordered_set<uint32_t> proven;
    if (!rt || ins.empty()) return proven;

    std::unordered_map<uint32_t, size_t> index_by_pc;
    for (size_t index = 0; index < ins.size(); ++index)
        index_by_pc.emplace(ins[index].pc, index);

    auto is_scalar_operand = [](const Operand& operand) {
        return operand.kind == OperandKind::SGPR ||
               (operand.kind == OperandKind::Special &&
                operand.value >= 106 && operand.value <= 124);
    };
    auto exact_buffer_resource = [&](const Rdna2Inst& consumer, bool scalar_buffer) {
        const ShaderResource* resource = rt->by_fetch_pc(consumer.pc);
        if (!resource || resource->fetch_pc != consumer.pc ||
            resource->srt_offset != 0xffffffffu ||
            resource->sgpr_base != 0xffffffffu || resource->table_index_count != 0)
            return false;
        if (scalar_buffer) return resource->cls == ResourceClass::ConstantBuffer;
        return resource->cls == ResourceClass::ConstantBuffer ||
               resource->cls == ResourceClass::VertexBuffer;
    };
    auto scalar_register_is = [&](const Operand& operand, int reg) {
        return is_scalar_operand(operand) && operand.value == reg;
    };
    auto instruction_reads_scc = [](const Rdna2Inst& in) {
        for (uint32_t source = 0; source < in.n_src; ++source)
            if (in.src[source].kind == OperandKind::Special && in.src[source].value == 253)
                return true;
        if (in.fmt == Rdna2Format::SOP2)
            return in.opcode == kSop2OpcodeAddcU32 || in.opcode == 0x05u ||
                   in.opcode == kSop2OpcodeCselectB32 ||
                   in.opcode == kSop2OpcodeCselectB64;
        if (in.fmt == Rdna2Format::SOPP)
            return in.opcode == kSoppOpcodeCbranchScc0 || in.opcode == 0x05u;
        if (in.fmt == Rdna2Format::SOP1)
            return in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64;
        return in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCmovkI32;
    };

    std::unordered_set<uint32_t> direct_branch_targets;
    for (const Rdna2Inst& in : ins)
        if (!in.is_end && in.fmt == Rdna2Format::SOPP &&
            sopp_opcode_is_direct_branch(in.opcode))
            direct_branch_targets.insert(branch_target(in));

    for (size_t load_index = 0; load_index < ins.size(); ++load_index) {
        const Rdna2Inst& load = ins[load_index];
        const bool register_offset =
            load.src[1].kind == OperandKind::SGPR ||
            (load.src[1].kind == OperandKind::Special &&
             load.src[1].value >= 106 && load.src[1].value <= 123);
        const bool optional_null_immediate =
            load.src[1].kind == OperandKind::Special && load.src[1].value == 125 &&
            load.literal == kGtaOptionalBufferPointerOffset;
        const bool aligned_immediate =
            load.src[1].kind == OperandKind::Special && load.src[1].value == 125 &&
            static_cast<int32_t>(load.literal) >= 0 && (load.literal & 3u) == 0;
        if (load.is_end || load.fmt != Rdna2Format::SMEM ||
            load.opcode != kSmemOpcodeLoadDwordX2 ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            (load.dst.value > 104 && load.dst.value != 106) ||
            (!register_offset && !aligned_immediate))
            continue;

        const int base = load.dst.value;
        struct DescriptorRelocation {
            size_t low_index = SIZE_MAX;
            size_t high_index = SIZE_MAX;
            int destination_base = -1;
        } relocation;

        // Some descriptor builders relocate a loaded 64-bit address with an ordinary carry pair:
        //   s_add_u32  dst.lo, loaded.lo, base.lo
        //   s_addc_u32 dst.hi, loaded.hi, base.hi
        // Admit only the exact pair in one straight-line SCC lifetime. Interposed scalar moves are
        // SCC-transparent, and the produced final carry must have no later observer. This keeps the
        // loaded address as descriptor provenance without treating general scalar arithmetic as a
        // descriptor transformation.
        for (size_t low_index = load_index + 1;
             low_index < ins.size() && relocation.low_index == SIZE_MAX; ++low_index) {
            const Rdna2Inst& low = ins[low_index];
            if (low.is_end ||
                (low.fmt == Rdna2Format::SOPP && !sopp_is_noop(low)) ||
                (low.fmt == Rdna2Format::SOP1 &&
                 low.opcode >= kSop1OpcodeSetpcB64 && low.opcode <= kSop1OpcodeRfeB64))
                break;
            if (low.fmt != Rdna2Format::SOP2 || low.opcode != kSop2OpcodeAddU32 ||
                low.dst.kind != OperandKind::SGPR || low.dst.value < 0 ||
                low.dst.value + 3 > 105 || low.n_src != 2)
                continue;
            if (low.dst.value <= base + 1 && base <= low.dst.value + 1)
                continue;
            int loaded_source = -1;
            if (scalar_register_is(low.src[0], base)) loaded_source = 0;
            if (scalar_register_is(low.src[1], base)) {
                if (loaded_source >= 0) continue;
                loaded_source = 1;
            }
            if (loaded_source < 0 ||
                !is_scalar_operand(low.src[static_cast<uint32_t>(1 - loaded_source)]))
                continue;
            const int other_base = low.src[static_cast<uint32_t>(1 - loaded_source)].value;
            if (other_base < 0 || other_base + 1 > 105 ||
                (other_base <= base + 1 && base <= other_base + 1) ||
                (other_base <= low.dst.value + 1 &&
                 low.dst.value <= other_base + 1))
                continue;

            for (size_t high_index = low_index + 1; high_index < ins.size(); ++high_index) {
                const Rdna2Inst& high = ins[high_index];
                if (direct_branch_targets.contains(high.pc)) break;
                if (high.fmt == Rdna2Format::SOP2 &&
                    high.opcode == kSop2OpcodeAddcU32 && high.n_src == 2 &&
                    high.dst.kind == OperandKind::SGPR &&
                    high.dst.value == low.dst.value + 1) {
                    const bool matched_sources =
                        (scalar_register_is(high.src[0], base + 1) &&
                         scalar_register_is(high.src[1], other_base + 1)) ||
                        (scalar_register_is(high.src[1], base + 1) &&
                         scalar_register_is(high.src[0], other_base + 1));
                    if (!matched_sources) break;
                    bool carry_observed = false;
                    for (size_t later = high_index + 1; later < ins.size(); ++later)
                        if (instruction_reads_scc(ins[later])) {
                            carry_observed = true;
                            break;
                        }
                    if (!carry_observed)
                        relocation = {low_index, high_index, low.dst.value};
                    break;
                }
                if (high.fmt != Rdna2Format::SOP1 ||
                    high.opcode != kSop1OpcodeMovB32)
                    break;
                if (instruction_reads_scc(high)) break;
                if (high.dst.value == base || high.dst.value == base + 1 ||
                    high.dst.value == low.dst.value ||
                    high.dst.value == low.dst.value + 1)
                    break;
            }
        }

        auto overlap = [base, &relocation](uint8_t live, int first, uint32_t words) {
            if (first < 0 || !words) return false;
            for (uint32_t word = 0; word < words; ++word) {
                const int reg = first + static_cast<int>(word);
                const int original_relative = reg - base;
                if (original_relative >= 0 && original_relative < 2 &&
                    (live & (1u << original_relative)))
                    return true;
                const int relocated_relative = reg - relocation.destination_base;
                if (relocation.destination_base >= 0 && relocated_relative >= 0 &&
                    relocated_relative < 2 && (live & (1u << (relocated_relative + 2))))
                    return true;
            }
            return false;
        };
        auto clear_written = [base, &relocation](uint8_t& live, int first, uint32_t words) {
            if (first < 0) return;
            for (uint32_t word = 0; word < words; ++word) {
                const int reg = first + static_cast<int>(word);
                const int original_relative = reg - base;
                if (original_relative >= 0 && original_relative < 2)
                    live &= static_cast<uint8_t>(~(1u << original_relative));
                const int relocated_relative = reg - relocation.destination_base;
                if (relocation.destination_base >= 0 && relocated_relative >= 0 &&
                    relocated_relative < 2)
                    live &= static_cast<uint8_t>(~(1u << (relocated_relative + 2)));
            }
        };

        struct PendingState { size_t index; uint8_t live; };
        std::vector<PendingState> pending;
        if (load_index + 1 < ins.size()) pending.push_back({load_index + 1, 0x3u});
        std::vector<uint16_t> visited(ins.size(), 0);
        bool consumed = false;
        bool valid = true;

        auto enqueue_pc = [&](int64_t target, uint8_t live) {
            if (!live || !valid) return;
            if (target < 0) { valid = false; return; }
            const auto found = index_by_pc.find(static_cast<uint32_t>(target));
            if (found != index_by_pc.end()) {
                pending.push_back({found->second, live});
                return;
            }
            // A forward target beyond the decoded program is a terminating early-out. Every other
            // unresolved edge could re-enter code whose scalar reads are unknown to this proof.
            if (target <= static_cast<int64_t>(ins.back().pc)) valid = false;
        };

        while (valid && !pending.empty()) {
            PendingState state = pending.back();
            pending.pop_back();
            if (!state.live || state.index >= ins.size()) continue;
            const uint16_t state_bit = static_cast<uint16_t>(1u << state.live);
            if (visited[state.index] & state_bit) continue;
            visited[state.index] |= state_bit;

            const Rdna2Inst& in = ins[state.index];
            if (in.is_end) continue;
            uint8_t live = state.live;

            // Captured builders patch descriptor control bits in the loaded second word before the
            // V# is consumed. Their results remain descriptor provenance: retain that word's live
            // marker so every later use is still checked, but do not mistake these exact RMW patches
            // for ordinary scalar observations of guest data.
            const bool bitset_descriptor_patch =
                in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeBitset1B32 &&
                in.dst.kind == OperandKind::SGPR && overlap(live, in.dst.value, 1) &&
                in.n_src == 1 && in.src[0].kind == OperandKind::InlineInt;
            const bool or_descriptor_patch =
                optional_null_immediate && in.fmt == Rdna2Format::SOP2 &&
                in.opcode == kSop2OpcodeOrB32 &&
                in.dst.kind == OperandKind::SGPR && in.dst.value == base + 1 &&
                in.n_src == 2 && in.literal == kGtaOptionalBufferStrideWord &&
                ((in.src[0].kind == OperandKind::SGPR && in.src[0].value == in.dst.value &&
                  in.src[1].kind == OperandKind::Literal) ||
                 (in.src[1].kind == OperandKind::SGPR && in.src[1].value == in.dst.value &&
                  in.src[0].kind == OperandKind::Literal));
            constexpr uint32_t kGtavBufferDescriptorHighControlBits = 0x000c0000u;
            const bool high_control_descriptor_patch =
                register_offset && in.fmt == Rdna2Format::SOP2 &&
                in.opcode == kSop2OpcodeOrB32 &&
                in.dst.kind == OperandKind::SGPR && in.dst.value == base + 1 &&
                in.n_src == 2 && in.src[0].kind == OperandKind::SGPR &&
                in.src[0].value == in.dst.value && in.src[1].kind == OperandKind::Literal &&
                in.literal == kGtavBufferDescriptorHighControlBits;
            const bool relocation_low = state.index == relocation.low_index;
            const bool relocation_high = state.index == relocation.high_index;
            const bool descriptor_patch = bitset_descriptor_patch || or_descriptor_patch ||
                high_control_descriptor_patch || relocation_low || relocation_high;

            // A load overlapping physical VCC also creates a mask lifetime. Reject every implicit
            // mask observation until a real VCC writer replaces the live half or pair; explicit
            // scalar observations remain covered by the ordinary operand walk below.
            const bool implicit_vcc_read =
                overlap(live, 106, wave_size == 32 ? 1u : 2u) &&
                ((in.fmt == Rdna2Format::SOPP &&
                  (in.opcode == 0x06u || in.opcode == 0x07u)) ||
                 (in.fmt == Rdna2Format::VOP2 &&
                  (in.opcode == 0x01u ||
                   (in.opcode >= 0x28u && in.opcode <= 0x2au))));
            if (implicit_vcc_read) {
                valid = false;
                break;
            }

            bool branch = false;
            bool fallthrough = true;
            if (in.fmt == Rdna2Format::SOPP) {
                if (sopp_opcode_is_direct_branch(in.opcode)) {
                    branch = true;
                    fallthrough = in.opcode != kSoppOpcodeBranch;
                } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::SOP1 &&
                       in.opcode >= kSop1OpcodeSetpcB64 &&
                       in.opcode <= kSop1OpcodeRfeB64) {
                valid = false;
                break;
            }

            bool descriptor_read = false;
            if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) &&
                in.src[1].kind == OperandKind::SGPR &&
                overlap(live, in.src[1].value, 4)) {
                if ((is_scalar_operand(in.src[2]) && overlap(live, in.src[2].value, 1)) ||
                    !exact_buffer_resource(in, false)) {
                    valid = false;
                    break;
                }
                descriptor_read = true;
                consumed = true;
            } else if (in.fmt == Rdna2Format::SMEM &&
                       smem_opcode_is_buffer_load(in.opcode) &&
                       is_scalar_operand(in.src[0]) && overlap(live, in.src[0].value, 4)) {
                if ((is_scalar_operand(in.src[1]) && overlap(live, in.src[1].value, 1)) ||
                    !exact_buffer_resource(in, true)) {
                    valid = false;
                    break;
                }
                descriptor_read = true;
                consumed = true;
            }

            if (in.fmt == Rdna2Format::SOPP) {
                // Branch operands are implicit architectural condition codes, never ordinary SGPR
                // data. Their control edges were validated above.
            } else if (in.fmt == Rdna2Format::SMEM) {
                const uint32_t base_words = smem_opcode_is_buffer_load(in.opcode) ? 4u : 2u;
                if ((!descriptor_read && is_scalar_operand(in.src[0]) &&
                     overlap(live, in.src[0].value, base_words)) ||
                    (is_scalar_operand(in.src[1]) && overlap(live, in.src[1].value, 1))) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) {
                if ((!descriptor_read && in.src[1].kind == OperandKind::SGPR &&
                     overlap(live, in.src[1].value, 4)) ||
                    (is_scalar_operand(in.src[2]) && overlap(live, in.src[2].value, 1))) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::MIMG) {
                if ((in.src[1].kind == OperandKind::SGPR &&
                     overlap(live, in.src[1].value, 8)) ||
                    (in.src[2].kind == OperandKind::SGPR &&
                     overlap(live, in.src[2].value, 4))) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::FLAT) {
                if (is_scalar_operand(in.src[1]) && overlap(live, in.src[1].value, 2)) {
                    valid = false;
                    break;
                }
            } else {
                const uint32_t implicit_read = scalar_implicit_destination_read_width(in);
                if (!descriptor_patch && implicit_read &&
                    overlap(live, in.dst.value, implicit_read)) {
                    valid = false;
                    break;
                }
                if (!descriptor_patch) {
                    for (uint32_t source = 0; source < in.n_src; ++source) {
                        if (!is_scalar_operand(in.src[source])) continue;
                        uint32_t words = scalar_alu_source_words(in, source);
                        if (words == UINT32_MAX) continue;
                        if (!words) words = 1; // decoded non-ALU scalar operands remain fail-closed
                        if (overlap(live, in.src[source].value, words)) {
                            valid = false;
                            break;
                        }
                    }
                }
                if (!valid) break;
            }

            if (relocation_low || relocation_high) {
                const uint8_t destination_bit = relocation_low ? 0x4u : 0x8u;
                const bool source_live = relocation_low
                    ? (live & 0x1u) != 0
                    : (live & (0x2u | 0x4u)) != 0;
                clear_written(live,
                              relocation.destination_base + (relocation_high ? 1 : 0), 1);
                if (source_live) live |= destination_bit;
            } else if (!descriptor_patch) {
                for_each_scalar_write(in, [&](int first, uint32_t words) {
                    clear_written(live, first, words);
                }, wave_size == 32);
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                    (in.dst.kind == OperandKind::SGPR ||
                     in.dst.kind == OperandKind::Special) && in.dst.value == 106)
                    clear_written(live, 106, wave_size == 32 ? 1u : 2u);
            }
            if (!live) continue;

            if (branch) {
                const int64_t target = static_cast<int64_t>(in.pc) +
                    static_cast<int64_t>(in.len_dwords) + static_cast<int64_t>(in.simm16);
                enqueue_pc(target, live);
            }
            if (fallthrough) {
                if (state.index + 1 < ins.size()) pending.push_back({state.index + 1, live});
                else valid = false; // live descriptor data fell off an unterminated stream
            }
        }
        if (valid && consumed) proven.insert(load.pc);
    }
    return proven;
}

// Prove the narrow S_LOAD_DWORDX16 descriptor-bundle shape used by GTA V compute kernels. A wide
// scalar load is not intrinsically a descriptor fetch: accepting every x16 as two T#s would replace
// real scalar data with zero placeholders. For each candidate, follow all sixteen loaded words to
// overwrite/end and require BOTH aligned eight-word halves to be consumed as MIMG SRSRCs. Every such
// consumer must have its own key-less exact-PC image resource; an SRT tag cannot name two descriptors
// packed under the load's one immediate offset.
//
// The only scalar transformation admitted is the captured compiler's T#.word3 patch (with at most
// one independent VOP scheduled between the two scalar operations):
//
//   s_and_b32 tmp, tword3, 0x0fffffff
//   s_or_b32  tword3, tmp, 0xd0000000
//
// All ordinary scalar/vector reads, partial descriptor writes, samplers overlapping the bundle, and
// scalar control flow reject the candidate. This is deliberately a use proof, not an opcode-wide
// declaration that x16 loads are descriptors. CONFIDENCE: HIGH for the admitted shape: the front-half
// snapshots both halves and publishes the live descriptor at each exact MIMG PC.
bool smem_x16_patch_gap_reads_implicit_state(const Rdna2Inst& in, int temporary) {
    // The AND's SCC is descriptor-derived until the matching OR overwrites it. Vector ALU can name
    // SCC as the scalar source encoding even though it is outside the ordinary SGPR/special range.
    for (uint32_t source = 0; source < in.n_src; ++source)
        if (in.src[source].kind == OperandKind::Special && in.src[source].value == 253)
            return true;

    // E32 cndmask and carry forms consume architectural VCC without exposing it in n_src. This is
    // observable descriptor-derived data when the compiler chose VCC_LO as its word3 temporary.
    return (temporary == 106 || temporary == 107) && in.fmt == Rdna2Format::VOP2 &&
           (in.opcode == 0x01u || (in.opcode >= 0x28u && in.opcode <= 0x2au));
}

std::unordered_set<uint32_t> proven_smem_x16_descriptor_loads(
        const std::vector<Rdna2Inst>& ins, const ShaderResourceTable* rt) {
    std::unordered_set<uint32_t> proven;
    if (!rt || ins.empty()) return proven;

    // Alternate entries would require path-sensitive lifetime/provenance joins. Keep this first
    // admission linear: hints, waits and barriers are transparent; every real scalar branch or
    // indirect PC transfer makes the whole candidate ineligible.
    for (const Rdna2Inst& in : ins) {
        if (in.is_end) continue;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u)
            return proven;
        if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) && in.opcode != 0x0au)
            return proven;
    }

    auto scalar_operand = [](const Operand& operand) {
        return operand.kind == OperandKind::SGPR ||
               (operand.kind == OperandKind::Special &&
                operand.value >= 106 && operand.value <= 124);
    };
    auto literal_is = [](const Rdna2Inst& in, const Operand& operand, uint32_t value) {
        return operand.kind == OperandKind::Literal && in.literal == value;
    };

    for (size_t load_index = 0; load_index < ins.size(); ++load_index) {
        const Rdna2Inst& load = ins[load_index];
        if (load.is_end || load.fmt != Rdna2Format::SMEM || load.opcode != 0x04u ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + 15 > 105 ||
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            static_cast<int32_t>(load.literal) < 0)
            continue;

        const int base = load.dst.value;
        uint16_t live = 0xffffu;
        bool consumed[2] = {false, false};
        bool valid = true;
        auto live_overlap = [&](int first, uint32_t words) {
            if (first < 0 || !words) return false;
            for (uint32_t word = 0; word < words; ++word) {
                const int relative = first + static_cast<int>(word) - base;
                if (relative >= 0 && relative < 16 &&
                    (live & static_cast<uint16_t>(1u << relative)))
                    return true;
            }
            return false;
        };
        auto clear_written = [&](int first, uint32_t words) {
            if (first < 0) return;
            for (uint32_t word = 0; word < words; ++word) {
                const int relative = first + static_cast<int>(word) - base;
                if (relative >= 0 && relative < 16)
                    live &= static_cast<uint16_t>(~(1u << relative));
            }
        };

        for (size_t index = load_index + 1; valid && index < ins.size(); ++index) {
            const Rdna2Inst& in = ins[index];
            if (in.is_end) break;

            // Recognize the complete word3 patch as one unit. One captured variant schedules an
            // independent VOP between the two scalar instructions; admit that exact one-instruction
            // gap only when it cannot observe or replace either the descriptor bundle or temporary.
            bool patched = false;
            if (index + 1 < ins.size() && in.fmt == Rdna2Format::SOP2 &&
                in.opcode == 0x0eu && in.dst.kind == OperandKind::SGPR &&
                in.src[0].kind == OperandKind::SGPR &&
                literal_is(in, in.src[1], 0x0fffffffu)) {
                const int descriptor_word = in.src[0].value;
                const int half = descriptor_word == base + 3 ? 0
                               : descriptor_word == base + 11 ? 1 : -1;
                // The retained GTA V shape uses VCC_LO exactly. Do not generalize this to M0 or
                // other architectural scalar registers: their implicit consumers are not all in the
                // ordinary SGPR liveness inventory (DS observes M0 without a decoded scalar source).
                const bool exact_patch_temporary = in.dst.value == 106;
                size_t join_index = index + 1;
                const Rdna2Inst& possible_gap = ins[join_index];
                if (possible_gap.fmt != Rdna2Format::SOP2 || possible_gap.opcode != 0x10u) {
                    bool transparent_gap = possible_gap.fmt == Rdna2Format::VOP1 ||
                                           possible_gap.fmt == Rdna2Format::VOP2 ||
                                           possible_gap.fmt == Rdna2Format::VOP3;
                    if (transparent_gap &&
                        smem_x16_patch_gap_reads_implicit_state(possible_gap, in.dst.value))
                        transparent_gap = false;
                    for (uint32_t source = 0;
                         transparent_gap && source < possible_gap.n_src; ++source) {
                        if (!scalar_operand(possible_gap.src[source])) continue;
                        if (possible_gap.src[source].value == in.dst.value ||
                            live_overlap(possible_gap.src[source].value, 2))
                            transparent_gap = false;
                    }
                    for_each_scalar_write(possible_gap, [&](int first, uint32_t words) {
                        if (live_overlap(first, words) ||
                            (in.dst.value >= first &&
                             in.dst.value < first + static_cast<int>(words)))
                            transparent_gap = false;
                    });
                    if (transparent_gap) ++join_index;
                }
                const Rdna2Inst& join = join_index < ins.size() ? ins[join_index] : in;
                const bool temporary_unobserved = join_index + 1 >= ins.size() ||
                    sgpr_dead_at_merge(ins, ins[join_index + 1].pc, in.dst.value);
                const bool exact_join = half >= 0 && exact_patch_temporary &&
                    join_index < ins.size() && join.fmt == Rdna2Format::SOP2 &&
                    join.opcode == 0x10u && join.dst.kind == OperandKind::SGPR &&
                    join.dst.value == descriptor_word && scalar_operand(join.src[0]) &&
                    join.src[0].value == in.dst.value &&
                    literal_is(join, join.src[1], 0xd0000000u) && temporary_unobserved;
                const uint16_t word_bit = half >= 0
                    ? static_cast<uint16_t>(1u << (descriptor_word - base)) : 0u;
                if (exact_join && (live & word_bit)) {
                    patched = true;
                    index = join_index;
                }
            }
            if (patched) continue;

            if (in.fmt == Rdna2Format::MIMG) {
                const bool t_is_scalar = in.src[1].kind == OperandKind::SGPR;
                const int half = t_is_scalar && in.src[1].value == base ? 0
                               : t_is_scalar && in.src[1].value == base + 8 ? 1 : -1;
                const bool touches_t = t_is_scalar && live_overlap(in.src[1].value, 8);
                const bool touches_sampler = scalar_operand(in.src[2]) &&
                                             live_overlap(in.src[2].value, 4);
                if (touches_sampler) { valid = false; break; }
                if (half >= 0) {
                    const uint16_t half_mask = static_cast<uint16_t>(0xffu << (half * 8));
                    const ShaderResource* resource = rt->by_fetch_pc(in.pc);
                    const bool exact_image = resource && resource->fetch_pc == in.pc &&
                        resource->srt_offset == 0xffffffffu &&
                        resource->sgpr_base == 0xffffffffu &&
                        (resource->cls == ResourceClass::Texture ||
                         resource->cls == ResourceClass::StorageImage);
                    if ((live & half_mask) != half_mask || !exact_image) {
                        valid = false;
                        break;
                    }
                    consumed[half] = true;
                    continue;
                }
                if (touches_t) { valid = false; break; }
            }

            // Bound implicit descriptor/address reads before the generic decoded operands. These
            // packet formats encode a base SGPR while consuming a wider range.
            if (in.fmt == Rdna2Format::SMEM) {
                const uint32_t base_words = in.opcode >= 0x08u ? 4u : 2u;
                if (scalar_operand(in.src[0]) && live_overlap(in.src[0].value, base_words)) {
                    valid = false;
                    break;
                }
                if (scalar_operand(in.src[1]) && live_overlap(in.src[1].value, 1)) {
                    valid = false;
                    break;
                }
            } else if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) {
                if ((in.src[1].kind == OperandKind::SGPR &&
                     live_overlap(in.src[1].value, 4)) ||
                    (scalar_operand(in.src[2]) && live_overlap(in.src[2].value, 1))) {
                    valid = false;
                    break;
                }
            } else {
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    if (!scalar_operand(in.src[source])) continue;
                    const uint32_t words =
                        in.fmt == Rdna2Format::SOP1 ||
                        in.fmt == Rdna2Format::SOP2 ||
                        in.fmt == Rdna2Format::SOPC ||
                        in.fmt == Rdna2Format::VOP3 ? 2u : 1u;
                    if (live_overlap(in.src[source].value, words)) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
            }

            // SOPK read/modify/write forms do not expose their implicit destination read through
            // n_src. Treat every live overlap conservatively; the target descriptor bundles use no
            // SOPK writes, so widening this is unnecessary.
            if (in.fmt == Rdna2Format::SOPK && in.dst.kind == OperandKind::SGPR &&
                live_overlap(in.dst.value, 1)) {
                valid = false;
                break;
            }
            for_each_scalar_write(in, clear_written);
        }
        if (valid && consumed[0] && consumed[1]) proven.insert(load.pc);
    }
    return proven;
}

// Scalar registers that MAY be overwritten while a loop executes. This is deliberately separate
// from loop_written_regs: mask-pair destinations overwrite physical SGPRs (and therefore descriptor
// provenance) but their values live in sreg_bool rather than the scalar-data SSA domain.
void loop_scalar_may_writes(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                            std::set<int>& sregs) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                sregs.insert(base + static_cast<int>(word));
        });
    }
}

void invalidate_loop_descriptor_provenance(RegState& rs, const std::set<int>& sregs) {
    for (int reg : sregs) {
        rs.sreg_written.insert(reg);
        rs.sreg_input.erase(reg);
        rs.sreg_srt.erase(reg);
        // A loop body that may write this register must not leave a copy alias standing: the alias
        // was established on one iteration's path and says nothing about the next one (#1773).
        rs.sreg_ud_alias.erase(reg);
    }
}

// Complex CFG dispatch and the narrow loop structurizers persist B64 mask values, but not the
// separate one-word-validity state required by Wave32 aliases. Conservatively find any B32 mask
// copy that the region could create. The source set is deliberately path-insensitive: a pair made a
// mask on any path may reach a later copy on another dispatcher edge, and rejecting an impossible
// ordering is safer than silently restoring a stale Boolean.
bool has_unpersisted_b32_mask_lifetime(const std::vector<Rdna2Inst>& ins,
                                      uint32_t lo, uint32_t hi,
                                      const RegState& entry) {
    std::set<int> possible_mask_sources{106, 126}; // VCC_LO and EXEC_LO
    for (const auto& kv : entry.sreg_bool) possible_mask_sources.insert(kv.first);
    for (int reg : entry.sreg_bool_b32) possible_mask_sources.insert(reg);
    for (const auto& in : ins) {
        if (in.is_end || in.pc < lo || in.pc >= hi) continue;
        for_each_scalar_write(in, [&](int base, uint32_t) {
            if (scalar_write_is_b64_mask(in, base)) possible_mask_sources.insert(base);
        });
    }
    for (const auto& in : ins) {
        if (in.is_end || in.pc < lo || in.pc >= hi || in.fmt != Rdna2Format::SOP1)
            continue;
        if (in.opcode == 0x09) return true; // s_wqm_b32 creates/consumes the same width state
        if (in.opcode != 0x03 && in.opcode != 0x07) continue;
        const bool register_source = in.src[0].kind == OperandKind::SGPR ||
                                     in.src[0].kind == OperandKind::Special;
        if ((register_source && possible_mask_sources.contains(in.src[0].value)) ||
            in.dst.value == 126)
            return true;
    }
    return false;
}

} // namespace

int shader_max_vgpr(const std::vector<Rdna2Inst>& ins) {
    int highest = 0;
    for (const auto& in : ins) {
        if (in.is_end) break;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const uint32_t source_span = rdna2_vgpr_source_span(in, source);
            if (source_span)
                highest = std::max(highest,
                    in.src[source].value + static_cast<int>(source_span) - 1);
        }
        const uint32_t destination_span = rdna2_vgpr_destination_span(in);
        if (destination_span)
            highest = std::max(highest,
                in.dst.value + static_cast<int>(destination_span) - 1);
    }
    return highest;
}

// Does [lo, hi) contain a `s_mov_b32 sX, m0` -- the instruction that starts an entry-M0 token
// lifetime (#3133)? A STATIC property of the decoded stream, so it can be asked before any block is
// emitted, which is what the loop emitters need: their header phis are built before the body runs,
// and a loop-carried token has no value to seed one with. Returns the destination register (>= 0)
// of the first such save, or -1.
int entry_m0_save_in_range(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x03 &&
            in.src[0].kind == OperandKind::Special && in.src[0].value == 124 &&
            in.dst.value <= 105)
            return in.dst.value;
    }
    return -1;
}

// Registers WRITTEN in the pc range [lo, hi): candidates for an OpPhi at the loop header. Over-
// approximation is safe (an extra phi for a non-carried value merges equal values). Mirrors emit_alu's
// write targets, INCLUDING multi-register writes (MIMG dmask -> N consecutive VGPRs, SMEM -> N SGPRs) so
// no genuinely-carried register is missed (a missing phi = an undominated use = invalid SPIR-V).
void loop_written_regs(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi,
                       std::set<int>& vregs, std::set<int>& sregs) {
    for (const auto& in : ins) {
        if (in.pc < lo || in.pc >= hi) continue;
        switch (in.fmt) {
            case Rdna2Format::VOP1:
                if (in.opcode == 0x02) sregs.insert(in.dst.value);        // v_readfirstlane -> SGPR
                else if (in.opcode == kVop1OpcodeMovreldB32)             // v_movreld: any observable
                    for (int reg = in.dst.value; reg <= shader_max_vgpr(ins); ++reg)
                        vregs.insert(reg);                               // VDST+M0 target
                else vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP2: case Rdna2Format::VOP3P:
                vregs.insert(in.dst.value); break;
            case Rdna2Format::VOP3:
                if (in.opcode == 0x360) sregs.insert(in.dst.value);       // v_readlane -> SGPR
                else {
                    for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                        vregs.insert(in.dst.value + (int)k);
                }
                break;                                                    // (writelane: slots, not SSA)
            case Rdna2Format::DS:
                for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                break;
            case Rdna2Format::MUBUF: case Rdna2Format::MTBUF: case Rdna2Format::MIMG:
            case Rdna2Format::FLAT:
                for (uint32_t k = 0; k < rdna2_vgpr_write_count(in); ++k)
                    vregs.insert(in.dst.value + (int)k);
                if (const int tfe_status = rdna2_tfe_status_vgpr(in); tfe_status >= 0)
                    vregs.insert(tfe_status);
                break;
            case Rdna2Format::SOP1:
                sregs.insert(in.dst.value); break;
            case Rdna2Format::SOPK:
                if (sopk_writes_scalar_data(in.opcode)) sregs.insert(in.dst.value);
                break;
            case Rdna2Format::SOP2:
                // s_lshr_b64 -> EXEC is modeled only in the per-lane mask domain. It does not
                // produce scalar SGPR data, so carrying a scalar value through a loop/if merge is
                // both unnecessary and semantically wrong.
                if (in.opcode != 0x21 || (in.dst.value != 126 && in.dst.value != 127)) {
                    uint32_t words = 1;
                    if (in.opcode == 0x0b)
                        words = scalar_write_width(in);
                    else if (in.opcode == 0x1f || in.opcode == 0x21)
                        words = 2;
                    for (uint32_t word = 0; word < words; ++word)
                        sregs.insert(in.dst.value + static_cast<int>(word));
                }
                break;
            case Rdna2Format::SMEM: {                                      // s_load/s_buffer_load: N consecutive SGPRs
                uint32_t n = 1; switch (in.opcode) { case 0x1: case 0x9: n=2; break; case 0x2: case 0xA: n=4; break;
                    case 0x3: case 0xB: n=8; break; case 0x4: case 0xC: n=16; break; }
                for (uint32_t k = 0; k < n; k++) sregs.insert(in.dst.value + (int)k); break;
            }
            default: break;                          // VOPC/SOPC write VCC/SCC — handled by their own phis
        }
    }
}

namespace {


} // namespace

// True when the guest program itself reads or writes GDS. The witness lives in the internal GDS
// buffer, which is guest-addressable, so instrumenting such a program would change its INPUT -- and
// a diagnostic that perturbs the state it measures can manufacture or suppress the behaviour under
// test. Decoded from the whole program rather than one phase: a GDS access in any phase disqualifies
// the program, and `ins` here is only the current phase's slice.
bool program_touches_guest_gds(const uint32_t* code, size_t dwords) {
    if (!code || !dwords) return false;
    std::vector<Rdna2Inst> decoded;
    if (!rdna2_walk(code, dwords, decoded)) return true;   // undecodable: refuse, fail closed
    return std::any_of(decoded.begin(), decoded.end(), [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::DS && in.ds_gds;
    });
}

uint32_t emitted_loop_trip_bound(uint64_t program_address, uint32_t phase,
                                uint32_t start_pc, uint32_t end_pc,
                                const uint32_t* code, size_t dwords) {
    const ComputeTripBoundSettings settings = compute_trip_bound_settings();
    const uint32_t bound = settings.bound;
    if (!bound) return 0u;
    if (settings.only_program && program_address != settings.only_program) return 0u;

    // A PHASE SELECTOR IS REQUIRED, and this is a coherence requirement rather than ergonomics.
    //
    // A barrier-phased program emits one dispatcher per phase, and each phase has its OWN dispatch
    // table -- ordinal 9 means different guest pcs in phase 0 and phase 2. The witness is a single
    // record: if two phases can hit during one dispatch, its phase field is whichever invocation
    // stored last and its ordinal extrema are a mixture of two incompatible maps, which the host then
    // prints as one phase's range. There is no reading of that record that is true.
    //
    // Discovery still works with the bound armed and no phase chosen: every phase prints its dispatch
    // map (see the caller), so one run tells you how many phases exist and what each covers. Only the
    // emission is withheld.
    if (settings.only_phase == ComputeTripBoundSettings::kAllPhases) {
        static std::once_flag once;
        std::call_once(once, [] {
            fprintf(stderr,
                    "[cfg-trip-bound] PROSPER_CFG_TRIP_BOUND_PHASE is REQUIRED and is unset: no "
                    "bound emitted. One witness record cannot describe two phases -- their dispatch "
                    "ordinals index different tables. The dispatch maps below list every phase; "
                    "re-run with PROSPER_CFG_TRIP_BOUND_PHASE=<k>.\n");
        });
        return 0u;
    }
    if (phase != settings.only_phase) return 0u;

    // Refuse to instrument a program that uses GDS itself; see program_touches_guest_gds.
    if (program_touches_guest_gds(code, dwords)) {
        static std::mutex refused_mutex;
        static std::set<uint64_t> refused;
        bool first = false;
        {
            std::lock_guard lock(refused_mutex);
            first = refused.insert(program_address).second;
        }
        if (first)
            fprintf(stderr,
                    "[cfg-trip-bound] program 0x%llx REFUSED: it accesses GDS itself, and the "
                    "witness would overwrite its data. Not instrumented.\n",
                    static_cast<unsigned long long>(program_address));
        return 0u;
    }

    static std::mutex announce_mutex;
    static std::set<std::pair<uint64_t, uint32_t>> announced;
    bool first = false;
    {
        std::lock_guard lock(announce_mutex);
        first = announced.insert({program_address, phase}).second;
    }
    if (first) {
        char scope[128] = "every back-edge traversal";
        if (settings.only_ordinal != ComputeTripBoundSettings::kAllOrdinals)
            snprintf(scope, sizeof scope,
                     "ONLY traversals about to dispatch ordinal %u (see this phase's dispatch map)",
                     settings.only_ordinal);
        fprintf(stderr,
                "[cfg-trip-bound] program 0x%llx phase %u (guest pc %u..<%u, end-exclusive) "
                "bounded at %u iterations counting %s "
                "(DIAGNOSTIC: truncates guest control flow)\n",
                static_cast<unsigned long long>(program_address), phase, start_pc, end_pc, bound,
                scope);
    }
    return bound;
}

namespace {

// #3231 — is the CFG region's ENTRY-BLOCK VCC value dead?
//
// The dispatcher stores one value into `vcc_var` before its loop, then dispatches block 0 first,
// exactly once, with every invocation active (`selector = active ? pc : UINT32_MAX`, and
// `active_var` is seeded true when the caller has no partial-workgroup extent). So that stored
// value is observable only until block 0 overwrites it: if block 0 DEFINES the complete VCC pair
// before any instruction in it can read VCC, nothing anywhere in the region can see the entry
// value, and persisting `false` for it invents nothing. Block 0's own `save_state` publishes the
// real definition before the iteration's common phases run, so the two direct `vcc_var` readers
// (portable readlane into 106, and the vote-to-VCC merge) see it too.
//
// This is deliberately narrow, because the failure the caller's gate prevents is silent-wrong
// rather than a crash. What it does NOT admit:
//   * anything but a `v_cmp_*` (VOPC, never `v_cmpx_*`) whose destination is the VCC pair. That is
//     the one encoding that defines both words for every lane in a single instruction, and it is
//     the form the live evidence uses. A VOP3B carry-out into VCC, a 64-bit scalar write of the
//     pair, and `v_cmpx_*`'s EXEC write are all left rejected.
//   * a b32 write of vcc_lo or vcc_hi alone — half the pair would still carry the entry value, so
//     the scan stops there rather than continuing to a later full define.
//   * an entry block that reads VCC first, in ANY form. "Reads" is over-approximated: the implicit
//     consumers this file already enumerates for the mask-domain analyses, plus any operand that
//     can name a word of the pair — including a wide scalar read rooted low enough to reach s106.
//     An operand the decoder left stale is read too (all four source slots, not `n_src`), because
//     over-reading only ever moves the answer to "not dead".
//
// `lo`/`hi` are the entry block's half-open pc range as the dispatcher itself partitions it
// (`starts[0]` and `starts[1]`), so a block split by a branch target, or by one of the synchronized
// cross-lane events that each get their own block, shortens the window rather than widening it.
bool entry_block_defines_vcc_before_any_read(const std::vector<Rdna2Inst>& ins,
                                             uint32_t lo, uint32_t hi) {
    auto may_name_vcc = [](const Operand& operand) {
        if (operand.kind == OperandKind::Special)
            return operand.value == 106 || operand.value == 107;
        // The widest scalar operand is an eight-word T#, so a root as low as s99 still covers s106.
        // Treat every scalar operand at or above that root as touching the pair.
        if (operand.kind == OperandKind::SGPR) return operand.value >= 99;
        return false;
    };
    for (const auto& in : ins) {
        if (in.pc < lo) continue;
        if (in.pc >= hi || in.is_end) return false;
        // Sources before the define: the defining compare may itself read a VCC word as scalar data.
        for (const Operand& source : in.src)
            if (may_name_vcc(source)) return false;
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06 || in.opcode == 0x07))
            return false;                       // s_cbranch_vccz / s_cbranch_vccnz
        if (in.fmt == Rdna2Format::VOP2 &&
            (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a)))
            return false;                       // e32 cndmask and the carry-in/out forms
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            !(in.dst.kind == OperandKind::SGPR && in.dst.value <= 105))
            return true;                        // the define: v_cmp_* into VCC
        if (may_name_vcc(in.dst) || may_name_vcc(in.sdst)) return false;
    }
    return false;
}

}  // namespace

bool emit_cfg_state_machine(
    SpirvCompute& b, RegState& initial, const std::vector<Rdna2Inst>& ins,
    const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
    bool allow_exec_update, bool allow_smem,
    const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
    const uint32_t* code, size_t dwords, uint32_t initial_active = 0,
    bool synchronize_lds_fminmax = false) {
    const bool graphics = b.is_fragment || b.is_vertex;
    auto reject_cfg = [&](uint32_t pc, const char* reason) {
        log_recompile_diagnostic(b.diagnostic,
                                 b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
                                 "terminal", "pc=%u reason=%s", pc, reason);
        return false;
    };
    if ((!b.is_compute && !graphics) || ins.empty()) return false;
    // `safe` branches have already been proven equivalent to straight-line predication by the
    // stage-specific analysis (fragment alpha-test wave early-outs, safe EXECZ regions, and the
    // bounded NGG terminal export gate).  The compact SSA emitter feeds them to emit_alu, which
    // deliberately no-ops the scalar branch while retaining the per-invocation EXEC effect.  Do the
    // same in the CFG fallback: treating one as a basic-block terminator makes a kill-mask SCC look
    // like an ordinary scalar boolean, even though the mask lowering intentionally poisons that
    // cross-lane SCC.  Astro Bot's complex material PS combines both shapes and was rejected there.
    auto linearized_branch = [&](const Rdna2Inst& in) {
        return graphics && in.fmt == Rdna2Format::SOPP && safe.contains(in.pc);
    };
    auto cfg_terminator = [&](const Rdna2Inst& in) {
        if (in.is_end) return true;
        if (linearized_branch(in) || in.fmt != Rdna2Format::SOPP) return false;
        const bool branch = in.opcode >= 0x02 && in.opcode <= 0x09 &&
            in.opcode != 0x03;
        return branch || in.opcode == 0x12; // s_trap terminates the guest wave
    };

    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins) if (in.is_end) { end_pc = in.pc; break; }
    if (end_pc == UINT32_MAX) return false;
    auto proven_exit_target = [&](uint32_t target) {
        if (target <= end_pc || !code || target >= dwords) return false;
        std::vector<Rdna2Inst> tail;
        rdna2_walk(code + target, dwords - target, tail);
        for (const auto& in : tail) {
            if (in.is_end) return true;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x00) continue;
            break;
        }
        return false;
    };
    if (b.is_compute &&
        ((b.wave_size != 32 && b.wave_size != 64) || !b.local_count || b.local_count > 1024))
        return false;
    const bool has_synchronized_lds_store = synchronize_lds_fminmax &&
        !b.atomicized_lds_store_pcs.empty();
    const bool has_synchronized_lds_fminmax = synchronize_lds_fminmax &&
        std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::DS && !in.ds_gds &&
                (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32);
        });
    // The float-atomic event is bracketed by Workgroup barriers. Keep even an exact native subgroup
    // on the common dispatcher path so lane zero's ordinary initializer stores are published before
    // the first CAS and the final CAS completes before the later gather. Admission below limits this
    // synthesized ordering to one guest wave, so ended/trapped lanes can safely remain participants.
    const bool direct_dispatch = (graphics || b.native_subgroup_size) &&
        !has_synchronized_lds_store && !has_synchronized_lds_fminmax;
    const bool proven_wave32_masks = b.allow_b32_masks &&
        (b.is_fragment || (b.is_compute && b.wave_size == 32));
    const bool compute_scalar_vcc_bridge = allows_compute_scalar_vcc_bridge(b);
    const uint32_t wave_count = b.is_compute
        ? (b.local_count + b.wave_size - 1) / b.wave_size : 0;
    const uint32_t padded_lanes = wave_count * b.wave_size;

    // Discover every scalar pair that lives in the per-lane mask domain.  Besides defining the
    // dispatcher variables below, this identifies s_cmp_{eq,lg}_u64 mask,0: its SCC result is a
    // whole-wave reduction and therefore must be lowered in the common synchronized phase.
    std::set<int> static_mask_keys;
    for (const auto& kv : initial.sreg_bool) static_mask_keys.insert(kv.first);
    for (const auto& in : ins) {
        if (in.is_end) break;
        for_each_scalar_write(in, [&](int base, uint32_t) {
            const bool wave32_one_word_mask = proven_wave32_masks &&
                ((in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) ||
                 (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                  in.opcode <= 0x12a && base == in.sdst.value) ||
                 (vop3b_fresh_carry_output(in) && base == in.sdst.value) ||
                 (compute_scalar_vcc_bridge &&
                  is_scalar_cselect_b32_to_vcc_lo(in) && base == 106));
            if (wave32_one_word_mask)
                static_mask_keys.insert(base);
            else if (base <= 105 && scalar_write_is_b64_mask(in, base))
                static_mask_keys.insert(base);
        }, proven_wave32_masks);
    }
    // Keep block discovery independent of the mask-domain dataflow below.  In particular, a
    // physical Wave32 SGPR can hold a mask in one lifetime and ordinary scalar data in another.
    // Conservatively split every syntactically eligible zero comparison here, then consult the
    // block-entry RegState while emitting it to decide which lifetime is actually live.
    auto mask_zero_compare_candidate_source = [&](const Rdna2Inst& in) -> int {
        if (in.fmt != Rdna2Format::SOPC) return -1;
        auto zero = [](const Operand& o) {
            return o.kind == OperandKind::InlineInt && o.value == 0;
        };
        auto possible_mask = [](const Operand& o) {
            return o.kind == OperandKind::SGPR || o.kind == OperandKind::Special;
        };
        // B64 EQ/LG and these B32 unsigned zero comparisons depend only on whether any mask bit is
        // set. In particular, Wave32 code commonly uses `s_cmp_gt_u32 vcc_lo, 0` after VOPC.
        const bool b64_compare = in.opcode == 0x12 || in.opcode == 0x13;
        const bool b32_mask_first = proven_wave32_masks &&
            (in.opcode == 0x06 || in.opcode == 0x07 ||
             in.opcode == 0x08 || in.opcode == 0x0b);
        const bool b32_mask_second = proven_wave32_masks &&
            (in.opcode == 0x06 || in.opcode == 0x07 ||
             in.opcode == 0x09 || in.opcode == 0x0a);
        if (possible_mask(in.src[0]) && zero(in.src[1]) &&
            (b64_compare || b32_mask_first))
            return in.src[0].value;
        if (possible_mask(in.src[1]) && zero(in.src[0]) &&
            (b64_compare || b32_mask_second))
            return in.src[1].value;
        return -1;
    };
    auto mask_zero_compare_inverts = [&](const Rdna2Inst& in) {
        if (in.opcode == 0x12 || in.opcode == 0x06) return true; // EQ mask,0 / 0,mask
        const bool mask_first =
            (in.src[0].kind == OperandKind::SGPR ||
             in.src[0].kind == OperandKind::Special) &&
            mask_zero_compare_candidate_source(in) == in.src[0].value;
        return mask_first ? in.opcode == 0x0b                 // mask <= 0
                          : in.opcode == 0x09;                // 0 >= mask
    };

    // Syberia's fullscreen compute pass compares EXEC with a mask written by an explicit Wave64
    // VOPC destination (`s_cmp_lg_u64 exec,s[16:17]`).  Both values live only in the per-lane Bool
    // domain, so their scalar inequality is exactly ANY(EXEC xor saved_mask) over the guest wave.
    // Keep this deliberately narrower than a general B64 comparison: one operand must be the
    // architectural EXEC pair and the other an ordinary saved-mask SGPR pair whose live RegState
    // value is checked while emitting the dispatcher case.
    auto exec_saved_mask_compare_source = [&](const Rdna2Inst& in) -> int {
        if (!b.is_compute || b.wave_size != 64 || in.fmt != Rdna2Format::SOPC ||
            (in.opcode != 0x12 && in.opcode != 0x13))
            return -1;
        auto is_exec = [](const Operand& operand) {
            return (operand.kind == OperandKind::SGPR ||
                    operand.kind == OperandKind::Special) &&
                   operand.value == 126;
        };
        auto saved_mask = [](const Operand& operand) {
            return operand.kind == OperandKind::SGPR &&
                   operand.value >= 0 && operand.value <= 105;
        };
        if (is_exec(in.src[0]) && saved_mask(in.src[1])) return in.src[1].value;
        if (saved_mask(in.src[0]) && is_exec(in.src[1])) return in.src[0].value;
        return -1;
    };

    // Unity fragment programs also compare two ordinary saved B64 mask pairs.  Keep candidate
    // discovery syntactic so block splitting does not depend on a path-local register lifetime;
    // the MUST dataflow below separately proves that BOTH pairs are masks at the exact compare.
    // A numeric pair, an EXEC/VCC special pair, or a path-dependent mask/data join therefore does
    // not enter this lowering and falls back to the ordinary scalar emitter (or rejects visibly).
    auto saved_mask_pair_compare_sources = [&](const Rdna2Inst& in)
            -> std::array<int, 2> {
        if (in.fmt != Rdna2Format::SOPC ||
            (in.opcode != 0x12 && in.opcode != 0x13) ||
            in.src[0].kind != OperandKind::SGPR ||
            in.src[1].kind != OperandKind::SGPR ||
            in.src[0].value < 0 || in.src[0].value > 105 ||
            in.src[1].value < 0 || in.src[1].value > 105)
            return {-1, -1};
        return {in.src[0].value, in.src[1].value};
    };

    // House of the Dead 2 reduces an unsigned value in place across one architectural DPP row
    // immediately after the two saved-mask comparisons above. Keep this dispatcher escape hatch
    // exact: fragment V_MIN_U32, unbounded ROW_SHR, full-mask/no-modifier DPP16 (proved by the
    // decoder's has_dpp contract), and one physical VGPR used as VDST/SRC0/SRC1. The subgroup
    // operation itself is emitted in the common phase below so lanes taking other CFG cases still
    // participate without supplying a false neighbor.
    auto fragment_dpp_min_row_shr = [&](const Rdna2Inst& in) {
        return b.is_fragment && in.fmt == Rdna2Format::VOP2 &&
            in.opcode == 0x13 && in.has_dpp && !in.dpp_bound_ctrl &&
            in.dpp_ctrl >= 0x111u && in.dpp_ctrl <= 0x11fu &&
            in.dst.kind == OperandKind::VGPR &&
            in.src[0].kind == OperandKind::VGPR &&
            in.src[1].kind == OperandKind::VGPR &&
            in.dst.value == in.src[0].value &&
            in.dst.value == in.src[1].value;
    };

    // GTA V's compute reductions use an in-place V_ADD_NC_U32 ladder over each architectural
    // DPP16 row. Keep this contract as narrow as the observed packets: unbounded ROW_SHR with no
    // modifier/mask (the decoder's has_dpp contract), and one physical VGPR as VDST/SRC0/SRC1.
    // A native exact-wave dispatcher can execute the shuffle in its uniform switch case. The
    // portable dispatcher publishes it as an event below because a host subgroup may be narrower
    // than Wave64 and another guest wave can be parked at a different static instruction.
    auto compute_dpp_add_row_shr = [&](const Rdna2Inst& in) {
        return b.is_compute && is_inplace_vadd_nc_u32_dpp_row_shr(in);
    };

    // GTA V's MOV/MIN/MAX ROW_ROR:8 family has the same synchronization requirement as the add
    // ladder: exact native waves can shuffle in the uniform dispatcher case, while portable waves
    // publish an event-tagged source through workgroup scratch in the common phase.
    auto compute_dpp_row_ror8 = [&](const Rdna2Inst& in) {
        return b.is_compute && dpp_row_ror8_op(in) != DppRowRor8Op::None;
    };

    // The row reduction is followed by an identity QUAD_PERM whose partial ROW_MASK selects rows
    // 1 and 3. No value crosses lanes: selected EXEC-active lanes add their current VDST/SRC0 to a
    // distinct SRC1, while masked rows preserve VDST. Keeping this a dedicated dispatcher case
    // avoids granting arbitrary partial DPP masks to the generic ALU emitter.
    auto compute_dpp_add_row_mask = [&](const Rdna2Inst& in) {
        return b.is_compute && is_vadd_nc_u32_dpp_partial_row(in);
    };

    // VOPC e64 can compare a complete 64-bit scalar mask as integer data. Generated Wave64 code
    // uses `v_cmp_gt_u64 vcc,vcc,0` to broadcast (old VCC != 0) back into every active VCC lane.
    // The per-invocation mask representation has no 64-bit scalar payload, but the comparison is
    // exactly one guest-wave ANY vote. Keep the admission deliberately narrow: unsigned B64,
    // architectural VCC destination, one proven mask source, literal zero, and predicates whose
    // result is either ANY or !ANY. Other B64 arithmetic comparisons remain fail-visible.
    auto vopc_mask_zero_compare_source = [&](const Rdna2Inst& in) -> int {
        if (in.fmt != Rdna2Format::VOPC || vopc_is_cmpx(in.opcode) ||
            in.dst.kind != OperandKind::SGPR || in.dst.value != 106 ||
            in.src_abs[0] || in.src_abs[1] || in.src_neg[0] || in.src_neg[1])
            return -1;
        auto zero = [](const Operand& operand) {
            return operand.kind == OperandKind::InlineInt && operand.value == 0;
        };
        auto possible_mask = [](const Operand& operand) {
            return operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special;
        };
        const bool mask_first = in.opcode == 0xe2 || in.opcode == 0xe3 ||
                                in.opcode == 0xe4 || in.opcode == 0xe5;
        const bool mask_second = in.opcode == 0xe1 || in.opcode == 0xe2 ||
                                 in.opcode == 0xe5 || in.opcode == 0xe6;
        if (mask_first && possible_mask(in.src[0]) && zero(in.src[1]))
            return in.src[0].value;
        if (mask_second && zero(in.src[0]) && possible_mask(in.src[1]))
            return in.src[1].value;
        return -1;
    };
    auto vopc_mask_zero_compare_inverts = [&](const Rdna2Inst& in) {
        const bool mask_first = in.src[0].kind != OperandKind::InlineInt;
        return mask_first ? in.opcode == 0xe2 || in.opcode == 0xe3 // mask ==/<= 0
                          : in.opcode == 0xe2 || in.opcode == 0xe6; // 0 ==/>= mask
    };

    // A Wave32 B32 logical writes SCC=any(result mask). The ordinary lane-local emitter poisons
    // that SCC because it cannot reduce a guest wave by itself. Inside an exact native dispatcher,
    // however, every lane reaches the same switch case and an immediately consuming SCC branch can
    // use one exact subgroup vote. Restrict the vote to the last architectural SCC writer before
    // the branch; generated traversal kernels often chain several mask intersections and only the
    // final result is live, so voting after every intermediate AND would add needless hot-loop work.
    std::unordered_set<uint32_t> native_b32_mask_scc_vote_pcs;
    if (b.native_subgroup_size && proven_wave32_masks) {
        for (size_t i = 0; i + 1 < ins.size(); ++i) {
            const Rdna2Inst& producer = ins[i];
            const Rdna2Inst& consumer = ins[i + 1];
            if (producer.fmt == Rdna2Format::SOP2 &&
                sop2_is_b32_logical(producer.opcode) &&
                consumer.fmt == Rdna2Format::SOPP &&
                (consumer.opcode == 0x04 || consumer.opcode == 0x05) &&
                producer.pc + producer.len_dwords == consumer.pc)
                native_b32_mask_scc_vote_pcs.insert(producer.pc);
        }
    }

    // A B64 mask logical writes SCC=(result mask != 0). Find the producer only when that SCC is
    // actually consumed by a later scalar branch, walking backwards across instructions that are
    // architecturally SCC-preserving. This avoids a synchronized vote after every intermediate
    // mask operation in branch-heavy kernels while retaining fail-closed behavior across another
    // scalar ALU/control-flow instruction whose SCC effect is not proven here.
    auto b64_mask_logical_opcode = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP2 &&
               (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 ||
                in.opcode == 0x15 || in.opcode == 0x17 || in.opcode == 0x19 ||
                in.opcode == 0x1b || in.opcode == 0x1d);
    };
    std::unordered_set<uint32_t> scalar_block_starts{ins.front().pc};
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& terminator = ins[i];
        if (!cfg_terminator(terminator)) continue;
        if (terminator.fmt == Rdna2Format::SOPP && terminator.opcode >= 0x02 &&
            terminator.opcode <= 0x09 && terminator.opcode != 0x03) {
            const uint32_t target = scalar_branch_target(terminator);
            if (target <= end_pc) scalar_block_starts.insert(target);
        }
        if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
            scalar_block_starts.insert(ins[i + 1].pc);
    }
    std::unordered_set<uint32_t> b64_mask_scc_vote_pcs;
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& consumer = ins[i];
        if (consumer.is_end) break;
        if (consumer.fmt != Rdna2Format::SOPP ||
            (consumer.opcode != 0x04 && consumer.opcode != 0x05))
            continue;
        // SCC at a block entry may come from more than one predecessor. Do not associate a linear
        // producer across that join: this narrow proof owns only the branch's current basic block.
        size_t block_begin = i;
        while (block_begin > 0 && !scalar_block_starts.contains(ins[block_begin].pc))
            --block_begin;
        for (size_t j = i; j-- > block_begin;) {
            const Rdna2Inst& candidate = ins[j];
            if (b64_mask_logical_opcode(candidate)) {
                b64_mask_scc_vote_pcs.insert(candidate.pc);
                break;
            }
            const bool preserves_scc =
                candidate.fmt == Rdna2Format::VOP1 ||
                candidate.fmt == Rdna2Format::VOP2 ||
                candidate.fmt == Rdna2Format::VOP3 ||
                candidate.fmt == Rdna2Format::VOP3P ||
                candidate.fmt == Rdna2Format::VOPC ||
                candidate.fmt == Rdna2Format::SMEM ||
                candidate.fmt == Rdna2Format::MUBUF ||
                candidate.fmt == Rdna2Format::MTBUF ||
                candidate.fmt == Rdna2Format::MIMG ||
                candidate.fmt == Rdna2Format::DS ||
                candidate.fmt == Rdna2Format::FLAT ||
                candidate.fmt == Rdna2Format::EXP ||
                candidate.fmt == Rdna2Format::VINTRP ||
                (candidate.fmt == Rdna2Format::SOP1 &&
                 candidate.opcode == kSop1OpcodeMovB64) ||
                sopp_is_noop(candidate);
            if (!preserves_scc) break;
        }
    }

    // GTA V scans one physical dword of a saved Wave64 predicate with V_FFBH_U32. In the portable
    // dispatcher the predicate exists only as one Bool per guest lane; no host subgroup-width
    // contract exists from which operand_bits could form the complete SGPR word. Split every plain
    // SGPR-fed candidate syntactically, then use the path-filtered B64 mask state at emission time to
    // select the synchronized scratch phase below. Ordinary scalar-data inputs still use emit_alu.
    auto portable_mask_ffbh_candidate = [&](const Rdna2Inst& in) {
        return b.is_compute && b.wave_size == 64 && !b.native_subgroup_size &&
            in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeFfbhU32 &&
            in.src[0].kind == OperandKind::SGPR && !in.has_sdwa && !in.has_dpp;
    };

    // Split at every branch target/fallthrough and around every cross-lane operation. Case values are
    // dense block indices, not guest PCs. A cross-lane op must end its block so the common synchronized
    // phase can publish its result before any invocation advances to the following guest instruction.
    std::set<uint32_t> start_set{ins.front().pc};
    std::unordered_map<uint32_t, uint32_t> mbcnt_event_for_pc;
    std::unordered_map<uint32_t, uint32_t> append_event_for_pc;
    bool has_gds_append = false;
    bool has_lds_append = false;
    std::unordered_set<uint32_t> swizzle_pcs;
    std::unordered_map<uint32_t, uint32_t> bpermute_event_for_pc;
    std::unordered_set<uint32_t> fragment_dpp_min_row_shr_pcs;
    std::unordered_map<uint32_t, uint32_t> fragment_dpp_min_event_for_pc;
    std::set<int> fragment_dpp_min_row_shr_dsts;
    std::unordered_set<uint32_t> compute_dpp_add_row_shr_pcs;
    std::unordered_map<uint32_t, uint32_t> compute_dpp_add_event_for_pc;
    std::set<int> compute_dpp_add_row_shr_dsts;
    std::unordered_set<uint32_t> compute_dpp_row_ror8_pcs;
    std::unordered_map<uint32_t, uint32_t> compute_dpp_ror8_event_for_pc;
    std::set<int> compute_dpp_row_ror8_dsts;
    uint32_t next_compute_dpp_event = 1;
    std::unordered_set<uint32_t> compute_dpp_add_row_mask_pcs;
    std::unordered_map<uint32_t, uint32_t> portable_mask_ffbh_event_for_pc;
    std::set<int> portable_mask_ffbh_dsts;
    std::unordered_map<uint32_t, uint32_t> portable_readlane_event_for_pc;
    std::set<int> portable_readlane_dsts;
    // A VGPR written by V_WRITELANE is a scalar spill array. Its lane slots must remain in one
    // dispatcher case so the exact slot lowering can resolve them; only ordinary VGPR lifetimes
    // use the synchronized generic readlane phase.
    std::set<int> writelane_spill_arrays;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361)
            writelane_spill_arrays.insert(in.dst.value);
    }
    std::unordered_set<uint32_t> synchronized_lds_store_pcs;
    std::unordered_set<uint32_t> lds_fminmax_pcs;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 &&
            (in.opcode == 0x365 || in.opcode == 0x366)) {
            mbcnt_event_for_pc.emplace(in.pc,
                static_cast<uint32_t>(mbcnt_event_for_pc.size()));
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::DS && (in.opcode == 0x3d || in.opcode == 0x3e)) {
            append_event_for_pc.emplace(in.pc,
                static_cast<uint32_t>(append_event_for_pc.size()));
            if (in.ds_gds) has_gds_append = true;
            else has_lds_append = true;
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::DS && in.opcode == 0x35) {
            swizzle_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::DS && in.opcode == kDsOpcodeBpermuteB32) {
            if (!b.is_compute || in.ds_gds || !b.native_subgroup_size ||
                b.native_subgroup_size != b.wave_size)
                return reject_cfg(in.pc, "ds-bpermute-native-wave-contract");
            bpermute_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(bpermute_event_for_pc.size() + 1));
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (fragment_dpp_min_row_shr(in)) {
            fragment_dpp_min_row_shr_pcs.insert(in.pc);
            fragment_dpp_min_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(fragment_dpp_min_event_for_pc.size() + 1));
            fragment_dpp_min_row_shr_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (compute_dpp_add_row_shr(in)) {
            compute_dpp_add_row_shr_pcs.insert(in.pc);
            compute_dpp_add_event_for_pc.emplace(
                in.pc, next_compute_dpp_event++);
            compute_dpp_add_row_shr_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (compute_dpp_row_ror8(in)) {
            compute_dpp_row_ror8_pcs.insert(in.pc);
            compute_dpp_ror8_event_for_pc.emplace(
                in.pc, next_compute_dpp_event++);
            compute_dpp_row_ror8_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (compute_dpp_add_row_mask(in)) {
            compute_dpp_add_row_mask_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (portable_mask_ffbh_candidate(in)) {
            portable_mask_ffbh_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(portable_mask_ffbh_event_for_pc.size() + 1));
            portable_mask_ffbh_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (b.is_compute && !b.native_subgroup_size &&
            in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 &&
            !writelane_spill_arrays.contains(in.src[0].value)) {
            portable_readlane_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(portable_readlane_event_for_pc.size() + 1));
            portable_readlane_dsts.insert(in.dst.value);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (synchronize_lds_fminmax && b.atomicized_lds_store_pcs.contains(in.pc)) {
            if (!b.is_compute || in.fmt != Rdna2Format::DS || in.ds_gds ||
                b.local_count > b.wave_size)
                return reject_cfg(in.pc, "lds-store-common-phase-contract");
            synchronized_lds_store_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (synchronize_lds_fminmax && in.fmt == Rdna2Format::DS &&
            (in.opcode == kDsOpcodeMinF32 || in.opcode == kDsOpcodeMaxF32)) {
            if (!b.is_compute || in.ds_gds || b.local_count > b.wave_size ||
                (in.words[1] & 0xffff0000u) != 0u ||
                b.compute_pgm_rsrc1 == UINT32_MAX)
                return reject_cfg(in.pc, "lds-fminmax-common-phase-contract");
            lds_fminmax_pcs.insert(in.pc);
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (mask_zero_compare_candidate_source(in) >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (exec_saved_mask_compare_source(in) >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (saved_mask_pair_compare_sources(in)[0] >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (vopc_mask_zero_compare_source(in) >= 0) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (b64_mask_scc_vote_pcs.contains(in.pc)) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12) {
            start_set.insert(in.pc);
            if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc)
                start_set.insert(ins[i + 1].pc);
            continue;
        }
        if (linearized_branch(in) || in.fmt != Rdna2Format::SOPP ||
            in.opcode < 0x02 || in.opcode > 0x09 ||
            in.opcode == 0x03) continue;
        const uint32_t target = branch_target(in);
        if (target <= end_pc) start_set.insert(target);
        if (i + 1 < ins.size() && ins[i + 1].pc <= end_pc) start_set.insert(ins[i + 1].pc);
    }
    const bool has_portable_compute_dpp_add =
        !b.native_subgroup_size && !compute_dpp_add_row_shr_pcs.empty();
    const bool has_portable_compute_dpp_ror8 =
        !b.native_subgroup_size && !compute_dpp_row_ror8_pcs.empty();
    const bool has_portable_compute_dpp =
        has_portable_compute_dpp_add || has_portable_compute_dpp_ror8;
    // Portable DPP needs a full-width value beside an event/EXEC word for every invocation. The
    // first plane remains reusable by MBCNT/votes after DPP's trailing barrier; only shaders that
    // actually contain this event pay for the second plane.
    const uint32_t dpp_value_base = 0;
    const uint32_t dpp_metadata_base = padded_lanes;
    const uint32_t wave_result_base = padded_lanes +
        (has_portable_compute_dpp ? padded_lanes : 0u);
    const uint32_t group_active_slot = wave_result_base + wave_count;
    if (b.is_compute && !direct_dispatch &&
        !b.declare_cfg_scratch(group_active_slot + 1))
        return reject_cfg(ins.front().pc, "cfg-scratch-too-small");
    start_set.insert(end_pc);
    std::vector<uint32_t> starts(start_set.begin(), start_set.end());
    if (starts.empty() || starts.front() != ins.front().pc) return false;
    std::unordered_map<uint32_t, uint32_t> block_for_pc;
    for (uint32_t i = 0; i < starts.size(); ++i) block_for_pc[starts[i]] = i;

    // Persist only registers that the stream reads or writes, plus the caller's initialized inputs.
    std::set<int> vregs, sregs;
    loop_written_regs(ins, 0, end_pc, vregs, sregs);
    for (const auto& in : ins) {
        if (in.is_end) break;
        for (uint32_t k = 0; k < in.n_src; ++k) {
            const Operand& src = in.src[k];
            if (src.kind == OperandKind::VGPR) vregs.insert(src.value);
            else if (src.kind == OperandKind::SGPR ||
                     (src.kind == OperandKind::Special && src.value >= 106 && src.value <= 124))
                sregs.insert(src.value);
        }
        // 64-bit scalar compares encode only the low SGPR of each pair in their two source fields.
        if (in.fmt == Rdna2Format::SOPC && (in.opcode == 0x12 || in.opcode == 0x13))
            for (uint32_t k = 0; k < 2; ++k)
                if (in.src[k].kind == OperandKind::SGPR) sregs.insert(in.src[k].value + 1);
        if (in.fmt == Rdna2Format::SOP2 &&
            (in.opcode == 0x1f || in.opcode == 0x21) &&
            (in.src[0].kind == OperandKind::SGPR ||
             (in.src[0].kind == OperandKind::Special &&
              in.src[0].value >= 106 && in.src[0].value < 124)))
            sregs.insert(in.src[0].value + 1);
    }
    for (const auto& kv : initial.vreg) vregs.insert(kv.first);
    for (const auto& kv : initial.sreg) sregs.insert(kv.first);

    // #3133: every scalar the stream touches gets a Function variable that is STORED AS ZERO when
    // the architectural word has no value on this path, and reloaded as ordinary tracked data at
    // each block entry -- the same fabricated zero a CFG phi would invent, one indirection further
    // away. An entry-M0 token is precisely a word with no value, so it cannot survive a dispatcher
    // edge. There is no MUST analysis for it here (the sibling one, `entry_wave64_scalar_words`,
    // exists only on the Wave64 mask path), so take the MAY set -- every register any save in this
    // stream can leave holding the token -- and re-establish the TOKEN on those words at every
    // block entry, dropping the reloaded placeholder. Erasing alone would not do: an untracked
    // ordinary SGPR reads as `uconst(0)` with `ok` left true, so it would fabricate the same word
    // silently. With the token, a save and its restore inside one block still work, and anything
    // that crosses a dispatcher edge rejects loudly at its read.
    //
    // This over-rejects a register that the shader reuses for real data in a LATER block, which is
    // the price of having no MUST analysis here; it is the safe direction, and it can only cost a
    // shader that contains `s_mov_b32 sX, m0`, every one of which rejected outright before #3133.
    // The set is empty for every other shader, so nothing that compiles today can change.
    std::set<int> entry_m0_may_hold;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x03 &&
            in.src[0].kind == OperandKind::Special && in.src[0].value == 124 &&
            in.dst.value <= 105)
            entry_m0_may_hold.insert(in.dst.value);
    }

    // Most vector destinations are one static consecutive range. V_MOVRELD is different: M0 can
    // select any observable VGPR at or above VDST, exactly the range emit_alu updates. Analyses that
    // protect v_writelane spill lifetimes must invalidate that full range as well, or a dynamic
    // ordinary write can leave a stale mask-half proof attached to an erased slot.
    const int max_observable_vgpr = shader_max_vgpr(ins);
    auto for_each_possible_vector_write = [&](const Rdna2Inst& in, const auto& callback) {
        if (in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32) {
            for (int reg = in.dst.value; reg <= max_observable_vgpr; ++reg) callback(reg);
            return;
        }
        const uint32_t writes = rdna2_vgpr_write_count(in);
        for (uint32_t word = 0; word < writes; ++word)
            callback(in.dst.value + static_cast<int>(word));
    };

    // Direct user-data descriptors are deliberately absent from the initial scalar SSA map: their
    // identity lives in the resource table, and presence in sreg means shader code overwrote them.
    // The dispatcher persists every referenced SGPR in a Function variable, so recover that
    // distinction per block until the first static write; otherwise a direct V# such as UE4's s[8:11]
    // looks rewritten merely because the dispatcher loaded its zero-initialized backing variable.
    std::set<int> direct_descriptor_sregs;
    if (rt) {
        for (const auto& resource : rt->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            const uint32_t words = (resource.cls == ResourceClass::Texture ||
                                    resource.cls == ResourceClass::StorageImage) ? 8u : 4u;
            for (uint32_t word = 0; word < words; ++word)
                direct_descriptor_sregs.insert(static_cast<int>(resource.sgpr_base + word));
        }
    }

    // The dispatcher reloads scalar values from Function variables, so map membership cannot say
    // whether shader code has overwritten an entry-time descriptor. Compute a forward MAY-write set
    // for every reachable basic-block entry instead. MAY is intentional: if one reachable predecessor
    // overwrote the descriptor, falling back to entry metadata after the join would be wrong on that
    // path. Entry-rooted reachability excludes writes in dead blocks, while backedges participate in
    // the fixed point and prevent stale fallback on later loop iterations.
    std::vector<std::unordered_set<int>> scalar_writes(starts.size());
    std::vector<std::set<int>> vector_writes(starts.size());
    std::vector<std::set<int>> vector_reads(starts.size());
    std::vector<std::vector<uint32_t>> successors(starts.size());
    for (uint32_t block = 0; block < starts.size(); ++block) {
        const uint32_t lo = starts[block];
        const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
        std::set<int> ignored_scalar_writes;
        loop_written_regs(ins, lo, hi, vector_writes[block], ignored_scalar_writes);
        vector_reads[block] = vector_writes[block];
        bool reads_dynamic_vector_range = false;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const uint32_t words = rdna2_vgpr_source_span(in, source);
                if (words) {
                    for (uint32_t word = 0; word < words; ++word)
                        vector_reads[block].insert(
                            in.src[source].value + static_cast<int>(word));
                }
            }
            // Buffer/image packets have format- and dmask-dependent implicit register ranges (and
            // stores encode VDATA in the decoded destination field). Relative VGPR reads similarly
            // select from a runtime range. Keep those uncommon blocks fully conservative; ordinary
            // ALU/branch blocks can still avoid loading the rest of the vector register file.
            reads_dynamic_vector_range |=
                in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF ||
                in.fmt == Rdna2Format::MIMG || in.fmt == Rdna2Format::FLAT ||
                (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43);
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                const bool wave32_vop3b = proven_wave32_masks &&
                    ((in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                      in.opcode <= 0x12a) || vop3b_fresh_carry_output(in)) &&
                    base == in.sdst.value;
                const uint32_t effective_width = wave32_vop3b ? 1u : width;
                for (uint32_t word = 0; word < effective_width; ++word)
                    scalar_writes[block].insert(base + static_cast<int>(word));
            }, proven_wave32_masks);
            if (b.allow_b32_masks && in.fmt == Rdna2Format::VOPC &&
                !vopc_is_cmpx(in.opcode) && in.dst.kind == OperandKind::SGPR &&
                (in.dst.value == 106 || in.dst.value == 107))
                scalar_writes[block].insert(in.dst.value);
        }

        const Rdna2Inst* terminator = nullptr;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            if (cfg_terminator(in)) {
                terminator = &in;
                break;
            }
        }
        auto add_successor = [&](uint32_t pc) {
            auto next = block_for_pc.find(pc);
            if (pc <= end_pc && next != block_for_pc.end() &&
                std::find(successors[block].begin(), successors[block].end(), next->second) ==
                    successors[block].end())
                successors[block].push_back(next->second);
        };
        if (!terminator) {
            add_successor(hi);
        } else if (!terminator->is_end && terminator->opcode != 0x12) {
            add_successor(branch_target(*terminator));
            if (terminator->opcode != 0x02)
                add_successor(terminator->pc + terminator->len_dwords);
        }
        if (reads_dynamic_vector_range) vector_reads[block] = vregs;
    }

    // #3133's `entry_m0_may_hold` is a WHOLE-STREAM MAY set, and `load_state` stamps it at EVERY
    // dispatcher block entry -- the program's own ENTRY block included. Its comment predicted the
    // cost as "over-rejects a register that the shader reuses for real data in a LATER block"; the
    // entry block is the case where that is not a judgement call but provably wrong. No instruction
    // has run there, so no save can have executed, and every scalar still holds the inbound word the
    // hardware placed in it.
    //
    // Stray's compute program `0x300e390000` is the worked example (#3308; the title-screen
    // background it belongs to is #3126). It saves M0 into s14 at pc157 and pc274 -- and s14 is ALSO
    // the compute stage's workgroup-id X, so `v_lshl_add_u32 v11, s14, 3, v0` at pc4, the shader's
    // own global-thread-index computation four dwords into the program, read a token for a save 153
    // dwords AHEAD of it and the whole dispatch was skipped.
    //
    // Narrow it to an entry-rooted forward MAY dataflow over the same CFG the mask analyses below
    // walk. GEN is #3133's own save shape; KILL is any other scalar write to the word, applied in
    // instruction order so a save and a later overwrite inside one block end the lifetime the save
    // started. The join is a UNION and deliberately not the MUST/equality join the mask analyses
    // use: the token is a MAY property whose whole purpose is to reject, so a register holding it on
    // one edge and data on another must keep rejecting.
    std::vector<std::set<int>> entry_m0_in(starts.size());
    std::vector<bool> entry_m0_reachable(starts.size(), false);
    if (!entry_m0_may_hold.empty() && !starts.empty()) {
        entry_m0_reachable.front() = true;   // the entry block's in-set stays EMPTY by construction
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> tokens = entry_m0_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;
                // Kill before gen, so the save's own destination write does not erase the token it
                // is about to create.
                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    for (uint32_t word = 0; word < width; ++word)
                        tokens.erase(base + static_cast<int>(word));
                }, proven_wave32_masks);
                if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x03 &&
                    in.src[0].kind == OperandKind::Special && in.src[0].value == 124 &&
                    in.dst.value <= 105)
                    tokens.insert(in.dst.value);
            }
            for (const uint32_t successor : successors[block]) {
                if (!entry_m0_reachable[successor]) {
                    entry_m0_reachable[successor] = true;
                    entry_m0_in[successor] = tokens;
                    pending.push_back(successor);
                    continue;
                }
                std::set<int> joined = entry_m0_in[successor];
                joined.insert(tokens.begin(), tokens.end());
                if (joined != entry_m0_in[successor]) {
                    entry_m0_in[successor] = std::move(joined);
                    pending.push_back(successor);
                }
            }
        }
    }

    // Wave32 saved masks can be replaced by ordinary scalar-data lifetimes. The dispatcher reloads
    // a statically-shaped register file at every case, so carry both the one-word B32 mask domain
    // and B64 saved-mask domain as compile-time properties of each basic-block entry. This is exact
    // whenever all reachable predecessors agree. A disagreement means that the same physical SGPR
    // is a mask on one edge and scalar data on another; keep that genuinely dynamic type join
    // fail-visible.
    //
    // This is deliberately a MUST/equality analysis rather than a union: loading a stale Boolean on
    // the scalar-data edge would be a silent miscompile. Compiler-generated save/restore regions,
    // including Astro Bot's large Wave32 material loop, have identical domains at their joins.
    std::vector<std::set<int>> b32_mask_in(starts.size());
    std::vector<std::set<int>> b32_mask_ambiguous_in(starts.size());
    std::vector<std::set<int>> b64_mask_in(starts.size());
    std::vector<std::set<int>> b64_mask_ambiguous_in(starts.size());
    std::vector<bool> b32_mask_reachable(starts.size(), false);
    std::unordered_set<uint32_t> proven_saved_mask_pair_compare_pcs;
    if (b.allow_b32_masks && !starts.empty()) {
        b32_mask_in.front().insert(
            initial.sreg_bool_b32.begin(), initial.sreg_bool_b32.end());
        for (const auto& mask : initial.sreg_bool) {
            if (!initial.sreg_bool_b32.contains(mask.first) &&
                mask.first != 106 && mask.first != 107)
                b64_mask_in.front().insert(mask.first);
        }
        // `vcc` is the implicit VCC_LO mask even when no instruction has needed an explicit
        // sreg_bool[106] alias yet. Preserve that valid entry lifetime in the same physical-domain
        // analysis as explicit Wave32 masks. A zero SSA id instead means the caller currently has
        // ordinary scalar data in the VCC words, so it deliberately does not seed the mask domain.
        if (initial.vcc) b32_mask_in.front().insert(106);
        b32_mask_reachable.front() = true;
        auto implicit_vcc_mask_source = [](const Rdna2Inst& in) -> int {
            if (in.fmt == Rdna2Format::SOPP &&
                (in.opcode == 0x06 || in.opcode == 0x07))
                return 106; // s_cbranch_vccz/nz
            if (in.fmt == Rdna2Format::VOP2 &&
                (in.opcode == 0x01 ||
                 (in.opcode >= 0x28 && in.opcode <= 0x2a)))
                return 106; // e32 cndmask and carry-in/out forms use implicit VCC
            if (in.fmt == Rdna2Format::VOP3 &&
                (in.opcode == 0x101 ||
                 (in.opcode >= 0x128 && in.opcode <= 0x12a))) {
                const Operand& mask = in.src[2];
                if ((mask.kind == OperandKind::SGPR || mask.kind == OperandKind::Special) &&
                    (mask.value == 106 || mask.value == 107))
                    return mask.value;
            }
            if (in.fmt == Rdna2Format::VOP3 &&
                (in.opcode == 0x365 || in.opcode == 0x366)) {
                const Operand& mask = in.src[0];
                if ((mask.kind == OperandKind::SGPR || mask.kind == OperandKind::Special) &&
                    (mask.value == 106 || mask.value == 107))
                    return mask.value;
            }
            return -1;
        };
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> masks = b32_mask_in[block];
            std::set<int> ambiguous = b32_mask_ambiguous_in[block];
            std::set<int> b64_masks = b64_mask_in[block];
            std::set<int> b64_ambiguous = b64_mask_ambiguous_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;

                const auto compare_sources = saved_mask_pair_compare_sources(in);
                if (compare_sources[0] >= 0) {
                    const bool proven =
                        b64_masks.contains(compare_sources[0]) &&
                        b64_masks.contains(compare_sources[1]) &&
                        !b64_ambiguous.contains(compare_sources[0]) &&
                        !b64_ambiguous.contains(compare_sources[1]);
                    if (proven)
                        proven_saved_mask_pair_compare_pcs.insert(in.pc);
                    else
                        proven_saved_mask_pair_compare_pcs.erase(in.pc);
                }

                // The dispatcher may use a false backing value for a physical VCC lifetime that is
                // provably dead. Do not let that implementation placeholder become observable: every
                // instruction whose ISA encoding requires a VCC mask must see one on all incoming
                // paths. Explicit SGPR operands remain governed by the data/mask-domain checks below.
                const int implicit_vcc = implicit_vcc_mask_source(in);
                if (implicit_vcc >= 0 &&
                    (!masks.contains(implicit_vcc) || ambiguous.contains(implicit_vcc))) {
                    log_recompile_diagnostic(
                        b.diagnostic,
                        b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
                        "terminal", "pc=%u reason=missing-wave32-vcc-mask", in.pc);
                    return false;
                }

                bool reads_ambiguous = false;
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    if ((in.src[source].kind == OperandKind::SGPR ||
                         in.src[source].kind == OperandKind::Special)) {
                        const int reg = in.src[source].value;
                        reads_ambiguous |= ambiguous.contains(reg);
                        for (int base : b64_ambiguous)
                            reads_ambiguous |= reg == base || reg == base + 1;
                    }
                }
                if (in.fmt == Rdna2Format::SOPP &&
                    (in.opcode == 0x06 || in.opcode == 0x07) &&
                    ambiguous.contains(106))
                    reads_ambiguous = true;
                if (reads_ambiguous) {
                    log_recompile_diagnostic(
                        b.diagnostic,
                        b.is_compute ? "compute-cfg-reject" : "graphics-cfg-reject",
                        "terminal", "pc=%u reason=wave32-ambiguous-mask-read", in.pc);
                    return false;
                }

                bool writes_b32_mask = false;
                auto register_mask = [&](const Operand& source) {
                    // EXEC_HI is the architectural zero dword in Wave32, not the second half of
                    // a live mask. Keep this transfer classification aligned with emit_alu's
                    // scalar materialization so an ordinary value derived from EXEC_HI survives
                    // dispatcher save/reload boundaries as scalar data.
                    return (source.kind == OperandKind::Special && source.value == 126) ||
                        ((source.kind == OperandKind::SGPR ||
                          source.kind == OperandKind::Special) &&
                         (masks.contains(source.value) ||
                          b64_masks.contains(source.value)));
                };
                auto source_mask = [&](const Operand& source) {
                    return register_mask(source) || source.kind == OperandKind::InlineInt ||
                        source.kind == OperandKind::Literal;
                };
                if (in.fmt == Rdna2Format::SOP1 &&
                    (in.opcode == 0x03 || in.opcode == 0x07 || in.opcode == 0x09 ||
                     sop1_opcode_is_emitted_saveexec_b32(in.opcode))) {
                    const bool source_is_mask = register_mask(in.src[0]) ||
                        (in.dst.value == 126 && in.src[0].kind == OperandKind::InlineInt);
                    writes_b32_mask = source_is_mask && in.dst.value != 127 &&
                        (!sop1_opcode_is_emitted_saveexec_b32(in.opcode) ||
                         in.dst.value != 126);
                }
                if (in.fmt == Rdna2Format::SOP2 &&
                    (in.opcode == 0x0a || (in.opcode >= 0x0e &&
                                           in.opcode <= 0x1c && (in.opcode & 1u) == 0))) {
                    const bool scalar_vcc_bridge = compute_scalar_vcc_bridge &&
                        is_scalar_cselect_b32_to_vcc_lo(in);
                    const bool scalar_vcchi_packet = compute_scalar_vcc_bridge &&
                        b.native_subgroup_size == 32 &&
                        is_gtav_wave32_vcchi_scalar_packet(in);
                    const bool mask_domain = in.dst.value == 126 || in.src[0].value == 126 ||
                        in.src[1].value == 126 ||
                        (((in.src[0].kind == OperandKind::SGPR ||
                           in.src[0].kind == OperandKind::Special) &&
                          masks.contains(in.src[0].value))) ||
                        (((in.src[1].kind == OperandKind::SGPR ||
                           in.src[1].kind == OperandKind::Special) &&
                          masks.contains(in.src[1].value)));
                    writes_b32_mask = !scalar_vcchi_packet &&
                        (scalar_vcc_bridge ||
                         (mask_domain && source_mask(in.src[0]) &&
                          source_mask(in.src[1]) && in.dst.value != 127));
                }

                int b32_write_reg = writes_b32_mask ? in.dst.value : -1;
                const bool wave32_b64_vcc_mask_write =
                    in.fmt == Rdna2Format::SOP2 && in.dst.value == 106 &&
                    in.opcode >= 0x0f && in.opcode <= 0x1d &&
                    (in.opcode & 1u) == 1 &&
                    source_mask(in.src[0]) && source_mask(in.src[1]);
                if (wave32_b64_vcc_mask_write)
                    b32_write_reg = 106;
                if (in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x128 &&
                    in.opcode <= 0x12a && in.sdst.kind == OperandKind::SGPR) {
                    const Operand& carry_in = in.src[2];
                    if (register_mask(carry_in)) b32_write_reg = in.sdst.value;
                } else if (vop3b_fresh_carry_output(in)) {
                    b32_write_reg = in.sdst.value;
                }

                // B64 mask-shaped instructions are not sufficient to prove a live mask lifetime:
                // s_mov_b64 and the logical family also move ordinary 64-bit scalar data. Require
                // their sources to be masks in the current path state. Saveexec always writes OLD
                // EXEC to its explicit destination, while BFM and VOP3B construct fresh masks.
                bool writes_b64_mask = false;
                if (in.fmt == Rdna2Format::SOP1 && in.dst.value <= 105) {
                    if (in.opcode == 0x04 || in.opcode == 0x08 || in.opcode == 0x0a)
                        writes_b64_mask = source_mask(in.src[0]);
                    else if ((in.opcode >= 0x24 && in.opcode <= 0x2b) ||
                             in.opcode == 0x37 || in.opcode == 0x38)
                        writes_b64_mask = true;
                }
                if (in.fmt == Rdna2Format::SOP2 && in.dst.value <= 105) {
                    if (in.opcode == 0x25)
                        writes_b64_mask = true;
                    else if (in.opcode == 0x0b ||
                             (in.opcode >= 0x0f && in.opcode <= 0x1d &&
                              (in.opcode & 1u) == 1))
                        writes_b64_mask = source_mask(in.src[0]) && source_mask(in.src[1]);
                }
                const bool vop3_b32_carry = in.fmt == Rdna2Format::VOP3 &&
                    ((in.opcode >= 0x128 && in.opcode <= 0x12a) ||
                     vop3b_fresh_carry_output(in));
                // rdna2_decode populates `sdst` only for its explicit ten-opcode VOP3B whitelist:
                // add/sub carry, div-scale flag, and 64-bit multiply-add carry outputs. Each SDST
                // is a per-lane scalar mask/flag; ordinary VOP3A scalar data never appears here
                // (v_readlane uses `dst`, not `sdst`).
                if (vop3_writes_mask_sdst(in) && !vop3_b32_carry &&
                    in.sdst.value <= 105)
                    writes_b64_mask = true;
                int b64_write_reg = writes_b64_mask ?
                    (in.fmt == Rdna2Format::VOP3 ? in.sdst.value : in.dst.value) : -1;

                auto erase_b64_overlapping = [&](int base, uint32_t width) {
                    auto erase = [&](std::set<int>& values) {
                        for (auto it = values.begin(); it != values.end();) {
                            const int mask_base = *it;
                            if (base < mask_base + 2 && mask_base < base + static_cast<int>(width))
                                it = values.erase(it);
                            else
                                ++it;
                        }
                    };
                    erase(b64_masks);
                    erase(b64_ambiguous);
                };

                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    const bool one_word_write = base == b32_write_reg;
                    const uint32_t effective_width = one_word_write ? 1u : width;
                    erase_b64_overlapping(base, effective_width);
                    for (uint32_t word = 0; word < effective_width; ++word) {
                        const int reg = base + static_cast<int>(word);
                        if (!one_word_write || reg != b32_write_reg) {
                            masks.erase(reg);
                            ambiguous.erase(reg);
                        }
                    }
                    if (one_word_write && b32_write_reg != 126) {
                        masks.insert(b32_write_reg);
                        ambiguous.erase(b32_write_reg);
                    }
                }, proven_wave32_masks);

                if (b64_write_reg >= 0) {
                    for (int word = 0; word < 2; ++word) {
                        masks.erase(b64_write_reg + word);
                        ambiguous.erase(b64_write_reg + word);
                    }
                    b64_masks.insert(b64_write_reg);
                    b64_ambiguous.erase(b64_write_reg);
                }

                // Every non-CMPX VOPC destination is a one-word mask in proven Wave32, including
                // explicit ordinary SGPRs. CMPX writes EXEC only and establishes no VCC lifetime.
                if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
                    const int destination = in.dst.kind == OperandKind::SGPR ? in.dst.value : 106;
                    erase_b64_overlapping(destination, 1);
                    masks.insert(destination);
                    ambiguous.erase(destination);
                }
            }

            for (uint32_t successor : successors[block]) {
                if (!b32_mask_reachable[successor]) {
                    b32_mask_reachable[successor] = true;
                    b32_mask_in[successor] = masks;
                    b32_mask_ambiguous_in[successor] = ambiguous;
                    b64_mask_in[successor] = b64_masks;
                    b64_mask_ambiguous_in[successor] = b64_ambiguous;
                    pending.push_back(successor);
                } else {
                    std::set<int> joined_masks;
                    std::set<int> joined_ambiguous = b32_mask_ambiguous_in[successor];
                    std::set_intersection(
                        b32_mask_in[successor].begin(), b32_mask_in[successor].end(),
                        masks.begin(), masks.end(),
                        std::inserter(joined_masks, joined_masks.end()));
                    for (int reg : b32_mask_in[successor])
                        if (!masks.contains(reg)) joined_ambiguous.insert(reg);
                    for (int reg : masks)
                        if (!b32_mask_in[successor].contains(reg)) joined_ambiguous.insert(reg);
                    joined_ambiguous.insert(ambiguous.begin(), ambiguous.end());
                    for (int reg : joined_ambiguous) joined_masks.erase(reg);
                    std::set<int> joined_b64_masks;
                    std::set<int> joined_b64_ambiguous = b64_mask_ambiguous_in[successor];
                    std::set_intersection(
                        b64_mask_in[successor].begin(), b64_mask_in[successor].end(),
                        b64_masks.begin(), b64_masks.end(),
                        std::inserter(joined_b64_masks, joined_b64_masks.end()));
                    for (int reg : b64_mask_in[successor])
                        if (!b64_masks.contains(reg)) joined_b64_ambiguous.insert(reg);
                    for (int reg : b64_masks)
                        if (!b64_mask_in[successor].contains(reg)) joined_b64_ambiguous.insert(reg);
                    joined_b64_ambiguous.insert(
                        b64_ambiguous.begin(), b64_ambiguous.end());
                    for (int reg : joined_b64_ambiguous) joined_b64_masks.erase(reg);
                    if (joined_masks != b32_mask_in[successor] ||
                        joined_ambiguous != b32_mask_ambiguous_in[successor] ||
                        joined_b64_masks != b64_mask_in[successor] ||
                        joined_b64_ambiguous != b64_mask_ambiguous_in[successor]) {
                        b32_mask_in[successor] = std::move(joined_masks);
                        b32_mask_ambiguous_in[successor] = std::move(joined_ambiguous);
                        b64_mask_in[successor] = std::move(joined_b64_masks);
                        b64_mask_ambiguous_in[successor] = std::move(joined_b64_ambiguous);
                        pending.push_back(successor);
                    }
                }
            }
        }
        for (const auto& masks : b32_mask_in)
            for (int reg : masks)
                if (reg <= 107) static_mask_keys.insert(reg);
        for (const auto& masks : b64_mask_in)
            for (int reg : masks)
                if (reg <= 105) static_mask_keys.insert(reg);
    }

    // Wave64 dispatcher Bool variables hold values, not lifetime tags. A scalar overwrite stores
    // false for a dead mask, and an unfiltered load at a later case can therefore make mere map
    // membership look like a valid saved mask. Prove the B64 mask domain separately at every
    // Wave64 whole-mask comparisons (mask-vs-zero, EXEC-vs-saved or saved-vs-saved): mask producers
    // generate the fact, every overlapping half/pair scalar write kills it, and joins retain it only
    // when all reachable predecessors agree. Architectural VCC lives in `state.vcc`, not in the
    // saved-mask map, so this proof is also the lifetime tag that lets the dispatcher distinguish a
    // live VCC mask from a physical VCC pair that has been recycled as scalar data.
    std::unordered_set<uint32_t> proven_wave64_mask_zero_compare_pcs;
    std::unordered_set<uint32_t> proven_exec_saved_mask_compare_pcs;
    std::unordered_set<uint32_t> proven_wave64_mask_reduction_pcs;
    std::unordered_map<uint32_t, int> proven_wave64_mbcnt_mask_root_for_pc;
    // Retain the entry facts beyond the consumer-specialization pass below. Function Bool
    // variables persist values only; this MUST set is the separate lifetime tag load_state needs
    // before reconstructing a saved-mask RegState entry in each dispatcher case. Domain conflicts
    // at joins remain ambiguous until a definite overwrite and reject on their first read.
    std::vector<std::set<int>> wave64_b64_mask_in(starts.size());
    std::vector<std::set<int>> wave64_b64_ambiguous_in(starts.size());
    // A mask/scalar join poisons the physical pair, but a later definite B32 write still makes the
    // addressed scalar word safe to consume. Track those post-conflict scalar definitions with a
    // separate MUST fact: the other half remains ambiguous until it too is definitely replaced.
    // This matters for GTA V's scalar scratch in VCC/ordinary mask pairs, where readfirstlane,
    // SMEM, or a B32 scalar ALU defines one half before a one-dword VALU/SALU consumer.
    std::vector<std::set<int>> wave64_scalar_word_in(starts.size());
    // Dispatcher Function variables persist SCC's Boolean value but not whether that value is an
    // architectural SCC or the false placeholder stored for an unrepresentable wave-mask result.
    // Carry a separate CFG MUST-validity bit and use it both for scalar-word provenance and when
    // reconstructing RegState at each dispatcher case.
    std::vector<bool> wave64_scalar_scc_valid_in(starts.size(), false);
    std::vector<bool> wave64_b64_reachable(starts.size(), false);
    auto wave64_mask_reduction_source = [&](const Rdna2Inst& in) -> int {
        if (!b.is_compute || b.wave_size != 64 || in.fmt != Rdna2Format::SOP1 ||
            (in.opcode != 0x10 && in.opcode != 0x14))
            return -1;
        if (in.src[0].kind == OperandKind::SGPR &&
            in.src[0].value >= 0 && in.src[0].value <= 105)
            return in.src[0].value;
        if (in.src[0].kind == OperandKind::Special && in.src[0].value == 106)
            return 106;
        // Architectural EXEC is a complete B64 mask source. emit_alu already materializes this
        // reduction from EXEC directly, but the MUST dataflow below has to agree, or the result is
        // a scalar the dispatcher erases at the next block entry -- see the note at the
        // exact_mask_reduction use. The exactness precondition is emit_alu's own: the ballot equals
        // the guest wave mask only when the native subgroup IS the guest wave.
        //
        // This covers BOTH opcodes admitted above, not only the population count: 0x10 is
        // s_bcnt1_i32_b64 and 0x14 is s_ff1_i32_b64, and `s_ff1_i32_b64 <- exec` (find-first-set over
        // the active mask) reaches this same return. The committed fixture exercises it at dword 336,
        // immediately before the bcnt at 337, so the wider coverage is tested rather than incidental.
        //
        // CONFIDENCE: HIGH — EXEC as a B64 mask source for these two reductions is architectural
        // (RDNA2 ISA 70648 §5.3/§12.2: both take a 64-bit scalar source, and EXEC is a legal SSRC),
        // and the wave-size guard is the same precondition emit_alu already relies on rather than a
        // new assumption. What is NOT covered is a 32-wide native subgroup, where the ballot is not
        // the guest wave: that returns -1 and the dispatcher declines, loudly, via
        // [subgroup-width] ... DISABLED. See #2429.
        if (in.src[0].kind == OperandKind::Special && in.src[0].value == 126 &&
            b.native_subgroup_size == b.wave_size)
            return 126;
        return -1;
    };
    auto wave64_mbcnt_mask_root = [&](const Rdna2Inst& in) -> int {
        if (!b.is_compute || b.wave_size != 64 || in.fmt != Rdna2Format::VOP3 ||
            (in.opcode != 0x365 && in.opcode != 0x366) ||
            in.src[0].kind != OperandKind::SGPR)
            return -1;
        // LOW names the mask pair's low word. HIGH names its high word, while the Bool-domain
        // value remains keyed by the low word. Architectural EXEC/VCC use Special operands and
        // continue through mbcnt_source_bit without this saved-SGPR lifetime proof.
        const int root = in.opcode == 0x366 ? in.src[0].value - 1 : in.src[0].value;
        return root >= 0 && root <= 105 ? root : -1;
    };
    if ((b.is_compute || b.is_fragment) && b.wave_size == 64 && !starts.empty()) {
        for (const auto& mask : initial.sreg_bool)
            if (!initial.sreg_bool_b32.contains(mask.first) && mask.first <= 105)
                wave64_b64_mask_in.front().insert(mask.first);
        if (initial.vcc) wave64_b64_mask_in.front().insert(106);
        for (const auto& value : initial.sreg)
            if (value.first <= 124) wave64_scalar_word_in.front().insert(value.first);
        for (const auto& value : initial.sreg_input)
            if (value.first <= 124) wave64_scalar_word_in.front().insert(value.first);
        // Direct descriptors intentionally live outside RegState's ordinary scalar map, but their
        // entry words are real scalar data until shader code overwrites them. Seed the MUST facts
        // from the same resource-table ranges used by the emitter's direct-descriptor fallback.
        for (int reg : direct_descriptor_sregs)
            if (reg <= 124) wave64_scalar_word_in.front().insert(reg);
        wave64_scalar_scc_valid_in.front() = initial.scc != 0;
        wave64_b64_reachable.front() = true;

        enum class ScalarSourceRead : uint8_t {
            None = 0, B32 = 1, Pair = 2, Quad = 4, Oct = 8,
        };
        auto scalar_source_read = [&](const Rdna2Inst& in, uint32_t source)
                -> ScalarSourceRead {
            const uint32_t alu_words = scalar_alu_source_words(in, source);
            if (alu_words == UINT32_MAX) return ScalarSourceRead::None;
            if (alu_words)
                return static_cast<ScalarSourceRead>(alu_words);
            switch (in.fmt) {
                case Rdna2Format::SMEM:
                    // Ordinary scalar memory uses a two-word base. S_BUFFER_* opcodes use a
                    // complete four-word descriptor; SOFFSET remains one scalar dword.
                    if (source == 1) return ScalarSourceRead::B32;
                    return in.opcode >= 0x08
                        ? ScalarSourceRead::Quad : ScalarSourceRead::Pair;
                case Rdna2Format::MUBUF:
                case Rdna2Format::MTBUF:
                    // The four-word V# starts at SRC1 while optional SOFFSET is one dword.
                    return source == 1 ? ScalarSourceRead::Quad : ScalarSourceRead::B32;
                case Rdna2Format::MIMG: {
                    // T# is eight words; the exact BVH form uses a four-word descriptor instead.
                    // Storage/load/resinfo/BVH packets have no sampler, while sampled operations
                    // consume the complete four-word S#.
                    if (source == 1)
                        return in.opcode == 0xe6
                            ? ScalarSourceRead::Quad : ScalarSourceRead::Oct;
                    const bool storage_only_op = in.opcode == 0x08 || in.opcode == 0x09 ||
                        in.opcode == 0x0f ||
                        (in.opcode >= 0x11 && in.opcode <= 0x1a && in.opcode != 0x13);
                    if (source == 2)
                        return in.opcode == 0x00 || in.opcode == 0x01 ||
                                   in.opcode == 0x0e || in.opcode == 0xe6 || storage_only_op
                            ? ScalarSourceRead::None : ScalarSourceRead::Quad;
                    return ScalarSourceRead::B32;
                }
                default:
                    return ScalarSourceRead::Pair;
            }
        };

        auto advance_wave64_b64_masks = [&](std::set<int>& masks,
                                            std::set<int>& ambiguous,
                                            std::set<int>& scalar_words,
                                            bool& scalar_scc,
                                            const Rdna2Inst& in,
                                            bool record_compare) {
            auto source_is_scalar_word = [&](const Operand& source) {
                switch (source.kind) {
                    case OperandKind::InlineInt:
                    case OperandKind::InlineFloat:
                    case OperandKind::Literal:
                        return true;
                    case OperandKind::SGPR:
                        return scalar_words.contains(source.value);
                    case OperandKind::Special:
                        if (source.value == 125) return true; // SGPR_NULL
                        if (source.value == 253) return scalar_scc;
                        return source.value >= 106 && source.value <= 124 &&
                               scalar_words.contains(source.value);
                    default:
                        return false;
                }
            };
            auto reads_scc = [](const Rdna2Inst& candidate) {
                return (candidate.fmt == Rdna2Format::SOP2 &&
                        (candidate.opcode == 0x04u || candidate.opcode == 0x05u ||
                         candidate.opcode == kSop2OpcodeCselectB32 ||
                         candidate.opcode == 0x0bu)) ||
                       (candidate.fmt == Rdna2Format::SOP1 &&
                        (candidate.opcode == kSop1OpcodeCmovB32 ||
                         candidate.opcode == kSop1OpcodeCmovB64)) ||
                       (candidate.fmt == Rdna2Format::SOPK &&
                        candidate.opcode == kSopkOpcodeCmovkI32);
            };
            const bool valid_scc_read = !reads_scc(in) || scalar_scc;
            const bool b32_vcc_scalar_write =
                is_wave64_vcc_lo_scalar_cselect(in) ||
                (in.fmt == Rdna2Format::SOP2 &&
                 (in.dst.value == 106 || in.dst.value == 107) &&
                 in.opcode >= 0x0e && in.opcode <= 0x1c &&
                 (in.opcode & 1u) == 0);
            const bool b32_vcc_scalar_result = b32_vcc_scalar_write &&
                source_is_scalar_word(in.src[0]) && source_is_scalar_word(in.src[1]);
            const int b32_vcc_sibling = in.dst.value == 106 ? 107 : 106;
            const bool b32_vcc_complete_scalar_pair =
                b32_vcc_scalar_result && scalar_words.contains(b32_vcc_sibling);
            if (record_compare && b32_vcc_scalar_write) {
                if (b32_vcc_scalar_result)
                    b.vcc_b32_scalar_result_pcs.insert(in.pc);
                else
                    b.vcc_b32_scalar_result_pcs.erase(in.pc);
                if (b32_vcc_complete_scalar_pair)
                    b.vcc_b32_scalar_pair_pcs.insert(in.pc);
                else
                    b.vcc_b32_scalar_pair_pcs.erase(in.pc);
            }
            // A block-entry join where the same physical pair is a mask on one predecessor and
            // scalar data on another has no runtime type tag. Reject the first observable read;
            // loading either the Bool's false placeholder or the scalar variable's zero placeholder
            // would silently choose one predecessor's domain for both paths.
            bool reads_ambiguous = false;
            auto source_is_mask = [&](const Operand& source) {
                if (source.kind == OperandKind::InlineInt) return true;
                if (source.kind != OperandKind::SGPR &&
                    source.kind != OperandKind::Special)
                    return false;
                if (source.value == 126 || source.value == 127) return true;
                return masks.contains(source.value);
            };
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const Operand& operand = in.src[source];
                if (operand.kind != OperandKind::SGPR &&
                    operand.kind != OperandKind::Special)
                    continue;
                // The mask-only emitter for B64 logical operations consumes a proved Bool-domain
                // source directly. A fresh VCC pair can overlap the high word of an older odd-rooted
                // ambiguity without becoming scalar data; only the scalar-pair operand below needs
                // per-word resolution. MBCNT likewise consumes one proved mask half, not a scalar
                // pair beginning at its encoded SGPR.
                const bool b64_logical_mask_source =
                    in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
                    in.opcode <= 0x1d && (in.opcode & 1u) && source_is_mask(operand);
                const int mbcnt_root = in.opcode == 0x366
                    ? operand.value - 1 : operand.value;
                const bool mbcnt_mask_source = in.fmt == Rdna2Format::VOP3 && source == 0 &&
                    (in.opcode == 0x365 || in.opcode == 0x366) &&
                    (operand.value == (in.opcode == 0x366 ? 127 : 126) ||
                     operand.value == (in.opcode == 0x366 ? 107 : 106) ||
                     masks.contains(mbcnt_root));
                if (b64_logical_mask_source || mbcnt_mask_source) continue;
                const ScalarSourceRead read = scalar_source_read(in, source);
                if (read == ScalarSourceRead::None) continue;
                const int first = operand.value;
                const int last = first + static_cast<int>(read);
                for (int base : ambiguous) {
                    const int overlap_first = std::max(first, base);
                    const int overlap_last = std::min(last, base + 2);
                    for (int word = overlap_first; word < overlap_last; ++word)
                        if (!scalar_words.contains(word)) reads_ambiguous = true;
                }
            }
            const bool implicit_vcc_read =
                (in.fmt == Rdna2Format::SOPP &&
                 (in.opcode == 0x06 || in.opcode == 0x07)) ||
                (in.fmt == Rdna2Format::VOP2 &&
                 (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a)));
            reads_ambiguous |= implicit_vcc_read && ambiguous.contains(106);
            const uint32_t implicit_scalar_words =
                scalar_implicit_destination_read_width(in);
            if (implicit_scalar_words) {
                const int first = in.dst.value;
                const int last = first + static_cast<int>(implicit_scalar_words);
                for (int word = first; word < last; ++word)
                    for (int mask_base : ambiguous)
                        if ((word == mask_base || word == mask_base + 1) &&
                            !scalar_words.contains(word))
                            reads_ambiguous = true;
            }
            if (reads_ambiguous)
                return reject_cfg(in.pc, "wave64-ambiguous-mask-read");

            const int reduction_source = wave64_mask_reduction_source(in);
            // EXEC needs no saved-mask lifetime: it is architectural state that always holds a
            // live mask, so it is exact wherever the source helper admits it.
            const bool exact_mask_reduction =
                reduction_source == 126 ||
                (reduction_source >= 0 && masks.contains(reduction_source));
            if (record_compare && exact_mask_reduction)
                proven_wave64_mask_reduction_pcs.insert(in.pc);
            const int mbcnt_root = wave64_mbcnt_mask_root(in);
            if (record_compare && mbcnt_root >= 0 && masks.contains(mbcnt_root))
                proven_wave64_mbcnt_mask_root_for_pc.emplace(in.pc, mbcnt_root);
            const int zero_compare_source = mask_zero_compare_candidate_source(in);
            if (record_compare && zero_compare_source >= 0 &&
                masks.contains(zero_compare_source))
                proven_wave64_mask_zero_compare_pcs.insert(in.pc);
            const int compare_source = exec_saved_mask_compare_source(in);
            if (record_compare && compare_source >= 0 && masks.contains(compare_source))
                proven_exec_saved_mask_compare_pcs.insert(in.pc);
            const auto pair_compare = saved_mask_pair_compare_sources(in);
            if (record_compare && pair_compare[0] >= 0 &&
                masks.contains(pair_compare[0]) && masks.contains(pair_compare[1]))
                proven_saved_mask_pair_compare_pcs.insert(in.pc);
            const bool writes_exact_wave_scc =
                (zero_compare_source >= 0 && masks.contains(zero_compare_source)) ||
                (compare_source >= 0 && masks.contains(compare_source)) ||
                (pair_compare[0] >= 0 && masks.contains(pair_compare[0]) &&
                 masks.contains(pair_compare[1]));

            // Scalar-data presence and SCC validity share the same provenance contract. SOPK
            // exposes its old SDST only as an implicit source; accepting ADDK, CMPK, or CMOVK after
            // a dispatcher reload therefore requires that exact word to be a MUST scalar value.
            // Check every dword of B64 inputs before the destination transfer ends the old lifetime.
            auto source_is_scalar_range = [&](const Operand& source, uint32_t width) {
                if (source.kind == OperandKind::InlineInt ||
                    source.kind == OperandKind::InlineFloat ||
                    source.kind == OperandKind::Literal)
                    return true;
                if (source.kind != OperandKind::SGPR &&
                    source.kind != OperandKind::Special)
                    return false;
                if (source.kind == OperandKind::Special && source.value == 125)
                    return true; // SGPR_NULL
                if (source.kind == OperandKind::Special && source.value == 253)
                    return width == 1 && scalar_scc;
                if (source.kind == OperandKind::Special &&
                    (source.value == 126 || source.value == 127))
                    return width == 1 && b.is_compute && b.wave_size == 64 &&
                        b.native_subgroup_size == 64;
                for (uint32_t word = 0; word < std::max(width, 1u); ++word)
                    if (!scalar_words.contains(source.value + static_cast<int>(word)))
                        return false;
                return true;
            };
            bool scalar_sources = true;
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const uint32_t width = scalar_alu_source_words(in, source);
                if (width != UINT32_MAX)
                    scalar_sources &= source_is_scalar_range(in.src[source], width);
            }
            bool implicit_scalar_source = true;
            if (implicit_scalar_words) {
                implicit_scalar_source = in.dst.kind == OperandKind::SGPR;
                for (uint32_t word = 0; word < implicit_scalar_words; ++word)
                    implicit_scalar_source &=
                        scalar_words.contains(in.dst.value + static_cast<int>(word));
            }
            const bool scalar_alu_result = scalar_sources && implicit_scalar_source;

            const bool wave64_vcc_b32_mask_not =
                in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeNotB32 &&
                (in.dst.value == 106 || in.dst.value == 107) &&
                in.src[0].value == in.dst.value && masks.contains(106) &&
                !scalar_words.contains(in.src[0].value);
            const bool exact_quadmask =
                in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeQuadmaskB64 &&
                b.wave_size == 64 && source_is_mask(in.src[0]) &&
                (b.is_fragment || (b.is_compute && b.native_subgroup_size == 64));

            int mask_write = -1;
            if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
                mask_write = in.dst.kind == OperandKind::SGPR && in.dst.value <= 105
                    ? in.dst.value : 106;
            } else if (in.fmt == Rdna2Format::SOP1 && in.dst.value <= 107) {
                if (wave64_vcc_b32_mask_not)
                    mask_write = 106;
                else if ((in.opcode == 0x04 || in.opcode == 0x08 || in.opcode == 0x0a) &&
                    source_is_mask(in.src[0]))
                    mask_write = in.dst.value;
                else if ((in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                          in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
                         in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
                         in.opcode == kSop1OpcodeOrn1SaveexecB64 || exact_quadmask)
                    mask_write = in.dst.value;
            } else if (in.fmt == Rdna2Format::SOP2 && in.dst.value <= 107) {
                if (b32_vcc_complete_scalar_pair)
                    mask_write = 106;
                else if (in.opcode == kSop2OpcodeBfmB64)
                    mask_write = in.dst.value;
                else if (in.opcode >= 0x0f && in.opcode <= 0x1d &&
                         (in.opcode & 1u) == 1)
                    // Every B64 logical has a Bool-domain result. Exact native Wave64 additionally
                    // materializes that result's ballot words below, but its mask lifetime remains
                    // the primary classification here.
                    mask_write = in.dst.value;
                else if (valid_scc_read && in.opcode == 0x0b && in.dst.value == 106 &&
                         !b.cselect_b64_low_only_pcs.contains(in.pc))
                    // A complete scalar-data pair selected into VCC has a dual lifetime: emit_alu
                    // derives its per-lane predicate even though neither input is a mask. The one
                    // incomplete GTA form deliberately has no predicate and is excluded here.
                    mask_write = 106;
                else if (valid_scc_read && in.opcode == 0x0b &&
                         source_is_mask(in.src[0]) && source_is_mask(in.src[1]))
                    mask_write = in.dst.value;
            } else if (vop3_writes_mask_sdst(in) && in.sdst.value <= 107) {
                mask_write = in.sdst.value;
            }

            auto erase_overlapping = [&](int base, uint32_t width) {
                for (auto it = masks.begin(); it != masks.end();) {
                    const int mask_base = *it;
                    if (base < mask_base + 2 &&
                        mask_base < base + static_cast<int>(width))
                        it = masks.erase(it);
                    else
                        ++it;
                }
                // A one-word scalar write kills the definite-mask fact for the physical pair, but
                // it resolves only the addressed half of an ambiguous pair. Keep the ambiguity
                // until one definite write covers both halves; otherwise the untouched word could
                // be reloaded from the wrong Function-variable domain. The separate scalar-word
                // MUST facts below can validate exactly the overwritten word while complete-pair
                // consumers remain fail-closed until both halves are definitely scalar.
                for (auto it = ambiguous.begin(); it != ambiguous.end();) {
                    const int mask_base = *it;
                    if (base <= mask_base &&
                        base + static_cast<int>(width) >= mask_base + 2)
                        it = ambiguous.erase(it);
                    else
                        ++it;
                }
            };
            std::vector<std::pair<int, uint32_t>> scalar_writes;
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                scalar_writes.emplace_back(base, width);
                erase_overlapping(base, width);
            }, /*wave32_one_word_masks*/false);
            // This exact B32 write resolves VCC_LO to scalar data while the whole-stream proof
            // guarantees the untouched high half cannot be observed before a complete replacement.
            // It is therefore safe to clear the pair-domain ambiguity for this path; a later join
            // with a mask path will recreate the ambiguity in the ordinary MUST merge below.
            if (b.is_compute && b.vcc_b32_low_only_pcs.contains(in.pc) &&
                b32_vcc_scalar_result)
                ambiguous.erase(106);
            // An implicit VOPC destination is architectural VCC and is absent from the explicit
            // scalar-writer inventory.
            if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                !(in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)) {
                erase_overlapping(106, 2);
                // VOPC writes a complete Wave64 mask even though it has no explicit scalar
                // destination in the generic write inventory. End both physical words' old
                // scalar lifetimes: retaining them can falsely classify a later one-word VCC_LO
                // overwrite as a complete scalar pair after a dispatcher reload.
                scalar_words.erase(106);
                scalar_words.erase(107);
            }
            if (mask_write >= 0) {
                masks.insert(mask_write);
                ambiguous.erase(mask_write);
                // An inline S_MOV_B64 is represented in both domains by emit_alu: its Bool
                // view is an exact wave mask and its two scalar words are the same architectural bit
                // pattern. GTA joins `s_mov_b64 s[16:17], 0` against an SMEM load, then consumes the
                // pair as scalar data. Preserve that dual definition; other mask writes still erase
                // scalar facts because their Boolean value cannot be materialized as two SGPR words.
                const bool mov_dual_domain = in.fmt == Rdna2Format::SOP1 &&
                    in.opcode == 0x04 &&
                    (in.src[0].kind == OperandKind::InlineInt ||
                     scalar_alu_result ||
                     (b.is_compute && b.wave_size == 64 &&
                      b.native_subgroup_size == 64 && source_is_mask(in.src[0])));
                const bool cselect_scalar_branch = valid_scc_read &&
                    in.fmt == Rdna2Format::SOP2 &&
                    in.opcode == 0x0b && in.dst.value == 106 &&
                    !b.cselect_b64_low_only_pcs.contains(in.pc) &&
                    !(source_is_mask(in.src[0]) && source_is_mask(in.src[1])) &&
                    scalar_alu_result;
                const bool logical_native_ballot = in.fmt == Rdna2Format::SOP2 &&
                    in.opcode >= 0x0f && in.opcode <= 0x1d &&
                    (in.opcode & 1u) == 1 && b.is_compute && b.wave_size == 64 &&
                    b.native_subgroup_size == 64;
                const bool quadmask_native_ballot = exact_quadmask;
                const bool dual_domain_scalar_write =
                    mov_dual_domain || cselect_scalar_branch || logical_native_ballot ||
                    quadmask_native_ballot || b32_vcc_complete_scalar_pair;
                if (dual_domain_scalar_write && valid_scc_read) {
                    for (const auto& [base, width] : scalar_writes)
                        for (uint32_t word = 0; word < width; ++word)
                            scalar_words.insert(base + static_cast<int>(word));
                } else {
                    for (const auto& [base, width] : scalar_writes)
                        for (uint32_t word = 0; word < width; ++word)
                            scalar_words.erase(base + static_cast<int>(word));
                }
            } else if (valid_scc_read &&
                       ((in.fmt != Rdna2Format::SOP1 &&
                         in.fmt != Rdna2Format::SOP2 &&
                         in.fmt != Rdna2Format::SOPK) || scalar_alu_result ||
                        exact_mask_reduction)) {
                for (const auto& [base, width] : scalar_writes)
                    for (uint32_t word = 0; word < width; ++word)
                        scalar_words.insert(base + static_cast<int>(word));
            } else {
                for (const auto& [base, width] : scalar_writes)
                    for (uint32_t word = 0; word < width; ++word)
                        scalar_words.erase(base + static_cast<int>(word));
            }
            // B32 VCC logicals have a separate mask-domain lowering. If either input lacks a MUST
            // scalar word, the emitter may take that path and erase its uint result. Never publish
            // a scalar fact merely because the architectural destination is one dword: dispatcher
            // Function variables contain zero placeholders for absent domains.
            if (b32_vcc_scalar_write && !b32_vcc_scalar_result)
                scalar_words.erase(in.dst.value);

            // SCC is persisted through a dispatcher Function variable without a runtime validity
            // tag. A scalar SOPC establishes a real Boolean, while the three exact whole-wave mask
            // comparisons above establish one through their synchronized vote phase. Other scalar
            // ALU writers need fully scalar sources or kill the fact. The resulting validity is a
            // CFG MUST property, so a placeholder from any incoming edge still poisons the join.
            if (in.fmt == Rdna2Format::SOPC) {
                scalar_scc = scalar_sources;
            } else if (in.fmt == Rdna2Format::SOP2) {
                const bool preserves_scc =
                    in.opcode == kSop2OpcodeCselectB32 || in.opcode == 0x0bu ||
                    in.opcode == kSop2OpcodeBfmB32 ||
                    in.opcode == kSop2OpcodeBfmB64 || in.opcode == 0x26u ||
                    (in.opcode >= 0x32u && in.opcode <= 0x36u);
                if (in.opcode == 0x04u || in.opcode == 0x05u)
                    scalar_scc = scalar_scc && scalar_sources;
                else if (in.opcode == kSop2OpcodeBfeU64)
                    scalar_scc = scalar_sources &&
                        in.dst.value != 106 && in.dst.value != 107 &&
                        in.dst.value != 126 && in.dst.value != 127;
                else if (!preserves_scc)
                    // `mask_write < 0` is what keeps a B64 logical's cross-lane
                    // SCC=(whole mask != 0) out of the scalar domain, and it must. It also fired on
                    // the ONE mask classification that is not a wave reduction at all:
                    // `b32_vcc_complete_scalar_pair` sets mask_write=106 precisely BECAUSE both
                    // source dwords are proved scalar data, and on exactly that proof emit_alu
                    // takes its scalar path and publishes `rs.scc = (complete 32-bit result != 0)`
                    // — an ordinary one-dword compare with no cross-lane component. The mask
                    // lifetime here is the SECOND, derived half of a dual-domain write, not
                    // evidence that the result is a wave mask.
                    //
                    // Charging that write an SCC loss cost Sonic Frontiers five stage compute
                    // programs, three hops downstream of the site (#2790). Every one of them runs
                    //     s_or_b32 vcc_lo, vcc_hi, scc      <- dual-domain scalar pair, mask_write=106
                    //     ... v_cmp_* vcc, 0, vcc_lo        <- ends both words' scalar lifetimes
                    //     s_cselect_b32 vcc_hi, 1, 0        <- reads SCC: valid_scc_read was FALSE,
                    //                                          so VCC_HI lost its scalar-word fact
                    //     s_cmp_eq_u32 <imm>, sN            <- re-arms a perfectly scalar SCC
                    //     s_or_b32 vcc_lo, scc, vcc_hi      <- REJECTED: VCC_HI is not a scalar word
                    // and the reject is reported at the last line, whose own SCC is fine — measured
                    // live, `scc` is a live SSA id there and both operands have scalar data. The
                    // deliberate 64-bit-mask SCC poison in rdna2_emit_alu.cpp is NOT what stops
                    // these programs; this transfer rule is.
                    //
                    // The SOP1 arm below already draws exactly this distinction for the identical
                    // situation -- `s_not_b32 vcc_lo,vcc_lo` keeps its scalar SCC unless
                    // `wave64_vcc_b32_mask_not` says the source really was a mask, rather than
                    // being denied it for having a coexisting Bool lifetime. This makes the SOP2
                    // arm agree with it. `dual_domain_scalar_write` above already publishes this
                    // instruction's DESTINATION words on the same `b32_vcc_complete_scalar_pair`
                    // proof; SCC was the one fact it withheld.
                    //
                    // That symmetry now holds FOR COMPUTE ONLY -- `dual_domain_scalar_write` is not
                    // narrowed, this is. Do NOT "restore consistency" by dropping `b.is_compute`:
                    // the asymmetry is deliberate and the reason is the next paragraph.
                    //
                    // `b.is_compute` is deliberately narrower than the rest of this transfer, which
                    // also runs for Wave64 FRAGMENT. A fragment B32 VCC logical can instead be
                    // claimed by emit_alu's Wave32-mask block, which publishes
                    // `rs.scc = fragment_wave_any(...)` -- a vote, not `(result != 0)`. Reaching
                    // that needs a source which is simultaneously a MUST scalar word and a live
                    // B32 mask, which nobody has constructed; the measured case is compute, so the
                    // exemption is granted only where the implication was actually traced. Review
                    // of #2801.
                    scalar_scc = scalar_sources &&
                        (mask_write < 0 ||
                         (b.is_compute && b32_vcc_complete_scalar_pair));
            } else if (in.fmt == Rdna2Format::SOP1) {
                const bool preserves_scc =
                    in.opcode == kSop1OpcodeMovB32 ||
                    in.opcode == kSop1OpcodeMovB64 ||
                    in.opcode == kSop1OpcodeCmovB32 ||
                    in.opcode == kSop1OpcodeCmovB64 ||
                    in.opcode == kSop1OpcodeBrevB32 ||
                    in.opcode == kSop1OpcodeBcnt1I32B64 ||
                    in.opcode == kSop1OpcodeFf1I32B64 ||
                    in.opcode == kSop1OpcodeFlbitI32B32 ||
                    in.opcode == kSop1OpcodeFlbitI32B64 ||
                    in.opcode == kSop1OpcodeBitset0B32 ||
                    in.opcode == kSop1OpcodeBitset1B32 ||
                    in.opcode == kSop1OpcodeGetpcB64;
                const bool saveexec =
                    (in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                     in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
                    in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
                    in.opcode == kSop1OpcodeOrn1SaveexecB64;
                if (in.opcode == kSop1OpcodeNotB32 ||
                    in.opcode == kSop1OpcodeAbsI32)
                    scalar_scc = scalar_alu_result && !wave64_vcc_b32_mask_not;
                else if (saveexec)
                    scalar_scc = b.is_fragment && initial.reads_scc &&
                        source_is_mask(in.src[0]);
                else if (in.opcode == kSop1OpcodeQuadmaskB64)
                    scalar_scc = exact_quadmask;
                else if (!preserves_scc)
                    scalar_scc = false;
            } else if (in.fmt == Rdna2Format::SOPK) {
                const bool writes_scc =
                    (in.opcode >= kSopkOpcodeCmpkFirst &&
                     in.opcode <= kSopkOpcodeCmpkLast) ||
                    in.opcode == kSopkOpcodeAddkI32;
                const bool preserves_scc =
                    in.opcode == kSopkOpcodeMovkI32 ||
                    in.opcode == kSopkOpcodeCmovkI32 ||
                    in.opcode == kSopkOpcodeMulkI32 ||
                    in.opcode == kSopkOpcodeSetregB32 ||
                    (in.opcode >= kSopkOpcodeWaitcntVscnt &&
                     in.opcode <= kSopkOpcodeWaitcntLgkmcnt);
                if (writes_scc)
                    scalar_scc = implicit_scalar_source;
                else if (!preserves_scc)
                    scalar_scc = false;
            }
            if (b64_mask_scc_vote_pcs.contains(in.pc) ||
                native_b32_mask_scc_vote_pcs.contains(in.pc) ||
                writes_exact_wave_scc)
                scalar_scc = true;
            return true;
        };

        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> masks = wave64_b64_mask_in[block];
            std::set<int> ambiguous = wave64_b64_ambiguous_in[block];
            std::set<int> scalar_words = wave64_scalar_word_in[block];
            bool scalar_scc = wave64_scalar_scc_valid_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;
                if (!advance_wave64_b64_masks(
                        masks, ambiguous, scalar_words, scalar_scc, in,
                        /*record_compare*/false))
                    return false;
            }
            for (uint32_t successor : successors[block]) {
                if (!wave64_b64_reachable[successor]) {
                    wave64_b64_reachable[successor] = true;
                    wave64_b64_mask_in[successor] = masks;
                    wave64_b64_ambiguous_in[successor] = ambiguous;
                    wave64_scalar_word_in[successor] = scalar_words;
                    wave64_scalar_scc_valid_in[successor] = scalar_scc;
                    pending.push_back(successor);
                    continue;
                }
                std::set<int> joined;
                std::set<int> joined_ambiguous = wave64_b64_ambiguous_in[successor];
                std::set_intersection(
                    wave64_b64_mask_in[successor].begin(),
                    wave64_b64_mask_in[successor].end(),
                    masks.begin(), masks.end(),
                    std::inserter(joined, joined.end()));
                for (int base : wave64_b64_mask_in[successor])
                    if (!masks.contains(base)) joined_ambiguous.insert(base);
                for (int base : masks)
                    if (!wave64_b64_mask_in[successor].contains(base))
                        joined_ambiguous.insert(base);
                joined_ambiguous.insert(ambiguous.begin(), ambiguous.end());
                for (int base : joined_ambiguous) joined.erase(base);
                std::set<int> joined_scalar_words;
                std::set_intersection(
                    wave64_scalar_word_in[successor].begin(),
                    wave64_scalar_word_in[successor].end(),
                    scalar_words.begin(), scalar_words.end(),
                    std::inserter(joined_scalar_words, joined_scalar_words.end()));
                const bool joined_scalar_scc =
                    wave64_scalar_scc_valid_in[successor] && scalar_scc;
                if (joined != wave64_b64_mask_in[successor] ||
                    joined_ambiguous != wave64_b64_ambiguous_in[successor] ||
                    joined_scalar_words != wave64_scalar_word_in[successor] ||
                    joined_scalar_scc != wave64_scalar_scc_valid_in[successor]) {
                    wave64_b64_mask_in[successor] = std::move(joined);
                    wave64_b64_ambiguous_in[successor] = std::move(joined_ambiguous);
                    wave64_scalar_word_in[successor] = std::move(joined_scalar_words);
                    wave64_scalar_scc_valid_in[successor] = joined_scalar_scc;
                    pending.push_back(successor);
                }
            }
        }
        for (uint32_t block = 0; block < starts.size(); ++block) {
            if (!wave64_b64_reachable[block]) continue;
            std::set<int> masks = wave64_b64_mask_in[block];
            std::set<int> ambiguous = wave64_b64_ambiguous_in[block];
            std::set<int> scalar_words = wave64_scalar_word_in[block];
            bool scalar_scc = wave64_scalar_scc_valid_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;
                if (!advance_wave64_b64_masks(
                        masks, ambiguous, scalar_words, scalar_scc, in,
                        /*record_compare*/true))
                    return false;
            }
        }
    }

    // V_WRITELANE/V_READLANE scalar spills can carry one physical half of a Wave64 mask through a
    // loop. The Bool slot alone loses whether it was LO or HI, while a dispatcher uint placeholder
    // is not a validity tag. Track that identity as a CFG MUST fact and publish it only at exact
    // native-Wave64 readlane PCs. Joins retain equal facts; every ordinary overwrite kills them.
    std::vector<std::map<int, uint32_t>> wave64_mask_half_sreg_in(starts.size());
    std::vector<bool> wave64_mask_half_reachable(starts.size(), false);
    if (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64 && !starts.empty()) {
        struct MaskHalfState {
            std::map<int, uint32_t> sreg;
            std::map<std::pair<int, int>, uint32_t> slot;
            bool operator==(const MaskHalfState&) const = default;
        };
        auto special_half = [](const Operand& source) -> int {
            if (source.kind != OperandKind::Special) return -1;
            // EXEC is always a live mask in RegState. VCC_LO/HI may instead be scalar scratch, and
            // their physical encodings do not carry a runtime domain tag; treating those words as
            // masks here can turn a dispatcher placeholder into a ballot of false. Admit VCC only
            // after a future proof is explicitly tied to the Wave64 mask-domain MUST analysis.
            if (source.value == 126) return 0;
            if (source.value == 127) return 1;
            return -1;
        };
        auto meet = [](MaskHalfState& dst, const MaskHalfState& incoming) {
            for (auto it = dst.sreg.begin(); it != dst.sreg.end();) {
                const auto other = incoming.sreg.find(it->first);
                if (other == incoming.sreg.end() || other->second != it->second)
                    it = dst.sreg.erase(it);
                else
                    ++it;
            }
            for (auto it = dst.slot.begin(); it != dst.slot.end();) {
                const auto other = incoming.slot.find(it->first);
                if (other == incoming.slot.end() || other->second != it->second)
                    it = dst.slot.erase(it);
                else
                    ++it;
            }
        };
        auto transfer = [&](MaskHalfState& state, const Rdna2Inst& in, bool record) {
            if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360) {
                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    for (uint32_t word = 0; word < width; ++word)
                        state.sreg.erase(base + static_cast<int>(word));
                }, /*wave32_one_word_masks*/false);
                if (in.src[1].kind == OperandKind::InlineInt &&
                    in.src[1].value >= 0 && in.src[1].value <= 63) {
                    const std::pair<int, int> key{in.src[0].value, in.src[1].value};
                    const auto half = state.slot.find(key);
                    if (half != state.slot.end()) {
                        state.sreg[in.dst.value] = half->second;
                        if (record)
                            b.wave64_mask_readlane_half_for_pc[in.pc] = half->second;
                    }
                }
                return;
            }

            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t word = 0; word < width; ++word)
                    state.sreg.erase(base + static_cast<int>(word));
            }, /*wave32_one_word_masks*/false);

            if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
                // A dynamic selector may overwrite any lane and therefore kills every known slot
                // in this VGPR. A constant selector updates only its named slot.
                if (in.src[1].kind != OperandKind::InlineInt ||
                    in.src[1].value < 0 || in.src[1].value > 63) {
                    for (auto it = state.slot.begin(); it != state.slot.end();) {
                        if (it->first.first == in.dst.value) it = state.slot.erase(it);
                        else ++it;
                    }
                    return;
                }
                const std::pair<int, int> key{in.dst.value, in.src[1].value};
                int half = special_half(in.src[0]);
                if (half < 0 && in.src[0].kind == OperandKind::SGPR) {
                    const auto source = state.sreg.find(in.src[0].value);
                    if (source != state.sreg.end()) {
                        half = source->second;
                        if (record)
                            b.wave64_mask_writelane_alias_pcs.insert(in.pc);
                    }
                }
                if (half < 0) state.slot.erase(key);
                else state.slot[key] = static_cast<uint32_t>(half);
                return;
            }

            for_each_possible_vector_write(in, [&](int vgpr) {
                for (auto it = state.slot.begin(); it != state.slot.end();) {
                    if (it->first.first == vgpr) it = state.slot.erase(it);
                    else ++it;
                }
            });
        };

        std::vector<MaskHalfState> half_in(starts.size());
        std::vector<bool> half_reachable(starts.size(), false);
        half_reachable.front() = true;
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            MaskHalfState state = half_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins)
                if (in.pc >= lo && in.pc < hi && !in.is_end)
                    transfer(state, in, /*record*/false);
            for (uint32_t successor : successors[block]) {
                if (!half_reachable[successor]) {
                    half_reachable[successor] = true;
                    half_in[successor] = state;
                    pending.push_back(successor);
                    continue;
                }
                MaskHalfState joined = half_in[successor];
                meet(joined, state);
                if (!(joined == half_in[successor])) {
                    half_in[successor] = std::move(joined);
                    pending.push_back(successor);
                }
            }
        }
        for (uint32_t block = 0; block < starts.size(); ++block) {
            if (!half_reachable[block]) continue;
            wave64_mask_half_reachable[block] = true;
            wave64_mask_half_sreg_in[block] = half_in[block].sreg;
            MaskHalfState state = half_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins)
                if (in.pc >= lo && in.pc < hi && !in.is_end)
                    transfer(state, in, /*record*/true);
        }
    }

    auto mask_zero_compare_is_proven = [&](const Rdna2Inst& in) {
        // Wave32 has its own one-word MUST-domain filtering in load_state. Wave64 needs the
        // pair-width analysis above because a persisted Bool value is not itself a lifetime tag.
        return b.wave_size != 64 || proven_wave64_mask_zero_compare_pcs.contains(in.pc);
    };
    auto mask_zero_compare_value = [&](const RegState& state, int source) -> uint32_t {
        // The current GTA V sites name canonical VCC_LO as the low word of a B64 source. Its exact
        // per-lane value is stored separately from ordinary saved SGPR masks.
        if (b.wave_size == 64 && source == 106) return state.vcc;
        const auto found = state.sreg_bool.find(source);
        return found == state.sreg_bool.end() ? 0 : found->second;
    };

    // Fragment scalar PCs are lane-local in the dispatcher.  A saved-mask pair comparison must
    // nevertheless vote over every lane in the guest wave, including lanes currently executing a
    // different switch case.  Give every proven static comparison an identity and retain its exact
    // operands/polarity for the uniform common-phase votes emitted below.
    struct SavedMaskPairCompareEvent {
        int first = -1;
        int second = -1;
        uint32_t opcode = 0;
    };
    std::unordered_map<uint32_t, uint32_t> saved_mask_pair_event_for_pc;
    std::vector<SavedMaskPairCompareEvent> saved_mask_pair_events;
    if (b.is_fragment) {
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (!proven_saved_mask_pair_compare_pcs.contains(in.pc)) continue;
            const auto sources = saved_mask_pair_compare_sources(in);
            if (sources[0] < 0) continue;
            saved_mask_pair_events.push_back({sources[0], sources[1], in.opcode});
            saved_mask_pair_event_for_pc.emplace(
                in.pc, static_cast<uint32_t>(saved_mask_pair_events.size()));
        }
    }

    std::vector<std::unordered_set<int>> scalar_may_write_in(starts.size());
    std::vector<bool> scalar_reachable(starts.size(), false);
    if (!scalar_may_write_in.empty()) {
        scalar_may_write_in.front() = initial.sreg_written;
        scalar_reachable.front() = true;
    }
    bool provenance_changed = true;
    while (provenance_changed) {
        provenance_changed = false;
        for (uint32_t block = 0; block < starts.size(); ++block) {
            if (!scalar_reachable[block]) continue;
            std::unordered_set<int> out = scalar_may_write_in[block];
            out.insert(scalar_writes[block].begin(), scalar_writes[block].end());
            for (uint32_t successor : successors[block]) {
                if (!scalar_reachable[successor]) {
                    scalar_reachable[successor] = true;
                    provenance_changed = true;
                }
                const size_t before = scalar_may_write_in[successor].size();
                scalar_may_write_in[successor].insert(out.begin(), out.end());
                provenance_changed |= scalar_may_write_in[successor].size() != before;
            }
        }
    }

    // The native exact-wave path does not need to return to a synchronized common phase after a
    // plain unconditional edge.  Fuse maximal forward chains whose successor has no other
    // predecessor; the successor cannot be entered from outside the chain, so keeping its register
    // state in SSA is equivalent to a dispatcher save/reload without duplicating guest code.  Keep
    // backward edges as dispatcher iterations (they form loops), and end a chain at every wave op
    // that must execute as one uniform switch case.
    std::vector<std::vector<uint32_t>> dispatch_blocks;
    std::vector<uint32_t> dispatch_for_block(starts.size(), UINT32_MAX);
    std::vector<uint32_t> predecessor_count(starts.size(), 0);
    for (const auto& edges : successors)
        for (uint32_t successor : edges)
            if (successor < predecessor_count.size()) ++predecessor_count[successor];
    std::vector<bool> synchronized_block(starts.size(), false);
    std::vector<bool> conditional_block(starts.size(), false);
    for (uint32_t block = 0; block < starts.size(); ++block) {
        const uint32_t lo = starts[block];
        const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
        for (const auto& in : ins) {
            if (in.pc < lo || in.pc >= hi) continue;
            synchronized_block[block] = synchronized_block[block] ||
                mbcnt_event_for_pc.contains(in.pc) || append_event_for_pc.contains(in.pc) ||
                swizzle_pcs.contains(in.pc) || bpermute_event_for_pc.contains(in.pc) ||
                fragment_dpp_min_row_shr_pcs.contains(in.pc) ||
                compute_dpp_add_row_shr_pcs.contains(in.pc) ||
                compute_dpp_row_ror8_pcs.contains(in.pc) ||
                compute_dpp_add_row_mask_pcs.contains(in.pc) ||
                lds_fminmax_pcs.contains(in.pc) ||
                mask_zero_compare_candidate_source(in) >= 0 ||
                exec_saved_mask_compare_source(in) >= 0 ||
                saved_mask_pair_compare_sources(in)[0] >= 0 ||
                vopc_mask_zero_compare_source(in) >= 0 ||
                b64_mask_scc_vote_pcs.contains(in.pc);
            conditional_block[block] = conditional_block[block] ||
                (!linearized_branch(in) && in.fmt == Rdna2Format::SOPP &&
                 in.opcode >= 0x04 && in.opcode <= 0x09 && in.opcode != 0x03);
        }
    }
    for (uint32_t first = 0; first < starts.size(); ++first) {
        if (dispatch_for_block[first] != UINT32_MAX) continue;
        const uint32_t dispatch = static_cast<uint32_t>(dispatch_blocks.size());
        dispatch_blocks.push_back({});
        uint32_t block = first;
        while (true) {
            dispatch_for_block[block] = dispatch;
            dispatch_blocks.back().push_back(block);
            if (!direct_dispatch || synchronized_block[block] || conditional_block[block] ||
                successors[block].size() != 1) break;
            const uint32_t successor = successors[block].front();
            if (successor <= block || predecessor_count[successor] != 1 ||
                dispatch_for_block[successor] != UINT32_MAX) break;
            block = successor;
        }
    }

    std::vector<std::set<int>> dispatch_vector_reads(dispatch_blocks.size());
    std::vector<std::set<int>> dispatch_vector_writes(dispatch_blocks.size());
    std::vector<std::unordered_set<int>> dispatch_scalar_writes(dispatch_blocks.size());
    for (uint32_t dispatch = 0; dispatch < dispatch_blocks.size(); ++dispatch) {
        for (uint32_t block : dispatch_blocks[dispatch]) {
            dispatch_vector_reads[dispatch].insert(
                vector_reads[block].begin(), vector_reads[block].end());
            dispatch_vector_writes[dispatch].insert(
                vector_writes[block].begin(), vector_writes[block].end());
            dispatch_scalar_writes[dispatch].insert(
                scalar_writes[block].begin(), scalar_writes[block].end());
        }
    }

    // Saved mask pairs and scalar-spill lane slots have their own value domains.
    std::set<int> mask_keys = static_mask_keys;
    std::set<int> mask_half_alias_keys;
    for (uint32_t block = 0; block < starts.size(); ++block)
        if (wave64_mask_half_reachable[block])
            for (const auto& alias : wave64_mask_half_sreg_in[block])
                mask_half_alias_keys.insert(alias.first);
    std::set<std::pair<int, int>> lane_slots, mask_lane_slots;
    // A lane spill can precede another static definition of its mask in a back-edge block, hence the
    // complete discovery pass above occurs before classifying any spill slots here.
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361 &&
            in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 && in.src[1].value <= 63) {
            const std::pair<int, int> slot{in.dst.value, in.src[1].value};
            // A fragment Wave64 spill of a saved B64 mask consumes one physical ballot dword.
            // Persist it in the scalar-data lane domain; the Bool-slot representation cannot
            // distinguish LO from HI and would reload false at a later dispatcher case. Direct
            // architectural EXEC/VCC sources and other stages retain their mask-slot treatment.
            const bool fragment_physical_mask_word =
                b.is_fragment && b.wave_size == 64 &&
                in.src[0].kind == OperandKind::SGPR &&
                (static_mask_keys.count(in.src[0].value) ||
                 (in.src[0].value > 0 &&
                  static_mask_keys.count(in.src[0].value - 1)));
            const bool is_mask = !fragment_physical_mask_word &&
                (in.src[0].value == 106 || in.src[0].value == 107 ||
                 in.src[0].value == 126 || in.src[0].value == 127 ||
                 (in.src[0].kind == OperandKind::SGPR &&
                  (static_mask_keys.count(in.src[0].value) ||
                   b.wave64_mask_writelane_alias_pcs.contains(in.pc))));
            (is_mask ? mask_lane_slots : lane_slots).insert(slot);
        }
    }
    for (const auto& vg : initial.vgpr_lane_slots)
        for (const auto& slot : vg.second) lane_slots.emplace(vg.first, slot.first);
    for (const auto& vg : initial.vgpr_lane_mask_slots)
        for (const auto& slot : vg.second) mask_lane_slots.emplace(vg.first, slot.first);
    // A physical spill lane may be recycled between scalar-data and wave-mask lifetimes. Persist
    // both domains across dispatcher blocks; v_readlane retains both destination views when both
    // are present, and the statically typed consumer selects the representation it needs.

    // Map presence is the scalar-spill validity bit in RegState. Function variables cannot encode
    // that compile-time type state: saving an erased slot as zero and reconstructing it in the next
    // dispatcher case would turn an invalid lifetime into a valid scalar zero. Conservatively reject
    // any CFG path on which an ordinary VGPR write can invalidate a spill array before v_readlane;
    // a later v_writelane starts a fresh lifetime and clears the tombstone, matching emit_alu.
    std::set<int> spill_vgprs;
    for (const auto& slot : lane_slots) spill_vgprs.insert(slot.first);
    for (const auto& slot : mask_lane_slots) spill_vgprs.insert(slot.first);
    std::unordered_set<int> terminal_invalidated_vgpr_lane_slots =
        initial.invalidated_vgpr_lane_slots;
    if (!spill_vgprs.empty()) {
        std::vector<std::set<int>> invalidated_in(starts.size());
        std::vector<bool> invalidated_reachable(starts.size(), false);
        invalidated_in.front().insert(initial.invalidated_vgpr_lane_slots.begin(),
                                      initial.invalidated_vgpr_lane_slots.end());
        invalidated_reachable.front() = true;
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            std::set<int> invalidated = invalidated_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi || in.is_end) continue;
                if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
                    invalidated.erase(in.dst.value);
                    continue;
                }
                if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 &&
                    invalidated.contains(in.src[0].value))
                    return reject_cfg(in.pc, "invalidated-vgpr-lane-slot");
                for_each_possible_vector_write(in, [&](int reg) {
                    if (spill_vgprs.contains(reg)) invalidated.insert(reg);
                });
                const int tfe_status = rdna2_tfe_status_vgpr(in);
                if (spill_vgprs.contains(tfe_status)) invalidated.insert(tfe_status);
            }
            for (uint32_t successor : successors[block]) {
                if (!invalidated_reachable[successor]) {
                    invalidated_reachable[successor] = true;
                    invalidated_in[successor] = invalidated;
                    pending.push_back(successor);
                    continue;
                }
                const size_t before = invalidated_in[successor].size();
                invalidated_in[successor].insert(invalidated.begin(), invalidated.end());
                if (invalidated_in[successor].size() != before) pending.push_back(successor);
            }
        }
        if (const auto terminal = block_for_pc.find(end_pc);
            terminal != block_for_pc.end() && invalidated_reachable[terminal->second])
            terminal_invalidated_vgpr_lane_slots = {
                invalidated_in[terminal->second].begin(),
                invalidated_in[terminal->second].end()};
    }

    uint32_t ptr_u32 = 0, ptr_bool = 0;
    std::map<int, uint32_t> vv, sv, mv, mhv;
    std::map<std::pair<int, int>, uint32_t> lv, lmv;
    for (int r : vregs) vv[r] = b.function_var(b.t_u32, ptr_u32);
    for (int r : sregs) sv[r] = b.function_var(b.t_u32, ptr_u32);
    for (int r : mask_keys) mv[r] = b.function_var(b.t_bool, ptr_bool);
    for (int r : mask_half_alias_keys) mhv[r] = b.function_var(b.t_bool, ptr_bool);
    for (const auto& slot : lane_slots) lv[slot] = b.function_var(b.t_u32, ptr_u32);
    for (const auto& slot : mask_lane_slots) lmv[slot] = b.function_var(b.t_bool, ptr_bool);
    const uint32_t scc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vcc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t exec_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t pc_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t active_var = b.function_var(b.t_bool, ptr_bool);
    // PROSPER_CFG_TRIP_BOUND=N — diagnostic only. Force the dispatcher out after N iterations.
    //
    // A dispatcher loop that never terminates hangs the GPU into a driver reset, which costs the
    // whole process its compute backend and every later indirect draw. That failure is
    // indistinguishable, from outside, from a shader that is merely slow or from a defect anywhere
    // else in the submit. Bounding the loop turns "is this a non-terminating dispatcher?" into a
    // one-run yes/no: if the device survives with a bound and dies without one, it is a loop.
    //
    // Unset, nothing is emitted and the module is byte-identical. It is NOT a fix — truncating a
    // guest program's control flow produces wrong results by construction — which is why it is
    // opt-in and says so.
    const uint32_t cfg_phase = b.cfg_phase_ordinal++;
    // ONE end-exclusive boundary for this phase, derived once and used by every report below.
    //
    // The two consumers disagreed before: the announcement used `ins.back().pc` (right for a phase
    // closed by the emitter's synthetic terminator, wrong for a real tail, where it drops the final
    // instruction) while the dispatch map used `pc + len_dwords` (right for a real tail, wrong for a
    // synthetic one, where it re-includes a boundary marker that is not guest code). The fixture made
    // them contradict each other in adjacent lines: `phase 0 guest pc 0..<14` above `6:pc14..<15`.
    const uint32_t cfg_phase_end =
        ins.empty() ? 0u
                    : (ins.back().synthetic_terminator ? ins.back().pc
                                                       : ins.back().pc + ins.back().len_dwords);
    const uint32_t cfg_trip_bound =
        emitted_loop_trip_bound(b.diagnostic.program_address, cfg_phase,
                                ins.empty() ? 0u : ins.front().pc, cfg_phase_end,
                                code, dwords);
    const uint32_t trip_var = cfg_trip_bound ? b.function_var(b.t_u32, ptr_u32) : 0u;
    if (trip_var) b.store_function(trip_var, b.uconst(0));
    // Publish the ordinal -> guest pc map for this phase, once, when a bound arms.
    //
    // Every number the witness reports is a dispatch ordinal, and an ordinal means nothing on its
    // own. Leaving the reader to reconstruct the mapping is not a documentation gap, it is a defect
    // in the instrument: an ordinal that happens to fall inside the block COUNT reads as a plausible
    // block index, and mapping it by hand onto a guest pc range is exactly how a wrong conclusion got
    // published here (instrument trap 172). Emitting the map costs one line per ordinal, once.
    if (compute_trip_bound_settings().bound) {
        std::string map;
        char entry[64];
        for (uint32_t dispatch = 0; dispatch < dispatch_blocks.size(); ++dispatch) {
            if (dispatch_blocks[dispatch].empty()) continue;
            const uint32_t entry_block = dispatch_blocks[dispatch].front();
            const uint32_t last_block = dispatch_blocks[dispatch].back();
            const uint32_t lo = starts[entry_block];
            // The final ordinal has no following block start to bound it. Use the phase's own end
            // -- one dword past its last decoded instruction -- rather than UINT32_MAX, which is not
            // a pc and made the last map entry unreadable. `len_dwords` is the decoded length, so
            // this is exact for a variable-length ISA rather than assuming one dword.
            const uint32_t hi = last_block + 1 < starts.size() ? starts[last_block + 1]
                                                              : cfg_phase_end;
            const int written = snprintf(entry, sizeof(entry), "%s%u:pc%u..<%u", dispatch ? " " : "",
                                         dispatch, lo, hi);
            if (written > 0) map.append(entry, static_cast<size_t>(
                std::min<size_t>(static_cast<size_t>(written), sizeof(entry) - 1)));
        }
        fprintf(stderr, "[cfg-trip-bound] program 0x%llx phase %u dispatch map: %s\n",
                static_cast<unsigned long long>(b.diagnostic.program_address), cfg_phase,
                map.c_str());
    }
    // The span of DISPATCH ORDINALS the state machine actually visited, tracked only while a bound
    // is armed.
    //
    // `pc_var` holds a dispatcher switch-case ordinal (`dispatch_for_block`), NOT a guest pc — read
    // the store sites, not the variable's name. One ordinal at the instant the cap ran out is a
    // single sample and cannot separate "spinning here" from "passing through here"; the extremes
    // can, because a state machine confined to a few ordinals is cycling among them.
    //
    // Ordinals are only meaningful against the map this phase announces below, which is why that map
    // is printed rather than left to be reconstructed by hand. Hand-mapping one of these numbers onto
    // a guest pc range is precisely what produced a wrong published conclusion — instrument trap 172.
    const uint32_t dispatch_min_var = cfg_trip_bound ? b.function_var(b.t_u32, ptr_u32) : 0u;
    const uint32_t dispatch_max_var = cfg_trip_bound ? b.function_var(b.t_u32, ptr_u32) : 0u;
    if (dispatch_min_var) b.store_function(dispatch_min_var, b.uconst(0xffffffffu));
    if (dispatch_max_var) b.store_function(dispatch_max_var, b.uconst(0));
    const uint32_t vote_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_value_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_invert_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_to_scc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_to_vcc_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t vote_taken_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t vote_next_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_mask_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_low_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_write_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t mbcnt_event_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_acc_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_dst_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t mbcnt_sum_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_active_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_event_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_consume_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_gds_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t append_idx_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_dst_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t append_count_var = b.function_var(b.t_u32, ptr_u32);
    const bool has_synchronized_lds_store_event = !synchronized_lds_store_pcs.empty();
    const uint32_t synchronized_lds_store_pending_var = has_synchronized_lds_store_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t synchronized_lds_store_active_var = has_synchronized_lds_store_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t synchronized_lds_store_count_var = has_synchronized_lds_store_event
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    std::array<uint32_t, 4> synchronized_lds_store_idx_vars{};
    std::array<uint32_t, 4> synchronized_lds_store_value_vars{};
    if (has_synchronized_lds_store_event) {
        for (uint32_t& var : synchronized_lds_store_idx_vars)
            var = b.function_var(b.t_u32, ptr_u32);
        for (uint32_t& var : synchronized_lds_store_value_vars)
            var = b.function_var(b.t_u32, ptr_u32);
    }
    const bool has_lds_fminmax_event = !lds_fminmax_pcs.empty();
    const uint32_t lds_fminmax_pending_var = has_lds_fminmax_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t lds_fminmax_active_var = has_lds_fminmax_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t lds_fminmax_min_var = has_lds_fminmax_event
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t lds_fminmax_idx_var = has_lds_fminmax_event
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t lds_fminmax_value_var = has_lds_fminmax_event
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t swizzle_pending_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t swizzle_active_var = b.function_var(b.t_bool, ptr_bool);
    const uint32_t swizzle_source_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t swizzle_source_lane_var = b.function_var(b.t_u32, ptr_u32);
    const uint32_t swizzle_dst_var = b.function_var(b.t_u32, ptr_u32);
    const bool has_bpermute = !bpermute_event_for_pc.empty();
    const uint32_t bpermute_pending_var = has_bpermute
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t bpermute_active_var = has_bpermute
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t bpermute_address_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t bpermute_source_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t bpermute_offset_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t bpermute_event_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t bpermute_dst_var = has_bpermute
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const bool has_saved_mask_pair_events = !saved_mask_pair_events.empty();
    const uint32_t saved_mask_pair_pending_var = has_saved_mask_pair_events
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t saved_mask_pair_event_var = has_saved_mask_pair_events
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const bool has_dpp_min_row_shr = !fragment_dpp_min_row_shr_pcs.empty();
    const uint32_t dpp_min_pending_var = has_dpp_min_row_shr
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_min_active_var = has_dpp_min_row_shr
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_min_source_var = has_dpp_min_row_shr
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_min_amount_var = has_dpp_min_row_shr
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_min_dst_var = has_dpp_min_row_shr
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_min_event_var = has_dpp_min_row_shr
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_add_pending_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_add_active_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_add_source_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_add_amount_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_add_dst_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_add_event_var = has_portable_compute_dpp_add
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_pending_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_ror8_active_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t dpp_ror8_src0_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_src1_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_op_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_dst_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t dpp_ror8_event_var = has_portable_compute_dpp_ror8
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const bool has_portable_mask_ffbh = !portable_mask_ffbh_event_for_pc.empty();
    const uint32_t mask_ffbh_pending_var = has_portable_mask_ffbh
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t mask_ffbh_mask_var = has_portable_mask_ffbh
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t mask_ffbh_write_var = has_portable_mask_ffbh
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t mask_ffbh_event_var = has_portable_mask_ffbh
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t mask_ffbh_half_var = has_portable_mask_ffbh
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t mask_ffbh_dst_var = has_portable_mask_ffbh
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const bool has_portable_readlane = !portable_readlane_event_for_pc.empty();
    const uint32_t readlane_pending_var = has_portable_readlane
        ? b.function_var(b.t_bool, ptr_bool) : 0;
    const uint32_t readlane_source_var = has_portable_readlane
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t readlane_selector_var = has_portable_readlane
        ? b.function_var(b.t_u32, ptr_u32) : 0;
    const uint32_t readlane_dst_var = has_portable_readlane
        ? b.function_var(b.t_u32, ptr_u32) : 0;

    const uint32_t zero = b.uconst(0), no = b.bfalse(), yes = b.btrue();
    for (const auto& kv : vv) {
        auto it = initial.vreg.find(kv.first);
        b.store_function(kv.second, it == initial.vreg.end() ? zero : it->second);
    }
    for (const auto& kv : sv) {
        auto it = initial.sreg.find(kv.first);
        b.store_function(kv.second, it == initial.sreg.end() ? zero : it->second);
    }
    for (const auto& kv : mv) {
        auto it = initial.sreg_bool.find(kv.first);
        b.store_function(kv.second, it == initial.sreg_bool.end() ? no : it->second);
    }
    for (const auto& kv : mhv) b.store_function(kv.second, no);
    for (const auto& kv : lv) {
        uint32_t value = zero;
        auto vg = initial.vgpr_lane_slots.find(kv.first.first);
        if (vg != initial.vgpr_lane_slots.end()) {
            auto slot = vg->second.find(kv.first.second);
            if (slot != vg->second.end()) value = slot->second;
        }
        b.store_function(kv.second, value);
    }
    for (const auto& kv : lmv) {
        uint32_t value = no;
        auto vg = initial.vgpr_lane_mask_slots.find(kv.first.first);
        if (vg != initial.vgpr_lane_mask_slots.end()) {
            auto slot = vg->second.find(kv.first.second);
            if (slot != vg->second.end()) value = slot->second;
        }
        b.store_function(kv.second, value);
    }
    b.store_function(scc_var, initial.scc ? initial.scc : no);
    // The dispatcher has no runtime type tag for a physical VCC word recycled as scalar data.
    // Wave32's compile-time mask-domain analysis above proves whether each implicit VCC consumer
    // sees a real mask. It is therefore safe to persist false while that lifetime is absent/dead;
    // load_state keeps the placeholder out of RegState on those entries. Other modes retain the
    // old fail-visible contract because they have no equivalent proof.
    // #3231: the entry value can also be provably DEAD. Astro Bot's world-map lighting consumer
    // `0x500571000` recycles the physical VCC words as scalar data before its final barrier phase,
    // then re-defines the pair with `v_cmp_le_u32 vcc, s12, v24` at the top of that phase's entry
    // block — pc 322, ahead of the block's first VCC read at 323 and its terminator at 325. Nothing
    // can observe what the dispatcher stored, so the old blanket refusal cost the title a
    // full-screen lighting pass for no correctness gain.
    //
    // Restricted to Wave64 COMPUTE, which is where the evidence is and where a second, independent
    // mechanism already stands behind this one: `load_state` only loads `vcc_var` into `state.vcc`
    // for a block whose `wave64_b64_mask_in` MUST set contains 106, and that set seeds 106 only
    // `if (initial.vcc)`. An absent entry VCC therefore reaches a consumer as *no value* (a
    // fail-visible reject) rather than as a fabricated `false`. Fragment Wave64 has the same
    // backstop but no live case, and every other stage has neither, so all of them keep the old
    // contract. A partial-workgroup extent is excluded as well: its padded invocations never
    // dispatch block 0, so they would keep the placeholder instead of the caller's value.
    const bool entry_vcc_dead = !initial.vcc && !proven_wave32_masks &&
        b.is_compute && b.wave_size == 64 && !initial_active &&
        entry_block_defines_vcc_before_any_read(
            ins, starts.front(), starts.size() > 1 ? starts[1] : UINT32_MAX);
    if (!initial.vcc && !proven_wave32_masks && !entry_vcc_dead)
        return reject_cfg(ins.front().pc, "missing-entry-vcc");
    b.store_function(vcc_var, initial.vcc ? initial.vcc : no);
    b.store_function(exec_var, initial.exec);
    b.store_function(pc_var, b.uconst(0));
    b.store_function(active_var, initial_active ? initial_active : yes);
    if (has_synchronized_lds_store_event) {
        b.store_function(synchronized_lds_store_count_var, zero);
        for (uint32_t var : synchronized_lds_store_idx_vars)
            b.store_function(var, zero);
        for (uint32_t var : synchronized_lds_store_value_vars)
            b.store_function(var, zero);
    }
    if (has_lds_fminmax_event) {
        b.store_function(lds_fminmax_min_var, no);
        b.store_function(lds_fminmax_idx_var, zero);
        b.store_function(lds_fminmax_value_var, zero);
    }
    b.store_function(swizzle_source_var, zero);
    b.store_function(swizzle_source_lane_var, zero);
    b.store_function(swizzle_dst_var, zero);
    if (has_bpermute) {
        b.store_function(bpermute_address_var, zero);
        b.store_function(bpermute_source_var, zero);
        b.store_function(bpermute_offset_var, zero);
        b.store_function(bpermute_event_var, zero);
        b.store_function(bpermute_dst_var, zero);
    }
    if (has_saved_mask_pair_events)
        b.store_function(saved_mask_pair_event_var, zero);
    if (has_dpp_min_row_shr) {
        b.store_function(dpp_min_source_var, zero);
        b.store_function(dpp_min_amount_var, zero);
        b.store_function(dpp_min_dst_var, zero);
        b.store_function(dpp_min_event_var, zero);
    }
    if (has_portable_compute_dpp_add) {
        b.store_function(dpp_add_source_var, zero);
        b.store_function(dpp_add_amount_var, zero);
        b.store_function(dpp_add_dst_var, zero);
        b.store_function(dpp_add_event_var, zero);
    }
    if (has_portable_compute_dpp_ror8) {
        b.store_function(dpp_ror8_src0_var, zero);
        b.store_function(dpp_ror8_src1_var, zero);
        b.store_function(dpp_ror8_op_var, zero);
        b.store_function(dpp_ror8_dst_var, zero);
        b.store_function(dpp_ror8_event_var, zero);
    }
    if (has_portable_mask_ffbh) {
        b.store_function(mask_ffbh_event_var, zero);
        b.store_function(mask_ffbh_half_var, zero);
        b.store_function(mask_ffbh_dst_var, zero);
    }
    if (has_portable_readlane) {
        b.store_function(readlane_source_var, zero);
        b.store_function(readlane_selector_var, zero);
        b.store_function(readlane_dst_var, zero);
    }

    auto load_state = [&](uint32_t dispatch = UINT32_MAX) {
        RegState state;
        state.scalar_presence_has_no_placeholders = false;
        state.max_vgpr = initial.max_vgpr;
        state.sreg_input = initial.sreg_input;
        state.smem_x16_descriptor_loads = initial.smem_x16_descriptor_loads;
        state.smem_x16_descriptor_analysis_done = initial.smem_x16_descriptor_analysis_done;
        state.smem_x2_descriptor_fragment_loads =
            initial.smem_x2_descriptor_fragment_loads;
        state.smem_x2_descriptor_fragment_analysis_done =
            initial.smem_x2_descriptor_fragment_analysis_done;
        state.reads_scc = initial.reads_scc;
        state.invalidated_vgpr_lane_slots = initial.invalidated_vgpr_lane_slots;
        for (const auto& kv : vv) {
            if (dispatch != UINT32_MAX &&
                !dispatch_vector_reads[dispatch].contains(kv.first)) continue;
            state.vreg[kv.first] = b.load_function(b.t_u32, kv.second);
        }
        for (const auto& kv : sv) state.sreg[kv.first] = b.load_function(b.t_u32, kv.second);
        const std::set<int>* entry_b32 = nullptr;
        const std::set<int>* entry_b64 = nullptr;
        const std::set<int>* entry_wave64_b64 = nullptr;
        const std::set<int>* entry_wave64_scalar_words = nullptr;
        const std::map<int, uint32_t>* entry_wave64_mask_halves = nullptr;
        uint32_t entry_block = UINT32_MAX;
        if (dispatch != UINT32_MAX)
            entry_block = dispatch_blocks[dispatch].front();
        else if (const auto terminal = block_for_pc.find(end_pc);
                 terminal != block_for_pc.end())
            entry_block = terminal->second;
        // #3133's token, narrowed to the saves that can actually REACH this block entry. A block the
        // walk never reached -- and the terminal call when `end_pc` maps to no block -- falls back
        // to the whole-stream set, so the unknown case keeps #3133's conservative behaviour instead
        // of silently dropping the token.
        const std::set<int>& entry_m0_here =
            (entry_block != UINT32_MAX && entry_m0_reachable[entry_block])
                ? entry_m0_in[entry_block]
                : entry_m0_may_hold;
        for (int reg : entry_m0_here) {
            state.sreg.erase(reg);
            state.sreg_entry_m0.insert(reg);
        }
        if (b.allow_b32_masks) {
            if (entry_block != UINT32_MAX && b32_mask_reachable[entry_block]) {
                entry_b32 = &b32_mask_in[entry_block];
                entry_b64 = &b64_mask_in[entry_block];
            }
        }
        const bool filters_wave64_b64 = (b.is_compute || b.is_fragment) &&
            b.wave_size == 64;
        if (filters_wave64_b64 && entry_block != UINT32_MAX &&
            wave64_b64_reachable[entry_block]) {
            entry_wave64_b64 = &wave64_b64_mask_in[entry_block];
            entry_wave64_scalar_words = &wave64_scalar_word_in[entry_block];
        }
        if (entry_block != UINT32_MAX &&
            wave64_mask_half_reachable[entry_block])
            entry_wave64_mask_halves = &wave64_mask_half_sreg_in[entry_block];
        for (const auto& kv : mv) {
            if (b.allow_b32_masks &&
                (!entry_b64 || !entry_b64->contains(kv.first)) &&
                (!entry_b32 || !entry_b32->contains(kv.first)))
                continue;
            if (filters_wave64_b64 &&
                (!entry_wave64_b64 || !entry_wave64_b64->contains(kv.first)))
                continue;
            state.sreg_bool[kv.first] = b.load_function(b.t_bool, kv.second);
            state.sreg_bool_narrowed[kv.first] = true;
        }
        if (entry_wave64_mask_halves)
            for (const auto& kv : mhv)
                if (const auto half = entry_wave64_mask_halves->find(kv.first);
                    half != entry_wave64_mask_halves->end()) {
                    state.sreg_wave64_mask_half[kv.first] =
                        b.load_function(b.t_bool, kv.second);
                    state.sreg_wave64_mask_half_index[kv.first] = half->second;
                }
        for (const auto& low : state.sreg_wave64_mask_half) {
            const int base = low.first;
            if (base & 1) continue;
            const auto high = state.sreg_wave64_mask_half.find(base + 1);
            const auto low_index = state.sreg_wave64_mask_half_index.find(base);
            const auto high_index = state.sreg_wave64_mask_half_index.find(base + 1);
            if (high == state.sreg_wave64_mask_half.end() ||
                low_index == state.sreg_wave64_mask_half_index.end() ||
                high_index == state.sreg_wave64_mask_half_index.end() ||
                low_index->second != 0 || high_index->second != 1)
                continue;
            const uint32_t lane = b.ibin(
                Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
            state.sreg_bool[base] = b.bsel(
                b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)),
                high->second, low.second);
            state.sreg_bool_narrowed[base] = true;
        }
        for (const auto& kv : lv)
            state.vgpr_lane_slots[kv.first.first][kv.first.second] =
                b.load_function(b.t_u32, kv.second);
        for (const auto& kv : lmv)
            state.vgpr_lane_mask_slots[kv.first.first][kv.first.second] =
                b.load_function(b.t_bool, kv.second);
        const bool filters_wave64_scalar_scc =
            (b.is_compute || b.is_fragment) && b.wave_size == 64;
        const bool live_scalar_scc = !filters_wave64_scalar_scc ||
            (entry_block != UINT32_MAX &&
             entry_block < wave64_scalar_scc_valid_in.size() &&
             wave64_scalar_scc_valid_in[entry_block]);
        state.scc = live_scalar_scc
            ? b.load_function(b.t_bool, scc_var) : 0;
        const bool live_vcc = filters_wave64_b64
            ? entry_wave64_b64 && entry_wave64_b64->contains(106)
            : (!entry_b32 || entry_b32->contains(106));
        state.vcc = live_vcc ? b.load_function(b.t_bool, vcc_var) : 0;
        state.exec = b.load_function(b.t_bool, exec_var);
        // `sv` has a Function variable for every statically observed scalar lifetime, including
        // zero placeholders stored while the same physical pair carries only a Bool-domain mask.
        // Do not let those placeholders shadow the live mask after a dispatcher reload. A genuinely
        // dual-domain value (an inline/native-ballot S_MOV or scalar CSELECT-to-VCC) retains exact
        // scalar-word MUST facts and therefore keeps its scalar loads.
        if (entry_wave64_b64) {
            for (int base : *entry_wave64_b64) {
                for (int word = base; word < base + 2; ++word) {
                    if (entry_wave64_scalar_words &&
                        entry_wave64_scalar_words->contains(word))
                        continue;
                    state.sreg.erase(word);
                    state.sreg_input.erase(word);
                }
            }
        }
        // Every referenced scalar has a Function variable, initialized to zero even when the
        // architectural word has no value on this path. The Wave64 MUST analysis is its separate
        // validity tag: remove all absent ordinary/special scalar words before emitting a case so
        // an uninitialized M0/SGPR cannot become a valid zero merely by crossing the dispatcher.
        if (entry_wave64_scalar_words) {
            for (auto it = state.sreg.begin(); it != state.sreg.end();) {
                if (it->first <= 124 && !entry_wave64_scalar_words->contains(it->first))
                    it = state.sreg.erase(it);
                else
                    ++it;
            }
            for (auto it = state.sreg_input.begin(); it != state.sreg_input.end();) {
                if (it->first <= 124 && !entry_wave64_scalar_words->contains(it->first))
                    it = state.sreg_input.erase(it);
                else
                    ++it;
            }
        }
        if (entry_b32) {
            for (int reg : *entry_b32) {
                state.sreg.erase(reg);
                state.sreg_input.erase(reg);
                state.sreg_bool_b32.insert(reg);
                if (reg == 106) {
                    state.sreg_bool[106] = state.vcc;
                    state.sreg_bool_narrowed[106] = true;
                }
            }
        }
        state.exec_narrowed = true; // state-machine joins may carry a narrowed EXEC edge
        state.mubuf_pcrel_tables = initial.mubuf_pcrel_tables;
        state.smem_pcrel_tables = initial.smem_pcrel_tables;
        state.mtbuf_pcrel_tables = initial.mtbuf_pcrel_tables;
        return state;
    };
    auto save_state = [&](const RegState& state, uint32_t dispatch) {
        // A dispatcher case starts from the persistent register file, executes exactly one guest
        // basic block, then returns to the common loop.  Only registers WRITTEN by that block need
        // to be copied back.  Saving every tracked VGPR/SGPR at every edge generated thousands of
        // redundant Function-memory stores for large kernels (Astro Bot's title compute shader has
        // around one hundred tracked registers and dozens of cases), creating severe register
        // pressure and scratch traffic in the host driver.  The conservative static writer sets
        // include multi-register loads and secondary scalar destinations; unchanged variables keep
        // the value loaded at the preceding dispatcher edge.
        for (const auto& kv : vv) {
            if (!dispatch_vector_writes[dispatch].contains(kv.first)) continue;
            auto it = state.vreg.find(kv.first);
            b.store_function(kv.second, it == state.vreg.end() ? zero : it->second);
        }
        for (const auto& kv : sv) {
            if (!dispatch_scalar_writes[dispatch].contains(kv.first)) continue;
            auto it = state.sreg.find(kv.first);
            b.store_function(kv.second, it == state.sreg.end() ? zero : it->second);
        }
        for (const auto& kv : mv) {
            if (!dispatch_scalar_writes[dispatch].contains(kv.first)) continue;
            auto it = state.sreg_bool.find(kv.first);
            b.store_function(kv.second, it == state.sreg_bool.end() ? no : it->second);
        }
        for (const auto& kv : mhv) {
            if (!dispatch_scalar_writes[dispatch].contains(kv.first)) continue;
            auto it = state.sreg_wave64_mask_half.find(kv.first);
            b.store_function(kv.second,
                             it == state.sreg_wave64_mask_half.end() ? no : it->second);
        }
        for (const auto& kv : lv) {
            uint32_t value = zero;
            auto vg = state.vgpr_lane_slots.find(kv.first.first);
            if (vg != state.vgpr_lane_slots.end()) {
                auto slot = vg->second.find(kv.first.second);
                if (slot != vg->second.end()) value = slot->second;
            }
            b.store_function(kv.second, value);
        }
        for (const auto& kv : lmv) {
            uint32_t value = no;
            auto vg = state.vgpr_lane_mask_slots.find(kv.first.first);
            if (vg != state.vgpr_lane_mask_slots.end()) {
                auto slot = vg->second.find(kv.first.second);
                if (slot != vg->second.end()) value = slot->second;
            }
            b.store_function(kv.second, value);
        }
        // A poisoned (0) SCC degrades to bfalse at the block boundary: the same-block consumers
        // reject on the sentinel; cross-block staleness matches the pre-poison model.
        b.store_function(scc_var, state.scc ? state.scc : b.bfalse());
        b.store_function(vcc_var, state.vcc ? state.vcc : no);
        b.store_function(exec_var, state.exec);
    };

    const uint32_t loop_header = b.id(), switch_header = b.id(), switch_merge = b.id();
    const uint32_t loop_continue = b.id(), loop_merge = b.id(), fallback = b.id();
    std::vector<uint32_t> labels(dispatch_blocks.size());
    std::vector<std::pair<uint32_t, uint32_t>> switch_cases;
    for (uint32_t i = 0; i < dispatch_blocks.size(); ++i) {
        labels[i] = b.id();
        switch_cases.emplace_back(i, labels[i]);
    }
    b.emit_branch(loop_header);
    b.emit_label(loop_header);
    // Every live hardware wave executes one guest basic block per dispatcher iteration. Inactive
    // invocations remain in the loop as synchronization participants until all waves finish.
    // With an exact native subgroup, one host subgroup is one guest wave and its scalar PC is
    // uniform.  Cross-lane operations can therefore execute directly in their switch case.  The
    // old publish/merge phase remains necessary only for the portable workgroup-scratch fallback;
    // resetting all of its mailboxes on every native dispatcher iteration was a surprisingly large
    // SALU/function-memory tax for branch-heavy kernels.
    if (!direct_dispatch) {
        b.store_function(vote_pending_var, no);
        b.store_function(vote_value_var, no);
        b.store_function(vote_invert_var, no);
        b.store_function(vote_to_scc_var, no);
        b.store_function(vote_to_vcc_var, no);
        b.store_function(vote_taken_var, zero);
        b.store_function(vote_next_var, zero);
        b.store_function(mbcnt_pending_var, no);
        b.store_function(mbcnt_mask_var, no);
        b.store_function(mbcnt_low_var, no);
        b.store_function(mbcnt_write_var, no);
        b.store_function(mbcnt_event_var, zero);
        b.store_function(mbcnt_acc_var, zero);
        b.store_function(mbcnt_dst_var, zero);
        b.store_function(append_pending_var, no);
        b.store_function(append_active_var, no);
        b.store_function(append_event_var, zero);
        b.store_function(append_consume_var, no);
        b.store_function(append_gds_var, no);
        b.store_function(append_idx_var, zero);
        b.store_function(append_dst_var, zero);
    }
    if (has_lds_fminmax_event) {
        b.store_function(lds_fminmax_pending_var, no);
        b.store_function(lds_fminmax_active_var, no);
    }
    if (has_synchronized_lds_store_event) {
        b.store_function(synchronized_lds_store_pending_var, no);
        b.store_function(synchronized_lds_store_active_var, no);
        b.store_function(synchronized_lds_store_count_var, zero);
    }
    b.store_function(swizzle_pending_var, no);
    b.store_function(swizzle_active_var, no);
    if (has_bpermute) {
        b.store_function(bpermute_pending_var, no);
        b.store_function(bpermute_active_var, no);
        b.store_function(bpermute_event_var, zero);
    }
    if (has_saved_mask_pair_events) {
        b.store_function(saved_mask_pair_pending_var, no);
        b.store_function(saved_mask_pair_event_var, zero);
    }
    if (has_dpp_min_row_shr) {
        b.store_function(dpp_min_pending_var, no);
        b.store_function(dpp_min_active_var, no);
        b.store_function(dpp_min_event_var, zero);
    }
    if (has_portable_compute_dpp_add) {
        b.store_function(dpp_add_pending_var, no);
        b.store_function(dpp_add_active_var, no);
        b.store_function(dpp_add_event_var, zero);
    }
    if (has_portable_compute_dpp_ror8) {
        b.store_function(dpp_ror8_pending_var, no);
        b.store_function(dpp_ror8_active_var, no);
        b.store_function(dpp_ror8_event_var, zero);
    }
    if (has_portable_mask_ffbh) {
        b.store_function(mask_ffbh_pending_var, no);
        b.store_function(mask_ffbh_mask_var, no);
        b.store_function(mask_ffbh_write_var, no);
        b.store_function(mask_ffbh_event_var, zero);
    }
    if (has_portable_readlane)
        b.store_function(readlane_pending_var, no);
    b.emit_loopmerge(loop_merge, loop_continue);
    b.emit_branch(switch_header);
    b.emit_label(switch_header);
    const uint32_t selector = b.sel(b.load_function(b.t_bool, active_var),
                                    b.load_function(b.t_u32, pc_var), b.uconst(UINT32_MAX));
    b.emit_selmerge(switch_merge);
    b.emit_switch(selector, fallback, switch_cases);

    auto set_next = [&](uint32_t pc) {
        auto found = block_for_pc.find(pc);
        if (pc > end_pc) {
            if (!proven_exit_target(pc)) return false;
            b.store_function(active_var, no);
        }
        else if (found == block_for_pc.end()) {
            if (getenv("PROSPER_DBG")) {
                log_recompile_diagnostic(
                    b.diagnostic, "compute-cfg-successor-reject", "terminal",
                    "pc=%u end=%u blocks=%zu", pc, end_pc, block_for_pc.size());
                std::string successors;
                char successor[32];
                for (const auto& entry : block_for_pc) {
                    const int written = std::snprintf(successor, sizeof successor, " %u",
                                                      entry.first);
                    if (written > 0)
                        successors.append(successor, std::min<size_t>(
                            static_cast<size_t>(written), sizeof(successor) - 1));
                }
                log_recompile_diagnostic(b.diagnostic, "compute-cfg-successors", "terminal",
                                         "%s", successors.c_str());
            }
            return false;
        }
        else b.store_function(pc_var, b.uconst(dispatch_for_block[found->second]));
        return true;
    };
    for (uint32_t dispatch = 0; dispatch < dispatch_blocks.size(); ++dispatch) {
        const uint32_t entry_block = dispatch_blocks[dispatch].front();
        b.emit_label(labels[dispatch]);
        RegState state = load_state(dispatch);
        state.sreg_written = scalar_may_write_in[entry_block];
        for (int reg : state.sreg_written) state.sreg_input.erase(reg);
        if (!direct_descriptor_sregs.empty()) {
            for (int reg : direct_descriptor_sregs)
                if (!state.sreg_written.count(reg)) state.sreg.erase(reg);
        }
        const Rdna2Inst* terminator = nullptr;
        const Rdna2Inst* mbcnt = nullptr;
        const Rdna2Inst* append = nullptr;
        const Rdna2Inst* synchronized_lds_store = nullptr;
        const Rdna2Inst* lds_fminmax = nullptr;
        const Rdna2Inst* swizzle = nullptr;
        const Rdna2Inst* bpermute = nullptr;
        const Rdna2Inst* dpp_min_row_shr = nullptr;
        const Rdna2Inst* dpp_add_row_shr = nullptr;
        const Rdna2Inst* dpp_row_ror8 = nullptr;
        const Rdna2Inst* dpp_add_row_mask = nullptr;
        const Rdna2Inst* mask_ffbh = nullptr;
        const Rdna2Inst* readlane = nullptr;
        const Rdna2Inst* mask_compare = nullptr;
        const Rdna2Inst* exec_saved_mask_compare = nullptr;
        const Rdna2Inst* saved_mask_pair_compare = nullptr;
        const Rdna2Inst* vopc_mask_compare = nullptr;
        const Rdna2Inst* b64_mask_scc_vote = nullptr;
        const uint32_t final_block = dispatch_blocks[dispatch].back();
        const uint32_t final_hi = final_block + 1 < starts.size()
            ? starts[final_block + 1] : UINT32_MAX;
        for (size_t member = 0; member < dispatch_blocks[dispatch].size(); ++member) {
            const uint32_t block = dispatch_blocks[dispatch][member];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            const Rdna2Inst* block_terminator = nullptr;
            const Rdna2Inst* block_mbcnt = nullptr;
            const Rdna2Inst* block_append = nullptr;
            const Rdna2Inst* block_synchronized_lds_store = nullptr;
            const Rdna2Inst* block_lds_fminmax = nullptr;
            const Rdna2Inst* block_swizzle = nullptr;
            const Rdna2Inst* block_bpermute = nullptr;
            const Rdna2Inst* block_dpp_min_row_shr = nullptr;
            const Rdna2Inst* block_dpp_add_row_shr = nullptr;
            const Rdna2Inst* block_dpp_row_ror8 = nullptr;
            const Rdna2Inst* block_dpp_add_row_mask = nullptr;
            const Rdna2Inst* block_mask_ffbh = nullptr;
            const Rdna2Inst* block_readlane = nullptr;
            const Rdna2Inst* block_mask_compare = nullptr;
            const Rdna2Inst* block_exec_saved_mask_compare = nullptr;
            const Rdna2Inst* block_saved_mask_pair_compare = nullptr;
            const Rdna2Inst* block_vopc_mask_compare = nullptr;
            const Rdna2Inst* block_b64_mask_scc_vote = nullptr;
            for (const auto& in : ins) {
                if (in.pc < lo || in.pc >= hi) continue;
                if (cfg_terminator(in)) {
                    block_terminator = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::VOP3 &&
                    (in.opcode == 0x365 || in.opcode == 0x366)) {
                    block_mbcnt = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::DS && (in.opcode == 0x3d || in.opcode == 0x3e)) {
                    block_append = &in;
                    break;
                }
                if (synchronized_lds_store_pcs.contains(in.pc)) {
                    block_synchronized_lds_store = &in;
                    break;
                }
                if (lds_fminmax_pcs.contains(in.pc)) {
                    block_lds_fminmax = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::DS && in.opcode == 0x35) {
                    block_swizzle = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::DS && in.opcode == kDsOpcodeBpermuteB32) {
                    block_bpermute = &in;
                    break;
                }
                if (fragment_dpp_min_row_shr_pcs.contains(in.pc)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[graphics-cfg-dpp-min-row-shr] "
                                     "pc=%u vgpr=v%d amount=%u\n",
                                     in.pc, in.dst.value,
                                     static_cast<uint32_t>(in.dpp_ctrl - 0x110u));
                    block_dpp_min_row_shr = &in;
                    break;
                }
                if (compute_dpp_add_row_shr_pcs.contains(in.pc)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-cfg-dpp-add-row-shr] "
                                     "pc=%u vgpr=v%d amount=%u\n",
                                     in.pc, in.dst.value,
                                     static_cast<uint32_t>(in.dpp_ctrl - 0x110u));
                    block_dpp_add_row_shr = &in;
                    break;
                }
                if (compute_dpp_row_ror8_pcs.contains(in.pc)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-cfg-dpp-row-ror8] "
                                     "pc=%u op=%u dst=v%d src0=v%d src1=v%d\n",
                                     in.pc, static_cast<uint32_t>(dpp_row_ror8_op(in)),
                                     in.dst.value, in.src[0].value,
                                     in.n_src > 1 ? in.src[1].value : -1);
                    block_dpp_row_ror8 = &in;
                    break;
                }
                if (compute_dpp_add_row_mask_pcs.contains(in.pc)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-cfg-dpp-add-row-mask] "
                                     "pc=%u dst=v%d src1=v%d row_mask=0x%x\n",
                                     in.pc, in.dst.value, in.src[1].value,
                                     in.dpp_row_mask);
                    block_dpp_add_row_mask = &in;
                    break;
                }
                if (portable_mask_ffbh_candidate(in) &&
                    !state.sreg.contains(in.src[0].value) &&
                    !state.sreg_input.contains(in.src[0].value)) {
                    const int source = in.src[0].value;
                    const bool source_is_b64_base = state.sreg_bool.contains(source) &&
                        !state.sreg_bool_b32.contains(source);
                    const bool previous_is_b64_base = source > 0 &&
                        state.sreg_bool.contains(source - 1) &&
                        !state.sreg_bool_b32.contains(source - 1);
                    if (source_is_b64_base || previous_is_b64_base) {
                        if (getenv("PROSPER_DBG"))
                            std::fprintf(stderr,
                                         "[compute-cfg-mask-ffbh] pc=%u source=s%d half=%u\n",
                                         in.pc, source,
                                         source_is_b64_base ? 0u : 1u);
                        block_mask_ffbh = &in;
                        break;
                    }
                }
                if (portable_readlane_event_for_pc.contains(in.pc) &&
                    !spill_vgprs.contains(in.src[0].value) &&
                    !state.vgpr_lane_slots.contains(in.src[0].value) &&
                    !state.vgpr_lane_mask_slots.contains(in.src[0].value)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-cfg-portable-readlane] program=0x%llx "
                                     "pc=%u dst=s%d src=v%d\n",
                                     (unsigned long long)b.diagnostic.program_address,
                                     in.pc, in.dst.value, in.src[0].value);
                    block_readlane = &in;
                    break;
                }
                const int mask_compare_source =
                    mask_zero_compare_candidate_source(in);
                if (mask_compare_source >= 0 &&
                    mask_zero_compare_is_proven(in) &&
                    mask_zero_compare_value(state, mask_compare_source)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-mask-compare] pc=%u source=s%d op=0x%x\n",
                                     in.pc, mask_compare_source, in.opcode);
                    block_mask_compare = &in;
                    break;
                }
                const int exec_saved_mask_source =
                    exec_saved_mask_compare_source(in);
                if (exec_saved_mask_source >= 0 &&
                    proven_exec_saved_mask_compare_pcs.contains(in.pc) &&
                    state.sreg_bool.contains(exec_saved_mask_source)) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-exec-saved-mask-compare] "
                                     "pc=%u source=s%d op=0x%x\n",
                                     in.pc, exec_saved_mask_source, in.opcode);
                    block_exec_saved_mask_compare = &in;
                    break;
                }
                const auto saved_pair_sources =
                    saved_mask_pair_compare_sources(in);
                if (saved_pair_sources[0] >= 0 &&
                    proven_saved_mask_pair_compare_pcs.contains(in.pc) &&
                    state.sreg_bool.contains(saved_pair_sources[0]) &&
                    state.sreg_bool.contains(saved_pair_sources[1])) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[%s-saved-mask-pair-compare] "
                                     "pc=%u sources=s[%d:%d],s[%d:%d] op=0x%x\n",
                                     b.is_compute ? "compute" : "graphics",
                                     in.pc, saved_pair_sources[0], saved_pair_sources[0] + 1,
                                     saved_pair_sources[1], saved_pair_sources[1] + 1,
                                     in.opcode);
                    block_saved_mask_pair_compare = &in;
                    break;
                }
                const int vopc_mask_compare_source =
                    vopc_mask_zero_compare_source(in);
                if (vopc_mask_compare_source >= 0 &&
                    (state.sreg_bool.contains(vopc_mask_compare_source) ||
                     (vopc_mask_compare_source == 106 && state.vcc))) {
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr,
                                     "[compute-vopc-mask-compare] pc=%u source=s%d op=0x%x\n",
                                     in.pc, vopc_mask_compare_source, in.opcode);
                    block_vopc_mask_compare = &in;
                    break;
                }
                if (in.fmt == Rdna2Format::EXP) {
                    if (!exp_fn(state, in)) return reject_cfg(in.pc, "export");
                    continue;
                }
                if (proven_wave64_mask_reduction_pcs.contains(in.pc)) {
                    const int source = wave64_mask_reduction_source(in);
                    const bool mask_available = source == 126
                        ? state.exec != 0
                        : source == 106
                        ? state.vcc != 0
                        : state.sreg_bool.contains(source);
                    if (source < 0 || !mask_available)
                        return reject_cfg(in.pc, "proven mask reduction missing mask state");
                    // The dispatcher persists the physical SGPR file and Bool-domain masks in
                    // separate function variables.  At this exact MUST-proven consumer, the u32
                    // pair is only a synthetic placeholder from an earlier scalar lifetime; leave
                    // the generic S_FF1/S_BCNT ambiguity guard intact and expose the proven mask by
                    // removing only those two stale data views.
                    for (int reg = source; reg <= source + 1; ++reg) {
                        state.sreg.erase(reg);
                        state.sreg_input.erase(reg);
                        state.sreg_srt.erase(reg);
                    }
                }
                bool ok = true;
                const SavedB64MaskSnapshot saved_masks = snapshot_saved_b64_masks(state, in);
                const bool handled = emit_alu(b, state, in, ok, allow_exec_update, &safe,
                                              allow_smem, rt, /*allow_wave*/false);
                if (handled && ok)
                    record_scalar_write(
                        state, in,
                        allows_compute_scalar_vcc_bridge(b), saved_masks);
                if (!handled || !ok) {
                    // NOT gated on PROSPER_DBG. log_recompile_diagnostic gates its own PRINTING
                    // on that variable and additionally RECORDS the reason for the unconditional
                    // `[compute] skip unsupported program 0x… reason=…` line. An outer gate here
                    // therefore suppressed the recording too, which is why five GTA V programs
                    // reported reason=unrecorded while their cause existed and was formatted.
                        // The raw INSTRUCTION WORDS, because without them this line cannot be acted
                        // on. `fmt` and `op` are our decoder's own labels, so they identify the
                        // instruction only if you already trust the decode -- and a reject is
                        // precisely the case where you should not. The word is ground truth and
                        // names the instruction in one command:
                        //     llvm-mc -arch=amdgcn -mcpu=gfx1010 -show-encoding
                        // assemble the candidate and compare, exactly as #2275 identified the image
                        // atomics and #2309 identified s_cbranch_vccz from `bf860051`.
                        // Its sibling at the ALU reject site already prints these; this one did not,
                        // so half the rejects from a run were unidentifiable and the two diagnostics
                        // could not be compared (#2309).
                        // `mode` as at the ALU reject site (#2412): `unknown-encoding` means no
                        // lowering exists and one must be written; `unresolved-operand` means the
                        // lowering ran and could not resolve an operand or a resource-table
                        // descriptor. Without it a census cannot tell "implement this" from
                        // "this instruction is fine, its descriptor is not".
                        log_recompile_diagnostic(
                            b.diagnostic, "cfg-recompile-reject", "terminal",
                            "mode=%s pc=%u words=%s len=%u fmt=%d op=0x%x",
                            handled ? "unresolved-operand" : "unknown-encoding",
                            in.pc, reject_words_text(in).c_str(), in.len_dwords,
                            static_cast<int>(in.fmt), in.opcode);
                    return false;
                }
                // Ordinary scalar B64 logicals already produced an exact nonzero SCC id above.
                // Only divert the mask-domain form, whose cross-wave SCC is deliberately poisoned.
                if (b64_mask_scc_vote_pcs.contains(in.pc) && !state.scc) {
                    block_b64_mask_scc_vote = &in;
                    break;
                }
                if (!state.scc && native_b32_mask_scc_vote_pcs.contains(in.pc)) {
                    const auto result = state.sreg_bool.find(in.dst.value);
                    if (result == state.sreg_bool.end() ||
                        !state.sreg_bool_b32.contains(in.dst.value))
                        return reject_cfg(in.pc, "missing-b32-mask-scc-source");
                    state.scc = b.native_wave_any(result->second);
                }
            }
            const bool last = member + 1 == dispatch_blocks[dispatch].size();
            if (!last) {
                // Group construction admits only one-successor plain blocks before the tail.
                if (block_mbcnt || block_append || block_synchronized_lds_store ||
                    block_lds_fminmax ||
                    block_swizzle || block_bpermute ||
                    block_dpp_min_row_shr || block_dpp_add_row_shr ||
                    block_dpp_row_ror8 ||
                    block_dpp_add_row_mask || block_mask_ffbh || block_readlane ||
                    block_mask_compare ||
                    block_exec_saved_mask_compare || block_saved_mask_pair_compare ||
                    block_vopc_mask_compare ||
                    block_b64_mask_scc_vote ||
                    (block_terminator && (block_terminator->is_end ||
                                          block_terminator->opcode != 0x02)))
                    return reject_cfg(starts[block], "invalid-fused-block");
                continue; // consume an unconditional guest branch without a dispatcher round-trip
            }
            terminator = block_terminator;
            mbcnt = block_mbcnt;
            append = block_append;
            synchronized_lds_store = block_synchronized_lds_store;
            lds_fminmax = block_lds_fminmax;
            swizzle = block_swizzle;
            bpermute = block_bpermute;
            dpp_min_row_shr = block_dpp_min_row_shr;
            dpp_add_row_shr = block_dpp_add_row_shr;
            dpp_row_ror8 = block_dpp_row_ror8;
            dpp_add_row_mask = block_dpp_add_row_mask;
            mask_ffbh = block_mask_ffbh;
            readlane = block_readlane;
            mask_compare = block_mask_compare;
            exec_saved_mask_compare = block_exec_saved_mask_compare;
            saved_mask_pair_compare = block_saved_mask_pair_compare;
            vopc_mask_compare = block_vopc_mask_compare;
            b64_mask_scc_vote = block_b64_mask_scc_vote;
        }
        if (mask_ffbh) {
            const int source = mask_ffbh->src[0].value;
            int mask_base = source;
            auto mask = state.sreg_bool.find(mask_base);
            if (mask == state.sreg_bool.end() || state.sreg_bool_b32.contains(mask_base)) {
                mask_base = source - 1;
                mask = state.sreg_bool.find(mask_base);
            }
            const auto event = portable_mask_ffbh_event_for_pc.find(mask_ffbh->pc);
            if (source < 0 || mask_base < 0 || source - mask_base < 0 ||
                source - mask_base > 1 || mask == state.sreg_bool.end() ||
                state.sreg_bool_b32.contains(mask_base) ||
                event == portable_mask_ffbh_event_for_pc.end())
                return reject_cfg(mask_ffbh->pc, "mask-ffbh-source");
            b.store_function(mask_ffbh_pending_var, yes);
            b.store_function(mask_ffbh_mask_var, mask->second);
            b.store_function(mask_ffbh_write_var, state.exec);
            b.store_function(mask_ffbh_event_var, b.uconst(event->second));
            b.store_function(mask_ffbh_half_var,
                b.uconst(static_cast<uint32_t>(source - mask_base)));
            b.store_function(mask_ffbh_dst_var,
                b.uconst(static_cast<uint32_t>(mask_ffbh->dst.value)));
        }
        if (readlane) {
            if (!portable_readlane_event_for_pc.contains(readlane->pc))
                return reject_cfg(readlane->pc, "portable-readlane-event");
            const auto source = state.vreg.find(readlane->src[0].value);
            if (source == state.vreg.end())
                return reject_cfg(readlane->pc, "portable-readlane-source");
            bool selector_ok = true;
            const uint32_t readlane_selector = operand_bits(
                b, state, *readlane, readlane->src[1], &selector_ok);
            if (!selector_ok || !readlane_selector)
                return reject_cfg(readlane->pc, "portable-readlane-selector");
            b.store_function(readlane_pending_var, yes);
            b.store_function(readlane_source_var, source->second);
            b.store_function(readlane_selector_var, readlane_selector);
            b.store_function(readlane_dst_var,
                b.uconst(static_cast<uint32_t>(readlane->dst.value)));
            state.sreg.erase(readlane->dst.value);
            state.sreg_input.erase(readlane->dst.value);
            state.sreg_srt.erase(readlane->dst.value);
        }
        if (mbcnt) {
            if (graphics) {
                log_recompile_diagnostic(b.diagnostic, "graphics-cfg-reject", "terminal",
                                         "pc=%u reason=mbcnt-cross-lane", mbcnt->pc);
                return false;
            }
            bool operand_ok = true;
            uint32_t mask = 0;
            if (b.wave_size == 64 && mbcnt->src[0].kind == OperandKind::SGPR) {
                // Dispatcher Bool variables preserve values after a scalar overwrite by storing
                // false, so map membership is not a lifetime proof. Admit a saved SGPR mask only
                // at the exact consumer where the Wave64 MUST analysis proves its pair root live.
                const auto proof = proven_wave64_mbcnt_mask_root_for_pc.find(mbcnt->pc);
                if (proof == proven_wave64_mbcnt_mask_root_for_pc.end())
                    return reject_cfg(mbcnt->pc, "mbcnt-unproven-saved-mask");
                const auto live = state.sreg_bool.find(proof->second);
                if (live == state.sreg_bool.end())
                    return reject_cfg(mbcnt->pc, "mbcnt-proven-mask-missing-state");
                mask = live->second;
            } else {
                mask = mbcnt_source_bit(
                    b, state, mbcnt->src[0], mbcnt->opcode == 0x366);
            }
            const uint32_t acc = operand_bits(b, state, *mbcnt, mbcnt->src[1], &operand_ok);
            const auto event = mbcnt_event_for_pc.find(mbcnt->pc);
            if (!mask || !operand_ok || event == mbcnt_event_for_pc.end()) return false;
            if (b.native_subgroup_size) {
                // The switch selector is scalar within this exact-size subgroup, so every guest
                // lane reaches the same case.  Execute the wave prefix count here and retain masked-
                // off lanes exactly as an RDNA VGPR write does, instead of dynamically selecting a
                // destination across the entire persistent register file in the common phase.
                const uint32_t result = b.native_compute_mbcnt(
                    mask, acc, mbcnt->opcode == 0x365 ? yes : no);
                const int dst = mbcnt->dst.value;
                const auto old = state.vreg.find(dst);
                state.vreg[dst] = b.sel(
                    state.exec, result, old == state.vreg.end() ? zero : old->second);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(mbcnt_pending_var, yes);
                b.store_function(mbcnt_mask_var, mask);
                b.store_function(mbcnt_low_var, mbcnt->opcode == 0x365 ? yes : no);
                // load_state conservatively marks EXEC narrowed. Writing under the current per-lane EXEC
                // is equivalent for a known-full mask and preserves inactive VGPR lanes for divergent code.
                b.store_function(mbcnt_write_var, state.exec);
                b.store_function(mbcnt_event_var, b.uconst(event->second));
                b.store_function(mbcnt_acc_var, acc);
                b.store_function(mbcnt_dst_var, b.uconst(static_cast<uint32_t>(mbcnt->dst.value)));
            }
        }
        if (append) {
            if (graphics) {
                log_recompile_diagnostic(b.diagnostic, "graphics-cfg-reject", "terminal",
                                         "pc=%u reason=gds-cross-lane", append->pc);
                return false;
            }
            const auto m0 = state.sreg.find(124);
            const auto event = append_event_for_pc.find(append->pc);
            if (m0 == state.sreg.end() || event == append_event_for_pc.end()) return false;
            if (!append->ds_gds) b.declare_lds();
            const uint32_t idx = ds_append_consume_index(
                b, m0->second, append->literal, append->ds_gds);
            if (b.native_subgroup_size) {
                const uint32_t result = append->ds_gds
                    ? b.native_gds_append(idx, state.exec, append->opcode == 0x3d)
                    : b.native_wave_append(
                          idx, state.exec, append->opcode == 0x3d ? yes : no);
                const int dst = append->dst.value;
                const auto old = state.vreg.find(dst);
                state.vreg[dst] = b.sel(
                    state.exec, result, old == state.vreg.end() ? zero : old->second);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(append_pending_var, yes);
                b.store_function(append_active_var, state.exec);
                b.store_function(append_event_var, b.uconst(event->second));
                b.store_function(append_consume_var, append->opcode == 0x3d ? yes : no);
                b.store_function(append_gds_var, append->ds_gds ? yes : no);
                b.store_function(append_idx_var, idx);
                b.store_function(append_dst_var,
                    b.uconst(static_cast<uint32_t>(append->dst.value)));
            }
        }
        if (synchronized_lds_store) {
            if (!synchronized_lds_store_pcs.contains(synchronized_lds_store->pc) ||
                synchronized_lds_store->fmt != Rdna2Format::DS ||
                synchronized_lds_store->ds_gds)
                return reject_cfg(synchronized_lds_store->pc,
                                  "lds-store-common-phase-contract");
            b.declare_lds();
            if (!b.lds_var)
                return reject_cfg(synchronized_lds_store->pc, "lds-store-common-phase-lds");
            auto vread = [&](int reg) {
                const auto value = state.vreg.find(reg);
                return value == state.vreg.end() ? zero : value->second;
            };
            std::vector<std::pair<uint32_t, uint32_t>> writes;
            auto append_write = [&](uint32_t idx, int value_reg) {
                writes.emplace_back(idx, vread(value_reg));
            };
            const Rdna2Inst& store = *synchronized_lds_store;
            if (store.opcode == 0xb0) {
                const auto m0 = state.sreg.find(124);
                if (m0 == state.sreg.end())
                    return reject_cfg(store.pc, "lds-store-common-phase-m0");
                const uint32_t base = b.ibin(
                    Op_BitwiseAnd, m0->second, b.uconst(0xffffu));
                const uint32_t tid = b.ibin(
                    Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1u));
                const uint32_t byte_address = b.ibin(
                    Op_IAdd, b.ibin(Op_IAdd, base, b.uconst(store.literal)),
                    b.ibin(Op_ShiftLeftLogical, tid, b.uconst(2)));
                append_write(
                    b.ibin(Op_ShiftRightLogical, byte_address, b.uconst(2)),
                    store.src[1].value);
            } else if (store.opcode == 0x0e || store.opcode == 0x4e) {
                const uint32_t base = b.ibin(
                    Op_ShiftRightLogical, vread(store.src[0].value), b.uconst(2));
                const uint32_t width = store.opcode == 0x4e ? 2u : 1u;
                const uint32_t offset0 = (store.literal & 0xffu) * width;
                const uint32_t offset1 = ((store.literal >> 8u) & 0xffu) * width;
                const uint32_t idx0 = offset0
                    ? b.ibin(Op_IAdd, base, b.uconst(offset0)) : base;
                const uint32_t idx1 = offset1
                    ? b.ibin(Op_IAdd, base, b.uconst(offset1)) : base;
                for (uint32_t word = 0; word < width; ++word)
                    append_write(word ? b.ibin(Op_IAdd, idx0, b.uconst(word)) : idx0,
                                 store.src[1].value + static_cast<int>(word));
                if ((store.literal & 0xffu) != ((store.literal >> 8u) & 0xffu))
                    for (uint32_t word = 0; word < width; ++word)
                        append_write(word ? b.ibin(Op_IAdd, idx1, b.uconst(word)) : idx1,
                                     store.src[2].value + static_cast<int>(word));
            } else if (store.opcode == 0x0d || store.opcode == 0x4d ||
                       store.opcode == 0xde || store.opcode == 0xdf) {
                const uint32_t byte_address = b.ibin(
                    Op_IAdd, vread(store.src[0].value), b.uconst(store.literal));
                const uint32_t base = b.ibin(
                    Op_ShiftRightLogical, byte_address, b.uconst(2));
                const uint32_t width = store.opcode == 0x0d ? 1u :
                    store.opcode == 0x4d ? 2u : store.opcode == 0xde ? 3u : 4u;
                for (uint32_t word = 0; word < width; ++word)
                    append_write(word ? b.ibin(Op_IAdd, base, b.uconst(word)) : base,
                                 store.src[1].value + static_cast<int>(word));
            }
            if (writes.empty() || writes.size() > synchronized_lds_store_idx_vars.size())
                return reject_cfg(store.pc, "lds-store-common-phase-shape");
            b.store_function(synchronized_lds_store_pending_var, yes);
            b.store_function(synchronized_lds_store_active_var, state.exec);
            b.store_function(synchronized_lds_store_count_var,
                             b.uconst(static_cast<uint32_t>(writes.size())));
            for (size_t word = 0; word < writes.size(); ++word) {
                b.store_function(synchronized_lds_store_idx_vars[word], writes[word].first);
                b.store_function(synchronized_lds_store_value_vars[word], writes[word].second);
            }
        }
        if (lds_fminmax) {
            if (!lds_fminmax_pcs.contains(lds_fminmax->pc) || lds_fminmax->ds_gds ||
                (lds_fminmax->words[1] & 0xffff0000u) != 0u ||
                (lds_fminmax->opcode != kDsOpcodeMinF32 &&
                 lds_fminmax->opcode != kDsOpcodeMaxF32))
                return reject_cfg(lds_fminmax->pc, "lds-fminmax-common-phase-contract");
            b.declare_lds();
            if (!b.lds_var) return reject_cfg(lds_fminmax->pc, "lds-fminmax-lds");
            const auto address = state.vreg.find(lds_fminmax->src[0].value);
            const auto value = state.vreg.find(lds_fminmax->src[1].value);
            const uint32_t byte_address = b.ibin(
                Op_IAdd, address == state.vreg.end() ? zero : address->second,
                b.uconst(lds_fminmax->literal));
            b.store_function(lds_fminmax_pending_var, yes);
            b.store_function(lds_fminmax_active_var, state.exec);
            b.store_function(lds_fminmax_min_var,
                lds_fminmax->opcode == kDsOpcodeMinF32 ? yes : no);
            b.store_function(lds_fminmax_idx_var,
                b.ibin(Op_ShiftRightLogical, byte_address, b.uconst(2)));
            b.store_function(lds_fminmax_value_var,
                value == state.vreg.end() ? zero : value->second);
        }
        if (swizzle) {
            if (!swizzle_pcs.contains(swizzle->pc)) return false;
            uint32_t source_lane = 0;
            if (!b.ds_swizzle_source_lane(swizzle->literal, &source_lane)) return false;
            const auto source = state.vreg.find(swizzle->src[0].value);
            b.store_function(swizzle_pending_var, yes);
            b.store_function(swizzle_active_var, state.exec);
            b.store_function(swizzle_source_var,
                source == state.vreg.end() ? zero : source->second);
            b.store_function(swizzle_source_lane_var, source_lane);
            b.store_function(swizzle_dst_var,
                b.uconst(static_cast<uint32_t>(swizzle->dst.value)));
        }
        if (bpermute) {
            const auto event = bpermute_event_for_pc.find(bpermute->pc);
            if (event == bpermute_event_for_pc.end() || !b.is_compute ||
                bpermute->ds_gds || !b.native_subgroup_size ||
                b.native_subgroup_size != b.wave_size)
                return reject_cfg(bpermute->pc, "ds-bpermute-native-wave-contract");
            // Publish lane-local operands in the selected case. The actual gathers run after the
            // switch merge, where every subgroup invocation participates in uniform control flow.
            const auto address = state.vreg.find(bpermute->src[0].value);
            const auto source = state.vreg.find(bpermute->src[1].value);
            b.store_function(bpermute_pending_var, yes);
            b.store_function(bpermute_active_var, state.exec);
            b.store_function(bpermute_address_var,
                address == state.vreg.end() ? zero : address->second);
            b.store_function(bpermute_source_var,
                source == state.vreg.end() ? zero : source->second);
            b.store_function(bpermute_offset_var, b.uconst(bpermute->literal));
            b.store_function(bpermute_event_var, b.uconst(event->second));
            b.store_function(bpermute_dst_var,
                b.uconst(static_cast<uint32_t>(bpermute->dst.value)));
        }
        if (dpp_min_row_shr) {
            if (!fragment_dpp_min_row_shr(*dpp_min_row_shr))
                return reject_cfg(dpp_min_row_shr->pc, "dpp-min-row-shr-contract");
            const auto event = fragment_dpp_min_event_for_pc.find(dpp_min_row_shr->pc);
            if (event == fragment_dpp_min_event_for_pc.end())
                return reject_cfg(dpp_min_row_shr->pc, "dpp-min-row-shr-event");
            const auto source = state.vreg.find(dpp_min_row_shr->src[0].value);
            b.store_function(dpp_min_pending_var, yes);
            b.store_function(dpp_min_active_var, state.exec);
            b.store_function(dpp_min_source_var,
                source == state.vreg.end() ? zero : source->second);
            b.store_function(dpp_min_amount_var,
                b.uconst(static_cast<uint32_t>(dpp_min_row_shr->dpp_ctrl - 0x110u)));
            b.store_function(dpp_min_dst_var,
                b.uconst(static_cast<uint32_t>(dpp_min_row_shr->dst.value)));
            b.store_function(dpp_min_event_var, b.uconst(event->second));
        }
        if (dpp_add_row_shr) {
            if (!compute_dpp_add_row_shr(*dpp_add_row_shr))
                return reject_cfg(dpp_add_row_shr->pc, "dpp-add-row-shr-contract");
            const auto event = compute_dpp_add_event_for_pc.find(dpp_add_row_shr->pc);
            if (event == compute_dpp_add_event_for_pc.end())
                return reject_cfg(dpp_add_row_shr->pc, "dpp-add-row-shr-event");
            const int dst = dpp_add_row_shr->dst.value;
            const auto source = state.vreg.find(dst);
            const uint32_t source_value =
                source == state.vreg.end() ? zero : source->second;
            const uint32_t amount = b.uconst(
                static_cast<uint32_t>(dpp_add_row_shr->dpp_ctrl - 0x110u));
            if (b.native_subgroup_size) {
                // One exact native subgroup is one guest wave, and the scalar dispatcher selector
                // is subgroup-uniform. The source EXEC bit still matters: FI=0/BOUND_CTRL=0
                // disables a destination whose shifted source lane is inactive.
                uint32_t valid_source = 0;
                const uint32_t shifted = b.subgroup_row_shr_dynamic(
                    source_value, state.exec, amount, 0, &valid_source);
                const uint32_t result = b.ibin(Op_IAdd, source_value, shifted);
                state.vreg[dst] = b.sel(
                    b.land(state.exec, valid_source), result, source_value);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(dpp_add_pending_var, yes);
                b.store_function(dpp_add_active_var, state.exec);
                b.store_function(dpp_add_source_var, source_value);
                b.store_function(dpp_add_amount_var, amount);
                b.store_function(dpp_add_dst_var,
                    b.uconst(static_cast<uint32_t>(dst)));
                b.store_function(dpp_add_event_var, b.uconst(event->second));
            }
        }
        if (dpp_row_ror8) {
            const DppRowRor8Op operation = dpp_row_ror8_op(*dpp_row_ror8);
            if (!compute_dpp_row_ror8(*dpp_row_ror8))
                return reject_cfg(dpp_row_ror8->pc, "dpp-row-ror8-contract");
            const auto event = compute_dpp_ror8_event_for_pc.find(dpp_row_ror8->pc);
            if (event == compute_dpp_ror8_event_for_pc.end())
                return reject_cfg(dpp_row_ror8->pc, "dpp-row-ror8-event");
            const int dst = dpp_row_ror8->dst.value;
            const auto old = state.vreg.find(dst);
            const auto src0 = state.vreg.find(dpp_row_ror8->src[0].value);
            const uint32_t old_value = old == state.vreg.end() ? zero : old->second;
            const uint32_t src0_value = src0 == state.vreg.end() ? zero : src0->second;
            uint32_t src1_value = zero;
            if (dpp_row_ror8->n_src > 1) {
                const auto src1 = state.vreg.find(dpp_row_ror8->src[1].value);
                src1_value = src1 == state.vreg.end() ? zero : src1->second;
            }
            if (b.native_subgroup_size) {
                // One exact native subgroup is one guest wave and this case is subgroup-uniform.
                // FI=0 makes an EXEC-inactive permuted source read as zero. Every stride in this
                // family XORs only bits 0..3, so the source is always an in-range lane of the same
                // 16-lane row and BOUND_CTRL does not decide this case.
                uint32_t valid_source = 0;
                const uint32_t rotated = b.subgroup_row_xor(
                    src0_value, state.exec,
                    [&]{ uint32_t st = 0; dpp_row_xor_ctrl(dpp_row_ror8->dpp_ctrl, &st); return st; }(),
                    &valid_source);
                const uint32_t bounded = b.sel(valid_source, rotated, zero);
                uint32_t result = bounded;
                if (operation == DppRowRor8Op::MinF32)
                    result = b.fext2(Glsl_NMin, bounded, src1_value);
                else if (operation == DppRowRor8Op::MaxF32)
                    result = b.fext2(Glsl_NMax, bounded, src1_value);
                state.vreg[dst] = b.sel(state.exec, result, old_value);
                for (auto& vg : state.vgpr_lane_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
                for (auto& vg : state.vgpr_lane_mask_slots)
                    if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
            } else {
                b.store_function(dpp_ror8_pending_var, yes);
                b.store_function(dpp_ror8_active_var, state.exec);
                b.store_function(dpp_ror8_src0_var, src0_value);
                b.store_function(dpp_ror8_src1_var, src1_value);
                b.store_function(dpp_ror8_op_var,
                    b.uconst(static_cast<uint32_t>(operation)));
                b.store_function(dpp_ror8_dst_var,
                    b.uconst(static_cast<uint32_t>(dst)));
                b.store_function(dpp_ror8_event_var, b.uconst(event->second));
            }
        }
        if (dpp_add_row_mask) {
            if (!compute_dpp_add_row_mask(*dpp_add_row_mask))
                return reject_cfg(dpp_add_row_mask->pc, "dpp-add-row-mask-contract");
            const int dst = dpp_add_row_mask->dst.value;
            const auto old = state.vreg.find(dst);
            const auto addend = state.vreg.find(dpp_add_row_mask->src[1].value);
            const uint32_t old_value = old == state.vreg.end() ? zero : old->second;
            const uint32_t addend_value =
                addend == state.vreg.end() ? zero : addend->second;
            const uint32_t lane = b.ibin(
                Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1u));
            const uint32_t row = b.ibin(
                Op_ShiftRightLogical, lane, b.uconst(4));
            const uint32_t row_bit = b.ibin(
                Op_ShiftLeftLogical, b.uconst(1), row);
            const uint32_t row_selected = b.ucmp(
                Op_INotEqual,
                b.ibin(Op_BitwiseAnd, row_bit,
                       b.uconst(dpp_add_row_mask->dpp_row_mask)),
                zero);
            const uint32_t result = b.ibin(Op_IAdd, old_value, addend_value);
            state.vreg[dst] = b.sel(
                b.land(state.exec, row_selected), result, old_value);
            for (auto& vg : state.vgpr_lane_slots)
                if (vg.first == dst) for (auto& slot : vg.second) slot.second = zero;
            for (auto& vg : state.vgpr_lane_mask_slots)
                if (vg.first == dst) for (auto& slot : vg.second) slot.second = no;
        }
        if (mask_compare) {
            if (b.is_vertex) {
                log_recompile_diagnostic(b.diagnostic, "graphics-cfg-reject", "terminal",
                                         "pc=%u reason=wave-mask-compare", mask_compare->pc);
                return false;
            }
            const int source = mask_zero_compare_candidate_source(*mask_compare);
            const uint32_t value = mask_zero_compare_value(state, source);
            if (!mask_zero_compare_is_proven(*mask_compare) || !value)
                return reject_cfg(mask_compare->pc, "missing-mask-compare-source");
            if (b.native_subgroup_size || b.is_fragment) {
                const uint32_t wave_any = b.is_fragment
                    ? b.fragment_wave_any(value)
                    : b.native_wave_any(value);
                if (!wave_any) return reject_cfg(mask_compare->pc, "mask-vote");
                state.scc = mask_zero_compare_inverts(*mask_compare)
                    ? b.logical_not(wave_any) : wave_any;
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, value);
                b.store_function(vote_invert_var,
                    mask_zero_compare_inverts(*mask_compare) ? yes : no);
                b.store_function(vote_to_scc_var, yes);
            }
        }
        if (exec_saved_mask_compare) {
            const int source =
                exec_saved_mask_compare_source(*exec_saved_mask_compare);
            const auto saved_mask = state.sreg_bool.find(source);
            if (saved_mask == state.sreg_bool.end())
                return reject_cfg(exec_saved_mask_compare->pc,
                                  "missing-exec-saved-mask-compare-source");
            const uint32_t mismatch = b.bsel(
                state.exec, b.logical_not(saved_mask->second), saved_mask->second);
            if (b.native_subgroup_size) {
                const uint32_t different = b.native_wave_any(mismatch);
                if (!different)
                    return reject_cfg(exec_saved_mask_compare->pc,
                                      "exec-saved-mask-vote");
                state.scc = exec_saved_mask_compare->opcode == 0x12
                    ? b.logical_not(different) : different;
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, mismatch);
                b.store_function(vote_invert_var,
                    exec_saved_mask_compare->opcode == 0x12 ? yes : no);
                b.store_function(vote_to_scc_var, yes);
            }
        }
        if (saved_mask_pair_compare) {
            if (b.is_fragment) {
                const auto event =
                    saved_mask_pair_event_for_pc.find(saved_mask_pair_compare->pc);
                if (event == saved_mask_pair_event_for_pc.end())
                    return reject_cfg(saved_mask_pair_compare->pc,
                                      "saved-mask-pair-event");
                b.store_function(saved_mask_pair_pending_var, yes);
                b.store_function(saved_mask_pair_event_var, b.uconst(event->second));
            } else {
                const auto sources =
                    saved_mask_pair_compare_sources(*saved_mask_pair_compare);
                const auto first = state.sreg_bool.find(sources[0]);
                const auto second = state.sreg_bool.find(sources[1]);
                if (first == state.sreg_bool.end() || second == state.sreg_bool.end())
                    return reject_cfg(saved_mask_pair_compare->pc,
                                      "missing-saved-mask-pair-compare-source");
                const uint32_t mismatch = b.bsel(
                    first->second, b.logical_not(second->second), second->second);
                if (b.native_subgroup_size) {
                    const uint32_t different = b.native_wave_any(mismatch);
                    if (!different)
                        return reject_cfg(saved_mask_pair_compare->pc,
                                          "saved-mask-pair-vote");
                    state.scc = saved_mask_pair_compare->opcode == 0x12
                        ? b.logical_not(different) : different;
                } else {
                    b.store_function(vote_pending_var, yes);
                    b.store_function(vote_value_var, mismatch);
                    b.store_function(vote_invert_var,
                        saved_mask_pair_compare->opcode == 0x12 ? yes : no);
                    b.store_function(vote_to_scc_var, yes);
                }
            }
        }
        if (vopc_mask_compare) {
            if (b.is_vertex) {
                log_recompile_diagnostic(b.diagnostic, "graphics-cfg-reject", "terminal",
                                         "pc=%u reason=vopc-wave-mask-compare",
                                         vopc_mask_compare->pc);
                return false;
            }
            const int source = vopc_mask_zero_compare_source(*vopc_mask_compare);
            const auto saved_value = state.sreg_bool.find(source);
            const uint32_t value = source == 106 && state.vcc
                ? state.vcc
                : saved_value != state.sreg_bool.end() ? saved_value->second : 0;
            if (!value)
                return reject_cfg(vopc_mask_compare->pc,
                                  "missing-vopc-mask-compare-source");
            if (b.native_subgroup_size || b.is_fragment) {
                const uint32_t wave_any = b.is_fragment
                    ? b.fragment_wave_any(value)
                    : b.native_wave_any(value);
                if (!wave_any) return reject_cfg(vopc_mask_compare->pc, "vopc-mask-vote");
                const uint32_t condition =
                    vopc_mask_zero_compare_inverts(*vopc_mask_compare)
                        ? b.logical_not(wave_any) : wave_any;
                state.vcc = b.land(state.exec, condition);
                state.sreg_bool[106] = state.vcc;
                state.sreg_bool_narrowed[106] = true;
                if (const auto saved_vcc = mv.find(106); saved_vcc != mv.end())
                    b.store_function(saved_vcc->second, state.vcc);
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, value);
                b.store_function(vote_invert_var,
                    vopc_mask_zero_compare_inverts(*vopc_mask_compare) ? yes : no);
                b.store_function(vote_to_vcc_var, yes);
            }
        }
        if (b64_mask_scc_vote) {
            uint32_t value = 0;
            if (b64_mask_scc_vote->dst.value == 126 ||
                b64_mask_scc_vote->dst.value == 127) {
                value = state.exec;
            } else if (b64_mask_scc_vote->dst.value == 106 ||
                       b64_mask_scc_vote->dst.value == 107) {
                value = state.vcc;
            } else {
                const auto saved = state.sreg_bool.find(b64_mask_scc_vote->dst.value);
                if (saved != state.sreg_bool.end()) value = saved->second;
            }
            if (!value)
                return reject_cfg(b64_mask_scc_vote->pc, "missing-b64-mask-scc-source");
            if (b.native_subgroup_size || b.is_fragment) {
                const uint32_t wave_any = b.is_fragment
                    ? b.fragment_wave_any(value)
                    : b.native_wave_any(value);
                if (!wave_any) return reject_cfg(b64_mask_scc_vote->pc, "b64-mask-scc-vote");
                state.scc = wave_any;
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var, value);
                b.store_function(vote_invert_var, no);
                b.store_function(vote_to_scc_var, yes);
            }
        }
        save_state(state, dispatch);
        if (mask_ffbh) {
            if (!set_next(mask_ffbh->pc + mask_ffbh->len_dwords))
                return reject_cfg(mask_ffbh->pc, "mask-ffbh-successor");
        } else if (readlane) {
            if (!set_next(readlane->pc + readlane->len_dwords))
                return reject_cfg(readlane->pc, "portable-readlane-successor");
        } else if (mbcnt) {
            if (!set_next(mbcnt->pc + mbcnt->len_dwords))
                return reject_cfg(mbcnt->pc, "mbcnt-successor");
        } else if (append) {
            if (!set_next(append->pc + append->len_dwords))
                return reject_cfg(append->pc, "append-successor");
        } else if (synchronized_lds_store) {
            if (!set_next(synchronized_lds_store->pc +
                          synchronized_lds_store->len_dwords))
                return reject_cfg(synchronized_lds_store->pc,
                                  "lds-store-successor");
        } else if (lds_fminmax) {
            if (!set_next(lds_fminmax->pc + lds_fminmax->len_dwords))
                return reject_cfg(lds_fminmax->pc, "lds-fminmax-successor");
        } else if (swizzle) {
            if (!set_next(swizzle->pc + swizzle->len_dwords))
                return reject_cfg(swizzle->pc, "swizzle-successor");
        } else if (bpermute) {
            if (!set_next(bpermute->pc + bpermute->len_dwords))
                return reject_cfg(bpermute->pc, "bpermute-successor");
        } else if (dpp_min_row_shr) {
            if (!set_next(dpp_min_row_shr->pc + dpp_min_row_shr->len_dwords))
                return reject_cfg(dpp_min_row_shr->pc,
                                  "dpp-min-row-shr-successor");
        } else if (dpp_add_row_shr) {
            if (!set_next(dpp_add_row_shr->pc + dpp_add_row_shr->len_dwords))
                return reject_cfg(dpp_add_row_shr->pc,
                                  "dpp-add-row-shr-successor");
        } else if (dpp_row_ror8) {
            if (!set_next(dpp_row_ror8->pc + dpp_row_ror8->len_dwords))
                return reject_cfg(dpp_row_ror8->pc, "dpp-row-ror8-successor");
        } else if (dpp_add_row_mask) {
            if (!set_next(dpp_add_row_mask->pc + dpp_add_row_mask->len_dwords))
                return reject_cfg(dpp_add_row_mask->pc,
                                  "dpp-add-row-mask-successor");
        } else if (mask_compare) {
            if (!set_next(mask_compare->pc + mask_compare->len_dwords))
                return reject_cfg(mask_compare->pc, "mask-compare-successor");
        } else if (exec_saved_mask_compare) {
            if (!set_next(exec_saved_mask_compare->pc +
                          exec_saved_mask_compare->len_dwords))
                return reject_cfg(exec_saved_mask_compare->pc,
                                  "exec-saved-mask-compare-successor");
        } else if (saved_mask_pair_compare) {
            if (!set_next(saved_mask_pair_compare->pc +
                          saved_mask_pair_compare->len_dwords))
                return reject_cfg(saved_mask_pair_compare->pc,
                                  "saved-mask-pair-compare-successor");
        } else if (vopc_mask_compare) {
            if (!set_next(vopc_mask_compare->pc + vopc_mask_compare->len_dwords))
                return reject_cfg(vopc_mask_compare->pc,
                                  "vopc-mask-compare-successor");
        } else if (b64_mask_scc_vote) {
            if (!set_next(b64_mask_scc_vote->pc + b64_mask_scc_vote->len_dwords))
                return reject_cfg(b64_mask_scc_vote->pc,
                                  "b64-mask-scc-successor");
        } else if (!terminator) {
            if (!set_next(final_hi)) return reject_cfg(starts[final_block], "fallthrough-successor");
        } else if (terminator->is_end || terminator->opcode == 0x12) {
            // Vulkan has no compute-stage trap instruction. Ending this dispatcher invocation is
            // the closest fail-closed model of s_trap when no guest trap handler is emulated; the
            // overwhelmingly common release-shader shape branches around the trap on valid data.
            b.store_function(active_var, no);
        } else if (terminator->opcode == 0x02) {
            if (!set_next(branch_target(*terminator)))
                return reject_cfg(terminator->pc, "branch-successor");
        } else {
            uint32_t condition = 0;
            switch (terminator->opcode) {
                // state.scc == 0 marks SCC poisoned by a 64-bit mask op inside this block (its
                // hardware SCC is a cross-lane reduction) — reject rather than branch on a stale
                // value an older s_cmp produced.
                case 0x04: if (!state.scc) return reject_cfg(terminator->pc, "poisoned-scc");
                           condition = b.logical_not(state.scc); break; // s_cbranch_scc0
                case 0x05: if (!state.scc) return reject_cfg(terminator->pc, "poisoned-scc");
                           condition = state.scc; break;
                case 0x06: case 0x07: case 0x08: case 0x09: break;
                default: return reject_cfg(terminator->pc, "branch-opcode");
            }
            const uint32_t target = branch_target(*terminator);
            const uint32_t fallthrough = terminator->pc + terminator->len_dwords;
            auto taken = block_for_pc.find(target), next = block_for_pc.find(fallthrough);
            const bool taken_exit = target > end_pc && proven_exit_target(target);
            const bool next_exit = fallthrough > end_pc && proven_exit_target(fallthrough);
            if ((!taken_exit && taken == block_for_pc.end()) ||
                (!next_exit && next == block_for_pc.end()))
                return reject_cfg(terminator->pc, "branch-target");
            if ((taken_exit || next_exit) && b.is_compute && !b.native_subgroup_size)
                return reject_cfg(terminator->pc, "portable-compute-exit");
            const uint32_t taken_dispatch = taken_exit ? 0 : dispatch_for_block[taken->second];
            const uint32_t next_dispatch = next_exit ? 0 : dispatch_for_block[next->second];
            auto route = [&](uint32_t branch_condition) {
                if (taken_exit || next_exit) {
                    const uint32_t remains_active = taken_exit
                        ? b.logical_not(branch_condition) : branch_condition;
                    b.store_function(active_var, remains_active);
                    b.store_function(pc_var, b.uconst(
                        taken_exit ? next_dispatch : taken_dispatch));
                } else {
                    b.store_function(pc_var,
                        b.sel(branch_condition, b.uconst(taken_dispatch),
                              b.uconst(next_dispatch)));
                }
            };
            if (terminator->opcode == 0x04 || terminator->opcode == 0x05) {
                route(condition);
            } else if (graphics) {
                const uint32_t lane_condition =
                    terminator->opcode <= 0x07 ? state.vcc : state.exec;
                const uint32_t branch_condition =
                    terminator->opcode == 0x06 || terminator->opcode == 0x08
                        ? b.logical_not(lane_condition) : lane_condition;
                route(branch_condition);
            } else if (b.native_subgroup_size) {
                const uint32_t wave_any = b.native_wave_any(
                    terminator->opcode <= 0x07 ? state.vcc : state.exec);
                const uint32_t wave_condition =
                    terminator->opcode == 0x06 || terminator->opcode == 0x08
                        ? b.logical_not(wave_any) : wave_any;
                route(wave_condition);
            } else {
                b.store_function(vote_pending_var, yes);
                b.store_function(vote_value_var,
                    terminator->opcode <= 0x07 ? state.vcc : state.exec);
                b.store_function(vote_invert_var,
                    terminator->opcode == 0x06 || terminator->opcode == 0x08 ? yes : no);
                b.store_function(vote_taken_var, b.uconst(taken_dispatch));
                b.store_function(vote_next_var, b.uconst(next_dispatch));
            }
        }
        b.emit_branch(switch_merge);
    }
    b.emit_label(fallback);
    b.store_function(active_var, no);
    b.emit_branch(switch_merge);
    b.emit_label(switch_merge);
    b.emit_branch(loop_continue);
    b.emit_label(loop_continue);

    // DS_SWIZZLE common phase. The dispatcher cases only publish source data and a lane selector;
    // every invocation executes the actual subgroup gathers here in uniform control flow. This is
    // required even though the instruction does not touch LDS: subgroup operations in a divergent
    // switch arm would have undefined participation on Vulkan.
    if (!swizzle_pcs.empty()) {
        const uint32_t swizzle_pending = b.load_function(b.t_bool, swizzle_pending_var);
        const uint32_t swizzle_active = b.load_function(b.t_bool, swizzle_active_var);
        const uint32_t swizzle_lane = b.load_function(b.t_u32, swizzle_source_lane_var);
        const uint32_t swizzle_value = b.subgroup_shuffle(
            b.load_function(b.t_u32, swizzle_source_var), swizzle_lane);
        const uint32_t source_active_word = b.sel(
            b.land(swizzle_pending, swizzle_active), b.uconst(1), zero);
        const uint32_t source_active = b.ucmp(
            Op_INotEqual, b.subgroup_shuffle(source_active_word, swizzle_lane), zero);
        const uint32_t swizzle_result = b.sel(source_active, swizzle_value, zero);
        const uint32_t swizzle_dst = b.load_function(b.t_u32, swizzle_dst_var);
        const uint32_t swizzle_write = b.land(swizzle_pending, swizzle_active);
        for (const auto& kv : vv) {
            const uint32_t selected = b.land(
                swizzle_write,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, swizzle_result, old));
        }
        // An ordinary VGPR definition ends a scalar lane-spill lifetime at that physical register.
        for (const auto& kv : lv) {
            const uint32_t selected = b.land(
                swizzle_pending,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, zero, old));
        }
        for (const auto& kv : lmv) {
            const uint32_t selected = b.land(
                swizzle_pending,
                b.ucmp(Op_IEqual, swizzle_dst,
                       b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_bool, kv.second);
            b.store_function(kv.second, b.bsel(selected, no, old));
        }
    }

    // DS_BPERMUTE common phase. Even the exact native dispatcher publishes operands in its switch
    // case and performs subgroup gathers here: keeping all lanes at one structurally uniform merge
    // avoids implementation-dependent participation in a case arm. Static event tags accompany
    // DATA0 and EXEC so adjacent BPERMUTE sites cannot consume one another's mailbox values.
    if (has_bpermute) {
        const uint32_t pending = b.load_function(b.t_bool, bpermute_pending_var);
        const uint32_t active = b.load_function(b.t_bool, bpermute_active_var);
        const uint32_t event = b.load_function(b.t_u32, bpermute_event_var);
        const uint32_t result = b.ds_bpermute_b32(
            b.load_function(b.t_u32, bpermute_address_var),
            b.load_function(b.t_u32, bpermute_source_var),
            b.land(pending, active),
            b.load_function(b.t_u32, bpermute_offset_var), event);
        const uint32_t dst = b.load_function(b.t_u32, bpermute_dst_var);
        const uint32_t write = b.land(pending, active);
        for (const auto& kv : vv) {
            const uint32_t selected = b.land(
                write, b.ucmp(Op_IEqual, dst,
                              b.uconst(static_cast<uint32_t>(kv.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, result, old));
        }
        // A physical VGPR definition ends any scalar lane-spill lifetime even when EXEC suppresses
        // this lane's data write, matching predicate_write and the DS_SWIZZLE phase above.
        for (const auto& kv : lv) {
            const uint32_t selected = b.land(
                pending, b.ucmp(Op_IEqual, dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, zero, old));
        }
        for (const auto& kv : lmv) {
            const uint32_t selected = b.land(
                pending, b.ucmp(Op_IEqual, dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_bool, kv.second);
            b.store_function(kv.second, b.bsel(selected, no, old));
        }
    }

    // Fragment saved-mask pair comparisons execute their subgroup votes here, outside the
    // lane-divergent dispatcher switch.  Every invocation evaluates every static event from the
    // persistent mask register file, so a mismatch bit remains visible even when that lane is
    // currently parked at another guest PC.  The publishing lane's event tag selects the exact
    // operand pair and EQ/LG polarity whose result updates its SCC.
    if (has_saved_mask_pair_events) {
        const uint32_t pending =
            b.load_function(b.t_bool, saved_mask_pair_pending_var);
        const uint32_t event =
            b.load_function(b.t_u32, saved_mask_pair_event_var);
        for (size_t index = 0; index < saved_mask_pair_events.size(); ++index) {
            const auto& compare = saved_mask_pair_events[index];
            const auto first_var = mv.find(compare.first);
            const auto second_var = mv.find(compare.second);
            if (first_var == mv.end() || second_var == mv.end())
                return reject_cfg(0, "missing-saved-mask-pair-common-source");
            const uint32_t first = b.load_function(b.t_bool, first_var->second);
            const uint32_t second = b.load_function(b.t_bool, second_var->second);
            const uint32_t mismatch = b.bsel(
                first, b.logical_not(second), second);
            const uint32_t different = b.fragment_wave_any(mismatch);
            if (!different) return reject_cfg(0, "saved-mask-pair-common-vote");
            const uint32_t result = compare.opcode == 0x12
                ? b.logical_not(different) : different;
            const uint32_t selected = b.land(
                pending,
                b.ucmp(Op_IEqual, event,
                       b.uconst(static_cast<uint32_t>(index + 1))));
            const uint32_t old = b.load_function(b.t_bool, scc_var);
            b.store_function(scc_var, b.bsel(selected, result, old));
        }
    }

    // Fragment DPP V_MIN_U32 common phase. A graphics dispatcher's PC is lane-local, so a source
    // lane is eligible only when it published this same static DPP event and its EXEC bit is active.
    // The source's event tag is shuffled beside its value and compared with the destination tag. The
    // unbounded ROW_SHR contract disables out-of-row/inactive-source lanes and preserves VDST;
    // active destination lanes reduce the shifted neighbor with their unchanged local value.
    if (!fragment_dpp_min_row_shr_pcs.empty()) {
        const uint32_t pending = b.load_function(b.t_bool, dpp_min_pending_var);
        const uint32_t active = b.load_function(b.t_bool, dpp_min_active_var);
        const uint32_t source = b.load_function(b.t_u32, dpp_min_source_var);
        uint32_t valid_source = 0;
        const uint32_t shifted = b.subgroup_row_shr_dynamic(
            source, b.land(pending, active),
            b.load_function(b.t_u32, dpp_min_amount_var),
            b.load_function(b.t_u32, dpp_min_event_var), &valid_source);
        const uint32_t result = b.uext2(Glsl_UMin, shifted, source);
        const uint32_t write = b.land(b.land(pending, active), valid_source);
        const uint32_t dst = b.load_function(b.t_u32, dpp_min_dst_var);
        for (int reg : fragment_dpp_min_row_shr_dsts) {
            const auto kv = vv.find(reg);
            if (kv == vv.end()) return reject_cfg(0, "missing-dpp-min-row-shr-dst");
            const uint32_t selected = b.land(
                write, b.ucmp(Op_IEqual, dst,
                              b.uconst(static_cast<uint32_t>(reg))));
            const uint32_t old = b.load_function(b.t_u32, kv->second);
            b.store_function(kv->second, b.sel(selected, result, old));
        }
        // A physical VGPR definition ends scalar lane-spill lifetimes even when EXEC suppresses
        // this lane's data write, matching predicate_write and the DS_SWIZZLE common phase above.
        for (const auto& kv : lv) {
            if (!fragment_dpp_min_row_shr_dsts.contains(kv.first.first)) continue;
            const uint32_t selected = b.land(
                pending, b.ucmp(Op_IEqual, dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_u32, kv.second);
            b.store_function(kv.second, b.sel(selected, zero, old));
        }
        for (const auto& kv : lmv) {
            if (!fragment_dpp_min_row_shr_dsts.contains(kv.first.first)) continue;
            const uint32_t selected = b.land(
                pending, b.ucmp(Op_IEqual, dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
            const uint32_t old = b.load_function(b.t_bool, kv.second);
            b.store_function(kv.second, b.bsel(selected, no, old));
        }
    }

    // Apply the diagnostic trip bound to whichever back-edge this dispatcher actually emits.
    //
    // This used to live only in the portable branch below, while the arm was announced before the
    // split — so a DIRECT dispatcher printed "bounded" and then emitted an unbounded loop. An
    // announced-but-inert lever is worse than no lever: it invites exactly the reading that the
    // instrument was applied, which is how a null result gets published as evidence.
    const uint32_t cfg_trip_only_ordinal = compute_trip_bound_settings().only_ordinal;
    auto apply_trip_bound = [&](uint32_t keep_going) {
        if (!trip_var) return keep_going;
        // Updated on EVERY back-edge traversal, not only on the hit, so the extremes describe the
        // whole run rather than its final instant.
        const uint32_t dispatch_now = b.load_function(b.t_u32, pc_var);
        // PROSPER_CFG_TRIP_BOUND_ORDINAL=K counts only the traversals about to dispatch ordinal K.
        // Unselected traversals still run -- they simply do not advance the counter -- so a cap
        // aimed at one guest loop leaves every other loop in the same program running to its own
        // natural end. That is what makes a NEGATIVE arm mean something: the loop was capped and the
        // device still died, so this loop is not the runaway.
        const uint32_t counts =
            cfg_trip_only_ordinal == ComputeTripBoundSettings::kAllOrdinals
                ? 0u
                : b.ucmp(Op_IEqual, dispatch_now, b.uconst(cfg_trip_only_ordinal));
        const uint32_t previous_trip = b.load_function(b.t_u32, trip_var);
        const uint32_t incremented = b.ibin(Op_IAdd, previous_trip, b.uconst(1));
        const uint32_t next_trip = counts ? b.sel(counts, incremented, previous_trip) : incremented;
        b.store_function(trip_var, next_trip);
        const uint32_t old_min = b.load_function(b.t_u32, dispatch_min_var);
        b.store_function(dispatch_min_var,
                         b.sel(b.ucmp(Op_ULessThan, dispatch_now, old_min), dispatch_now, old_min));
        const uint32_t old_max = b.load_function(b.t_u32, dispatch_max_var);
        b.store_function(dispatch_max_var,
                         b.sel(b.ucmp(Op_UGreaterThan, dispatch_now, old_max), dispatch_now,
                               old_max));
        const uint32_t under_bound = b.ucmp(Op_ULessThan, next_trip, b.uconst(cfg_trip_bound));
        // WITNESS. A cap that is armed but never reached proves nothing, so record the hit where the
        // host can read it: the internal GDS buffer's top five dwords (kComputeTripWitnessDword),
        // which the host prepares before this dispatch and only touches while armed.
        // Predicated on "still nominally running AND the bound just ran out", so a loop that exits
        // normally writes nothing and the ABSENCE of a record is itself an answer.
        const uint32_t hit = b.land(keep_going, b.logical_not(under_bound));
        // THE WITNESS IS COMPUTE-ONLY, AND THE BOUND IS NOT. Only the compute executor binds the
        // internal GDS buffer (kComputeInternalGdsBinding, gpu_executor.cpp) and only the compute
        // host prepares/reads/restores its top dwords. A graphics pipeline binds nothing at that
        // slot, so emitting the witness in a vertex or fragment module declares a descriptor the
        // pipeline layout does not have -- which does not merely lose the record, it can stop the
        // instrumented draw from being created at all. That failure mode is the exact one this
        // instrument exists to prevent: the draw disappears, the device survives, and the run reads
        // as "the bound fixed it" when the bound never executed. So bound the loop in every stage,
        // publish only where a reader exists, and say which case this is (#3193).
        if (b.is_compute) {
            // EVERY field is published with a device-scope atomic, including the two that are
            // invocation-invariant. Concurrent non-atomic stores of the same value are still a data race
            // by the memory model, and "they happen to agree" is not a publication protocol -- it is the
            // same reasoning that made the last-writer range look coherent. Max is idempotent for a flag
            // and for a compile-time constant, so this costs nothing and removes the exception.
            b.compute_gds_atomic_minmax(Op_AtomicUMax, b.uconst(kComputeTripWitnessDword + 0),
                                        b.uconst(1), hit);
            b.compute_gds_atomic_minmax(Op_AtomicUMax, b.uconst(kComputeTripWitnessDword + 1),
                                        b.uconst(cfg_phase), hit);
            // Fields 2..4 are per-invocation and must be REDUCED, not overwritten. The deleted field
            // here was the dispatcher ordinal at the instant one invocation hit the cap: a single
            // sample, unusable for the question ("is it cycling?"), and the field whose label was
            // published wrongly twice. The span it belonged to is what actually answers that, so only
            // the span survives -- and as true extremes over every invocation rather than whichever
            // wrote last.
            b.compute_gds_atomic_minmax(Op_AtomicUMax, b.uconst(kComputeTripWitnessDword + 2),
                                        next_trip, hit);
            b.compute_gds_atomic_minmax(Op_AtomicUMin, b.uconst(kComputeTripWitnessDword + 3),
                                        b.load_function(b.t_u32, dispatch_min_var), hit);
            b.compute_gds_atomic_minmax(Op_AtomicUMax, b.uconst(kComputeTripWitnessDword + 4),
                                        b.load_function(b.t_u32, dispatch_max_var), hit);
        } else {
            // Announced once per program, not per back-edge, and phrased as what the run CANNOT
            // tell you: without a witness, a surviving device is evidence about the bound only if
            // the draw is independently shown to have still executed (the census counts it).
            static std::mutex graphics_note_mutex;
            static std::set<uint64_t> graphics_noted;
            bool first_note = false;
            {
                std::lock_guard lock(graphics_note_mutex);
                first_note = graphics_noted.insert(b.diagnostic.program_address).second;
            }
            if (first_note)
                fprintf(stderr,
                        "[cfg-trip-bound] program 0x%llx is a GRAPHICS stage: the back edge IS "
                        "bounded, but no witness is emitted (the internal GDS buffer is bound by "
                        "the compute executor only). A surviving device therefore does not by "
                        "itself prove the cap was reached -- confirm the draw still ran, e.g. with "
                        "PROSPER_DRAW_PROGRAM_CENSUS.\n",
                        static_cast<unsigned long long>(b.diagnostic.program_address));
        }
        return b.land(keep_going, under_bound);
    };
    if (direct_dispatch) {
        // Native wave operations and votes were already resolved in the selected case.  PC and
        // ACTIVE are scalar guest-wave state, so every invocation in this exact-size subgroup has
        // the same value; another subgroup in the workgroup may still leave independently because
        // this path contains no workgroup barriers.
        b.emit_condbranch(apply_trip_bound(b.load_function(b.t_bool, active_var)),
                          loop_header, loop_merge);
    } else {
    // Atomicized LDS-store common phase. Each guest DS_WRITE packet is one dispatcher event. The
    // trailing barrier completes every lane's complete packet before another guest store packet can
    // begin, preserving RDNA wave instruction order even when two packets' address footprints alias.
    if (has_synchronized_lds_store_event) {
        b.barrier();
        const uint32_t pending_and_active = b.land(
            b.load_function(b.t_bool, synchronized_lds_store_pending_var),
            b.load_function(b.t_bool, synchronized_lds_store_active_var));
        const uint32_t count = b.load_function(
            b.t_u32, synchronized_lds_store_count_var);
        for (uint32_t word = 0; word < synchronized_lds_store_idx_vars.size(); ++word) {
            const uint32_t word_block = b.id(), word_merge = b.id();
            const uint32_t perform_word = b.land(
                pending_and_active,
                b.ucmp(Op_UGreaterThan, count, b.uconst(word)));
            b.emit_selmerge(word_merge);
            b.emit_condbranch(perform_word, word_block, word_merge);
            b.emit_label(word_block);
            b.lds_atomic(
                Op_AtomicExchange,
                b.load_function(b.t_u32, synchronized_lds_store_idx_vars[word]),
                b.load_function(b.t_u32, synchronized_lds_store_value_vars[word]),
                false, yes);
            b.emit_branch(word_merge);
            b.emit_label(word_merge);
            // Wide/write2 DS instructions have ordered component effects. Keep each component's
            // exchange globally complete before the following one can begin; the barrier remains
            // outside the conditional so every workgroup invocation participates.
            b.barrier();
        }
    }

    // DS_MIN/MAX_F32 common phase. Dispatcher cases publish one lane-local atomic request without
    // touching LDS. Every invocation, including lanes whose guest wave trapped or ended, reaches the
    // publication barrier here. The trailing barrier completes this event before the next dispatcher
    // iteration can execute another atomic or the lane-zero gather. Admission requires a single guest
    // wave whenever ordinary initializer stores precede the atomics, so this does not invent ownership
    // between independently executing guest waves.
    if (has_lds_fminmax_event) {
        b.barrier();
        const uint32_t perform = b.land(
            b.load_function(b.t_bool, lds_fminmax_pending_var),
            b.load_function(b.t_bool, lds_fminmax_active_var));
        const uint32_t atomic_block = b.id(), atomic_merge = b.id();
        b.emit_selmerge(atomic_merge);
        b.emit_condbranch(perform, atomic_block, atomic_merge);
        b.emit_label(atomic_block);
        const uint32_t min_block = b.id(), max_block = b.id(), operation_merge = b.id();
        const uint32_t is_min = b.load_function(b.t_bool, lds_fminmax_min_var);
        b.emit_selmerge(operation_merge);
        b.emit_condbranch(is_min, min_block, max_block);
        b.emit_label(min_block);
        b.lds_atomic_fminmax(
            b.load_function(b.t_u32, lds_fminmax_idx_var),
            b.load_function(b.t_u32, lds_fminmax_value_var), true, false, yes);
        b.emit_branch(operation_merge);
        b.emit_label(max_block);
        b.lds_atomic_fminmax(
            b.load_function(b.t_u32, lds_fminmax_idx_var),
            b.load_function(b.t_u32, lds_fminmax_value_var), false, false, yes);
        b.emit_branch(operation_merge);
        b.emit_label(operation_merge);
        b.emit_branch(atomic_merge);
        b.emit_label(atomic_merge);
        b.barrier();
    }

    // Portable compute DPP V_ADD_NC_U32 common phase. Host subgroup shuffles cannot model a
    // Wave64 guest on a subgroup32 device, and different guest waves can reach different static
    // DPP sites in one dispatcher iteration. Publish a full source value plus an event/EXEC word
    // for every workgroup invocation, then address the shifted lane directly inside the same guest
    // 16-lane row. Two barriers bracket the scratch lifetime so later MBCNT/vote phases can reuse
    // the value plane without observing a peer's previous dispatcher iteration.
    if (has_portable_compute_dpp_add) {
    const uint32_t dpp_pending = b.load_function(b.t_bool, dpp_add_pending_var);
    const uint32_t dpp_active = b.load_function(b.t_bool, dpp_add_active_var);
    const uint32_t dpp_source = b.load_function(b.t_u32, dpp_add_source_var);
    const uint32_t dpp_event = b.load_function(b.t_u32, dpp_add_event_var);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(dpp_value_base), b.linear_localid), dpp_source);
    const uint32_t dpp_metadata = b.sel(
        dpp_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, dpp_event, b.uconst(1)),
               b.sel(dpp_active, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(dpp_metadata_base), b.linear_localid),
        dpp_metadata);
    b.barrier();

    const uint32_t dpp_amount = b.load_function(b.t_u32, dpp_add_amount_var);
    const uint32_t dpp_row_lane = b.ibin(
        Op_BitwiseAnd, b.linear_localid, b.uconst(15));
    const uint32_t dpp_in_bounds = b.ucmp(
        Op_UGreaterThanEqual, dpp_row_lane, dpp_amount);
    // Keep even the disabled lane's scratch address valid. BOUND_CTRL=0 uses the validity gate
    // below to preserve old VDST rather than consuming this self-addressed placeholder.
    const uint32_t dpp_source_index = b.sel(
        dpp_in_bounds,
        b.ibin(Op_ISub, b.linear_localid, dpp_amount),
        b.linear_localid);
    const uint32_t dpp_shifted = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(dpp_value_base), dpp_source_index));
    const uint32_t dpp_source_metadata = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(dpp_metadata_base), dpp_source_index));
    const uint32_t dpp_source_event = b.ibin(
        Op_ShiftRightLogical, dpp_source_metadata, b.uconst(1));
    const uint32_t dpp_source_active = b.ucmp(
        Op_INotEqual,
        b.ibin(Op_BitwiseAnd, dpp_source_metadata, b.uconst(1)), zero);
    uint32_t dpp_valid_source = b.land(
        dpp_in_bounds, dpp_source_active);
    dpp_valid_source = b.land(
        dpp_valid_source, b.ucmp(Op_IEqual, dpp_source_event, dpp_event));
    const uint32_t dpp_result = b.ibin(Op_IAdd, dpp_source, dpp_shifted);
    const uint32_t dpp_write = b.land(
        b.land(dpp_pending, dpp_active), dpp_valid_source);
    const uint32_t dpp_dst = b.load_function(b.t_u32, dpp_add_dst_var);
    for (int reg : compute_dpp_add_row_shr_dsts) {
        const auto kv = vv.find(reg);
        if (kv == vv.end()) return reject_cfg(0, "missing-dpp-add-row-shr-dst");
        const uint32_t selected = b.land(
            dpp_write, b.ucmp(Op_IEqual, dpp_dst,
                              b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, kv->second);
        b.store_function(kv->second, b.sel(selected, dpp_result, old));
    }
    // The physical VGPR definition invalidates scalar lane-spill aliases even when EXEC or the
    // shifted source suppresses this invocation's data write.
    for (const auto& kv : lv) {
        if (!compute_dpp_add_row_shr_dsts.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        if (!compute_dpp_add_row_shr_dsts.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    // Portable compute DPP ROW_ROR:8 common phase. This is deliberately separate from the ROW_SHR
    // add phase above: each phase publishes its own pending state, consumes it between two workgroup
    // barriers, and only then permits the other operation to reuse the scratch planes. Event IDs
    // distinguish static sites; the operation tag distinguishes MOV/MIN/MAX semantics per invocation.
    if (has_portable_compute_dpp_ror8) {
    const uint32_t dpp_pending = b.load_function(b.t_bool, dpp_ror8_pending_var);
    const uint32_t dpp_active = b.load_function(b.t_bool, dpp_ror8_active_var);
    const uint32_t dpp_src0 = b.load_function(b.t_u32, dpp_ror8_src0_var);
    const uint32_t dpp_src1 = b.load_function(b.t_u32, dpp_ror8_src1_var);
    const uint32_t dpp_operation = b.load_function(b.t_u32, dpp_ror8_op_var);
    const uint32_t dpp_event = b.load_function(b.t_u32, dpp_ror8_event_var);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(dpp_value_base), b.linear_localid), dpp_src0);
    const uint32_t dpp_metadata = b.sel(
        dpp_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, dpp_event, b.uconst(1)),
               b.sel(dpp_active, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(dpp_metadata_base), b.linear_localid),
        dpp_metadata);
    b.barrier();

    // XOR 8 exchanges the two eight-lane halves without crossing an architectural DPP16 row.
    // This uses guest linear-local order, not the implementation-defined Vulkan subgroup lane ID.
    const uint32_t dpp_rotated_index = b.ibin(
        Op_BitwiseXor, b.linear_localid, b.uconst(8));
    const uint32_t dpp_source_in_bounds = b.ucmp(
        Op_ULessThan, dpp_rotated_index, b.uconst(b.local_count));
    // A partial final DPP16 row has no invocation to initialize the rotated slot. Address this
    // lane's initialized placeholder and let FI=0's validity gate supply zero for the missing peer.
    const uint32_t dpp_source_index = b.sel(
        dpp_source_in_bounds, dpp_rotated_index, b.linear_localid);
    const uint32_t dpp_rotated = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(dpp_value_base), dpp_source_index));
    const uint32_t dpp_source_metadata = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(dpp_metadata_base), dpp_source_index));
    const uint32_t dpp_source_event = b.ibin(
        Op_ShiftRightLogical, dpp_source_metadata, b.uconst(1));
    const uint32_t dpp_source_active = b.ucmp(
        Op_INotEqual,
        b.ibin(Op_BitwiseAnd, dpp_source_metadata, b.uconst(1)), zero);
    const uint32_t dpp_valid_source = b.land(
        dpp_source_in_bounds,
        b.land(dpp_source_active,
               b.ucmp(Op_IEqual, dpp_source_event, dpp_event)));
    // FI=0 requires an EXEC-active source. BOUND_CTRL=1 supplies zero when that source is invalid,
    // but the active destination still writes the operation's result. MOV uses the bounded source;
    // MIN/MAX combine it with the destination lane's unpermuted SRC1.
    const uint32_t dpp_bounded = b.sel(dpp_valid_source, dpp_rotated, zero);
    uint32_t dpp_result = dpp_bounded;
    dpp_result = b.sel(
        b.ucmp(Op_IEqual, dpp_operation,
               b.uconst(static_cast<uint32_t>(DppRowRor8Op::MinF32))),
        b.fext2(Glsl_NMin, dpp_bounded, dpp_src1), dpp_result);
    dpp_result = b.sel(
        b.ucmp(Op_IEqual, dpp_operation,
               b.uconst(static_cast<uint32_t>(DppRowRor8Op::MaxF32))),
        b.fext2(Glsl_NMax, dpp_bounded, dpp_src1), dpp_result);
    const uint32_t dpp_write = b.land(dpp_pending, dpp_active);
    const uint32_t dpp_dst = b.load_function(b.t_u32, dpp_ror8_dst_var);
    for (int reg : compute_dpp_row_ror8_dsts) {
        const auto kv = vv.find(reg);
        if (kv == vv.end()) return reject_cfg(0, "missing-dpp-row-ror8-dst");
        const uint32_t selected = b.land(
            dpp_write, b.ucmp(Op_IEqual, dpp_dst,
                              b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, kv->second);
        b.store_function(kv->second, b.sel(selected, dpp_result, old));
    }
    // A physical destination definition invalidates scalar lane aliases even when EXEC suppresses
    // this invocation's data write, matching predicate_write and the existing DPP add phase.
    for (const auto& kv : lv) {
        if (!compute_dpp_row_ror8_dsts.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        if (!compute_dpp_row_ror8_dsts.contains(kv.first.first)) continue;
        const uint32_t selected = b.land(
            dpp_pending, b.ucmp(Op_IEqual, dpp_dst,
                                b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    // MBCNT common phase. Cases publish an event-tagged mask bit and accumulator, but never emit a
    // barrier themselves. Every invocation—including ended waves and lanes currently at a different
    // guest block—therefore executes these two barriers in identical structured control flow. Event
    // tags keep contributions from different static MBCNT sites isolated if malformed/nonuniform
    // scalar state ever lets waves reach different sites during the same dispatcher iteration.
    const uint32_t mbcnt_wave_index = b.ibin(
        Op_ShiftRightLogical, b.linear_localid,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    const uint32_t mbcnt_lane = b.ibin(
        Op_BitwiseAnd, b.linear_localid, b.uconst(b.wave_size - 1));

    if (has_portable_mask_ffbh) {
    // Portable Wave64 saved-mask FFBH phase. Each publishing lane contributes its one predicate bit
    // with a static-event tag. Lane zero assembles the selected architectural 32-bit half in LDS,
    // after which every lane applies the ordinary V_FFBH_U32 semantics and predicates the VGPR
    // write by its own EXEC. This deliberately does not use a host subgroup ballot: the portable
    // route has no exact-width contract, and a narrower ballot would silently lose guest lanes.
    const uint32_t mask_ffbh_pending =
        b.load_function(b.t_bool, mask_ffbh_pending_var);
    const uint32_t mask_ffbh_mask = b.load_function(b.t_bool, mask_ffbh_mask_var);
    const uint32_t mask_ffbh_tag = b.load_function(b.t_u32, mask_ffbh_event_var);
    const uint32_t mask_ffbh_encoded = b.sel(
        mask_ffbh_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, mask_ffbh_tag, b.uconst(1)),
               b.sel(mask_ffbh_mask, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(b.linear_localid, mask_ffbh_encoded);
    b.barrier();

    const uint32_t mask_ffbh_leader = b.id(), mask_ffbh_assembled = b.id();
    const uint32_t mask_ffbh_is_leader = b.land(
        mask_ffbh_pending, b.ucmp(Op_IEqual, mbcnt_lane, zero));
    b.emit_selmerge(mask_ffbh_assembled);
    b.emit_condbranch(mask_ffbh_is_leader, mask_ffbh_leader, mask_ffbh_assembled);
    b.emit_label(mask_ffbh_leader);
    const uint32_t mask_ffbh_wave_base = b.ibin(
        Op_ShiftLeftLogical, mbcnt_wave_index, b.uconst(6));
    const uint32_t mask_ffbh_half = b.load_function(b.t_u32, mask_ffbh_half_var);
    uint32_t mask_ffbh_word = zero;
    for (uint32_t bit = 0; bit < 32; ++bit) {
        const uint32_t candidate_lane = b.ibin(
            Op_IAdd, b.uconst(bit),
            b.ibin(Op_ShiftLeftLogical, mask_ffbh_half, b.uconst(5)));
        const uint32_t candidate_index = b.ibin(
            Op_IAdd, mask_ffbh_wave_base, candidate_lane);
        const uint32_t candidate = b.cfg_scratch_load(candidate_index);
        const uint32_t candidate_tag = b.ibin(
            Op_ShiftRightLogical, candidate, b.uconst(1));
        uint32_t include = b.ucmp(Op_IEqual, candidate_tag, mask_ffbh_tag);
        include = b.land(
            include, b.ucmp(Op_ULessThan, candidate_index, b.uconst(b.local_count)));
        const uint32_t candidate_bit = b.ibin(
            Op_BitwiseAnd, candidate, b.uconst(1));
        const uint32_t positioned = b.ibin(
            Op_ShiftLeftLogical, candidate_bit, b.uconst(bit));
        mask_ffbh_word = b.ibin(
            Op_BitwiseOr, mask_ffbh_word, b.sel(include, positioned, zero));
    }
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index),
        mask_ffbh_word);
    b.emit_branch(mask_ffbh_assembled);
    b.emit_label(mask_ffbh_assembled);
    b.barrier();

    const uint32_t mask_ffbh_result = b.ffbh_u32(b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index)));
    const uint32_t mask_ffbh_dst = b.load_function(b.t_u32, mask_ffbh_dst_var);
    const uint32_t mask_ffbh_write = b.land(
        mask_ffbh_pending, b.load_function(b.t_bool, mask_ffbh_write_var));
    for (int reg : portable_mask_ffbh_dsts) {
        const auto destination = vv.find(reg);
        if (destination == vv.end()) return reject_cfg(0, "missing-mask-ffbh-dst");
        const uint32_t selected = b.land(
            mask_ffbh_write,
            b.ucmp(Op_IEqual, mask_ffbh_dst, b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, destination->second);
        b.store_function(destination->second, b.sel(selected, mask_ffbh_result, old));
    }
    // As for every ordinary VALU destination, the physical write ends scalar-spill aliases even
    // where EXEC suppresses this lane's data update.
    for (const auto& kv : lv) {
        const uint32_t selected = b.land(
            mask_ffbh_pending,
            b.ucmp(Op_IEqual, mask_ffbh_dst,
                   b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected = b.land(
            mask_ffbh_pending,
            b.ucmp(Op_IEqual, mask_ffbh_dst,
                   b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    if (has_portable_readlane) {
    // Every invocation publishes its ordinary VGPR value unconditionally; V_READLANE ignores EXEC.
    // A pending wave then reads the selected lane from its own guest-wave slice. This common region
    // is reached uniformly by every dispatcher invocation, so the workgroup barriers are exact even
    // when the host subgroup is narrower than the guest Wave64.
    const uint32_t readlane_pending = b.load_function(b.t_bool, readlane_pending_var);
    const uint32_t readlane_source = b.load_function(b.t_u32, readlane_source_var);
    b.cfg_scratch_store(b.linear_localid, readlane_source);
    b.barrier();

    const uint32_t readlane_shift = b.uconst(b.wave_size == 32 ? 5u : 6u);
    const uint32_t readlane_wave_base = b.ibin(
        Op_ShiftLeftLogical,
        b.ibin(Op_ShiftRightLogical, b.linear_localid, readlane_shift),
        readlane_shift);
    const uint32_t readlane_lane = b.ibin(
        Op_BitwiseAnd, b.load_function(b.t_u32, readlane_selector_var),
        b.uconst(b.wave_size - 1u));
    const uint32_t readlane_index = b.ibin(
        Op_IAdd, readlane_wave_base, readlane_lane);
    const uint32_t readlane_valid = b.ucmp(
        Op_ULessThan, readlane_index, b.uconst(b.local_count));
    const uint32_t readlane_result = b.sel(
        readlane_valid,
        b.cfg_scratch_load(b.sel(readlane_valid, readlane_index, zero)), zero);
    const uint32_t readlane_dst = b.load_function(b.t_u32, readlane_dst_var);
    for (int reg : portable_readlane_dsts) {
        const auto destination = sv.find(reg);
        if (destination == sv.end())
            return reject_cfg(0, "missing-portable-readlane-dst");
        const uint32_t selected = b.land(
            readlane_pending,
            b.ucmp(Op_IEqual, readlane_dst,
                   b.uconst(static_cast<uint32_t>(reg))));
        const uint32_t old = b.load_function(b.t_u32, destination->second);
        b.store_function(destination->second, b.sel(selected, readlane_result, old));
    }
    for (const auto& kv : mv) {
        if (!portable_readlane_dsts.contains(kv.first)) continue;
        const uint32_t selected = b.land(
            readlane_pending,
            b.ucmp(Op_IEqual, readlane_dst,
                   b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    for (const auto& kv : mhv) {
        if (!portable_readlane_dsts.contains(kv.first)) continue;
        const uint32_t selected = b.land(
            readlane_pending,
            b.ucmp(Op_IEqual, readlane_dst,
                   b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    if (portable_readlane_dsts.contains(106)) {
        const uint32_t selected = b.land(
            readlane_pending,
            b.ucmp(Op_IEqual, readlane_dst, b.uconst(106u)));
        const uint32_t bit = b.ucmp(
            Op_INotEqual,
            b.ibin(Op_BitwiseAnd, readlane_result, b.uconst(1u)), zero);
        const uint32_t old = b.load_function(b.t_bool, vcc_var);
        b.store_function(vcc_var, b.bsel(selected, bit, old));
    }
    b.barrier();
    }

    if (!mbcnt_event_for_pc.empty()) {
    const uint32_t mbcnt_pending = b.load_function(b.t_bool, mbcnt_pending_var);
    const uint32_t mbcnt_mask = b.load_function(b.t_bool, mbcnt_mask_var);
    const uint32_t mbcnt_tag = b.ibin(
        Op_IAdd, b.load_function(b.t_u32, mbcnt_event_var), b.uconst(1));
    const uint32_t mbcnt_encoded = b.sel(
        mbcnt_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, mbcnt_tag, b.uconst(1)),
               b.sel(mbcnt_mask, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(b.linear_localid, mbcnt_encoded);
    b.barrier();

    const uint32_t mbcnt_low = b.load_function(b.t_bool, mbcnt_low_var);
    b.store_function(mbcnt_sum_var, zero);
    const uint32_t mbcnt_scan = b.id(), mbcnt_scanned = b.id();
    b.emit_selmerge(mbcnt_scanned);
    b.emit_condbranch(mbcnt_pending, mbcnt_scan, mbcnt_scanned);
    b.emit_label(mbcnt_scan);
    uint32_t mbcnt_sum = zero;
    const uint32_t mbcnt_wave_base = b.ibin(
        Op_ShiftLeftLogical, mbcnt_wave_index,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    const uint32_t half_size = std::min(32u, b.wave_size);
    for (uint32_t i = 0; i < half_size; ++i) {
        const uint32_t candidate_lane = b.sel(
            mbcnt_low, b.uconst(i), b.uconst(i + (b.wave_size == 64 ? 32u : 0u)));
        const uint32_t candidate = b.cfg_scratch_load(
            b.ibin(Op_IAdd, mbcnt_wave_base, candidate_lane));
        const uint32_t candidate_tag = b.ibin(
            Op_ShiftRightLogical, candidate, b.uconst(1));
        const uint32_t candidate_bit = b.ibin(Op_BitwiseAnd, candidate, b.uconst(1));
        const uint32_t below = b.ucmp(
            Op_ULessThan, candidate_lane, mbcnt_lane);
        uint32_t include = below;
        if (b.wave_size == 32) include = b.land(include, mbcnt_low);
        include = b.land(include, b.ucmp(Op_IEqual, candidate_tag, mbcnt_tag));
        mbcnt_sum = b.b_iadd(mbcnt_sum, b.sel(include, candidate_bit, zero));
    }
    b.store_function(mbcnt_sum_var, mbcnt_sum);
    b.emit_branch(mbcnt_scanned);
    b.emit_label(mbcnt_scanned);
    const uint32_t mbcnt_result = b.b_iadd(
        b.load_function(b.t_u32, mbcnt_acc_var),
        b.load_function(b.t_u32, mbcnt_sum_var));
    const uint32_t mbcnt_dst = b.load_function(b.t_u32, mbcnt_dst_var);
    const uint32_t mbcnt_write = b.land(
        mbcnt_pending, b.load_function(b.t_bool, mbcnt_write_var));
    for (const auto& kv : vv) {
        const uint32_t selected = b.land(
            mbcnt_write, b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, mbcnt_result, old));
    }
    // An ordinary VGPR write ends any scalar-spill lifetime even for lanes masked off by EXEC.
    for (const auto& kv : lv) {
        const uint32_t selected = b.land(
            mbcnt_pending,
            b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected = b.land(
            mbcnt_pending,
            b.ucmp(Op_IEqual, mbcnt_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    if (!append_event_for_pc.empty()) {
    // DS_APPEND/DS_CONSUME common phase. Each wave reduces popcount(EXEC), its lane zero performs
    // exactly one LDS atomic add/subtract, and the pre-op value is broadcast through the result slot.
    // The event tag prevents a wave at one static append from counting another event's lanes.
    const uint32_t append_pending = b.load_function(b.t_bool, append_pending_var);
    const uint32_t append_active = b.load_function(b.t_bool, append_active_var);
    const uint32_t append_tag = b.ibin(
        Op_IAdd, b.load_function(b.t_u32, append_event_var), b.uconst(1));
    const uint32_t append_encoded = b.sel(
        append_pending,
        b.ibin(Op_BitwiseOr,
               b.ibin(Op_ShiftLeftLogical, append_tag, b.uconst(1)),
               b.sel(append_active, b.uconst(1), zero)),
        zero);
    b.cfg_scratch_store(b.linear_localid, append_encoded);
    b.barrier();

    b.store_function(append_count_var, zero);
    const uint32_t append_scan = b.id(), append_scanned = b.id();
    b.emit_selmerge(append_scanned);
    b.emit_condbranch(append_pending, append_scan, append_scanned);
    b.emit_label(append_scan);
    uint32_t append_count = zero;
    const uint32_t append_wave_base = b.ibin(
        Op_ShiftLeftLogical, mbcnt_wave_index,
        b.uconst(b.wave_size == 32 ? 5u : 6u));
    for (uint32_t i = 0; i < b.wave_size; ++i) {
        const uint32_t candidate_index = b.ibin(
            Op_IAdd, append_wave_base, b.uconst(i));
        const uint32_t candidate = b.cfg_scratch_load(candidate_index);
        const uint32_t candidate_tag = b.ibin(
            Op_ShiftRightLogical, candidate, b.uconst(1));
        uint32_t include = b.ucmp(
            Op_ULessThan, candidate_index, b.uconst(b.local_count));
        include = b.land(include, b.ucmp(Op_IEqual, candidate_tag, append_tag));
        append_count = b.b_iadd(
            append_count,
            b.sel(include, b.ibin(Op_BitwiseAnd, candidate, b.uconst(1)), zero));
    }
    b.store_function(append_count_var, append_count);
    b.emit_branch(append_scanned);
    b.emit_label(append_scanned);
    const uint32_t append_is_leader = b.land(
        append_pending, b.ucmp(Op_IEqual, mbcnt_lane, zero));
    const uint32_t append_leader = b.id(), append_reduced = b.id();
    b.emit_selmerge(append_reduced);
    b.emit_condbranch(append_is_leader, append_leader, append_reduced);
    b.emit_label(append_leader);
    const uint32_t append_count_value = b.load_function(b.t_u32, append_count_var);
    const uint32_t append_delta = b.sel(
        b.load_function(b.t_bool, append_consume_var),
        b.ibin(Op_ISub, zero, append_count_value), append_count_value);
    const uint32_t append_index = b.load_function(b.t_u32, append_idx_var);
    uint32_t append_old = 0;
    if (has_gds_append && has_lds_append) {
        const uint32_t gds_block = b.id(), lds_block = b.id(), memory_merge = b.id();
        b.emit_selmerge(memory_merge);
        b.emit_condbranch(b.load_function(b.t_bool, append_gds_var),
                          gds_block, lds_block);
        b.emit_label(gds_block);
        const uint32_t gds_old = b.compute_gds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta);
        const uint32_t gds_end = b.cur_block;
        b.emit_branch(memory_merge);
        b.emit_label(lds_block);
        const uint32_t lds_old = b.lds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta, false, yes, zero);
        const uint32_t lds_end = b.cur_block;
        b.emit_branch(memory_merge);
        b.emit_label(memory_merge);
        append_old = b.emit_phi_2way(
            b.t_u32, gds_old, gds_end, lds_old, lds_end);
    } else if (has_gds_append) {
        append_old = b.compute_gds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta);
    } else {
        append_old = b.lds_atomic_rtn(
            Op_AtomicIAdd, append_index, append_delta, false, yes, zero);
    }
    b.cfg_scratch_store(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index), append_old);
    b.emit_branch(append_reduced);
    b.emit_label(append_reduced);
    b.barrier();

    const uint32_t append_result = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), mbcnt_wave_index));
    const uint32_t append_dst = b.load_function(b.t_u32, append_dst_var);
    const uint32_t append_write = b.land(append_pending, append_active);
    for (const auto& kv : vv) {
        const uint32_t selected = b.land(
            append_write,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, append_result, old));
    }
    for (const auto& kv : lv) {
        const uint32_t selected = b.land(
            append_pending,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_u32, kv.second);
        b.store_function(kv.second, b.sel(selected, zero, old));
    }
    for (const auto& kv : lmv) {
        const uint32_t selected = b.land(
            append_pending,
            b.ucmp(Op_IEqual, append_dst, b.uconst(static_cast<uint32_t>(kv.first.first))));
        const uint32_t old = b.load_function(b.t_bool, kv.second);
        b.store_function(kv.second, b.bsel(selected, no, old));
    }
    b.barrier();
    }

    // Publish this lane's pending vote bit and liveness. The switch merge is reached by every
    // invocation on every iteration, including lanes whose emulated wave has already ended.
    const uint32_t pending = b.load_function(b.t_bool, vote_pending_var);
    const uint32_t vote_value = b.load_function(b.t_bool, vote_value_var);
    const uint32_t vote_bit = b.sel(b.land(pending, vote_value), b.uconst(1), zero);
    const uint32_t active_bit = b.sel(b.load_function(b.t_bool, active_var), b.uconst(2), zero);
    b.cfg_scratch_store(b.linear_localid, b.ibin(Op_BitwiseOr, vote_bit, active_bit));
    b.barrier();

    const uint32_t wave_shift = b.wave_size == 32 ? 5u : 6u;
    const uint32_t wave_index = b.ibin(Op_ShiftRightLogical, b.linear_localid,
                                       b.uconst(wave_shift));
    const uint32_t wave_base = b.ibin(Op_ShiftLeftLogical, wave_index,
                                      b.uconst(wave_shift));
    const uint32_t lane_in_wave = b.ibin(Op_BitwiseAnd, b.linear_localid,
                                         b.uconst(b.wave_size - 1));

    // One lane per hardware wave reduces that wave's vote. Padding keeps every dynamic access in
    // bounds for a partial final wave; padded values are explicitly masked out.
    uint32_t wave_leader = b.id(), wave_reduced = b.id();
    const uint32_t is_wave_leader = b.ucmp(Op_IEqual, lane_in_wave, zero);
    b.emit_selmerge(wave_reduced);
    b.emit_condbranch(is_wave_leader, wave_leader, wave_reduced);
    b.emit_label(wave_leader);
    uint32_t wave_flags = zero;
    for (uint32_t lane = 0; lane < b.wave_size; ++lane) {
        const uint32_t idx = b.ibin(Op_IAdd, wave_base, b.uconst(lane));
        uint32_t flags = b.cfg_scratch_load(idx);
        if (padded_lanes != b.local_count)
            flags = b.sel(b.ucmp(Op_ULessThan, idx, b.uconst(b.local_count)), flags, zero);
        wave_flags = b.ibin(Op_BitwiseOr, wave_flags,
                            b.ibin(Op_BitwiseAnd, flags, b.uconst(1)));
    }
    b.cfg_scratch_store(b.ibin(Op_IAdd, b.uconst(wave_result_base), wave_index), wave_flags);
    b.emit_branch(wave_reduced);
    b.emit_label(wave_reduced);

    // Lane zero also reduces workgroup liveness. This uniform dispatcher loop is what makes the
    // internal barriers legal even after one hardware wave reaches S_ENDPGM before another.
    uint32_t group_leader = b.id(), group_reduced = b.id();
    const uint32_t is_group_leader = b.ucmp(Op_IEqual, b.linear_localid, zero);
    b.emit_selmerge(group_reduced);
    b.emit_condbranch(is_group_leader, group_leader, group_reduced);
    b.emit_label(group_leader);
    uint32_t group_flags = zero;
    for (uint32_t lane = 0; lane < b.local_count; ++lane) {
        const uint32_t flags = b.cfg_scratch_load(b.uconst(lane));
        group_flags = b.ibin(Op_BitwiseOr, group_flags,
                             b.ibin(Op_BitwiseAnd, flags, b.uconst(2)));
    }
    b.cfg_scratch_store(b.uconst(group_active_slot), group_flags);
    b.emit_branch(group_reduced);
    b.emit_label(group_reduced);
    b.barrier();

    const uint32_t wave_flags_result = b.cfg_scratch_load(
        b.ibin(Op_IAdd, b.uconst(wave_result_base), wave_index));
    const uint32_t wave_any = b.ucmp(
        Op_INotEqual, b.ibin(Op_BitwiseAnd, wave_flags_result, b.uconst(1)), zero);
    const uint32_t vote_condition = b.bsel(
        b.load_function(b.t_bool, vote_invert_var), b.logical_not(wave_any), wave_any);
    const uint32_t selected_pc = b.sel(vote_condition,
        b.load_function(b.t_u32, vote_taken_var), b.load_function(b.t_u32, vote_next_var));
    const uint32_t vote_to_scc = b.load_function(b.t_bool, vote_to_scc_var);
    const uint32_t write_scc = b.land(pending, vote_to_scc);
    b.store_function(scc_var,
        b.bsel(write_scc, vote_condition, b.load_function(b.t_bool, scc_var)));
    const uint32_t vote_to_vcc = b.load_function(b.t_bool, vote_to_vcc_var);
    const uint32_t write_vcc = b.land(pending, vote_to_vcc);
    const uint32_t compared_vcc = b.land(
        b.load_function(b.t_bool, exec_var), vote_condition);
    b.store_function(vcc_var,
        b.bsel(write_vcc, compared_vcc, b.load_function(b.t_bool, vcc_var)));
    if (const auto saved_vcc = mv.find(106); saved_vcc != mv.end())
        b.store_function(saved_vcc->second,
            b.bsel(write_vcc, compared_vcc,
                   b.load_function(b.t_bool, saved_vcc->second)));
    const uint32_t write_pc = b.land(
        pending, b.land(b.logical_not(vote_to_scc), b.logical_not(vote_to_vcc)));
    b.store_function(pc_var, b.sel(write_pc, selected_pc, b.load_function(b.t_u32, pc_var)));
    uint32_t group_active = b.ucmp(
        Op_INotEqual, b.cfg_scratch_load(b.uconst(group_active_slot)), zero);
    group_active = apply_trip_bound(group_active);
    b.emit_condbranch(group_active, loop_header, loop_merge);
    }
    b.emit_label(loop_merge);

    // Expose the final emulated state to the caller. Graphics exports are emitted in their exact
    // cases, while any post-body bookkeeping still sees this invocation's final register values.
    initial = load_state();
    if (const auto terminal = block_for_pc.find(end_pc);
        terminal != block_for_pc.end() && scalar_reachable[terminal->second]) {
        initial.sreg_written = scalar_may_write_in[terminal->second];
        for (int reg : initial.sreg_written) initial.sreg_input.erase(reg);
    }
    initial.invalidated_vgpr_lane_slots =
        std::move(terminal_invalidated_vgpr_lane_slots);
    // sreg_srt and lds_addtid are path-sensitive SSA provenance without dispatcher function
    // variables. Not reconstructing them across a phase is intentional and fail-closed: a later
    // descriptor/LDS consumer rejects instead of reviving provenance from an arbitrary path.
    if (proven_wave32_masks) {
        const auto end_block = block_for_pc.find(end_pc);
        if (end_block != block_for_pc.end() &&
            (!b32_mask_reachable[end_block->second] ||
             !b32_mask_in[end_block->second].contains(106))) {
            initial.vcc = 0;
            initial.sreg_bool.erase(106);
            initial.sreg_bool_narrowed.erase(106);
            initial.sreg_bool_b32.erase(106);
        }
    }
    return true;
}

// Emit the instruction body (shared by every stage). Handles a single recognized COUNTED loop as a real
// structured SPIR-V loop (OpLoopMerge + OpPhi for loop-carried registers); loop-FREE streams walk straight
// through, byte-identical to the pre-loop-feature behavior. `exp_fn` handles an EXP instruction per stage
// (compute: reject; fragment: MRT color; vertex: POS/PARAM). Returns false if any instruction is
// unsupported. allow_exec_update / allow_smem match the stage's emit_alu flags.
bool emit_body(SpirvCompute& b, RegState& rs, const std::vector<Rdna2Inst>& ins,
               const std::unordered_set<uint32_t>& safe, const ShaderResourceTable* rt,
               bool allow_exec_update, bool allow_smem,
               const std::function<bool(RegState&, const Rdna2Inst&)>& exp_fn,
               const uint32_t* code, size_t dwords,
               const std::unordered_set<uint32_t>* inherited_dead_masks,
               bool allow_cfg_dispatcher,
               uint32_t initial_dispatch_active,
               bool force_barrier_phases,
               bool force_lds_fminmax_dispatcher) {
               // code/dwords: raw stream for forward-if target checks; inherited_dead_masks keeps
               // whole-shader liveness valid when a barrier-separated body is compiled in phases.
    rs.max_vgpr = std::max(rs.max_vgpr, shader_max_vgpr(ins));
    if (!b.cselect_b64_low_only_analysis_done) {
        b.cselect_b64_low_only_pcs = proven_cselect_b64_low_only_pcs(ins);
        b.cselect_b64_low_only_analysis_done = true;
    }
    if (!b.vcc_b32_low_only_analysis_done) {
        b.vcc_b32_low_only_pcs = proven_wave64_vcc_b32_low_only_pcs(ins);
        b.vcc_b32_low_only_analysis_done = true;
    }
    if (!b.structured_wave64_mask_reduction_analysis_done) {
        b.structured_wave64_mask_reduction_pcs =
            proven_structured_wave64_mask_reduction_pcs(ins);
        b.structured_wave64_mask_reduction_analysis_done = true;
    }
    if (!rs.smem_x16_descriptor_analysis_done) {
        rs.smem_x16_descriptor_loads = proven_smem_x16_descriptor_loads(ins, rt);
        rs.smem_x16_descriptor_analysis_done = true;
    }
    if (!rs.smem_x2_descriptor_fragment_analysis_done) {
        rs.smem_x2_descriptor_fragment_loads =
            proven_smem_x2_descriptor_fragment_loads(ins, rt, b.wave_size);
        rs.smem_x2_descriptor_fragment_analysis_done = true;
    }
    // Fold PC-relative embedded-table loads (s_getpc_b64-built V#s) before the walk — emit_alu's
    // SMEM/MUBUF/SOP1 handlers consult the proven table maps (#273/#1054).
    if (code) {
        PcrelTables tables = detect_pcrel_tables(ins, code, dwords);
        rs.mubuf_pcrel_tables = std::move(tables.mubuf);
        rs.smem_pcrel_tables = std::move(tables.smem);
        rs.mtbuf_pcrel_tables = std::move(tables.mtbuf);
    }
    std::unordered_set<uint32_t> local_dead_masks;
    if (!inherited_dead_masks) local_dead_masks = dead_wave_mask_writes(ins);
    const std::unordered_set<uint32_t>& dead_masks = inherited_dead_masks
        ? *inherited_dead_masks : local_dead_masks;

    // A large generated compute kernel may put a workgroup-uniform scalar early-out around several
    // barrier-separated phases, then use arbitrary (but barrier-free) control flow in its final
    // phase. The whole-stream CFG dispatcher cannot legally place an OpControlBarrier in one switch
    // case, while the narrow structurizer rejects the final phase's branch graph. Peel the terminal
    // scalar guard and compile each barrier-free phase independently. Every invocation in one
    // workgroup takes the guard together, and no guest edge may cross a split, so each explicit
    // barrier remains uniform while the final phase can use the ordinary CFG dispatcher.
    if (b.is_compute) {
        const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
        if (phased.found &&
            (phased.guarded || initial_dispatch_active || force_barrier_phases)) {
            // Every phase shares one immutable Workgroup OpTypeArray. Size it from the complete
            // phased stream before the first dispatcher: a later portable DPP operation needs a
            // second per-lane plane even when the earlier phase needed only votes/liveness.
            if (!b.native_subgroup_size) {
                const uint32_t wave_count =
                    (b.local_count + b.wave_size - 1) / b.wave_size;
                const uint32_t padded_lanes = wave_count * b.wave_size;
                const bool has_portable_dpp = std::any_of(
                    ins.begin(), ins.begin() + phased.end_index,
                    [](const Rdna2Inst& in) {
                        return is_inplace_vadd_nc_u32_dpp_row_shr(in) ||
                            dpp_row_ror8_op(in) != DppRowRor8Op::None;
                    });
                const uint32_t scratch_dwords = padded_lanes +
                    (has_portable_dpp ? padded_lanes : 0u) + wave_count + 1;
                if (!b.declare_cfg_scratch(scratch_dwords)) return false;
            }
            uint32_t merge_label = 0;
            if (phased.guarded) {
                // Padded invocations cannot follow a divergent extent guard around this selection.
                // Keep that uncommon combination on the existing fail-closed path; the unguarded
                // phase form below is the one proved safe for partial workgroups.
                if (initial_dispatch_active) return false;
                const std::vector<Rdna2Inst> prefix(
                    ins.begin(), ins.begin() + phased.guard_index);
                if (!prefix.empty() &&
                    !emit_body(b, rs, prefix, safe, rt, allow_exec_update, allow_smem,
                               exp_fn, code, dwords, &dead_masks))
                    return false;
                if (!rs.scc) return false;
                const uint32_t execute_body = ins[phased.guard_index].opcode == 0x04
                    ? rs.scc : b.logical_not(rs.scc);
                const uint32_t body_label = b.id();
                merge_label = b.id();
                b.emit_selmerge(merge_label);
                b.emit_condbranch(execute_body, body_label, merge_label);
                b.emit_label(body_label);
            }

            auto emit_phase = [&](std::vector<Rdna2Inst>& phase) {
                // Unguarded phases are admitted specifically because the whole-stream exact-wave
                // dispatcher was blocked by a guest barrier. Use that dispatcher directly for each
                // now barrier-free region; re-entering the narrow structurizer could reject the same
                // backward-else/complex shape before it ever reaches the fallback.
                if (!phased.guarded || initial_dispatch_active)
                    return emit_cfg_state_machine(
                        b, rs, phase, safe, rt, allow_exec_update, allow_smem,
                        exp_fn, code, dwords, initial_dispatch_active,
                        force_lds_fminmax_dispatcher);
                return emit_body(b, rs, phase, safe, rt, allow_exec_update, allow_smem,
                                 exp_fn, code, dwords, &dead_masks, true, 0, false,
                                 force_lds_fminmax_dispatcher);
            };

            size_t phase_begin = phased.guarded ? phased.guard_index + 1 : 0;
            for (size_t barrier_index : phased.barriers) {
                std::vector<Rdna2Inst> phase(
                    ins.begin() + phase_begin, ins.begin() + barrier_index);
                if (!phase.empty()) {
                    // A phase is a complete control-flow region even though the guest's real
                    // S_ENDPGM follows the final phase. The arbitrary-CFG fallback requires an end
                    // block so its persistent dispatcher can become inactive and rejoin this outer
                    // barrier sequence. Give each proven split a synthetic, emitter-only terminator
                    // at the boundary; no branch crosses the boundary (proved above), and the raw
                    // barrier remains emitted exactly once by this outer shell.
                    Rdna2Inst phase_end;
                    phase_end.pc = ins[barrier_index].pc;
                    phase_end.fmt = Rdna2Format::SOPP;
                    phase_end.opcode = 0x01u;
                    phase_end.len_dwords = 1;
                    phase_end.is_end = true;
                    phase_end.synthetic_terminator = true;
                    phase.push_back(phase_end);
                    if (getenv("PROSPER_DBG"))
                        std::fprintf(stderr, "[compute-phase] begin=%u end=%u barrier=%u\n",
                                     phase.front().pc, phase[phase.size() - 2].pc,
                                     ins[barrier_index].pc);
                    if (!emit_phase(phase)) {
                        log_recompile_diagnostic(
                            b.diagnostic, "compute-phase-reject", "consequent",
                            "begin=%u end=%u", phase.front().pc, phase[phase.size() - 2].pc);
                        return false;
                    }
                }
                b.barrier();
                phase_begin = barrier_index + 1;
            }
            std::vector<Rdna2Inst> tail(ins.begin() + phase_begin, ins.end());
            if (!tail.empty()) {
                if (getenv("PROSPER_DBG"))
                    std::fprintf(stderr, "[compute-phase] begin=%u end=%u tail=1\n",
                                 tail.front().pc, tail.back().pc);
                if (!emit_phase(tail)) {
                    log_recompile_diagnostic(b.diagnostic, "compute-phase-reject", "consequent",
                                             "begin=%u end=%u tail=1",
                                             tail.front().pc, tail.back().pc);
                    return false;
                }
            }
            if (phased.guarded) {
                b.emit_branch(merge_label);
                b.emit_label(merge_label);
                // This join deliberately merges NOTHING -- no phis, and no meet for sreg_srt,
                // sreg_bool, sreg_written or sreg_ud_alias. That is safe only because the peeled
                // guard is TERMINAL: `tail` runs to ins.end(), so no guest instruction follows this
                // label and nothing can consume a per-path fact that escaped it. If
                // analyze_barrier_phased_compute is ever relaxed to peel a NON-terminal guard, every
                // one of those domains becomes a silent hole here at once, and each needs the
                // treatment the ordinary if-only region gets (#1773 added merge_ud_alias there).
            }
            if (initial_dispatch_active)
                b.partial_barrier_phases_emitted = true;
            return true;
        }
    }
    if (force_lds_fminmax_dispatcher)
        return emit_cfg_state_machine(
            b, rs, ins, safe, rt, allow_exec_update, allow_smem,
            exp_fn, code, dwords, initial_dispatch_active, true);
    std::unordered_set<uint32_t> effective_safe = safe;
    const CountedLoop L = detect_counted_loop(ins);
    size_t idx = 0;
    // Cross-lane wave ops (mbcnt) emit LDS + barriers, which are only valid at wave-uniform points — so
    // they're allowed ONLY in the straight-line path (no divergent loop/if around them). Set true below.
    bool wave_ok = false;
    auto emit_range = [&](uint32_t pc_lo, uint32_t pc_hi) -> bool {   // emit ins whose pc ∈ [pc_lo, pc_hi)
        for (; idx < ins.size(); ++idx) {
            const Rdna2Inst& in = ins[idx];
            if (in.is_end || in.pc >= pc_hi) break;
            if (in.pc < pc_lo) continue;
            if (dead_masks.count(in.pc)) continue;
            if (in.fmt == Rdna2Format::EXP) { if (!exp_fn(rs, in)) return false; continue; }
            bool ok = true;
            const SavedB64MaskSnapshot saved_masks = snapshot_saved_b64_masks(rs, in);
            const bool handled = emit_alu(
                b, rs, in, ok, allow_exec_update, &effective_safe, allow_smem, rt, wave_ok);
            if (handled && ok)
                record_scalar_write(
                    rs, in,
                    allows_compute_scalar_vcc_bridge(b), saved_masks);
            // Shader I/O tap: snapshot this instruction's destination VGPR (+3) if it is the tapped PC.
            if (handled && ok && in.pc == b.tap_pc && in.dst.kind == OperandKind::VGPR) {
                auto tv = [&](int r) { auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
                b.set_tap(tv(in.dst.value), tv(in.dst.value + 1), tv(in.dst.value + 2), tv(in.dst.value + 3));
            }
            if (!handled || !ok) {
                // PROSPER_DBG (gated, off by default): report the instruction that fails recompilation —
                // the first unsupported op / unresolved resource that makes a shader return empty.
                // NOT gated on PROSPER_DBG, for the reason at the CFG reject site above: the gate
                // suppressed the RECORDING as well as the printing, and the recording is what the
                // unconditional skip line reads. This site fires once per failing compile, so the
                // formatting cost it now always pays is one string per rejected shader.
                {
                    // `mode` separates the two rejections that used to print identically and want
                    // OPPOSITE work (#2412). `unknown-encoding` (handled=false) means no lowering
                    // exists — write the emitter. `unresolved-operand` (handled=true, ok=false)
                    // means the lowering exists and could not resolve an operand or a V#/T#/S#
                    // through the resource table — the emitter is fine and the descriptor is the
                    // defect. GTA V's black 3D world is the worked example: its top three rejected
                    // instructions are buffer_store_dword / buffer_load_dwordx2 / buffer_load_dword,
                    // all lowered at :9343-9362, so a census without this field reads as "implement
                    // MUBUF" when nothing about MUBUF is missing.
                    // Shader identity (#2412): the reject lines carry a program-local pc and nothing
                    // else, so a census cannot group them by SHADER -- which is the unit that matters,
                    // since 24,485 skipped GTA V draws turned out to be 43 distinct shaders. The first
                    // code dword plus the span identifies one cheaply and stably.
                    // A third `mode`. The emitter sets `stage_reject_*` when it DECODED the
                    // instruction, resolved every operand, and still cannot lower it in THIS stage
                    // -- a cross-lane operation in a shell with no guest wave. Neither existing
                    // value describes that: `unknown-encoding` sends the reader to write an
                    // emitter that already exists, and `unresolved-operand` sends them to the
                    // resource table, which is not involved. The reason itself is appended so the
                    // ONE terminal line records the cause; a second line would be recorded after
                    // this one and overwrite it (#3135).
                    const bool stage_unsupported =
                        b.stage_reject_pc == in.pc && !b.stage_reject_reason.empty();
                    log_recompile_diagnostic(
                        b.diagnostic, "recompile-reject", "terminal",
                        "sh=%08x/%zu mode=%s%s%s pc=%u words=%s fmt=%d op=0x%x "
                        "dst=%d(kind%d) src=%d(k%d),%d(k%d),%d(k%d) dmask=0x%x "
                        "dim=%u glc=%d len=%u modifier=%d dpp=%d sdwa=%u/%u/%u/%u "
                        "sext=%d/%d",
                        dwords ? code[0] : 0u, dwords,
                        stage_unsupported
                            ? "unsupported-in-stage"
                            : (handled ? "unresolved-operand" : "unknown-encoding"),
                        stage_unsupported ? " reason=" : "",
                        stage_unsupported ? b.stage_reject_reason.c_str() : "",
                        in.pc, reject_words_text(in).c_str(), (int)in.fmt, in.opcode,
                        in.dst.value, (int)in.dst.kind,
                        in.src[0].value, (int)in.src[0].kind,
                        in.src[1].value, (int)in.src[1].kind,
                        in.src[2].value, (int)in.src[2].kind,
                        in.mimg_dmask, in.mimg_dim, (int)in.mimg_glc, in.len_dwords, (int)in.has_modifier,
                        (int)in.has_dpp, in.sdwa_dst_sel, in.sdwa_dst_unused,
                        in.sdwa_src0_sel, in.sdwa_src1_sel,
                        // The four selects alone cannot distinguish the zero- from the sign-extending
                        // form of one encoding, and they are different operations (#2013).
                        (int)in.sdwa_src0_sext, (int)in.sdwa_src1_sext);
                    // The primary line historically printed only the fixed MIMG pair. That hid the
                    // address VGPRs which distinguished Asterix's rejected NSA form from the accepted
                    // consecutive-vaddr form. Keep a separate MIMG-only line so an address-shape
                    // diagnosis sees every decoded extra dword without adding zeros to unrelated ops.
                    if (in.fmt == Rdna2Format::MIMG && in.len_dwords > 2u)
                        log_recompile_diagnostic(
                            b.diagnostic, "recompile-reject-mimg-address", "terminal",
                            "pc=%u extra=%u words=%08x,%08x,%08x",
                            in.pc, in.len_dwords - 2u,
                            in.words[2], in.words[3], in.words[4]);
                }
                return false;
            }
        }
        return true;
    };
    auto& safe_branches = effective_safe;
    if (L.found) {
        auto vget = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
        auto sget = [&](int r){ auto it = rs.sreg.find(r); return it == rs.sreg.end() ? b.uconst(0) : it->second; };
        bool guarded_narrow_entry = false;
        // saveexec -> execz -> matching EXEC restore around a side-effect-free counted region is a
        // whole-wave empty-work optimization. In the per-invocation shell we may run the uniform
        // scalar loop for every invocation while narrowed EXEC predicates vector writes; inactive
        // lanes retain their old VGPRs until the exact restore. Reject stores/exports/barriers and
        // unclassified memory so this never becomes a general branch-linearization escape hatch.
        // Scan inside-out so an already-proven nested guard may contribute its balanced save/restore
        // pair without making an otherwise-safe outer guarded loop look like it leaks narrowed EXEC.
        struct GuardedExecRegion { uint32_t save_pc, restore_pc; };
        std::vector<GuardedExecRegion> guarded_exec_regions;
        for (size_t branch_index = ins.size(); branch_index-- > 0;) {
            const Rdna2Inst& branch = ins[branch_index];
            if (branch.fmt != Rdna2Format::SOPP || branch.opcode != 0x08 || branch.simm16 <= 0)
                continue;
            size_t previous = branch_index;
            while (previous > 0) {
                --previous;
                if (!sopp_is_noop(ins[previous])) break;
            }
            if (previous >= branch_index) continue;
            const Rdna2Inst& saveexec = ins[previous];
            if (saveexec.fmt != Rdna2Format::SOP1 ||
                (saveexec.opcode != 0x24 && saveexec.opcode != 0x25) ||
                saveexec.dst.kind != OperandKind::SGPR || saveexec.dst.value > 104) continue;
            const uint32_t target = branch_target(branch);
            const Rdna2Inst* restore = nullptr;
            for (const auto& candidate : ins) if (candidate.pc == target) { restore = &candidate; break; }
            if (!restore || restore->fmt != Rdna2Format::SOP1 || restore->opcode != 0x04 ||
                restore->dst.value < 126 || !reg_operand(restore->src[0], saveexec.dst.value)) continue;
            // A lexical save/restore pair is not necessarily balanced along the counted-loop CFG.
            // In particular, a save in the body with its restore after the backedge leaves EXEC
            // narrowed between iterations (EXEC has no loop phi), and a zero-trip path reaches an
            // undominated restore. Accept only a pair contained in one straight-line loop segment,
            // or a true preheader-to-postloop wrapper around the complete loop.
            const bool same_preloop = saveexec.pc < L.header_pc && target < L.header_pc;
            const bool same_condition = saveexec.pc >= L.header_pc && target < L.exit_branch_pc;
            const bool same_body = saveexec.pc > L.exit_branch_pc && target < L.backedge_pc;
            const bool same_postloop = saveexec.pc >= L.exit_pc;
            const bool wraps_loop = saveexec.pc < L.header_pc && target >= L.exit_pc;
            if (!same_preloop && !same_condition && !same_body && !same_postloop && !wraps_loop)
                continue;
            bool side_effect_free = true;
            for (const auto& candidate : ins) {
                if (candidate.pc <= branch.pc || candidate.pc >= target) continue;
                bool clobbers_guard_mask = false;
                for_each_scalar_write(candidate, [&](int base, uint32_t width) {
                    clobbers_guard_mask |= base < saveexec.dst.value + 2 &&
                        saveexec.dst.value < base + static_cast<int>(width);
                });
                bool balanced_nested_exec = false;
                for (const auto& nested : guarded_exec_regions) {
                    if (nested.save_pc > branch.pc && nested.restore_pc < target &&
                        (candidate.pc == nested.save_pc || candidate.pc == nested.restore_pc)) {
                        balanced_nested_exec = true;
                        break;
                    }
                }
                if (candidate.fmt == Rdna2Format::EXP || candidate.fmt == Rdna2Format::DS ||
                    candidate.fmt == Rdna2Format::MUBUF || candidate.fmt == Rdna2Format::MTBUF ||
                    candidate.fmt == Rdna2Format::MIMG || candidate.fmt == Rdna2Format::FLAT ||
                    (rdna2_instruction_may_change_exec(candidate) && !balanced_nested_exec) ||
                    clobbers_guard_mask ||
                    (candidate.fmt == Rdna2Format::SOPP && candidate.opcode == 0x0a)) {
                    side_effect_free = false;
                    break;
                }
            }
            if (!side_effect_free) continue;
            effective_safe.insert(branch.pc);
            guarded_exec_regions.push_back({saveexec.pc, target});
            if (branch.pc < L.header_pc && target >= L.exit_pc) guarded_narrow_entry = true;
        }
        // 1. Pre-loop body. A compiler may place one ordinary uniform if/else before the canonical
        // counted loop (Evergate selects one of two constant blocks this way; Astro's NGG culling
        // prelude also has a one-arm conditional). Structure that choice with the same two-arm PHIs
        // as the general forward-if path, then enter the existing counted-loop lowering. Anything
        // nested/more complex stays unsupported and rejects visibly.
        std::vector<Rdna2Inst> preloop;
        for (const auto& in : ins) {
            if (in.pc >= L.header_pc) break;
            // A forward execz spanning the counted loop is a redundant wave-empty guard when the
            // prelude reaches it with full EXEC. Leave it in the emitted stream (emit_alu proves
            // full EXEC and otherwise rejects), but do not ask the pre-loop uniform-if detector to
            // model a region whose body deliberately extends beyond its artificial end marker.
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x08 &&
                branch_target(in) >= L.header_pc) continue;
            preloop.push_back(in);
        }
        Rdna2Inst preloop_end;
        preloop_end.pc = L.header_pc;
        preloop_end.is_end = true;
        preloop.push_back(preloop_end);
        bool preloop_rejected = false;
        const std::vector<ForwardIf> preloop_ifs = detect_forward_ifs(
            preloop, /*allow_vcc*/!b.is_compute, code, dwords, &effective_safe, nullptr,
            &preloop_rejected, /*compute_wave_branches*/b.is_compute, b.diagnostic);
        // detect_forward_ifs clamps a branch to an immediate s_endpgm at its artificial end marker
        // and records it as early_out. In this truncated prelude that can be a real branch over the
        // entire counted loop, so it cannot be structured as an ordinary one-arm conditional.
        const bool preloop_if_unsupported = std::any_of(
            preloop_ifs.begin(), preloop_ifs.end(), [&](const ForwardIf& branch) {
                return branch.early_out ||
                    (branch.has_else ? branch.merge_pc : branch.target_pc) > L.header_pc;
            });
        if (preloop_rejected || preloop_if_unsupported) {
            log_recompile_diagnostic(
                b.diagnostic, "recompile-reject", "terminal",
                "counted-loop prelude cfg rejected=%u ifs=%zu header=%u",
                preloop_rejected, preloop_ifs.size(), L.header_pc);
            return false;
        }
        if (preloop_ifs.empty()) {
            if (!emit_range(0, L.header_pc)) return false;
        } else if (preloop_ifs.size() > 1) {
            // The general structurizer already owns nested and disjoint ForwardIf trees, including
            // exact guest-wave EXEC/VCC tests in compute. Reuse it for a multi-choice prefix instead
            // of duplicating another recursive IF emitter inside the counted-loop path. Keep every
            // real prefix instruction here (including a proven-safe guard spanning the whole loop):
            // the truncated scan above intentionally omitted that guard only so it was not mistaken
            // for an early-out at the artificial end marker. effective_safe still makes emit_alu
            // linearize a proven spanning guard, and inherited dead masks retain whole-shader liveness.
            std::vector<Rdna2Inst> preloop_body;
            for (const auto& in : ins) {
                if (in.pc >= L.header_pc) break;
                preloop_body.push_back(in);
            }
            preloop_body.push_back(preloop_end);
            if (!emit_body(b, rs, preloop_body, effective_safe, rt, allow_exec_update,
                           allow_smem, exp_fn, code, dwords, &dead_masks,
                           /*allow_cfg_dispatcher*/false))
                return false;
            while (idx < ins.size() && ins[idx].pc < L.header_pc) ++idx;
            if (idx >= ins.size() || ins[idx].pc != L.header_pc) return false;
        } else {
            const ForwardIf F = preloop_ifs[0];
            if (!emit_range(0, F.branch_pc)) return false;
            if (idx >= ins.size() || ins[idx].pc != F.branch_pc) return false;
            ++idx; // consume the conditional branch
            // The structured PHIs below use zero for an SGPR absent on one predecessor. From this
            // point onward map presence alone is no longer a scalar-lifetime MUST fact.
            rs.scalar_presence_has_no_placeholders = false;
            // An scc-conditioned forward-if with a POISONED SCC (rs.scc == 0: a 64-bit mask op was
            // the last architectural SCC writer, unrepresentable per-lane) must reject — this is
            // exactly the non-adjacent stale-SCC consumer from the ISA audit (#879).
            if (!F.on_exec && !F.on_vcc && !rs.scc) return false;
            if (F.on_vcc && !rs.vcc) return false;
            uint32_t condition = F.on_exec ? rs.exec : (F.on_vcc ? rs.vcc : rs.scc);
            if (b.is_fragment && (F.on_exec || F.on_vcc) &&
                !(F.on_vcc && rs.vcc == rs.vcc_wave_uniform &&
                  vcc_exit_is_wave_uniform(ins, F.branch_pc)))
                condition = b.fragment_wave_any(condition);
            condition = F.on_scc0 ? condition : b.logical_not(condition);
            const RegState before = rs;
            std::set<int> written_v, written_s;
            const uint32_t then_end = F.has_else ? F.sb_pc : F.target_pc;
            const uint32_t merge_pc = F.has_else ? F.merge_pc : F.target_pc;
            loop_written_regs(ins, F.branch_pc + 1, then_end, written_v, written_s);
            if (F.has_else)
                loop_written_regs(ins, F.target_pc, merge_pc, written_v, written_s);
            const uint32_t then_label = b.id(), else_label = b.id(), merge_label = b.id();
            b.emit_selmerge(merge_label);
            b.emit_condbranch(condition, then_label, else_label);

            b.emit_label(then_label);
            if (!emit_range(F.branch_pc + 1, then_end)) return false;
            if (F.has_else) {
                if (idx >= ins.size() || ins[idx].pc != F.sb_pc) return false;
                ++idx; // consume the then arm's jump to the merge
            }
            const uint32_t then_block = b.cur_block;
            std::unordered_map<int, uint32_t> then_v, then_s;
            for (int reg : written_v) then_v[reg] = vget(reg);
            for (int reg : written_s) then_s[reg] = sget(reg);
            // #3133: which of those scalars is the THEN edge holding as an entry-M0 token? `sget`
            // rendered each of them as uconst(0) above; the meet below decides whether that value
            // may be phi'd at all.
            std::set<int> then_entry_m0;
            for (int reg : written_s)
                if (entry_m0_live(rs, reg)) then_entry_m0.insert(reg);
            const uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
            const bool then_narrowed = rs.exec_narrowed;
            const auto then_bool = rs.sreg_bool;
            const auto then_bool_b32 = rs.sreg_bool_b32;
            const auto then_written = rs.sreg_written;
            const auto then_ud_alias = rs.sreg_ud_alias;   // the then edge's copy-alias claims
            b.emit_branch(merge_label);

            rs = before;
            b.emit_label(else_label);
            if (F.has_else && !emit_range(F.target_pc, merge_pc)) return false;
            const uint32_t else_block = b.cur_block;
            b.emit_branch(merge_label);
            b.emit_label(merge_label);
            for (int reg : written_v) {
                const uint32_t else_value = vget(reg);
                if (then_v[reg] != else_value)
                    rs.vreg[reg] = b.emit_phi_2way(
                        b.t_u32, then_v[reg], then_block, else_value, else_block);
            }
            for (int reg : written_s) {
                const uint32_t else_value = sget(reg);
                // `rs` is the else edge here; skip the phi entirely when the token decides (#3133).
                if (!join_entry_m0(rs, reg, then_entry_m0.count(reg) != 0)) continue;
                if (then_s[reg] != else_value)
                    rs.sreg[reg] = b.emit_phi_2way(
                        b.t_u32, then_s[reg], then_block, else_value, else_block);
            }
            if (then_scc != rs.scc)
                // A poisoned (0) input degrades to bfalse across the merge — no invalid SPIR-V and
                // no stricter rejection than the pre-poison behavior; straight-line consumers of a
                // poisoned SCC are still rejected at their own sites.
                rs.scc = b.emit_phi_2way(b.t_bool, then_scc ? then_scc : b.bfalse(), then_block,
                                         rs.scc ? rs.scc : b.bfalse(), else_block);
            if (then_vcc != rs.vcc)
                rs.vcc = !then_vcc || !rs.vcc ? 0u : b.emit_phi_2way(
                    b.t_bool, then_vcc, then_block, rs.vcc, else_block);
            if (then_exec != rs.exec)
                rs.exec = b.emit_phi_2way(b.t_bool, then_exec, then_block, rs.exec, else_block);
            rs.exec_narrowed = then_narrowed || rs.exec_narrowed;
            rs.sreg_written.insert(then_written.begin(), then_written.end());
            for (int reg : then_written) rs.sreg_input.erase(reg);
            merge_ud_alias(rs, then_ud_alias);   // `rs` holds the else edge here (#1773)
            if (then_bool != rs.sreg_bool || then_bool_b32 != rs.sreg_bool_b32) {
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "counted-loop prelude changes mask domain");
                return false; // no mask-domain PHIs in this narrow composition
            }
            if (!emit_range(merge_pc, L.header_pc)) return false;
        }
        // Ordinary counted loops require full EXEC at entry. An NGG vertex is already represented by
        // one independent Vulkan invocation, however, so its EXEC bit is an ordinary per-invocation
        // Boolean. Carry that bit through the loop just like SCC/VCC instead of rejecting Astro Bot's
        // fixed-trip culling loop merely because some guest lanes were masked before its header.
        const bool carry_vertex_exec = b.ngg_one_lane && rs.exec_narrowed;
        if (rs.exec_narrowed && !guarded_narrow_entry && !carry_vertex_exec) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "counted-loop enters with narrowed EXEC");
            return false;
        }
        // 2. Loop-carried registers -> a header OpPhi each. `cond_written` = regs written in the CONDITION
        // region [header, exit_branch): those execute on the exiting iteration too, so their post-loop
        // value is the condition-block value (which dominates the merge), NOT the phi (defect A). SCC/VCC
        // always get a phi so any cross-iteration carry is valid SSA (defect C); they're recomputed each
        // iteration in practice, so the phi is usually dead — harmless.
        std::set<int> cv, cs, condv, conds, scalar_may_writes;
        loop_written_regs(ins, L.header_pc, L.backedge_pc, cv, cs);
        loop_written_regs(ins, L.header_pc, L.exit_branch_pc, condv, conds);
        loop_scalar_may_writes(ins, L.header_pc, L.backedge_pc, scalar_may_writes);
        for (int reg : rs.sreg_bool_b32)
            if (scalar_may_writes.contains(reg)) return false;
        if (b.allow_b32_masks &&
            has_unpersisted_b32_mask_lifetime(
                ins, L.header_pc, L.backedge_pc, rs))
            return false;
        const uint32_t preheader = b.cur_block;
        const uint32_t hdr = b.id(), check = b.id(), body = b.id(), cont = b.id(), merge = b.id();
        // Loop-carried PHIs likewise seed a missing preheader SGPR with zero.
        rs.scalar_presence_has_no_placeholders = false;
        b.emit_branch(hdr); b.emit_label(hdr);
        struct PhiRec { int reg; int dom; uint32_t phi; size_t patch; };   // dom: 0=vreg,1=sreg,2=scc,3=vcc,4=exec
        std::vector<PhiRec> phis;
        // #3133: a loop-carried entry-M0 token has no value to seed a header phi with, and `sget`
        // would supply the `uconst(0)` the token exists to withhold. Unlike an if merge, "untracked
        // after the join" is not expressible here -- the phi is built before the body is emitted,
        // and its back-edge operand is patched from whatever the body left, so both operands would
        // have to be fabricated. Reject the region instead, which is exactly what happened to every
        // shader containing this instruction before #3133.
        if (const int m0_dst = entry_m0_save_in_range(ins, L.header_pc, L.backedge_pc); m0_dst >= 0) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "entry-M0 save inside a loop body (s%d)", m0_dst);
            return false;
        }
        for (int r : cs)
            if (entry_m0_live(rs, r)) {
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "entry-M0 token is loop-carried (s%d)", r);
                return false;
            }
        for (int r : cv) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, vget(r), preheader, p); rs.vreg[r] = ph; phis.push_back({r, 0, ph, p}); }
        for (int r : cs) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, sget(r), preheader, p); rs.sreg[r] = ph; phis.push_back({r, 1, ph, p}); }
        // A poisoned (0) SCC live-in degrades to bfalse — the loop shapes re-produce SCC via their
        // in-loop s_cmp before any read, so the phi seed is dead in practice; 0 would be invalid SSA.
        { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.scc ? rs.scc : b.bfalse(), preheader, p); rs.scc = ph; phis.push_back({0, 2, ph, p}); }
        if (rs.vcc) { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.vcc, preheader, p); rs.vcc = ph; phis.push_back({0, 3, ph, p}); }
        if (carry_vertex_exec) {
            size_t p;
            uint32_t ph = b.emit_phi2(b.t_bool, rs.exec, preheader, p);
            rs.exec = ph;
            phis.push_back({0, 4, ph, p});
        }
        // The header executes again after the back-edge. A direct/SRT descriptor overwritten
        // anywhere in the loop is therefore not an invariant entry descriptor at header compile
        // time. An exact descriptor load in the header may establish fresh provenance afterward.
        invalidate_loop_descriptor_provenance(rs, scalar_may_writes);
        // Snapshot AFTER that invalidation: this is the alias state on the path that skips the body
        // entirely (a top-tested loop may run zero times). Met against at the loop merge so an alias
        // the BODY establishes cannot outlive it (#1773).
        const auto loop_entry_ud_alias = rs.sreg_ud_alias;
        b.emit_loopmerge(merge, cont); b.emit_branch(check); b.emit_label(check);
        // 3. Condition block: emit [header, exit_branch); the SCC exit becomes OpBranchConditional.
        if (!emit_range(L.header_pc, L.exit_branch_pc)) return false;
        if (!rs.scc) return false;   // condition region left SCC poisoned: the exit test is unknowable
        // Snapshot condition-region register values (these dominate the merge — see defect A above).
        std::unordered_map<int,uint32_t> condv_val, conds_val;
        for (int r : condv) condv_val[r] = vget(r);
        for (int r : conds) conds_val[r] = sget(r);
        const uint32_t cond_exec = rs.exec;
        const bool cond_exec_narrowed = rs.exec_narrowed;
        // s_cbranch_scc0 exits when SCC==0 (so the loop CONTINUES when SCC!=0); scc1 is the inverse.
        uint32_t loop_cond = L.exit_on_scc0 ? rs.scc : b.bsel(rs.scc, b.bfalse(), b.btrue());
        b.emit_condbranch(loop_cond, body, merge);
        while (idx < ins.size() && ins[idx].pc < L.exit_branch_pc) ++idx;   // (already past it)
        if (idx < ins.size() && ins[idx].pc == L.exit_branch_pc) ++idx;     // skip the exit branch itself
        b.emit_label(body);
        // 4. Body [after exit_branch, back-edge). Must restore EXEC before looping (bail if left narrowed).
        if (!emit_range(L.exit_branch_pc + 1, L.backedge_pc)) return false;
        if (rs.exec_narrowed && !guarded_narrow_entry && !carry_vertex_exec) {
            log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                     "counted-loop body leaves EXEC narrowed");
            return false;
        }
        if (idx < ins.size() && ins[idx].pc == L.backedge_pc) ++idx;        // skip the back-edge branch
        // 5. Continue block branches back to the header; patch each phi's back-edge (value = current, cont).
        b.emit_branch(cont); b.emit_label(cont);
        for (auto& pr : phis) {
            uint32_t nv = pr.dom == 0 ? vget(pr.reg)
                        : pr.dom == 1 ? sget(pr.reg)
                        : pr.dom == 2 ? rs.scc
                        : pr.dom == 3 ? rs.vcc : rs.exec;
            if (!nv && pr.dom == 3) return false;
            if (!nv && pr.dom == 2)
                nv = b.bfalse(); // poisoned SCC back-edge value: false when dead in practice
            b.patch_phi(pr.patch, nv, cont);
        }
        b.emit_branch(hdr);
        // 6. Merge (loop exit): a condition-region reg keeps its exit-iteration (%check) value; a body-only
        //    reg (and scc/vcc) takes the header phi (its value when the loop exited).
        b.emit_label(merge);
        merge_ud_alias(rs, loop_entry_ud_alias);   // body-established aliases die here (#1773)
        for (auto& pr : phis) {
            if (pr.dom == 0)      rs.vreg[pr.reg] = condv.count(pr.reg) ? condv_val[pr.reg] : pr.phi;
            else if (pr.dom == 1) rs.sreg[pr.reg] = conds.count(pr.reg) ? conds_val[pr.reg] : pr.phi;
            // SCC/VCC take the phi (not a condition-region snapshot): reading a wave flag AFTER a loop is
            // not a real codegen pattern (flags are transient, consumed by their branch), so the A-class
            // exit-iteration refinement is intentionally omitted for them.
            else if (pr.dom == 2) rs.scc = pr.phi;
            else if (pr.dom == 3) rs.vcc = pr.phi;
            else                  rs.exec = cond_exec;
        }
        if (carry_vertex_exec) rs.exec_narrowed = cond_exec_narrowed;
        // 7. Post-loop body. Feed the suffix back through the ordinary body selector: with this
        // counted back-edge removed it can use the established nested forward-if/divergent-loop
        // structurizer. This composes a counted loop with a non-trivial shared postlude without
        // duplicating that CFG machinery or admitting arbitrary branches inside the counted loop.
        std::vector<Rdna2Inst> postloop;
        for (const auto& in : ins) if (in.pc >= L.exit_pc) postloop.push_back(in);
        if (!postloop.empty() &&
            !emit_body(b, rs, postloop, effective_safe, rt, allow_exec_update, allow_smem,
                       exp_fn, code, dwords)) return false;
    } else if (std::vector<DivLoop> Ls; true) {
        // EXEC/VCC/SCC-exit loops (#273/#615/#1554) + structured scalar IFs. Each is a real SPIR-V
        // loop with header phis for carried register/mask state. Fragment conditions are exact wave64
        // votes; vertex and the guarded compute cases retain their per-invocation form. The IF
        // machinery recurses into loop bodies and handles their nested forward-execz regions.
        Ls = detect_divergent_loops(ins, safe, b.is_fragment, b.diagnostic);
        if (b.is_compute) {
            // Compute VCC-exit loops (#590, extending #615): the fragment-stage uniformity proof is
            // data-provenance-based, not stage-based — vcc_exit_is_wave_uniform accepts a compare only
            // when every input is scalar/inline/literal or a VGPR whose nearest definition is an
            // unmodified uniform VOP1 move from a scalar. With that proven, every lane's compare bool
            // is identical, so the wave-empty vccz exit lowers to THIS invocation's !cond exactly as
            // in the fragment shell (tid-derived/varying inputs fail the proof and keep rejecting).
            // One compute-specific guard (see the per-condition detail below for why both Vcc- and
            // Exec-condition loops are safe here — the Exec case is GTA V's exec_cs_2042d47600 grid-stride
            // decode loop, #1183):
            //   * the body must be barrier/LDS/cross-lane-free — the proof is per-WAVE, and a barrier
            //     inside a loop whose trip count could differ across the workgroup's waves would be
            //     workgroup-divergent control flow (UB). DOLL's blocked light/fill kernels are
            //     straight-line bodies, so nothing observed is lost. CONFIDENCE: MED-HIGH (shared
            //     emit machinery; spirv-val + coverage tests + Messenger guard gate it).
            // SCC-condition loops use exact architectural scalar control and may retain ordinary
            // LDS effects; their synchronized/cross-lane operations remain rejected below.
            // Condition::Vcc loops are accepted under the detector's uniformity proof (#615/#590).
            // Condition::Exec loops (#590 — DOLL's nested post-process kernel is the observed compute
            // case) lower with the per-invocation model: this invocation
            // iterates while ITS EXEC bit holds after the header's v_cmpx recompute. Both flavors
            // require the barrier/LDS/cross-lane-free body below — the per-invocation trip count can
            // differ across a workgroup, so a barrier inside the loop would be workgroup-divergent
            // control flow (UB); barriers AFTER the loop are fine (the loop merge reconverges).
            bool compute_ok = !Ls.empty();
            const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
            auto has_stable_post_barrier_lds_reads = [&](const DivLoop& loop) {
                // The structured emitter performs the Workgroup load before predicating the VGPR
                // result. Limit this exception to a top-tested EXEC loop, where the body entry
                // proves this invocation active, and keep that proof live through every DS read.
                if (!phased.found || loop.condition != DivLoop::Condition::Exec ||
                    loop.bottom_tested) return false;
                size_t phase_begin = ins.size();
                size_t phase_end = phased.end_index;
                for (size_t barrier : phased.barriers) {
                    if (ins[barrier].pc < loop.header_pc) {
                        phase_begin = barrier + 1;
                        continue;
                    }
                    phase_end = barrier;
                    break;
                }
                if (phase_begin == ins.size() ||
                    (phase_end < ins.size() && loop.exit_pc > ins[phase_end].pc))
                    return false;
                // This narrow exception is read-only for the COMPLETE phase. A preceding proved
                // top-level barrier publishes earlier LDS initialization; with no later LDS write,
                // atomic, GDS access, or other DS opcode, each per-lane loop may repeat its own
                // ordinary DS_READ_B32 without introducing visibility or synchronization edges.
                for (size_t i = phase_begin; i < phase_end; ++i)
                    if (ins[i].fmt == Rdna2Format::DS &&
                        (ins[i].opcode != 0x36 || ins[i].ds_gds))
                        return false;
                for (const auto& read : ins) {
                    if (read.pc < loop.header_pc || read.pc >= loop.backedge_pc ||
                        read.fmt != Rdna2Format::DS) continue;
                    // The condition region runs before the top-of-loop EXEC test. A load there has
                    // no active-lane proof yet, so the emitter's unconditional Workgroup access is
                    // not safe even though the rest of the phase is read-only.
                    if (read.pc <= loop.exit_branch_pc) return false;
                    for (const auto& prior : ins) {
                        if (prior.pc <= loop.exit_branch_pc || prior.pc >= read.pc) continue;
                        if (rdna2_instruction_may_change_exec(prior)) return false;
                    }
                }
                return true;
            };
            for (const auto& L : Ls) {
                if (!compute_ok) break;
                const bool stable_post_barrier_reads =
                    has_stable_post_barrier_lds_reads(L);
                for (const auto& in : ins) {
                    if (in.is_end || in.pc >= L.exit_pc) break;
                    if (in.pc < L.header_pc) continue;
                    const bool ds_wave_collective = in.fmt == Rdna2Format::DS &&
                        (in.opcode == 0x35 || in.opcode == 0x3d || in.opcode == 0x3e ||
                         in.opcode == kDsOpcodeBpermuteB32);
                    // SCC is architectural scalar control, so an SCC loop executes ordinary LDS
                    // effects uniformly within each guest wave just like the existing CountedLoop
                    // path. EXEC/VCC loops retain their per-invocation approximation: only ordinary
                    // read-only LDS in the proved stable post-barrier phase above is safe. Cross-lane
                    // DS collectives, writes/atomics, MBCNT, and a guest barrier remain unavailable.
                    const bool stable_lds_read = stable_post_barrier_reads &&
                        in.fmt == Rdna2Format::DS && in.opcode == 0x36 && !in.ds_gds;
                    if ((in.fmt == Rdna2Format::DS &&
                         ((L.condition != DivLoop::Condition::Scc && !stable_lds_read) ||
                          ds_wave_collective)) ||
                        (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a) ||
                        (in.fmt == Rdna2Format::VOP3 &&
                         (in.opcode == 0x365 || in.opcode == 0x366))) { compute_ok = false; break; }
                }
                // Fail-visible marker for the per-invocation approximation's one known divergence: a
                // POST-loop read of an SGPR advanced inside an Exec-condition loop. With a lane-varying
                // bound, hardware advances in-loop scalars to the wave's MAX trip count while the
                // per-invocation lowering yields each invocation its own exit value. Uniform bounds
                // (all observed shapes) are exact. Diagnose loudly instead of silently diverging; if a
                // real kernel trips this AND has a varying bound, that is the evidence to revisit.
                if (compute_ok && L.condition == DivLoop::Condition::Exec && getenv("PROSPER_DBG")) {
                    std::set<int> lv, lsr;
                    loop_written_regs(ins, L.header_pc, L.backedge_pc, lv, lsr);
                    for (const auto& in : ins) {
                        if (in.is_end) break;
                        if (in.pc < L.exit_pc) continue;
                        for (uint32_t oi = 0; oi < in.n_src; ++oi)
                            if (in.src[oi].kind == OperandKind::SGPR && lsr.count((int)in.src[oi].value))
                                fprintf(stderr,
                                        "[compute-exec-loop] post-loop read of loop-advanced s%u at pc=%u "
                                        "(per-invocation value; wave max-trip on hardware)\n",
                                        (unsigned)in.src[oi].value, in.pc);
                    }
                }
            }
            if (!compute_ok) Ls.clear();   // unchanged behavior: the branch reaches emit_alu -> loud reject
        }
        bool cf_rejected = false;
        const std::vector<ForwardIf> Fs = detect_forward_ifs(ins, /*allow_vcc*/!b.is_compute, code, dwords, &safe,
                                                             Ls.empty() ? nullptr : &Ls, &cf_rejected,
                                                             /*compute_wave_branches*/b.is_compute,
                                                             b.diagnostic);

        // UE4's volume-lighting kernels combine nested EXEC loops with a backward VCC vote loop.
        // That shape is intentionally outside the narrow pattern structurizer above. Use the
        // dispatcher fallback only when there is unmistakably complex compute control flow and the
        // body is free of operations whose workgroup-uniform placement the dispatcher cannot prove.
        // Raw DS operations are wave-local memory effects and are valid in a case. Guest barriers
        // remain unsupported; MBCNT is hoisted into the dispatcher's uniform synchronized phase.
        // Existing straight-line, single-if, and recognized single-loop shaders keep their old path.
        size_t cfg_branches = 0;
        const bool graphics_cfg = b.is_fragment || b.is_vertex;
        bool cfg_has_backedge = false, cfg_dispatch_safe = b.is_compute || graphics_cfg;
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x0a)
                cfg_dispatch_safe = false;
            if (in.fmt == Rdna2Format::SOPP && in.opcode >= 0x02 && in.opcode <= 0x09 &&
                in.opcode != 0x03) {
                ++cfg_branches;
                if (branch_target(in) <= in.pc) cfg_has_backedge = true;
            }
        }
        // A lone exit + back-edge is the ordinary single-loop shape. Keep rejecting it when its
        // wave-uniformity proof fails; the dispatcher is reserved for genuinely nested/multi-branch CFGs.
        // The fallback is equally valid for an acyclic branch tree: Astro Bot's screen-space kernels
        // contain several nested forward EXEC/SCC early-outs but no back-edge. Requiring a back-edge
        // left those valid CFGs in the straight-line path, where their first branch rejected.
        const bool complex_compute_cfg = b.is_compute && cfg_dispatch_safe && cfg_branches > 2;
        const bool complex_graphics_cfg = graphics_cfg && cfg_dispatch_safe && cfg_branches > 2;
        const bool exact_compute_wave_cfg = b.is_compute &&
            std::any_of(Fs.begin(), Fs.end(), [](const ForwardIf& branch) {
                return branch.on_exec || branch.on_vcc;
            });
        // A structured shader may contain top-level guest-wave branches and workgroup-uniform barriers
        // between those regions (DOLL's title grading kernel and UE4's barrier-separated reductions).
        // The generic CFG dispatcher cannot contain guest barriers, but routing the entire shader there
        // is unnecessary: reduce each top-level branch with exact 32/64-lane scratch votes and retain the
        // structured emitter. Loops are not required; a sequence of forward top-level reductions has the
        // same safety argument. Neither a vote nor a guest barrier may be nested in a varying region.
        auto top_level_pc = [&](uint32_t pc) {
            for (const auto& parent : Fs) {
                // A proved workgroup-uniform region is entered or skipped by EVERY invocation of the
                // workgroup together (#1554), so it does not make a nested barrier or scratch vote
                // divergent. Only genuinely per-wave regions hide their contents from this test.
                if (parent.uniform_workgroup) continue;
                const uint32_t parent_end = parent.has_else ? parent.merge_pc : parent.target_pc;
                if (parent.branch_pc < pc && pc < parent_end)
                    return false;
            }
            for (const auto& loop : Ls)
                if (pc >= loop.header_pc && pc <= loop.backedge_pc)
                    return false;
            return true;
        };
        const bool barriers_are_top_level =
            std::all_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
                return in.fmt != Rdna2Format::SOPP || in.opcode != 0x0a ||
                       top_level_pc(in.pc);
            });
        const bool structured_has_cross_lane_mbcnt =
            std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
                return in.fmt == Rdna2Format::VOP3 &&
                       (in.opcode == 0x365 || in.opcode == 0x366) &&
                       !(in.src[0].kind == OperandKind::InlineInt && in.src[0].value == -1);
            });
        const bool structured_wave_forward_ifs_ok =
            std::all_of(Fs.begin(), Fs.end(), [&](const ForwardIf& branch) {
                // An exact native guest-size subgroup performs the vote without synthesized
                // workgroup barriers, so nested wave branches are safe. Portable scratch votes
                // must remain top-level so every workgroup invocation reaches their barriers.
                return (!branch.on_exec && !branch.on_vcc) || b.native_subgroup_size ||
                       top_level_pc(branch.branch_pc);
            });
        const bool structured_compute_wave_cfg = exact_compute_wave_cfg && !cf_rejected &&
            barriers_are_top_level && !structured_has_cross_lane_mbcnt &&
            structured_wave_forward_ifs_ok;
        // PROSPER_DBG: name WHICH conjunct denied structured wave emission, and what that
        // actually costs THIS stream.
        //
        // When `exact_compute_wave_cfg` holds and this conjunction does not, the stream loses
        // structured wave emission. What happens instead is NOT one outcome, and an earlier revision
        // of this reporter asserted the loop-emulating one unconditionally. That was wrong in a way
        // worth recording, because the wrong case is not a corner:
        //
        //   `barriers_are_top_level == false` requires a barrier inside a ForwardIf/DivLoop region.
        //   Any barrier before `is_end` also clears `cfg_dispatch_safe`, so the gate below always
        //   takes its `!cfg_dispatch_safe` arm -- and `analyze_barrier_phased_compute` marks a
        //   branch that crosses a barrier invalid, so the phased retry does not fire either and the
        //   shader is REJECTED OUTRIGHT. A line claiming "emulates loops" would then describe loop
        //   emulation for a program that emitted no SPIR-V at all.
        //
        // The census that accompanied that revision could not have caught it: the conjuncts it
        // measured were the two that leave `cfg_dispatch_safe` alone, so every sampled line landed
        // in the one arm where the sentence happened to be true. A control drawn from the arm that
        // works cannot test the arm that does not.
        //
        // So the outcome is computed rather than assumed, and it is in the KEY as well as the
        // message. `emit_body` reaches this point for one program address from the whole stream, the
        // counted-loop prelude (dispatcher deliberately withheld), the post-loop suffix and per-phase
        // sub-streams; keying on (program, conjunct) alone lets whichever ran first suppress the
        // rest, so the line with the wrong consequence would win. That is the same shared-key
        // suppression fixed on `divloop_reject` (#2684) and not inherited here until review.
        //
        // The neighbouring `[compute-cfg]` line prints `structured_wave=` as a single bool, which
        // cannot name a conjunct at all.
        if (b.is_compute && exact_compute_wave_cfg && !structured_compute_wave_cfg &&
            recompile_diagnostic_verbose(b.diagnostic.program_address) &&
            b.diagnostic.program_address != 0) {
            const char* outcome =
                !allow_cfg_dispatcher
                    ? "dispatcher withheld for this sub-stream; emission continues here"
                    : (!cfg_dispatch_safe
                           ? "dispatcher unsafe (guest barrier): phased retry, else the whole "
                             "program is rejected and emits nothing"
                           : "stream goes to the CFG dispatcher, which emulates loops");
            const std::pair<const char*, bool> conjuncts[] = {
                {"cf-rejected",                  !cf_rejected},
                {"barrier-not-top-level",        barriers_are_top_level},
                {"cross-lane-mbcnt",             !structured_has_cross_lane_mbcnt},
                {"nested-wave-forward-if",       structured_wave_forward_ifs_ok},
            };
            static std::mutex swr_mu;
            static std::set<std::tuple<uint64_t, std::string, std::string>> swr_seen;
            std::lock_guard<std::mutex> lk(swr_mu);
            // Clearing, not refusing: past the cap the reporter degrades into REPEATING lines rather
            // than dropping them. For a diagnostic, losing tidiness beats losing findings.
            if (swr_seen.size() >= 4096u) swr_seen.clear();
            for (const auto& c : conjuncts) {
                // EVERY false conjunct is named, not just the first. They are independent, so
                // fixing the first one reported can leave the decision unchanged -- a report that
                // stopped at one would send a reader to widen a guard that changes nothing.
                if (c.second) continue;
                if (!swr_seen.emplace(b.diagnostic.program_address, c.first, outcome).second)
                    continue;
                std::fprintf(stderr,
                             "[structured-wave-reject] program=0x%llx %s (%s)\n",
                             (unsigned long long)b.diagnostic.program_address, c.first, outcome);
            }
        }
        if (b.is_compute && cfg_branches && recompile_diagnostic_verbose(b.diagnostic.program_address))
            std::fprintf(stderr,
                         "[compute-cfg] branches=%zu backedge=%d dispatch_safe=%d complex=%d "
                         "structured_ifs=%zu loops=%zu cf_rejected=%d exact_wave=%d "
                         "structured_wave=%d local=%u wave=%u\n",
                         cfg_branches, cfg_has_backedge, cfg_dispatch_safe, complex_compute_cfg,
                         Fs.size(), Ls.size(), cf_rejected, exact_compute_wave_cfg,
                         structured_compute_wave_cfg,
                         b.local_count, b.wave_size);
        if (graphics_cfg && cfg_branches && recompile_diagnostic_verbose(b.diagnostic.program_address))
            std::fprintf(stderr,
                         "[graphics-cfg] stage=%s branches=%zu backedge=%d complex=%d "
                         "structured_ifs=%zu loops=%zu cf_rejected=%d\n",
                         b.is_fragment ? "fragment" : "vertex", cfg_branches,
                         cfg_has_backedge, complex_graphics_cfg, Fs.size(), Ls.size(), cf_rejected);
        if (allow_cfg_dispatcher && exact_compute_wave_cfg && !structured_compute_wave_cfg) {
            // Native Vulkan subgroup widths may be 8/16/32 while the guest wave is 32/64. A native
            // subgroupAny would let different pieces of one guest wave take different scalar edges.
            // The dispatcher performs the reduction through Workgroup scratch and synchronized common
            // phases. Top-level wave branches use the compact structured reduction above even without
            // a guest barrier; nested wave branches still need this dispatcher. If a guest barrier
            // makes that transformation unsafe, reject rather than silently changing the branch domain.
            if (!cfg_dispatch_safe) {
                const BarrierPhasedCompute phased = analyze_barrier_phased_compute(ins);
                if (phased.found && !phased.guarded)
                    return emit_body(
                        b, rs, ins, safe, rt, allow_exec_update, allow_smem,
                        exp_fn, code, dwords, &dead_masks, allow_cfg_dispatcher,
                        /*initial_dispatch_active=*/0, /*force_barrier_phases=*/true);
                log_recompile_diagnostic(
                    b.diagnostic, "compute-cfg-reject", "terminal",
                    "reason=exact-wave-dispatcher-unsafe guest-barrier=1");
                return false;
            }
            if (!emit_cfg_state_machine(b, rs, ins, safe, rt,
                                        allow_exec_update, allow_smem, exp_fn, code, dwords,
                                        initial_dispatch_active))
                return false;
            return true;
        }
        // Attempting the dispatcher is only recoverable while it has emitted NOTHING.
        // emit_cfg_state_machine writes the dispatcher loop's OpLoopMerge, its OpSelectionMerge and
        // the OpSwitch into b.code *before* it emits the per-case block bodies, and it can still
        // reject inside a case body (an unlowerable instruction, an unresolvable successor). Falling
        // through to the structurizer after that leaves the half-written construct in the module: the
        // switch and both merges survive, referencing case / default / continue / merge labels that
        // are never emitted. That is structurally invalid SPIR-V, and nothing downstream catches it —
        // `spirv-val` is a CI gate over representative modules, not over game shaders, so the module
        // reaches the driver, `spirv_to_nir` returns NULL, and RADV dereferences that NULL (#2396).
        // So the attempt is transactional: fall back only when the buffer is untouched, and reject
        // loudly once anything has been written.
        auto try_cfg_dispatcher = [&]() -> int {           // 1 = emitted, 0 = clean reject, -1 = partial
            const size_t checkpoint = b.code.size();
            if (emit_cfg_state_machine(b, rs, ins, safe, rt,
                                       allow_exec_update, allow_smem, exp_fn, code, dwords,
                                       initial_dispatch_active))
                return 1;
            if (b.code.size() == checkpoint) return 0;
            if (getenv("PROSPER_DBG"))
                std::fprintf(stderr,
                             "[cfg-dispatcher-partial] rejected after emitting %zu words; failing the "
                             "shader instead of shipping a half-written dispatcher\n",
                             b.code.size() - checkpoint);
            return -1;
        };
        const bool portable_compute_dpp_ror8 = b.is_compute &&
            !b.native_subgroup_size &&
            std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
                return dpp_row_ror8_op(in) != DppRowRor8Op::None;
            });
        if (allow_cfg_dispatcher && portable_compute_dpp_ror8) {
            // The generic straight-line/structured emitter can use ROW_ROR only when the backend
            // guarantees one exact native guest wave. Otherwise the complete program must enter the
            // synchronized dispatcher so every invocation reaches both scratch barriers uniformly.
            if (!cfg_dispatch_safe) {
                log_recompile_diagnostic(
                    b.diagnostic, "compute-cfg-reject", "terminal",
                    "reason=portable-dpp-row-ror8-dispatcher-unsafe guest-barrier=1");
                return false;
            }
            const int emitted = try_cfg_dispatcher();
            if (emitted > 0) return true;
            return false;
        }
        std::set<int> cfg_writelane_spill_arrays;
        for (const auto& in : ins) {
            if (in.is_end) break;
            if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361)
                cfg_writelane_spill_arrays.insert(in.dst.value);
        }
        const bool portable_compute_cfg_readlane = b.is_compute &&
            !b.native_subgroup_size && (!Fs.empty() || !Ls.empty()) &&
            std::any_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
                return !in.is_end && in.fmt == Rdna2Format::VOP3 &&
                    in.opcode == 0x360 &&
                    !cfg_writelane_spill_arrays.contains(in.src[0].value);
            });
        if (allow_cfg_dispatcher && portable_compute_cfg_readlane) {
            if (!cfg_dispatch_safe) {
                log_recompile_diagnostic(
                    b.diagnostic, "compute-cfg-reject", "terminal",
                    "reason=portable-readlane-dispatcher-unsafe guest-barrier=1");
                return false;
            }
            const int emitted = try_cfg_dispatcher();
            if (emitted > 0) return true;
            return false;
        }
        if (allow_cfg_dispatcher && complex_compute_cfg && (cf_rejected || Ls.empty())) {
            const int emitted = try_cfg_dispatcher();
            if (emitted > 0) return true;
            if (emitted < 0) return false;
        }
        // In graphics, the SPIR-V invocation already represents one guest lane. Complex reducible
        // control flow therefore needs no workgroup vote: the dispatcher selects the next block from
        // this pixel/vertex's SCC, VCC, or EXEC bit. Keep ordinary structured shaders on their compact
        // SSA path and use the Function-variable fallback only after the narrow structurizer rejects.
        // Only one of these two blocks can run: complex_compute_cfg requires b.is_compute and
        // complex_graphics_cfg requires graphics_cfg (b.is_fragment || b.is_vertex), and a module is
        // one or the other. That exclusivity is what makes a clean reject (0) from the compute block
        // safe to fall through — it reaches this block only in a stage that cannot enter it. If the
        // stage predicates above ever stop being disjoint, this second attempt would run against
        // builder state the first had already touched, and the checkpoint would no longer describe an
        // untouched buffer.
        if (allow_cfg_dispatcher && complex_graphics_cfg && cf_rejected) {
            const int emitted = try_cfg_dispatcher();
            if (emitted > 0) return true;
            if (emitted < 0) return false;
        }
        if (cf_rejected) Ls.clear();   // unmodeled CF somewhere: fall through to straight-line (loud reject)
        if (Fs.empty() && Ls.empty()) {
            wave_ok = true;   // straight-line: barriers are wave-uniform, so cross-lane mbcnt is safe here
            if (!emit_range(0, UINT32_MAX)) return false;   // loop-free: straight-line, unchanged behavior
            (void)safe_branches;
            return true;
        }
        // Structured uniform IFs (forward s_cbranch_scc*/vcc*), possibly SEQUENTIAL and/or NESTED
        // (detect_forward_ifs verified the region tree). Each if emits as OpSelectionMerge +
        // OpBranchConditional on the SCC/VCC bool, with an OpPhi per register written in the
        // conditional block that is live after the merge — the same phi machinery as the original
        // single-if path (which this generalizes 1:1: a single if takes exactly the old shape).
        // Recursion handles nesting: the then-block emitter re-enters for inner branches.
        // CONFIDENCE: MED-HIGH — guarded by the test suite + exec-diff; DOLL's two-vccz color-grade
        // PS and nested-vccz lighting PS are the motivating real shaders (#273).
        auto vget = [&](int r){ auto it = rs.vreg.find(r); return it == rs.vreg.end() ? b.uconst(0) : it->second; };
        auto sget = [&](int r){ auto it = rs.sreg.find(r); return it == rs.sreg.end() ? b.uconst(0) : it->second; };
        size_t bi = 0;   // next unconsumed branch in Fs (pc order; recursion consumes nested ones)
        size_t li = 0;   // next unconsumed loop in Ls (pc order)
        const DivLoop* active_direct_wave_loop = nullptr;
        uint32_t* active_direct_wave_continue = nullptr;
        // `cont` = the pc control flows to after `hi` — the enclosing construct's merge chain. An
        // if/else whose merge ESCAPES the current region (the shared-outer-merge cascade) is legal
        // only when it targets exactly this continuation (no skipped instructions); anything else
        // rejects, fail-visible. Fragment EXEC/VCC branches vote over the enforced wave64; compute
        // uses its exact guest-wave reduction paths. Either may enter/leave with EXEC narrowed, and
        // EXEC is phi'd across the merge like any other value.
        std::function<bool(uint32_t, uint32_t, uint32_t)> emit_structured;
        // Emit one EXEC/VCC/SCC-exit loop (#273/#615/#1554) as structured SPIR-V. Same block shape as
        // the counted-loop path (hdr -> chk -> body -> cont -> hdr, exit chk->merge) with three
        // differences: (1) fragment votes the complete wave's EXEC/VCC after the header recompute
        // (other guarded stages consume this lane's bool); (2) the body is
        // emitted RECURSIVELY (nested forward-execz if regions live inside it); (3) per-lane MASKS
        // (VCC/EXEC/saved sreg_bool pairs) are loop-carried too, so each gets a header phi. At the
        // merge, a register written in the condition region keeps its exit-iteration (chk) value —
        // it dominates the merge — while body-written state takes the header phi (its value when the
        // exiting check ran); masks CREATED inside the loop are dropped at the merge (an SSA id from
        // inside the body does not dominate it — a later read then rejects loudly instead of
        // emitting invalid SPIR-V). Execution tests cover both wave-vote outcomes and direct breaks.
        std::function<bool(const DivLoop&)> emit_divloop = [&](const DivLoop& L) -> bool {
            // A loop-carried SGPR absent at the preheader receives a zero PHI input, so compact map
            // presence inside or after the loop cannot stand in for a scalar-lifetime MUST proof.
            rs.scalar_presence_has_no_placeholders = false;
            const bool entry_exec_narrowed = rs.exec_narrowed;
            std::set<int> cv, cs, condv, conds, scalar_may_writes;
            loop_written_regs(ins, L.header_pc, L.backedge_pc, cv, cs);
            loop_written_regs(ins, L.header_pc, L.exit_branch_pc, condv, conds);
            loop_scalar_may_writes(ins, L.header_pc, L.backedge_pc, scalar_may_writes);
            for (int reg : rs.sreg_bool_b32)
                if (scalar_may_writes.contains(reg)) return false;
            if (b.allow_b32_masks &&
                has_unpersisted_b32_mask_lifetime(
                    ins, L.header_pc, L.backedge_pc, rs))
                return false;
            const uint32_t preheader = b.cur_block;
            const uint32_t hdr = b.id(), chk = b.id(), body = b.id(), cont = b.id(), merge = b.id();
            b.emit_branch(hdr); b.emit_label(hdr);
            struct PhiRec { int reg; int dom; uint32_t phi; size_t patch; };  // dom: 0=vreg,1=sreg,2=scc,3=vcc,4=exec,5=mask
            std::vector<PhiRec> phis;
            // #3133: a loop-carried entry-M0 token has no value to seed a header phi with, and `sget`
            // would supply the `uconst(0)` the token exists to withhold. Unlike an if merge, "untracked
            // after the join" is not expressible here -- the phi is built before the body is emitted,
            // and its back-edge operand is patched from whatever the body left, so both operands would
            // have to be fabricated. Reject the region instead, which is exactly what happened to every
            // shader containing this instruction before #3133.
            if (const int m0_dst = entry_m0_save_in_range(ins, L.header_pc, L.backedge_pc); m0_dst >= 0) {
                log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                         "entry-M0 save inside a loop body (s%d)", m0_dst);
                return false;
            }
            for (int r : cs)
                if (entry_m0_live(rs, r)) {
                    log_recompile_diagnostic(b.diagnostic, "recompile-reject", "terminal",
                                             "entry-M0 token is loop-carried (s%d)", r);
                    return false;
                }
            for (int r : cv) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, vget(r), preheader, p); rs.vreg[r] = ph; phis.push_back({r, 0, ph, p}); }
            for (int r : cs) { size_t p; uint32_t ph = b.emit_phi2(b.t_u32, sget(r), preheader, p); rs.sreg[r] = ph; phis.push_back({r, 1, ph, p}); }
            // A poisoned (0) SCC live-in degrades to bfalse (invalid as an SSA phi input; dead in
            // practice — the loop shapes re-produce SCC before any read).
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.scc ? rs.scc : b.bfalse(), preheader, p); rs.scc = ph; phis.push_back({0, 2, ph, p}); }
            if (rs.vcc) { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.vcc, preheader, p); rs.vcc = ph; phis.push_back({0, 3, ph, p}); }
            { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.exec, preheader, p); rs.exec = ph; phis.push_back({0, 4, ph, p}); }
            std::vector<int> mask_keys;                        // saved masks live at entry: loop-carried bools
            for (auto& kv : rs.sreg_bool) mask_keys.push_back(kv.first);
            std::sort(mask_keys.begin(), mask_keys.end());     // deterministic emission order
            for (int k : mask_keys) { size_t p; uint32_t ph = b.emit_phi2(b.t_bool, rs.sreg_bool[k], preheader, p); rs.sreg_bool[k] = ph; phis.push_back({k, 5, ph, p}); }
            invalidate_loop_descriptor_provenance(rs, scalar_may_writes);
            // See the sibling loop above: the zero-trip path carries these aliases, not the body's.
            const auto loop_entry_ud_alias = rs.sreg_ud_alias;
            b.emit_loopmerge(merge, cont); b.emit_branch(chk); b.emit_label(chk);
            // An EXEC-governed loop predicates vector writes. VCC/SCC-governed loops branch on their
            // represented predicate but do not themselves change EXEC, matching the hardware body.
            if (L.condition == DivLoop::Condition::Exec) rs.exec_narrowed = true;
            // A top-tested condition region is branch-free. For a bottom-tested EXEC loop the
            // same interval is the complete do-while body and may contain validated nested loops;
            // recurse so those children keep their own structured merges before the latch test.
            const uint32_t condition_entry_scc = rs.scc;
            if (L.bottom_tested) {
                if (!emit_structured(
                        L.header_pc, L.exit_branch_pc, L.exit_branch_pc)) return false;
            } else if (!emit_range(L.header_pc, L.exit_branch_pc)) {
                return false;
            }
            // A canonical SCC loop must compute a fresh representable scalar predicate on every
            // header visit. Reusing the header phi would admit stale SCC, while a B64 wave-mask
            // producer poisons rs.scc to zero; both remain fail-visible.
            if (L.condition == DivLoop::Condition::Scc &&
                (!rs.scc || rs.scc == condition_entry_scc))
                return false;
            // chk-end snapshots: the exit path flows THROUGH this block, so these dominate the merge.
            std::unordered_map<int, uint32_t> condv_val, conds_val;
            for (int r : condv) condv_val[r] = vget(r);
            for (int r : conds) conds_val[r] = sget(r);
            const uint32_t exec_chk = rs.exec, vcc_chk = rs.vcc, scc_chk = rs.scc;
            const std::unordered_map<int, uint32_t> bool_chk = rs.sreg_bool;
            uint32_t loop_cond = L.condition == DivLoop::Condition::Exec ? rs.exec
                               : L.condition == DivLoop::Condition::Vcc ? rs.vcc : rs.scc;
            if (!loop_cond) return false;
            if (!L.continue_on_set)
                loop_cond = b.logical_not(loop_cond);
            // EXECZ/VCCZ are scalar wave decisions. Keeping every fragment invocation in the loop
            // until the complete guest wave becomes empty makes scalar state and nested wave votes
            // exact; vector writes remain predicated by the per-lane EXEC bool.
            if (b.is_fragment && L.condition != DivLoop::Condition::Scc &&
                !(L.condition == DivLoop::Condition::Vcc &&
                  rs.vcc == rs.vcc_wave_uniform &&
                  vcc_exit_is_wave_uniform(ins, L.exit_branch_pc)))
                loop_cond = b.fragment_wave_any(loop_cond);
            const uint32_t chk_end = b.cur_block;
            b.emit_condbranch(loop_cond, body, merge);         // canonical exit: branch on continue predicate
            while (idx < ins.size() && ins[idx].pc < L.exit_branch_pc) ++idx;
            if (idx < ins.size() && ins[idx].pc == L.exit_branch_pc) ++idx;   // consume the exit branch
            b.emit_label(body);
            // Body (recursive: nested if regions + breaks); ends just before the back-edge.
            uint32_t direct_wave_continue = b.btrue();
            const DivLoop* prior_direct_wave_loop = active_direct_wave_loop;
            uint32_t* prior_direct_wave_continue = active_direct_wave_continue;
            active_direct_wave_loop = &L;
            active_direct_wave_continue = &direct_wave_continue;
            const bool body_ok = L.bottom_tested || emit_structured(
                L.exit_branch_pc + 1, L.backedge_pc, L.backedge_pc);
            active_direct_wave_loop = prior_direct_wave_loop;
            active_direct_wave_continue = prior_direct_wave_continue;
            if (!body_ok) return false;
            const bool body_exec_narrowed = rs.exec_narrowed;
            if (idx < ins.size() && ins[idx].pc == L.backedge_pc) ++idx;      // consume the back-edge
            const uint32_t body_end = b.cur_block;
            // An unconditional back-edge can re-enable EXEC at the header. When an interior EXECZ
            // targets the loop exit, send the inactive invocation straight to the loop merge; the
            // ordinary path still reaches the continue block and patches the loop-carried phis.
            uint32_t continue_condition = direct_wave_continue;
            if (L.direct_exec_breaks) continue_condition = b.land(continue_condition, rs.exec);
            if (L.direct_exec_breaks || L.direct_wave_breaks)
                b.emit_condbranch(continue_condition, cont, merge);
            else
                b.emit_branch(cont);
            b.emit_label(cont);
            for (auto& pr : phis) {
                uint32_t nv = pr.dom == 0 ? vget(pr.reg)
                            : pr.dom == 1 ? sget(pr.reg)
                            : pr.dom == 2 ? rs.scc
                            : pr.dom == 3 ? rs.vcc
                            : pr.dom == 4 ? rs.exec
                            : (rs.sreg_bool.count(pr.reg) ? rs.sreg_bool[pr.reg] : pr.phi);
                if (!nv && pr.dom == 3) return false;
                if (!nv && pr.dom == 2) nv = b.bfalse();
                b.patch_phi(pr.patch, nv, cont);
            }
            b.emit_branch(hdr);
            b.emit_label(merge);
            merge_ud_alias(rs, loop_entry_ud_alias);   // body-established aliases die here (#1773)
            for (auto& pr : phis) {
                if (pr.dom == 3 && (!vcc_chk || !rs.vcc)) return false;
                uint32_t chk_value = pr.dom == 0 ? (condv.count(pr.reg) ? condv_val[pr.reg] : pr.phi)
                                   : pr.dom == 1 ? (conds.count(pr.reg) ? conds_val[pr.reg] : pr.phi)
                                   : pr.dom == 2 ? (scc_chk ? scc_chk : b.bfalse())
                                   : pr.dom == 3 ? vcc_chk
                                   : pr.dom == 4 ? exec_chk
                                   : (bool_chk.count(pr.reg) ? bool_chk.at(pr.reg) : pr.phi);
                uint32_t body_value = pr.dom == 0 ? vget(pr.reg)
                                    : pr.dom == 1 ? sget(pr.reg)
                                    : pr.dom == 2 ? (rs.scc ? rs.scc : b.bfalse())
                                    : pr.dom == 3 ? rs.vcc
                                    : pr.dom == 4 ? rs.exec
                                    : (rs.sreg_bool.count(pr.reg) ? rs.sreg_bool[pr.reg] : pr.phi);
                const uint32_t merged = (L.direct_exec_breaks || L.direct_wave_breaks) &&
                                                chk_value != body_value
                    ? b.emit_phi_2way(pr.dom <= 1 ? b.t_u32 : b.t_bool,
                                      chk_value, chk_end, body_value, body_end)
                    : chk_value;
                if (pr.dom == 0)      rs.vreg[pr.reg] = merged;
                else if (pr.dom == 1) rs.sreg[pr.reg] = merged;
                else if (pr.dom == 2) rs.scc = merged;
                else if (pr.dom == 3) rs.vcc = merged;
                else if (pr.dom == 4) rs.exec = merged;
                else                  rs.sreg_bool[pr.reg] = merged;
            }
            // Masks CREATED inside the loop: their ids do not dominate the merge — drop them.
            for (auto it = rs.sreg_bool.begin(); it != rs.sreg_bool.end();) {
                if (!std::binary_search(mask_keys.begin(), mask_keys.end(), it->first)) {
                    rs.sreg_bool_narrowed.erase(it->first);
                    rs.sreg_bool_b32.erase(it->first);
                    it = rs.sreg_bool.erase(it);
                } else ++it;
            }
            // An execz exit leaves this lane inactive until the compiled restore. A vccz exit leaves
            // EXEC unchanged; preserve any narrowing that existed on entry or occurred in the body.
            rs.exec_narrowed = L.condition == DivLoop::Condition::Exec
                ? true : (entry_exec_narrowed || body_exec_narrowed);
            return true;
        };
        emit_structured =
            [&](uint32_t lo, uint32_t hi, uint32_t cont) -> bool {
            for (;;) {
                const uint32_t next_br = (bi < Fs.size() && Fs[bi].branch_pc < hi) ? Fs[bi].branch_pc : hi;
                const uint32_t next_lp = (li < Ls.size() && Ls[li].header_pc < hi) ? Ls[li].header_pc : hi;
                if (next_lp < next_br) {                     // a loop begins before the next if
                    if (!emit_range(lo, next_lp)) return false;
                    const DivLoop& L = Ls[li++];
                    if (!emit_divloop(L)) return false;
                    lo = L.exit_pc;
                    continue;
                }
                if (!emit_range(lo, next_br)) return false;
                if (next_br == hi) return true;
                const ForwardIf F = Fs[bi++];
                if (idx < ins.size() && ins[idx].pc == F.branch_pc) ++idx;   // skip the branch itself
                // Arm merges synthesize zero for a scalar absent on either predecessor. Keep the
                // branch-free prefix eligible for exact path-local proofs, then turn them off for
                // both arms and every successor of this construct.
                rs.scalar_presence_has_no_placeholders = false;
                // scc0/vccz/execz: branch (skip block) taken when the flag==0 → the block runs when
                // flag!=0; scc1/vccnz are the inverse. Compute and fragment reduce per-invocation
                // mask bits to the architecture's wave-wide "any lane active" predicate.
                // A poisoned SCC (0: last written by a 64-bit mask op) cannot condition a real
                // structured if — reject (the ISA-audit #879 stale-SCC consumer).
                if (!F.on_exec && !F.on_vcc && !rs.scc) {
                    log_recompile_diagnostic(b.diagnostic, "compute-struct-reject", "terminal",
                                             "poisoned SCC at branch pc=%u", F.branch_pc);
                    return false;
                }
                if (F.on_vcc && !rs.vcc) {
                    log_recompile_diagnostic(b.diagnostic, "compute-struct-reject", "terminal",
                                             "missing VCC at branch pc=%u", F.branch_pc);
                    return false;
                }
                // Compute VCC/EXEC branches normally return through the exact guest-wave dispatcher.
                // The narrow structured-wave path above accepts only top-level vote sites, where all
                // workgroup invocations may participate in the scratch barriers uniformly.
                if (b.is_compute && (F.on_exec || F.on_vcc) && !structured_compute_wave_cfg) {
                    log_recompile_diagnostic(b.diagnostic, "compute-struct-reject", "terminal",
                                             "unavailable wave vote at branch pc=%u", F.branch_pc);
                    return false;
                }
                uint32_t cond_reg = F.on_exec ? rs.exec : (F.on_vcc ? rs.vcc : rs.scc);
                if (b.is_compute && (F.on_exec || F.on_vcc))
                    cond_reg = b.native_subgroup_size
                        ? b.native_wave_any(cond_reg)
                        : b.guest_wave_any(cond_reg);
                else if (b.is_fragment && (F.on_exec || F.on_vcc) &&
                         !(F.on_vcc && rs.vcc == rs.vcc_wave_uniform &&
                           vcc_exit_is_wave_uniform(ins, F.branch_pc)))
                    cond_reg = b.fragment_wave_any(cond_reg);
                uint32_t exec_cond = F.on_scc0 ? cond_reg : b.bsel(cond_reg, b.bfalse(), b.btrue());
                if (active_direct_wave_loop && active_direct_wave_continue &&
                    active_direct_wave_loop->direct_wave_breaks && F.on_vcc &&
                    std::find(active_direct_wave_loop->break_pcs.begin(),
                              active_direct_wave_loop->break_pcs.end(), F.branch_pc) !=
                        active_direct_wave_loop->break_pcs.end())
                    *active_direct_wave_continue =
                        b.land(*active_direct_wave_continue, exec_cond);
                const uint32_t preblock = b.cur_block;      // block holding the OpBranchConditional
                if (!F.has_else) {
                    std::set<int> ifv, ifs;
                    loop_written_regs(ins, F.branch_pc + 1, F.target_pc, ifv, ifs);
                    std::unordered_map<int,uint32_t> pre_v, pre_s;
                    for (int r : ifv) pre_v[r] = vget(r);
                    for (int r : ifs) pre_s[r] = sget(r);
                    // #3133: the SKIPPED edge's entry-M0 tokens. `sget` rendered each as uconst(0)
                    // just now, which is exactly the fabrication the meet below refuses to phi.
                    std::set<int> pre_entry_m0;
                    for (int r : ifs)
                        if (entry_m0_live(rs, r)) pre_entry_m0.insert(r);
                    uint32_t pre_scc = rs.scc, pre_vcc = rs.vcc, pre_exec = rs.exec;
                    const bool pre_narrowed = rs.exec_narrowed;
                    const std::unordered_map<int,uint32_t> pre_bool = rs.sreg_bool;   // mask-domain snapshot
                    const auto pre_bool_b32 = rs.sreg_bool_b32;
                    // The SKIPPED edge keeps the pre-branch bits, so it keeps the pre-branch copy
                    // aliases. Anything the arm establishes is true on one edge only (#1773).
                    const auto pre_ud_alias = rs.sreg_ud_alias;
                    uint32_t thenL = b.id(), mergeL = b.id();
                    b.emit_selmerge(mergeL); b.emit_condbranch(exec_cond, thenL, mergeL);
                    b.emit_label(thenL);
                    if (!emit_structured(F.branch_pc + 1, F.target_pc, F.target_pc)) return false;
                    const uint32_t thenEnd = b.cur_block;   // last block of the then-body (nested ifs move it)
                    std::unordered_map<int,uint32_t> then_v, then_s;
                    for (int r : ifv) then_v[r] = vget(r);
                    for (int r : ifs) then_s[r] = sget(r);
                    uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
                    const bool then_narrowed = rs.exec_narrowed;
                    // The skipped edge retains the entry-time physical-word lifetime. A B32 mask
                    // created or invalidated only in the taken arm therefore needs a validity phi,
                    // which this narrow merge does not represent. Reject rather than attach the
                    // taken arm's marker to the synthesized bool value on both paths.
                    std::set<int> dead_b32_at_merge;
                    std::set_symmetric_difference(
                        rs.sreg_bool_b32.begin(), rs.sreg_bool_b32.end(),
                        pre_bool_b32.begin(), pre_bool_b32.end(),
                        std::inserter(dead_b32_at_merge, dead_b32_at_merge.end()));
                    const bool dead_domain_difference =
                        std::all_of(dead_b32_at_merge.begin(), dead_b32_at_merge.end(),
                                    [&](int reg) {
                                        return sgpr_dead_at_merge(ins, F.target_pc, reg);
                                    });
                    if (!dead_domain_difference) {
                        log_recompile_diagnostic(
                            b.diagnostic, "compute-struct-reject", "terminal",
                            "live b32 mask domain differs across branch pc=%u merge=%u",
                            F.branch_pc, F.target_pc);
                        return false;
                    }
                    b.emit_branch(mergeL); b.emit_label(mergeL);
                    for (int r : ifv) rs.vreg[r] = b.emit_phi_2way(b.t_u32,  pre_v[r], preblock, then_v[r], thenEnd);
                    for (int r : ifs) {   // `rs` is the taken arm; the skipped edge is `pre_*`
                        if (!join_entry_m0(rs, r, pre_entry_m0.count(r) != 0)) continue;   // #3133
                        rs.sreg[r] = b.emit_phi_2way(b.t_u32,  pre_s[r], preblock, then_s[r], thenEnd);
                    }
                    if (then_scc != pre_scc)   // poisoned (0) inputs degrade to bfalse across the merge
                        rs.scc = b.emit_phi_2way(b.t_bool, pre_scc ? pre_scc : b.bfalse(), preblock,
                                                 then_scc ? then_scc : b.bfalse(), thenEnd);
                    if (then_vcc != pre_vcc)
                        rs.vcc = !pre_vcc || !then_vcc ? 0u : b.emit_phi_2way(
                            b.t_bool, pre_vcc, preblock, then_vcc, thenEnd);
                    // EXEC changed inside the arm (saveexec / v_cmpx / restore): merge it like any value.
                    // Narrowed-ness is sticky (either edge narrowed → post-merge writes stay predicated).
                    if (then_exec != pre_exec) rs.exec = b.emit_phi_2way(b.t_bool, pre_exec, preblock, then_exec, thenEnd);
                    rs.exec_narrowed = pre_narrowed || then_narrowed;
                    // A saved MASK (sreg_bool) created or changed inside the block must dominate both
                    // merge predecessors. A newly-created per-lane mask is false on the skipped edge.
                    for (auto& kv : rs.sreg_bool) {
                        auto p = pre_bool.find(kv.first);
                        const uint32_t before = p != pre_bool.end() ? p->second : b.bfalse();
                        if (before != kv.second) {
                            kv.second = b.emit_phi_2way(b.t_bool, before, preblock, kv.second, thenEnd);
                            rs.sreg_bool_narrowed[kv.first] = true;   // conservative: provenance now mixed
                        }
                    }
                    // A differing physical-word domain needs no validity phi when that word is
                    // provably overwritten before every post-merge read. Drop its stale typed view
                    // on both synthesized paths; the later defining instruction recreates the
                    // appropriate scalar or mask lifetime.
                    for (int reg : dead_b32_at_merge) {
                        rs.sreg_bool_b32.erase(reg);
                        rs.sreg_bool.erase(reg);
                        rs.sreg_bool_narrowed.erase(reg);
                        if (reg == 106) rs.vcc = 0;
                    }
                    merge_ud_alias(rs, pre_ud_alias);   // meet against the skipped edge (#1773)
                    lo = F.target_pc;   // continue after the merge (further sequential ifs handled here)
                } else {
                    // IF/ELSE: then = [branch_pc+1, sb_pc) (its s_branch terminator is consumed);
                    // else = [target_pc, merge). A merge escaping this region must be exactly the
                    // enclosing continuation `cont` (the cascade shape) — the else-arm then runs to
                    // `hi` and the merge coincides with the region end.
                    uint32_t else_hi = F.merge_pc;
                    if (F.merge_pc >= hi) {
                        if (F.merge_pc != cont && F.merge_pc != hi) {
                            log_recompile_diagnostic(
                                b.diagnostic, "compute-struct-reject", "terminal",
                                "escaping merge pc=%u merge=%u region=%u continuation=%u",
                                F.branch_pc, F.merge_pc, hi, cont);
                            return false;
                        }
                        else_hi = hi;
                    }
                    const RegState pre = rs;                // FULL snapshot: the else-arm re-runs from it
                    std::set<int> wv, ws;                   // regs written in EITHER arm
                    loop_written_regs(ins, F.branch_pc + 1, F.sb_pc, wv, ws);
                    loop_written_regs(ins, F.target_pc, else_hi, wv, ws);
                    uint32_t thenL = b.id(), elseL = b.id(), mergeL = b.id();
                    b.emit_selmerge(mergeL); b.emit_condbranch(exec_cond, thenL, elseL);
                    b.emit_label(thenL);
                    if (!emit_structured(F.branch_pc + 1, F.sb_pc, F.merge_pc)) return false;
                    if (idx < ins.size() && ins[idx].pc == F.sb_pc) ++idx;   // consume the arm's s_branch
                    const uint32_t thenEnd = b.cur_block;
                    std::unordered_map<int,uint32_t> then_v, then_s;
                    for (int r : wv) then_v[r] = vget(r);
                    for (int r : ws) then_s[r] = sget(r);
                    std::set<int> then_entry_m0;                 // #3133, the then edge's tokens
                    for (int r : ws)
                        if (entry_m0_live(rs, r)) then_entry_m0.insert(r);
                    uint32_t then_scc = rs.scc, then_vcc = rs.vcc, then_exec = rs.exec;
                    const bool then_narrowed = rs.exec_narrowed;
                    const std::unordered_map<int,uint32_t> then_bool = rs.sreg_bool;
                    const auto then_bool_b32 = rs.sreg_bool_b32;
                    const auto then_written = rs.sreg_written;
                    const auto then_ud_alias = rs.sreg_ud_alias;   // the then edge's alias claims
                    b.emit_branch(mergeL);
                    rs = pre;                               // else-arm starts from the pre-branch state
                    b.emit_label(elseL);
                    if (!emit_structured(F.target_pc, else_hi, F.merge_pc)) return false;
                    const uint32_t elseEnd = b.cur_block;
                    b.emit_branch(mergeL); b.emit_label(mergeL);
                    for (int r : wv) { uint32_t ev = vget(r);
                        if (then_v[r] != ev) rs.vreg[r] = b.emit_phi_2way(b.t_u32, then_v[r], thenEnd, ev, elseEnd); }
                    for (int r : ws) { uint32_t es = sget(r);
                        // `rs` is the else edge here (the then arm's state was rolled back).
                        if (!join_entry_m0(rs, r, then_entry_m0.count(r) != 0)) continue;   // #3133
                        if (then_s[r] != es) rs.sreg[r] = b.emit_phi_2way(b.t_u32, then_s[r], thenEnd, es, elseEnd); }
                    if (then_scc != rs.scc)   // poisoned (0) inputs degrade to bfalse across the merge
                        rs.scc = b.emit_phi_2way(b.t_bool, then_scc ? then_scc : b.bfalse(), thenEnd,
                                                 rs.scc ? rs.scc : b.bfalse(), elseEnd);
                    if (then_vcc != rs.vcc)
                        rs.vcc = !then_vcc || !rs.vcc ? 0u : b.emit_phi_2way(
                            b.t_bool, then_vcc, thenEnd, rs.vcc, elseEnd);
                    if (then_exec != rs.exec) rs.exec = b.emit_phi_2way(b.t_bool, then_exec, thenEnd, rs.exec, elseEnd);
                    rs.exec_narrowed = then_narrowed || rs.exec_narrowed;
                    rs.sreg_written.insert(then_written.begin(), then_written.end());
                    for (int reg : then_written) rs.sreg_input.erase(reg);
                    merge_ud_alias(rs, then_ud_alias);   // `rs` holds the else edge here (#1773)
                    if (then_bool_b32 != rs.sreg_bool_b32) {
                        log_recompile_diagnostic(
                            b.diagnostic, "compute-struct-reject", "terminal",
                            "b32 mask domain differs across if/else pc=%u merge=%u",
                            F.branch_pc, F.merge_pc);
                        return false;
                    }
                    // Merge the UNION of mask keys. A mask created in only one arm is false in the
                    // other arm; leaving that arm-local SSA id live after the merge is invalid SPIR-V.
                    std::set<int> bool_keys;
                    for (const auto& kv : then_bool) bool_keys.insert(kv.first);
                    for (const auto& kv : rs.sreg_bool) bool_keys.insert(kv.first);
                    for (int key : bool_keys) {
                        auto t = then_bool.find(key);
                        auto e = rs.sreg_bool.find(key);
                        const uint32_t tv = t != then_bool.end() ? t->second : b.bfalse();
                        const uint32_t ev = e != rs.sreg_bool.end() ? e->second : b.bfalse();
                        rs.sreg_bool[key] = tv == ev ? tv
                            : b.emit_phi_2way(b.t_bool, tv, thenEnd, ev, elseEnd);
                        if (tv != ev) rs.sreg_bool_narrowed[key] = true;
                    }
                    lo = else_hi;   // continue after the merge (== hi for the escaping-cascade shape)
                }
            }
        };
        // Every scalar branch and loop condition is subgroup-uniform in the fragment shell (SCC is
        // scalar already; EXECZ/VCCZ use fragment_wave_any), so native wave operations in any
        // structured arm observe the complete guest wave.
        wave_ok = b.is_fragment;
        if (!emit_structured(0, UINT32_MAX, UINT32_MAX)) {
            if (getenv("PROSPER_DBG")) {
                const uint32_t next_pc = idx < ins.size() ? ins[idx].pc : UINT32_MAX;
                const uint32_t next_if = bi < Fs.size() ? Fs[bi].branch_pc : UINT32_MAX;
                const uint32_t next_loop = li < Ls.size() ? Ls[li].header_pc : UINT32_MAX;
                log_recompile_diagnostic(
                    b.diagnostic, "compute-struct-reject", "consequent",
                    "structured emission stopped next-pc=%u next-if=%u next-loop=%u",
                    next_pc, next_if, next_loop);
            }
            return false;
        }
    }
    (void)safe_branches;
    return true;
}

namespace {

// A no-GS NGG program is split into two machine-code allocations by the guest compiler: the
// logical vertex producer writes one compact per-vertex LDS record, then a compiler-generated NGG
// wrapper culls/compacts primitives and exports fields from that record.  Vulkan's vertex stage
// already launches exactly the logical draw vertices and performs primitive assembly itself.  When
// both sides of this ABI can be proven from the machine code, execute only the producer and export
// the same LDS fields directly.  This avoids pretending that Function-private LDS can communicate
// between independent Vulkan vertex invocations.

} // namespace

} // namespace prosper::gpu
