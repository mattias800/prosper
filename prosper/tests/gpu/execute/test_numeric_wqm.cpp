// Numeric WQM expands nonzero four-bit groups, independently of EXEC and host subgroup width.
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "fixtures/compute_runner.h"
#include <array>
#include <bit>
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t N = 128;
constexpr uint32_t Adjacent = 0xa5a5f00fu;
struct Checks {
    unsigned failures = 0;
    void check(bool ok, const char* label) {
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", label);
        failures += !ok;
    }
};
struct Case { uint32_t operand, lo, hi; };
constexpr std::array<Case, 16> Cases{{
    {255, 0, 0}, {129, 1, 0}, {131, 3, 0}, {143, 15, 0},
    {194, 0xfffffffeu, UINT32_MAX}, {193, UINT32_MAX, UINT32_MAX},
    {255, 8, 0}, {255, 16, 0}, {255, 0x80000000u, 0},
    {255, 0x80000001u, 0}, {255, 0x01000010u, 0}, {242, 0x3f800000u, 0},
    {4, 0, 1}, {4, 0, 0x80000000u}, {4, 0x80000000u, 0x80000000u},
    {4, 0x01000010u, 0x00100001u},
}};

uint32_t widened(uint32_t bits) {
    uint32_t result = 0;
    for (unsigned shift = 0; shift < 32; shift += 4)
        if ((bits >> shift) & 15u) result |= 15u << shift;
    return result;
}

void mov(std::vector<uint32_t>& code, unsigned dst, uint32_t bits) {
    code.push_back(0xbe8003ffu | (dst << 16));
    code.push_back(bits);
}

void source(std::vector<uint32_t>& code, const Case& c) {
    if (c.operand != 4) return;
    mov(code, 4, c.lo);
    mov(code, 5, c.hi);
}

void wqm(std::vector<uint32_t>& code, const Case& c, bool wide, unsigned dst) {
    code.push_back(0xbe800000u | (dst << 16) | ((wide ? 10u : 9u) << 8) | c.operand);
    if (c.operand == 255) code.push_back(c.lo);
}

void output(std::vector<uint32_t>& code, unsigned reg) {
    code.insert(code.end(), {0x7e000200u | reg, 0xbf810000u});
}

std::vector<uint32_t> packet(const Case& c, bool wide, bool high, bool in_place = false) {
    std::vector<uint32_t> code;
    code.reserve(16);
    mov(code, 21, Adjacent);
    source(code, c);
    const unsigned dst = in_place ? 4 : 20;
    wqm(code, c, wide, dst);
    output(code, dst + high);
    return code;
}

std::vector<uint32_t> compile(const std::vector<uint32_t>& code, bool dispatcher) {
    return recompile_valu(code.data(), code.size(), 1, 0, nullptr, 0,
                          kDefaultComputePgmRsrc1, dispatcher);
}

bool no_subgroup_capability(const std::vector<uint32_t>& spirv) {
    for (size_t at = 5; at < spirv.size();) {
        const uint32_t words = spirv[at] >> 16;
        if (!words || words > spirv.size() - at) return false;
        if ((spirv[at] & 0xffffu) == Op_Capability && words == 2 &&
            spirv[at + 1] >= Cap_GroupNonUniform && spirv[at + 1] <= Cap_GroupNonUniformQuad)
            return false;
        at += words;
    }
    return true;
}

void execute_words(Checks& checks, const std::vector<uint32_t>& code, bool dispatcher,
                   const std::vector<uint32_t>& expected) {
    const auto spirv = compile(code, dispatcher);
    checks.check(!spirv.empty(), "numeric WQM compiles in the claimed route");
    if (spirv.empty()) return;
    checks.check(no_subgroup_capability(spirv), "numeric WQM requires no subgroup capability");
    const auto actual = prosper::test::run_compute(spirv, std::vector<float>(N), N, N);
    unsigned bad = 0;
    for (size_t i = 0; i < actual.size(); ++i)
        bad += i >= expected.size() || std::bit_cast<uint32_t>(actual[i]) != expected[i];
    std::printf("dispatcher=%d expected=%08x outputs=%zu bad=%u\n",
                dispatcher, expected.front(), actual.size(), bad);
    checks.check(actual.size() == N && bad == 0, "all 128 output words match the exact scalar oracle");
}

void execute(Checks& checks, const std::vector<uint32_t>& code, bool dispatcher,
             uint32_t expected) {
    execute_words(checks, code, dispatcher, std::vector<uint32_t>(N, expected));
}

void matrix(Checks& checks) {
    for (const auto& c : Cases) {
        for (bool wide : {false, true}) {
            if (wide && c.operand == 242) continue; // distinct B64 inline-float contract is not admitted
            if (c.operand == 193 || (wide && c.operand == 143)) continue; // exact legacy mask spellings
            for (bool dispatcher : {false, true}) {
                if (dispatcher && c.operand == 4) continue; // scalar placeholders are not DATA proof
                execute(checks, packet(c, wide, false), dispatcher, widened(c.lo));
                execute(checks, packet(c, wide, true), dispatcher, wide ? widened(c.hi) : Adjacent);
            }
        }
    }
}

std::vector<uint32_t> scc_packet(const Case& c, bool wide, bool edge, bool empty_exec) {
    std::vector<uint32_t> code;
    code.reserve(16);
    const bool nonzero = c.lo || (wide && c.hi);
    source(code, c);
    if (empty_exec) code.push_back(0xbefe0480u); // scalar WQM still executes under empty EXEC
    code.push_back(nonzero ? 0xbf068180u : 0xbf068080u); // deliberately opposite preceding SCC
    wqm(code, c, wide, 20);
    if (edge) code.insert(code.end(), {0xbf820001u, 0xbf800000u});
    code.push_back(0xbe9e03fdu); // s_mov_b32 s30,scc
    if (empty_exec) code.push_back(0xbefe04c1u);
    output(code, 30);
    return code;
}

void scalar_conditions(Checks& checks) {
    for (const auto& c : {Cases[0], Cases[1], Cases[4]})
        for (bool wide : {false, true})
            for (bool dispatcher : {false, true})
                for (bool empty_exec : {false, true})
                    execute(checks, scc_packet(c, wide, dispatcher, empty_exec), dispatcher,
                            (c.lo || (wide && c.hi)) ? 1u : 0u);
    // A high-only B64 result must set SCC even though its low word remains zero.
    // Raw register sources have no dispatcher provenance proof yet.
    for (const auto& c : {Cases[12], Cases[13]})
        for (bool wide : {false, true})
            for (bool empty_exec : {false, true})
                execute(checks, scc_packet(c, wide, false, empty_exec), false, wide ? 1u : 0u);
}

void lifetime_execution(Checks& checks) {
    for (bool wide : {false, true}) {
        execute(checks, packet(Cases.back(), wide, false, true), false, widened(Cases.back().lo));
        execute(checks, packet(Cases.back(), wide, true, true), false,
                wide ? widened(Cases.back().hi) : Cases.back().hi);
    }
    const std::vector<uint32_t> stale{
        0xbe94047eu,             // save full EXEC into s[20:21]
        0xbe950981u,             // numeric WQM overwrites its high physical word
        0xbefe0414u,             // restoring the old Bool must fail: no complete raw pair exists
        0x7e000281u, 0xbf810000u,
    };
    checks.check(compile(stale, false).empty(), "a high-half WQM write ends the old saved B64 mask");
    const std::vector<uint32_t> fresh{
        0xbe94047eu, 0xbe950981u, 0xbe940382u, // replace both words with real DATA
        0xbe9e1014u, 0x7e00021eu, 0xbf810000u,
    };
    execute(checks, fresh, false, 5); // popcount(2) + popcount(WQM(1))
}

void mask_consumers(Checks& checks) {
    for (const auto& c : {Case{128, 0, 0}, Case{143, 15, 0},
                          Case{193, UINT32_MAX, UINT32_MAX},
                          Case{208, 0xfffffff0u, UINT32_MAX}})
        for (bool dispatcher : {false, true}) {
            for (bool copy : {false, true})
                for (bool edge : {false, true})
                    for (bool invert : {false, true}) {
                        // The compact structurizer cannot persist this saved-mask lifetime over
                        // the branch. Exercise that edge in the existing dispatcher route instead.
                        if (edge && !dispatcher) continue;
                        std::vector<uint32_t> code;
                        source(code, c);
                        code.push_back(0x7e000287u); // initialize v0=7 before changing EXEC
                        wqm(code, c, true, 4);
                        if (copy) code.push_back(0xbe880404u); // complete DATA copy s[8:9]<-s[4:5]
                        if (edge) code.insert(code.end(), {0xbf820001u, 0xbf800000u});
                        code.push_back((invert ? 0xbefe0800u : 0xbefe0400u) | (copy ? 8u : 4u));
                        code.insert(code.end(), {0x7e000281u, 0xbefe04c1u, 0xbf810000u});
                        std::vector<uint32_t> expected(N);
                        for (uint32_t i = 0; i < N; ++i) {
                            const uint32_t lane = i & 63u;
                            const uint32_t word = widened(lane < 32 ? c.lo : c.hi);
                            const bool active = ((word >> (lane & 31u)) & 1u) != 0;
                            expected[i] = (active != invert) ? 1u : 7u;
                        }
                        execute_words(checks, code, dispatcher, expected);
                    }
            // Other former saved-mask consumers must retain their predicate view too. These are
            // already quad-closed constants: this is compatibility, not dynamic Boolean WQM.
            for (unsigned consumer = 0; consumer < 5; ++consumer) {
                std::vector<uint32_t> code{0x7e000287u};
                wqm(code, c, true, 4);
                if (consumer == 0) code.insert(code.end(), {0xd5010000u, 0x00110287u}); // cndmask
                else if (consumer == 1) code.insert(code.end(), {0xd5286a00u, 0x00110481u}); // carry-in
                else {
                    if (consumer == 2) code.insert(code.end(), {0xbf068080u, 0x85fe8004u}); // cselect EXEC
                    else code.push_back(consumer == 3 ? 0x87fec104u : 0xbe942404u); // AND / SAVEEXEC
                    code.insert(code.end(), {0x7e000281u, 0xbefe04c1u});
                }
                code.push_back(0xbf810000u);
                std::vector<uint32_t> expected(N);
                for (uint32_t i = 0; i < N; ++i) {
                    const uint32_t lane = i & 63u;
                    const uint32_t word = lane < 32 ? c.lo : c.hi;
                    const bool active = ((word >> (lane & 31u)) & 1u) != 0;
                    expected[i] = consumer == 1 ? (active ? 4u : 3u) : (active ? 1u : 7u);
                }
                execute_words(checks, code, dispatcher, expected);
            }
            const std::vector<uint32_t> direct{
                0x7e000287u, 0xbefe0a00u | c.operand,
                0x7e000281u, 0xbefe04c1u, 0xbf810000u,
            };
            std::vector<uint32_t> expected(N);
            for (uint32_t i = 0; i < N; ++i) {
                const uint32_t lane = i & 63u;
                expected[i] = (((lane < 32 ? c.lo : c.hi) >> (lane & 31u)) & 1u) ? 1u : 7u;
            }
            execute_words(checks, direct, dispatcher, expected);
        }
}

void mask_provenance_controls(Checks& checks) {
    const std::vector<uint32_t> half_overwrite{
        0xbe840ac1u, 0xbe850380u, // WQM pair then a fresh high-word scalar definition
        0xbf820001u, 0xbf800000u, 0xbefe0404u, 0xbefe04c1u, 0xbf810000u,
    };
    checks.check(compile(half_overwrite, true).empty(),
                 "an overlapping write kills the saved-mask dispatcher MUST fact");
    const std::vector<uint32_t> missing_path{
        0xbf068080u, 0xbf840001u, 0xbe840ac1u, // only one predecessor defines the pair
        0xbefe0404u, 0xbefe04c1u, 0xbf810000u,
    };
    checks.check(compile(missing_path, true).empty(),
                 "a saved-mask lifetime is a MUST fact at a dispatcher join");
    for (uint32_t operand : {128u, 193u}) {
        const std::vector<uint32_t> vertex{
            0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
            0xbe840a00u | operand, 0xbefe0404u, 0x7e000281u, 0xbefe04c1u,
            0xf80008cfu, 0x03020100u, 0xbf810000u,
        };
        checks.check(!recompile_vertex(vertex.data(), vertex.size()).empty(),
                     "vertex saved empty/full WQM masks retain compatibility");
        auto fragment = vertex;
        fragment[8] = 0xf800000fu;
        const auto spirv = recompile_fragment(fragment.data(), fragment.size());
        checks.check(!spirv.empty() && no_subgroup_capability(spirv),
                     "fragment saved empty/full WQM masks add no subgroup-width dependency");
    }
}

void provenance_controls(Checks& checks) {
    Rdna2Inst in;
    in.fmt = Rdna2Format::SOP1;
    in.opcode = 10;
    in.src[0] = {OperandKind::SGPR, 4};
    in.dst = {OperandKind::SGPR, 20};
    RegState rs;
    rs.sreg[4] = 11;
    rs.sreg[5] = 12;
    checks.check(wqm_has_scalar_numeric_source(rs, in), "complete raw pair is numeric proof");
    rs.sreg_srt[5] = 0;
    checks.check(!wqm_has_scalar_numeric_source(rs, in), "descriptor placeholder is not raw DATA");
    rs.sreg_srt.clear();
    rs.scalar_presence_has_no_placeholders = false;
    checks.check(!wqm_has_scalar_numeric_source(rs, in), "dispatcher placeholder is not raw DATA");
    rs.scalar_presence_has_no_placeholders = true;
    rs.sreg_bool[3] = 13;
    checks.check(!wqm_has_scalar_numeric_source(rs, in), "overlapping saved mask is not raw DATA");
    rs.sreg_bool.clear();
    rs.sreg_wave64_mask_half[5] = 14;
    checks.check(!wqm_has_scalar_numeric_source(rs, in), "dedicated mask half is not raw DATA");
    rs.sreg_wave64_mask_half.clear();
    rs.sreg.erase(5);
    checks.check(!wqm_has_scalar_numeric_source(rs, in), "missing source half is not fabricated as zero");
}

void overlap_controls(Checks& checks) {
    RegState rs;
    for (int root : {19, 20, 21, 22}) {
        rs.sreg_bool[root] = static_cast<uint32_t>(root);
        rs.sreg_bool_narrowed[root] = true;
        rs.sreg_srt[root] = static_cast<uint32_t>(root);
    }
    rs.sreg_bool_b32.insert(19);
    rs.sreg_wave64_mask_half[20] = 20;
    rs.sreg_wave64_mask_half[21] = 21;
    rs.sreg_wave64_mask_half_index[20] = 0;
    rs.sreg_wave64_mask_half_index[21] = 1;
    invalidate_wqm_numeric_mask_aliases(rs, 20, 2);
    checks.check(rs.sreg_bool.contains(19) && rs.sreg_bool_b32.contains(19) &&
                 rs.sreg_bool.contains(22), "adjacent non-overlapping predicates survive");
    checks.check(!rs.sreg_bool.contains(20) && !rs.sreg_bool.contains(21) &&
                 !rs.sreg_bool_narrowed.contains(20) && !rs.sreg_bool_narrowed.contains(21),
                 "both overlapping mask roots and narrowed tags expire");
    checks.check(rs.sreg_wave64_mask_half.empty() && rs.sreg_wave64_mask_half_index.empty(),
                 "overwritten dedicated mask-half values and indices expire together");
    checks.check(!rs.sreg_srt.contains(20) && !rs.sreg_srt.contains(21) &&
                 rs.sreg_srt.contains(19) && rs.sreg_srt.contains(22), "only overwritten descriptor tags expire");
    rs.sreg_bool_b32.erase(19);
    invalidate_wqm_numeric_mask_aliases(rs, 20, 1);
    checks.check(!rs.sreg_bool.contains(19), "a B32 write expires an overlapping B64 high half");
}

void rejection_controls(Checks& checks) {
    for (const auto& c : {Cases[12], Cases[13]})
        for (bool wide : {false, true})
            checks.check(compile(packet(c, wide, false), true).empty(),
                         "raw dispatcher SGPR WQM requires a separate provenance proof");
    checks.check(compile(packet(Cases[11], true, false), false).empty(),
                 "B64 inline-float operand bits are not approximated as B32");
    for (bool wide : {false, true})
        for (unsigned dst : {106u, 107u, 126u, 127u}) {
            std::vector<uint32_t> code;
            wqm(code, Cases[1], wide, dst);
            output(code, 0);
            checks.check(compile(code, false).empty(), "numeric-to-special projection is not silently admitted");
        }
    std::vector<uint32_t> unaligned;
    wqm(unaligned, Cases[1], true, 21);
    output(unaligned, 0);
    checks.check(compile(unaligned, false).empty(), "an unaligned B64 pair does not gain a Bool fallback");
    checks.check(compile(packet(Cases[1], true, false, true), false).size() > 0,
                 "ordinary aligned numeric destination remains a positive control");
}
} // namespace

int main() {
    Checks checks;
    matrix(checks);
    scalar_conditions(checks);
    lifetime_execution(checks);
    mask_consumers(checks);
    mask_provenance_controls(checks);
    provenance_controls(checks);
    overlap_controls(checks);
    rejection_controls(checks);
    return checks.failures ? 1 : 0;
}
