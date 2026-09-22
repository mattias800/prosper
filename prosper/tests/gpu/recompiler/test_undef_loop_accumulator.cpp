// test_undef_loop_accumulator — the executable fixture for #2790's dropped HDR producer.
//
// WHAT THIS PINS TODAY: that a divergent-boolean accumulator with no reaching definition on the
// loop-entry path reaches the divergent-loop emitter and is rejected AT THE ACCUMULATOR OPERAND,
// rather than being refused earlier for unsupported control flow. That distinction decides where a
// repair belongs, and two rounds of analysis on the real shader got it wrong in both directions
// before this fixture existed.
//
// It is deliberately a REJECT fixture. When the admission analysis lands, this file is where the
// positive expectation goes; until then it exists so the reject is attributed, not assumed.
//
// Provenance. Seven of the 23 dwords are byte-identical to Sonic Frontiers' own shader
// (`0x200584ca00`, submit 138828, draw 0), which is the point — a fixture that merely resembles the
// shape would not tell us the emitter takes the same route:
//
//   0xBED6047E  s_mov_b64 s[86:87], exec          the witness's pc=0280   (E := entry EXEC)
//   0x8AC47E54  s_andn2_b64 s[68:69], s[84:85], exec          pc=0328     (THE REJECT)
//   0xBEEA045C  s_mov_b64 vcc, s[92:93]                       pc=0329     (dominates the vccz)
//   0x88D4445C  s_or_b64 s[84:85], s[92:93], s[68:69]         pc=0330     (A := cond | T)
//   0xBEFE045C  s_mov_b64 exec, s[92:93]                      pc=0331
//   0x87EA5654  s_and_b64 vcc, s[84:85], s[86:87]             pc=0371     (the masked use)
//   0xBECC246A  s_and_saveexec_b64 s[76:77], vcc              pc=0373
//   0x7D0200F9,0x0686DC13  the witness's own v_cmp -> s[92:93]  pc=0325
//
// Shape preserved from the witness: NESTED loops (the real one is an inner 324..369 inside an outer
// 267..525); an EXEC-governed backedge (`s_cbranch_execnz`); an empty condition region at each
// header (`s_cbranch_execz` is the first instruction, which `detect_divergent_loops` requires for an
// execnz backedge); a VCC overwrite dominating the `s_cbranch_vccz`; and the post-loop mask by E
// before `s_and_saveexec_b64`. `s[84:85]` is never written before the loop — that is the defect.
//
// KNOWN DIVERGENCE FROM THE WITNESS, recorded rather than smoothed over: the probe reports
// `sreg=1/0` here against `sreg=1/1` on the real shader. `loop_written_regs`'s SOP2 case
// (`rdna2_emit_cfg.cpp`) defaults to one word and widens only for opcodes 0x0b/0x1f/0x21, so
// `s_or_b64` (0x11) contributes the base register alone. Why the witness also has the HIGH word
// present is not yet explained. Anything that depends on both words of the pair being placeholdered
// is therefore NOT exercised by this fixture, and must not be claimed to be.
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// pc  instruction                                     (branch targets computed, not hand-counted)
//  0  s_mov_b64  s[86:87], exec                       E := entry EXEC
//  1  s_cbranch_execz -> 16                           OUTER header, empty condition region
//  2  s_cbranch_execz -> 12                           INNER header, empty condition region
//  3  v_cmp ... -> s[92:93]                           cond  (2 dwords)
//  5  s_andn2_b64 s[68:69], s[84:85], exec            <<< the reject: s[84:85] has no definition
//  6  s_mov_b64  vcc, s[92:93]
//  7  s_or_b64   s[84:85], s[92:93], s[68:69]
//  8  s_mov_b64  exec, s[92:93]
//  9  s_cbranch_vccz -> 12
// 10  s_mov_b64  exec, s[92:93]                       EXEC writer before the backedge
// 11  s_cbranch_execnz -> 2                           INNER backedge
// 12  s_mov_b64  exec, s[86:87]
// 13  s_and_b64  vcc, s[84:85], s[86:87]              the only unmasked-looking use, masked by E
// 14  s_and_saveexec_b64 s[76:77], vcc
// 15  s_cbranch_execnz -> 1                           OUTER backedge
// 16  s_mov_b64  exec, s[86:87]                       exit
// 17  export + s_endpgm
static const uint32_t kUndefAccumulatorPs[] = {
    0xBED6047Eu, 0xBF88000Eu, 0xBF880009u, 0x7D0200F9u,
    0x0686DC13u, 0x8AC47E54u, 0xBEEA045Cu, 0x88D4445Cu,
    0xBEFE045Cu, 0xBF860002u, 0xBEFE045Cu, 0xBF89FFF6u,
    0xBEFE0456u, 0x87EA5654u, 0xBECC246Au, 0xBF89FFF1u,
    0xBEFE0456u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
    0xF800180Fu, 0x03020100u, 0xBF810000u,
};

int main() {
    printf("== test_undef_loop_accumulator ==\n");

    const uint64_t addr = 0xA1A1A1A1ull;
    const std::vector<uint32_t> frag = recompile_fragment(
        kUndefAccumulatorPs, std::size(kUndefAccumulatorPs), nullptr, nullptr,
        UINT32_MAX, nullptr, /*wave32=*/false, {RecompileDiagnosticStage::Fragment, addr});
    const std::string reason = last_terminal_reject_reason(addr);

    CHECK(frag.empty(), "today: the undef loop accumulator rejects (this flips when it is admitted)");

    // The load-bearing assertions. `frag.empty()` alone is satisfied by ANY malformed shader, which
    // is exactly how a fixture ends up proving nothing -- these name WHERE and WHY.
    CHECK(reason.find("mode=unresolved-operand") != std::string::npos,
          "...for the intended reason: an unresolved mask operand");
    CHECK(reason.find("pc=5") != std::string::npos,
          "...at the intended pc: the s_andn2_b64 reading the accumulator");
    CHECK(reason.find("words=8ac47e54") != std::string::npos,
          "...on the byte-identical instruction from the real shader");
    CHECK(reason.find("src=84(k1)") != std::string::npos,
          "...naming s84 as the unresolved source");

    // CONTROL, and the reason this fixture took several attempts. A reject is worthless as evidence
    // if the shader was refused earlier for unsupported control flow -- the repair would then belong
    // somewhere else entirely. These exclude the refusals that would mean that.
    CHECK(reason.find("execnz") == std::string::npos,
          "CONTROL: NOT refused for an unclaimed s_cbranch_execnz");
    CHECK(reason.find("divloop") == std::string::npos && reason.find("backedge") == std::string::npos,
          "CONTROL: NOT refused by divergent-loop detection -- the loops were claimed, so the "
          "failure is inside a loop body and the loop-PHI machinery is present at it");

    if (fails) printf("== FAIL: %d ==\n", fails);
    else       printf("== all passed ==\n");
    return fails ? 1 : 0;
}
