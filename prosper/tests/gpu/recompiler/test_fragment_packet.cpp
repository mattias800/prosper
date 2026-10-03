#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_packet_fixture.hpp"
#include "fixtures/fragment_packet_scalar_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {
int failures = 0, checks = 0;
void check(bool value, const std::string& name) {
    ++checks;
    if (!value) { ++failures; std::fprintf(stderr, "[FAIL] %s\n", name.c_str()); }
}
uint32_t count(const std::vector<uint32_t>& m, uint32_t op) {
    uint32_t n = 0;
    for (size_t pc = 5; pc < m.size();) {
        const uint32_t words = m[pc] >> 16;
        if (!words || words > m.size() - pc) { check(false, "module framing"); return 0; }
        n += (m[pc] & 0xffffu) == op;
        pc += words;
    }
    return n;
}
bool workgroup64(const std::vector<uint32_t>& m) {
    bool compute = false, size64 = false;
    for (size_t pc = 5; pc < m.size();) {
        const uint32_t n = m[pc] >> 16, op = m[pc] & 0xffffu;
        if (!n || n > m.size() - pc) return false;
        if (op == 15 && n >= 4) compute |= m[pc + 1] == 5; // GLCompute
        if (op == 16 && n == 6 && m[pc + 2] == 17)
            size64 |= m[pc + 3] == 64 && m[pc + 4] == 1 && m[pc + 5] == 1;
        pc += n;
    }
    return compute && size64;
}
uint32_t szi32(const std::vector<uint32_t>& m) {
    bool capability = false, mode = false;
    uint32_t entry = 0;
    for (size_t pc = 5; pc < m.size();) {
        const uint32_t n = m[pc] >> 16, op = m[pc] & 0xffffu;
        if (!n || n > m.size() - pc) return UINT32_MAX;
        if (op == 15 && n >= 4) entry = m[pc + 2];
        if (op == 17 && n == 2) capability |= m[pc + 1] == 4466;
        if (op == 16 && n == 4 && entry && m[pc + 1] == entry)
            mode |= m[pc + 2] == 4461 && m[pc + 3] == 32;
        pc += n;
    }
    return uint32_t(capability) | (uint32_t(mode) << 1);
}
void reject(prosper::gpu::FragmentInvocationPacket p, const std::string& reason,
            const std::string& name) {
    const auto result = prosper::gpu::recompile_fragment_packet(
        p, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4093});
    check(result.spirv.empty() && result.input_words.empty() && result.output_words.empty(),
          name + " transactional refusal");
    check(result.rejection == reason, name + " precise reason: " + result.rejection);
}
void sink_case(const prosper::gpu::FragmentInvocationPacket& input,
               const std::vector<uint32_t>& want, const std::string& name,
               const std::filesystem::path& directory, bool bitcount_controls = false) {
    const auto result = prosper::gpu::recompile_fragment_packet(
        input, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4093});
    check(!result.spirv.empty() && result.rejection.empty(), name + " emits actual guest");
    if (result.spirv.empty()) return;
    bpermute_oracle::Interpreter vm(result.spirv);
    const auto actual = vm.run_packet(result.input_words, result.output_words);
    check(vm.error.empty(), name + " typed actual SOURCE: " + vm.error);
    check(actual == want, name + " all logical64 raw sinks");
    if (actual.size() == want.size())
        for (size_t word = 0; word < want.size(); ++word)
            check(actual[word] == want[word], name + " word=" + std::to_string(word));
    if (bitcount_controls) {
        size_t bitcount = 0;
        uint32_t boolean_type = 0, zero = 0;
        for (size_t pc = 5; pc < result.spirv.size();) {
            const uint32_t n = result.spirv[pc] >> 16, op = result.spirv[pc] & 0xffffu;
            if (!n || n > result.spirv.size() - pc) break;
            if (op == 205 && n == 4 && !bitcount) bitcount = pc;
            if (op == 20 && n == 2) boolean_type = result.spirv[pc + 1];
            pc += n;
        }
        if (bitcount) for (size_t pc = 5; pc < result.spirv.size();) {
            const uint32_t n = result.spirv[pc] >> 16, op = result.spirv[pc] & 0xffffu;
            if (!n || n > result.spirv.size() - pc) break;
            if (op == 43 && n == 4 && result.spirv[pc + 1] == result.spirv[bitcount + 1] &&
                result.spirv[pc + 3] == 0) zero = result.spirv[pc + 2];
            pc += n;
        }
        check(bitcount && boolean_type && zero, name + " actual typed BitCount control operands");
        if (bitcount && boolean_type && zero) {
            auto wrong_type = result.spirv;
            wrong_type[bitcount + 1] = boolean_type;
            bpermute_oracle::Interpreter typed(wrong_type);
            (void)typed.run_packet(result.input_words, result.output_words);
            check(typed.error == "unsupported BitCount result", name + " BitCount type fails closed");
            // Valid typed IAdd(source,0) does not count bits. This is a live numeric
            // mutation of the emitted module, not a precomputed expected-data flip.
            auto no_count = result.spirv;
            no_count[bitcount] = (5u << 16) | 128u;
            no_count.insert(no_count.begin() + bitcount + 4, zero);
            bpermute_oracle::Interpreter changed(no_count);
            const auto changed_words = changed.run_packet(result.input_words, result.output_words);
            check(changed.error.empty() && changed_words != want,
                  name + " actual live scalar popcount omission changes raw sink");
            if (!directory.empty()) {
                std::ofstream control(directory / (name + "_oracle_nopopcount.spv"), std::ios::binary);
                control.write(reinterpret_cast<const char*>(no_count.data()),
                              static_cast<std::streamsize>(no_count.size() * 4));
                control.close();
                check(bool(control), name + " valid typed numeric mutation dump");
            }
        }
    }
    if (!directory.empty()) {
        std::ofstream file(directory / (name + ".spv"), std::ios::binary);
        file.write(reinterpret_cast<const char*>(result.spirv.data()),
                   static_cast<std::streamsize>(result.spirv.size() * 4));
        file.close();
        check(bool(file), name + " strict-validation dump");
    }
}
// Append an independently specified one-channel EXP to the existing one-record
// fixture. The scalar input is uniform; only the guest VGPR destination write is
// EXEC-predicated. Export eligibility is retained separately in the raw record.
std::vector<uint32_t> extra_scalar_record(const std::vector<uint32_t>& first,
                                        const prosper::gpu::FragmentInvocationPacket& input,
                                        uint32_t scalar) {
    std::vector<uint32_t> want(24 * 64, 0);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        std::copy_n(first.begin() + lane * 12, 12, want.begin() + lane * 24);
        const bool exec = (input.exec_mask >> lane) & 1u;
        const uint32_t header[] = {1, uint32_t(exec), input.export_enabled[lane], 0, 1, 0, 1, 1};
        std::copy(std::begin(header), std::end(header), want.begin() + lane * 24 + 12);
        want[lane * 24 + 20] = exec ? scalar : prosper::test::fragment_packet::poison_sentinel;
    }
    return want;
}
} // namespace
int main(int argc, char** argv) {
    using namespace prosper::test::fragment_packet;
    std::filesystem::path directory;
    if (argc == 3 && std::strcmp(argv[1], "--dump-directory") == 0) directory = argv[2];
    else if (argc != 1) return 2;
    namespace s = prosper::test::fragment_packet::scalar;
    uint32_t scalar_ordinal = 0;
    for (const auto& c : s::cases())
        sink_case(s::packet(c),s::expected(c),"scalar_initialization_" +
            std::to_string(scalar_ordinal++),directory,c.kind == s::Kind::Pair);
    check(scalar_ordinal == 13,"all supplied-entry/scratch CFG scalar cases compiled");
    for (bool initial_scc : {false,true}) {
        auto one_arm = s::packet({s::Kind::BothArms,initial_scc});
        one_arm.guest_code[4] = one_arm.guest_code[5] = 0xbf800000u; // exact two-word NOP twin
        reject(one_arm,"packet-sgpr-read-before-definition","one arm missing despite known SCC");
    }
    auto read_first = s::packet({});
    read_first.guest_code.insert(read_first.guest_code.begin(),0x7e02020eu); // v1 <- missing s14 before both writers
    reject(read_first,"packet-sgpr-read-before-definition","read precedes later overwrite");
    auto pair_high = s::packet({s::Kind::Pair});
    pair_high.guest_code[2] = pair_high.guest_code[3] = 0xbf800000u;
    reject(pair_high,"packet-sgpr-read-before-definition","missing physical high word before BCNT");
    auto self_pair = s::packet({s::Kind::Pair});
    self_pair.guest_code.insert(self_pair.guest_code.begin(),0xbe940414u); // s20:s21 <- its own absent old pair
    reject(self_pair,"packet-sgpr-read-before-definition","pair alias reads OLD state before write");
    auto entry_absent = s::packet({s::Kind::SuppliedEntry}); entry_absent.sgprs.clear();
    reject(entry_absent,"packet-sgpr-read-before-definition","no invented value for a real entry read");
    uint32_t ordinal = 0;
    for (uint32_t selected : {63u, 31u, 64u})
        for (uint32_t variant = 0; variant < 8; ++variant) {
            Case c;
            c.selected_lane = selected;
            c.inactive_source = variant >= 1 && variant <= 3;
            c.leave_source_inactive = variant == 2 || variant == 3;
            c.second_export = variant == 3 || variant == 7;
            c.scalar_selector = variant == 6;
            c.data_base = variant == 1 ? 0x7f800001u : 0x51000000u + ordinal * 0x1000;
            auto input = packet(c);
            if (variant == 4) input.exec_mask = uint64_t(1) << 63;
            if (variant == 5) input.exec_mask = 0xffffffffu;
            input.export_enabled[7] = 0; // export eligibility never changes wave participation
            std::vector<prosper::gpu::Rdna2Inst> decoded;
            prosper::gpu::rdna2_walk(input.guest_code.data(), input.guest_code.size(), decoded);
            uint32_t compares = 0, reads = 0, exports = 0;
            for (const auto& i : decoded) {
                if (i.fmt == prosper::gpu::Rdna2Format::VOPC) {
                    ++compares;
                    check(i.opcode == 0xc2 || i.opcode == 0xd5, "actual uint VOPC opcode");
                    check(i.src[1].kind == prosper::gpu::OperandKind::VGPR && i.src[1].value == 0,
                          "actual packet lane operand");
                }
                if (i.fmt == prosper::gpu::Rdna2Format::VOP3) {
                    ++reads;
                    check(i.opcode == 0x360 &&
                          i.src[0].kind == prosper::gpu::OperandKind::VGPR && i.src[0].value == 8 &&
                          i.src[1].kind == (c.scalar_selector ? prosper::gpu::OperandKind::SGPR
                                                             : prosper::gpu::OperandKind::InlineInt) &&
                          i.src[1].value == (c.scalar_selector ? 22 : 40),
                          "actual READLANE40 decode");
                }
                exports += i.fmt == prosper::gpu::Rdna2Format::EXP;
            }
            check(compares == (c.inactive_source ? 2u : 1u) &&
                  reads == (c.second_export ? 2u : 1u) && exports == reads,
                  "actual guest static operation inventory");
            const auto result = prosper::gpu::recompile_fragment_packet(
                input, {prosper::gpu::RecompileDiagnosticStage::Fragment, 0x4093});
            const std::string name = "packet_" + std::to_string(ordinal++);
            check(!result.spirv.empty() && result.rejection.empty(), name + " actual guest emitted");
            if (result.spirv.empty()) continue;
            check(result.exports_per_lane == exports && result.output_words.size() == exports * 12 * 64,
                  name + " owned output extent");
            check(count(result.spirv, 251) && count(result.spirv, 224) >= 2,
                  name + " actual common dispatcher/rendezvous");
            check(workgroup64(result.spirv), name + " actual physical64 workgroup shell");
            check(count(result.spirv, 124) == 0, name + " raw uint input/output, no float transport");
            check(count(result.spirv, 252) == 0, name + " no physical worker kill");
            for (uint32_t op = 333; op <= 366; ++op)
                check(count(result.spirv, op) == 0, name + " no host subgroup " + std::to_string(op));
            bpermute_oracle::Interpreter vm(result.spirv);
            const auto actual = vm.run_packet(result.input_words, result.output_words);
            const auto want = expected(c, input);
            check(vm.error.empty(), name + " typed actual SOURCE: " + vm.error);
            check(actual == want, name + " all raw EXP records match logical64 oracle");
            if (actual.size() == want.size()) {
                auto wrong = actual;
                wrong[63 * exports * 12 + 10] ^= 1;
                check(wrong != want, name + " high-lane sink load-bearing");
                for (uint32_t lane = 0; lane < 64; ++lane)
                    for (uint32_t word = 0; word < exports * 12; ++word)
                        check(actual[lane * exports * 12 + word] == want[lane * exports * 12 + word],
                              name + " lane=" + std::to_string(lane) + " word=" + std::to_string(word));
            }
            if (!directory.empty()) {
                std::ofstream file(directory / (name + ".spv"), std::ios::binary);
                file.write(reinterpret_cast<const char*>(result.spirv.data()),
                           static_cast<std::streamsize>(result.spirv.size() * 4));
                file.close();
                check(bool(file), name + " strict-validation dump");
            }
        }
    auto clean = packet({});
    // FF1 scans the full saved logical mask and preserves SCC established by
    // BCNT. In particular, an empty mask returns -1 but must NOT set SCC true.
    // Both a real SCC branch and an independent CSELECT make that preservation
    // observable, rather than checking only the FF1 marker/instruction count.
    for (uint32_t selected : {63u, 31u, 64u}) {
        Case c; c.selected_lane = selected;
        auto p = packet(c);
        p.scc = selected == 64; // deliberately opposite to BCNT's new SCC
        p.sgprs.emplace_back(13, poison_sentinel);
        p.sgprs.emplace_back(24, branch_true); p.sgprs.emplace_back(25, branch_false);
        const auto count_at = std::find(p.guest_code.begin(), p.guest_code.end(), 0xbe8c1014u);
        check(count_at != p.guest_code.end(), "FF1 fixture retains saved-mask BCNT");
        if (count_at == p.guest_code.end()) continue;
        p.guest_code.insert(count_at + 1, {0xbe8c1414u, 0x850d1918u}); // FF1 s12,s20; CSELECT s13,s24,s25
        const auto branch_at = std::find(p.guest_code.begin(), p.guest_code.end(), 0xbf860002u);
        check(branch_at != p.guest_code.end(), "FF1 fixture retains real conditional branch");
        if (branch_at == p.guest_code.end()) continue;
        *branch_at = 0xbf840002u; // SCCZ instead of VCCZ, same paired forward destination
        p.guest_code.insert(p.guest_code.end() - 1, {0x7e0a020du, 0xf8001801u, 5u});
        auto first = expected(c, p);
        for (uint32_t lane = 0; lane < 64; ++lane)
            first[lane * 12 + 8] = selected < 64 ? selected : UINT32_MAX;
        sink_case(p, extra_scalar_record(first, p, selected < 64 ? branch_true : branch_false),
                  "ff1_scc_" + std::to_string(selected), directory);
    }
    // Read the supplied initial VCC before the compare replaces it, including
    // a high-half bit belonging to an EXEC-off lane. Scalar mask reduction reads
    // the full VCC data, not the current EXEC or the later compare result.
    for (uint64_t initial : {uint64_t(0), uint64_t(1), uint64_t(1) << 40,
                             uint64_t(1) << 63, (uint64_t(1) << 63) | 1}) {
        auto p = clean; p.vcc_mask = initial;
        if (initial == (uint64_t(1) << 40)) p.exec_mask &= ~(uint64_t(1) << 40);
        for (uint32_t reg : {26u, 27u, 28u}) p.sgprs.emplace_back(reg, poison_sentinel);
        p.guest_code.insert(p.guest_code.begin(), {0xbe9a046au, 0xbe9c101au});
        p.guest_code.insert(p.guest_code.end() - 1, {0x7e0a021cu, 0xf8001801u, 5u});
        uint32_t count = 0;
        for (uint32_t lane = 0; lane < 64; ++lane) count += (initial >> lane) & 1u;
        sink_case(p, extra_scalar_record(expected({}, p), p, count),
                  "initial_vcc_" + std::to_string(initial), directory);
    }
    // Ordinary supplied scalar-pair BCNT takes a different emitted numeric
    // route from a saved Boolean wave mask. Test its raw result, SCC-dependent
    // select/branch, then an independent ADD carry and select through live EXPs.
    for (uint64_t raw : {uint64_t(0), uint64_t(1) << 63, uint64_t(0x8000000180000001ull)})
        for (bool carry : {false, true}) {
            auto p = clean;
            p.sgprs = {{12, poison_sentinel}, {14, poison_sentinel}, {15, branch_false},
                {16, poison_sentinel}, {19, poison_sentinel}, {20, uint32_t(raw)},
                {21, uint32_t(raw >> 32)}, {24, branch_true}, {25, branch_false},
                {26, carry ? UINT32_MAX : 3u}, {27, carry ? 2u : 4u}};
            p.guest_code = {0xbe8c1014u, 0x850e1918u, 0xbf840002u};
            smov(p.guest_code, 15, branch_true);
            p.guest_code.insert(p.guest_code.end(), {0x80131b1au, 0x85101918u});
            vmov(p.guest_code, 1, 12); vmov(p.guest_code, 2, 14);
            vmov(p.guest_code, 3, 15); vmov(p.guest_code, 8, 19);
            exp(p.guest_code, 15, 0x08030201u);
            vmov(p.guest_code, 5, 16); exp(p.guest_code, 1, 5);
            p.guest_code.push_back(0xbf810000u);
            uint32_t bits = 0;
            for (uint32_t position = 0; position < 64; ++position) bits += (raw >> position) & 1u;
            std::vector<uint32_t> first(12 * 64);
            for (uint32_t lane = 0; lane < 64; ++lane) {
                const uint32_t record[] = {1, 1, 1, 0, 15, 0, 1, 1,
                    bits, raw ? branch_true : branch_false, raw ? branch_true : branch_false,
                    carry ? 1u : 7u};
                std::copy(std::begin(record), std::end(record), first.begin() + lane * 12);
            }
            sink_case(p, extra_scalar_record(first, p, carry ? branch_true : branch_false),
                "scalar_bcnt_add_" + std::to_string(raw) + "_" + std::to_string(carry), directory,
                raw == 0x8000000180000001ull && !carry);
        }
    // Change the ACTUAL decoded operand fields while keeping both register banks
    // supplied. A missing-register refusal must not conceal bank aliasing or an
    // inadmissible varying selector flowing into the scalar/PC state.
    for (uint32_t arm = 0; arm < 7; ++arm) {
        auto p = clean;
        p.sgprs.emplace_back(8, 0x87654321u); // not the supplied VGPR8 values
        const auto read = std::find(p.guest_code.begin(), p.guest_code.end(), 0xd760000fu);
        check(read != p.guest_code.end(), "operand-kind control targets actual READLANE");
        if (read == p.guest_code.end()) continue;
        const size_t pc = static_cast<size_t>(read - p.guest_code.begin());
        const uint32_t source = arm == 0 ? 8u : arm == 6 ? 254u : 264u;
        const uint32_t selector = arm == 1 ? 256u : arm == 2 ? 192u :
                                  arm == 3 ? 193u : arm == 4 ? 240u : arm == 5 ? 255u : 168u;
        p.guest_code[pc + 1] = source | (selector << 9);
        if (arm == 5) p.guest_code.insert(p.guest_code.begin() + pc + 2, 40u);
        const auto decoded = prosper::gpu::rdna2_decode_one(p.guest_code.data() + pc,
                                                           p.guest_code.size() - pc);
        check(decoded.fmt == prosper::gpu::Rdna2Format::VOP3 && decoded.opcode == 0x360 &&
              decoded.src[0].kind == (arm == 0 ? prosper::gpu::OperandKind::SGPR
                  : arm == 6 ? prosper::gpu::OperandKind::Special : prosper::gpu::OperandKind::VGPR) &&
              decoded.src[1].kind == (arm == 1 ? prosper::gpu::OperandKind::VGPR
                  : arm == 4 ? prosper::gpu::OperandKind::InlineFloat
                  : arm == 5 ? prosper::gpu::OperandKind::Literal
                             : prosper::gpu::OperandKind::InlineInt),
              "actual unsupported READLANE operand kinds retained by decoder");
        reject(p, arm == 0 || arm == 6 ? "packet-readlane-source-kind-unimplemented"
                          : "packet-readlane-selector-kind-unimplemented",
               "READLANE bank/domain control " + std::to_string(arm));
    }
    // A saved mask now defines its physical words, not the supplied initial poison. Keep both
    // the zero low half (bit63) and nonzero low half (bit31) live in EXP beside the count twin.
    for (uint32_t selected : {31u, 63u})
        for (uint32_t source : {12u, 20u}) {
            Case c;
            c.selected_lane = selected;
            auto p = packet(c);
            const auto first_exp = std::find(p.guest_code.begin(), p.guest_code.end(), 0xf800180fu);
            check(first_exp != p.guest_code.end(), "live count/mask control retains EXP fixture");
            if (first_exp == p.guest_code.end()) continue;
            const size_t pc = static_cast<size_t>(first_exp - p.guest_code.begin());
            p.guest_code.insert(p.guest_code.begin() + pc,
                                0x7e0a0200u | source);   // v_mov v5,sSOURCE
            p.guest_code[pc + 2] = (p.guest_code[pc + 2] & ~0xffu) | 5u;   // live MRT channel0
            auto want = expected(c, p);
            if (source == 20)
                for (uint32_t lane = 0; lane < 64; ++lane)
                    want[lane * 12 + 8] = uint32_t(uint64_t(1) << selected);
            sink_case(p, want,
                      "saved_mask_word_echo_" + std::to_string(selected) + "_" +
                          std::to_string(source),
                      directory);
        }
    auto missing = clean;
    missing.slots_available[63] = false;
    reject(missing, "packet-invocation-state-unavailable", "missing high-half slot");
    missing = clean; missing.mask_state_available = false;
    reject(missing, "packet-entry-exec-unavailable", "missing demanded initial EXEC");
    missing = clean;
    missing.vgprs.erase(missing.vgprs.begin() + 5);
    {
        const auto compiled = prosper::gpu::recompile_fragment_packet(missing);
        check(!compiled.spirv.empty(),
              "missing READLANE source retains runtime validity obligation");
        if (!compiled.spirv.empty()) {
            bpermute_oracle::Interpreter vm(compiled.spirv);
            const auto words = vm.run_packet(compiled.input_words, compiled.output_words);
            const auto result =
                prosper::gpu::decode_fragment_packet(compiled, words, vm.error.empty());
            check(vm.error.empty(),
                  "missing READLANE source completes actual SOURCE without split barriers");
            check(result.exports.empty() &&
                      result.rejection == "packet-vgpr-selected-peer-unavailable" &&
                      result.reg == 8 && result.kind == 3,
                  "missing selected source cannot publish fabricated zero");
        }
    }
    missing = s::packet({s::Kind::Readlane}); missing.sgprs.clear();
    reject(missing, "packet-sgpr-read-before-definition", "missing supplied scalar READLANE selector");
    missing = clean; missing.export_enabled[40] = 2;
    reject(missing, "packet-launch-state-invalid", "invalid eligibility");
    missing = clean; missing.vgprs.push_back(missing.vgprs.front());
    reject(missing, "packet-vgpr-input-invalid", "duplicate VGPR column");
    missing = clean; missing.guest_code.insert(missing.guest_code.begin(), 0xbefe0a7eu);
    reject(missing, "packet-quad-topology-unavailable", "WQM needs supplied topology");
    for (const uint32_t op : {0x13u, 0x15u, 0x24u, 0x25u}) {
        missing = clean;
        std::vector<uint32_t> mode = op < 0x20
            ? std::vector<uint32_t>{0xb0000000u | (op << 23) | 1u}
            : std::vector<uint32_t>{0xbf800000u | (op << 16)};
        if (op == 0x15) mode.push_back(0);
        missing.guest_code.insert(missing.guest_code.begin(), mode.begin(), mode.end());
        reject(missing, "packet-mode-write", "MODE write " + std::to_string(op));
    }
    missing = clean; missing.guest_code.insert(missing.guest_code.begin(), 0xc8020002u);
    reject(missing, "packet-interpolation-unavailable", "no guessed interpolation");
    missing = clean;
    missing.guest_code.insert(missing.guest_code.begin(), {0xf0800f08u, 0x00820000u});
    reject(missing, "packet-image-derivative-or-effect-unimplemented", "no implicit LOD0 substitution");
    missing = clean;
    missing.guest_code.insert(missing.guest_code.begin(), {0xd8340000u, 0x00000302u});
    reject(missing, "packet-ds-op-unimplemented", "no unowned LDS/effect");
    missing = clean;
    missing.guest_code.insert(missing.guest_code.begin(), {0xf4000000u, 0xfa000004u});
    reject(missing, "packet-memory-effect-unimplemented", "no unowned scalar buffer read");
    missing = clean;
    missing.guest_code.insert(missing.guest_code.begin(), {0xd760800fu, 264u | (168u << 9)});
    reject(missing, "packet-modifier-unimplemented", "no silently lost READLANE modifier");
    missing = clean;
    *std::find(missing.guest_code.begin(), missing.guest_code.end(), 0xf800180fu) |= 0x400u;
    reject(missing, "packet-compressed-export-unimplemented", "no guessed compressed export");
    // Inventory all owned words before emission, including a genuinely branch-skipped instruction.
    for (bool unsupported : {false, true}) {
        auto p = clean;
        p.guest_code.insert(p.guest_code.begin(),
                            {0xbf820001u, unsupported ? 0xc8020002u : 0xbf800000u});
        if (unsupported) reject(p, "packet-interpolation-unavailable", "dead interpolation still refuses");
        else {
            const auto r = prosper::gpu::recompile_fragment_packet(p);
            check(!r.spirv.empty() && r.rejection.empty(), "matching skipped NOP control emits");
            if (!r.spirv.empty()) {
                bpermute_oracle::Interpreter vm(r.spirv);
                const auto actual = vm.run_packet(r.input_words, r.output_words);
                check(vm.error.empty() && actual == expected({}, p),
                      "matching forward branch control retains actual uint sinks");
            }
        }
    }
    missing = clean; missing.guest_code.insert(missing.guest_code.begin(), 0xbefc0380u);
    reject(missing, "packet-scalar-destination-unimplemented", "no guessed M0 destination");
    missing = clean; missing.guest_code.insert(missing.guest_code.begin(), 0xbe940415u);
    reject(missing, "packet-scalar-pair-input-invalid", "unaligned scalar pair");
    missing = clean;
    missing.guest_code.insert(missing.guest_code.begin(), {0x7c0400ffu, 0u});
    reject(missing, "packet-fp-or-compare-unimplemented", "no guest FP mode substitution");
    missing = clean; missing.guest_code.insert(missing.guest_code.begin(), 0xbf8200ffu);
    reject(missing, "packet-branch-target-invalid", "no guessed branch exit");
    missing = clean; missing.guest_code.insert(missing.guest_code.begin(), 0xbf82ffffu);
    reject(missing, "packet-backedge-unimplemented", "no repeated export overwrite");
    missing = clean; missing.guest_code.pop_back();
    reject(missing, "packet-code-not-one-complete-program", "unterminated input");
    {
        prosper::gpu::ComputeTripBoundSettings settings;
        settings.bound = 1;
        prosper::gpu::TripBoundOperation pin(settings);
        reject(clean, "packet-diagnostic-trip-bound-unimplemented", "no internal GDS/lossy trip cap");
    }
    // The small packet-owned launch carrier is not live launch extraction or FP arithmetic.
    // Known-clear is distinct from unavailable, but canonical facts cannot change integer SOURCE.
    const auto integer_baseline = prosper::gpu::recompile_fragment_packet(clean);
    for (uint32_t flags = 0; flags < 5; ++flags) {
        auto p = clean;
        p.float_flags = {flags != 4, flags != 4 && bool(flags & 1u),
                        flags != 4 && bool(flags & 2u)};
        const auto r = prosper::gpu::recompile_fragment_packet(p);
        check(!r.spirv.empty() && r.rejection.empty(), "canonical packet launch flags emit");
        check(r.spirv == integer_baseline.spirv && r.input_words == integer_baseline.input_words &&
              r.output_words == integer_baseline.output_words,
              "known/unknown flag carrier does not change integer SOURCE or owned buffers");
    }
    for (uint32_t flags : {1u, 2u, 3u}) {
        missing = clean; missing.float_flags = {false, bool(flags & 1u), bool(flags & 2u)};
        reject(missing, "packet-launch-state-invalid", "unavailable flag carrier cannot invent bits");
    }
    // The new entry inherits the same actual declaration gate through begin(), not a second
    // inferred device verdict. CPU publication here is structural, not a measured device claim.
    for (uint32_t arm = 0; arm < 4; ++arm) {
        prosper::gpu::reset_float_controls_support_for_test();
        if (arm != 3) prosper::gpu::publish_float_controls_support(arm != 1, arm != 2);
        const auto m = prosper::gpu::recompile_fragment_packet(clean);
        check(!m.spirv.empty(), "packet float-controls gate emits " + std::to_string(arm));
        check(szi32(m.spirv) == (arm == 0 ? 3u : 0u),
              "packet actual declaration follows published gate " + std::to_string(arm));
        if (arm != 0) check(count(m.spirv, 10) == 0,
                            "packet unavailable gate does not name an unsupported extension");
    }
    prosper::gpu::reset_float_controls_support_for_test();
    std::printf("fragment_packet: checks=%d failures=%d (owned fixture packet, no raster claim)\n",
                checks, failures);
    return failures ? 1 : 0;
}
