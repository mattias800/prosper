// #4086: unavailable launch inputs refuse only when the modifier needs them.
// Project-owned programs; no Vulkan device, host capability guess, or numeric-boundary claim.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/diagnostics/fragment_arithmetic.hpp"
#include <cstdio>
#include <string>
#include <vector>
using namespace prosper::gpu;
namespace {
unsigned checks = 0, failures = 0, calls = 0;
void check(bool value, const std::string& name) {
    ++checks;
    if (!value) { ++failures; std::printf("[FAIL] %s\n", name.c_str()); }
}
enum class Path { Vop1, Vop2, Vop3, F16 };
std::vector<uint32_t> program(Path path, unsigned omod, bool clamp) {
    std::vector<uint32_t> w;
    const bool half = path == Path::Vop1 || path == Path::F16;
    const auto mov = [&](unsigned dst, uint32_t value) {
        w.insert(w.end(), {0x7e0002ffu | (dst << 17), value});
    };
    mov(1, half ? 0x3c00u : 0x3f800000u);
    mov(2, path == Path::F16 ? 0x3c00u : 0x3f800000u);
    mov(3, 0); mov(4, 0x35550000u);
    std::vector<uint32_t> packet;
    unsigned opcode = 0x108u;
    Rdna2Format format = Rdna2Format::VOP3;
    if (path == Path::Vop1) {
        opcode = 0x0bu; format = Rdna2Format::VOP1;
        packet = {0x7e000000u | (4u << 17) | (opcode << 9) | 0xf9u,
                  1u | (6u << 8) | (6u << 16) | (unsigned(clamp) << 13) | (omod << 14)};
    } else if (path == Path::Vop2) {
        opcode = 8; format = Rdna2Format::VOP2;
        packet = {(opcode << 25) | (4u << 17) | (2u << 9) | 0xf9u,
                  1u | (6u << 8) | (6u << 16) | (6u << 24) | (unsigned(clamp) << 13) | (omod << 14)};
    } else {
        if (path == Path::F16) opcode = 0x34bu;
        packet = {0xd4000000u | (opcode << 16) | 4u | (unsigned(clamp) << 15),
                  257u | (258u << 9) | (259u << 18) | (omod << 27)};
    }
    const auto decoded = rdna2_decode_one(packet.data(), packet.size());
    check(decoded.fmt == format && decoded.opcode == opcode && decoded.omod == omod &&
          decoded.clamp == clamp && decoded.dst.kind == OperandKind::VGPR && decoded.dst.value == 4 &&
          decoded.src[0].kind == OperandKind::VGPR && decoded.src[0].value == 1 &&
          !decoded.has_modifier, "actual positive/reject instruction decoder control");
    w.insert(w.end(), packet.begin(), packet.end());
    w.push_back(0x7e000000u | (5u << 17) | ((path == Path::F16 ? 0x0bu : 1u) << 9) | 260u);
    mov(6, 0x3e800000u); mov(7, 0xbf000000u); mov(8, 0x3f800000u);
    w.insert(w.end(), {0xf800180fu, 0x08070605u, 0xbf810000u});
    return w;
}
std::vector<uint32_t> compile(Path path, unsigned wave, unsigned omod, bool clamp,
                              FragmentFloatFlags flags, FragmentFloatMode mode,
                              const char* rejection = nullptr) {
    const auto raw = program(path, omod, clamp);
    const uint64_t address = 0x4086a000ull + ++calls * 0x100u;
    FragmentArithmeticObservation observation;
    const auto source = recompile_fragment(raw.data(), raw.size(), nullptr, nullptr, UINT32_MAX,
        nullptr, wave == 32, {RecompileDiagnosticStage::Fragment, address}, mode, &observation, {}, flags);
    // The public direct API returns the observation to this owner rather than announcing twice.
    observe_fragment_arithmetic(observation, address, !source.empty());
    check(source.empty() == (rejection != nullptr), "modifier admission uses actual required launch inputs");
    if (rejection) {
        const auto reason = last_terminal_reject_reason(address);
        check(reason.find(rejection) != std::string::npos,
              std::string("named terminal reason: ") + rejection + " actual=" + reason);
        check(reason.find("pc=") != std::string::npos && reason.find("result_width=") != std::string::npos,
              "missing launch diagnostic retains actual site and guest result width");
    } else {
        const bool active = omod && flags.available && !flags.ieee_mode && mode.available &&
            !(path == Path::F16 ? mode.preserves_f16_outputs() : mode.preserves_f32_outputs());
        const auto family = path == Path::F16 ? FragmentArithmeticFamily::OutputModifierF16
                                              : FragmentArithmeticFamily::OutputModifierF32;
        const uint8_t expected = active ? uint8_t(1u << static_cast<uint8_t>(family)) : 0;
        check((observation.families & 12u) == expected,
              "only actual active modifiers retain the result-width unverified-arithmetic observation");
        if (active) {
            bool found = false;
            for (unsigned i = 0; i < observation.site_count; ++i)
                found |= observation.sites[i].family == family && observation.sites[i].pc == 8;
            check(found, "unverified modifier records the actual decoded guest PC");
        }
    }
    return source;
}
} // namespace
int main() {
    for (Path path : {Path::Vop1, Path::Vop2, Path::Vop3, Path::F16}) for (unsigned wave : {32u, 64u}) {
        const uint8_t preserved = path == Path::F16 ? 0x80u : 0x20u;
        const uint8_t wrong_width = path == Path::F16 ? 0x20u : 0x80u;
        compile(path, wave, 1, false, {}, {true, 0}, "fragment-float-flags-unavailable");
        compile(path, wave, 1, false, {true, false, true}, {}, "fragment-float-output-mode-unavailable");
        compile(path, wave, 1, false, {}, {}, "fragment-float-flags-unavailable");
        const auto ieee_known = compile(path, wave, 1, false, {true, true, true}, {true, 0});
        check(compile(path, wave, 1, false, {true, true, true}, {}) == ieee_known,
              "observed IEEE independently disables OMOD despite missing MODE");
        check(compile(path, wave, 1, false, {}, {true, preserved}) == ieee_known,
              "observed result-width output preservation independently disables OMOD despite missing flags");
        compile(path, wave, 1, false, {}, {true, wrong_width}, "fragment-float-flags-unavailable");
        compile(path, wave, 1, true, {}, {true, preserved}, "fragment-float-flags-unavailable");
        compile(path, wave, 0, true, {}, {true, 0}, "fragment-float-flags-unavailable");
        compile(path, wave, 0, true, {true, false, true}, {}); // CLAMP does not require denorm MODE
        compile(path, wave, 0, false, {}, {}); // no modifier means no new launch-input requirement
        compile(path, wave, 1, true, {true, false, true}, {true, 0}); // paired positive of reject program
    }
    check(calls == 96, "all four emitters, both widths, twelve actual launch controls compiled");
    std::printf("fragment_float_modifier_inputs: %u calls, %u checks, %u failures\n", calls, checks, failures);
    return failures ? 1 : 0;
}
