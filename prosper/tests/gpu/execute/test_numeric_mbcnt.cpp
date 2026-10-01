// Numeric MBCNT counts source-word bits below the physical lane, even if those lanes do not
// participate in EXEC. It is not a population scan. AMD RDNA2 ISA 70648, section 12.12.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "fixtures/compute_runner.h"
#include <array>
#include <bit>
#include <cstdio>
#include <map>
#include <vector>

using namespace prosper::gpu;

// Hardware results alone need not expose an undefined shift in an unselected OpSelect operand.
// Constrain the MBCNT prefix shift, not unrelated shifts emitted by the CFG dispatcher.
bool numeric_shifts_are_bounded(const std::vector<uint32_t>& module) {
    std::map<uint32_t, uint32_t> constants;
    struct Definition { uint32_t op; std::vector<uint32_t> args; };
    std::map<uint32_t, Definition> definitions;
    unsigned prefixes = 0;
    for (size_t i = 5; i < module.size();) {
        const uint32_t words = module[i] >> 16, op = module[i] & 0xffffu;
        if (!words || words > module.size() - i) return false;
        if (op == Op_Constant && words == 4) constants[module[i + 2]] = module[i + 3];
        if ((op == Op_BitwiseAnd || op == Op_ISub || op == Op_ShiftLeftLogical) && words == 5)
            definitions[module[i + 2]] = {op, {module[i + 3], module[i + 4]}};
        if (op == Op_Select && words == 6)
            definitions[module[i + 2]] = {op, {module[i + 3], module[i + 4], module[i + 5]}};
        if (op == Op_BitCount && words == 4) {
            const auto masked = definitions.find(module[i + 3]);
            if (masked == definitions.end() || masked->second.op != Op_BitwiseAnd) return false;
            bool bounded_prefix = false;
            for (uint32_t operand : masked->second.args) {
                const auto select = definitions.find(operand);
                if (select == definitions.end() || select->second.op != Op_Select) continue;
                const auto full = constants.find(select->second.args[1]);
                const auto subtract = definitions.find(select->second.args[2]);
                if (full == constants.end() || full->second != UINT32_MAX ||
                    subtract == definitions.end() || subtract->second.op != Op_ISub) continue;
                const auto one = constants.find(subtract->second.args[1]);
                const auto shift = definitions.find(subtract->second.args[0]);
                if (one == constants.end() || one->second != 1 ||
                    shift == definitions.end() || shift->second.op != Op_ShiftLeftLogical) continue;
                const auto amount = definitions.find(shift->second.args[1]);
                if (amount == definitions.end() || amount->second.op != Op_BitwiseAnd) continue;
                for (uint32_t term : amount->second.args) {
                    const auto constant = constants.find(term);
                    bounded_prefix |= constant != constants.end() && constant->second == 31;
                }
            }
            if (!bounded_prefix) return false;
            ++prefixes;
        }
        i += words;
    }
    return prefixes == 1;
}

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char* name) {
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", name);
        if (!ok) ++failures;
    };
    enum class Source { Inline, InlineFloat, Literal, Scalar, Vector };
    struct Case { Source source; uint32_t mask; int inline_value; };
    const Case cases[] = {
        {Source::Inline, 0, 0}, {Source::Inline, 1, 1}, {Source::Inline, 3, 3},
        {Source::Inline, 15, 15}, {Source::Inline, 0xfffffffeu, -2},
        {Source::InlineFloat, 0x3f800000u, 0}, // 1.0 operand contributes its raw bits
        {Source::Literal, 0x80000001u, 0}, {Source::Literal, 0x55555555u, 0},
        {Source::Scalar, 0xaaaaaaaau, 0}, {Source::Vector, 0, 0},
    };
    constexpr uint32_t N = 128;
    std::vector<float> input(N);
    for (uint32_t i = 0; i < N; ++i) input[i] = static_cast<float>((i % 11) * 3 + 1);
    for (const auto& c : cases) {
        for (bool hi : {false, true}) {
            for (bool dispatcher : {false, true}) {
                // Raw SGPR words still use the existing dispatcher event route. This change does
                // not infer a numeric lifetime from a dispatcher scalar placeholder.
                if (dispatcher && c.source == Source::Scalar) continue;
                std::vector<uint32_t> code;
                code.reserve(12);
                uint32_t operand = 0;
                if (c.source == Source::Inline)
                    operand = c.inline_value >= 0 ? 128u + c.inline_value
                                                  : 192u - c.inline_value;
                else if (c.source == Source::InlineFloat) operand = 242;
                else if (c.source == Source::Literal) operand = 255;
                else if (c.source == Source::Scalar) {
                    code.insert(code.end(), {0xbe8403ffu, c.mask}); // s_mov_b32 s4,literal
                    operand = 4;
                } else {
                    code.push_back(0x7e020f00u); // v_cvt_u32_f32 v1,v0: distinct mask per lane
                    operand = 257;
                }
                code.insert(code.end(), {hi ? 0xd7660000u : 0xd7650000u,
                                         operand | (133u << 9)}); // MBCNT v0,source,5
                if (c.source == Source::Literal) code.push_back(c.mask);
                code.insert(code.end(), {0x7e000d00u, 0xbf810000u}); // cvt_f32_u32 v0,v0; end
                const auto spv = recompile_valu(code.data(), code.size(), 1, 0,
                    nullptr, 0, kDefaultComputePgmRsrc1, dispatcher);
                check(!spv.empty(), "numeric source compiles in its claimed route");
                if (spv.empty()) continue;
                check(numeric_shifts_are_bounded(spv),
                      "the numeric prefix shift has a structurally bounded amount");
                const auto output = prosper::test::run_compute(spv, input, N, N);
                unsigned bad = 0;
                if (output.size() == N) {
                    for (uint32_t i = 0; i < N; ++i) {
                        const unsigned lane = i & 63u;
                        const unsigned width = hi ? (lane > 32 ? lane - 32 : 0)
                                                  : (lane < 32 ? lane : 32);
                        const uint32_t prefix = width == 32 ? UINT32_MAX : (1u << width) - 1;
                        const uint32_t mask = c.source == Source::Vector
                            ? static_cast<uint32_t>(input[i]) : c.mask;
                        const float expected = static_cast<float>(5 + std::popcount(mask & prefix));
                        if (output[i] != expected) ++bad;
                    }
                }
                std::printf("source=%u hi=%d dispatcher=%d bad=%u outputs=%zu\n",
                    static_cast<unsigned>(c.source), hi, dispatcher, bad, output.size());
                check(output.size() == N && bad == 0,
                      "every lane counts its own source word under the physical LO/HI prefix");
            }
        }
    }

    // Explicit EXEC divergence must not make a numeric source mask count only active lanes.
    // Inactive destination lanes keep their previous VGPR value.
    const uint32_t divergent[] = {
        0x7e020f00u,             // v_cvt_u32_f32 v1,v0
        0x7e0402a5u,            // v_mov_b32 v2,37
        0xd4c1006au, 0x00012101u, // v_cmp_lt_u32_e64 vcc_lo,v1,16
        0xbe88246au,            // s_and_saveexec_b64 s[8:9],vcc
        0xd7650002u, 0x00020283u, // v_mbcnt_lo_u32_b32 v2,3,v1
        0xbefe0408u,            // s_mov_b64 exec,s[8:9]
        0x7e000d02u, 0xbf810000u,
    };
    for (bool dispatcher : {false, true}) {
        const auto spv = recompile_valu(divergent, std::size(divergent), 1, 0,
            nullptr, 0, kDefaultComputePgmRsrc1, dispatcher);
        check(!spv.empty(), "numeric MBCNT under narrowed EXEC compiles");
        if (spv.empty()) continue;
        const auto output = prosper::test::run_compute(spv, input, N, N);
        unsigned bad = 0;
        for (unsigned i = 0; i < output.size() && i < N; ++i) {
            const unsigned lane = i & 63;
            const unsigned count = lane == 0 ? 0 : lane == 1 ? 1 : 2;
            const float expected = input[i] < 16 ? input[i] + count : 37;
            if (output[i] != expected) ++bad;
        }
        std::printf("divergent dispatcher=%d bad=%u outputs=%zu\n", dispatcher, bad, output.size());
        check(output.size() == N && bad == 0,
              "active destinations use physical bits; inactive destinations preserve their VGPR");
    }

    // The overflowing LO prefix is exactly 32 at every lane >=32. Losing just one bit turns the
    // expected zero into a huge float, so this arm remains discriminating despite float readback.
    const uint32_t overflow[] = {
        0xbe8603ffu, 0xffffffe0u, // s_mov_b32 s6,UINT32_MAX-31
        0x7e0202c1u,            // v_mov_b32 v1,-1
        0xd7650000u, 0x00000d01u, // v_mbcnt_lo_u32_b32 v0,v1,s6
        0x7e000d00u, 0xbf810000u,
    };
    for (bool dispatcher : {false, true}) {
        const auto spv = recompile_valu(overflow, std::size(overflow), 1, 0,
            nullptr, 0, kDefaultComputePgmRsrc1, dispatcher);
        check(!spv.empty(), "wrapping accumulator compiles");
        if (spv.empty()) continue;
        const auto output = prosper::test::run_compute(spv, input, N, N);
        unsigned bad = 0;
        for (unsigned i = 0; i < output.size() && i < N; ++i) {
            const unsigned lane = i & 63;
            const uint32_t expected = 0xffffffe0u + (lane < 32 ? lane : 32);
            if (output[i] != static_cast<float>(expected)) ++bad;
        }
        check(output.size() == N && bad == 0, "numeric MBCNT uses wrapping uint32 addition");
    }

    const uint32_t positive[] = {0xd7650000u, 0x00010a83u, 0xbf810000u};
    check(!recompile_valu(positive, std::size(positive), 1, 0).empty(),
          "plain numeric form is the positive counterpart of modifier rejects");
    for (const auto& change : std::array<std::array<uint32_t, 2>, 5>{{
            {1u << 8, 0}, {1u << 11, 0}, {1u << 15, 0},
            {0, 1u << 29}, {0, 1u << 27}}}) {
        const uint32_t code[] = {positive[0] | change[0], positive[1] | change[1], positive[2]};
        check(recompile_valu(code, std::size(code), 1, 0).empty(),
              "undefined float/output/OPSEL modifiers reject instead of silently disappearing");
    }
    return failures ? 1 : 0;
}
