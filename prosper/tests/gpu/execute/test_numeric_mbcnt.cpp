// Numeric MBCNT counts source-word bits below the physical lane, even if those lanes do not
// participate in EXEC. It is not a population scan. AMD RDNA2 ISA 70648, section 12.12.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "fixtures/compute_runner.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <map>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t N = 128;
struct Checks {
    int failures = 0;
    void check(bool ok, const char* name) {
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", name);
        if (!ok) ++failures;
    }
};

struct Definition { uint32_t op; std::vector<uint32_t> args; };
struct PrefixInspector {
    std::map<uint32_t, uint32_t> constants;
    std::map<uint32_t, Definition> definitions;

    const Definition* find(uint32_t value, uint32_t op) const {
        const auto found = definitions.find(value);
        if (found == definitions.end() || found->second.op != op) return nullptr;
        return &found->second;
    }
    bool constant(uint32_t value, uint32_t expected) const {
        const auto found = constants.find(value);
        return found != constants.end() && found->second == expected;
    }
    bool bounded_prefix(uint32_t value) const {
        const auto* select = find(value, Op_Select);
        if (!select || !constant(select->args[1], UINT32_MAX)) return false;
        const auto* subtract = find(select->args[2], Op_ISub);
        if (!subtract || !constant(subtract->args[1], 1)) return false;
        const auto* shift = find(subtract->args[0], Op_ShiftLeftLogical);
        if (!shift) return false;
        const auto* amount = find(shift->args[1], Op_BitwiseAnd);
        if (!amount) return false;
        return constant(amount->args[0], 31) || constant(amount->args[1], 31);
    }
    bool bounded_count(uint32_t value) const {
        const auto* masked = find(value, Op_BitwiseAnd);
        if (!masked) return false;
        return bounded_prefix(masked->args[0]) || bounded_prefix(masked->args[1]);
    }
    void record(const uint32_t* instruction, uint32_t words, uint32_t op) {
        if (op == Op_Constant && words == 4) constants[instruction[2]] = instruction[3];
        if ((op == Op_BitwiseAnd || op == Op_ISub || op == Op_ShiftLeftLogical) && words == 5)
            definitions[instruction[2]] = {op, {instruction[3], instruction[4]}};
        if (op == Op_Select && words == 6)
            definitions[instruction[2]] = {op, {instruction[3], instruction[4], instruction[5]}};
    }
};

// Hardware results alone need not expose an undefined shift in an unselected OpSelect operand.
// Constrain the MBCNT prefix shift, not unrelated shifts emitted by the CFG dispatcher.
bool numeric_shifts_are_bounded(const std::vector<uint32_t>& spirv) {
    PrefixInspector inspector;
    unsigned prefixes = 0;
    for (size_t i = 5; i < spirv.size();) {
        const uint32_t words = spirv[i] >> 16;
        const uint32_t op = spirv[i] & 0xffffu;
        if (!words || words > spirv.size() - i) return false;
        inspector.record(&spirv[i], words, op);
        if (op == Op_BitCount && words == 4) {
            if (!inspector.bounded_count(spirv[i + 3])) return false;
            ++prefixes;
        }
        i += words;
    }
    return prefixes == 1;
}

enum class Source : unsigned { Inline, InlineFloat, Literal, Scalar, Vector };
struct Case { Source source; uint32_t mask; int inline_value; };

std::vector<uint32_t> numeric_code(const Case& c, bool hi) {
    std::vector<uint32_t> code;
    code.reserve(12);
    uint32_t operand = 0;
    switch (c.source) {
    case Source::Inline:
        operand = c.inline_value >= 0 ? 128u + c.inline_value : 192u - c.inline_value;
        break;
    case Source::InlineFloat: operand = 242; break;
    case Source::Literal: operand = 255; break;
    case Source::Scalar:
        code.insert(code.end(), {0xbe8403ffu, c.mask}); // s_mov_b32 s4,literal
        operand = 4;
        break;
    case Source::Vector:
        code.push_back(0x7e020f00u); // v_cvt_u32_f32 v1,v0: distinct mask per lane
        operand = 257;
        break;
    }
    code.insert(code.end(), {hi ? 0xd7660000u : 0xd7650000u,
                             operand | (133u << 9)}); // MBCNT v0,source,5
    if (c.source == Source::Literal) code.push_back(c.mask);
    code.insert(code.end(), {0x7e000d00u, 0xbf810000u}); // cvt_f32_u32 v0,v0; end
    return code;
}

uint32_t physical_prefix(unsigned lane, bool hi) {
    unsigned width = std::min(lane, 32u);
    if (hi) width = lane > 32 ? lane - 32 : 0;
    return width == 32 ? UINT32_MAX : (1u << width) - 1;
}

template <typename Expected>
unsigned mismatches(const std::vector<float>& output, Expected expected) {
    unsigned bad = 0;
    for (unsigned i = 0; i < output.size() && i < N; ++i)
        if (output[i] != expected(i)) ++bad;
    return bad;
}

void run_numeric_case(Checks& checks, const Case& c, bool hi, bool dispatcher,
                      const std::vector<float>& input) {
    const auto code = numeric_code(c, hi);
    const auto spv = recompile_valu(code.data(), code.size(), 1, 0,
        nullptr, 0, kDefaultComputePgmRsrc1, dispatcher);
    checks.check(!spv.empty(), "numeric source compiles in its claimed route");
    if (spv.empty()) return;
    checks.check(numeric_shifts_are_bounded(spv),
                 "the numeric prefix shift has a structurally bounded amount");
    const auto output = prosper::test::run_compute(spv, input, N, N);
    const unsigned bad = mismatches(output, [&](unsigned i) {
        const uint32_t mask = c.source == Source::Vector ? static_cast<uint32_t>(input[i]) : c.mask;
        return static_cast<float>(5 + std::popcount(mask & physical_prefix(i & 63u, hi)));
    });
    std::printf("source=%u hi=%d dispatcher=%d bad=%u outputs=%zu\n",
        static_cast<unsigned>(c.source), hi, dispatcher, bad, output.size());
    checks.check(output.size() == N && bad == 0,
                 "every lane counts its own source word under the physical LO/HI prefix");
}

void run_numeric_matrix(Checks& checks, const std::vector<float>& input) {
    const std::array<Case, 10> cases{{
        {Source::Inline, 0, 0}, {Source::Inline, 1, 1}, {Source::Inline, 3, 3},
        {Source::Inline, 15, 15}, {Source::Inline, 0xfffffffeu, -2},
        {Source::InlineFloat, 0x3f800000u, 0}, // 1.0 operand contributes its raw bits
        {Source::Literal, 0x80000001u, 0}, {Source::Literal, 0x55555555u, 0},
        {Source::Scalar, 0xaaaaaaaau, 0}, {Source::Vector, 0, 0},
    }};
    for (const auto& c : cases) {
        for (bool hi : {false, true}) {
            run_numeric_case(checks, c, hi, false, input);
            // Raw SGPR words still use the existing dispatcher event route. This change does
            // not infer a numeric lifetime from a dispatcher scalar placeholder.
            if (c.source != Source::Scalar) run_numeric_case(checks, c, hi, true, input);
        }
    }
}

void run_divergent(Checks& checks, const std::vector<float>& input) {
    // Explicit EXEC divergence must not make a numeric source mask count only active lanes.
    // Inactive destination lanes keep their previous VGPR value.
    const std::array divergent = {
        0x7e020f00u,             // v_cvt_u32_f32 v1,v0
        0x7e0402a5u,            // v_mov_b32 v2,37
        0xd4c1006au, 0x00012101u, // v_cmp_lt_u32_e64 vcc_lo,v1,16
        0xbe88246au,            // s_and_saveexec_b64 s[8:9],vcc
        0xd7650002u, 0x00020283u, // v_mbcnt_lo_u32_b32 v2,3,v1
        0xbefe0408u,            // s_mov_b64 exec,s[8:9]
        0x7e000d02u, 0xbf810000u,
    };
    for (bool dispatcher : {false, true}) {
        const auto spv = recompile_valu(divergent.data(), divergent.size(), 1, 0,
            nullptr, 0, kDefaultComputePgmRsrc1, dispatcher);
        checks.check(!spv.empty(), "numeric MBCNT under narrowed EXEC compiles");
        if (spv.empty()) continue;
        const auto output = prosper::test::run_compute(spv, input, N, N);
        const unsigned bad = mismatches(output, [&](unsigned i) {
            const unsigned count = std::min(i & 63u, 2u);
            return input[i] < 16 ? input[i] + count : 37;
        });
        std::printf("divergent dispatcher=%d bad=%u outputs=%zu\n", dispatcher, bad, output.size());
        checks.check(output.size() == N && bad == 0,
                     "active destinations use physical bits; inactive destinations preserve their VGPR");
    }
}

void run_overflow(Checks& checks, const std::vector<float>& input) {
    // The overflowing LO prefix is exactly 32 at every lane >=32. Losing just one bit turns the
    // expected zero into a huge float, so this arm remains discriminating despite float readback.
    const std::array overflow = {
        0xbe8603ffu, 0xffffffe0u, // s_mov_b32 s6,UINT32_MAX-31
        0x7e0202c1u,            // v_mov_b32 v1,-1
        0xd7650000u, 0x00000d01u, // v_mbcnt_lo_u32_b32 v0,v1,s6
        0x7e000d00u, 0xbf810000u,
    };
    for (bool dispatcher : {false, true}) {
        const auto spv = recompile_valu(overflow.data(), overflow.size(), 1, 0,
            nullptr, 0, kDefaultComputePgmRsrc1, dispatcher);
        checks.check(!spv.empty(), "wrapping accumulator compiles");
        if (spv.empty()) continue;
        const auto output = prosper::test::run_compute(spv, input, N, N);
        const unsigned bad = mismatches(output, [](unsigned i) {
            const uint32_t expected = 0xffffffe0u + std::min(i & 63u, 32u);
            return static_cast<float>(expected);
        });
        checks.check(output.size() == N && bad == 0, "numeric MBCNT uses wrapping uint32 addition");
    }
}

void run_modifier_rejects(Checks& checks) {
    const std::array positive = {0xd7650000u, 0x00010a83u, 0xbf810000u};
    checks.check(!recompile_valu(positive.data(), positive.size(), 1, 0).empty(),
                 "plain numeric form is the positive counterpart of modifier rejects");
    for (const auto& change : std::array<std::array<uint32_t, 2>, 5>{{
            {1u << 8, 0}, {1u << 11, 0}, {1u << 15, 0},
            {0, 1u << 29}, {0, 1u << 27}}}) {
        const std::array code = {positive[0] | change[0], positive[1] | change[1], positive[2]};
        checks.check(recompile_valu(code.data(), code.size(), 1, 0).empty(),
                     "undefined float/output/OPSEL modifiers reject instead of silently disappearing");
    }
}
} // namespace

int main() {
    Checks checks;
    std::vector<float> input(N);
    for (uint32_t i = 0; i < N; ++i) input[i] = static_cast<float>((i % 11) * 3 + 1);
    run_numeric_matrix(checks, input);
    run_divergent(checks, input);
    run_overflow(checks, input);
    run_modifier_rejects(checks);
    return checks.failures ? 1 : 0;
}
