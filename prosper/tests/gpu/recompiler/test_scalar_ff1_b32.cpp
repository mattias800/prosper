// #4060: ordinary FF1_B32 data is not a guest-wave reduction.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

using namespace prosper::gpu;
#define CHECK(c, m) EXPECT_TRUE(c) << (m)

static size_t instructions(const std::vector<uint32_t>& words, uint32_t opcode,
                           uint32_t extended = UINT32_MAX) {
    size_t found = 0;
    for (size_t n = 5; n < words.size();) {
        const size_t size = words[n] >> 16;
        if (!size || n + size > words.size()) return 0;
        if ((words[n] & 0xffffu) == opcode &&
            (extended == UINT32_MAX || (size >= 6 && words[n + 4] == extended))) ++found;
        n += size;
    }
    return found;
}

static std::vector<uint32_t> compile(const std::vector<uint32_t>& code,
                                     uint32_t wave = 64, uint32_t native = 0) {
    ComputeShaderConfig config;
    config.wave_size = wave;
    config.native_subgroup_size = native;
    return recompile_compute(code.data(), code.size(), nullptr, config);
}

// Exercise the actual emitter with explicit competing representations. A source has the same
// numeric SSA word in each control; only the provenance claim changes.
static void state_contracts(const Rdna2Inst& ff1) {
    for (int arm = 0; arm < 7; ++arm) {
        SpirvCompute b;
        b.begin(0);
        b.is_compute = true;
        RegState rs;
        rs.sreg[36] = b.uconst(0x10000u);
        rs.exec = b.btrue();
        if (arm == 1) rs.scalar_presence_has_no_placeholders = false;
        if (arm == 2) rs.sreg_bool[36] = b.btrue();
        if (arm == 3) rs.sreg_bool[35] = b.btrue();
        if (arm == 4) rs.sreg_srt[36] = 0;
        if (arm == 5) { rs.sreg.erase(36); rs.sreg_entry_m0.insert(36); }
        if (arm == 6) rs.sreg_wave64_mask_half[36] = b.btrue();
        const uint32_t scc = rs.scc = b.bfalse();
        bool ok = true;
        const bool handled = emit_alu(b, rs, ff1, ok, true);
        CHECK(arm == 0 ? handled && ok : !handled || !ok,
              "only the represented ordinary word is admitted; placeholders/tokens/masks reject");
        CHECK(rs.scc == scc, "FF1 never changes SCC, including refused operands");
    }
    // Both controls contain the same represented source word. The existing uniformity predicate
    // must distinguish their FF1 results even though each result has a newly computed SSA id.
    for (bool tainted : {false, true}) for (bool self_source : {false, true}) {
        SpirvCompute b;
        b.begin(0);
        RegState rs;
        const uint32_t source = rs.sreg[36] = b.uconst(0x10000u);
        if (tainted) rs.lane_local_scalars.insert(source);
        auto write = ff1;
        if (self_source) write.dst.value = 36;
        bool ok = true;
        CHECK(emit_alu(b, rs, write, ok, true) && ok,
              "known lane-local and ordinary scalar words both admit the numeric FF1 route");
        const auto result = rs.sreg.find(write.dst.value);
        CHECK(result != rs.sreg.end() && result->second != source,
              "FF1 computes a fresh SSA value, including a self-source write");
        CHECK(scalar_is_lane_local(rs, write.dst.value) == tainted,
              "downstream uniformity predicate retains lane-local FF1 provenance");
        CHECK(rs.lane_local_scalars.contains(source) == tainted,
              "retaining result provenance leaves the original source identity unchanged");
    }
    for (int mask_base : {36, 37, 38}) {
        SpirvCompute b;
        b.begin(0);
        RegState rs;
        rs.sreg[10] = b.uconst(0x10000u);
        rs.sreg_bool[mask_base] = b.btrue();
        rs.sreg_bool_narrowed[mask_base] = true;
        auto write = ff1;
        write.src[0].value = 10;
        bool ok = true;
        CHECK(emit_alu(b, rs, write, ok, true) && ok, "numeric result overwrites an ordinary SGPR");
        CHECK(rs.sreg_bool.contains(mask_base) == (mask_base == 38),
              "destination retires both overlapping B64 roots but preserves the next pair");
        CHECK(rs.sreg_bool_narrowed.contains(mask_base) == (mask_base == 38),
              "overlapping narrowed-mask metadata retires with its root");
    }
    SpirvCompute b;
    b.begin(0);
    RegState rs;
    rs.sreg[10] = b.uconst(8);
    rs.sreg_bool[36] = b.btrue();
    rs.sreg_bool_b32.insert(36);
    auto write = ff1;
    write.src[0].value = 10;
    bool ok = true;
    CHECK(emit_alu(b, rs, write, ok, true) && ok && rs.sreg_bool.contains(36) &&
              rs.sreg_bool_b32.contains(36), "preceding B32 mask is an independent physical word");

    // An immutable input carries real bits even where the CFG's general scalar map may contain
    // placeholders. Inspect the actual FindILsb operand, not just whether translation accepted.
    SpirvCompute input_builder;
    input_builder.begin(0);
    RegState input_state;
    input_state.scalar_presence_has_no_placeholders = false;
    input_state.sreg[36] = input_builder.uconst(0);
    const uint32_t input = input_state.sreg_input[36] = input_builder.uconst(8);
    ok = true;
    CHECK(emit_alu(input_builder, input_state, ff1, ok, true) && ok,
          "immutable scalar input is usable without admitting general CFG placeholders");
    bool input_consumed = false;
    for (size_t n = 0; n < input_builder.code.size();) {
        const size_t size = input_builder.code[n] >> 16;
        if (!size || size > input_builder.code.size() - n) break;
        if ((input_builder.code[n] & 0xffffu) == 12 && size == 6 &&
            input_builder.code[n + 4] == 73 && input_builder.code[n + 5] == input)
            input_consumed = true;
        n += size;
    }
    CHECK(input_consumed, "FindILsb reads the immutable input rather than the placeholder zero");

    SpirvCompute narrow_builder;
    narrow_builder.begin(0);
    RegState narrow_state;
    narrow_state.sreg[36] = narrow_builder.uconst(8);
    narrow_state.sreg_bool[35] = narrow_builder.btrue();
    narrow_state.sreg_bool_b32.insert(35);
    ok = true;
    CHECK(emit_alu(narrow_builder, narrow_state, ff1, ok, true) && ok,
          "a preceding B32 mask does not make the source its high half");

    // Preserve the established architectural mask path and its exact-native-width requirement.
    for (uint32_t native : {0u, 32u}) for (bool placeholder : {false, true}) {
        SpirvCompute mask_builder;
        mask_builder.begin(0);
        mask_builder.is_compute = true;
        mask_builder.wave_size = 32;
        mask_builder.native_subgroup_size = native;
        RegState mask_state;
        mask_state.exec = mask_builder.btrue();
        if (placeholder) mask_state.sreg[126] = mask_builder.uconst(0);
        auto exec_source = ff1;
        exec_source.src[0] = {OperandKind::Special, 126};
        ok = true;
        const bool handled = emit_alu(mask_builder, mask_state, exec_source, ok, true);
        CHECK(native == 32 && !placeholder ? handled && ok && mask_state.sreg.contains(37)
                                          : !handled || !ok,
              "EXEC FF1 retains the exact Wave32 mask route and rejects competing scalar data");
    }
}

TEST(ScalarFf1B32, Contract) {
    constexpr uint32_t captured = 0xbea51324u;
    const auto decoded = rdna2_decode_one(&captured, 1);
    CHECK(decoded.fmt == Rdna2Format::SOP1 && decoded.len_dwords == 1 &&
              decoded.opcode == 0x13 && decoded.dst.value == 37 &&
              decoded.src[0].kind == OperandKind::SGPR && decoded.src[0].value == 36,
          "independent guest word decodes as FF1_B32 s37,s36");
    if (::testing::Test::HasFailure()) return;
    state_contracts(decoded);

    const std::vector<uint32_t> numeric{0xbea403ffu, 0x10000u, captured, 0xbf810000u};
    auto move = numeric;
    move[2] = 0xbea50324u;
    for (uint32_t wave : {32u, 64u}) for (uint32_t native : {0u, wave}) {
        const auto control = compile(move, wave, native);
        const auto result = compile(numeric, wave, native);
        CHECK(!control.empty(), "same-shape MOV control compiles");
        CHECK(!result.empty() && instructions(result, 12, 73) == 1,
              "numeric FF1 uses exactly GLSL FindILsb73 across wave/subgroup configurations");
        size_t control_groups = 0, result_groups = 0;
        for (uint32_t op = 333; op <= 366; ++op) {
            control_groups += instructions(control, op);
            result_groups += instructions(result, op);
        }
        CHECK(!result.empty() && result_groups == control_groups,
              "scalar FF1 adds no subgroup vote/reduction requirement");
    }
    for (uint32_t source : {128u, 193u, 240u, 255u}) {
        std::vector<uint32_t> code{0xbea51300u | source};
        if (source == 255) code.push_back(0x80000000u);
        code.push_back(0xbf810000u);
        CHECK(!compile(code).empty(), "32-bit integer/float inline and literal bits compile");
    }
    CHECK(compile({captured, 0xbf810000u}).empty(), "unknown SGPR never becomes a fabricated zero");
    CHECK(compile({0xbea5137eu, 0xbf810000u}).empty(), "EXEC scalar placeholders remain unavailable");
    CHECK(compile({0xbefe1381u, 0xbf810000u}).empty(), "FF1 numeric writes to EXEC remain refused");
    CHECK(compile({0xbeea1381u, 0xbf810000u}).empty(), "new numeric path does not invent VCC mask bits");
}
