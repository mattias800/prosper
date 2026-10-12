// A fresh VOPC predicate reads only the lanes its own EXEC enables. A never-written VGPR that a
// partial VALU write under EXEC G leaves unwritten outside G is therefore never observed by a
// compare under the same G, and must not refuse it as fabricated (fabricated-fresh-mask-source).
// The negative control keeps the refusal when the compare runs under a different EXEC value.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {
// v_mov_b32 v6, <literal>
constexpr uint32_t kMovV6[] = {0x7e0c02ffu, 0x00001234u};
// v_add-class VOP2 (op 0x1b): v8 = <literal> op v6
constexpr uint32_t kV8FromV6[] = {0x36100cffu, 0x00003fffu};
// VOPC compare, vsrc1 = v8, src0 = v6
constexpr uint32_t kCmpV8[] = {0x7c201106u};

std::vector<Rdna2Inst> decode(std::vector<uint32_t> words) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    return ins;
}

struct Harness {
    SpirvCompute b;
    RegState rs;
    Harness() {
        b.wave_size = 64;
        b.begin_fragment();
        rs.exec = b.btrue();
        rs.vector_wave_size = 64;
        rs.vector_full_exec = false;
    }
};

// Runs mov v6, partial v8 = f(lit, v6) under EXEC `writer_exec`, then returns the predicate verdict
// of the compare executed under `reader_exec`.
bool predicate_defined(uint32_t writer_exec_kind, uint32_t reader_exec_kind) {
    Harness h;
    const auto mov = decode({kMovV6[0], kMovV6[1]});
    const auto v8 = decode({kV8FromV6[0], kV8FromV6[1]});
    const auto cmp = decode({kCmpV8[0]});
    if (mov.size() != 1 || v8.size() != 1 || cmp.size() != 1) return false;
    std::vector<Rdna2Inst> all = {mov[0], v8[0], cmp[0]};
    // The writer's EXEC is narrowed and has a value of its own. The seed reads narrowedness, so
    // it must see the narrowed state.
    const uint32_t writer_exec = writer_exec_kind == 0 ? h.b.btrue() : h.b.uconst(77);
    h.rs.exec = writer_exec;
    h.rs.exec_narrowed = true;
    seed_vector_word_definedness(h.rs, all, false, 64);
    record_vector_word_marks(h.rs, mov[0], scalar_source_marks(h.rs, mov[0]));
    record_vector_word_marks(h.rs, v8[0], scalar_source_marks(h.rs, v8[0]));
    // The same SSA value is the same EXEC. Only a different value is a different lane set.
    h.rs.exec = reader_exec_kind == 0 ? writer_exec : h.b.uconst(78);
    h.rs.exec_narrowed = true;
    return fresh_mask_sources_defined(h.b, h.rs, cmp[0]);
}
}   // namespace

TEST(FreshMaskExecConfinement, CompareUnderTheWritersExecIsDefined) {
    // Lanes outside EXEC G keep the never-written v8; a compare under the same G never reads them.
    EXPECT_TRUE(predicate_defined(/*writer=*/0, /*reader=*/0));
}

TEST(FreshMaskExecConfinement, CompareUnderAnotherExecStillRefuses) {
    // A different EXEC value does not prove its lanes were written: the refusal must remain.
    EXPECT_FALSE(predicate_defined(/*writer=*/0, /*reader=*/1));
}
