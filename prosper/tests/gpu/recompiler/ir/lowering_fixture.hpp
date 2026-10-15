// A SpirvCompute + RegState pair for calling one scalar-ALU lowering directly.
#pragma once

#include "gpu/recompiler/ir/lower/scalar_alu_lowering.hpp"
#include "gpu/recompiler/ir/scalar_encoding.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace prosper::test::ir {

struct LoweringFixture {
    gpu::SpirvCompute b;
    gpu::RegState rs;
    bool ok = true;

    LoweringFixture() {
        b.begin(16, nullptr, 64, 1, 1, 64, 0);
        rs.exec = b.btrue();
    }
    static gpu::Rdna2Inst decode(const std::vector<uint32_t>& words) {
        std::vector<gpu::Rdna2Inst> ins;
        gpu::rdna2_walk(words.data(), words.size(), ins);
        EXPECT_FALSE(ins.empty());
        return ins.at(0);
    }
    // An SGPR holding a known word, as an earlier s_mov would leave it.
    void set_sgpr(int sgpr, uint32_t value) { rs.sreg[sgpr] = b.uconst(value); }
    bool constant_cached(uint32_t value) const { return b.uconst_cache.contains(value); }
    size_t code_words() const { return b.code.size(); }
};

} // namespace prosper::test::ir
