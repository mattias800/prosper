#pragma once
#include "gpu/recompiler/ir/lowering_fixture.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <cstdint>
#include <vector>
namespace prosper::test::ir {
inline std::vector<uint32_t> scalar64_finish(LoweringFixture& t, uint32_t id) {
    t.b.store_output_word(id, 1u, 0u);
    bpermute_oracle::Interpreter vm(t.b.finish());
    vm.legacy_float_word_storage = true;
    auto out = vm.run_buffers(64, {{1, std::vector<uint32_t>(64, 0xdeadbeef)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    out.resize(64);
    return out;
}
inline void expect_scalar64_word(LoweringFixture& t, uint32_t id, uint32_t want) {
    const auto out = scalar64_finish(t, id);
    ASSERT_EQ(out.size(), 64u);
    for (uint32_t word : out) ASSERT_EQ(word, want);
}
} // namespace prosper::test::ir
