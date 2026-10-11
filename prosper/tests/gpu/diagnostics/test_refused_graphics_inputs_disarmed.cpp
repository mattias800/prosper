// #4877: a boot-disarmed input diagnostic must not copy even the raw code for an unavailable
// compiler invocation. The process-start environment is separate because the flag is cached.
#include "gpu/diagnostics/refused_graphics_inputs.hpp"
#include "gpu/diagnostics/refused_shader_dump.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

TEST(RefusedGraphicsInputsDisarmed, UnavailableHelperStopsBeforeCodeOwnership) {
    using namespace prosper::gpu;
    reset_refused_shader_dump_for_test(
        prosper_test::test_scratch_path("graphics_disarmed").string());
    const auto code = std::make_shared<const std::vector<uint32_t>>(
        std::initializer_list<uint32_t>{0x7e000280u, 0xbf810000u});
    RefusedShaderMemo memo;
    const RefusedShaderSource source{code, &memo};
    ASSERT_TRUE(refused_shader_dump_full()) << "the process-start disarmed control must be real";
    note_unavailable_graphics_inputs("vs", 0x1000, source, "disarmed unavailable control");
    const auto after = refused_shader_dump_stats();
    EXPECT_EQ(after.graphics_source_copy_dwords, 0u);
    EXPECT_EQ(after.hash_evaluations, 0u);
    EXPECT_EQ(after.content_records, 0u);
    EXPECT_TRUE(refused_shader_dump_directory().empty());
}
