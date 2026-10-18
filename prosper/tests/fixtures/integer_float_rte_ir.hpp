// Representative RTE32 module for the same definitions used by the native e64 route.
#pragma once
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include "gpu/recompiler/ir/ops/vector_integer_float_rte_e64.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
namespace prosper::test::ir::integer_float_rte {
inline void emit_fixture(prosper::gpu::SpirvCompute& b, uint32_t opcode) {
    using namespace prosper::gpu;
    const auto* def = gpu::ir::find_vector_integer_float_rte_e64_op(Rdna2Format::VOP3, opcode);
    b.declare_float32_rte();
    b.store_output_word(gpu::ir::emit_conversion_op(b, *def, b.load_input(0)), 1, 0);
}
} // namespace prosper::test::ir::integer_float_rte
