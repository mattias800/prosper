// A real module through every buffer-store value definition, shared by the numeric and strict
// SPIR-V validation gates. Arbitrary input words remain runtime values throughout emission.
#pragma once
#include "gpu/recompiler/ir/ops/buffer_store.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace prosper::test::buffer_store_ir {
inline std::vector<uint32_t> module() {
    using namespace prosper::gpu;
    using namespace prosper::gpu::ir;
    SpirvCompute b;
    b.begin(5, nullptr, 32, 1, 1, 32, 0, true, true);
    std::array<uint32_t, 5> inputs{};
    for (uint32_t k = 0; k < 5; ++k) inputs[k] = b.load_input(k);
    std::vector<uint32_t> outputs;
    outputs.reserve(14);
    const auto append = [&](const BufferStoreBlock& built) {
        const auto ids =
            emit_buffer_store_values(b, built, [&](uint32_t slot) { return inputs[slot]; });
        outputs.insert(outputs.end(), ids.begin(), ids.end());
    };
    append(build_buffer_store_fields(3, 2, 65535));
    append(build_buffer_store_index(3));
    for (auto kind : {BufferStoreKind::Integer, BufferStoreKind::Unorm, BufferStoreKind::Snorm,
                      BufferStoreKind::Ufloat})
        append(build_buffer_store_word({kind, 4, kind == BufferStoreKind::Ufloat ? 3u : 4u, 0}));
    append(build_buffer_store_dword({BufferStoreKind::Half, 4, 4, 2}, 1));
    append(build_buffer_store_dword({BufferStoreKind::Unorm, 4, 4, 1}, 0));
    append(build_buffer_store_dword({BufferStoreKind::Snorm, 4, 4, 2}, 1));
    for (uint32_t k = 0; k < outputs.size(); ++k) b.store_output_word(outputs[k], 14, k);
    return b.finish();
}
} // namespace prosper::test::buffer_store_ir
