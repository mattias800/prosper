// The strict-validation DS address module: two byte indexes, a pair, its wide words, and a
// component index, all emitted through the production IR definitions.
#pragma once

#include "gpu/recompiler/ir/ops/ds_address.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace prosper::test::ir::ds_address {

inline void emit_fixture(prosper::gpu::SpirvCompute& b) {
    using namespace prosper::gpu::ir;
    const uint32_t a = b.load_input(0);
    const uint32_t c = b.load_input(1);
    std::vector<uint32_t> outputs;
    outputs.reserve(9);
    const auto append = [&](DsAddressBlock built, const DsAddressValueReader& read) {
        const auto values = emit_ds_address_values(b, built, read);
        outputs.insert(outputs.end(), values.begin(), values.end());
    };
    append(build_ds_byte_address(3, false), [&](uint32_t) { return a; });
    append(build_ds_byte_address(65535, true), [&](uint32_t) { return a; });
    append(build_ds_pair_address(3, 5), [&](uint32_t) { return a; });
    append(build_ds_pair_words(), [&](uint32_t slot) { return slot ? c : a; });
    append(build_ds_word_index(3), [&](uint32_t) { return c; });
    for (uint32_t k = 0; k < outputs.size(); ++k) b.store_output_word(outputs[k], 9, k);
}

} // namespace prosper::test::ir::ds_address
