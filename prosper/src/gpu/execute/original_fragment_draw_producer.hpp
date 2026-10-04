#pragma once
#include "gpu/execute/fragment_scalar_bank.hpp"
#include "gpu/execute/native_graphics_source_lineage.hpp"
#include "gpu/recompiler/original_fragment_producer.hpp"

namespace prosper::gpu {
std::shared_ptr<const OriginalFragmentDrawProducer>
seal_original_fragment_draw_producer(const OrderedScalarBankReadPoint&, const GpuState&,
                                     uint64_t command_order,
                                     std::shared_ptr<const NativeGraphicsStageCompilation>,
                                     std::shared_ptr<const FragmentScalarBank>);
} // namespace prosper::gpu
