// Names for checked ordered-memory overlay diagnostics; classification stays in the processor.
#include "gpu/pm4/command_processor.hpp"

namespace prosper::gpu {

const char* ordered_wait_effect_class_name(OrderedWaitEffectClass effect_class) {
    switch (effect_class) {
        case OrderedWaitEffectClass::Disjoint: return "disjoint";
        case OrderedWaitEffectClass::ImmediateDma: return "immediate-dma";
        case OrderedWaitEffectClass::AddressDma: return "address-dma";
        case OrderedWaitEffectClass::FixedRelease32: return "fixed-release32";
        case OrderedWaitEffectClass::FixedRelease64: return "fixed-release64";
        case OrderedWaitEffectClass::InterruptOnlyRelease: return "interrupt-only-release";
        case OrderedWaitEffectClass::DynamicRelease: return "dynamic-release";
        case OrderedWaitEffectClass::WriteData: return "write-data";
        case OrderedWaitEffectClass::WriteDataPartial: return "write-data-partial";
        case OrderedWaitEffectClass::WriteDataOversized: return "write-data-oversized";
        case OrderedWaitEffectClass::WriteDataUnowned: return "write-data-unowned";
        case OrderedWaitEffectClass::EventTimestamp: return "event-timestamp";
        case OrderedWaitEffectClass::OffsetOverlap: return "offset-overlap";
        case OrderedWaitEffectClass::AliasedOverlap: return "aliased-overlap";
        case OrderedWaitEffectClass::Unsupported: return "unsupported";
    }
    return "unsupported";
}

const char* ordered_wait_effect_overlay_name(OrderedWaitEffectOverlay overlay) {
    switch (overlay) {
        case OrderedWaitEffectOverlay::None: return "none";
        case OrderedWaitEffectOverlay::Applied: return "applied";
        case OrderedWaitEffectOverlay::Ambiguous: return "ambiguous";
    }
    return "ambiguous";
}

} // namespace prosper::gpu
