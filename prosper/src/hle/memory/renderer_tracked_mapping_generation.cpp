// renderer_tracked_mapping_generation.cpp -- the tracked-mapping generation, readable without the table.
#include "hle/memory/renderer_tracked_mapping_cache.hpp"

extern "C" uint64_t prosper_renderer_tracked_mapping_generation() {
    return prosper::detail::renderer_tracked_mapping_generation().load();
}
