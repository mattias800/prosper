#include "gpu/execute/indirect_draw_arguments.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "diagnostics/env_cache.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>

namespace prosper::gpu {

bool resolve_indirect_draw_arguments(const GpuState& submit, const GpuState::Draw& source,
                                     GpuState::Draw& resolved) {
    resolved = source;
    if (!source.indirect) return true;
    if (!source.indexed) {
        // sceAgcDcbDrawIndirect (#2929) — the NON-indexed form. Its argument buffer is FOUR dwords,
        // not five: (vertexCount, instanceCount, startVertex, startInstance). Reading the indexed
        // five-dword layout here would demand a fifth dword the guest never wrote and then reject
        // the draw for having no index base, which is the same silent disappearance the missing
        // registration caused. Resolves to the DrawIndexAuto shape the realizer already renders:
        // vertex count in index_count with `indexed` false, and startVertex through the same
        // per-draw vertex-offset override the indexed path uses (the realizer hands it to
        // vkCmdDraw's firstVertex).
        constexpr uint32_t kArgumentBytes = 4u * sizeof(uint32_t);
        if (!source.indirect_args_addr || (source.indirect_args_addr & 3u) ||
            !guest_readable(source.indirect_args_addr, kArgumentBytes)) {
            static std::atomic<int> warned{0};
            if (warned.fetch_add(1) < 24)
                std::fprintf(stderr,
                             "[agc] indirect draw skipped: unreadable arguments at 0x%llx\n",
                             static_cast<unsigned long long>(source.indirect_args_addr));
            return false;
        }
        uint32_t args[4] = {};
        std::memcpy(args, reinterpret_cast<const void*>(source.indirect_args_addr), sizeof(args));
        const uint32_t vertex_count = args[0];
        const uint32_t instance_count = args[1];
        const uint32_t first_vertex = args[2];
        const uint32_t first_instance = args[3];
        constexpr uint32_t kMaxIndirectCount = 1u << 20;
        if (!vertex_count || !instance_count) return false;   // hardware no-op
        if (vertex_count > kMaxIndirectCount || instance_count > kMaxIndirectCount ||
            first_instance != 0) {
            static std::atomic<int> warned{0};
            if (warned.fetch_add(1) < 24)
                std::fprintf(
                    stderr,
                    "[agc] indirect draw skipped: vertices=%u instances=%u first_vertex=%u "
                    "first_instance=%u\n",
                    vertex_count, instance_count, first_vertex, first_instance);
            return false;
        }
        resolved.index_count = vertex_count;
        resolved.instance_count = instance_count;
        resolved.indexed = false;
        resolved.indirect_vertex_offset = static_cast<int32_t>(first_vertex);
        resolved.has_vertex_offset_override = true;
        resolved.indirect = false;
        if (PROSPER_ENV_ON("PROSPER_INDIRECTLOG")) {
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 256)
                std::fprintf(stderr,
                             "[agc-indirect] draw args=0x%llx vertices=%u instances=%u "
                             "first_vertex=%u\n",
                             static_cast<unsigned long long>(source.indirect_args_addr),
                             vertex_count, instance_count, first_vertex);
        }
        return true;
    }
    constexpr uint32_t kArgumentBytes = 5u * sizeof(uint32_t);
    if (!source.indirect_args_addr || (source.indirect_args_addr & 3u) ||
        !guest_readable(source.indirect_args_addr, kArgumentBytes)) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] indexed indirect draw skipped: unreadable arguments at 0x%llx\n",
                         static_cast<unsigned long long>(source.indirect_args_addr));
        return false;
    }
    uint32_t args[5] = {};
    std::memcpy(args, reinterpret_cast<const void*>(source.indirect_args_addr), sizeof(args));
    const uint32_t index_count = args[0];
    const uint32_t instance_count = args[1];
    const uint32_t first_index = args[2];
    const int32_t vertex_offset = static_cast<int32_t>(args[3]);
    const uint32_t first_instance = args[4];
    constexpr uint32_t kMaxIndirectCount = 1u << 20;
    if (!index_count || !instance_count) return false;   // hardware no-op
    if (index_count > kMaxIndirectCount || instance_count > kMaxIndirectCount ||
        first_instance != 0 || !source.index_base) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            std::fprintf(stderr,
                         "[agc] indexed indirect draw skipped: count=%u instances=%u first=%u "
                         "vertex_offset=%d first_instance=%u index_base=0x%llx\n",
                         index_count, instance_count, first_index, vertex_offset, first_instance,
                         static_cast<unsigned long long>(source.index_base));
        return false;
    }
    const GpuState& draw_state = source.state ? *source.state : submit;
    const uint64_t element_bytes = index_elem_bytes(draw_state.index_type);
    if (!element_bytes) return false;
    if (first_index > (UINT64_MAX - source.index_base) / element_bytes) return false;
    resolved.index_count = index_count;
    resolved.instance_count = instance_count;
    resolved.indexed = true;
    resolved.index_offset = first_index;
    resolved.index_addr = source.index_base + static_cast<uint64_t>(first_index) * element_bytes;
    resolved.from_offset = true;
    resolved.indirect_vertex_offset = vertex_offset;
    resolved.has_vertex_offset_override = true;
    resolved.indirect = false;
    if (PROSPER_ENV_ON("PROSPER_INDIRECTLOG")) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 256)
            std::fprintf(stderr,
                         "[agc-indirect] draw args=0x%llx count=%u instances=%u first=%u "
                         "vertex_offset=%d index_base=0x%llx\n",
                         static_cast<unsigned long long>(source.indirect_args_addr), index_count,
                         instance_count, first_index, vertex_offset,
                         static_cast<unsigned long long>(source.index_base));
    }
    return true;
}

}   // namespace prosper::gpu
