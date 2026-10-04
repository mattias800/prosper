#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/execute/shader_source_window.hpp"

namespace prosper::gpu {
namespace {
struct ShaderDecodeCache {
    std::mutex mutex;
    std::unordered_map<uintptr_t, DecodedShaderEntry> entries;
    ShaderDecodeCacheStats stats;
    uint64_t use_counter = 0;
};

ShaderDecodeCache& shader_decode_cache() {
    static ShaderDecodeCache cache;
    return cache;
}

uint64_t shader_decode_cache_limit_bytes() {
    constexpr uint64_t default_bytes = 64ull * 1024 * 1024;
    const char* value = getenv("PROSPER_SHADER_DECODE_CACHE_MB");
    if (!value || !*value) return default_bytes;
    char* end = nullptr;
    const unsigned long long mib = strtoull(value, &end, 10);
    if (end == value || *end != '\0') return default_bytes;
    return std::min<uint64_t>(mib, 1024ull) * 1024 * 1024;
}

} // namespace

std::shared_ptr<const DecodedShader> decode_shader_cached(const uint32_t* code, size_t dwords) {
    dwords = shader_source_dwords(uint64_t(uintptr_t(code)), dwords);
    auto decode = [&] {
        auto result = std::make_shared<DecodedShader>();
        result->source_dwords = dwords;
        if (!code || !dwords) return result;
        // Fold/proof share one bounded byte version; truncated encoded operands are not zeros.
        const std::vector<uint32_t> snapshot = shader_source_snapshot(code, dwords);
        std::vector<Rdna2Inst> decoded;
        const size_t consumed = rdna2_walk(snapshot.data(), snapshot.size(), decoded);
        result->code.assign(snapshot.begin(), snapshot.begin() + consumed);
        result->packet_requirements = fragment_packet_vgpr_requirements(
            result->code, decoded, FragmentPacketExportObservation::Architectural);
        for (const auto [index, stage] : {std::pair{0u, ShaderProgramStage::Vertex},
                                          std::pair{1u, ShaderProgramStage::Fragment}}) {
            result->original_effects[index] =
                original_graphics_stage_effects(result->code, decoded, stage);
            if (consumed != snapshot.size())
                result->original_effects[index].rejection =
                    "original-stage-registered-tail-unproved";
        }
        if (!decoded.empty()) {
            const Rdna2Inst& last = decoded.back();
            result->terminated = last.is_end || last.fmt == Rdna2Format::Unknown ||
                                 last.len_dwords == 0;
        }
        // The fold ignores most vector/control/export instructions unless the decoder reports an SGPR
        // destination (the conservative unknown-value invalidation). Scalar lane spills are the exception:
        // retain their spill VGPR writes too, so an ordinary VGPR write invalidates any saved lane values.
        std::set<int> scalar_spill_vgprs;
        std::set<int> fetch_vaddr_vgprs;
        std::set<int> zero_mip_vgprs;
        for (const Rdna2Inst& instruction : decoded) {
            if (instruction.fmt == Rdna2Format::VOP3 && instruction.opcode == 0x361 &&
                instruction.dst.value >= 0 && instruction.dst.value < 256)
                result->scalar_spill_written_vgprs.set(
                    static_cast<size_t>(instruction.dst.value));
            if ((instruction.fmt == Rdna2Format::MUBUF ||
                 instruction.fmt == Rdna2Format::MTBUF) &&
                instruction.src[0].kind == OperandKind::VGPR)
                fetch_vaddr_vgprs.insert(instruction.src[0].value);
            uint32_t mip_vgpr = 0;
            if (rdna2_mimg_zero_mip_shape(instruction, &mip_vgpr))
                zero_mip_vgprs.insert(static_cast<int>(mip_vgpr));
            if (instruction.fmt == Rdna2Format::VOP3) {
                if (instruction.opcode == 0x361 && instruction.dst.kind == OperandKind::VGPR)
                    scalar_spill_vgprs.insert(instruction.dst.value);       // v_writelane_b32
                else if (instruction.opcode == 0x360 && instruction.src[0].kind == OperandKind::VGPR)
                    scalar_spill_vgprs.insert(instruction.src[0].value);    // v_readlane_b32
            }
        }
        auto retain_fold_instructions = [&](const std::vector<Rdna2Inst>& source,
                                            std::vector<Rdna2Inst>& retained) {
            retained.reserve(source.size());
            for (const Rdna2Inst& instruction : source) {
                if (instruction.is_end) break;
                const bool scalar_spill = instruction.fmt == Rdna2Format::VOP3 &&
                                          (instruction.opcode == 0x360 || instruction.opcode == 0x361);
                const bool vector_index_select =
                    ((instruction.fmt == Rdna2Format::VOP3 && instruction.opcode == 0x101) ||
                     (instruction.fmt == Rdna2Format::VOP2 && instruction.opcode == 0x01)) &&
                    instruction.dst.kind == OperandKind::VGPR &&
                    fetch_vaddr_vgprs.contains(instruction.dst.value);
                // Index provenance begins at the hardware ABI VGPRs and is killed by any later shader
                // computation of a register used as VADDR. Keep only writes to actual fetch-address
                // registers; retaining every VALU instruction would make the otherwise-small scalar fold
                // walk large UE shaders in full. This is what distinguishes DQ's first v5=vertex_id fetch
                // from its later v5=3*vertex_id+1 packed-attribute fetch.
                const bool fetch_vaddr_write = instruction.dst.kind == OperandKind::VGPR &&
                                               fetch_vaddr_vgprs.contains(instruction.dst.value);
                const bool scalar_spill_invalidation = instruction.dst.kind == OperandKind::VGPR &&
                                                       scalar_spill_vgprs.contains(instruction.dst.value);
                // The zero-mip proof needs the unambiguous reaching definition of one exact address
                // VGPR. Retain every possible writer whose (at most four-dword) result overlaps it;
                // false-positive retention is cheap, while dropping one would accept a stale v_mov.
                bool zero_mip_write = false;
                if (instruction.dst.kind == OperandKind::VGPR) {
                    for (int reg : zero_mip_vgprs)
                        if (instruction.dst.value <= reg && instruction.dst.value + 3 >= reg) {
                            zero_mip_write = true;
                            break;
                        }
                }
                const bool zero_mip_definition = zero_mip_write &&
                    instruction.fmt == Rdna2Format::VOP1 && instruction.opcode == 0x01u &&
                    instruction.len_dwords == 1u && !instruction.has_modifier &&
                    instruction.src[0].kind == OperandKind::SGPR;
                const bool zero_mip_intervening_write =
                    zero_mip_write && !zero_mip_definition;
                const bool fold_format = instruction.fmt == Rdna2Format::SOP1 ||
                                         instruction.fmt == Rdna2Format::SOP2 ||
                                         instruction.fmt == Rdna2Format::SOPC ||
                                         instruction.fmt == Rdna2Format::SOPK ||
                                         instruction.fmt == Rdna2Format::SOPP ||
                                         instruction.fmt == Rdna2Format::SMEM ||
                                         instruction.fmt == Rdna2Format::MIMG ||
                                         instruction.fmt == Rdna2Format::MUBUF ||
                                         instruction.fmt == Rdna2Format::MTBUF;
                if (fold_format || rdna2_instruction_may_change_exec(instruction) ||
                    scalar_spill || vector_index_select || fetch_vaddr_write ||
                    scalar_spill_invalidation || zero_mip_definition ||
                    zero_mip_intervening_write ||
                    instruction.dst.kind == OperandKind::SGPR)
                    retained.push_back(instruction);
            }
        };
        // Retain only instructions that can affect fold state or emit a descriptor use, preserving
        // their original order and PCs. Prove shader-constant branches against the FULL decoded
        // stream first: the compact fold stream intentionally omits most VALU, including implicit
        // VCC writers that must invalidate a scalar-data proof through s106:s107.
        result->raw_x2_data_load_pcs = rdna2_proven_raw_x2_data_loads(decoded);
        result->raw_immediate_wide_data_load_pcs =
            rdna2_proven_raw_immediate_wide_data_loads(decoded);
        result->raw_register_wide_data_load_pcs =
            rdna2_proven_raw_register_wide_data_loads(decoded, &result->raw_offset_scalar_source_pcs);
        result->raw_owned_wide_data_load_pcs = rdna2_owned_raw_wide_data_loads(decoded);
        result->raw_nested_wide_data_load_pcs =
            rdna2_proven_raw_nested_wide_data_loads(decoded);
        result->owned_nested_wide_chains = rdna2_owned_nested_wide_chains(decoded);
        result->owned_raw_x2_chains = rdna2_owned_raw_x2_chains(decoded);
        if (!result->owned_raw_x2_chains.empty())
            result->owned_raw_x2_write_plan =
                raw_snapshot_write_plan(decoded, result->owned_raw_x2_chains);
        result->raw_nested_numeric_load_pcs = rdna2_raw_nested_numeric_loads(decoded);
        retain_fold_instructions(decoded, result->instructions);
        std::vector<Rdna2Inst> shader_constant_decoded = decoded;
        result->shader_constant_specialized =
            rdna2_specialize_shader_constant_branches(shader_constant_decoded) != 0;
        if (result->shader_constant_specialized)
            retain_fold_instructions(shader_constant_decoded,
                                     result->shader_constant_instructions);
        if (fold_control_cache_enabled()) {
            result->control_plan = build_fold_control_plan(result->instructions);
            if (result->shader_constant_specialized)
                result->shader_constant_control_plan =
                    build_fold_control_plan(result->shader_constant_instructions);
        }
        result->bytes =
            static_cast<uint64_t>(result->code.size()) * sizeof(uint32_t) +
            result->packet_requirements.retained_bytes() +
            result->original_effects[0].retained_bytes() +
            result->original_effects[1].retained_bytes() +
            static_cast<uint64_t>(result->instructions.size() +
                                  result->shader_constant_instructions.size()) *
                sizeof(Rdna2Inst) +
            sizeof(result->scalar_spill_written_vgprs) + result->control_plan.allocated_bytes() +
            result->shader_constant_control_plan.allocated_bytes() +
            (result->raw_x2_data_load_pcs.capacity() +
             result->raw_immediate_wide_data_load_pcs.capacity() +
             result->raw_register_wide_data_load_pcs.capacity() +
             result->raw_offset_scalar_source_pcs.capacity() +
             result->raw_owned_wide_data_load_pcs.capacity() +
             result->raw_nested_wide_data_load_pcs.capacity() +
             result->raw_nested_numeric_load_pcs.capacity() +
             result->owned_raw_x2_write_plan.descriptor_pcs.capacity() +
             result->owned_raw_x2_write_plan.storage_write_pcs.capacity()) *
                sizeof(uint32_t) +
            (result->owned_nested_wide_chains.capacity() + result->owned_raw_x2_chains.capacity()) *
                sizeof(RawNestedWideChain);
        return result;
    };

    if (!code || !dwords) return decode();
    auto& cache = shader_decode_cache();
    if (PROSPER_ENV_ON("PROSPER_NO_SHADER_DECODE_CACHE")) {
        std::lock_guard lock(cache.mutex);
        ++cache.stats.bypasses;
        return decode();
    }

    const uintptr_t address = reinterpret_cast<uintptr_t>(code);
    {
        std::lock_guard lock(cache.mutex);
        auto found = cache.entries.find(address);
        if (found != cache.entries.end()) {
            const auto& cached = found->second.shader;
            const bool compatible_length = cached->terminated || cached->source_dwords == dwords;
            const uint64_t code_bytes = static_cast<uint64_t>(cached->code.size()) * sizeof(uint32_t);
            const bool readable = code_bytes == 0 ||
                                  (code_bytes <= UINT32_MAX && guest_readable(address, (uint32_t)code_bytes));
            if (compatible_length && cached->code.size() <= dwords && readable &&
                (code_bytes == 0 || memcmp(code, cached->code.data(), (size_t)code_bytes) == 0)) {
                ++cache.stats.hits;
                found->second.last_use = ++cache.use_counter;
                return cached;
            }
            cache.stats.bytes -= cached->bytes;
            cache.entries.erase(found);
            ++cache.stats.invalidations;
        }
        ++cache.stats.misses;
    }

    auto decoded = decode();
    std::lock_guard lock(cache.mutex);
    // Another cold worker may have populated this address while decode ran. Replace it only after
    // removing its accounted bytes; returned shared_ptrs remain valid independently of the map entry.
    auto concurrent = cache.entries.find(address);
    if (concurrent != cache.entries.end()) {
        cache.stats.bytes -= concurrent->second.shader->bytes;
        cache.entries.erase(concurrent);
    }
    constexpr size_t max_entries = 4096;
    const uint64_t limit = shader_decode_cache_limit_bytes();
    while (!cache.entries.empty() &&
           (cache.entries.size() >= max_entries || cache.stats.bytes + decoded->bytes > limit)) {
        auto oldest = cache.entries.begin();
        for (auto it = std::next(cache.entries.begin()); it != cache.entries.end(); ++it)
            if (it->second.last_use < oldest->second.last_use) oldest = it;
        cache.stats.bytes -= oldest->second.shader->bytes;
        cache.entries.erase(oldest);
        ++cache.stats.evictions;
    }
    if (!decoded->code.empty() && decoded->bytes <= limit && max_entries != 0) {
        cache.entries[address] = {decoded, ++cache.use_counter};
        cache.stats.bytes += decoded->bytes;
    }
    cache.stats.entries = cache.entries.size();
    return decoded;
}

ShaderDecodeCacheStats shader_decode_cache_stats() {
    auto& cache = shader_decode_cache();
    std::lock_guard lock(cache.mutex);
    ShaderDecodeCacheStats stats = cache.stats;
    stats.entries = cache.entries.size();
    return stats;
}

void clear_shader_decode_cache() {
    auto& cache = shader_decode_cache();
    std::lock_guard lock(cache.mutex);
    cache.entries.clear();
    cache.stats = {};
    cache.use_counter = 0;
}

} // namespace prosper::gpu
