#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <cstdint>

namespace prosper::gpu {
// Field-wise code-generation inputs. Kept out of the oversized executor; no raw-state presence
// or word can be lost while computing the cache key. User values remain runtime push constants.
void populate_compute_shader_key(ShaderCompileKey& key, const ComputeShaderConfig& config) {
    key.compute_dispatch_launch = config.dispatch_launch;
    key.compute_user_sgpr_count = static_cast<uint32_t>(config.user_sgprs.size());
    key.compute_local_x = config.local_x;
    key.compute_local_y = config.local_y;
    key.compute_local_z = config.local_z;
    key.compute_exact_thread_extent = config.exact_thread_extent;
    key.compute_threads_x = config.threads_x;
    key.compute_threads_y = config.threads_y;
    key.compute_threads_z = config.threads_z;
    key.compute_wave_size = config.wave_size;
    key.compute_pgm_rsrc1 = config.compute_pgm_rsrc1;
    key.compute_tidig_comp_cnt = config.tidig_comp_cnt;
    key.compute_tgid_x_en = config.tgid_x_en;
    key.compute_tgid_y_en = config.tgid_y_en;
    key.compute_tgid_z_en = config.tgid_z_en;
    key.compute_tg_size_en = config.tg_size_en;
    key.compute_lds_bytes = config.lds_bytes;
    key.compute_native_subgroup_size = config.native_subgroup_size;
    key.compute_wave64_exchange_width = config.wave64_exchange_width;
    key.compute_native_storage_format_support = config.native_storage_format_support;
    key.compute_storage_buffer_int64_atomics = config.storage_buffer_int64_atomics;
    key.compute_packed_r11_storage = config.packed_r11_storage;
    key.compute_choices = config.choices;
}
} // namespace prosper::gpu
