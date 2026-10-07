// compute_transfer_gate_stats.hpp -- counters of the compute transfer-gate census (diagnostic).
#pragma once

#include <cstdint>

namespace prosper::frontend {

struct ComputeTransferGateStats {
    uint64_t reflected_images = 0;
    uint64_t reflected_storage = 0;
    uint64_t reflected_sampled = 0;
    uint64_t reflected_dim_2d = 0;
    uint64_t reflected_nonarrayed = 0;
    uint64_t reflected_nonmsaa = 0;
    uint64_t ordinary_2d = 0;
    uint64_t storage_float = 0;
    uint64_t storage_native_semantic = 0;
    uint64_t storage_native_device = 0;
    uint64_t storage_cache_evaluated = 0;
    uint64_t storage_renderer_owned = 0;
    uint64_t storage_dcc_cache_safe = 0;
    uint64_t storage_poison_verify = 0;
    uint64_t storage_exact = 0;
    uint64_t storage_write_only = 0;
    uint64_t storage_persistent_enabled = 0;
    uint64_t storage_cache_candidate = 0;
    uint64_t storage_persistent_setup = 0;
    uint64_t storage_publish_evaluated = 0;
    uint64_t storage_publish_native = 0;
    uint64_t storage_publish_unique = 0;
    uint64_t storage_publish_candidate = 0;
    uint64_t storage_publish_persistent = 0;
    uint64_t storage_publish_authorized = 0;
    uint64_t sampled_gate_evaluated = 0;
    uint64_t sampled_cache_candidate = 0;
    uint64_t sampled_persistent_setup = 0;
    uint64_t sampled_ordinary_2d = 0;
    uint64_t sampled_format_compatible = 0;
    uint64_t sampled_transfer_dimension = 0;
    uint64_t sampled_hostless = 0;
    uint64_t sampled_format_match = 0;
    uint64_t sampled_validation_enabled = 0;
    uint64_t sampled_native_defined = 0;
    uint64_t borrow_attempts = 0;
    uint64_t borrow_hits = 0;
    uint64_t borrow_no_cache = 0;
    uint64_t borrow_invalid_cache = 0;
    uint64_t borrow_authority_changed = 0;
    uint64_t borrow_metadata_unproven = 0;
};

}  // namespace prosper::frontend
