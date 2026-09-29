// Process-lifetime callback prelude; see callback_state.hpp (#3892).
#include "shared/live/submit_renderer/callback_state.hpp"
#include "diagnostics/env_cache.hpp"
#include "diagnostics/env_numeric.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace prosper::frontend {
namespace {

uint64_t host_physical_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? status.ullTotalPhys : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0) return 0;
    const uint64_t count = static_cast<uint64_t>(pages);
    const uint64_t bytes = static_cast<uint64_t>(page_bytes);
    return count <= UINT64_MAX / bytes ? count * bytes : UINT64_MAX;
#endif
}
} // namespace

namespace submit_renderer {

CallbackState& CallbackState::instance() {
    static CallbackState state;
    return state;
}

const size_t& CallbackState::write_watch_promotion_budget_bytes() {
    static const auto& value = write_watch_promotion_budget_bytes_.emplace([] {
        // 0 is not "off": an empty budget makes WriteWatchPromotionBudget::try_consume
        // return true unconditionally, i.e. unbounded arming per submit. A typo must keep
        // the default rather than select that (#3253).
        const char* value = PROSPER_ENV_VALUE("PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB");
        const uint64_t mib = prosper::diag::env_u64_or_default_capped(
            "PROSPER_TEXTURE_WRITE_WATCH_PROMOTE_MB", value, 8ull,
            SIZE_MAX / (1024ull * 1024ull), "MiB");
        return static_cast<size_t>(mib * (1024ull * 1024ull));
    }());
    return value;
}

std::atomic<int>& CallbackState::g_submit_idx() {
    static auto& value = g_submit_idx_.emplace(0);
    return value;
}

int& CallbackState::g_render_first() {
    static auto& value = g_render_first_.emplace(getenv("PROSPER_RENDER_FIRST") ? atoi(getenv("PROSPER_RENDER_FIRST")) : 0);
    return value;
}

const int64_t& CallbackState::g_render_delay_ms() {
    static const auto& value = g_render_delay_ms_.emplace(getenv("PROSPER_RENDER_DELAY_MS")
        ? std::max<int64_t>(0, atoll(PROSPER_ENV_VALUE("PROSPER_RENDER_DELAY_MS"))) : 0);
    return value;
}

const std::chrono::steady_clock::time_point& CallbackState::g_render_delay_start() {
    static const auto& value = g_render_delay_start_.emplace(std::chrono::steady_clock::now());
    return value;
}

std::atomic<bool>& CallbackState::g_render_delay_announced() {
    static auto& value = g_render_delay_announced_.emplace(false);
    return value;
}

int& CallbackState::g_render_last() {
    static auto& value = g_render_last_.emplace(getenv("PROSPER_RENDER_LAST") ? atoi(getenv("PROSPER_RENDER_LAST")) : INT_MAX);
    return value;
}

const int& CallbackState::g_rttlog_min_submit() {
    static const auto& value = g_rttlog_min_submit_.emplace(getenv("PROSPER_RTTLOG_MIN_SUBMIT")
        ? std::max(0, atoi(PROSPER_ENV_VALUE("PROSPER_RTTLOG_MIN_SUBMIT"))) : 0);
    return value;
}

const int& CallbackState::g_rttlog_max_submit() {
    static const auto& value = g_rttlog_max_submit_.emplace(getenv("PROSPER_RTTLOG_MAX_SUBMIT")
        ? std::max(0, atoi(PROSPER_ENV_VALUE("PROSPER_RTTLOG_MAX_SUBMIT"))) : INT_MAX);
    return value;
}

const bool& CallbackState::validation_census_requested() {
    static const auto& value = validation_census_requested_.emplace(PROSPER_ENV_VALUE("PROSPER_TEXTURE_VALIDATION_CENSUS") != nullptr);
    return value;
}

const uint64_t& CallbackState::rtt_timing_min_draws() {
    static const auto& value = rtt_timing_min_draws_.emplace(getenv("PROSPER_RTT_TIMING_MIN_DRAWS")
        ? strtoull(PROSPER_ENV_VALUE("PROSPER_RTT_TIMING_MIN_DRAWS"), nullptr, 0) : 0);
    return value;
}

const bool& CallbackState::submit_decode_scope_disabled() {
    static const auto& value = submit_decode_scope_disabled_.emplace(PROSPER_ENV_VALUE("PROSPER_NO_SUBMIT_TEXTURE_DECODE_SCOPE") != nullptr ||
        PROSPER_ENV_VALUE("PROSPER_RESOURCE_HASH_DIM") != nullptr);
    return value;
}

const bool& CallbackState::use_tracked_buffer_membership_cache() {
    static const auto& value = use_tracked_buffer_membership_cache_.emplace(PROSPER_ENV_VALUE("PROSPER_NO_TRACKED_BUFFER_GATE") == nullptr);
    return value;
}

std::unordered_map<TextureDecodeKey, PersistentDecodedTexture, TextureDecodeKeyHash>& CallbackState::persistent_decoded_textures() {
    static auto& value = persistent_decoded_textures_.emplace();
    return value;
}

size_t& CallbackState::persistent_decoded_texture_bytes() {
    static auto& value = persistent_decoded_texture_bytes_.emplace(0);
    return value;
}

uint64_t& CallbackState::persistent_decode_generation() {
    static auto& value = persistent_decode_generation_.emplace(0);
    return value;
}

size_t& CallbackState::retired_submit_bytes() {
    static auto& value = retired_submit_bytes_.emplace(0);
    return value;
}

uint64_t& CallbackState::persistent_texture_id() {
    static auto& value = persistent_texture_id_.emplace(0);
    return value;
}

std::vector<uint8_t>& CallbackState::persistent_validation_scratch() {
    static auto& value = persistent_validation_scratch_.emplace();
    return value;
}

const size_t& CallbackState::persistent_decode_limit() {
    static const auto& value = persistent_decode_limit_.emplace([] {
        const uint64_t physical_bytes = host_physical_memory_bytes();
        const size_t limit = texture_decode_cache_limit_bytes(
            PROSPER_ENV_VALUE("PROSPER_TEXTURE_DECODE_CACHE_MB"), physical_bytes);
        fprintf(stderr,
                "[render] decoded texture cache budget = %.1f MiB "
                "(host physical %.1f GiB)\n",
                limit / (1024.0 * 1024.0),
                physical_bytes / (1024.0 * 1024.0 * 1024.0));
        return limit;
    }());
    return value;
}

const bool& CallbackState::reserve_frame_resources() {
    static const auto& value = reserve_frame_resources_.emplace([] {
        const char* setting = std::getenv("PROSPER_FRAME_RESOURCE_RESERVE");
        return !setting || std::strcmp(setting, "0") != 0;
    }());
    return value;
}

const bool& CallbackState::disable_guest_depth_layers() {
    static const auto& value = disable_guest_depth_layers_.emplace(std::getenv("PROSPER_NO_GUEST_DEPTH_LAYERS") != nullptr);
    return value;
}

} // namespace submit_renderer
} // namespace prosper::frontend
