// gpu_retile_census.hpp -- why each compute image did not take the GPU retile path (diagnostic).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include "gpu/resources/shader_resources.hpp"

namespace prosper::frontend {

// Why a storage image took the CPU tiling path instead of the GPU retile compute shader.
//
// `layout_ms` -- the CPU tile_surface/tile_volume pass -- is one of the largest leaves in the compute
// profile (about 7 s of a 33 s summed-thread total on a 70 s Sonic Frontiers route), and it runs
// exactly when this admission declines. The admission has a dozen `continue`s and no counter on any of
// them, so "the CPU path is hot" and "which condition sent it there" were separate questions with only
// the first answerable. Nothing here changes a decision; each reason is recorded where the decision was
// already being made.
//
// Counted per reason and per program, because the interesting quantity is which SHAPE a hot program
// presents, not the global mix: a title can decline ten thousand times for a reason that costs nothing
// and a hundred times for the one that owns the frame.
enum class GpuRetileDecline : uint8_t {
    Admitted = 0,
    Aliased,               // this binding is an alias of an earlier one; the owner does the writeback
    Imported,              // guest-imported image, not a storage result prosper produced
    PartialWrite,          // a write mask means the result does not cover the whole surface
    InexactBytes,          // staged bytes are not the guest's exact storage extent
    MipTailOrOffset,       // in a mip tail, or at a nonzero layer/mip offset
    NoResource,            // the binding carries no ShaderResource at all
    NoStaging,             // no staging buffer for this image
    UnsupportedShape,      // layer/depth/dimension combination outside the admitted set
    PackedDisabled,        // packed byte/halfword words with the extension switched off
    PackedUnsupported,     // array-and-not-packed, or a packed descriptor the layout cannot express
    LayoutMismatch,        // parameters would not reproduce the guest's exact tiled/linear extents
    PrepareFailed,         // pipeline/buffer/memory/descriptor setup declined at runtime
    Disabled,              // PROSPER_NO_GPU_RETILE
    Count
};

inline const char* gpu_retile_decline_name(GpuRetileDecline reason) {
    switch (reason) {
        case GpuRetileDecline::Admitted: return "admitted";
        case GpuRetileDecline::Aliased: return "aliased";
        case GpuRetileDecline::Imported: return "imported";
        case GpuRetileDecline::PartialWrite: return "partial-write";
        case GpuRetileDecline::InexactBytes: return "inexact-bytes";
        case GpuRetileDecline::MipTailOrOffset: return "mip-tail-or-offset";
        case GpuRetileDecline::NoResource: return "no-resource";
        case GpuRetileDecline::NoStaging: return "no-staging";
        case GpuRetileDecline::UnsupportedShape: return "unsupported-shape";
        case GpuRetileDecline::PackedDisabled: return "packed-disabled";
        case GpuRetileDecline::PackedUnsupported: return "packed-unsupported";
        case GpuRetileDecline::LayoutMismatch: return "layout-mismatch";
        case GpuRetileDecline::PrepareFailed: return "prepare-failed";
        case GpuRetileDecline::Disabled: return "disabled";
        default: return "?";
    }
}

struct GpuRetileCensus {
    struct Row {
        std::array<uint64_t, static_cast<size_t>(GpuRetileDecline::Count)> reasons{};
        // One representative shape PER REASON, and printed next to that reason's name. A single
        // sample per program was the first shape of this and it is not enough to act on: it is
        // whichever reason declined first, it carries no label saying which, and quoting it beside a
        // count generalizes one image to every decline in the row. Per reason, first seen wins --
        // a later one would need a policy for which is representative, and there is none.
        std::array<std::array<char, 96>, static_cast<size_t>(GpuRetileDecline::Count)> samples{};
    };
    std::mutex mutex;
    std::unordered_map<uint64_t, Row> by_code;   // guest code address of the dispatching program
    std::atomic<bool> any{false};

    void record(uint64_t code_addr, GpuRetileDecline reason,
                const prosper::gpu::ShaderResource* r, uint32_t bpe) {
        any.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex);
        Row& row = by_code[code_addr];
        row.reasons[static_cast<size_t>(reason)]++;
        auto& sample = row.samples[static_cast<size_t>(reason)];
        if (reason != GpuRetileDecline::Admitted && !sample[0] && r) {
            std::snprintf(sample.data(), sample.size(),
                          "%ux%ux%u dim=%u tile=%u fmt=%u bpe=%u",
                          r->width, r->height, r->depth, (unsigned)r->img_dim,
                          (unsigned)r->tile_mode, (unsigned)r->format, bpe);
        }
    }

    void report() {
        if (!any.load(std::memory_order_relaxed)) return;
        std::lock_guard<std::mutex> lock(mutex);
        std::fprintf(stderr, "[gpu-retile-census] %zu program(s)\n", by_code.size());
        for (const auto& [code, row] : by_code) {
            std::string line;
            uint64_t total = 0;
            for (size_t i = 0; i < row.reasons.size(); ++i) {
                if (!row.reasons[i]) continue;
                total += row.reasons[i];
                line += " ";
                line += gpu_retile_decline_name(static_cast<GpuRetileDecline>(i));
                line += "=" + std::to_string(row.reasons[i]);
                // The sample belongs to THIS reason, and says so. A shape printed once per row with
                // no label cannot be attributed to any bucket, and reads as representative of all of
                // them. "first" rather than "e.g." because it is the first seen, not a typical one:
                // nothing here establishes that the other declines in this bucket share its shape.
                if (row.samples[i][0]) {
                    line += "(first ";
                    line += row.samples[i].data();
                    line += ")";
                }
            }
            std::fprintf(stderr, "[gpu-retile-census] code=0x%llx images=%llu%s\n",
                         (unsigned long long)code, (unsigned long long)total, line.c_str());
        }
    }
};

inline GpuRetileCensus& gpu_retile_census() { static GpuRetileCensus* value = new GpuRetileCensus; return *value; }

}  // namespace prosper::frontend
