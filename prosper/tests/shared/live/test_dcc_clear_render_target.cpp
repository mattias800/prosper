// A DCC fast clear reaches the pass that RENDERS to the cleared target (#4556).
//
// A compute span that rewrites a retained colour target's DCC metadata clears it: a uniform clear
// code is the whole surface. That clear used to be materialized only for a span that SAMPLES the
// target, because the decode took its inputs from the sampling descriptor. A span that renders to
// it loads it too -- a blend reads the destination, and what the draws do not cover stays as
// cleared -- and that span started from the draw's own clear colour instead.
//
// MOUSE: P.I. For Hire's decal buffer is that case. Each frame a compute dispatch fills its
// metadata with 0x40, the code for (0,0,0,1); the decal draws then blend with the destination
// alpha as the running transmittance; and the G-buffer pass samples the result. The decal pass
// was starting from alpha 0.
//
// Each case goes through the production functions on both sides: note_rtt_dcc_descriptor, which
// is what registers a sampling descriptor's view of the metadata, invalidate_cpu_rtt_entry, which
// is what a metadata write does to the entry, and materialize_dirty_dcc_clears itself. What these
// cases cannot see is the renderer ceasing to call the first of them when it registers a
// descriptor; that line is covered by the title, not by this file.
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/submit_renderer/callback_prelude.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <vector>

namespace {
using prosper::frontend::LiveRttGuestWriteEffect;
using prosper::frontend::RttCache;
using prosper::frontend::RttSurf;

constexpr uint64_t kTarget = 0x4045c10000ull;
constexpr uint32_t kWidth = 64, kHeight = 64;
constexpr uint8_t kClear0000 = 0x00, kClear0001 = 0x40, kClear1111 = 0xc0;

class DccClearRenderTarget : public ::testing::Test {
protected:
    static constexpr uint64_t kMetadataBytes = 0x1000;
    static constexpr uint64_t kMappedBytes = 0x10000;

    // The target, retained with content, after a span that sampled it through a DCC-enabled
    // four-component descriptor with alpha in the top byte. Its metadata plane is tracked guest
    // memory, which is the only kind the renderer reads through an address.
    void SetUp() override {
        prosper::register_builtin_hle();
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        unmap_ = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
        ASSERT_TRUE(map && unmap_);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&mapped_), kMappedBytes, 3, 0,
                      reinterpret_cast<uint64_t>("dcc-clear-render-target"), 0),
                  0u);
        ASSERT_NE(mapped_, 0u);
        metadata_ = {reinterpret_cast<uint8_t*>(mapped_), kMetadataBytes};
        std::fill(metadata_.begin(), metadata_.end(), uint8_t{0xff});
        RttSurf& surface = cache_[kTarget];
        surface.w = kWidth;
        surface.h = kHeight;
        surface.format = VK_FORMAT_R8G8B8A8_UNORM;
        surface.gpu_valid = true;
        prosper::frontend::note_rtt_dcc_descriptor(surface, descriptor(), metadata_.size());
    }
    // The descriptor that sampled the target: its metadata plane, its guest extent, and how it
    // reads a clear code.
    prosper::gpu::ShaderResource descriptor(uint32_t components = 4,
                                            bool alpha_is_on_msb = true) const {
        prosper::gpu::ShaderResource sampled;
        sampled.metadata_addr = reinterpret_cast<uint64_t>(metadata_.data());
        sampled.num_components = components;
        sampled.alpha_is_on_msb = alpha_is_on_msb;
        sampled.width = kWidth;
        sampled.height = kHeight;
        return sampled;
    }
    void TearDown() override {
        if (mapped_) unmap_(mapped_, kMappedBytes, 0, 0, 0, 0);
    }

    // A compute span rewrites the whole metadata plane with `code`.
    void fast_clear(uint8_t code) {
        std::fill(metadata_.begin(), metadata_.end(), code);
        prosper::frontend::invalidate_cpu_rtt_entry(cache_, cache_.find(kTarget),
                                                    LiveRttGuestWriteEffect::dcc_metadata, nullptr);
    }

    // One draw with `target` in colour slot 0 at the retained extent, writing `write_mask`.
    static prosper::gpu::DrawItem draw_to(uint64_t target, uint32_t write_mask,
                                          uint32_t width = kWidth) {
        prosper::gpu::DrawItem draw;
        draw.color0_base = target;
        draw.color0_width = width;
        draw.color0_height = kHeight;
        draw.ps.color_write_mask = write_mask;
        return draw;
    }

    // The graphics span that follows, made of `draw` alone, in a renderer configured with
    // `render_scale`. Returns how many targets it cleared.
    uint64_t span(const prosper::gpu::DrawItem& draw, uint32_t render_scale = 1) {
        const std::vector<prosper::gpu::DrawItem> items{draw};
        prosper::frontend::submit_renderer::RenderTiming timing;
        const bool timing_enabled = true;
        prosper::frontend::submit_renderer::DccClearContext context{.g_rtt = cache_,
                                                                    .items = items,
                                                                    .pending_timing = timing,
                                                                    .timing_enabled =
                                                                        timing_enabled,
                                                                    .render_scale = render_scale};
        prosper::frontend::submit_renderer::materialize_dirty_dcc_clears(context);
        return timing.dcc_materialize_surfaces;
    }

    const RttSurf& surface() const { return cache_.at(kTarget); }

    uint64_t mapped_ = 0;
    prosper::HleFn unmap_ = nullptr;
    std::span<uint8_t> metadata_;
    RttCache cache_;
};

TEST_F(DccClearRenderTarget, APassThatRendersToAClearedTargetStartsFromItsClearColour) {
    fast_clear(kClear0001);
    ASSERT_TRUE(surface().dcc_metadata_dirty);
    ASSERT_FALSE(surface().has_uniform_color);
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 1u);
    EXPECT_TRUE(surface().has_uniform_color);
    EXPECT_EQ(surface().uniform_color, (std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
    EXPECT_FALSE(surface().dcc_metadata_dirty);
}

TEST_F(DccClearRenderTarget, TheColourIsTheOneTheCodeNames) {
    fast_clear(kClear0000);
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 1u);
    EXPECT_EQ(surface().uniform_color, (std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}));
    fast_clear(kClear1111);
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 1u);
    EXPECT_EQ(surface().uniform_color, (std::array<float, 4>{1.0f, 1.0f, 1.0f, 1.0f}));
}

// The case that already worked, through the same decode: a span that samples the target.
TEST_F(DccClearRenderTarget, ASpanThatSamplesAClearedTargetStillStartsFromItsClearColour) {
    fast_clear(kClear0001);
    prosper::gpu::ShaderResource sampled;
    sampled.cls = prosper::gpu::ResourceClass::Texture;
    sampled.gpu_addr = kTarget;
    sampled.format = prosper::gpu::DataFormat::Unorm8;
    sampled.num_components = 4;
    sampled.alpha_is_on_msb = true;
    sampled.width = kWidth;
    sampled.height = kHeight;
    sampled.tile_mode = static_cast<uint32_t>(prosper::gpu::TileMode::Sw64KbRX);
    sampled.compression_enabled = true;
    sampled.metadata_addr = reinterpret_cast<uint64_t>(metadata_.data());
    ASSERT_EQ(prosper::gpu::gpu_capture_dcc_metadata_footprint(sampled), kMetadataBytes);
    prosper::gpu::DrawItem draw;
    draw.prt = std::make_shared<prosper::gpu::ShaderResourceTable>();
    draw.prt->resources.push_back(sampled);
    EXPECT_EQ(span(draw), 1u);
    EXPECT_EQ(surface().uniform_color, (std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
    EXPECT_FALSE(surface().dcc_metadata_dirty);
}

// What the span does not load, it does not clear.
TEST_F(DccClearRenderTarget, ASpanThatDoesNotWriteTheTargetLeavesItPending) {
    fast_clear(kClear0001);
    EXPECT_EQ(span(draw_to(kTarget, 0)), 0u);   // bound, every channel masked
    EXPECT_EQ(span(draw_to(kTarget + 0x10000, 0xf)), 0u);   // another target
    EXPECT_EQ(span(draw_to(kTarget, 0xf, kWidth * 2)), 0u);   // another extent at this address
    EXPECT_FALSE(surface().has_uniform_color);
    EXPECT_TRUE(surface().dcc_metadata_dirty);
}

// Metadata that is not one clear code throughout is rendered content, not a clear.
TEST_F(DccClearRenderTarget, MetadataThatIsNotOneClearCodeIsNotAClear) {
    fast_clear(kClear0001);
    metadata_.back() = 0xff;
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 0u);
    fast_clear(0x20);   // the clear-register code, which this decode does not resolve
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 0u);
    EXPECT_FALSE(surface().has_uniform_color);
    EXPECT_TRUE(surface().dcc_metadata_dirty);
}

// The decode reads the code as the registered descriptor does: a three-component view has no
// alpha to clear, and one with alpha in the low byte puts it first.
TEST_F(DccClearRenderTarget, TheRegisteredDescriptorDecidesWhereAlphaIs) {
    prosper::frontend::note_rtt_dcc_descriptor(cache_[kTarget], descriptor(4, false),
                                               metadata_.size());
    fast_clear(kClear0001);
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 1u);
    EXPECT_EQ(surface().uniform_color, (std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f}));

    prosper::frontend::note_rtt_dcc_descriptor(cache_[kTarget], descriptor(3, false),
                                               metadata_.size());
    fast_clear(kClear0001);
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 1u);
    EXPECT_EQ(surface().uniform_color, (std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
}

// Under PROSPER_RENDER_SCALE the retained target is smaller than the guest's, and the draw still
// names the guest extent. That reduction, and no other mismatch, is the same surface.
TEST_F(DccClearRenderTarget, ATargetRetainedAtTheRenderScaleIsTheSameSurface) {
    cache_[kTarget].w = kWidth / 4;
    cache_[kTarget].h = kHeight / 4;
    fast_clear(kClear0001);
    EXPECT_EQ(span(draw_to(kTarget, 0xf), 1), 0u);
    EXPECT_EQ(span(draw_to(kTarget, 0xf), 2), 0u);
    EXPECT_TRUE(surface().dcc_metadata_dirty);
    EXPECT_EQ(span(draw_to(kTarget, 0xf), 4), 1u);
    EXPECT_EQ(surface().uniform_color, (std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
}

// The recorded facts belong to the surface the descriptor described. A narrower target that now
// lives at the address, and was never sampled, is not cleared from them.
TEST_F(DccClearRenderTarget, FactsFromAnotherSurfaceAtTheAddressAreNotApplied) {
    cache_[kTarget].w = kWidth / 2;
    fast_clear(kClear0001);
    EXPECT_EQ(span(draw_to(kTarget, 0xf, kWidth / 2)), 0u);
    EXPECT_FALSE(surface().has_uniform_color);
    EXPECT_TRUE(surface().dcc_metadata_dirty);
}

// Slot 1 starts from a retained uniform colour as slot 0 does. The slots above it do not, so
// taking their clear here would only lose it.
TEST_F(DccClearRenderTarget, OnlyASlotThePassCanStartFromIsCleared) {
    fast_clear(kClear0001);
    prosper::gpu::DrawItem third;
    third.color_targets[2].base = kTarget;
    third.color_targets[2].width = kWidth;
    third.color_targets[2].height = kHeight;
    third.ps.color_targets[2].write_mask = 0xf;
    ASSERT_EQ(prosper::frontend::mrt_write_mask(third, 2), 0xfu);
    EXPECT_EQ(span(third), 0u);
    EXPECT_TRUE(surface().dcc_metadata_dirty);

    prosper::gpu::DrawItem second;
    second.color1_base = kTarget;
    second.color1_width = kWidth;
    second.color1_height = kHeight;
    second.ps.color1_write_mask = 0xf;
    EXPECT_EQ(span(second), 1u);
    EXPECT_EQ(surface().uniform_color, (std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
}

// On a volume target the same flag also marks a partial colour-plane write, which is not a clear.
TEST_F(DccClearRenderTarget, AVolumeTargetIsLeftAlone) {
    fast_clear(kClear0001);
    cache_[kTarget].volume_depth = 4;
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 0u);
    cache_[kTarget].volume_depth = 0;
    cache_[kTarget].volume_guest_bytes = 4096;
    EXPECT_EQ(span(draw_to(kTarget, 0xf)), 0u);
    EXPECT_FALSE(surface().has_uniform_color);
}

// A target no descriptor has sampled has no metadata range on record, so there is nothing to read.
TEST_F(DccClearRenderTarget, ATargetWithNoRegisteredMetadataIsLeftAlone) {
    RttSurf& fresh = cache_[kTarget + 0x10000];
    fresh.w = kWidth;
    fresh.h = kHeight;
    fresh.format = VK_FORMAT_R8G8B8A8_UNORM;
    fresh.dcc_metadata_dirty = true;
    EXPECT_EQ(span(draw_to(kTarget + 0x10000, 0xf)), 0u);
    EXPECT_FALSE(fresh.has_uniform_color);
}
}   // namespace
