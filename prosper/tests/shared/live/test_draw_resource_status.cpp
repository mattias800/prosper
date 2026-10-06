// Exercise the shipping resource builder's rejection/skip/continuation contract before extraction.
// These are seeded reflection-memo metadata arms: they observe real builder outputs and census/
// timing hooks, not shader reflection, rendered pixels or GPU completion.
#include "shared/live/submit_renderer/guest_reads.hpp"   // safe_span
#include "shared/live/submit_renderer/image_resources.hpp"

#include "diagnostics/transfer_pressure.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::frontend::submit_renderer;

static int failures = 0;
static void check(bool ok, const char* arm, const char* message) {
    std::printf("[%s] %s: %s\n", ok ? "ok" : "FAIL", arm, message);
    std::fflush(stdout);
    failures += !ok;
}

// The builder's native TLS reference aliases bind once. Keep the same fixture-owned objects
// alive for every call, and reset their contents rather than binding to successive stack owners.
template <class T> using Owned = std::remove_cvref_t<T>;
// One ordered field list keeps fixture ownership and its designated reference bindings aligned.
#define IMAGE_STATUS_CONTEXT_FIELDS(FIELD) \
    FIELD(g_rtt) \
    FIELD(g_pass_log_submit) \
    FIELD(invalidate_ds) \
    FIELD(frame_no) \
    FIELD(rtt_on) \
    FIELD(live_gpu_targets) \
    FIELD(write_watch_promotion_budget) \
    FIELD(pinned_renderer_mip_targets) \
    FIELD(g_this_submit) \
    FIELD(rtt_log) \
    FIELD(pending_timing) \
    FIELD(validation_census) \
    FIELD(timing_enabled) \
    FIELD(render_timing_detail) \
    FIELD(texstore) \
    FIELD(texstore_pinned) \
    FIELD(texstore_used) \
    FIELD(decoded_textures) \
    FIELD(decode_span_ordinal) \
    FIELD(use_direct_buffer_views) \
    FIELD(use_tracked_buffer_membership_cache) \
    FIELD(persistent_decoded_textures) \
    FIELD(persistent_decoded_texture_bytes) \
    FIELD(decode_generation) \
    FIELD(retired_submit_pixels) \
    FIELD(retired_submit_bytes) \
    FIELD(persistent_texture_id) \
    FIELD(persistent_validation_scratch) \
    FIELD(persistent_decode_limit) \
    FIELD(resource_hash_w) \
    FIELD(resource_hash_h) \
    FIELD(rtt_injection_cache) \
    FIELD(reflect_memo) \
    FIELD(compact_buffer_resources) \
    FIELD(reserve_frame_resources) \
    FIELD(depth_array_census) \
    FIELD(depth_array_snapshots) \
    FIELD(depth_array_snapshot_bytes) \
    FIELD(depth_array_snapshot_admitted) \
    FIELD(reuse_depth_arrays_across_draws) \
    FIELD(compact_depth_array_snapshots) \
    FIELD(gpu_depth_array_snapshots) \
    FIELD(disable_guest_depth_layers) \
    FIELD(depth_array_guest_scans) \
    FIELD(depth_array_guest_scan_epoch) \
    FIELD(gpu_depth_cube_snapshots) \
    FIELD(depth_cube_gpu_snapshots)

struct FixtureState {
#define IMAGE_STATUS_OWN_FIELD(name) Owned<decltype(DrawResourceContext::name)> name{};
    IMAGE_STATUS_CONTEXT_FIELDS(IMAGE_STATUS_OWN_FIELD)
#undef IMAGE_STATUS_OWN_FIELD
    RenderTiming previous_timing;
    uint64_t previous_census_refs = 0;
    uint64_t previous_texture_references = 0;
    uint64_t expected_texture_references = 0;

    FixtureState() {
        timing_enabled = true;
        compact_buffer_resources = true;
        reserve_frame_resources = true;
    }

    DrawResourceContext context() {
        return {
#define IMAGE_STATUS_BIND_FIELD(name) .name = name,
            IMAGE_STATUS_CONTEXT_FIELDS(IMAGE_STATUS_BIND_FIELD)
#undef IMAGE_STATUS_BIND_FIELD
        };
    }

    void reset() {
        ++g_this_submit;
        ++decode_generation;
        ++decode_span_ordinal;
        previous_timing = pending_timing;
        previous_texture_references =
            prosper::diagnostics::perf::thread_texture_references();
        expected_texture_references = 0;
        previous_census_refs = 0;
        for (const auto& klass : prosper::frontend::texture_reference_census().cls)
            previous_census_refs += klass.refs;
        texstore_used = 0;
        decoded_textures.clear();
        texstore.clear();
        texstore_pinned.clear();
        reflect_memo.clear();
    }
};
#undef IMAGE_STATUS_CONTEXT_FIELDS

struct Input {
    ShaderResource resource;
    SpirvDescriptorBinding descriptor;
};

static std::array<uint8_t, 16> ImageBytes{
    17, 34, 51, 255, 68, 85, 102, 255,
    119, 136, 153, 255, 170, 187, 204, 255};
static std::array<uint32_t, 4> BufferWords{0x12345678, 2, 3, 4};

static Input ordinary_image(uint32_t binding = 21) {
    Input input{};
    auto& r = input.resource;
    r.cls = ResourceClass::Texture;
    r.format = DataFormat::Unorm8;
    r.num_components = 4;
    r.binding = binding;
    r.img_dim = 1;
    r.width = 2; r.height = 2; r.depth = 1;
    r.sample_count = 1;
    r.declared_mip_levels = 1;
    r.gpu_addr = 0x71000000 + binding * 0x10000ull;
    r.size = ImageBytes.size();
    r.host_data = ImageBytes.data();
    r.host_data_size = ImageBytes.size();
    r.swizzle[0] = 4; r.swizzle[1] = 5;
    r.swizzle[2] = 6; r.swizzle[3] = 7;
    auto& d = input.descriptor;
    d.binding = binding;
    d.kind = SpirvDescriptorKind::CombinedImageSampler;
    d.readable = true;
    d.normalized_sampling = true;
    d.sampled_float = true;
    d.image_numeric_class = SpirvImageNumericClass::Float;
    d.image_dim = 1;
    return input;
}

static Input ordinary_buffer() {
    Input input{};
    auto& r = input.resource;
    r.cls = ResourceClass::ConstantBuffer;
    r.format = DataFormat::Float32;
    r.num_components = 4;
    r.binding = 20;
    r.gpu_addr = 0x72000000;
    r.size = sizeof(BufferWords);
    r.host_data = reinterpret_cast<uint8_t*>(BufferWords.data());
    r.host_data_size = sizeof(BufferWords);
    auto& d = input.descriptor;
    d.binding = r.binding;
    d.kind = SpirvDescriptorKind::StorageBuffer;
    d.readable = true;
    d.required_bytes = sizeof(BufferWords);
    return input;
}

// A buffer binding in GUEST memory: `declared` bytes at `address`, of which the shader's own
// layout needs `required`.
static Input guest_buffer(uint64_t address, uint32_t declared, uint64_t required) {
    Input input = ordinary_buffer();
    input.resource.gpu_addr = address;
    input.resource.size = declared;
    input.resource.host_data = nullptr;
    input.resource.host_data_size = 0;
    input.descriptor.required_bytes = required;
    return input;
}

static Input shape_rejection() {
    Input input = ordinary_image(4);
    input.resource.format = DataFormat::Float32;
    input.resource.img_dim = 5;
    input.resource.depth = 2;
    // A real Float32 layered descriptor with a 1D reflected image reaches ArrayFloat32Shape.
    input.descriptor.image_dim = 0;
    input.descriptor.image_arrayed = true;
    return input;
}

static Input budget_rejection() {
    Input input = ordinary_image(5);
    input.resource.format = DataFormat::Float32;
    input.resource.img_dim = 5;
    input.resource.depth = 2049;
    input.descriptor.image_arrayed = true;
    // The production 2,048-layer ceiling refuses this before allocation/device format queries.
    return input;
}

static Input msaa_input(bool short_source) {
    Input input = ordinary_image(4);
    auto& r = input.resource;
    r.format = DataFormat::Float32;
    r.num_components = 1;
    r.img_dim = 6;
    r.sample_count = short_source ? 4 : 2;
    r.width = 8; r.height = 8;
    r.tile_mode = 24;
    r.host_data_size = 1;
    auto& d = input.descriptor;
    d.normalized_sampling = false;
    d.texel_access = true;
    d.image_arrayed = true;
    // The supported shape has a deliberately one-byte captured source. The other arm changes
    // only sample count, so it fails the earlier exact-shape gate instead of the copy gate.
    return input;
}

static BuiltFrameResources build(FixtureState& state, const std::vector<Input>& vertex,
                                const std::vector<Input>& fragment) {
    state.reset();
    check(state.decoded_textures.empty(), "setup",
          "every arm starts cache-cold so decoded reuse cannot bypass a refusal");
    ShaderResourceTable vrt, prt;
    auto seed = [&](const std::vector<Input>& inputs, uint32_t set, SpirvShaderStage stage,
                    uint64_t identity, ShaderResourceTable& table) {
        ReflectMemoEntry entry;
        entry.set = set; entry.stage = stage;
        for (const Input& input : inputs) {
            if (input.resource.cls == ResourceClass::Texture)
                ++state.expected_texture_references;
            table.resources.push_back(input.resource);
            auto descriptor = input.descriptor;
            descriptor.set = set; descriptor.stage = stage;
            entry.report.descriptors.push_back(descriptor);
        }
        state.reflect_memo.emplace(identity, std::move(entry));
    };
    seed(vertex, 0, SpirvShaderStage::Vertex, 1, vrt);
    seed(fragment, 1, SpirvShaderStage::Fragment, 2, prt);
    DrawItem draw;
    draw.vs_identity = 1; draw.fs_identity = 2;
    auto context = state.context();
    return build_draw_frame_resources(context, draw, &vrt, &prt, nullptr);
}

static void check_tail(const char* arm, const BuiltFrameResources& result,
                       const FixtureState& state) {
    // Both successful resources follow every refused input in the fragment-stage table.
    check(result.full.size() == 1 && result.buffers.size() == 1, arm,
          "only the later image and buffer survive; refused bindings are suppressed");
    check(result.order == std::vector<uint32_t>{kCompactBufferResourceBit, 0}, arm,
          "the later buffer/image preserve the original resource order");
    if (result.full.size() == 1) {
        const auto& image = result.full[0];
        check(image.binding == 21 && image.set == 1 && image.tw == 2 && image.th == 2 &&
                  image.tex_rgba && std::memcmp(image.tex_rgba, ImageBytes.data(),
                                                ImageBytes.size()) == 0,
              arm, "the later image reaches the real fill/emplace tail with exact hosted bytes");
    }
    if (result.buffers.size() == 1) {
        const auto& buffer = result.buffers[0];
        check(buffer.binding == 20 && buffer.set == 1 && buffer.buffer_word_count() == 4 &&
                  buffer.buffer_words_data() &&
                  std::memcmp(buffer.buffer_words_data(), BufferWords.data(),
                              sizeof(BufferWords)) == 0,
              arm, "the later compact buffer reaches the real materialization/emplace tail");
    }
    const auto& census = prosper::frontend::texture_reference_census();
    uint64_t refs = 0;
    for (const auto& klass : census.cls) refs += klass.refs;
    check(refs == state.previous_census_refs + 1 && census.this_submit.size() == 1, arm,
          "only the successful image increments the real texture-reference census");
    check(prosper::diagnostics::perf::thread_texture_references() ==
              state.previous_texture_references + state.expected_texture_references,
          arm, "the reference constructor counts refused and successful texture inputs");
    const auto& timing = state.pending_timing;
    const auto& before = state.previous_timing;
    check(timing.textures == before.textures + 1 && timing.buffers == before.buffers + 1 &&
              timing.texture_bytes == before.texture_bytes + 16 &&
              timing.buffer_bytes == before.buffer_bytes + 16 &&
              timing.compact_buffer_resources == before.compact_buffer_resources + 1 &&
              timing.full_buffer_resources == before.full_buffer_resources,
          arm, "only successful resources reach the real timing counters");
    check(timing.build_reflect_memo_hits == before.build_reflect_memo_hits + 2 &&
              timing.build_reflect_calls == before.build_reflect_calls, arm,
          "both stages actually consume the seeded manifests on the production memo path");
}

int main(int argc, char** argv) {
    check(prosper::frontend::TextureReferenceCensus::enabled(), "setup",
          "the census is armed before its first cached read");
    static FixtureState state;
    state.texstore.resize(2);
    state.texstore_pinned = {false, true};
    auto slot_context = state.context();
    check(acquire_image_texstore_slot(slot_context) == 0 && state.texstore_used == 1,
          "slot-counter", "the returned old index advances the referenced caller counter");
    check(acquire_image_texstore_slot(slot_context) == 2 && state.texstore_used == 3,
          "slot-counter", "the next acquisition skips a pinned slot and advances past its result");
    check(state.texstore.size() == 3 && state.texstore_pinned.size() == 3 &&
              state.texstore_pinned[1] && !state.texstore_pinned[2],
          "slot-counter", "pool growth preserves the existing pin and initializes the new slot");
    const auto buffer = ordinary_buffer();
    const auto image = ordinary_image();

    {
        const auto result = build(state, {}, {buffer, image});
        check(result.complete && result.drop_reason == DropReason::Unattributed, "success",
              "ordinary resources keep the draw complete and unattributed");
        check_tail("success", result, state);
    }
    {
        const auto result = build(state, {}, {shape_rejection(), buffer, image});
        check(!result.complete && result.drop_reason == DropReason::ArrayFloat32Shape,
              "early-reject", "the real shape gate rejects with its exact reason");
        check_tail("early-reject", result, state);
    }
    {
        const auto result = build(state, {shape_rejection()}, {budget_rejection(), buffer, image});
        check(!result.complete && result.drop_reason == DropReason::ArrayFloat32Shape,
              "shape-first", "the first real rejection wins across shader stages");
        check_tail("shape-first", result, state);
    }
    {
        const auto result = build(state, {budget_rejection()}, {shape_rejection(), buffer, image});
        check(!result.complete && result.drop_reason == DropReason::ArrayShapeBudget,
              "budget-first", "reversing real rejection order changes only the first reason");
        check_tail("budget-first", result, state);
    }
    for (bool short_source : {false, true}) {
        const auto msaa = msaa_input(short_source);
        const char* arm = short_source ? "msaa-short-source" : "msaa-unsupported-shape";
        std::printf("[arm-begin] %s\n", arm);
        std::fflush(stdout);
        check(prosper::frontend::sampled_msaa_fetch_shape_supported(
                  msaa.resource, false, true) == short_source,
              arm, "the two inputs distinguish the early shape and later copy gates");
        const auto result = build(state, {}, {msaa, buffer, image});
        check(result.complete && result.drop_reason == DropReason::Unattributed, arm,
              "the real MSAA refusal skips the binding without rejecting the builder result");
        check(state.decoded_textures.size() == 1, arm,
              "the refused MSAA source never creates a decoded entry");
        check_tail(arm, result, state);
        std::printf("[arm-end] %s\n", arm);
        std::fflush(stdout);
    }
    // A draw buffer whose DECLARED range runs past the end of mapped guest memory (#4556). A
    // vertex attribute declared as "the whole vertex buffer, from my offset" is one: it overshoots
    // the buffer by that offset. The readable start is borrowed in place; only what cannot be
    // borrowed is still copied into a zero-filled vector of the declared size.
    {
        using prosper::diagnostics::Transfer;
        using prosper::diagnostics::transfer_bytes;
        constexpr uint64_t kMapped = 0x10000, kReadable = 0x2000;
        constexpr uint32_t kDeclared = 0x100000;
        prosper::register_builtin_hle();
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        uint64_t mapped = 0;
        check(map &&
                  map(reinterpret_cast<uint64_t>(&mapped), kMapped, 3, 0,
                      reinterpret_cast<uint64_t>("draw-buffer-view"), 0) == 0 &&
                  mapped,
              "guest-buffer", "the arms below have tracked guest memory to point into");
        if (mapped) {
            auto* words = reinterpret_cast<uint32_t*>(mapped);
            for (uint32_t i = 0; i < kMapped / 4; ++i) words[i] = 0xb0000000u | i;
            const uint64_t start =
                mapped + kMapped - kReadable;   // readable for kReadable, no more
            const uint32_t first_word =
                0xb0000000u | static_cast<uint32_t>((kMapped - kReadable) / 4);
            check(safe_span(start, kDeclared) == kReadable && safe_span(mapped, 0x1000) == 0x1000,
                  "guest-buffer", "the mapping ends where the arms assume, and nothing follows it");
            state.use_direct_buffer_views = true;
            // No device exists in this process; say what a certified one says (see the
            // "uncertified-device" arm for the other answer).
            prosper::gpu::device_reads_zero_past_a_binding() = true;
            // The comparison arm of the measurement is its own ctest case: the same binary with
            // the switch set and `--expect-copy`. The two must agree, so a switch left exported
            // in the shell cannot quietly turn the default case into the comparison one.
            // NOLINTNEXTLINE(concurrency-mt-unsafe): one read in a single-threaded test
            const bool borrows = std::getenv("PROSPER_NO_SHORT_BUFFER_VIEW") == nullptr;
            const bool expect_copy = argc > 1 && std::strcmp(argv[1], "--expect-copy") == 0;
            check(borrows != expect_copy, "guest-buffer",
                  "PROSPER_NO_SHORT_BUFFER_VIEW is set exactly when --expect-copy is passed");
            const auto staged = [] { return transfer_bytes(Transfer::DrawBufferStage); };
            const auto one = [&](const char* arm, const Input& input) {
                auto result = build(state, {}, {input});
                check(result.complete && result.buffers.size() == 1, arm,
                      "the binding is kept and the draw stays complete");
                return result;
            };
            const auto borrowed = [&](const auto& result, uint64_t address, uint64_t bytes) {
                if (result.buffers.size() != 1) return false;
                const auto& buffer = result.buffers[0];
                return buffer.dwords.empty() &&
                       reinterpret_cast<uintptr_t>(buffer.buffer_words_data()) == address &&
                       buffer.buffer_word_count() == bytes / 4 &&
                       buffer.direct_guest_buffer_addr == address;
            };
            const auto copied = [&](const auto& result, uint32_t word0, size_t zero_from_word) {
                if (result.buffers.size() != 1) return false;
                const auto& buffer = result.buffers[0];
                return buffer.dwords.size() == kDeclared / 4 && !buffer.direct_guest_buffer_addr &&
                       buffer.dwords[0] == word0 && buffer.dwords[zero_from_word - 1] != 0 &&
                       buffer.dwords[zero_from_word] == 0 && buffer.dwords.back() == 0;
            };
            {
                const uint64_t before = staged();
                const auto result = one("over-declared", guest_buffer(start, kDeclared, 16));
                if (borrows) {
                    check(borrowed(result, start, kReadable), "over-declared",
                          "the readable start is borrowed in place, exactly as long as it is "
                          "readable");
                    check(staged() == before, "over-declared", "a borrowed range stages no bytes");
                } else {
                    check(copied(result, first_word, kReadable / 4), "over-declared",
                          "with the switch set the declared range is copied and zero-filled as "
                          "before");
                    check(staged() == before + kDeclared, "over-declared",
                          "the copy is charged at its declared size");
                }
            }
            {
                // The shader's own layout reaches past what is readable: a binding shorter than
                // that would not be a valid one, so this still takes the copy.
                const uint64_t before = staged();
                const auto result =
                    one("needs-more-than-readable", guest_buffer(start, kDeclared, kReadable + 4));
                check(copied(result, first_word, kReadable / 4), "needs-more-than-readable",
                      "a start shorter than the shader's layout is not borrowed");
                check(staged() == before + kDeclared, "needs-more-than-readable",
                      "the copy is charged at its declared size");
            }
            {
                // A binding the shader WRITES: a store past the readable bytes landed in the
                // copy, and a shorter binding has nowhere for it to land.
                Input written = guest_buffer(start, kDeclared, 16);
                written.descriptor.writable = true;
                const auto result = one("writable", written);
                check(copied(result, first_word, kReadable / 4), "writable",
                      "a binding the shader writes is copied at its declared size");
                Input atomic = guest_buffer(start, kDeclared, 16);
                atomic.descriptor.atomic_access = true;
                check(copied(one("atomic", atomic), first_word, kReadable / 4), "atomic",
                      "a binding the shader updates atomically is copied at its declared size");
            }
            {
                // A device that does not promise zero past a binding's end: the explicit zeros
                // of the copy are the only way to keep them.
                prosper::gpu::device_reads_zero_past_a_binding() = false;
                const auto result = one("uncertified-device", guest_buffer(start, kDeclared, 16));
                prosper::gpu::device_reads_zero_past_a_binding() = true;
                check(copied(result, first_word, kReadable / 4), "uncertified-device",
                      "without the device's zero guarantee the declared range is copied");
            }
            {
                // Two bytes in: not a word-aligned source, so no view of it can be handed out.
                const auto result = one("misaligned", guest_buffer(start + 2, kDeclared, 16));
                check(result.buffers.size() == 1 && !result.buffers[0].dwords.empty() &&
                          !result.buffers[0].direct_guest_buffer_addr,
                      "misaligned", "an unaligned start is copied, never borrowed");
            }
            {
                // The control: a range that is readable to its declared end was always borrowed
                // whole, and still is at its declared length.
                const uint64_t before = staged();
                const auto result = one("fully-readable", guest_buffer(mapped, 0x1000, 16));
                check(borrowed(result, mapped, 0x1000), "fully-readable",
                      "a fully readable range is borrowed at its declared length");
                check(staged() == before, "fully-readable", "and stages no bytes");
            }
            {
                // A CAPTURED source that is shorter than its declared size, at an address where
                // guest memory happens to be readable too. Its bytes are the capture's, so the
                // guest's must not be borrowed in their place.
                Input captured = guest_buffer(start, kDeclared, 16);
                captured.resource.host_data = reinterpret_cast<uint8_t*>(BufferWords.data());
                captured.resource.host_data_size = sizeof(BufferWords);
                const auto result = one("captured-short", captured);
                check(copied(result, BufferWords[0], BufferWords.size()), "captured-short",
                      "a short captured source is copied from the capture, not borrowed from guest "
                      "memory");
            }
            state.use_direct_buffer_views = false;
            {
                // With views switched off altogether every guest buffer is copied, and that
                // route charges its copy too.
                const uint64_t before = staged();
                const auto result = one("views-off", guest_buffer(start, kDeclared, 16));
                check(copied(result, first_word, kReadable / 4), "views-off",
                      "without buffer views the declared range is copied and zero-filled");
                check(staged() == before + kDeclared, "views-off",
                      "and the copy is charged at its declared size");
            }
        }
    }
    std::printf("draw resource status: %d failures (seeded builder metadata; no rendered-pixel claim)\n",
                failures);
    return failures ? 1 : 0;
}
