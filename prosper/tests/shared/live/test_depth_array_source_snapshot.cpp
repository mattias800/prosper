// Retained depth arrays preserve Float32 contents and publish only complete, ordered snapshots.
#include "fixtures/render_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include <bit>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace prosper::gpu;
using namespace prosper::test;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

template<class Action> static std::string capture_stderr(Action action) {
#ifdef _WIN32
    const auto fd = &_fileno;
    const auto duplicate = &_dup;
    const auto redirect = &_dup2;
    const auto close_fd = &_close;
#else
    const auto fd = &fileno;
    const auto duplicate = &dup;
    const auto redirect = &dup2;
    const auto close_fd = &close;
#endif
    FILE* capture = std::tmpfile();
    if (!capture) { check(false, "diagnostic capture file opens"); return {}; }
    std::fflush(stderr);
    const int saved = duplicate(fd(stderr));
    if (saved < 0 || redirect(fd(capture), fd(stderr)) < 0) {
        check(false, "diagnostic capture redirects stderr");
        if (saved >= 0) close_fd(saved);
        std::fclose(capture);
        return {};
    }
    action();
    std::fflush(stderr);
    check(redirect(saved, fd(stderr)) >= 0, "diagnostic capture restores stderr");
    close_fd(saved);
    std::rewind(capture);
    std::string text;
    char buffer[1024];
    while (const size_t count = std::fread(buffer, 1, sizeof(buffer), capture))
        text.append(buffer, count);
    std::fclose(capture);
    std::fputs(text.c_str(), stderr);
    return text;
}

int main() {
    constexpr uint32_t W = 8, H = 8, Layers = 4;
    constexpr uint64_t Base = 0x7d400000;
    constexpr size_t Pixels = W * H;
    using Status = PersistentDsDepthArrayStatus;
    std::string error;
    std::vector<float> output{19.0f, 23.0f};
    const auto sentinel = output;
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::NoIdentity && output == sentinel,
          "absent retained identity permits guest fallback without modifying output");

    auto seed_layer = [&](uint32_t layer, float value) {
        BackendPersistentResourceGuard guard;
        GpuCaptureDsSeed seed;
        seed.depth_read_base = seed.depth_write_base = Base;
        seed.width = W; seed.height = H; seed.slice = layer;
        seed.format = GpuCaptureDsFormat::D32Float; seed.depth_valid = true;
        seed.depth.resize(Pixels * sizeof(float));
        for (size_t i = 0; i < Pixels; ++i)
            std::memcpy(seed.depth.data() + i * sizeof(float), &value, sizeof(value));
        const bool ok = restore_persistent_ds_image(seed, error);
        if (!ok) std::fprintf(stderr, "seed: %s\n", error.c_str());
        check(ok, "real Vulkan depth layer restored");
        if (ok) for (auto& [key, image] : persistent_ds_cache())
            if (key.dr == Base && key.slice == layer)
                note_persistent_ds_depth_write(image, true, true);
        return ok;
    };
    std::array<float, Layers> expected{0.1250457763671875f, 0.25006103515625f,
                                       0.5000762939453125f, 0.750091552734375f};
    for (uint32_t layer = 0; layer < Layers - 1; ++layer)
        if (!seed_layer(layer, expected[layer])) return 1;
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == sentinel,
          "missing final layer refuses stale guest fallback and partial publication");
    if (!seed_layer(3, expected[3])) return 1;
    auto exact = [&] {
        if (output.size() != Pixels * Layers) return false;
        for (uint32_t layer = 0; layer < Layers; ++layer)
            for (size_t i = 0; i < Pixels; ++i)
                if (std::bit_cast<uint32_t>(output[layer * Pixels + i]) !=
                    std::bit_cast<uint32_t>(expected[layer])) return false;
        return true;
    };
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Ready && exact(),
          "four retained layers preserve every Float32 bit, including values lost by half or RGBA8");
    expected[2] = 0.6251068115234375f;
    if (!seed_layer(2, expected[2])) return 1;
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Ready && exact(), "rewritten depth layer replaces the prior snapshot");
    std::vector<float> subset;
    check(read_persistent_ds_depth_array(Base, W, H, 1, 2, subset, error) == Status::Ready &&
              subset.size() == 2 * Pixels && subset.front() == expected[1] &&
              subset.back() == expected[2], "nonzero first layer selects the requested subrange");
    const auto before_failure = output;
    depth_array_readback_failure_after_layers() = 2;
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_failure,
          "injected later-layer readback failure preserves the whole previous snapshot");
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Ready && exact(), "unchanged identity retries successfully after readback failure");
    check(read_persistent_ds_depth_array(Base, W + 1, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_failure,
          "retained base with wrong extent cannot fall back to guest memory");
    check(read_persistent_ds_depth_array(Base, W, H, 0, 2049, output, error) ==
              Status::Unavailable && output == before_failure,
          "oversized layer count is rejected before allocation");
    check(read_persistent_ds_depth_array(Base, UINT32_MAX, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_failure,
          "oversized dimensions are rejected before allocation");

    // Metadata adversaries use no aliased Vulkan handles. The newer invalid key must beat the
    // real older image; an incompatible format must never be interpreted as Float32 bytes.
    PersistentDsKey adversary{Base, Base, 0, 0, 1, W, H, VK_FORMAT_D32_SFLOAT, 0};
    {
        BackendPersistentResourceGuard guard;
        persistent_ds_cache()[adversary] = {};
    }
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_failure,
          "unversioned invalid alias cannot be proven older than retained pixels");
    {
        BackendPersistentResourceGuard guard;
        note_persistent_ds_depth_write(persistent_ds_cache()[adversary], true, true);
    }
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_failure,
          "newer invalid identity never resurrects an older valid layer");
    {
        BackendPersistentResourceGuard guard;
        persistent_ds_cache().erase(adversary);
        adversary.fmt = VK_FORMAT_D16_UNORM;
        persistent_ds_cache()[adversary] = {};
    }
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_failure,
          "incompatible retained format is refused despite an older readable Float32 identity");
    {
        BackendPersistentResourceGuard guard;
        persistent_ds_cache().erase(adversary);
        adversary.fmt = VK_FORMAT_D32_SFLOAT_S8_UINT;
        persistent_ds_cache()[adversary] = {};
    }
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_failure,
          "overlapping depth-only and combined identities are conservatively ambiguous");
    {
        BackendPersistentResourceGuard guard;
        persistent_ds_cache().erase(adversary);
    }

    const uint32_t vs_words[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u,
        0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    const uint32_t fs_words[]{0x7E0002F2u, 0x7E020280u, 0x7E040280u,
        0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u};
    ResolvedPipelineState producer_state{};
    producer_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    producer_state.depth_test_enable = producer_state.depth_write_enable = true;
    producer_state.depth_compare_op = VK_COMPARE_OP_ALWAYS;
    producer_state.depth_read_base = producer_state.depth_write_base = Base;
    producer_state.db_depth_view = 1u | (1u << 13);
    producer_state.has_viewport = true;
    producer_state.viewport_w = W; producer_state.viewport_h = H;
    expected[1] = 0.8751220703125f;
    producer_state.min_depth = producer_state.max_depth = expected[1];
    BackendDraw producer;
    producer.vs = recompile_vertex(vs_words, std::size(vs_words));
    producer.fs = recompile_fragment(fs_words, std::size(fs_words));
    producer.ps = &producer_state; producer.vcount = 3;
    check(!producer.vs.empty() && !producer.fs.empty(), "depth producer shaders compile");
    BackendSubmissionBatch batch;
    (void)render_draws_rgba({producer}, W, H, nullptr, nullptr, true, nullptr,
                          nullptr, nullptr, nullptr, &batch, false, nullptr, false);
    check(batch.pending(), "depth producer remains unsubmitted until its consumer");
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error, &batch) ==
              Status::Ready && !batch.pending() && exact(),
          "array snapshot flushes the pending producer and observes its exact new depth");

    // Sample every gathered layer through the real Float32 array uploader. Amplify the difference
    // before the final RGBA8 readback, so Float16 conversion cannot hide in presentation rounding.
    std::vector<float> rgba(output.size() * 4, 1.0f);
    for (size_t i = 0; i < output.size(); ++i)
        rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = output[i];
    FrameResource resource{};
    resource.binding = 4; resource.set = 1;
    resource.img_dim = 5; resource.guest_array = true;
    resource.tw = W; resource.th = H; resource.sample_count = Layers;
    resource.texture_format = VK_FORMAT_R32G32B32A32_SFLOAT;
    resource.tex_rgba = reinterpret_cast<const uint8_t*>(rgba.data());
    resource.tex_byte_size = rgba.size() * sizeof(float);
    resource.min_filter = resource.mag_filter = 0;
    ShaderResourceTable table;
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture; texture.format = DataFormat::Float32;
    texture.binding = 4; texture.sgpr_base = 8; texture.img_dim = 5;
    texture.width = W; texture.height = H; texture.depth = Layers; texture.num_components = 4;
    table.resources.push_back(texture);
    ResolvedPipelineState sample_state{};
    sample_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    sample_state.color_write_mask = 15;
    for (uint32_t layer = 0; layer < Layers; ++layer) {
        std::vector<uint32_t> ps{0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
            0x7e0402ffu, std::bit_cast<uint32_t>(float(layer)), 0x7e060280u,
            0xf0900f28u, 0x00820000u, 0x080000ffu, std::bit_cast<uint32_t>(expected[layer]),
            0x100000ffu, std::bit_cast<uint32_t>(65536.0f), 0x060000ffu, 0x3f000000u,
            0xf800000fu, 0x00000000u, 0xbf810000u};
        BackendDraw sample;
        sample.vs = producer.vs; sample.fs = recompile_fragment(ps.data(), ps.size(), &table);
        sample.ps = &sample_state; sample.vcount = 3; sample.R = {resource};
        const auto pixels = render_draws_rgba({sample}, W, H);
        const size_t center = (W * (H / 2) + W / 2) * 4;
        check(pixels.size() == Pixels * 4 && pixels[center] >= 127 && pixels[center] <= 128,
              "sampler consumes the precise retained Float32 array layer");
    }

    // Exercise the production resource builder as well as the backend helper. Captured guest
    // bytes intentionally disagree with every retained plane, and remain unchanged on rewrites.
    prosper::register_builtin_hle();
    prosper::frontend::register_live_renderer(".", false);
    std::vector<uint8_t> stale_guest(Pixels * Layers * sizeof(float), 0);
    auto live_table = std::make_shared<ShaderResourceTable>();
    texture.num_components = 1;
    texture.gpu_addr = Base; texture.size = stale_guest.size();
    texture.host_data = stale_guest.data(); texture.host_data_size = stale_guest.size();
    texture.layer_stride_bytes = Pixels * sizeof(float);
    texture.linear_row_pitch_bytes = W * sizeof(float);
    texture.mag_filter = texture.min_filter = 0;
    texture.swizzle[0] = 4; texture.swizzle[1] = 0;
    texture.swizzle[2] = 0; texture.swizzle[3] = 1;
    live_table->resources.push_back(texture);
    DrawItem live_draw;
    live_draw.vs = producer.vs; live_draw.vertex_count = 3;
    live_draw.ps = sample_state; live_draw.prt = live_table;
    live_draw.color0_base = Base + 0x100000;
    live_draw.color0_width = W; live_draw.color0_height = H;
    auto live_shader = [&](uint32_t layer) {
        std::vector<uint32_t> ps{0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
            0x7e0402ffu, std::bit_cast<uint32_t>(float(layer)), 0x7e060280u,
            0xf0900f28u, 0x00820000u};
        const float channels[]{expected[layer], 0.0f, 0.0f, 1.0f};
        for (uint32_t c = 0; c < 4; ++c) {
            ps.insert(ps.end(), {0x080000ffu | (c << 17) | (c << 9),
                std::bit_cast<uint32_t>(channels[c]),
                0x100000ffu | (c << 17) | (c << 9), std::bit_cast<uint32_t>(65536.0f),
                0x060000ffu | (c << 17) | (c << 9), 0x3f000000u});
        }
        ps.insert(ps.end(), {0xf800000fu, 0x03020100u, 0xbf810000u});
        live_draw.fs = recompile_fragment(ps.data(), ps.size(), live_table.get());
        check(!live_draw.fs.empty(), "frontend array precision comparison shader compiles");
    };
    auto live_matches = [&] {
        const auto pixels = render_submit_items({live_draw}, W, H);
        if (pixels.size() != Pixels * 4) return false;
        const size_t center = (W * (H / 2) + W / 2) * 4;
        for (size_t c = 0; c < 4; ++c)
            if (pixels[center + c] < 127 || pixels[center + c] > 128) return false;
        return true;
    };
    for (uint32_t layer = 0; layer < Layers; ++layer) {
        live_shader(layer);
        check(live_matches(), "production frontend selects retained depth over zero hosted bytes");
    }
    expected[3] = 0.9376373291015625f;
    if (!seed_layer(3, expected[3])) return 1;
    live_shader(3);
    check(live_matches(), "frontend observes renderer-only rewrite with unchanged guest identity");
    depth_array_readback_failure_after_layers() = 2;
    const auto failure_diagnostic = capture_stderr([&] {
        check(live_matches(), "rejected draw preserves the prior completed color target");
    });
    check(failure_diagnostic.find("[render-array-reject] binding=4 retained depth: "
          "injected retained depth array readback failure") != std::string::npos,
          "frontend explicitly refuses a partially read retained array");
    check(live_matches(), "frontend retries the same retained identity after transient failure");

    const auto supported = live_table->resources[0];
    for (unsigned shape = 0; shape < 5; ++shape) {
        auto& unsupported = live_table->resources[0];
        unsupported = supported;
        if (shape == 0) unsupported.num_components = 2;
        if (shape == 1) unsupported.num_components = 4;
        if (shape == 2) unsupported.layer_mip_offset_bytes = 32;
        if (shape == 3) unsupported.in_mip_tail = true;
        if (shape == 4) {
            unsupported.gpu_addr += unsupported.layer_stride_bytes;
            unsupported.host_data += unsupported.layer_stride_bytes;
            unsupported.host_data_size -= unsupported.layer_stride_bytes;
            unsupported.size -= unsupported.layer_stride_bytes;
            unsupported.depth = Layers - 1;
        }
        const auto diagnostic = capture_stderr([&] {
            (void)render_submit_items({live_draw}, W, H);
        });
        constexpr const char* labels[]{
            "two-component retained depth view explicitly refuses guest fallback",
            "four-component retained depth view explicitly refuses guest fallback",
            "retained depth selected mip explicitly refuses guest fallback",
            "retained depth mip tail explicitly refuses guest fallback",
            "rebased retained depth subview explicitly refuses guest fallback"};
        check(diagnostic.find("[render-array-reject] binding=4 unsupported retained depth view") !=
                  std::string::npos, labels[shape]);
    }
    live_table->resources[0] = supported;
    check(live_matches(), "supported retained descriptor still works after rejected aliases");

    // The production callback groups a depth-only writer separately from its colored consumer.
    // With no color readback, the first group stays pending until build_R forwards that same
    // BackendSubmissionBatch to the retained-array helper. Reading before that flush samples the
    // previous generation, even though the producer eventually executes before callback return.
    DrawItem live_producer;
    live_producer.vs = producer.vs; live_producer.fs = producer.fs;
    live_producer.vertex_count = 3; live_producer.ps = producer_state;
    live_producer.color0_width = W; live_producer.color0_height = H;
    for (unsigned generation = 0; generation < 2; ++generation) {
        expected[1] = generation == 0 ? 0.3751373291015625f : 0.6876678466796875f;
        live_producer.ps.min_depth = live_producer.ps.max_depth = expected[1];
        live_shader(1);
        // A fresh target prevents a rejected consumer from passing by preserving an earlier
        // successful comparison image. The depth producer has no color target and no CPU pixels.
        live_draw.color0_base = Base + 0x300000 + generation * 0x10000;
        const auto pixels = render_submit_items({live_producer, live_draw}, W, H);
        const size_t center = (W * (H / 2) + W / 2) * 4;
        bool matches = pixels.size() == Pixels * 4;
        if (matches) for (size_t c = 0; c < 4; ++c)
            matches &= pixels[center + c] >= 127 && pixels[center + c] <= 128;
        check(matches, "frontend batch forwards depth producer ordering through array materialization");
    }
    return failures ? 1 : 0;
}
