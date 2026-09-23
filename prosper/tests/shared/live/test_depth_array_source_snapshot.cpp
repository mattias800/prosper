// Retained depth arrays preserve Float32 contents and publish only complete, ordered snapshots.
#include "fixtures/render_runner.h"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include <bit>
#include <cstdio>
#include <cstring>
#include <sstream>
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

int main(int argc, char** argv) {
    bool per_draw_control = false, expanded_control = false;
    bool texture_path_census = false, texture_path_census_budget = false;
    bool texture_path_census_empty = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--per-draw-control") == 0) per_draw_control = true;
        else if (std::strcmp(argv[i], "--expanded-control") == 0) expanded_control = true;
        else if (std::strcmp(argv[i], "--texture-path-census") == 0) texture_path_census = true;
        else if (std::strcmp(argv[i], "--texture-path-census-budget") == 0)
            texture_path_census_budget = true;
        else if (std::strcmp(argv[i], "--texture-path-census-empty") == 0)
            texture_path_census_empty = true;
        else {
            std::fprintf(stderr, "usage: %s [--per-draw-control] [--expanded-control] "
                                 "[--texture-path-census|--texture-path-census-budget|"
                                 "--texture-path-census-empty]\n", argv[0]);
            return 2;
        }
    }
    if (texture_path_census_budget) {
        check(std::getenv("PROSPER_BACKEND_TEXTURE_PATH_CENSUS") != nullptr,
              "texture path budget arm is explicitly enabled before first backend use");
        const char* min_draws_text = std::getenv("PROSPER_BACKEND_TEXTURE_PATH_CENSUS_MIN_DRAWS");
        check(min_draws_text && std::strcmp(min_draws_text, "10") == 0,
              "budget arm has a non-default minimum draw count");
        const auto log = capture_stderr([&] {
            prosper::frontend::ScopedInteractivePerformanceTiming timing(true);
            for (unsigned below = 0; below < 3; ++below)
                BackendTexturePathCensus subthreshold(1);
            for (unsigned call = 0; call < BackendTexturePathCensus::kMaxCalls + 2; ++call) {
                BackendTexturePathCensus census(10);
                for (size_t row = 0; row < BackendTexturePathCensus::kMaxRows + 1; ++row)
                    census.record({});
                census.reached_resource_phase_end();
            }
        });
        size_t calls = 0;
        std::istringstream lines(log);
        for (std::string line; std::getline(lines, line);)
            calls += line.starts_with("[texture-path] call=");
        check(calls == BackendTexturePathCensus::kMaxCalls,
              "four-call budget refuses all later backend calls");
        check(log.find("threshold-decline draws=1 min_draws=10") != std::string::npos &&
                  log.find("min_draws=10 considered=4 below_threshold=3") != std::string::npos,
              "subthreshold calls are counted but do not consume the first admitted slot");
        check(log.find("budget-exhausted cap_calls=4 min_draws=10") != std::string::npos,
              "the first over-budget eligible call reports its refusal");
        check(log.find("rows=512 omitted=1 complete_population=0 resource_phase_reached=1") !=
                  std::string::npos,
              "row cap reports omitted unique-key decisions instead of silent truncation");
        return failures ? 1 : 0;
    }
    if (texture_path_census_empty) {
        const char* minimum = std::getenv("PROSPER_BACKEND_TEXTURE_PATH_CENSUS_MIN_DRAWS");
        check(minimum && !*minimum, "empty-threshold arm passes an explicitly empty setting");
        const auto log = capture_stderr([&] {
            prosper::frontend::ScopedInteractivePerformanceTiming timing(true);
            BackendTexturePathCensus census(10);
        });
        check(log.find("REFUSED PROSPER_BACKEND_TEXTURE_PATH_CENSUS_MIN_DRAWS=''") !=
                  std::string::npos &&
                  log.find("[texture-path] call=") == std::string::npos,
              "explicitly empty threshold refuses the census instead of taking default admission");
        return failures ? 1 : 0;
    }
    // These production controls are cached at first use. Set them before ANY renderer callback;
    // the separate process arm supplies the per-draw oracle without changing startup semantics.
#ifdef _WIN32
    _putenv_s("PROSPER_DEPTH_ARRAY_SNAPSHOT_CENSUS", "1");
    _putenv_s("PROSPER_ARRAY_REJECT_LOG_ALL", "1");
    if (per_draw_control) _putenv_s("PROSPER_NO_SUBMIT_DEPTH_ARRAY_SNAPSHOT_REUSE", "1");
    if (expanded_control) _putenv_s("PROSPER_NO_COMPACT_DEPTH_ARRAY_SNAPSHOT", "1");
#else
    setenv("PROSPER_DEPTH_ARRAY_SNAPSHOT_CENSUS", "1", 1);
    setenv("PROSPER_ARRAY_REJECT_LOG_ALL", "1", 1);
    if (per_draw_control) setenv("PROSPER_NO_SUBMIT_DEPTH_ARRAY_SNAPSHOT_REUSE", "1", 1);
    if (expanded_control) setenv("PROSPER_NO_COMPACT_DEPTH_ARRAY_SNAPSHOT", "1", 1);
#endif
    constexpr uint32_t W = 8, H = 8, Layers = 4;
    // Numeric disjointness is not physical disjointness. Give the fixture's ordinary DS
    // identities tracked private backing so the invalidator can prove their independence.
    prosper::register_builtin_hle();
    auto map_flexible = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto map_direct = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
    auto alloc_direct = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    auto release_direct = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
    struct GuestBacking {
        prosper::HleFn unmap, release;
        std::vector<std::pair<uint64_t, uint64_t>> mappings, physical;
        ~GuestBacking() {
            for (auto [address, bytes] : mappings) unmap(address, bytes, 0, 0, 0, 0);
            for (auto [address, bytes] : physical) release(address, bytes, 0, 0, 0, 0);
        }
    } backing{unmap, release_direct, {}, {}};
    check(map_flexible && map_direct && alloc_direct && unmap && release_direct,
          "guest mapping and physical-alias APIs available");
    if (!map_flexible || !map_direct || !alloc_direct || !unmap || !release_direct) return 1;
    uint64_t Base = 0;
    constexpr uint64_t FixtureBytes = 0x2000000;
    const bool base_mapped = map_flexible(reinterpret_cast<uint64_t>(&Base), FixtureBytes, 2, 0,
        reinterpret_cast<uint64_t>("depth-array-identities"), 0) == 0 && Base;
    check(base_mapped, "ordinary depth fixture has real tracked guest backing");
    if (!base_mapped) return 1;
    backing.mappings.emplace_back(Base, FixtureBytes);
    constexpr size_t Pixels = W * H;
    using Status = PersistentDsDepthArrayStatus;
    std::string error;
    std::vector<float> output{19.0f, 23.0f};
    const auto sentinel = output;
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::NoIdentity && output == sentinel,
          "absent retained identity permits guest fallback without modifying output");

    uint64_t seed_base = Base;
    auto seed_layer = [&](uint32_t layer, float value) {
        BackendPersistentResourceGuard guard;
        GpuCaptureDsSeed seed;
        seed.depth_read_base = seed.depth_write_base = seed_base;
        seed.width = W; seed.height = H; seed.slice = layer;
        seed.format = GpuCaptureDsFormat::D32Float; seed.depth_valid = true;
        seed.depth.resize(Pixels * sizeof(float));
        for (size_t i = 0; i < Pixels; ++i)
            std::memcpy(seed.depth.data() + i * sizeof(float), &value, sizeof(value));
        const bool ok = restore_persistent_ds_image(seed, error);
        if (!ok) std::fprintf(stderr, "seed: %s\n", error.c_str());
        check(ok, "real Vulkan depth layer restored");
        if (ok) for (auto& [key, image] : persistent_ds_cache())
            if (key.dr == seed_base && key.slice == layer)
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
    prosper::frontend::register_live_renderer(".", false);
    VkFormatProperties compact_properties{};
    vkGetPhysicalDeviceFormatProperties(render_vk_ctx().phys, VK_FORMAT_R32_SFLOAT, &compact_properties);
    const auto payload_bpp = [&](bool linear) -> uint64_t {
        const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            (linear ? VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT : 0u);
        return !expanded_control && (compact_properties.optimalTilingFeatures & needed) == needed
            ? sizeof(float) : 4 * sizeof(float);
    };
    const uint64_t snapshot_bpp = payload_bpp(false);
    std::printf("[snapshot-fixture] representation=%s expected_bpp=%llu R32features=0x%x\n",
                expanded_control ? "expanded-control" : "compact-with-feature-fallback",
                (unsigned long long)snapshot_bpp, compact_properties.optimalTilingFeatures);
    std::vector<uint8_t> stale_guest(Pixels * Layers * sizeof(float), 0);
    auto live_table = std::make_shared<ShaderResourceTable>();
    texture.num_components = 1;
    texture.gpu_addr = Base; texture.size = stale_guest.size();
    texture.host_data = stale_guest.data(); texture.host_data_size = stale_guest.size();
    texture.layer_stride_bytes = Pixels * sizeof(float);
    texture.linear_row_pitch_bytes = W * sizeof(float);
    texture.mag_filter = texture.min_filter = 0;
    // Identity view selectors make G/B/A come from real format substitution, not swizzle constants.
    texture.swizzle[0] = 4; texture.swizzle[1] = 5;
    texture.swizzle[2] = 6; texture.swizzle[3] = 7;
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
    // Exercise missing channels and view swizzling through the actual retained-array frontend.
    // Gather taps are uniform within each seeded layer here; the separate gather fixture owns
    // spatial tap-order coverage. This guard specifically binds format-dependent component values.
    const auto component_shader = [&](bool gather, uint32_t component, float u,
                                      const std::array<float, 4>& wanted, bool fetch = false) {
        std::vector<uint32_t> words;
        auto mov = [&](uint32_t reg, uint32_t value) {
            words.insert(words.end(), {0x7e0002ffu | (reg << 17), value});
        };
        if (gather) {
            mov(0, 0x3f01u); // signed offset (+1,-1), separate from the layer
            mov(1, std::bit_cast<uint32_t>(u)); mov(2, 0x3f000000u); mov(3, 0x40000000u);
            words.insert(words.end(), {0xf0000028u | (0x57u << 18) | (1u << (8 + component)),
                                      0x00820c00u});
        } else if (fetch) {
            mov(0, UINT32_MAX); mov(1, 0u); mov(2, 2u);
            words.insert(words.end(), {0xf0000f28u, 0x00020c00u});
        } else {
            mov(0, std::bit_cast<uint32_t>(u)); mov(1, 0x3f000000u);
            mov(2, 0x40000000u); mov(3, 0u);
            words.insert(words.end(), {0xf0900f28u, 0x00820c00u});
        }
        for (uint32_t c = 0; c < 4; ++c) {
            const uint32_t reg = 12 + c;
            words.insert(words.end(), {
                0x080000ffu | (reg << 17) | (reg << 9), std::bit_cast<uint32_t>(wanted[c]),
                0x100000ffu | (reg << 17) | (reg << 9), std::bit_cast<uint32_t>(65536.0f),
                0x060000ffu | (reg << 17) | (reg << 9), 0x3f000000u});
        }
        words.insert(words.end(), {0xf800000fu, 0x0f0e0d0cu, 0xbf810000u});
        live_draw.fs = recompile_fragment(words.data(), words.size(), live_table.get());
        check(!live_draw.fs.empty(), "retained array component shader compiles");
        live_draw.color0_base += 0x10000;
    };
    const std::array<float, 4> natural{expected[2], 0.0f, 0.0f, 1.0f};
    for (uint32_t channel = 0; channel < 4; ++channel) {
        component_shader(true, channel, 0.5f,
            {natural[channel], natural[channel], natural[channel], natural[channel]});
        check(live_matches(), "array gather preserves precise R and substituted G/B/A components");
        check(backend_texture_upload_stats().upload_bytes == Pixels * Layers * snapshot_bpp,
              "gather uses the expected compact or explicitly expanded payload size");
    }
    auto& component_resource = live_table->resources[0];
    component_resource.swizzle[0] = 7; component_resource.swizzle[1] = 4;
    component_resource.swizzle[2] = 5; component_resource.swizzle[3] = 6;
    component_shader(false, 0, 0.5f, {1.0f, expected[2], 0.0f, 0.0f});
    check(live_matches(), "array sampling applies view swizzle after missing-component substitution");
    component_shader(true, 1, 0.5f, {expected[2], expected[2], expected[2], expected[2]});
    check(live_matches(), "array gather selects the swizzled depth component");
    component_resource.swizzle[0] = 4; component_resource.swizzle[1] = 5;
    component_resource.swizzle[2] = 6; component_resource.swizzle[3] = 7;
    VkFormatProperties expanded_properties{};
    vkGetPhysicalDeviceFormatProperties(render_vk_ctx().phys, VK_FORMAT_R32G32B32A32_SFLOAT,
                                        &expanded_properties);
    if (expanded_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) {
        component_resource.mag_filter = component_resource.min_filter = 1;
        component_shader(false, 0, 0.5f, natural);
        check(live_matches(), "linear array sampler preserves exact uniform-layer components");
        check(backend_texture_upload_stats().upload_bytes == Pixels * Layers * payload_bpp(true),
              "R32 linear support is checked separately and otherwise retains expanded sampling");
        component_resource.mag_filter = component_resource.min_filter = 0;
    } else {
        std::puts("[skip] existing RGBA32 array route cannot support the linear-sampler control");
    }

    // Border replacement precedes missing-component substitution and is format-sensitive.
    // Preserve RGBA32 for border samplers: transparent A=0 and opaque-white G/B=1 differ in R32.
    component_resource.addr_uvw[0] = 6;
    for (uint32_t border : {0u, 2u}) {
        component_resource.border_color_type = border;
        const float border_value = border == 2 ? 1.0f : 0.0f;
        component_shader(false, 0, -4.0f,
            {border_value, border_value, border_value, border_value});
        check(live_matches(), "border sampler preserves expanded RGBA channel values outside the image");
        component_shader(true, border == 2 ? 1u : 3u, -4.0f,
            {border_value, border_value, border_value, border_value});
        check(live_matches(), "out-of-range gather preserves transparent alpha and opaque-white green");
        check(backend_texture_upload_stats().upload_bytes == Pixels * Layers * 4 * sizeof(float),
              "border-sensitive retained array deliberately keeps the expanded representation");
    }
    component_resource.addr_uvw[0] = 0; component_resource.border_color_type = 0;
    component_shader(false, 0, 0.0f, {0.0f, 0.0f, 0.0f, 0.0f}, true);
    check(live_matches(), "out-of-bounds array fetch preserves expanded robust zero including alpha");
    check(backend_texture_upload_stats().upload_bytes == Pixels * Layers * 4 * sizeof(float),
          "any image fetch keeps the expanded retained-array representation");

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
    // Keep every color and DS attachment identical: only the write enable differs. The
    // consumer must be prepared after the producer, rather than reading the preceding generation
    // while build_bds is still preparing their shared backend group.
    for (unsigned generation = 0; generation < 2; ++generation) {
        expected[1] = generation == 0 ? 0.1876373291015625f : 0.8126373291015625f;
        live_producer.ps.min_depth = live_producer.ps.max_depth = expected[1];
        live_producer.ps.color_write_mask = sample_state.color_write_mask;
        live_draw.color0_base = live_producer.color0_base =
            Base + 0x500000 + generation * 0x10000;
        live_draw.ps.depth_read_base = live_producer.ps.depth_read_base;
        live_draw.ps.depth_write_base = live_producer.ps.depth_write_base;
        live_draw.ps.db_depth_view = live_producer.ps.db_depth_view;
        live_draw.ps.depth_test_enable = true;
        live_draw.ps.depth_write_enable = false;
        live_draw.ps.depth_compare_op = VK_COMPARE_OP_ALWAYS;
        live_shader(1);
        const auto pixels = render_submit_items({live_producer, live_draw}, W, H);
        const size_t center = (W * (H / 2) + W / 2) * 4;
        bool matches = pixels.size() == Pixels * 4;
        if (matches) for (size_t c = 0; c < 4; ++c)
            matches &= pixels[center + c] >= 127 && pixels[center + c] <= 128;
        check(matches, "same-attachment depth producer completes before array consumer preparation");
    }

    // Only the production resource builder can learn this descriptor's layer stride. A write
    // into layer 2 must revoke that layer without falsely invalidating the other three planes.
    check(ds_layer_stride_for(Base, W, H) == Pixels * sizeof(float),
          "exact array consumer publishes its depth layer stride");
    {
        BackendPersistentResourceGuard guard;
        check(invalidate_persistent_ds_guest_write(Base + 2 * Pixels * sizeof(float),
                                                  sizeof(float)) == 1,
              "nonzero array layer guest write invalidates precisely that retained layer");
    }
    const auto before_guest_write = output;
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_guest_write,
          "array snapshot refuses a guest-invalidated nonzero layer");
    if (!seed_layer(2, expected[2])) return 1;
    check(read_persistent_ds_depth_array(Base, W, H, 0, Layers, output, error) ==
              Status::Ready && exact(),
          "renderer rewrite restores complete array after nonzero layer guest invalidation");
    // A guest write may drain through a target query before the array is ever sampled. There
    // is then no known stride for the ordinary invalidator to locate layer 2. First admission
    // must not bless those old images merely by learning the stride afterward.
    seed_base = Base + 0x800000;
    live_table->resources[0].gpu_addr = seed_base;
    live_draw.ps = sample_state;
    live_draw.color0_base = seed_base + 0x100000;
    for (uint32_t layer = 0; layer < Layers; ++layer)
        if (!seed_layer(layer, expected[layer])) return 1;
    check(ds_layer_stride_for(seed_base, W, H) == 0,
          "fresh retained array has no previously learned consumer stride");
    notify_guest_gpu_write_preserving_bytes(seed_base + 2 * Pixels * sizeof(float), sizeof(float));
    (void)is_live_render_target(seed_base + 0x200000); // Drain before the first consumer.
    // Exercise the ordinary publisher used by depth cubes before the array's own builder.
    // Otherwise a cube could turn "unknown" into "known" and bypass an array-only history check.
    note_ds_layer_stride(seed_base, W, H, Pixels * sizeof(float));
    check(ds_layer_stride_for(seed_base, W, H) == Pixels * sizeof(float),
          "ordinary cube stride publisher records the first consumer stride");
    const auto before_cube_publication = output;
    check(read_persistent_ds_depth_array(seed_base, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_cube_publication,
          "first cube stride publication cannot authorize depth older than an unknown-stride write");
    live_shader(2);
    const auto historical_diagnostic = capture_stderr([&] {
        (void)render_submit_items({live_draw}, W, H);
    });
    check(historical_diagnostic.find("[render-array-reject] binding=4 retained depth: "
          "retained depth array has missing or invalid layers") != std::string::npos,
          "write drained before first stride refuses historically unproven array layers");
    for (uint32_t layer = 0; layer < Layers; ++layer)
        if (!seed_layer(layer, expected[layer])) return 1;
    check(live_matches(), "renderer rewrites after historical write recover array admission");

    // The watermark intentionally does not remember addresses. An unrelated intervening write
    // also refuses an older unknown-stride array; this is conservative, never evidence of overlap.
    seed_base = Base + 0xa00000;
    live_table->resources[0].gpu_addr = seed_base;
    live_draw.color0_base = seed_base + 0x100000;
    for (uint32_t layer = 0; layer < Layers; ++layer)
        if (!seed_layer(layer, expected[layer])) return 1;
    notify_guest_gpu_write(Base + 0xf00000, sizeof(float));
    (void)is_live_render_target(Base + 0xf00000);
    const auto unrelated_diagnostic = capture_stderr([&] {
        (void)render_submit_items({live_draw}, W, H);
    });
    check(unrelated_diagnostic.find("[render-array-reject] binding=4 retained depth: "
          "retained depth array has missing or invalid layers") != std::string::npos,
          "unrelated intervening write conservatively refuses unknown-stride history");

    // A fresh renderer generation AFTER the last guest write does not need retrospective range
    // knowledge, even when the process has an older watermark from a different allocation.
    seed_base = Base + 0xc00000;
    live_table->resources[0].gpu_addr = seed_base;
    live_draw.color0_base = seed_base + 0x100000;
    for (uint32_t layer = 0; layer < Layers; ++layer)
        if (!seed_layer(layer, expected[layer])) return 1;
    check(ds_layer_stride_for(seed_base, W, H) == 0 && live_matches(),
          "untouched freshly rendered array admits its first exact consumer after older writes");
    // A recycled write address can coexist with an unrelated old attachment. Neither a
    // different extent nor a slice outside this consumer's range is one of its source planes.
    // A rejected draw can return an earlier color image, so positive arms must also assert the
    // specific decline is absent. The matching-alias negative arm below proves that lever logs.
    const uint64_t saved_color_base = live_draw.color0_base;
    const auto canonical_survives_alias = [&](const char* pixel_label, const char* log_label) {
        bool pixels_match = false;
        const auto diagnostic = capture_stderr([&] { pixels_match = live_matches(); });
        check(pixels_match, pixel_label);
        check(diagnostic.find("[render-array-reject] binding=4 unproven retained depth "
              "write-base alias") == std::string::npos, log_label);
    };
    PersistentDsKey coexisting_alias{Base + 0xe00000, seed_base, 0, 0, 0,
                                    W / 2, H / 2, VK_FORMAT_D32_SFLOAT, 0};
    {
        BackendPersistentResourceGuard guard;
        persistent_ds_cache()[coexisting_alias] = {};
    }
    live_draw.color0_base = Base + 0x1400000;
    canonical_survives_alias("different-extent alias preserves canonical pixels",
                             "different-extent alias does not trigger noncanonical veto");
    {
        BackendPersistentResourceGuard guard;
        persistent_ds_cache().erase(coexisting_alias);
        coexisting_alias.w = W; coexisting_alias.h = H; coexisting_alias.slice = Layers;
        persistent_ds_cache()[coexisting_alias] = {};
    }
    live_draw.color0_base = Base + 0x1410000;
    canonical_survives_alias("out-of-range alias preserves canonical pixels",
                             "out-of-range alias does not trigger noncanonical veto");
    {
        BackendPersistentResourceGuard guard;
        persistent_ds_cache().erase(coexisting_alias);
        coexisting_alias.slice = 0;
        uint64_t canonical_generation = 0;
        for (const auto& [key, image] : persistent_ds_cache())
            if (key.dr == seed_base && key.w == W && key.h == H && key.slice == 0)
                canonical_generation = std::max(canonical_generation, image.last_depth_write);
        check(canonical_generation > 1, "canonical layer has a version newer than stale alias");
        auto& old_alias = persistent_ds_cache()[coexisting_alias];
        old_alias.last_depth_write = canonical_generation - 1;
    }
    live_draw.color0_base = Base + 0x1420000;
    canonical_survives_alias("older same-shape alias preserves newer canonical pixels",
                             "older same-shape alias does not trigger noncanonical veto");
    {
        BackendPersistentResourceGuard guard;
        note_persistent_ds_depth_write(persistent_ds_cache()[coexisting_alias], true, true);
    }
    live_draw.color0_base = Base + 0x1430000;
    const auto coexisting_diagnostic = capture_stderr([&] {
        (void)render_submit_items({live_draw}, W, H);
    });
    check(coexisting_diagnostic.find("[render-array-reject] binding=4 unproven retained depth "
          "write-base alias") != std::string::npos,
          "newer matching write-base alias still refuses selected noncanonical source");
    {
        BackendPersistentResourceGuard guard;
        persistent_ds_cache().erase(coexisting_alias);
    }
    live_draw.color0_base = Base + 0x1440000;
    check(live_matches(), "canonical array recovers after matching alias is removed");
    live_draw.color0_base = saved_color_base;
    // Re-key the real images without duplicating their Vulkan ownership. Sampling a distinct
    // write-base alias must not publish a stride that the read-base invalidator never consults.
    const uint64_t write_alias = Base + 0x1000000;
    {
        BackendPersistentResourceGuard guard;
        auto& cache = persistent_ds_cache();
        std::vector<PersistentDsKey> keys;
        for (const auto& [key, image] : cache)
            if (key.dr == seed_base) keys.push_back(key);
        for (const auto& key : keys) {
            auto entry = cache.extract(key);
            entry.key().dw = write_alias;
            cache.insert(std::move(entry));
        }
    }
    live_table->resources[0].gpu_addr = write_alias;
    const auto alias_diagnostic = capture_stderr([&] {
        (void)render_submit_items({live_draw}, W, H);
    });
    check(alias_diagnostic.find("[render-array-reject] binding=4 unproven retained depth "
          "write-base alias") != std::string::npos,
          "noncanonical write-base array alias is explicitly refused");
    check(ds_layer_stride_for(write_alias, W, H) == 0,
          "rejected write-base alias never publishes an unrelated stride identity");
    live_table->resources[0].gpu_addr = seed_base;
    check(live_matches(), "canonical read-base array remains usable after write-alias rejection");
    {
        BackendPersistentResourceGuard guard;
        auto& cache = persistent_ds_cache();
        std::vector<PersistentDsKey> keys;
        for (const auto& [key, image] : cache)
            if (key.dr == seed_base && key.dw == write_alias) keys.push_back(key);
        for (const auto& key : keys) {
            auto entry = cache.extract(key);
            entry.key().dw = seed_base;
            cache.insert(std::move(entry));
        }
    }

    // Multiple bindings of one retained array must share one immutable payload/upload inside
    // this draw. Their samplers and selected layers differ, so descriptor-state dedup alone
    // cannot remove the repeated 64-MiB image/staging allocations seen with real shadow arrays.
    const uint64_t shared_base = seed_base;
    auto repeated_table = std::make_shared<ShaderResourceTable>();
    for (uint32_t i = 0; i < 3; ++i) {
        auto resource = live_table->resources[0];
        resource.binding = 4 + i;
        resource.sgpr_base = 8 + i * 8;
        resource.addr_uvw[0] = i; // Distinct samplers; the interior coordinates still agree.
        repeated_table->resources.push_back(resource);
    }
    live_draw.prt = repeated_table;
    auto repeated_shader = [&](std::array<float, 3> expected_values) {
        std::vector<uint32_t> words;
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t reg = 20 + i;
            const uint32_t layer = i == 2 && repeated_table->resources[i].depth == 2 ? 1u : i;
            words.insert(words.end(), {0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
                0x7e0402ffu, std::bit_cast<uint32_t>(float(layer)), 0x7e060280u,
                0xf0900128u, 0x00800000u | ((repeated_table->resources[i].sgpr_base / 4) << 16) |
                                (reg << 8),
                0x080000ffu | (reg << 17) | (reg << 9), std::bit_cast<uint32_t>(expected_values[i]),
                0x100000ffu | (reg << 17) | (reg << 9), std::bit_cast<uint32_t>(65536.0f),
                0x060000ffu | (reg << 17) | (reg << 9), 0x3f000000u});
        }
        words.insert(words.end(), {0xf800000fu, 0x14161514u, 0xbf810000u});
        live_draw.fs = recompile_fragment(words.data(), words.size(), repeated_table.get());
        check(!live_draw.fs.empty(), "three independent array consumers compile");
    };
    expected[1] = 0.3282623291015625f;
    live_producer.ps.depth_read_base = live_producer.ps.depth_write_base = shared_base;
    live_producer.ps.min_depth = live_producer.ps.max_depth = expected[1];
    live_producer.ps.color_write_mask = 0;
    live_producer.color0_base = 0;
    repeated_shader({expected[0], expected[1], expected[2]});
    live_draw.color0_base = Base + 0x1200000;
    std::vector<uint8_t> shared_pixels;
    std::string texture_path_log;
    if (texture_path_census) {
        texture_path_log = capture_stderr([&] {
            prosper::frontend::ScopedInteractivePerformanceTiming timing(true);
            shared_pixels = render_submit_items({live_producer, live_draw}, W, H);
        });
    } else {
        shared_pixels = render_submit_items({live_producer, live_draw}, W, H);
    }
    const size_t shared_center = (W * (H / 2) + W / 2) * 4;
    bool shared_matches = shared_pixels.size() == Pixels * 4;
    if (shared_matches) for (size_t c = 0; c < 4; ++c)
        shared_matches &= shared_pixels[shared_center + c] >= 127 &&
                          shared_pixels[shared_center + c] <= 128;
    check(shared_matches, "shared array payload preserves layers/samplers after pending producer flush");
    auto uploads = backend_texture_upload_stats();
    check(uploads.references == 3 && uploads.unique_uploads == 1 &&
              uploads.upload_bytes == Pixels * Layers * snapshot_bpp,
          "three array bindings produce one actual backend image/staging upload");
    if (texture_path_census) {
        bool copied_row = false, complete_call = false;
        std::istringstream lines(texture_path_log);
        for (std::string line; std::getline(lines, line);) {
            complete_call |= line.starts_with("[texture-path] call=") &&
                             line.find("complete_population=1 resource_phase_reached=1 "
                                       "skipped_resource_draws=0") !=
                                 std::string::npos;
            if (!line.starts_with("[texture-path-row] call=") ||
                line.find("path=cpu_staging_copy") == std::string::npos ||
                line.find("cpu_copy=" + std::to_string(Pixels * Layers * snapshot_bpp)) ==
                    std::string::npos) continue;
            const auto value_after = [&line](const char* key) {
                const size_t pos = line.find(key);
                return pos == std::string::npos ? -1.0
                    : std::strtod(line.c_str() + pos + std::strlen(key), nullptr);
            };
            const double prepare = value_after("prepare_ms=");
            const double copy = value_after("cpu_copy_ms=");
            copied_row |= prepare > 0 && copy > 0 && prepare >= copy;
        }
        check(complete_call, "live backend census reaches resource-complete call site");
        check(copied_row, "live unique-upload branch reports actual CPU staging-copy bytes");
    }

    // Separate draws in one real callback share a stable retained generation. Distinct samplers
    // still yield six references, while the immutable pixel owner permits one actual upload. The
    // old draw-local memo makes two unique uploads here even when the final pixels happen to match.
    live_draw.color0_base += 0x10000;
    const auto cross_draw_pixels = render_submit_items({live_draw, live_draw}, W, H);
    bool cross_draw_matches = cross_draw_pixels.size() == Pixels * 4;
    if (cross_draw_matches) for (size_t c = 0; c < 4; ++c)
        cross_draw_matches &= cross_draw_pixels[shared_center + c] >= 127 &&
                              cross_draw_pixels[shared_center + c] <= 128;
    check(cross_draw_matches, "two same-generation array consumers preserve exact sampled pixels");
    uploads = backend_texture_upload_stats();
    const uint64_t expected_uploads = per_draw_control ? 2u : 1u;
    const auto report_uploads = [&](const char* arm) {
        std::printf("[snapshot-fixture] arm=%s policy=%s refs=%llu uploads=%llu bytes=%llu\n",
            arm, per_draw_control ? "per-draw" : "callback",
            (unsigned long long)uploads.references, (unsigned long long)uploads.unique_uploads,
            (unsigned long long)uploads.upload_bytes);
    };
    report_uploads("same-target");
    check(uploads.references == 6, "two draws retain all six texture references");
    check(uploads.unique_uploads == expected_uploads,
          "callback policy shares one upload; startup per-draw control requires two");
    check(uploads.upload_bytes == expected_uploads * Pixels * Layers * snapshot_bpp,
          "actual uploaded bytes agree with the independently expected policy count");

    // Different color targets force separate backend groups. Completing unrelated color work
    // must retain an unchanged depth snapshot rather than turning every group boundary into a miss.
    const DrawItem first_color_target = live_draw;
    live_draw.color0_base += 0x10000;
    std::vector<uint8_t> cross_group_pixels;
    const auto cross_group_census = capture_stderr([&] {
        cross_group_pixels = render_submit_items({first_color_target, live_draw}, W, H);
    });
    bool cross_group_matches = cross_group_pixels.size() == Pixels * 4;
    if (cross_group_matches) for (size_t c = 0; c < 4; ++c)
        cross_group_matches &= cross_group_pixels[shared_center + c] >= 127 &&
                              cross_group_pixels[shared_center + c] <= 128;
    check(cross_group_matches, "different color groups preserve exact sampled depth pixels");
    const char* expected_group_census = per_draw_control
        ? "scope=callback reads=2 reuses=4 " : "scope=callback reads=1 reuses=5 ";
    check(cross_group_census.find(expected_group_census) != std::string::npos,
          "different color groups use one callback read or two explicit control reads");

    // Warm that same callback memo, then record an actual depth rewrite between consumers. The
    // last consumer has a different expected depth; stale reuse cannot pass by keeping old pixels.
    const DrawItem before_rewrite = live_draw;
    expected[1] = 0.4532623291015625f;
    live_producer.ps.min_depth = live_producer.ps.max_depth = expected[1];
    repeated_shader({expected[0], expected[1], expected[2]});
    live_draw.color0_base += 0x10000;
    std::vector<uint8_t> rewritten_in_callback;
    const auto rewritten_census = capture_stderr([&] {
        rewritten_in_callback = render_submit_items({before_rewrite, live_producer, live_draw}, W, H);
    });
    bool rewritten_matches = rewritten_in_callback.size() == Pixels * 4;
    if (rewritten_matches) for (size_t c = 0; c < 4; ++c)
        rewritten_matches &= rewritten_in_callback[shared_center + c] >= 127 &&
                             rewritten_in_callback[shared_center + c] <= 128;
    check(rewritten_matches,
          "depth producer between consumers replaces the earlier sampled pixels");
    check(rewritten_census.find("scope=callback reads=2 reuses=4 ") != std::string::npos,
          "both policies require two reads when a depth producer changes the generation");

    // A different base in the SAME preparation scope must retain independent pixels.
    seed_base = Base;
    constexpr float different_value = 0.4376373291015625f;
    if (!seed_layer(2, different_value)) return 1;
    repeated_table->resources[2].gpu_addr = Base;
    repeated_shader({expected[0], expected[1], different_value});
    live_draw.color0_base += 0x10000;
    check(live_matches(), "different retained array bases keep independent sampled values");
    uploads = backend_texture_upload_stats();
    check(uploads.references == 3 && uploads.unique_uploads == 2,
          "different array base does not reuse the first snapshot/upload");

    repeated_table->resources[2].gpu_addr = shared_base;
    repeated_table->resources[2].depth = 2;
    repeated_shader({expected[0], expected[1], expected[1]});
    live_draw.color0_base += 0x10000;
    check(live_matches(), "different array layer count retains the requested view");
    uploads = backend_texture_upload_stats();
    check(uploads.references == 3 && uploads.unique_uploads == 2 &&
              uploads.upload_bytes == Pixels * (Layers + 2) * snapshot_bpp,
          "different layer count owns a distinct correctly sized upload");

    repeated_table->resources[2].depth = Layers;
    repeated_table->resources[2].width = W + 1;
    repeated_table->resources[2].layer_stride_bytes = (W + 1) * H * sizeof(float);
    const auto extent_diagnostic = capture_stderr([&] {
        (void)render_submit_items({live_draw}, W, H);
    });
    check(extent_diagnostic.find("[render-array-reject] binding=6 retained depth:") != std::string::npos,
          "incompatible extent cannot borrow the earlier binding's complete array");
    repeated_table->resources[2].width = W;
    repeated_table->resources[2].layer_stride_bytes = Pixels * sizeof(float);

    seed_base = shared_base;
    expected[2] = 0.5626373291015625f;
    if (!seed_layer(2, expected[2])) return 1;
    repeated_shader({expected[0], expected[1], expected[2]});
    live_draw.color0_base += 0x10000;
    check(live_matches(), "later draw observes renderer rewrite rather than a prior draw's shared snapshot");
    uploads = backend_texture_upload_stats();
    check(uploads.references == 3 && uploads.unique_uploads == 1,
          "new generation deduplicates only its own draw's bindings");
    // Real direct-memory aliases: A and B are VA-disjoint views of the same physical bytes;
    // C is another physical range. Warm an array through A before notifying a layer-2 write
    // through B. Old numeric-only invalidation retains all four stale layers and this arm fails.
    uint64_t physical = 0;
    constexpr uint64_t MappingBytes = 0x10000;
    const bool allocated = alloc_direct(0, 0x200000000ull, MappingBytes * 2, MappingBytes, 0,
                                       reinterpret_cast<uint64_t>(&physical)) == 0;
    check(allocated, "physical backing for actual A/B aliases allocated");
    if (!allocated) return 1;
    backing.physical.emplace_back(physical, MappingBytes * 2);
    uint64_t alias_a = 0, alias_b = 0, unrelated = 0;
    for (auto [address, offset] : {std::pair{&alias_a, physical}, std::pair{&alias_b, physical},
                                  std::pair{&unrelated, physical + MappingBytes}}) {
        const bool mapped = map_direct(reinterpret_cast<uint64_t>(address), MappingBytes, 2, 0,
                                       offset, MappingBytes) == 0 && *address;
        check(mapped, "actual guest direct-memory view mapped");
        if (!mapped) return 1;
        backing.mappings.emplace_back(*address, MappingBytes);
    }
    check(alias_a != alias_b && prosper::guest_memory_topology_relation(
              alias_a, Pixels * Layers * sizeof(float), alias_b, Pixels * Layers * sizeof(float)) ==
              prosper::GuestMemoryTopologyRelation::Overlap,
          "A and B are numerically distinct but authoritatively physical aliases");
    check(prosper::guest_memory_topology_relation(alias_a, Pixels * Layers * sizeof(float),
              unrelated, Pixels * Layers * sizeof(float)) == prosper::GuestMemoryTopologyRelation::Disjoint,
          "C is authoritatively disjoint from the retained array");
    seed_base = alias_a;
    live_draw.prt = live_table;
    live_table->resources[0] = texture;
    live_table->resources[0].gpu_addr = alias_a;
    live_table->resources[0].host_data = nullptr;
    live_table->resources[0].host_data_size = 0;
    live_draw.ps = sample_state;
    live_draw.color0_base = Base + 0x1800000;
    for (uint32_t layer = 0; layer < Layers; ++layer)
        if (!seed_layer(layer, expected[layer])) return 1;
    live_shader(2);
    check(live_matches() && ds_layer_stride_for(alias_a, W, H) == Pixels * sizeof(float),
          "actual array A is warm and consumer stride established before alias writes");
    const float changed = 0.3126373291015625f;
    std::memcpy(reinterpret_cast<void*>(unrelated + 2 * Pixels * sizeof(float)), &changed, sizeof(changed));
    notify_guest_gpu_write(unrelated + 2 * Pixels * sizeof(float), sizeof(changed));
    check(live_matches(), "known nonalias C write preserves the warm A array and its pixels");
    const uint64_t layer2_offset = 2 * Pixels * sizeof(float);
    std::memcpy(reinterpret_cast<void*>(alias_b + layer2_offset), &changed, sizeof(changed));
    float observed_alias = 0;
    std::memcpy(&observed_alias, reinterpret_cast<const void*>(alias_a + layer2_offset), sizeof(observed_alias));
    check(std::bit_cast<uint32_t>(observed_alias) == std::bit_cast<uint32_t>(changed),
          "write through B actually changed guest bytes observed through A");
    notify_guest_gpu_write(alias_b + layer2_offset, sizeof(changed));
    const auto physical_alias_diagnostic = capture_stderr([&] {
        (void)render_submit_items({live_draw}, W, H);
    });
    check(physical_alias_diagnostic.find("[render-array-reject] binding=4 retained depth: "
          "retained depth array has missing or invalid layers") != std::string::npos,
          "real frontend refuses warm A after layer-2 physical-alias B write");
    {
        BackendPersistentResourceGuard guard;
        unsigned valid = 0, invalid = 0;
        for (const auto& [key, image] : persistent_ds_cache()) if (key.dr == alias_a) {
            valid += image.depth_valid;
            invalid += !image.depth_valid && key.slice == 2;
        }
        check(valid == Layers - 1 && invalid == 1,
              "physical alias invalidates only its known nonzero layer");
    }
    const auto retained_before_alias = output;
    check(read_persistent_ds_depth_array(alias_a, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == retained_before_alias,
          "physical-alias invalidation cannot republish an old complete snapshot");
    expected[2] = changed;
    if (!seed_layer(2, expected[2])) return 1;
    live_shader(2);
    check(live_matches(), "actual renderer layer rewrite recovers A with new exact pixels");
    // A notification without authoritative mapping coverage must not be treated as disjoint.
    constexpr uint64_t UnknownWrite = 0x1000;
    check(prosper::guest_memory_topology_relation(alias_a, Pixels * Layers * sizeof(float),
              UnknownWrite, sizeof(float)) == prosper::GuestMemoryTopologyRelation::Unknown,
          "untracked notification has no physical-disjointness proof");
    notify_guest_gpu_write(UnknownWrite, sizeof(float));
    (void)is_live_render_target(alias_a);
    const auto before_unknown = output;
    check(read_persistent_ds_depth_array(alias_a, W, H, 0, Layers, output, error) ==
              Status::Unavailable && output == before_unknown,
          "unknown mapping relation conservatively revokes retained array authority");
    return failures ? 1 : 0;
}
