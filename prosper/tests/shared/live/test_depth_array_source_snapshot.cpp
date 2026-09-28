// Retained depth arrays preserve Float32 contents and publish only complete, ordered snapshots.
//
// Every live-frontend arm runs against one of two routes: the CPU readback + re-upload bridge
// (default arms, PROSPER_NO_GPU_DEPTH_ARRAY=1) or the GPU-resident copy (`--gpu`). Both must
// produce the same sampled pixels on every check below; the representation checks count CPU
// upload bytes on the CPU route and GPU-copied output bytes (with zero CPU upload) on the GPU one.
#include "fixtures/render_runner.h"
#include "fixtures/retained_depth_array_gpu.h"
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
    bool per_draw_control = false, expanded_control = false, gpu_route = false;
    bool texture_path_census_budget = false;
    bool texture_path_census_empty = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--per-draw-control") == 0) per_draw_control = true;
        else if (std::strcmp(argv[i], "--expanded-control") == 0) expanded_control = true;
        else if (std::strcmp(argv[i], "--gpu") == 0) gpu_route = true;
        else if (std::strcmp(argv[i], "--texture-path-census-budget") == 0)
            texture_path_census_budget = true;
        else if (std::strcmp(argv[i], "--texture-path-census-empty") == 0)
            texture_path_census_empty = true;
        else {
            std::fprintf(stderr, "usage: %s [--gpu] [--per-draw-control] [--expanded-control] "
                                 "[--texture-path-census-budget|"
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
    _putenv_s("PROSPER_RENDER_TIMING", "1");
    if (per_draw_control) _putenv_s("PROSPER_NO_SUBMIT_DEPTH_ARRAY_SNAPSHOT_REUSE", "1");
    if (expanded_control) _putenv_s("PROSPER_NO_COMPACT_DEPTH_ARRAY_SNAPSHOT", "1");
    if (!gpu_route) _putenv_s("PROSPER_NO_GPU_DEPTH_ARRAY", "1");
#else
    setenv("PROSPER_DEPTH_ARRAY_SNAPSHOT_CENSUS", "1", 1);
    setenv("PROSPER_ARRAY_REJECT_LOG_ALL", "1", 1);
    setenv("PROSPER_RENDER_TIMING", "1", 1);
    if (per_draw_control) setenv("PROSPER_NO_SUBMIT_DEPTH_ARRAY_SNAPSHOT_REUSE", "1", 1);
    if (expanded_control) setenv("PROSPER_NO_COMPACT_DEPTH_ARRAY_SNAPSHOT", "1", 1);
    if (!gpu_route) setenv("PROSPER_NO_GPU_DEPTH_ARRAY", "1", 1);
    else unsetenv("PROSPER_NO_GPU_DEPTH_ARRAY");
#endif
    check(gpu_depth_array_snapshots_enabled() == gpu_route,
          "the selected retained-array route is the one this arm asked for");
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
    if (gpu_route) {
        // GPU snapshot ownership (#3882 review). A snapshot is filled by a command buffer in ONE
        // ordered batch: it may be served only to consumers recorded later into that same batch,
        // and a failure of that batch after the copy was enqueued must revoke it.
        const auto& ctx = render_vk_ctx();
        auto copy_into = [&](BackendSubmissionBatch& batch) {
            std::shared_ptr<PersistentDsDepthArrayGpuImage> snapshot;
            std::string copy_error;
            const auto result = copy_persistent_ds_depth_array_gpu(
                Base, W, H, 0, Layers, VK_FORMAT_R32_SFLOAT, batch, snapshot, copy_error);
            if (result != DepthArrayGpuResult::Ready)
                std::fprintf(stderr, "gpu copy: %s\n", copy_error.c_str());
            return result == DepthArrayGpuResult::Ready ? snapshot : nullptr;
        };
        BackendSubmissionBatch completed_batch, other_batch;
        auto completed = copy_into(completed_batch);
        check(completed && completed_batch.pending(),
              "GPU snapshot copy is recorded into the caller's ordered batch, not submitted");
        check(completed && completed->servable_to(&completed_batch),
              "GPU snapshot serves a consumer in its own batch");
        check(completed && !completed->servable_to(nullptr) &&
                  !completed->servable_to(&other_batch),
              "GPU snapshot refuses a direct submission or another batch that could run first");
        const auto submitted = completed_batch.submit_and_wait(ctx.dev, ctx.queue, false);
        check(submitted.submit_result == VK_SUCCESS && submitted.wait_result == VK_SUCCESS &&
                  completed && completed->valid.load(),
              "a completed owning batch leaves the snapshot valid (positive control)");
        BackendSubmissionBatch failed_batch;
        auto failed = copy_into(failed_batch);
        check(failed && failed->servable_to(&failed_batch),
              "second snapshot is servable before its batch fails");
        failed_batch.discard();
        check(failed && !failed->valid.load() && !failed->servable_to(&failed_batch),
              "a batch failing AFTER the copy was enqueued revokes the snapshot");
    }
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
    // The sampled representation produced since mark_payload(): CPU route -> the last backend call's
    // CPU upload; GPU route -> GPU-copied output bytes, and the CPU upload must be exactly zero (a
    // silent fallback to the CPU route fails here even when the pixels are right).
    uint64_t gpu_payload_mark = 0, gpu_copy_mark = 0;
    const auto mark_payload = [&] {
        gpu_payload_mark = depth_array_gpu_copy_stats().output_bytes;
        gpu_copy_mark = depth_array_gpu_copy_stats().copies;
    };
    // Distinct materialized array images since mark_payload(): backend CPU uploads on the CPU
    // route; GPU copies on the GPU route, whose borrowed images are not backend uploads at all.
    const auto materialized_is = [&](size_t expected_images) {
        const size_t cpu = backend_texture_upload_stats().unique_uploads;
        if (!gpu_route) return cpu == expected_images;
        return cpu == 0 &&
               depth_array_gpu_copy_stats().copies - gpu_copy_mark == expected_images;
    };
    const auto payload_is = [&](uint64_t expected_bytes) {
        const uint64_t cpu = backend_texture_upload_stats().upload_bytes;
        if (!gpu_route) return cpu == expected_bytes;
        const uint64_t gpu = depth_array_gpu_copy_stats().output_bytes - gpu_payload_mark;
        if (cpu != 0 || gpu != expected_bytes)
            std::printf("[snapshot-fixture] gpu route payload cpu=%llu gpu=%llu expected=%llu\n",
                        (unsigned long long)cpu, (unsigned long long)gpu,
                        (unsigned long long)expected_bytes);
        return cpu == 0 && gpu == expected_bytes;
    };
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
        mark_payload();
        check(live_matches(), "array gather preserves precise R and substituted G/B/A components");
        check(payload_is(Pixels * Layers * snapshot_bpp),
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
        mark_payload();
        check(live_matches(), "linear array sampler preserves exact uniform-layer components");
        check(payload_is(Pixels * Layers * payload_bpp(true)),
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
        mark_payload();
        check(live_matches(), "out-of-range gather preserves transparent alpha and opaque-white green");
        check(payload_is(Pixels * Layers * 4 * sizeof(float)),
              "border-sensitive retained array deliberately keeps the expanded representation");
    }
    component_resource.addr_uvw[0] = 0; component_resource.border_color_type = 0;
    component_shader(false, 0, 0.0f, {0.0f, 0.0f, 0.0f, 0.0f}, true);
    mark_payload();
    check(live_matches(), "out-of-bounds array fetch preserves expanded robust zero including alpha");
    check(payload_is(Pixels * Layers * 4 * sizeof(float)),
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
    mark_payload();
    const auto shared_pixels = render_submit_items({live_producer, live_draw}, W, H);
    const size_t shared_center = (W * (H / 2) + W / 2) * 4;
    bool shared_matches = shared_pixels.size() == Pixels * 4;
    if (shared_matches) for (size_t c = 0; c < 4; ++c)
        shared_matches &= shared_pixels[shared_center + c] >= 127 &&
                          shared_pixels[shared_center + c] <= 128;
    check(shared_matches, "shared array payload preserves layers/samplers after pending producer flush");
    auto uploads = backend_texture_upload_stats();
    check(uploads.references == 3 && materialized_is(1) &&
              payload_is(Pixels * Layers * snapshot_bpp),
          "three array bindings produce one actual backend image/staging upload");
    const auto valid_bindings = backend_resource_reuse_stats();
    const auto valid_timing = backend_render_timing_stats();
    check(valid_bindings.texture_binding_references >= 3 &&
              valid_bindings.unique_texture_bindings > 0,
          "valid call populates backend texture binding counters before refusal");
    check(valid_timing.res_texture_upload_ms > 0 && valid_timing.res_texture_bind_ms > 0,
          "valid call populates backend texture subphase times before refusal");
    // A valid call is the positive control for this two-call diagnostic: the early compact-order
    // refusal must publish zero texture work, not leave the previous call's plausible counters.
    BackendDraw malformed_order;
    malformed_order.B.push_back({});
    const auto refusal = capture_stderr([&] {
        const auto rejected = render_draws_rgba({malformed_order}, W, H);
        check(rejected.empty(), "malformed compact order refuses before rendering");
    });
    check(refusal.find("compact resources require explicit order metadata") != std::string::npos,
          "the preflight arm reached its intended early refusal");
    uploads = backend_texture_upload_stats();
    check(uploads.references == 0 && uploads.unique_uploads == 0 && uploads.upload_bytes == 0,
          "preflight refusal clears the previous backend texture counters");
    const auto refused_bindings = backend_resource_reuse_stats();
    const auto refused_timing = backend_render_timing_stats();
    check(refused_bindings.texture_binding_references == 0 &&
              refused_bindings.unique_texture_bindings == 0,
          "preflight refusal clears the previous backend texture binding counters");
    check(refused_timing.res_texture_upload_ms == 0 && refused_timing.res_texture_bind_ms == 0,
          "preflight refusal clears the previous backend texture subphase times");

    // Separate draws in one real callback share a stable retained generation. Distinct samplers
    // still yield six references, while the immutable pixel owner permits one actual upload. The
    // old draw-local memo makes two unique uploads here even when the final pixels happen to match.
    live_draw.color0_base += 0x10000;
    mark_payload();
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
    check(materialized_is(expected_uploads),
          "callback policy shares one upload; startup per-draw control requires two");
    check(payload_is(expected_uploads * Pixels * Layers * snapshot_bpp),
          "actual uploaded bytes agree with the independently expected policy count");
    const auto empty_pass = render_draw_pass_rgba(std::span<const BackendDraw>{}, W, H);
    check(empty_pass.empty(), "empty direct backend pass exits before any Vulkan work");
    uploads = backend_texture_upload_stats();
    check(uploads.references == 0 && uploads.unique_uploads == 0 && uploads.upload_bytes == 0,
          "empty direct backend pass clears the previous texture counters");

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
    mark_payload();
    check(live_matches(), "different retained array bases keep independent sampled values");
    uploads = backend_texture_upload_stats();
    check(uploads.references == 3 && materialized_is(2),
          "different array base does not reuse the first snapshot/upload");

    repeated_table->resources[2].gpu_addr = shared_base;
    repeated_table->resources[2].depth = 2;
    repeated_shader({expected[0], expected[1], expected[1]});
    live_draw.color0_base += 0x10000;
    mark_payload();
    check(live_matches(), "different array layer count retains the requested view");
    uploads = backend_texture_upload_stats();
    check(uploads.references == 3 && materialized_is(2) &&
              payload_is(Pixels * (Layers + 2) * snapshot_bpp),
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
    mark_payload();
    check(live_matches(), "later draw observes renderer rewrite rather than a prior draw's shared snapshot");
    uploads = backend_texture_upload_stats();
    check(uploads.references == 3 && materialized_is(1),
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
    // #3893: a cascade array with only layer 0 ever rendered as depth. Sonic Frontiers' menus
    // render cascade 0 of a four-layer D32 shadow array and sample all four; no DS pass names
    // layers 1-3 at any extent, so their guest bytes are what the hardware samples. The bridge used
    // to refuse the whole binding, dropping the draw.
    {
        const uint64_t guest_base = Base + 0x1c00000;
        const uint64_t layer_bytes = Pixels * sizeof(float);
        auto* guest_words = reinterpret_cast<float*>(guest_base);
        // Layer 1 and 3 uniform (a cleared or never-written cascade), layer 2 a gradient, so both
        // the uniform fast path and the ordinary linear copy are exercised. Layer 0's guest bytes
        // disagree with its retained pixels and must never be used.
        const float guest_uniform1 = 0.2813720703125f, guest_uniform3 = 0.0f;
        for (size_t i = 0; i < Pixels; ++i) {
            guest_words[i] = 0.9f;
            guest_words[Pixels + i] = guest_uniform1;
            guest_words[2 * Pixels + i] = 0.015625f * float(i);
            guest_words[3 * Pixels + i] = guest_uniform3;
        }
        const size_t center_texel = W * (H / 2) + W / 2;
        const std::array<float, Layers> guest_expected{0.3438720703125f, guest_uniform1,
            0.015625f * float(center_texel), guest_uniform3};
        seed_base = guest_base;
        if (!seed_layer(0, guest_expected[0])) return 1;
        const DepthArrayGuestSource source{0u, layer_bytes, W * sizeof(float)};
        std::vector<float> partial{7.0f};
        const auto untouched = partial;
        check(read_persistent_ds_depth_array(guest_base, W, H, 0, Layers, partial, error) ==
                  Status::Unavailable && partial == untouched &&
                  error.find("missing or invalid layers") != std::string::npos,
              "without a guest source a partially rendered array is still refused (pre-#3893)");
        check(read_persistent_ds_depth_array(guest_base, W, H, 0, Layers, partial, error,
                                             nullptr, &source) == Status::Ready,
              "never-rendered layers are admitted from their guest bytes");
        bool partial_exact = partial.size() == Pixels * Layers;
        for (size_t i = 0; partial_exact && i < Pixels; ++i) {
            partial_exact &= std::bit_cast<uint32_t>(partial[i]) ==
                             std::bit_cast<uint32_t>(guest_expected[0]);
            partial_exact &= partial[Pixels + i] == guest_uniform1;
            partial_exact &= partial[2 * Pixels + i] == 0.015625f * float(i);
            partial_exact &= partial[3 * Pixels + i] == guest_uniform3;
        }
        check(partial_exact,
              "retained layer 0 keeps renderer pixels; layers 1-3 carry their exact guest values");
        // A layer an identity names -- even at another extent -- is renderer-owned: its guest bytes
        // are stale, so it must keep refusing rather than silently fall back.
        PersistentDsKey other_extent{guest_base, guest_base, 0, 0, 0, W * 2, H,
                                     VK_FORMAT_D32_SFLOAT, 2};
        {
            BackendPersistentResourceGuard guard;
            persistent_ds_cache()[other_extent] = {};
        }
        check(read_persistent_ds_depth_array(guest_base, W, H, 0, Layers, partial, error,
                                             nullptr, &source) == Status::Unavailable,
              "a layer named by a retained identity at another extent is never read from guest bytes");
        {
            BackendPersistentResourceGuard guard;
            persistent_ds_cache().erase(other_extent);
            for (auto& [key, image] : persistent_ds_cache())
                if (key.dr == guest_base && key.slice == 0) image.depth_valid = false;
        }
        check(read_persistent_ds_depth_array(guest_base, W, H, 0, Layers, partial, error,
                                             nullptr, &source) == Status::Unavailable,
              "an invalidated retained layer still refuses; guest bytes never replace it");
        if (!seed_layer(0, guest_expected[0])) return 1;

        // The production frontend, for every layer, on both routes.
        live_table->resources[0] = texture;
        live_table->resources[0].gpu_addr = guest_base;
        live_table->resources[0].host_data = nullptr;
        live_table->resources[0].host_data_size = 0;
        live_table->resources[0].size = layer_bytes * Layers;
        live_draw.prt = live_table;
        live_draw.ps = sample_state;
        live_draw.color0_base = Base + 0x1d00000;
        const auto saved_expected = expected;
        expected = guest_expected;
        for (uint32_t layer = 0; layer < Layers; ++layer) {
            live_shader(layer);
            live_draw.color0_base += 0x10000;
            mark_payload();
            const auto diagnostic = capture_stderr([&] {
                check(live_matches(), "frontend samples a partially rendered array at every layer");
            });
            check(diagnostic.find("[render-array-reject]") == std::string::npos,
                  "a partially rendered array is not rejected by the frontend");
            // Layer 2 is a gradient: the GPU route cannot fill it and must hand the whole array to
            // the CPU route, never publish a partial GPU copy.
            if (gpu_route)
                check(depth_array_gpu_copy_stats().copies == gpu_copy_mark &&
                          backend_texture_upload_stats().upload_bytes ==
                              Pixels * Layers * snapshot_bpp,
                      "a non-uniform guest layer takes the CPU route (no GPU copy)");
        }
        // The GPU route fills uniform guest layers on the GPU; a non-uniform one needs a detile,
        // so it takes the CPU route. Make layer 2 uniform and require a GPU copy.
        for (size_t i = 0; i < Pixels; ++i) guest_words[2 * Pixels + i] = 0.6563720703125f;
        expected[2] = 0.6563720703125f;
        live_shader(2);
        live_draw.color0_base += 0x10000;
        mark_payload();
        check(live_matches(), "frontend observes a guest rewrite of a never-rendered layer");
        if (gpu_route)
            check(materialized_is(1) && payload_is(Pixels * Layers * snapshot_bpp),
                  "uniform guest layers stay on the GPU route (fill, no CPU upload)");
        expected = saved_expected;

        // #3893 review (blocking): a DS view with SLICE_MAX > SLICE_START -- one clear over the
        // whole shadow array, or layered rendering -- writes layers the key (SLICE_START only)
        // never names. Through the production DS path, attach slice 0 with SLICE_MAX = 3: layers
        // 1-3 were written by the renderer and their guest bytes are stale, so the array must keep
        // refusing rather than read them from guest memory.
        const uint64_t span_base = Base + 0x1e00000;
        std::memset(reinterpret_cast<void*>(span_base), 0, layer_bytes * Layers);
        ResolvedPipelineState span_state = producer_state;
        span_state.depth_read_base = span_state.depth_write_base = span_base;
        span_state.db_depth_view = 0u | (3u << 13);   // SLICE_START 0, SLICE_MAX 3
        span_state.min_depth = span_state.max_depth = 0.75f;
        BackendDraw span_producer = producer;
        span_producer.ps = &span_state;
        BackendSubmissionBatch span_batch;
        (void)render_draws_rgba({span_producer}, W, H, nullptr, nullptr, true, nullptr,
                                nullptr, nullptr, nullptr, &span_batch, false, nullptr, false);
        {
            BackendPersistentResourceGuard guard;
            uint32_t recorded = 0;
            for (const auto& [key, image] : persistent_ds_cache())
                if (key.dr == span_base) recorded = image.programmed_slice_max;
            check(recorded == 3, "a multi-slice DS view records its programmed SLICE_MAX");
        }
        const DepthArrayGuestSource span_source{0u, layer_bytes, W * sizeof(float)};
        check(read_persistent_ds_depth_array(span_base, W, H, 0, Layers, partial, error,
                                             &span_batch, &span_source) == Status::Unavailable,
              "layers inside a multi-slice DS view's span are never read from guest bytes");

        // #3906: the span is the widest over EVERY depth-using draw of the pass. A narrow first
        // draw (SLICE_MAX 0) followed by a whole-array one (SLICE_MAX 3) in the same pass wrote
        // layers 1-3 too.
        const uint64_t span2_base = Base + 0x1f00000;
        std::memset(reinterpret_cast<void*>(span2_base), 0, layer_bytes * Layers);
        ResolvedPipelineState narrow_state = span_state;
        narrow_state.depth_read_base = narrow_state.depth_write_base = span2_base;
        narrow_state.db_depth_view = 0u;               // SLICE_START 0, SLICE_MAX 0
        ResolvedPipelineState wide_state = narrow_state;
        wide_state.db_depth_view = 0u | (3u << 13);   // SLICE_START 0, SLICE_MAX 3
        BackendDraw narrow_producer = producer, wide_producer = producer;
        narrow_producer.ps = &narrow_state;
        wide_producer.ps = &wide_state;
        BackendSubmissionBatch span2_batch;
        (void)render_draws_rgba({narrow_producer, wide_producer}, W, H, nullptr, nullptr, true,
                                nullptr, nullptr, nullptr, nullptr, &span2_batch, false, nullptr,
                                false);
        {
            BackendPersistentResourceGuard guard;
            uint32_t recorded = 0;
            for (const auto& [key, image] : persistent_ds_cache())
                if (key.dr == span2_base) recorded = image.programmed_slice_max;
            check(recorded == 3, "a later, wider draw of the same pass records its SLICE_MAX");
        }
        const DepthArrayGuestSource span2_source{0u, layer_bytes, W * sizeof(float)};
        check(read_persistent_ds_depth_array(span2_base, W, H, 0, Layers, partial, error,
                                             &span2_batch, &span2_source) == Status::Unavailable,
              "layers written by a later, wider draw of the pass are never read from guest bytes");

        // Rebased planes name the layers their own slices and extents can touch. Layers 0 and 1
        // are retained at the base; a plane rebased to layer 1 with its own slice 1 writes layer 2,
        // and a plane rebased to layer 1 whose extent is two layers tall covers layer 2 too.
        const uint64_t rebased_base = Base + 0x1e80000;
        std::memset(reinterpret_cast<void*>(rebased_base), 0, layer_bytes * Layers);
        seed_base = rebased_base;
        if (!seed_layer(0, 0.25f) || !seed_layer(1, 0.5f)) return 1;
        const DepthArrayGuestSource rebased_source{0u, layer_bytes, W * sizeof(float)};
        check(read_persistent_ds_depth_array(rebased_base, W, H, 0, Layers, partial, error,
                                             nullptr, &rebased_source) == Status::Ready,
              "control: layers 2-3 of a two-layer retained array are admitted from guest bytes");
        const uint64_t layer1 = rebased_base + layer_bytes;
        const PersistentDsKey rebased_slice{layer1, layer1, 0, 0, 0, W, H,
                                            VK_FORMAT_D32_SFLOAT, 1};
        const PersistentDsKey rebased_tall{layer1, layer1, 0, 0, 0, W, H * 2,
                                           VK_FORMAT_D32_SFLOAT, 0};
        // #3906: an identity of ANOTHER extent is padded to whole 256x256 texels. A 4x4 plane at
        // the array base covers 64 bytes unpadded (layer 0, already retained), but its real
        // tiled surface can reach layers 2-3.
        const PersistentDsKey small_other{rebased_base, rebased_base, 0, 0, 0, 4, 4,
                                          VK_FORMAT_D32_SFLOAT, 0};
        for (const auto& [plane, label] :
             {std::pair{rebased_slice,
                        "a rebased plane's own nonzero slice names the layer it actually writes"},
              std::pair{rebased_tall,
                        "a rebased plane taller than one layer names every layer it covers"},
              std::pair{small_other,
                        "an other-extent identity is padded to 256x256 texels and names the "
                        "layers its tiled surface can reach"}}) {
            {
                BackendPersistentResourceGuard guard;
                persistent_ds_cache()[plane] = {};
            }
            check(read_persistent_ds_depth_array(rebased_base, W, H, 0, Layers, partial, error,
                                                 nullptr, &rebased_source) ==
                      Status::Unavailable, label);
            {
                BackendPersistentResourceGuard guard;
                persistent_ds_cache().erase(plane);
            }
        }

        // A guest layer whose bytes are a live COLOR target (renderer-owned pixels) is refused,
        // including layer 0 and a target that starts inside a layer rather than at its start.
        live_table->resources[0].gpu_addr = rebased_base;
        live_shader(0);
        live_draw.color0_base += 0x10000;
        const auto admitted = capture_stderr([&] { (void)render_submit_items({live_draw}, W, H); });
        check(admitted.find("[render-array-reject]") == std::string::npos,
              "control: the two-layer retained array is admitted before any colour target");
        // #3906: an unresolved MSAA colour target's guest footprint is `samples` times its
        // single-sample size. A target 512 KiB before layer 2 misses it single-sample (256 KiB
        // padded footprint) and covers it at 4x.
        DrawItem msaa_writer;
        msaa_writer.vs = producer.vs; msaa_writer.fs = producer.fs;
        msaa_writer.vertex_count = 3; msaa_writer.ps = sample_state;
        msaa_writer.color0_base = rebased_base + 2 * layer_bytes - 0x80000;
        msaa_writer.color0_width = W; msaa_writer.color0_height = H;
        (void)render_submit_items({msaa_writer}, W, H);
        live_draw.color0_base += 0x10000;
        const auto single_sample = capture_stderr([&] { (void)render_submit_items({live_draw}, W, H); });
        check(single_sample.find("[render-array-reject]") == std::string::npos,
              "control: a single-sample colour target 512 KiB before a guest layer does not overlap it");
        msaa_writer.ps.color_targets[0].log2_samples = 2;
        (void)render_submit_items({msaa_writer}, W, H);
        live_draw.color0_base += 0x10000;
        const auto four_sample = capture_stderr([&] { (void)render_submit_items({live_draw}, W, H); });
        check(four_sample.find("overlaps a live color target") != std::string::npos,
              "a 4x MSAA colour target's sample-scaled footprint overlaps the guest layer");
        DrawItem color_writer;
        color_writer.vs = producer.vs; color_writer.fs = producer.fs;
        color_writer.vertex_count = 3; color_writer.ps = sample_state;
        color_writer.color0_base = rebased_base + 2 * layer_bytes + 64;
        color_writer.color0_width = W; color_writer.color0_height = H;
        (void)render_submit_items({color_writer}, W, H);
        live_draw.color0_base += 0x10000;
        const auto overlapped = capture_stderr([&] { (void)render_submit_items({live_draw}, W, H); });
        check(overlapped.find("overlaps a live color target") != std::string::npos,
              "a guest layer overlapping a live colour target keeps the refusal");
    }

    const auto gpu_stats = depth_array_gpu_copy_stats();
    std::printf("[snapshot-fixture] route=%s gpu_copies=%llu gpu_layers=%llu pool_hits=%llu "
                "pool_misses=%llu\n", gpu_route ? "gpu" : "cpu",
                (unsigned long long)gpu_stats.copies, (unsigned long long)gpu_stats.layers,
                (unsigned long long)gpu_stats.pool_hits, (unsigned long long)gpu_stats.pool_misses);
    check(gpu_route ? gpu_stats.copies > 0 : gpu_stats.copies == 0,
          "only the GPU arm records GPU copies; the CPU arms never touch the new route");
    return failures ? 1 : 0;
}
