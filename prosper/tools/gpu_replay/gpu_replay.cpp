#include "gpu/gpu_capture.hpp"
#include "gpu/gpu_execute.hpp"
#include "live_renderer.hpp"
#include "render_runner.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool set_environment(const std::string& name, const std::string& value) {
#ifdef _WIN32
    return _putenv_s(name.c_str(), value.c_str()) == 0;
#else
    return setenv(name.c_str(), value.c_str(), 1) == 0;
#endif
}

void usage(const char* argv0) {
    std::fprintf(stderr, "usage: %s [--inspect|--inspect-only] [--draw N[:M]] [--allow-mismatch] "
                         "<capture.prgcap> [output.bmp]\n", argv0);
}

const char* class_name(prosper::gpu::ResourceClass c) {
    using RC = prosper::gpu::ResourceClass;
    switch (c) {
    case RC::ConstantBuffer: return "CB"; case RC::VertexBuffer: return "VB";
    case RC::Texture: return "TEX"; case RC::Sampler: return "SAMP"; case RC::StorageImage: return "STORAGE";
    }
    return "?";
}

void inspect_table(const char* stage, const prosper::gpu::ShaderResourceTable* table) {
    if (!table) { std::printf("  %s: none\n", stage); return; }
    for (const auto& r : table->resources) {
        size_t nz = 0; uint32_t first = 0;
        if (r.host_data) {
            for (uint64_t i = 0; i < r.host_data_size; ++i) nz += r.host_data[i] != 0;
            if (r.host_data_size >= 4) std::memcpy(&first, r.host_data, 4);
        }
        uint64_t hash = r.host_data ? prosper::gpu::gpu_capture_hash(r.host_data, static_cast<size_t>(r.host_data_size)) : 0;
        std::printf("  %s %-7s b=%u addr=%016llx bytes=%llu nz=%zu hash=%016llx first=%08x "
                    "fmt=%u nc=%u stride=%u %ux%u tile=%u srt=%08x sgpr=%08x pc=%08x\n",
                    stage, class_name(r.cls), r.binding, static_cast<unsigned long long>(r.gpu_addr),
                    static_cast<unsigned long long>(r.host_data_size), nz, static_cast<unsigned long long>(hash), first,
                    static_cast<unsigned>(r.format), r.num_components, r.stride, r.width, r.height, r.tile_mode,
                    r.srt_offset, r.sgpr_base, r.fetch_pc);
    }
}

void inspect_frame(const prosper::gpu::GpuReplayFrame& replay) {
    for (size_t i = 0; i < replay.items.size(); ++i) {
        const auto& d = replay.items[i];
        std::printf("draw[%zu] target=%016llx vcount=%u indices=%zu topo=%u fmt=%u cwm=%x "
                    "depth=%d/%d/%u stencil=%d blend=%d viewport=%d %.1f,%.1f %.1fx%.1f "
                    "vs=%zu/%016llx fs=%zu/%016llx\n",
                    i, static_cast<unsigned long long>(d.color0_base), d.vertex_count, d.indices.size(),
                    d.ps.topology, d.ps.color0_format, d.ps.color_write_mask, d.ps.depth_test_enable,
                    d.ps.depth_write_enable, d.ps.depth_compare_op, d.ps.stencil_enable, d.ps.blend_enable,
                    d.ps.has_viewport, d.ps.viewport_x, d.ps.viewport_y, d.ps.viewport_w, d.ps.viewport_h,
                    d.vs.size(), static_cast<unsigned long long>(prosper::gpu::gpu_capture_hash(
                        reinterpret_cast<const uint8_t*>(d.vs.data()), d.vs.size() * 4)),
                    d.fs.size(), static_cast<unsigned long long>(prosper::gpu::gpu_capture_hash(
                        reinterpret_cast<const uint8_t*>(d.fs.data()), d.fs.size() * 4)));
        std::printf("  stencil clear=%u cmp=%u/%u fail=%u/%u pass=%u/%u zfail=%u/%u "
                    "ref=%u/%u opval=%u/%u cmask=%02x/%02x wmask=%02x/%02x\n",
                    d.ps.stencil_clear_value, d.ps.stencil_compare_op[0], d.ps.stencil_compare_op[1],
                    d.ps.stencil_fail_op[0], d.ps.stencil_fail_op[1], d.ps.stencil_pass_op[0],
                    d.ps.stencil_pass_op[1], d.ps.stencil_depth_fail_op[0], d.ps.stencil_depth_fail_op[1],
                    d.ps.stencil_ref[0], d.ps.stencil_ref[1], d.ps.stencil_op_val[0], d.ps.stencil_op_val[1],
                    d.ps.stencil_compare_mask[0], d.ps.stencil_compare_mask[1],
                    d.ps.stencil_write_mask[0], d.ps.stencil_write_mask[1]);
        std::printf("  ds rc=%08x clear=%d/%d programmed=%d/%d base z=%016llx/%016llx s=%016llx/%016llx "
                    "rawops=%u,%u,%u/%u,%u,%u shaderctl=%08x stexport=%d/%d\n",
                    d.ps.db_render_control, d.ps.depth_clear_enable, d.ps.stencil_clear_enable,
                    d.ps.has_depth_clear, d.ps.has_stencil_clear,
                    static_cast<unsigned long long>(d.ps.depth_read_base),
                    static_cast<unsigned long long>(d.ps.depth_write_base),
                    static_cast<unsigned long long>(d.ps.stencil_read_base),
                    static_cast<unsigned long long>(d.ps.stencil_write_base),
                    d.ps.raw_stencil_op[0][0], d.ps.raw_stencil_op[0][1], d.ps.raw_stencil_op[0][2],
                    d.ps.raw_stencil_op[1][0], d.ps.raw_stencil_op[1][1], d.ps.raw_stencil_op[1][2],
                    d.ps.db_shader_control, d.ps.stencil_test_val_export_enable,
                    d.ps.stencil_op_val_export_enable);
        inspect_table("VS", d.vrt.get()); inspect_table("PS", d.prt.get());
    }
}

} // namespace

int main(int argc, char** argv) {
    bool inspect = false, inspect_only = false, allow_mismatch = false; int draw_first = -1, draw_last = -1;
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--inspect") inspect = true;
        else if (std::string(argv[i]) == "--inspect-only") inspect = inspect_only = true;
        else if (std::string(argv[i]) == "--allow-mismatch") allow_mismatch = true;
        else if (std::string(argv[i]) == "--draw" && i + 1 < argc) {
            std::string range = argv[++i]; size_t colon = range.find(':');
            draw_first = std::atoi(range.c_str());
            draw_last = colon == std::string::npos ? draw_first : std::atoi(range.c_str() + colon + 1);
        }
        else positional.push_back(argv[i]);
    }
    if (positional.empty() || positional.size() > 2) { usage(argv[0]); return 2; }
    prosper::gpu::GpuCaptureFile capture; std::string error;
    if (!prosper::gpu::read_gpu_capture(positional[0], capture, error)) {
        std::fprintf(stderr, "gpu_replay: %s: %s\n", positional[0], error.c_str()); return 2;
    }
    for (const auto& [name, value] : capture.metadata.renderer_env) {
        if (!set_environment(name, value)) {
            std::fprintf(stderr, "gpu_replay: cannot set %s\n", name.c_str()); return 2;
        }
    }
    // A replay is already at its target submit; live-run windowing must never skip it.
    set_environment("PROSPER_RENDER_FIRST", "0"); set_environment("PROSPER_RENDER_LAST", "2147483647");
    prosper::gpu::GpuReplayFrame replay;
    if (!prosper::gpu::materialize_gpu_replay(capture, replay, error)) {
        std::fprintf(stderr, "gpu_replay: cannot materialize: %s\n", error.c_str()); return 2;
    }
    if (draw_first >= 0) {
        if (draw_last < draw_first || static_cast<size_t>(draw_last) >= replay.items.size()) {
            std::fprintf(stderr, "gpu_replay: draw range %d:%d is out of range\n", draw_first, draw_last); return 2;
        }
        std::vector<prosper::gpu::DrawItem> selected;
        for (int i = draw_first; i <= draw_last; ++i) selected.push_back(std::move(replay.items[i]));
        replay.items = std::move(selected); allow_mismatch = true;
        std::fprintf(stderr, "[gpureplay] selected original draws %d:%d\n", draw_first, draw_last);
    }
    const auto& m = replay.metadata;
    std::fprintf(stderr, "[gpureplay] rev=%s title=%s submit=%llu %ux%u draws=%zu blobs=%zu\n",
                 m.revision.c_str(), m.title_id.c_str(), static_cast<unsigned long long>(m.submit_index),
                 m.width, m.height, replay.items.size(), replay.blobs.size());
    if (inspect) inspect_frame(replay);
    if (inspect_only) return 0;
    prosper::frontend::register_live_renderer(".", false);
    std::vector<uint8_t> pixels = prosper::gpu::render_submit_items(replay.items, m.width, m.height);
    uint64_t hash = prosper::gpu::gpu_capture_hash(pixels);
    std::fprintf(stderr, "[gpureplay] output_bytes=%zu hash=%016llx expected_bytes=%llu expected_hash=%016llx\n",
                 pixels.size(), static_cast<unsigned long long>(hash),
                 static_cast<unsigned long long>(replay.expected_output_bytes),
                 static_cast<unsigned long long>(replay.expected_output_hash));
    if (positional.size() == 2 && !pixels.empty() && !prosper::test::dump_bmp(positional[1], pixels, m.width, m.height)) {
        std::fprintf(stderr, "gpu_replay: cannot write %s\n", positional[1]); return 2;
    }
    if (!allow_mismatch && (pixels.size() != replay.expected_output_bytes || hash != replay.expected_output_hash)) {
        std::fprintf(stderr, "gpu_replay: output mismatch\n"); return 1;
    }
    return 0;
}
