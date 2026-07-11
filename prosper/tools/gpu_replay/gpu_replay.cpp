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
    std::fprintf(stderr, "usage: %s [--inspect|--inspect-only|--validate] [--draw N[:M]] "
                         "[--dump-resource DRAW:vs|ps:BINDING PATH] [--allow-mismatch] "
                         "[--dump-shader DRAW:vs|fs PATH] "
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
                    "fmt=%u nc=%u stride=%u %ux%u tile=%u addr=%u%u%u swz=%u%u%u%u filt=%u/%u/%u "
                    "srt=%08x sgpr=%08x pc=%08x\n",
                    stage, class_name(r.cls), r.binding, static_cast<unsigned long long>(r.gpu_addr),
                    static_cast<unsigned long long>(r.host_data_size), nz, static_cast<unsigned long long>(hash), first,
                    static_cast<unsigned>(r.format), r.num_components, r.stride, r.width, r.height, r.tile_mode,
                    r.addr_uvw[0], r.addr_uvw[1], r.addr_uvw[2],
                    r.swizzle[0], r.swizzle[1], r.swizzle[2], r.swizzle[3],
                    r.mag_filter, r.min_filter, r.mip_filter,
                    r.srt_offset, r.sgpr_base, r.fetch_pc);
        if (r.host_data && r.host_data_size >= 16 &&
            (r.cls == prosper::gpu::ResourceClass::ConstantBuffer ||
             r.cls == prosper::gpu::ResourceClass::VertexBuffer)) {
            uint32_t u[4]; float f[4]; std::memcpy(u, r.host_data, sizeof(u)); std::memcpy(f, u, sizeof(f));
            std::printf("    first4 u32=%08x,%08x,%08x,%08x f32=%g,%g,%g,%g\n",
                        u[0], u[1], u[2], u[3], f[0], f[1], f[2], f[3]);
        }
    }
}

void inspect_frame(const prosper::gpu::GpuReplayFrame& replay) {
    for (size_t i = 0; i < replay.items.size(); ++i) {
        const auto& d = replay.items[i];
        std::printf("draw[%zu] target=%016llx extent=%ux%u vcount=%u indices=%zu topo=%u fmt=%u cwm=%x "
                    "depth=%d/%d/%u stencil=%d blend=%d viewport=%d %.1f,%.1f %.1fx%.1f "
                    "vs=%zu/%016llx fs=%zu/%016llx\n",
                    i, static_cast<unsigned long long>(d.color0_base), d.color0_width, d.color0_height,
                    d.vertex_count, d.indices.size(),
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

bool validate_frame(const prosper::gpu::GpuReplayFrame& replay) {
    bool valid = true;
    for (size_t i = 0; i < replay.items.size(); ++i) {
        const auto& d = replay.items[i];
        struct StageInput {
            const char* name;
            const std::vector<uint32_t>* spirv;
            const prosper::gpu::ShaderResourceTable* table;
            uint32_t set;
            prosper::gpu::SpirvShaderStage stage;
        } stages[] = {
            {"VS", &d.vs, d.vrt.get(), 0, prosper::gpu::SpirvShaderStage::Vertex},
            {"PS", &d.fs, d.prt.get(), 1, prosper::gpu::SpirvShaderStage::Fragment},
        };
        for (const auto& s : stages) {
            auto report = prosper::gpu::validate_spirv_descriptor_interface(
                *s.spirv, s.table, s.set, s.stage, true);
            std::printf("draw[%zu] %s descriptors=%zu runtime=%zu result=%s\n", i, s.name,
                        report.descriptors.size(), s.table ? s.table->resources.size() : 0,
                        report.ok() ? "accept" : "reject");
            for (const auto& binding : report.descriptors)
                std::printf("  set=%u binding=%u type=%s required=%llu%s\n",
                            binding.set, binding.binding,
                            prosper::gpu::spirv_descriptor_kind_name(binding.kind),
                            static_cast<unsigned long long>(binding.required_bytes),
                            binding.dynamic_access ? "+dynamic" : "");
            for (const auto& issue : report.issues) {
                std::printf("  %s %s set=%u binding=%u expected=%s actual=%s required=%llu available=%llu\n",
                            issue.error ? "ERROR" : "warn",
                            prosper::gpu::descriptor_issue_name(issue.code), issue.set, issue.binding,
                            prosper::gpu::spirv_descriptor_kind_name(issue.expected),
                            prosper::gpu::spirv_descriptor_kind_name(issue.actual),
                            static_cast<unsigned long long>(issue.required_bytes),
                            static_cast<unsigned long long>(issue.available_bytes));
                valid &= !issue.error;
            }
        }
    }
    return valid;
}

} // namespace

int main(int argc, char** argv) {
    bool inspect = false, inspect_only = false, validate_only = false, allow_mismatch = false;
    int draw_first = -1, draw_last = -1;
    std::string dump_spec, dump_path, shader_spec, shader_path;
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--inspect") inspect = true;
        else if (std::string(argv[i]) == "--inspect-only") inspect = inspect_only = true;
        else if (std::string(argv[i]) == "--validate") validate_only = true;
        else if (std::string(argv[i]) == "--allow-mismatch") allow_mismatch = true;
        else if (std::string(argv[i]) == "--draw" && i + 1 < argc) {
            std::string range = argv[++i]; size_t colon = range.find(':');
            draw_first = std::atoi(range.c_str());
            draw_last = colon == std::string::npos ? draw_first : std::atoi(range.c_str() + colon + 1);
        }
        else if (std::string(argv[i]) == "--dump-resource" && i + 2 < argc) {
            dump_spec = argv[++i]; dump_path = argv[++i];
        }
        else if (std::string(argv[i]) == "--dump-shader" && i + 2 < argc) {
            shader_spec = argv[++i]; shader_path = argv[++i];
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
    if (!dump_spec.empty()) {
        size_t c1 = dump_spec.find(':'), c2 = c1 == std::string::npos ? c1 : dump_spec.find(':', c1 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos) { usage(argv[0]); return 2; }
        int di = std::atoi(dump_spec.c_str()); std::string stage = dump_spec.substr(c1 + 1, c2 - c1 - 1);
        int binding = std::atoi(dump_spec.c_str() + c2 + 1);
        if (di < 0 || static_cast<size_t>(di) >= replay.items.size() || (stage != "vs" && stage != "ps")) {
            std::fprintf(stderr, "gpu_replay: invalid resource selector %s\n", dump_spec.c_str()); return 2;
        }
        const auto* table = stage == "vs" ? replay.items[di].vrt.get() : replay.items[di].prt.get();
        const prosper::gpu::ShaderResource* found = nullptr;
        if (table) for (const auto& resource : table->resources)
            if (static_cast<int>(resource.binding) == binding) { found = &resource; break; }
        if (!found || !found->host_data) {
            std::fprintf(stderr, "gpu_replay: resource %s not found or has no captured bytes\n", dump_spec.c_str()); return 2;
        }
        FILE* f = std::fopen(dump_path.c_str(), "wb");
        if (!f || std::fwrite(found->host_data, 1, static_cast<size_t>(found->host_data_size), f) != found->host_data_size) {
            if (f) std::fclose(f); std::fprintf(stderr, "gpu_replay: cannot write %s\n", dump_path.c_str()); return 2;
        }
        std::fclose(f);
        std::fprintf(stderr, "[gpureplay] dumped %s (%llu bytes) to %s\n", dump_spec.c_str(),
                     static_cast<unsigned long long>(found->host_data_size), dump_path.c_str());
    }
    if (!shader_spec.empty()) {
        size_t colon = shader_spec.find(':');
        int di = std::atoi(shader_spec.c_str()); std::string stage =
            colon == std::string::npos ? std::string{} : shader_spec.substr(colon + 1);
        if (colon == std::string::npos || di < 0 || static_cast<size_t>(di) >= replay.items.size() ||
            (stage != "vs" && stage != "fs")) {
            std::fprintf(stderr, "gpu_replay: invalid shader selector %s\n", shader_spec.c_str()); return 2;
        }
        const auto& words = stage == "vs" ? replay.items[di].vs : replay.items[di].fs;
        FILE* f = std::fopen(shader_path.c_str(), "wb"); size_t bytes = words.size() * sizeof(uint32_t);
        if (!f || std::fwrite(words.data(), 1, bytes, f) != bytes) {
            if (f) std::fclose(f); std::fprintf(stderr, "gpu_replay: cannot write %s\n", shader_path.c_str()); return 2;
        }
        std::fclose(f);
        std::fprintf(stderr, "[gpureplay] dumped %s (%zu bytes) to %s\n", shader_spec.c_str(), bytes,
                     shader_path.c_str());
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
    if (validate_only) return validate_frame(replay) ? 0 : 1;
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
