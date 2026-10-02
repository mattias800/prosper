// #4056: producing launch authority, cache identity and capture/replay round trip. CPU only.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/capture/gpu_capture_bundle.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <utility>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
namespace {
int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++failures; } } while (0)
alignas(256) constexpr uint32_t vertex[] = {
    0x36020081u,0x2c040081u,0x7e020d01u,0x7e040d02u,0x7e0a02f6u,0x7e0c02f2u,
    0x10020b01u,0x08020d01u,0x10040b02u,0x08040d02u,0x7e060280u,0x7e0802f2u,
    0xf80008cfu,0x04030201u,0xbf810000u,
};
// Smallest positive subnormal versus zero. MRT0.x consumes the actual VCC, not a marker.
alignas(256) constexpr uint32_t fragment[] = {
    0x7e0002ffu,1u,0x7c040080u,
    0xd5010001u,128u | (242u << 9) | (106u << 18),
    0x7e040280u,0x7e0602f2u,0xf800180fu,0x03020201u,0xbf810000u,
};
alignas(256) constexpr uint32_t failed_fragment[] = {
    0xbfA40001u, // S_ROUND_MODE: ordered MODE writes remain unsupported, not ignored
    0x7e000280u,0xf800180fu,0u,0xbf810000u,
};
alignas(256) constexpr uint32_t utility[] = {0x7e000280u,0xf8001803u,0u,0xbf810000u};
void program(GpuState& state, uint32_t lo, uint32_t hi, const uint32_t* words) {
    const auto address = reinterpret_cast<uint64_t>(words);
    state.sh[lo] = static_cast<uint32_t>(address >> 8);
    state.sh[hi] = static_cast<uint32_t>((address >> 40) & 255u);
}
GpuState state_for(FragmentFloatMode mode, const uint32_t* fs = fragment, bool wave32 = false) {
    GpuState state;
    program(state,P::SPI_SHADER_PGM_LO_ES,P::SPI_SHADER_PGM_HI_ES,vertex);
    program(state,P::SPI_SHADER_PGM_LO_PS,P::SPI_SHADER_PGM_HI_PS,fs);
    state.uc[P::VGT_PRIMITIVE_TYPE] = 4;
    state.cx[P::CB_TARGET_MASK] = 15;
    state.cx[P::CB_COLOR_CONTROL] = P::CB_COLOR_CONTROL_MODE_NORMAL << P::CB_COLOR_CONTROL_MODE_SHIFT;
    state.cx[P::SPI_PS_IN_CONTROL] = wave32 ? 1u << P::SPI_PS_IN_CONTROL_PS_W32_EN_SHIFT : 0u;
    if (mode.available)
        state.sh[P::SPI_SHADER_PGM_RSRC1_PS] = uint32_t(mode.value) << P::SPI_SHADER_PGM_RSRC1_PS_FLOAT_MODE_SHIFT;
    return state;
}
size_t reader(uint64_t address, uint8_t* destination, size_t count) {
    for (const auto span : {std::pair{vertex,sizeof(vertex)},std::pair{fragment,sizeof(fragment)},
                            std::pair{failed_fragment,sizeof(failed_fragment)},std::pair{utility,sizeof(utility)}}) {
        const uint64_t base = reinterpret_cast<uint64_t>(span.first);
        if (address < base || address - base >= span.second) continue;
        const size_t offset = static_cast<size_t>(address - base), n = std::min(count,span.second-offset);
        std::memcpy(destination,reinterpret_cast<const uint8_t*>(span.first)+offset,n);
        return n;
    }
    return 0;
}
void set32(std::vector<uint8_t>& bytes, size_t at, uint32_t value) {
    for (unsigned i=0;i<4;++i) bytes[at+i] = static_cast<uint8_t>(value>>(i*8));
}
std::vector<uint32_t> direct(FragmentFloatMode mode, bool wave32=false) {
    return recompile_fragment(fragment,std::size(fragment),nullptr,nullptr,UINT32_MAX,nullptr,
                              wave32,{RecompileDiagnosticStage::Fragment,0},mode);
}
}
int main(int argc, char** argv) {
    std::printf("== fragment_float_mode_capture ==\n");
    clear_shader_recompile_cache();
    const auto flush = direct({true,0}), preserve = direct({true,16}), unknown = direct({});
    CHECK(!flush.empty() && !preserve.empty() && !unknown.empty() && flush != preserve && unknown != flush,
          "mode discriminates real integer comparison modules and legacy FP fallback");
    std::vector<DrawItem> realized;
    for (bool shared : {false,true}) for (bool wave32 : {false,true}) {
        for (const auto mode : {FragmentFloatMode{true,0},FragmentFloatMode{true,16},
                                FragmentFloatMode{true,0},FragmentFloatMode{true,16},FragmentFloatMode{}}) {
            DrawItem draw;
            CHECK(realize_draw_item(state_for(mode,fragment,wave32),nullptr,3,std::size(vertex),false,
                                    draw,nullptr,shared), "actual successful draw realizes");
            CHECK(draw.ps_float_mode == mode && draw.fs_words() == direct(mode,wave32),
                  "copied/shared cold/warm draw owns exact compiler mode and matching bytes");
            draw.draw_index = realized.size(); draw.command_order = realized.size()+1;
            realized.push_back(std::move(draw));
        }
    }
    std::set<uint64_t> identities;
    for (unsigned value=0; value<256; ++value) {
        uint64_t id = 0, warm_id = 0;
        const FragmentFloatMode mode{true,static_cast<uint8_t>(value)};
        const auto source = recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
            fragment,std::size(fragment),nullptr,nullptr,nullptr,&id,false,0,false,{},mode);
        const auto warm = recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
            fragment,std::size(fragment),nullptr,nullptr,nullptr,&warm_id,false,0,false,{},mode);
        CHECK(source && warm && source == warm && id && id == warm_id && *source == direct(mode),
              "all eight mode bits retained; identical keys warm-reuse exact source ownership");
        identities.insert(id);
    }
    CHECK(identities.size()==256, "full launch mode, not only currently consumed denorm bit, partitions cache");
    uint64_t invalid_id = 99;
    CHECK(!recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,fragment,std::size(fragment),
            nullptr,nullptr,nullptr,&invalid_id,false,0,false,{}, {false,16}) && invalid_id==0,
          "noncanonical unavailable mode refuses before cache identity publication");
    GpuState mixed = state_for({true,255});
    for (uint8_t mode : {uint8_t{0},uint8_t{16}}) {
        GpuState::Draw draw; draw.index_count=3; draw.command_order=mixed.draws.size()+1;
        draw.state=std::make_shared<const GpuState>(state_for({true,mode}));
        mixed.draws.push_back(std::move(draw));
    }
    const auto per_draw = realize_gpustate_draws(mixed,std::size(vertex),1,1,nullptr,true,false);
    CHECK(per_draw.size()==2 && per_draw[0].ps_float_mode==FragmentFloatMode({true,0}) &&
              per_draw[1].ps_float_mode==FragmentFloatMode({true,16}) &&
              per_draw[0].fs_words()==flush && per_draw[1].fs_words()==preserve,
          "per-draw snapshots beat conflicting submit-end mode and warm cache");
    auto util = state_for({true,0xab},utility,true);
    util.cx[P::CB_COLOR_CONTROL]=P::CB_COLOR_CONTROL_MODE_DCC_DECOMPRESS << P::CB_COLOR_CONTROL_MODE_SHIFT;
    DrawItem replaced;
    CHECK(realize_draw_item(util,nullptr,3,std::size(vertex),false,replaced) &&
              replaced.ps_float_mode==FragmentFloatMode({true,0xab}) &&
              fragment_spirv_required_subgroup_size(replaced.fs_words())==0,
          "utility replacement retains launch mode rather than inferring authority from effective SPIR-V");
    GpuCaptureMetadata metadata; metadata.width=metadata.height=1;
    GpuCaptureFile capture, loaded;
    std::string error;
    std::vector<uint8_t> bytes;
    CHECK(capture_draw_items(realized,metadata,reader,capture,error) &&
              serialize_gpu_capture(capture,bytes,error) && deserialize_gpu_capture(bytes,loaded,error) &&
              loaded.format_version==66 && loaded.draws.size()==realized.size(),
          "actual collector and current production codecs round trip realized mode");
    GpuReplayFrame replay;
    CHECK(materialize_gpu_replay(loaded,replay,error) && replay.items.size()==realized.size(),
          "production replay materializes each immutable mode");
    for (size_t i=0; i<realized.size() && i<replay.items.size(); ++i)
        CHECK(loaded.draws[i].ps_float_mode==realized[i].ps_float_mode &&
                  replay.items[i].ps_float_mode==realized[i].ps_float_mode &&
                  replay.items[i].fs_words()==realized[i].fs_words(),
              "mode availability/value and stored source bytes survive collector/codec/replay");
    GpuCaptureBundle bundle;
    GpuCaptureFile full, manifest;
    CHECK(append_gpu_capture_bundle(bundle,capture,error) &&
              materialize_gpu_capture_bundle_submit(bundle,0,full,error) &&
              materialize_gpu_capture_bundle_manifest(bundle,0,manifest,error),
          "bundle payload and metadata-only manifest materialize mode tail");
    for (const auto* result : {&full,&manifest}) for (size_t i=0;i<realized.size();++i)
        CHECK(i<result->draws.size() && result->draws[i].ps_float_mode==realized[i].ps_float_mode,
              "bundle paths preserve unknown, explicitly programmed zero and full-byte values");
    const bool no_resources = capture.computes.empty() && capture.failure_diagnostics.empty() &&
        std::all_of(capture.draws.begin(),capture.draws.end(),[](const auto& draw) {
            return draw.vrt.resources.empty() && draw.prt.resources.empty();
        });
    CHECK(no_resources && bytes.size() >= 12u+capture.draws.size()+4u+8u+2u*capture.draws.size(),
          "mode controls require the resource-free combined fixture");
    if (!no_resources || bytes.size() < 12u+capture.draws.size()+4u+8u+2u*capture.draws.size()) return 1;
    auto mode_bytes = bytes;
    mode_bytes.resize(mode_bytes.size()-12u-capture.draws.size()-4u); set32(mode_bytes,8,64);
    GpuCaptureFile official64;
    CHECK(deserialize_gpu_capture(mode_bytes,official64,error) && official64.format_version==64 &&
              official64.draws.size()==capture.draws.size(),
          "genuine official v64 prefix retains complete producing-mode records");
    for (size_t i=0;i<capture.draws.size() && i<official64.draws.size();++i)
        CHECK(official64.draws[i].ps_float_mode==capture.draws[i].ps_float_mode,
              "official v64 preserves known zero, all mode bits and unknown availability");
    const size_t tail = mode_bytes.size() - 8u - 2u*capture.draws.size();
    auto legacy_bytes = mode_bytes; legacy_bytes.resize(tail); set32(legacy_bytes,8,63);
    GpuCaptureFile legacy;
    CHECK(deserialize_gpu_capture(legacy_bytes,legacy,error) && legacy.format_version==63,
          "genuine append-only v63 prefix loads without FLOAT_MODE");
    for (const auto& draw : legacy.draws)
        CHECK(!draw.ps_float_mode.available && draw.ps_float_mode.value==0,
              "legacy unknown is not fabricated mode0 despite source markers and guest width");
    std::vector<uint8_t> upgraded_bytes;
    CHECK(serialize_gpu_capture(legacy,upgraded_bytes,error) && deserialize_gpu_capture(upgraded_bytes,legacy,error),
          "legacy rewrite succeeds without inventing mode authority");
    for (const auto& draw : legacy.draws) CHECK(!draw.ps_float_mode.available,"rewrite preserves unknown mode");
    for (uint32_t count : {0u,1u,UINT32_MAX}) {
        auto corrupt=mode_bytes; set32(corrupt,tail,count);
        CHECK(!deserialize_gpu_capture(corrupt,loaded,error) && error=="invalid realized-draw fragment float mode count",
              "hostile mode count rejected before allocating additional records");
    }
    for (uint8_t tag : {uint8_t{2},uint8_t{255}}) {
        auto corrupt=mode_bytes; corrupt[tail+4]=tag;
        CHECK(!deserialize_gpu_capture(corrupt,loaded,error) && error=="invalid realized-draw fragment float mode",
              "reserved mode availability tags reject");
    }
    auto corrupt=mode_bytes; corrupt[tail+4]=0; corrupt[tail+5]=1;
    CHECK(!deserialize_gpu_capture(corrupt,loaded,error), "unknown nonzero mode fails closed");
    corrupt=bytes; set32(corrupt,mode_bytes.size()-4,UINT32_MAX);
    CHECK(!deserialize_gpu_capture(corrupt,loaded,error) && error=="invalid failed-draw fragment float mode count",
          "failure mode count must match already bounded diagnostics");
    for (size_t end : {tail,tail+3,tail+4,tail+5,mode_bytes.size()-1}) {
        corrupt=bytes; corrupt.resize(end);
        CHECK(!deserialize_gpu_capture(corrupt,loaded,error), "truncated v64 tail rejects");
    }
    corrupt=bytes; corrupt.push_back(0);
    CHECK(!deserialize_gpu_capture(corrupt,loaded,error) && error=="capture has trailing data",
          "trailing bytes after complete canonical tail reject");
    auto bad=capture; bad.draws[0].ps_float_mode={false,16};
    CHECK(!serialize_gpu_capture(bad,upgraded_bytes,error) && !materialize_gpu_replay(bad,replay,error),
          "writer/direct materializer reject noncanonical untrusted mode");
    auto bad_draws=realized; bad_draws[0].ps_float_mode={false,16};
    CHECK(!capture_draw_items(bad_draws,metadata,reader,bad,error),"collector rejects noncanonical mode");
    OperationRealizationFailure failure;
    DrawItem rejected;
    CHECK(!realize_draw_item(state_for({true,0x9b},failed_fragment),nullptr,3,std::size(vertex),false,
                             rejected,&failure) && failure.reason==RealizationFailureReason::ShaderRecompile &&
              failure.ps_float_mode==FragmentFloatMode({true,0x9b}),
          "actual failed-stage compiler retains known launch mode");
    failure.index=7; failure.command_order=1;
    GpuCaptureFile failed;
    CHECK(capture_submit_items({}, {},{{SubmitOperationKind::Draw,7,1}},metadata,reader,failed,error,{}, {failure}) &&
              serialize_gpu_capture(failed,upgraded_bytes,error) && deserialize_gpu_capture(upgraded_bytes,loaded,error) &&
              loaded.failure_diagnostics.size()==1 && loaded.failure_diagnostics[0].ps_float_mode==failure.ps_float_mode,
          "actual failure collector/codec retain mode independently of successful draws");
    if (argc==3 && std::string(argv[1])=="--write-fixture") {
        const std::filesystem::path directory(argv[2]);
        for (const auto entry : {std::pair{"mode0",FragmentFloatMode{true,0}},
                                 std::pair{"mode16",FragmentFloatMode{true,16}},
                                 std::pair{"unknown",FragmentFloatMode{}}}) {
            DrawItem draw;
            CHECK(realize_draw_item(state_for(entry.second),nullptr,3,std::size(vertex),false,draw),"CLI fixture realizes");
            draw.draw_index=7; draw.command_order=1;
            GpuCaptureFile single;
            CHECK(capture_draw_items({draw},metadata,reader,single,error),"CLI fixture captures exact producing input");
            // Wrong stored pixels must not pass for regenerated source. Oppose the known mode.
            single.draws[0].fs=entry.second.available && entry.second.value==0 ? preserve : flush;
            CHECK(write_gpu_capture((directory/(std::string(entry.first)+".prgcap")).string(),single,error),"CLI fixture writes");
            auto retry=single; retry.draws.clear(); retry.operations={{SubmitOperationKind::Draw,7,1,false}};
            retry.failure_diagnostics_available=true;
            GpuCapturedOperationFailure f; f.source_index=7; f.command_order=1;
            f.reason=RealizationFailureReason::ShaderRecompile; f.vertex_retry_config_available=true;
            f.fragment_retry_config_available=true; f.ps_float_mode=entry.second;
            GpuCapturedStageDiagnostic stage; stage.stage=ShaderProgramStage::Fragment;
            stage.program_addr=reinterpret_cast<uint64_t>(fragment); stage.raw_shader_index=single.draws[0].fs_raw_shader_index;
            f.stages.push_back(stage); retry.failure_diagnostics.push_back(f);
            // Remove the otherwise unreferenced vertex raw version and remap fragment index.
            const auto raw=retry.raw_shader_versions[stage.raw_shader_index]; retry.raw_shader_versions={raw};
            retry.failure_diagnostics[0].stages[0].raw_shader_index=0;
            CHECK(write_gpu_capture((directory/(std::string(entry.first)+"-failed.prgcap")).string(),retry,error),"retry CLI fixture writes");
        }
    }
    std::printf("== %s (%d failures) ==\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
