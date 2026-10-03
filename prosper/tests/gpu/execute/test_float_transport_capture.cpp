// #4066: actual producing profile, immutable cache ownership and append-only codecs. CPU only.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/capture/gpu_capture_bundle.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;
namespace {
int failures = 0, checks = 0;
#define CHECK(c, m) do { ++checks; if (!(c)) { std::printf("[FAIL] %s\n", m); ++failures; } } while (0)
constexpr FloatTransportConfig unknown{}, implicit{FloatTransportProfile::Implicit},
    explicit32{FloatTransportProfile::ExplicitNonFinite32};
alignas(256) constexpr uint32_t vertex[] = {
    0x36020081u,0x2c040081u,0x7e020d01u,0x7e040d02u,0x7e0a02f6u,0x7e0c02f2u,
    0x10020b01u,0x08020d01u,0x10040b02u,0x08040d02u,0x7e060280u,0x7e0802f2u,
    0xf80008cfu,0x04030201u,0xbf810000u,
};
alignas(256) constexpr uint32_t fragment[] = {
    0x7e0002ffu,0x7f800000u,0x7e0202ffu,0x7fc12345u,
    0x7e040280u,0x7e0602f2u,0xf800180fu,0x03020100u,0xbf810000u,
};
alignas(256) constexpr uint32_t bad_fragment[] = {0xbfA40001u,0xbf810000u};
alignas(256) constexpr uint32_t dcc_fragment[] = {
    0x7e000280u,0xf8001803u,0x00000000u,0xbf810000u,
};
void set32(std::vector<uint8_t>& bytes, size_t at, uint32_t value) {
    for (size_t i=0; i<4; ++i) bytes[at+i] = static_cast<uint8_t>(value >> (i*8));
}
void append32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (size_t i=0; i<4; ++i) bytes.push_back(static_cast<uint8_t>(value >> (i*8)));
}
GpuState state(const uint32_t* fs=fragment) {
    GpuState result;
    const auto program = [&](uint32_t lo, uint32_t hi, const uint32_t* words) {
        const auto address=reinterpret_cast<uint64_t>(words);
        result.sh[lo]=static_cast<uint32_t>(address >> 8);
        result.sh[hi]=static_cast<uint32_t>((address >> 40) & 255u);
    };
    program(P::SPI_SHADER_PGM_LO_ES,P::SPI_SHADER_PGM_HI_ES,vertex);
    program(P::SPI_SHADER_PGM_LO_PS,P::SPI_SHADER_PGM_HI_PS,fs);
    result.uc[P::VGT_PRIMITIVE_TYPE]=4;
    result.cx[P::CB_TARGET_MASK]=15;
    result.cx[P::CB_COLOR_CONTROL]=P::CB_COLOR_CONTROL_MODE_NORMAL << P::CB_COLOR_CONTROL_MODE_SHIFT;
    result.sh[P::SPI_SHADER_PGM_RSRC1_PS]=16u << P::SPI_SHADER_PGM_RSRC1_PS_FLOAT_MODE_SHIFT;
    return result;
}
size_t reader(uint64_t address, uint8_t* destination, size_t count) {
    for (const auto entry : {std::pair{vertex,sizeof(vertex)}, std::pair{fragment,sizeof(fragment)},
                             std::pair{bad_fragment,sizeof(bad_fragment)},
                             std::pair{dcc_fragment,sizeof(dcc_fragment)}}) {
        const auto base=reinterpret_cast<uint64_t>(entry.first);
        if (address<base || address>=base+entry.second) continue;
        count=std::min(count,entry.second-static_cast<size_t>(address-base));
        std::memcpy(destination,reinterpret_cast<const void*>(address),count); return count;
    }
    return 0;
}
bool controls2(const std::vector<uint32_t>& words) {
    for (size_t at=5; at<words.size();) {
        const size_t count=words[at] >> 16;
        if (!count || count>words.size()-at) return false;
        if ((words[at]&65535u)==17u && count==2 && words[at+1]==6029u) return true;
        at+=count;
    }
    return false;
}
}
int main(int argc, char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    for (bool api : {false,true}) for (bool feature : {false,true})
        for (bool preserve : {false,true}) for (bool request : {false,true})
            CHECK(select_float_transport_request(api,feature,preserve,request)==
                      (api&&feature&&preserve&&request ? explicit32 : implicit),
                  "all four request prerequisites independent; advertised feature alone is insufficient");
    reset_float_transport_config_for_test();
    CHECK(published_float_transport_config()==unknown,"offline state is unknown, not capability-off");
    publish_float_transport_config(explicit32);
    const auto retained=published_float_transport_config();
    publish_float_transport_config(unknown);
    CHECK(retained==explicit32 && published_float_transport_config()==explicit32,
          "unknown publisher cannot fabricate an off decision");
    publish_float_transport_config(implicit); publish_float_transport_config(explicit32);
    CHECK(retained==explicit32 && published_float_transport_config()==implicit,
          "mixed device owners conservatively AND known enabled profiles without changing retained snapshots");
    std::vector<DrawItem> draws;
    std::vector<OperationRealizationFailure> failures_to_capture;
    std::vector<SubmitOperation> operations;
    clear_shader_recompile_cache();
    uint64_t rejected_identity=0x4066;
    const FloatTransportConfig invalid{static_cast<FloatTransportProfile>(3)};
    CHECK(!recompile_vertex_chain_cached_shared(nullptr,0,nullptr,0,nullptr,nullptr,
              &rejected_identity,0,false,invalid) && rejected_identity==0,
          "invalid chain profile clears stale caller identity before rejecting");
    rejected_identity=0x4066;
    CHECK(!recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
              fragment,std::size(fragment),nullptr,nullptr,nullptr,&rejected_identity,
              false,0,false,{}, {true,16},invalid) && rejected_identity==0,
          "invalid ordinary profile clears stale caller identity before rejecting");
    std::set<uint64_t> identities;
    std::vector<std::shared_ptr<const std::vector<uint32_t>>> modules;
    for (const auto config : {unknown,implicit,explicit32}) {
        uint64_t identity=0,warm_identity=0;
        const auto source=recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
            fragment,std::size(fragment),nullptr,nullptr,nullptr,&identity,false,0,false,{}, {true,16},config);
        const auto warm=recompile_graphics_shader_cached_shared(ShaderProgramStage::Fragment,
            fragment,std::size(fragment),nullptr,nullptr,nullptr,&warm_identity,false,0,false,{}, {true,16},config);
        CHECK(source && !source->empty() && source==warm && identity && identity==warm_identity,
              "same pinned producing profile reuses exact immutable source ownership");
        if (source) CHECK(controls2(*source)==config.explicit_nonfinite32(),
                          "actual module words distinguish explicit transport from unknown/implicit");
        identities.insert(identity); modules.push_back(source);
        reset_float_transport_config_for_test(); publish_float_transport_config(config);
        for (bool shared : {false,true}) {
            DrawItem item;
            CHECK(realize_draw_item(state(),nullptr,3,std::size(vertex),false,item,nullptr,shared) &&
                      item.float_transport==config && controls2(item.fs_words())==config.explicit_nonfinite32(),
                  "actual copied/shared realization retains producing snapshot and exact selected module");
            item.draw_index=draws.size()+1; item.command_order=item.draw_index;
            operations.push_back({SubmitOperationKind::Draw,item.draw_index,item.command_order});
            draws.push_back(std::move(item));
        }
        DrawItem rejected; OperationRealizationFailure failure;
        CHECK(!realize_draw_item(state(bad_fragment),nullptr,3,std::size(vertex),false,rejected,&failure) &&
                  failure.float_transport==config && failure.reason==RealizationFailureReason::ShaderRecompile,
              "actual failed compiler invocation retains producing profile independently of success");
        failure.index=100+failures_to_capture.size(); failure.command_order=failure.index;
        operations.push_back({SubmitOperationKind::Draw,failure.index,failure.command_order});
        failures_to_capture.push_back(std::move(failure));
    }
    CHECK(identities.size()==3,"all profile states partition cache identity even when legacy words coincide");
    CHECK(modules.size()==3 && modules[0] && modules[1] && modules[2] &&
              *modules[0]==*modules[1] && *modules[1]!=*modules[2],
          "unknown and implicit can share bytes but cannot alias producing provenance; explicit changes bytes");
    // This helper bypasses the ordinary fragment cache. Its first call must not pin later
    // decompression draws to the wrong feature envelope, in either initialization order.
    const bool explicit_first=argc==2 && std::strcmp(argv[1],"--dcc-explicit-first")==0;
    const std::array<FloatTransportConfig,5> dcc_profiles = explicit_first
        ? std::array{explicit32,unknown,implicit,explicit32,unknown}
        : std::array{unknown,explicit32,implicit,explicit32,unknown};
    std::array<std::vector<uint32_t>,3> dcc_sources;
    for (const auto config:dcc_profiles) {
        reset_float_transport_config_for_test(); publish_float_transport_config(config);
        auto dcc_state=state(dcc_fragment);
        dcc_state.cx[P::CB_COLOR_CONTROL]=
            P::CB_COLOR_CONTROL_MODE_DCC_DECOMPRESS << P::CB_COLOR_CONTROL_MODE_SHIFT;
        for (bool shared:{false,true}) {
            DrawItem item;
            CHECK(realize_draw_item(dcc_state,nullptr,3,std::size(vertex),false,item,nullptr,shared) &&
                      item.float_transport==config && !item.fs_words().empty() && item.fs_identity==0,
                  "actual DCC null-export realization retains the current producing profile");
            CHECK(controls2(item.fs_words())==config.explicit_nonfinite32(),
                  "DCC null-export source envelope follows each profile, not the first caller");
            auto& retained_source=dcc_sources[static_cast<size_t>(config.profile)];
            CHECK(retained_source.empty() || retained_source==item.fs_words(),
                  "DCC per-profile source stays immutable across warm transitions and copied/shared routes");
            retained_source=item.fs_words();
            DrawItem ordinary;
            CHECK(realize_draw_item(state(dcc_fragment),nullptr,3,std::size(vertex),false,
                      ordinary,nullptr,shared) && ordinary.fs_identity!=0 &&
                      controls2(ordinary.fs_words())==config.explicit_nonfinite32(),
                  "same helper program without DCC mode keeps its ordinary keyed fragment");
        }
    }
    CHECK(dcc_sources[0]==dcc_sources[1] && dcc_sources[0]!=dcc_sources[2],
          "DCC unknown/implicit legacy bytes match while explicit envelope is distinct");
    for (const auto device:{unknown,implicit,explicit32}) for (size_t i=0;i<modules.size();++i)
        if (modules[i]) CHECK(float_transport_module_supported(modules[i]->data(),modules[i]->size(),device)==
                                 (i!=2 || device.explicit_nonfinite32()),
                             "actual stored words, not captured profile, require executing device enablement");
    std::vector<uint32_t> malformed{0x07230203u,0x00010300u,0,2,0,0};
    CHECK(!float_transport_module_supported(malformed.data(),malformed.size(),explicit32),
          "zero-length instruction is not a capability witness");
    malformed.back()=(2u<<16)|17u;
    CHECK(!float_transport_module_supported(malformed.data(),malformed.size(),explicit32),
          "truncated capability instruction rejects at executing-device gate");
    // Change publication after every producer exists. Capture MUST read retained values, not current host state.
    reset_float_transport_config_for_test(); publish_float_transport_config(implicit);
    GpuCaptureMetadata metadata; metadata.width=metadata.height=1;
    GpuCaptureFile capture,loaded; std::string error;
    CHECK(capture_submit_items(draws,{},operations,metadata,reader,capture,error,{},failures_to_capture),
          "actual deferred collector captures retained draw and failure profiles");
    // Exercise independent compute and failed-compute config fields in the external codec. These are
    // synthetic metadata controls, NOT a claim that a fixture acquired an enabled Vulkan device.
    for (const auto config : {unknown,implicit,explicit32}) {
        GpuCapturedCompute compute; compute.recompile_config.float_transport=config;
        compute.dispatch_index=200+capture.computes.size(); compute.command_order=compute.dispatch_index;
        capture.operations.push_back({SubmitOperationKind::Dispatch,compute.dispatch_index,compute.command_order,true});
        capture.computes.push_back(compute);
        GpuCapturedOperationFailure failure; failure.kind=SubmitOperationKind::Dispatch;
        failure.reason=RealizationFailureReason::ShaderRecompile;
        failure.source_index=300+capture.computes.size(); failure.command_order=failure.source_index;
        capture.operations.push_back({SubmitOperationKind::Dispatch,failure.source_index,failure.command_order,false});
        GpuCapturedStageDiagnostic stage; stage.stage=ShaderProgramStage::Compute;
        stage.recompile_config.float_transport=config; failure.stages.push_back(stage);
        capture.failure_diagnostics.push_back(failure);
    }
    std::vector<uint8_t> bytes;
    const bool roundtrip=serialize_gpu_capture(capture,bytes,error) && deserialize_gpu_capture(bytes,loaded,error);
    CHECK(roundtrip && loaded.format_version == 70,
          "current codec accepts exact bounded producing profile tails");
    if (!roundtrip) { std::printf("codec error: %s\n",error.c_str()); return 1; }
    GpuReplayFrame replay;
    CHECK(materialize_gpu_replay(loaded,replay,error) && replay.items.size()==draws.size(),
          "materializer owns exact profiles independent of current device publication");
    for (size_t i=0;i<draws.size() && i<replay.items.size();++i)
        CHECK(replay.items[i].float_transport==draws[i].float_transport &&
                  replay.items[i].fs_words()==draws[i].fs_words(),"source words and producer profile round trip together");
    for (size_t i=0;i<3 && i<loaded.computes.size();++i)
        CHECK(loaded.failure_diagnostics.size()>3+i && !loaded.failure_diagnostics[3+i].stages.empty() &&
                  loaded.computes[i].recompile_config.float_transport.profile==static_cast<FloatTransportProfile>(i) &&
                  loaded.failure_diagnostics[3+i].stages[0].recompile_config.float_transport.profile==
                      static_cast<FloatTransportProfile>(i),"compute/stage profile is independent of failure parent");
    GpuCaptureBundle bundle; GpuCaptureFile full,manifest;
    CHECK(append_gpu_capture_bundle(bundle,capture,error) &&
              materialize_gpu_capture_bundle_submit(bundle,0,full,error) &&
              materialize_gpu_capture_bundle_manifest(bundle,0,manifest,error),"both bundle routes retain v66 profile tail");
    for (const auto* result : {&full,&manifest}) for (size_t i=0;i<draws.size();++i)
        CHECK(result->draws[i].float_transport==draws[i].float_transport,"full/manifest do not replace producer with current profile");
    std::vector<uint8_t> tail;
    append32(tail,static_cast<uint32_t>(capture.draws.size()));
    for (const auto& item:capture.draws) tail.push_back(static_cast<uint8_t>(item.float_transport.profile));
    append32(tail,static_cast<uint32_t>(capture.computes.size()));
    for (const auto& item:capture.computes) tail.push_back(static_cast<uint8_t>(item.recompile_config.float_transport.profile));
    append32(tail,static_cast<uint32_t>(capture.failure_diagnostics.size()));
    for (const auto& item:capture.failure_diagnostics) {
        tail.push_back(static_cast<uint8_t>(item.float_transport.profile));
        append32(tail,static_cast<uint32_t>(item.stages.size()));
        for (const auto& stage:item.stages) tail.push_back(static_cast<uint8_t>(stage.recompile_config.float_transport.profile));
    }
    // These ordinary draws have no wave plan: v70 is exactly count plus zero flags.
    const size_t wave_tail = 4u + capture.draws.size();
    auto official69 = bytes;
    official69.resize(official69.size() - wave_tail);
    set32(official69, 8, 69);
    CHECK(deserialize_gpu_capture(official69, loaded, error) && loaded.format_version == 69,
          "genuine official v69 retains entry facts without owned-wave authority");
    auto official68 = official69;
    official68.resize(official68.size() - 4u - kGpuCaptureFragmentEntryRecordBytes*capture.draws.size());
    set32(official68,8,68);
    CHECK(deserialize_gpu_capture(official68,loaded,error) && loaded.format_version==68 &&
          std::all_of(loaded.draws.begin(),loaded.draws.end(),[](const auto& d) { return !d.ps_entry.observed; }),
          "genuine official v68 lacks new entry facts without inferring them from producer profiles");
    CHECK(official68.size() > 4u && std::all_of(official68.end()-4, official68.end(), [](uint8_t b) { return b == 0; }),
          "resource-free v68 fixture has an exact zero nested count");
    auto official67 = official68; official67.resize(official67.size()-4u); set32(official67,8,67);
    CHECK(deserialize_gpu_capture(official67,loaded,error) && loaded.format_version==67,
          "genuine official v67 retains producing launch evidence without nested ownership");
    auto v66_bytes = official67;
    v66_bytes.resize(v66_bytes.size() - 8u - 8u*(capture.draws.size()+capture.failure_diagnostics.size()));
    set32(v66_bytes,8,66);
    CHECK(deserialize_gpu_capture(v66_bytes,loaded,error) && loaded.format_version==66,
          "genuine v66 profile prefix survives without fabricated launch flags");
    CHECK(v66_bytes.size()>tail.size() && std::equal(tail.rbegin(),tail.rend(),v66_bytes.rbegin()),
          "canonical tail is exact, not a version-only relabel");
    if (bytes.size()<=tail.size()) return 1;
    const size_t start=v66_bytes.size()-tail.size();
    auto legacy=v66_bytes; legacy.resize(start); set32(legacy,8,65);
    CHECK(deserialize_gpu_capture(legacy,loaded,error) && loaded.format_version==65,
          "genuine v65 prefix retains owned markers without fabricated profile authority");
    for (const auto& item:loaded.draws) CHECK(item.float_transport==unknown,"old explicit source markers do not infer producing profile");
    for (const auto& item:loaded.computes) CHECK(item.recompile_config.float_transport==unknown,"old compute config remains unknown");
    for (const auto& item:loaded.failure_diagnostics) {
        CHECK(item.float_transport==unknown,"old failed parent remains unknown");
        for (const auto& stage:item.stages) CHECK(stage.recompile_config.float_transport==unknown,"old failed stage remains unknown");
    }
    const size_t compute_count=start+4+capture.draws.size();
    const size_t failure_count=compute_count+4+capture.computes.size();
    for (const auto offset:{start,compute_count,failure_count}) {
        auto corrupt=bytes; set32(corrupt,offset,UINT32_MAX);
        CHECK(!deserialize_gpu_capture(corrupt,loaded,error),"hostile counts reject without second allocations");
    }
    for (uint8_t tag:{uint8_t{3},uint8_t{255}}) for (const auto offset:{start+4,compute_count+4,failure_count+4}) {
        auto corrupt=bytes; corrupt[offset]=tag;
        CHECK(!deserialize_gpu_capture(corrupt,loaded,error),"reserved profile tags reject rather than becoming true");
    }
    auto corrupt=bytes; set32(corrupt,failure_count+5,UINT32_MAX);
    CHECK(!deserialize_gpu_capture(corrupt,loaded,error),"failed-stage count must match previously bounded inventory");
    for (size_t end=start;end<v66_bytes.size();++end) {
        corrupt=v66_bytes; corrupt.resize(end);
        CHECK(!deserialize_gpu_capture(corrupt,loaded,error),"every truncated v66 suffix refuses");
    }
    for (size_t end=v66_bytes.size();end<bytes.size();++end) {
        corrupt=bytes; corrupt.resize(end);
        CHECK(!deserialize_gpu_capture(corrupt, loaded, error),
              "every truncated v67 launch/v68 nested/v69 entry/v70 wave suffix refuses");
    }
    corrupt=bytes; corrupt.push_back(0);
    CHECK(!deserialize_gpu_capture(corrupt,loaded,error) && error=="capture has trailing data","strict EOF after new tail");
    corrupt=bytes; set32(corrupt,8,66);
    CHECK(!deserialize_gpu_capture(corrupt,loaded,error) && error=="capture has trailing data","version-only downgrade is not legacy");
    auto bad=capture; bad.draws[0].float_transport.profile=static_cast<FloatTransportProfile>(3);
    CHECK(!serialize_gpu_capture(bad,bytes,error) && !materialize_gpu_replay(bad,replay,error),
          "writer and direct materializer reject noncanonical profile");
    bad=capture; bad.computes[0].recompile_config.float_transport.profile=static_cast<FloatTransportProfile>(3);
    CHECK(!serialize_gpu_capture(bad,bytes,error) && !materialize_gpu_replay(bad,replay,error),
          "compute profile rejects even when older recompile-config availability is false");
    bad=capture; bad.failure_diagnostics[0].float_transport.profile=static_cast<FloatTransportProfile>(3);
    CHECK(!serialize_gpu_capture(bad,bytes,error) && !materialize_gpu_replay(bad,replay,error),
          "failed parent profile rejects noncanonical tags");
    bad=capture; bad.failure_diagnostics[0].stages[0].recompile_config.float_transport.profile=static_cast<FloatTransportProfile>(3);
    CHECK(!serialize_gpu_capture(bad,bytes,error) && !materialize_gpu_replay(bad,replay,error),
          "failed stage profile rejects independently of older compute-config availability");
    if (argc==3 && std::strcmp(argv[1],"--write-fixture")==0 && modules.size()==3) {
        const std::filesystem::path directory(argv[2]);
        CHECK(write_gpu_capture((directory/"inventory.prgcap").string(),capture,error),
              "inspect fixture retains draw, compute, failed parent and failed-compute profiles");
        // Already independently stripped and decoded above: keep the genuine owned v65 prefix.
        { std::ofstream output(directory/"legacy-inventory.prgcap",std::ios::binary);
          output.write(reinterpret_cast<const char*>(legacy.data()),legacy.size());
          CHECK(output.good(),"legacy inspect fixture has no producing transport tail"); }
        for (size_t i=0;i<3;++i) {
            auto item=draws[i*2]; item.draw_index=7; item.command_order=1;
            GpuCaptureFile single;
            CHECK(capture_draw_items({item},metadata,reader,single,error),"CLI fixture captures actual producer");
            if (single.draws.empty() || !modules[i]) return 1;
            // Opposing stored capability markers must not replace captured producer configuration.
            single.draws[0].fs=*modules[i==2 ? 1 : 2];
            const std::string name=float_transport_profile_name(item.float_transport);
            CHECK(write_gpu_capture((directory/(name+".prgcap")).string(),single,error),"CLI fixture writes current capture");
            auto retry=single; retry.draws.clear(); retry.operations={{SubmitOperationKind::Draw,7,1,false}};
            retry.failure_diagnostics_available=true;
            GpuCapturedOperationFailure failed; failed.source_index=7; failed.command_order=1;
            failed.reason=RealizationFailureReason::ShaderRecompile; failed.vertex_retry_config_available=true;
            failed.fragment_retry_config_available=true; failed.ps_float_mode=item.ps_float_mode;
            failed.float_transport=item.float_transport;
            failed.ps_float_flags=item.ps_float_flags;
            failed.ps_launch_rsrc1=item.ps_launch_rsrc1;
            GpuCapturedStageDiagnostic stage; stage.stage=ShaderProgramStage::Fragment;
            stage.program_addr=reinterpret_cast<uint64_t>(fragment);
            stage.raw_shader_index=single.draws[0].fs_raw_shader_index;
            failed.stages.push_back(stage); retry.failure_diagnostics.push_back(failed);
            const auto raw=retry.raw_shader_versions[stage.raw_shader_index]; retry.raw_shader_versions={raw};
            retry.failure_diagnostics[0].stages[0].raw_shader_index=0;
            CHECK(write_gpu_capture((directory/(name+"-failed.prgcap")).string(),retry,error),"synthetic retry metadata writes exact producer inputs");
        }
    }
    reset_float_transport_config_for_test();
    std::printf("float transport capture: %d checks, %d failures\n",checks,failures);
    return failures ? 1 : 0;
}
