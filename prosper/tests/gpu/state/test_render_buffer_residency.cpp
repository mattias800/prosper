// Immutable renderer uploads: real vertex fetches distinguish stale contents from valid reuse.
// No guest mapping/watch is simulated here: FrameResource's current materialized bytes are the
// authority. Distinct host vectors deliberately advertise one guest identity. A queued pass must
// retain its own upload even after that identity changes or cache pressure removes the entry.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/state/render_state.hpp"
#include "fixtures/render_runner.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test;
namespace {
int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++failures; } \
                        else std::printf("[ok] %s\n", m); } while (0)
constexpr uint32_t W = 32, H = 32;
constexpr size_t Words = 2048, Bytes = Words * sizeof(uint32_t);
constexpr std::array<uint32_t, 3> Quads{0, 508, 1020};
constexpr float Clear[4]{0, 0, 1, 1};
void disabled(bool value) {
#ifdef _WIN32
    _putenv_s("PROSPER_NO_BACKEND_BUFFER_RESIDENCY", value ? "1" : "");
#else
    if (value) setenv("PROSPER_NO_BACKEND_BUFFER_RESIDENCY", "1", 1);
    else unsetenv("PROSPER_NO_BACKEND_BUFFER_RESIDENCY");
#endif
}
void quad(std::vector<uint32_t>& words, uint32_t first, bool visible) {
    const float bias = visible ? 0.f : 10.f;
    const float positions[]{-1,-1, -1,1, 1,1, 1,-1};
    for (size_t i = 0; i < 8; ++i)
        words[first * 2 + i] = std::bit_cast<uint32_t>(positions[i] + bias);
}
std::vector<uint32_t> payload() {
    std::vector<uint32_t> result(Words, 0);
    for (auto first : Quads) quad(result, first, true);
    return result;
}
bool solid(const std::vector<uint8_t>& image, bool green) {
    if (image.size() != W * H * 4) return false;
    for (size_t i = 0; i < image.size(); i += 4)
        if (image[i] > 8 || image[i + (green ? 1 : 2)] < 247 ||
            image[i + (green ? 2 : 1)] > 8 || image[i + 3] < 247) return false;
    return true;
}
struct Fixture {
    ResolvedPipelineState state;
    std::vector<uint32_t> vs, fs;
    Fixture() {
        const uint32_t vertex[]{0x7e060280u,0x7e0802f2u,0xe0042000u,0x80020100u,
                                0xf80008cfu,0x04030201u,0xbf810000u};
        const uint32_t fragment[]{0x7e000280u,0x7e0202f2u,0x7e040280u,0x7e0602f2u,
                                  0xf800180fu,0x03020100u,0xbf810000u};
        ShaderResourceTable table;
        ShaderResource buffer{};
        buffer.cls = ResourceClass::VertexBuffer; buffer.format = DataFormat::Float32;
        buffer.num_components = 2; buffer.binding = 3; buffer.stride = 8; buffer.sgpr_base = 8;
        table.resources.push_back(buffer);
        vs = recompile_vertex(vertex, std::size(vertex), &table);
        fs = recompile_fragment(fragment, std::size(fragment));
        state.topology = 3;
    }
    BackendDraw draw(const std::vector<uint32_t>& source, uint64_t identity,
                     uint32_t first = 0) const {
        BackendDraw d; d.vs = vs; d.fs = fs; d.ps = &state;
        d.vcount = static_cast<uint32_t>(source.size() / 2);
        d.indices = {first,first+1,first+2, first+2,first+3,first};
        FrameResource r; r.binding = 3; r.set = 0; r.buffer_identity = identity;
        r.dwords_view = source.data(); r.dwords_view_count = source.size();
        d.R.push_back(std::move(r)); return d;
    }
    std::vector<uint8_t> render(const std::vector<uint32_t>& source, uint64_t identity,
                                uint32_t first = 0) const {
        return render_draws_rgba({draw(source, identity, first)}, W, H, nullptr, Clear);
    }
};
// Add a real atomic SSBO access to an otherwise green fragment shader. Atomic OR zero at a zero
// payload word is deterministic even when many fragments address it. The SSBO has the same exact
// source/identity as the VS buffer but another stage/binding: whole-pass writer exclusion matters.
std::vector<uint32_t> atomic_fragment(std::vector<uint32_t> input, bool copy_pointer=false) {
    if (input.size()<5 || input[0]!=0x07230203u) return {};
    // SPIR-V forbids duplicate non-aggregate type declarations. The actual recompiled FS already
    // declares uint even when its visible output only uses float. Reuse that scalar and an existing
    // StorageBuffer pointer to it; the new aggregate block has its own distinct pointee identity.
    uint32_t existing_uint=0, existing_uint_pointer=0;
    for (size_t at=5; at<input.size();) {
        const uint32_t count=input[at]>>16, op=input[at]&0xffffu;
        if (!count || at+count>input.size()) return {};
        if (op==21 && count==4 && input[at+2]==32 && input[at+3]==0)
            existing_uint=input[at+1];
        at+=count;
    }
    if (existing_uint) {
        for (size_t at=5; at<input.size();) {
            const uint32_t count=input[at]>>16, op=input[at]&0xffffu;
            if (op==32 && count==4 && input[at+2]==12 && input[at+3]==existing_uint)
                existing_uint_pointer=input[at+1];
            at+=count; // The complete instruction stream was validated above.
        }
    }
    auto next = input[3];
    auto id = [&] { return next++; };
    const uint32_t u=existing_uint ? existing_uint : id();
    const uint32_t arr=id(), block=id(), pb=id();
    const uint32_t pu=existing_uint_pointer ? existing_uint_pointer : id();
    const uint32_t zero=id(), index=id(), scope=id(), variable=id(), pointer=id(), value=id();
    const uint32_t atomic_pointer=copy_pointer ? id() : pointer;
    auto emit=[](std::vector<uint32_t>& out,uint32_t op,std::initializer_list<uint32_t> args) {
        out.push_back((uint32_t(args.size()+1)<<16)|op); out.insert(out.end(),args);
    };
    std::vector<uint32_t> annotations, declarations, code;
    emit(annotations,71,{arr,6,4}); emit(annotations,71,{block,2});
    emit(annotations,72,{block,0,35,0});
    emit(annotations,71,{variable,34,1}); emit(annotations,71,{variable,33,4});
    if (!existing_uint) emit(declarations,21,{u,32,0});
    emit(declarations,29,{arr,u});
    emit(declarations,30,{block,arr}); emit(declarations,32,{pb,12,block});
    if (!existing_uint_pointer) emit(declarations,32,{pu,12,u});
    emit(declarations,43,{u,zero,0});
    emit(declarations,43,{u,index,16}); emit(declarations,43,{u,scope,1});
    emit(declarations,59,{pb,variable,12});
    emit(code,65,{pu,pointer,variable,zero,index});
    // A legal pointer copy deliberately exceeds the reflector's direct-root attribution. The
    // whole-module incomplete-write proof must therefore veto immutable residency, even though
    // the ordinary descriptor declarations and Vulkan shader remain valid.
    if (copy_pointer) emit(code,83,{pu,atomic_pointer,pointer}); // OpCopyObject
    emit(code,241,{u,value,atomic_pointer,scope,zero,zero}); // OpAtomicOr, Device, Relaxed, 0
    std::vector<uint32_t> result(input.begin(), input.begin()+5);
    bool added_annotations=false, added_declarations=false, added_code=false;
    for (size_t at=5; at<input.size();) {
        const uint32_t count=input[at]>>16, op=input[at]&0xffffu;
        if (!count || at+count>input.size()) return {};
        if (!added_annotations && op>=19 && op<=39) {
            result.insert(result.end(),annotations.begin(),annotations.end()); added_annotations=true;
        }
        if (!added_declarations && op==54) {
            result.insert(result.end(),declarations.begin(),declarations.end()); added_declarations=true;
        }
        if (op==253) {
            result.insert(result.end(),code.begin(),code.end()); added_code=true;
        }
        const size_t start=result.size();
        result.insert(result.end(),input.begin()+at,input.begin()+at+count);
        if (op==15 && input[1]>=0x00010400u) {
            result[start]+=1u<<16; result.push_back(variable);
        }
        at+=count;
    }
    result[3]=next;
    return added_annotations && added_declarations && added_code ? result : std::vector<uint32_t>{};
}
}

int main(int argc, char** argv) {
    std::printf("== renderer buffer residency ==\n");
    Fixture f;
    CHECK(!f.vs.empty() && !f.fs.empty(), "real vertex-fetch and green fragment shaders compile");
    if (f.vs.empty() || f.fs.empty()) return 1;
    if (argc != 1) {
        if (argc != 3 || std::string(argv[1]) != "--dump-spv") return 2;
        const std::filesystem::path directory(argv[2]);
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) return 2;
        const auto atomic = atomic_fragment(f.fs);
        const auto unresolved = atomic_fragment(f.fs,true);
        if (atomic.empty() || unresolved.empty()) return 1;
        const auto save = [&](const char* name, const std::vector<uint32_t>& words) {
            const auto path = directory / name;
            if (std::filesystem::exists(path)) return false;
            std::ofstream stream(path, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(words.data()),
                         static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
            stream.close();
            return !stream.fail();
        };
        CHECK(save("vertex.spv", f.vs), "export actual vertex-fetch module");
        CHECK(save("fragment.spv", f.fs), "export actual green fragment module");
        CHECK(save("fragment-atomic.spv", atomic), "export actual alias-writer module");
        CHECK(save("fragment-unresolved.spv", unresolved), "export actual copied-pointer atomic module");
        return failures ? 1 : 0; // Export requires no Vulkan device or runtime execution.
    }
    CHECK(backend_module_has_readonly_buffers(f.vs) && backend_module_has_readonly_buffers(f.fs),
          "final shaders have complete read-only storage-buffer proof");
    auto current=payload();
    constexpr uint64_t Identity=0x348900001ull;
    disabled(true);
    auto green=f.render(current, Identity);
    CHECK(solid(green,true), "CPU-upload control covers every pixel green");
    CHECK(backend_resource_reuse_stats().buffer_resident_hits==0,
          "explicit disabled control does not borrow resident input");
    disabled(false);
    CHECK(f.render(current,Identity)==green,"cold retained upload matches control pixels");
    CHECK(backend_resource_reuse_stats().buffer_resident_admitted_bytes==Bytes,
          "one complete current source is admitted");
    auto& cache=resident_render_buffer_cache();
    std::weak_ptr<ResidentRenderBuffer> completed_owner;
    VkBuffer completed_buffer=VK_NULL_HANDLE;
    {
        BackendPersistentResourceGuard guard;
        const auto found=cache.index.find({Identity,Bytes});
        CHECK(found!=cache.index.end(),"completed cold upload remains resident");
        if (found!=cache.index.end()) {
            completed_owner=found->second->owner;
            completed_buffer=found->second->owner->storage.buffer;
            CHECK(found->second->owner.use_count()==1,
                  "completed upload has no outstanding submission owner before refresh");
            CHECK(found->second->owner->snapshot_bytes == Bytes &&
                      found->second->owner->retained_bytes() ==
                          found->second->owner->storage.allocation_bytes + Bytes,
                  "residency budget includes the exact CPU comparison snapshot and Vulkan allocation");
            // No queued commands use this completed upload. Make its mapping unreadable to this
            // lookup without changing the real Vulkan allocation: an equal hit must only inspect
            // ordinary CPU snapshot memory. A mapped-memory comparison would fault here.
            auto& owner = found->second->owner;
            void* mapping = owner->storage.mapped;
            owner->storage.mapped = nullptr;
            BackendResourceReuseStats observed;
            auto hit = cache.find(owner->device, {Identity,Bytes}, current.data(), observed);
            owner->storage.mapped = mapping;
            CHECK(hit == owner && observed.buffer_resident_hits == 1,
                  "equal lookup does not read the potentially uncached Vulkan mapping");
        }
    }
    for (auto first:Quads) {
        CHECK(f.render(current,Identity,first)==green,
              "unchanged first/middle/last vertex fetch sees the retained complete payload");
        CHECK(backend_resource_reuse_stats().buffer_resident_hits==1 &&
              backend_resource_reuse_stats().buffer_resident_reused_bytes==Bytes,
              "unchanged renderer call reuses the complete immutable upload");
        quad(current,first,false);
        CHECK(solid(f.render(current,Identity,first),false),
              "changed first/middle/last source region produces the actual clear-blue frame");
        const auto refreshed=backend_resource_reuse_stats();
        CHECK(refreshed.buffer_resident_hits==0 && refreshed.buffer_resident_reused_bytes==0 &&
                  refreshed.buffer_resident_refreshed_bytes==Bytes &&
                  refreshed.buffer_resident_admitted_bytes==0,
              "changed completed source performs a full refresh, not an unchanged hit or allocation");
        {
            BackendPersistentResourceGuard guard;
            const auto found=cache.index.find({Identity,Bytes});
            const auto original=completed_owner.lock();
            CHECK(original && found!=cache.index.end() && found->second->owner==original &&
                      original->storage.buffer==completed_buffer,
                  "idle exact-key refresh preserves the same owner and actual Vulkan buffer");
        }
        quad(current,first,true);
        CHECK(f.render(current,Identity,first)==green,"restored source repairs stale retained bytes");
    }
    auto hosted=payload(); quad(hosted,Quads.back(),false);
    CHECK(hosted.data()!=current.data(),"replacement host backing is simultaneously distinct");
    CHECK(solid(f.render(hosted,Identity,Quads.back()),false),
          "same guest identity with replacement host bytes does not reuse stale pixels");
    CHECK(f.render(current,Identity,Quads.back())==green,
          "switching back to original host backing restores correct pixels");

    // Cold versions recorded before either submission completes. Inspect existing owner handles
    // only to prove lifetime/budget; both independent images are the content oracle.
    constexpr uint64_t PendingIdentity=0x348900002ull;
    BackendColorTarget target_a{0x3489a001ull,false,false}, target_b{0x3489b001ull,false,true};
    BackendSubmissionBatch batch;
    auto pending=payload();
    auto no_pixels=render_draws_rgba({f.draw(pending,PendingIdentity,Quads.back())},W,H,
        nullptr,Clear,false,&target_a,nullptr,nullptr,nullptr,&batch,false,nullptr,false);
    CHECK(no_pixels.empty() && batch.pending(),"first real render retains an unsubmitted upload");
    std::weak_ptr<ResidentRenderBuffer> old;
    uint64_t old_charge=0;
    {
        BackendPersistentResourceGuard guard;
        auto found=cache.index.find({PendingIdentity,Bytes});
        CHECK(found!=cache.index.end(),"pending source has a resident allocation");
        if (found!=cache.index.end()) {
            old=found->second->owner; old_charge=found->second->owner->retained_bytes();
        }
    }
    quad(pending,Quads.back(),false);
    auto b=render_draws_rgba({f.draw(pending,PendingIdentity,Quads.back())},W,H,
        nullptr,Clear,false,&target_b,nullptr,nullptr,nullptr,&batch,false,nullptr,false);
    CHECK(b.empty() && batch.pending() && !old.expired(),
          "changed source leaves the older recorded allocation alive until completion");
    {
        BackendPersistentResourceGuard guard;
        CHECK(cache.charged_bytes->load()>=2*old_charge && old_charge!=0,
              "detached recorded version remains charged alongside its replacement");
        const auto idle = cache.index.find({Identity,Bytes});
        CHECK(idle!=cache.index.end() && idle->second->owner.use_count()==1,
              "an independent completed upload is idle and available for reclamation");
        std::weak_ptr<ResidentRenderBuffer> idle_owner;
        if (idle!=cache.index.end()) idle_owner=idle->second->owner;
        const auto before=cache.charged_bytes->load();
        BackendResourceReuseStats declined;
        CHECK(!cache.admit(render_vk_ctx(), {PendingIdentity+100,Bytes}, current.data(),
                           old_charge, declined),
              "pinned versions cannot be reclaimed to fake available budget");
        CHECK(!idle_owner.expired() && cache.index.find({Identity,Bytes})!=cache.index.end() &&
                  cache.charged_bytes->load()==before,
              "some idle bytes cannot satisfy admission: preserve that unrelated cache entry");
    }
    const auto& ctx=render_vk_ctx();
    BackendSubmissionBatchResult submitted;
    {
        BackendPersistentResourceGuard guard;
        submitted=batch.submit_and_wait(ctx.dev,ctx.queue,false);
        if (submitted.submit_result==VK_SUCCESS && submitted.wait_result==VK_SUCCESS) batch.complete();
    }
    CHECK(submitted.submit_result==VK_SUCCESS && submitted.wait_result==VK_SUCCESS &&
              submitted.command_buffers==2, "both recorded passes complete successfully");
    if (submitted.submit_result!=VK_SUCCESS || submitted.wait_result!=VK_SUCCESS) return 1;
    CHECK(!batch.pending() && old.expired(),"completed cleanup releases the detached old version");
    std::vector<uint8_t> a_pixels,b_pixels; std::string error;
    CHECK(readback_persistent_color_target(target_a.persistent_id,W,H,VK_FORMAT_UNDEFINED,a_pixels,error) &&
              a_pixels==green,"earlier queued draw reads original bytes after later source mutation");
    CHECK(readback_persistent_color_target(target_b.persistent_id,W,H,VK_FORMAT_UNDEFINED,b_pixels,error) &&
              solid(b_pixels,false),"later queued draw independently reads replacement bytes");

    // A different exact range is a different upload, even when the entire old prefix matches.
    auto larger=current; larger.resize(Words*2,0);
    const uint32_t new_tail=static_cast<uint32_t>(larger.size()/2-4);
    quad(larger,new_tail,true);
    CHECK(f.render(larger,Identity,new_tail)==green,
          "larger same-identity range reads newly added tail vertices correctly");
    CHECK(backend_resource_reuse_stats().buffer_resident_hits==0 &&
              backend_resource_reuse_stats().buffer_resident_admitted_bytes==larger.size()*4,
          "changed extent cannot borrow the previous shorter descriptor range");
    CHECK(f.render(current,Identity)==green && backend_resource_reuse_stats().buffer_resident_hits==1,
          "original exact-size entry remains independently reusable");
    quad(larger,new_tail,false);
    CHECK(solid(f.render(larger,Identity,new_tail),false),
          "mutation beyond the original range changes the real rendered result");

    // Discard is a never-submitted path. Its normal complete() releases recorded resource closures;
    // cached latest versions remain owned, but a detached older version must not leak its charge.
    constexpr uint64_t DiscardIdentity=0x348900004ull;
    BackendColorTarget discard_a{0x3489c001ull,false,false}, discard_b{0x3489d001ull,false,false};
    BackendSubmissionBatch discarded;
    auto discard_source=payload();
    (void)render_draws_rgba({f.draw(discard_source,DiscardIdentity)},W,H,nullptr,Clear,false,
        &discard_a,nullptr,nullptr,nullptr,&discarded,false,nullptr,false);
    std::weak_ptr<ResidentRenderBuffer> detached;
    uint64_t detached_charge=0;
    {
        BackendPersistentResourceGuard guard;
        const auto found=cache.index.find({DiscardIdentity,Bytes});
        CHECK(found!=cache.index.end(),"discard control records a resident source");
        if (found!=cache.index.end()) {
            detached=found->second->owner; detached_charge=found->second->owner->retained_bytes();
        }
    }
    quad(discard_source,0,false);
    (void)render_draws_rgba({f.draw(discard_source,DiscardIdentity)},W,H,nullptr,Clear,false,
        &discard_b,nullptr,nullptr,nullptr,&discarded,false,nullptr,false);
    CHECK(discarded.pending() && !detached.expired(),
          "discard control has an old detached version retained by unsubmitted commands");
    {
        BackendPersistentResourceGuard guard;
        const auto before=cache.charged_bytes->load();
        discarded.discard();
        discarded.complete();
        CHECK(!discarded.pending() && detached.expired() && detached_charge!=0 &&
                  cache.charged_bytes->load()==before-detached_charge,
              "never-submitted discard cleanup releases exactly the detached version charge");
    }
    CHECK(solid(f.render(discard_source,DiscardIdentity),false) &&
              backend_resource_reuse_stats().buffer_resident_hits==1,
          "discarded commands leave the immutable current host-upload snapshot reusable");

    auto small=current; small.resize(16);
    CHECK(solid(f.render(small,0x348900003ull),true),"small source retains correct ordinary upload");
    CHECK(backend_resource_reuse_stats().buffer_resident_admitted_bytes==0,
          "small sources remain outside residency admission");
    CHECK(f.render(current,0)==green,"zero-identity input remains a valid rendering source");
    CHECK(backend_resource_reuse_stats().buffer_resident_admitted_bytes==0,
          "zero identity is not upgraded to persistent guest authority");

    if (ctx.fragment_stores_atomics) {
        auto shared_reader=f.draw(current,Identity);
        shared_reader.vs_shared=std::make_shared<const std::vector<uint32_t>>(f.vs);
        shared_reader.fs_shared=std::make_shared<const std::vector<uint32_t>>(f.fs);
        shared_reader.vs.clear();
        // Raw words deliberately disagree: shared words are the effective shader until set_fs.
        shared_reader.fs=atomic_fragment(f.fs);
        shared_reader.fs_identity=0x3489f001ull;
        CHECK(!shared_reader.fs.empty() && shared_reader.fs_words()==f.fs,
              "shared fragment words take precedence over a raw writer decoy");
        CHECK(render_draws_rgba({shared_reader},W,H,nullptr,Clear)==green &&
                  backend_resource_reuse_stats().buffer_resident_hits==1,
              "actual shared-module read-only pass warms its proof and reuses resident input");
        CHECK(render_draws_rgba({shared_reader},W,H,nullptr,Clear)==green &&
                  backend_resource_reuse_stats().buffer_resident_hits==1,
              "the same live shared owners remain eligible on a repeated pass");
        auto writer=shared_reader;
        writer.set_fs(atomic_fragment(f.fs));
        CHECK(!writer.fs_shared && writer.fs_identity==0,
              "set_fs replaces shared fragment authority and clears the old pipeline identity");
        CHECK(!writer.fs.empty() && !backend_module_has_readonly_buffers(writer.fs),
              "actual fragment atomic prevents whole-pass immutable admission");
        FrameResource alias=writer.R[0]; alias.set=1; alias.binding=4;
        writer.R.push_back(alias);
        auto output=render_draws_rgba({shared_reader,writer},W,H,nullptr,Clear);
        CHECK(output==green,"reader plus atomic alias writer renders correctly through ordinary uploads");
        CHECK(backend_resource_reuse_stats().buffer_resident_hits==0 &&
              backend_resource_reuse_stats().buffer_resident_admitted_bytes==0,
              "later fragment writer excludes resident input for the entire multi-draw pass");
        auto unresolved_writer=shared_reader;
        unresolved_writer.set_fs(atomic_fragment(f.fs,true));
        CHECK(!unresolved_writer.fs.empty(), "copied-pointer atomic module is constructed");
        if (!unresolved_writer.fs.empty()) {
            const auto report=validate_spirv_descriptor_interface(
                unresolved_writer.fs,nullptr,0,SpirvShaderStage::Fragment,false);
            CHECK(!report.storage_buffer_writes_complete &&
                      !backend_module_has_readonly_buffers(unresolved_writer.fs),
                  "unresolved atomic pointer fails the complete negative-write proof");
            unresolved_writer.R.push_back(alias);
            auto unresolved_pixels=render_draws_rgba(
                {shared_reader,unresolved_writer},W,H,nullptr,Clear);
            CHECK(unresolved_pixels==green,
                  "valid copied-pointer atomic executes with correct pixels using ordinary uploads");
            CHECK(backend_resource_reuse_stats().buffer_resident_hits==0 &&
                      backend_resource_reuse_stats().buffer_resident_admitted_bytes==0 &&
                      backend_resource_reuse_stats().buffer_resident_refreshed_bytes==0,
                  "incomplete write attribution excludes resident inputs for the whole actual pass");
        }
    } else std::printf("[UNSUPPORTED] fragment atomics; writer runtime arm not exercised\n");
    {
        BackendPersistentResourceGuard guard;
        auto temporary=std::make_shared<const std::vector<uint32_t>>(f.fs);
        std::weak_ptr<const std::vector<uint32_t>> observed=temporary;
        CHECK(backend_module_has_readonly_buffers(*temporary,temporary),
              "an independently owned shared module can populate the proof memo");
        temporary.reset();
        CHECK(observed.expired(),"proof memo does not keep an otherwise dead shader owner alive");
    }
    // A working set larger than residency must not allocate/free Vulkan objects on each scan.
    // All previous submissions completed. Exercise pressured rekeying in the actual global cache,
    // then use a real draw to prove its replacement mapping contains the new snapshot bytes.
    {
        BackendPersistentResourceGuard guard;
        cache.index.clear();
        cache.lru.clear();
        CHECK(cache.charged_bytes->load() == 0, "completed residency releases both memory charges");
    }
    constexpr uint64_t RekeyIdentity = 0x348900100ull;
    CHECK(f.render(current, RekeyIdentity) == green, "rekey control starts from a real green upload");
    auto replacement = current;
    quad(replacement, 0, false);
    {
        BackendPersistentResourceGuard guard;
        const auto before = cache.charged_bytes->load();
        const auto found = cache.index.find({RekeyIdentity, Bytes});
        CHECK(found != cache.index.end(), "rekey control has one retained allocation");
        if (found != cache.index.end()) {
            const VkBuffer original_buffer = found->second->owner->storage.buffer;
            auto pin = found->second->owner;
            BackendResourceReuseStats stats;
            CHECK(!cache.admit(ctx, {RekeyIdentity+1,Bytes}, replacement.data(), before, stats),
                  "full pinned cache declines replacement without overwriting its recorded owner");
            pin.reset();
            auto incompatible = current; incompatible.resize(Words*2);
            CHECK(!cache.admit(ctx, {RekeyIdentity+2,Bytes*2}, incompatible.data(), before, stats) &&
                      cache.index.contains({RekeyIdentity,Bytes}) &&
                      cache.charged_bytes->load() == before,
                  "full incompatible cache declines without eviction or extra allocation");
            auto reused = cache.admit(ctx, {RekeyIdentity+1,Bytes}, replacement.data(), before, stats);
            CHECK(reused && reused->storage.buffer == original_buffer &&
                      cache.charged_bytes->load() == before &&
                      !cache.index.contains({RekeyIdentity,Bytes}) &&
                      cache.index.contains({RekeyIdentity+1,Bytes}),
                  "pressured equal-size admission rekeys an idle Vulkan allocation at constant charge");
            CHECK(reused && stats.buffer_resident_admitted_bytes == Bytes &&
                      stats.buffer_resident_hits == 0 && stats.buffer_upload_bytes == Bytes,
                  "rekey is a new copied admission, never an unchanged source hit");
        }
    }
    CHECK(solid(f.render(replacement, RekeyIdentity+1), false),
          "actual rekeyed GPU upload draws the changed blue result rather than stale green geometry");
    // A non-power-of-two payload needs a rounded Vulkan allocation AND an exact CPU snapshot.
    // 13 KiB spare fits twice a 6 KiB payload, but cannot fit its 8+6 KiB resident allocation.
    auto rounded_source = current;
    rounded_source.resize(1536);
    constexpr VkDeviceSize RoundedBytes = 6144;
    CHECK(f.render(rounded_source, RekeyIdentity+3) == green,
          "non-power-of-two control uploads real six-KiB vertex data");
    quad(rounded_source,0,false);
    {
        BackendPersistentResourceGuard guard;
        const auto before = cache.charged_bytes->load();
        const auto found = cache.index.find({RekeyIdentity+3,RoundedBytes});
        CHECK(found != cache.index.end(), "rounded control owns the six-KiB upload");
        if (found != cache.index.end()) {
            const auto original_buffer = found->second->owner->storage.buffer;
            BackendResourceReuseStats stats;
            auto rekeyed = cache.admit(ctx, {RekeyIdentity+4,RoundedBytes}, rounded_source.data(),
                                      before + 13*1024, stats);
            CHECK(rekeyed && rekeyed->storage.buffer == original_buffer &&
                      cache.charged_bytes->load() == before,
                  "rounded allocation pressure rekeys instead of allocating and rejecting repeatedly");
        }
    }
    CHECK(solid(f.render(rounded_source, RekeyIdentity+4),false),
          "rounded rekey uploads the changed bytes to the actual GPU draw");
    disabled(false);
    std::printf("== %s (%d failures) ==\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
