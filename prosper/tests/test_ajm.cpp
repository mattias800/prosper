// test_ajm (#187) — libSceAjm headless decode lifecycle. prosper doesn't decode compressed audio, but
// the handle lifecycle must be REAL: Initialize/InstanceCreate/BatchStartBuffer fill their handle
// out-param with a valid value (never the caller's garbage — the bug the generic stub caused), inputs
// are validated, and a batch reports started->complete so the guest's audio pipeline proceeds. Names
// hash to the game's import NIDs, so R()-registration reaches the guest.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static constexpr uint64_t kInvalidContext   = 0xffffffff80930002ull;
static constexpr uint64_t kInvalidInstance  = 0xffffffff80930003ull;
static constexpr uint64_t kInvalidBatch     = 0xffffffff80930004ull;
static constexpr uint64_t kInvalidParameter = 0xffffffff80930005ull;

using Hle8Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t);
using Hle10Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                             uint64_t, uint64_t, uint64_t, uint64_t);

struct JobChunk { uint32_t header, size; };
struct FlagsChunk { uint32_t header, flags_low; };
struct BufferChunk { uint32_t header, size; uint64_t address; };
struct GuestBuffer { uint64_t address, size; };
static_assert(sizeof(JobChunk) == 8 && sizeof(FlagsChunk) == 8 &&
              sizeof(BufferChunk) == 16 && sizeof(GuestBuffer) == 16);

static uint32_t ident(uint32_t header) { return header & 0x3f; }
static uint32_t payload(uint32_t header) { return (header >> 6) & 0xfffff; }

template <typename T>
static T chunk_at(const uint8_t* bytes, size_t offset) {
    T value{};
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

int main() {
    printf("== test_ajm ==\n");
    register_builtin_hle();

    // The names must hash to the game's import NIDs (#187 table), else R() registers under the wrong key.
    CHECK(nid_hash("sceAjmInitialize") == "dl+4eHSzUu4", "sceAjmInitialize hashes to its import NID");
    CHECK(nid_hash("sceAjmModuleRegister") == "Q3dyFuwGn64", "sceAjmModuleRegister hashes to its import NID");
    CHECK(nid_hash("sceAjmBatchCancel") == "NVDXiUesSbA" &&
          nid_hash("sceAjmBatchJobControlBufferRa") == "dmDybN--Fn8" &&
          nid_hash("sceAjmBatchJobInlineBuffer") == "stlghnic3Jc" &&
          nid_hash("sceAjmBatchJobRunBufferRa") == "ElslOCpOIns" &&
          nid_hash("sceAjmBatchJobRunSplitBufferRa") == "7jdAXK+2fMo",
          "AJM cancel/cursor builders hash to the PS5 3.20 import NIDs");

    HleFn init = Hle::lookup(nid_hash("sceAjmInitialize"));
    HleFn modreg = Hle::lookup(nid_hash("sceAjmModuleRegister"));
    HleFn icreate = Hle::lookup(nid_hash("sceAjmInstanceCreate"));
    HleFn idestroy = Hle::lookup(nid_hash("sceAjmInstanceDestroy"));
    HleFn bstart = Hle::lookup(nid_hash("sceAjmBatchStartBuffer"));
    HleFn bwait = Hle::lookup(nid_hash("sceAjmBatchWait"));
    HleFn bcancel = Hle::lookup(nid_hash("sceAjmBatchCancel"));
    auto control = reinterpret_cast<Hle8Fn>(
        Hle::lookup(nid_hash("sceAjmBatchJobControlBufferRa")));
    HleFn inline_buffer = Hle::lookup(nid_hash("sceAjmBatchJobInlineBuffer"));
    auto run = reinterpret_cast<Hle10Fn>(
        Hle::lookup(nid_hash("sceAjmBatchJobRunBufferRa")));
    auto run_split = reinterpret_cast<Hle10Fn>(
        Hle::lookup(nid_hash("sceAjmBatchJobRunSplitBufferRa")));
    HleFn fin = Hle::lookup(nid_hash("sceAjmFinalize"));
    CHECK(init && modreg && icreate && idestroy && bstart && bwait && bcancel && control &&
          inline_buffer && run && run_split && fin, "AJM functions registered");
    if (!(init && modreg && icreate && idestroy && bstart && bwait && bcancel && control &&
          inline_buffer && run && run_split && fin)) { printf("== FAIL ==\n"); return 1; }

    // Initialize fills a valid (non-zero) context handle.
    uint32_t ctx = 0xDEAD;
    CHECK(init(0, (uint64_t)(uintptr_t)&ctx, 0, 0, 0, 0) == 0, "sceAjmInitialize -> OK");
    CHECK(ctx != 0 && ctx != 0xDEAD, "Initialize WROTE a valid non-zero context (not left garbage)");
    CHECK(init(0, 0, 0, 0, 0, 0) == kInvalidParameter, "Initialize(NULL out) -> INVALID_PARAMETER");
    CHECK(init(0, 1, 0, 0, 0, 0) == kInvalidParameter,
          "Initialize(inaccessible out) -> INVALID_PARAMETER without host fault");
    uint32_t rejected_ctx = 0xCAFE;
    CHECK(init(1, (uint64_t)(uintptr_t)&rejected_ctx, 0, 0, 0, 0) == kInvalidParameter,
          "Initialize(nonzero reserved) -> INVALID_PARAMETER");
    CHECK(rejected_ctx == 0xCAFE, "invalid Initialize leaves the context output untouched");

    // ModuleRegister needs a context.
    CHECK(modreg(ctx, 1 /*At9Dec*/, 0, 0, 0, 0) == 0, "sceAjmModuleRegister(ctx) -> OK");
    CHECK(modreg(0, 1, 0, 0, 0, 0) == kInvalidContext, "ModuleRegister(context 0) -> INVALID_CONTEXT");

    // InstanceCreate fills a valid instance handle, distinct from the context.
    uint32_t inst = 0xBEEF;
    CHECK(icreate(ctx, 1 /*At9Dec*/, 0 /*flags*/, (uint64_t)(uintptr_t)&inst, 0, 0) == 0, "sceAjmInstanceCreate -> OK");
    CHECK(inst != 0 && inst != 0xBEEF && inst != ctx, "InstanceCreate WROTE a distinct non-zero instance");
    CHECK(icreate(0, 1, 0, (uint64_t)(uintptr_t)&inst, 0, 0) == kInvalidContext, "InstanceCreate(ctx 0) -> INVALID_CONTEXT");
    CHECK(icreate(ctx, 1, 0, 1, 0, 0) == kInvalidParameter,
          "InstanceCreate(inaccessible out) -> INVALID_PARAMETER without host fault");

    // Batch: StartBuffer fills a batch id; Wait completes it.
    uint32_t batch = 0xF00D;
    CHECK(bstart(ctx, 0 /*batch buf*/, 0 /*size*/, 0 /*prio*/, 0 /*err*/, (uint64_t)(uintptr_t)&batch) == 0,
          "sceAjmBatchStartBuffer -> OK");
    CHECK(batch != 0 && batch != 0xF00D, "BatchStartBuffer WROTE a valid non-zero batch id");
    CHECK(bstart(ctx, 0, 0, 0, 0, 1) == kInvalidParameter,
          "BatchStartBuffer(inaccessible out) -> INVALID_PARAMETER without host fault");
    CHECK(bwait(ctx, batch, 0, 0, 0, 0) == 0, "sceAjmBatchWait -> OK (batch completed)");
    CHECK(bcancel(ctx, batch, 0, 0, 0, 0) == 0, "sceAjmBatchCancel accepts a valid batch handle");
    CHECK(bcancel(0, batch, 0, 0, 0, 0) == kInvalidContext,
          "sceAjmBatchCancel rejects context 0");
    CHECK(bcancel(ctx, 0, 0, 0, 0, 0) == kInvalidBatch,
          "sceAjmBatchCancel rejects batch 0");

    // The guest chains each builder's returned write cursor. Exercise byte-exact chunks, including
    // the return-address and flag fields that live past the first six register arguments.
    alignas(8) uint8_t job[256];
    std::memset(job, 0xEE, sizeof job);
    const uint64_t base = (uint64_t)(uintptr_t)job;
    const uint64_t input_addr = 0x1111222233334444ull;
    const uint64_t output_addr = 0x5555666677778888ull;
    const uint64_t sideband_addr = 0x9999aaaabbbbccccull;
    const uint64_t return_addr = 0xddddeeeeffff0000ull;
    uint64_t cursor = control(base, 0x12345, ~uint64_t{0}, input_addr, 0x21,
                              sideband_addr, 0x32, return_addr);
    const JobChunk control_job = chunk_at<JobChunk>(job, 0);
    const BufferChunk control_ra = chunk_at<BufferChunk>(job, 8);
    const BufferChunk control_in = chunk_at<BufferChunk>(job, 24);
    const FlagsChunk control_flags = chunk_at<FlagsChunk>(job, 40);
    const BufferChunk control_out = chunk_at<BufferChunk>(job, 48);
    CHECK(cursor == base + 64 && ident(control_job.header) == 0 &&
          payload(control_job.header) == 0x12345 && control_job.size == 56,
          "ControlBufferRa returns the next cursor and writes the job header");
    CHECK(ident(control_ra.header) == 6 && control_ra.size == 0 &&
          control_ra.address == return_addr && ident(control_in.header) == 2 &&
          control_in.size == 0x21 && control_in.address == input_addr,
          "ControlBufferRa serializes return and input-control chunks");
    CHECK(ident(control_flags.header) == 3 &&
          payload(control_flags.header) == 0x6000 && control_flags.flags_low == 0x0000e7ff &&
          ident(control_out.header) == 18 && control_out.size == 0x32 &&
          control_out.address == sideband_addr,
          "ControlBufferRa masks flags and serializes the output-control chunk");
    CHECK(job[64] == 0xEE, "ControlBufferRa writes exactly through its returned cursor");

    std::memset(job, 0xEE, sizeof job);
    const uint8_t inline_data[5] = {1, 2, 3, 4, 5};
    uint64_t inline_address = 0;
    cursor = inline_buffer(base, (uint64_t)(uintptr_t)inline_data, sizeof inline_data,
                           (uint64_t)(uintptr_t)&inline_address, 0, 0);
    const JobChunk inline_job = chunk_at<JobChunk>(job, 0);
    CHECK(cursor == base + 16 && inline_address == base + 8 &&
          ident(inline_job.header) == 7 && inline_job.size == 8,
          "InlineBuffer returns an aligned cursor and reports its in-batch payload address");
    CHECK(std::memcmp(job + 8, inline_data, sizeof inline_data) == 0 &&
          job[13] == 0 && job[14] == 0 && job[15] == 0 && job[16] == 0xEE,
          "InlineBuffer copies data, zero-pads to 8 bytes, and preserves the canary");
    CHECK(inline_buffer(base, 1, 4, (uint64_t)(uintptr_t)&inline_address, 0, 0) == 0,
          "InlineBuffer rejects inaccessible input without a host fault");

    std::memset(job, 0xEE, sizeof job);
    cursor = run(base, 0x23456, ~uint64_t{0}, input_addr, 0x43, output_addr, 0x54,
                 sideband_addr, 0x65, return_addr);
    const JobChunk run_job = chunk_at<JobChunk>(job, 0);
    const BufferChunk run_ra = chunk_at<BufferChunk>(job, 8);
    const BufferChunk run_in = chunk_at<BufferChunk>(job, 24);
    const FlagsChunk run_flags = chunk_at<FlagsChunk>(job, 40);
    const BufferChunk run_out = chunk_at<BufferChunk>(job, 48);
    const BufferChunk run_sideband = chunk_at<BufferChunk>(job, 64);
    CHECK(cursor == base + 80 && run_job.size == 72 && payload(run_job.header) == 0x23456,
          "RunBufferRa consumes all ten ABI arguments and advances by its exact job size");
    CHECK(ident(run_ra.header) == 6 && run_ra.address == return_addr &&
          ident(run_in.header) == 1 && run_in.size == 0x43 && run_in.address == input_addr &&
          ident(run_flags.header) == 4 && payload(run_flags.header) == 0xe000 &&
          run_flags.flags_low == 0x00001fff,
          "RunBufferRa serializes return/input chunks and masked run flags");
    CHECK(ident(run_out.header) == 17 && run_out.size == 0x54 &&
          run_out.address == output_addr && ident(run_sideband.header) == 18 &&
          run_sideband.size == 0x65 && run_sideband.address == sideband_addr && job[80] == 0xEE,
          "RunBufferRa serializes matching output chunks without overrunning the cursor");

    std::memset(job, 0xEE, sizeof job);
    const GuestBuffer inputs[2] = {{input_addr, 0x10}, {input_addr + 0x100, 0x20}};
    const GuestBuffer outputs[1] = {{output_addr, 0x30}};
    cursor = run_split(base, 0x34567, ~uint64_t{0}, (uint64_t)(uintptr_t)inputs, 2,
                       (uint64_t)(uintptr_t)outputs, 1, sideband_addr, 0x40, return_addr);
    const JobChunk split_job = chunk_at<JobChunk>(job, 0);
    const BufferChunk split_in0 = chunk_at<BufferChunk>(job, 24);
    const BufferChunk split_in1 = chunk_at<BufferChunk>(job, 40);
    const FlagsChunk split_flags = chunk_at<FlagsChunk>(job, 56);
    const BufferChunk split_out = chunk_at<BufferChunk>(job, 64);
    const BufferChunk split_sideband = chunk_at<BufferChunk>(job, 80);
    CHECK(cursor == base + 96 && split_job.size == 88 &&
          ident(split_in0.header) == 1 && split_in0.address == inputs[0].address &&
          split_in1.address == inputs[1].address,
          "RunSplitBufferRa serializes every input descriptor and advances the cursor");
    CHECK(ident(split_flags.header) == 4 && ident(split_out.header) == 17 &&
          split_out.address == outputs[0].address && split_out.size == outputs[0].size &&
          ident(split_sideband.header) == 18 && split_sideband.address == sideband_addr &&
          job[96] == 0xEE,
          "RunSplitBufferRa serializes flags and matching output/sideband chunks exactly");
    CHECK(run_split(base, 1, 0, 1, 1, (uint64_t)(uintptr_t)outputs, 1,
                    sideband_addr, 0x40, 0) == 0,
          "RunSplitBufferRa rejects an inaccessible descriptor array without a host fault");

    // Teardown.
    CHECK(idestroy(ctx, inst, 0, 0, 0, 0) == 0, "sceAjmInstanceDestroy -> OK");
    CHECK(idestroy(0, inst, 0, 0, 0, 0) == kInvalidContext, "InstanceDestroy(ctx 0) -> INVALID_CONTEXT");
    CHECK(idestroy(ctx, 0, 0, 0, 0, 0) == kInvalidInstance, "InstanceDestroy(instance 0) -> INVALID_INSTANCE");
    CHECK(fin(0, 0, 0, 0, 0, 0) == 0, "sceAjmFinalize -> OK");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
