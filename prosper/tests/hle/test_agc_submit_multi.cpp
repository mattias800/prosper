// Multi-DCB imports must fold every array entry in order as one submission. Calling the single
// submit HLE repeatedly loses earlier draws, resets cross-buffer barriers, and leaks import scopes.
// Sony error semantics for malformed batches are unproved: unsupported shapes must diagnose and abort
// rather than claim success. The one proved shape is count == 0 (#4741): the shared submit worker
// tests the count first and returns an empty-batch code, so that case returns it and folds nothing.
// All command words and mappings below are synthetic; no renderer is created.
#include "hle/dispatch/dispatch.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "support/death_test.hpp"
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace prosper;
extern "C" void prosper_agc_submit_stats(uint64_t* submits, uint64_t* draws);
extern "C" bool prosper_agc_submit_sh_reg(uint64_t queue, uint32_t offset, uint32_t* value);

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++fails; } } while (0)
static uint64_t U(const void* p) { return reinterpret_cast<uintptr_t>(p); }
static uint32_t header(uint32_t words, uint32_t op, uint32_t sub = 0) {
    return 0xc0000000u | ((words - 2) << 16) | (op << 8) | (sub << 2);
}
static void emit_release(uint32_t* out, const void* target, uint32_t value) {
    const uint64_t address = U(target);
    out[0] = header(7, gpu::IT_NOP, gpu::R_RELEASE_MEM);
    out[1] = static_cast<uint32_t>(address); out[2] = static_cast<uint32_t>(address >> 32);
    out[3] = 1; out[4] = value; out[5] = 0; out[6] = 0x2d;
}
static void emit_wait(uint32_t* out, const void* target) {
    const uint64_t address = U(target);
    out[0] = header(8, gpu::IT_NOP, gpu::R_WAIT_MEM_64);
    out[1] = static_cast<uint32_t>(address); out[2] = static_cast<uint32_t>(address >> 32);
    out[3] = 0xffffffffu; out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 3;
}

// Put the supplied descriptor/stream immediately before an unreadable page. Full-range validation
// must refuse it before decoding; checking only its first byte would fault in the guard page.
static uint8_t* guarded_end() {
#ifdef _WIN32
    SYSTEM_INFO info{}; GetSystemInfo(&info);
    const size_t page = info.dwPageSize;
    auto* mapping = static_cast<uint8_t*>(VirtualAlloc(nullptr, page * 2,
                                                      MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    DWORD old = 0;
    if (!mapping || !VirtualProtect(mapping + page, page, PAGE_NOACCESS, &old)) return nullptr;
#else
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* mapping = static_cast<uint8_t*>(mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
                                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (mapping == MAP_FAILED || mprotect(mapping + page, page, PROT_NONE) != 0) return nullptr;
#endif
    return mapping + page;
}

static int child(const char* mode, HleFn submit) {
    test::suppress_coredump();
#ifdef _WIN32
    // Microsoft's CRT documents exit 3 when report-fault is disabled. Keep intentional aborts
    // headless; the parent's independent abort control verifies the actual status on this host.
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    if (!std::strcmp(mode, "probe")) return 0;
    if (!std::strcmp(mode, "abort-control")) std::abort();
    if (!std::strcmp(mode, "wrong-failure-control")) return 2;
    if (!submit) return 74;
    uint32_t stream[] = {0x80000000u};
    uint64_t pointers[] = {U(stream), U(stream)};
    uint32_t lengths[] = {1, 1};
    uint64_t p = U(pointers), n = U(lengths), count = 1;
    if (!std::strcmp(mode, "null-pointers"))
        p = 0;
    else if (!std::strcmp(mode, "null-lengths")) n = 0;
    else if (!std::strcmp(mode, "array-size")) count = 0xffffffffu;
    else if (!std::strcmp(mode, "null-stream")) pointers[0] = 0;
    else if (!std::strcmp(mode, "unaligned-stream")) pointers[0] = U(stream) + 1;
    else if (!std::strcmp(mode, "zero-length")) lengths[0] = 0;
    else if (!std::strcmp(mode, "stream-size")) lengths[0] = 0x40000000u;
    else if (!std::strcmp(mode, "short-packet")) stream[0] = header(3, gpu::IT_SET_SH_REG);
    else if (!std::strcmp(mode, "unknown-packet")) {
        uint32_t unknown[] = {header(2, 0xfe), 0};
        pointers[0] = U(unknown); lengths[0] = 2;
        submit(p, n, count, 0, 0, 0);
        return 73;
    } else {
        auto* end = guarded_end();
        if (!end) return 74;
        if (!std::strcmp(mode, "pointer-guard")) {
            std::memcpy(end - 8, pointers, 8); p = U(end - 8); count = 2;
        } else if (!std::strcmp(mode, "length-guard")) {
            std::memcpy(end - 4, lengths, 4); n = U(end - 4); count = 2;
        } else if (!std::strcmp(mode, "stream-guard")) {
            std::memcpy(end - 4, stream, 4); pointers[0] = U(end - 4); lengths[0] = 2;
        } else return 74;
    }
    submit(p, n, count, 0, 0, 0);
    return 73; // A normal HLE return is a failure, regardless of the guessed error value.
}
static int run_child(const char* executable, const char* mode) {
    std::string command = "\"" + std::string(executable) + "\" " + mode;
    return std::system(command.c_str());
}
static bool aborted(int status) {
#ifdef _WIN32
    return status == 3; // CRT abort with _CALL_REPORTFAULT disabled; not any nonzero failure.
#else
    // system() launches a shell, which reports signal termination as 128 + signal on these hosts.
    return (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) ||
           (WIFEXITED(status) && WEXITSTATUS(status) == 128 + SIGABRT);
#endif
}

int main(int argc, char** argv) {
#ifdef _WIN32
    _putenv_s("PROSPER_WAIT_DEFER", "1");
#else
    setenv("PROSPER_WAIT_DEFER", "1", 1);
#endif
    register_builtin_hle();
    auto submit = Hle::lookup("6UzEidRZwkg");
    if (argc == 2) return child(argv[1], submit);
    CHECK(submit != nullptr, "MultiDcbs import registered");
    auto hook = Hle::return_hook_of("6UzEidRZwkg");
    CHECK(hook == &::prosper_gpu_submit_scope_end, "MultiDcbs owns the established return hook");
    if (!submit || !hook) return 1;

    uint32_t first[] = {header(3, gpu::IT_SET_SH_REG), 0x123, 0x11223344,
                       header(3, gpu::IT_NOP, gpu::R_DRAW_INDEX_AUTO), 3, 0};
    uint32_t second[] = {0x80000000u, header(2, gpu::IT_NOP), 0xa5a5a5a5,
                        header(3, gpu::IT_SET_SH_REG), 0x123, 0x55667788,
                        header(3, gpu::IT_NOP, gpu::R_DRAW_INDEX_AUTO), 5, 0};
    uint64_t pointers[] = {U(first), U(second)};
    uint32_t lengths[] = {6, 9};
    uint64_t before = 0, draws = 0; prosper_agc_submit_stats(&before, &draws);
    submit(U(pointers), U(lengths), 1, 0, 0, 0);
    uint64_t after = 0; prosper_agc_submit_stats(&after, &draws);
    uint32_t value = 0;
    CHECK(after == before + 1 && draws == 1, "one-entry array executes its draw as one submission");
    CHECK(prosper_agc_submit_sh_reg(0, 0x123, &value) && value == 0x11223344,
          "one-entry array applies the pointed-to stream");
    CHECK(::prosper_gpu_submit_scope_active(), "import scope remains active before return hook");
    hook();
    CHECK(!::prosper_gpu_submit_scope_active(), "one return hook retires one import scope");

    before = after;
    submit(U(pointers), U(lengths), 2, 0, 0, 0);
    prosper_agc_submit_stats(&after, &draws);
    CHECK(after == before + 1 && draws == 2, "two buffers retain both draws in one submission");
    CHECK(prosper_agc_submit_sh_reg(0, 0x123, &value) && value == 0x55667788,
          "later array entry overwrites the earlier register in order");
    hook();
    CHECK(!::prosper_gpu_submit_scope_active(), "multi-buffer submission opens only one scope");

    // count == 0 returns the library's empty-batch code and folds nothing. The constant is
    // build-dependent (hle_agc.cpp, kAgcDriverErrEmptyBatch): this pins the project module's value,
    // written as a literal so a mutated handler constant cannot agree with itself.
    before = after;
    CHECK(submit(U(pointers), U(lengths), 0, 0, 0, 0) == 0x8a6d0000ull,
          "count == 0 returns the empty-batch code instead of aborting");
    prosper_agc_submit_stats(&after, &draws);
    CHECK(after == before, "an empty batch folds nothing");
    hook();
    CHECK(!::prosper_gpu_submit_scope_active(),
          "the empty batch's scope is retired by the return hook");

    // Distinct target addresses matter: per-address gating would conceal a fold reset if both
    // buffers wrote the same label. A blocked WAIT must gate even the fresh second-buffer address.
    uint64_t condition = 0, upstream = 0, downstream = 0;
    uint32_t waiting[15], tail[7];
    emit_release(waiting, &upstream, 1); emit_wait(waiting + 7, &condition);
    emit_release(tail, &downstream, 9);
    const gpu::CommandBuffer barrier_buffers[] = {{waiting, 15}, {tail, 7}};
    gpu::GpuState barrier_state;
    size_t consumed[2]{};
    CHECK(gpu::run_command_buffers(barrier_buffers, 2, barrier_state, consumed) == 3 &&
          consumed[0] == 15 && consumed[1] == 7, "batch processor consumes both complete streams");
    CHECK(gpu::last_fold_deferred() && gpu::deferred_pending(),
          "batch still reports a first-buffer wait after folding the second buffer");
    ::prosper_gpu_drain_completion_writes();
    CHECK(upstream == 1 && downstream == 0,
          "upstream write drains while distinct second-buffer effect remains gated");
    condition = 1;
    // This processor-only case starts no HLE watchdog, so release and inspection stay on one thread.
    gpu::flush_deferred_streams();
    ::prosper_gpu_drain_completion_writes();
    CHECK(downstream == 9 && !gpu::deferred_pending(), "satisfied barrier releases the batch tail");

    CHECK(run_child(argv[0], "probe") == 0, "death-test child launches successfully");
    CHECK(aborted(run_child(argv[0], "abort-control")), "hand-built abort control is recognized");
    CHECK(!aborted(run_child(argv[0], "wrong-failure-control")), "ordinary nonzero exit is not an abort");
    const char* modes[] = {"null-pointers",    "null-lengths",  "array-size",   "null-stream",
                           "unaligned-stream", "zero-length",   "stream-size",  "short-packet",
                           "unknown-packet",   "pointer-guard", "length-guard", "stream-guard"};
    for (const char* mode : modes) CHECK(aborted(run_child(argv[0], mode)), mode);
    std::printf("%d failures\n", fails);
    return fails ? 1 : 0;
}
