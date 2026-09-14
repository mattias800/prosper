#pragma once

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/resources/fold_reader.hpp"
#include <filesystem>
#include <span>

namespace prosper::gpu {

inline constexpr uint32_t kFoldMaxCodeDwords = 65536;
inline constexpr uint32_t kFoldMaxEvents = 65536;
inline constexpr uint32_t kFoldMaxOutputs = 16384;
inline constexpr uint32_t kFoldMaxPayloadBytes = 4 * 1024 * 1024;
inline constexpr uint32_t kFoldMaxFileBytes = 16 * 1024 * 1024;

struct FoldInputs {
    uint64_t invocation = 0, code_address = 0;
    uint32_t stage = UINT32_MAX; // The fold API has no stage parameter.
    std::string revision;
    std::vector<uint32_t> code, user, system;
    uint32_t user_base = 0, absent_system_count = 0;
    bool user_present = true, system_present = false, srt_requested = false;
    bool dispatch_present = false, branch_exclusive_disabled = false;
    uint32_t dispatch_target = UINT32_MAX;
    PcrelDispatchInfo dispatch;
};

enum class FoldEventKind : uint32_t { Probe, Word, Prefix };
struct FoldEvent {
    FoldEventKind kind = FoldEventKind::Probe;
    FoldProbe probe = FoldProbe::Raw;
    uint32_t pc = 0, bytes = 0;
    uint64_t address = 0;
    bool readable = false;
    std::vector<uint8_t> data;
};

struct FoldCapture {
    FoldInputs input;
    bool complete = true;
    std::string error;
    std::vector<FoldEvent> events;
    std::vector<DynFetch> fetches;
    std::vector<SrtUse> uses;
    uint64_t evaluated_instructions = 0;
    uint32_t decoded_dwords = 0;
    bool shader_constant_specialized = false;
};

// Own code/register inputs; dynamic resource addresses remain logical identities. The default
// backing performs live reads. Supplying a backing is useful for deterministic independent tests.
FoldCapture capture_fold(FoldInputs input, FoldReader* backing = nullptr);
struct FoldReplayResult {
    std::vector<DynFetch> fetches;
    std::vector<SrtUse> uses;
    uint64_t evaluated_instructions = 0;
};
FoldReplayResult replay_fold(const FoldCapture& capture, bool compare_outputs = true);
void validate_fold_capture(const FoldCapture& capture);
void write_fold_capture(const std::filesystem::path& path, const FoldCapture& capture);
FoldCapture read_fold_capture(const std::filesystem::path& path);
std::vector<uint8_t> encode_fold_capture(const FoldCapture& capture);
FoldCapture decode_fold_capture(std::span<const uint8_t> bytes);

// Called by the production evaluator only when the capture environment switch is enabled.
// False means no evaluation occurred and the caller should take its normal path. Once evaluated,
// recording/I/O failure cannot cause a retry or publish a replacement empty result.
bool try_capture_live_fold(const uint32_t* code, size_t dwords,
    const uint32_t* user, uint32_t nuser, uint32_t user_base, std::vector<SrtUse>* uses,
    uint32_t target, const PcrelDispatchInfo* dispatch,
    const uint32_t* system, uint32_t nsystem, std::vector<DynFetch>& result);

} // namespace prosper::gpu
