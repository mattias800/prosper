#pragma once

#include "diagnostics/diag_clock.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace prosper::diagnostics::readback_refusal {

enum class ObservedBool : uint8_t { NotObserved, No, Yes };
enum class Reason : uint8_t {
    NotRefused, GlobalUnproven, LookupNotPerformed, ExactKeyAbsent, PresentInvalid, NoImage,
    UndefinedLayout
};
enum class Caller : uint8_t { Unknown, ComputeSnapshot, LazySampled };

// Plain values, passed by the caller. A zero identifier without its known bit is not a witness.
enum ContextField : uint32_t { SubmitKnown = 1u, DrawKnown = 2u, ProgramKnown = 4u };
struct Context {
    Caller caller;
    uint32_t known;
    uint64_t submit;
    uint64_t draw;
    uint64_t program;
    ObservedBool frontend_gpu_valid;
    // Shared-vector presence only; not exact size, current bytes or CPU producer authority.
    ObservedBool frontend_cpu_pixels;
};

// Trivial POD: callers may leave storage untouched when they pass nullptr.
struct LookupObservation {
    ObservedBool key_present;
    ObservedBool valid;
};
struct Record {
    Reason reason;
    uint64_t source_us;
    uint64_t address;
    uint32_t width;
    uint32_t height;
    uint32_t volume_depth;
    uint32_t requested_format;
    uint32_t canonical_format;
    bool canonical_known;
    LookupObservation lookup;
    ObservedBool image_present;
    uint32_t layout;
    bool layout_known;
    Context context;
};

inline void stamp_refusal(Record* record, Reason reason) {
    if (!record) return;
    record->reason = reason;
    record->source_us = diag_now_us();
}

// Initialize at live-renderer registration, outside resource/queue locks and before guest work.
// Environment selection is immutable thereafter. This does not initialize from a backend hook.
void initialize();
bool selected(uint64_t address);
const char* reason_name(Reason reason);
// Formatting and output happen in the frontend after the backend helper returns, never inside
// a counted submission region. No lifecycle, producer-token or final-publication proof is added.
// Returns the required length, as snprintf does; NotRefused produces an empty string.
size_t format_record(char* output, size_t capacity, const Record& record);
void emit(const Record& record, FILE* stream = stderr);

} // namespace prosper::diagnostics::readback_refusal
