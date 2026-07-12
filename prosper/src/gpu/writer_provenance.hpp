#pragma once

#include <cstdint>
#include <optional>

namespace prosper::gpu {

enum class GuestWriterKind : uint8_t {
    ColorTarget,
    ComputeBuffer,
    DmaData,
    WriteData,
};

struct GuestWriteEvent {
    uint64_t sequence = 0;
    GuestWriterKind kind = GuestWriterKind::ColorTarget;
    uint64_t addr = 0;
    uint64_t size = 0;
    uint64_t submit = 0;
    uint64_t item = 0;
    uint64_t order = 0;
    uint64_t identity = 0;
    uint32_t width = 0, height = 0;
};

// The opt-in live history is enabled by either the existing dimension probe or its explicit alias.
bool writer_provenance_enabled();
const char* guest_writer_kind_name(GuestWriterKind kind);

uint64_t record_guest_write(GuestWriterKind kind, uint64_t addr, uint64_t size,
                            uint64_t submit = 0, uint64_t item = 0, uint64_t order = 0,
                            uint64_t identity = 0,
                            uint32_t width = 0, uint32_t height = 0);
std::optional<GuestWriteEvent> last_guest_write_overlap(uint64_t addr, uint64_t size,
                                                        uint64_t before_sequence = UINT64_MAX);

// Test-only reset for the process-global bounded history.
void reset_guest_write_history_for_test();

} // namespace prosper::gpu
