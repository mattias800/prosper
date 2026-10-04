#pragma once

#include <cstdint>

namespace prosper::gpu {

// Nonminting activity at actual public graphics/compute entries. Starting another execution
// permanently invalidates earlier read-point permissions, including successful/reentrant work.
// This is NOT byte-write exclusion: no lock or wait protects a CPU memcpy against an overlapping
// writer. Submitted guest input must retain its existing live/stable-backing contract; normal
// HLE submits already serialize their execution. Concurrent direct backing writes remain unsupported.
class GraphicsExecutionActivity {
    friend struct OrderedGraphicsReadPointIssuer;
    friend class OrderedGraphicsReadPoint;
    friend class OrderedScalarBankReadPoint;
    static uint64_t current_version();

public:
    GraphicsExecutionActivity();
    ~GraphicsExecutionActivity();
    GraphicsExecutionActivity(const GraphicsExecutionActivity&) = delete;
    GraphicsExecutionActivity& operator=(const GraphicsExecutionActivity&) = delete;
};

}   // namespace prosper::gpu
