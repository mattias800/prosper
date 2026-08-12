# Design Sample — Illustrative Boot Timeline Output

> **Note:** This document contains **illustrative examples only**. These are not real measurements from any specific title or hardware configuration. Actual timestamps vary by host hardware, game dump, and system load.

## Sample JSON Output

```json
[
  {"phase": "PROCESS_START", "timestamp_ms": 0.00},
  {"phase": "LINKING", "timestamp_ms": 2.15},
  {"phase": "HLE_REGISTERED", "timestamp_ms": 2.87},
  {"phase": "MODULES_MAPPED", "timestamp_ms": 8.42},
  {"phase": "STUBS_INSTALLED", "timestamp_ms": 9.01},
  {"phase": "GUEST_INITS_RUNNING", "timestamp_ms": 9.15},
  {"phase": "BOOT_COMPLETE", "timestamp_ms": 47.83}
]
```

## Phase Duration Breakdown

| Phase | Duration (ms) | Cumulative (ms) |
|-------|---------------|-----------------|
| PROCESS_START → LINKING | ~2.15 | 2.15 |
| LINKING → HLE_REGISTERED | ~0.72 | 2.87 |
| HLE_REGISTERED → MODULES_MAPPED | ~5.55 | 8.42 |
| MODULES_MAPPED → STUBS_INSTALLED | ~0.59 | 9.01 |
| STUBS_INSTALLED → GUEST_INITS_RUNNING | ~0.14 | 9.15 |
| GUEST_INITS_RUNNING → BOOT_COMPLETE | ~38.68 | 47.83 |

## Usage Example

```cpp
#include "diagnostics/diagnostics.hpp"
#include <iostream>

void capture_boot_timeline() {
    auto& ctx = prosper::diagnostics::DiagnosticContext::instance();
    ctx.enable();
    ctx.clear();

    // ... boot_program() runs here, recording phases ...

    // Export timeline
    std::string json = prosper::diagnostics::JsonWriter::write_events(ctx.events());
    std::cout << json << std::endl;

    // Or iterate events
    for (const auto& ev : ctx.events()) {
        printf("%s: %.3f ms\n",
               prosper::diagnostics::phase_name(ev.phase),
               ev.timestamp_ms);
    }
}
```

## EventBus Subscription Example

```cpp
#include "diagnostics/diagnostics.hpp"

void setup_live_monitor() {
    auto handle = prosper::diagnostics::event_bus().subscribe(
        [](const prosper::diagnostics::BootEvent& ev) {
            printf("[boot] %s at %.3f ms\n",
                   prosper::diagnostics::phase_name(ev.phase),
                   ev.timestamp_ms);
        });

    // Events now stream to callback in real-time
    // Unsubscribe when done:
    // prosper::diagnostics::event_bus().unsubscribe(handle);
}
```
