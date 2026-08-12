# Diagnostics Layer — Minimal Boot Timeline

**Status:** Optional, observer-only instrumentation (OFF by default)

## Overview

This directory contains a minimal diagnostics layer for capturing boot timeline events during PS4 program startup. It is designed to have **zero overhead when disabled** — all calls compile to inline no-ops.

## Architecture

```
src/diagnostics/
├── diagnostics.hpp          # Single include entry point (stubs when disabled)
├── core/
│   ├── types.hpp            # BootPhase enum, BootEvent struct
│   ├── event_bus.hpp/.cpp   # Thread-safe pub/sub event bus
│   └── context.hpp/.cpp     # DiagnosticContext singleton coordinator
└── storage/
    └── json_writer.hpp/.cpp # JSON output for boot events
```

## Building

### Default (Disabled)

```bash
cmake -B build -S prosper
cmake --build build
```

No diagnostics code is compiled. The `diagnostics.hpp` header provides stubs that optimize to nothing.

### Enabled

```bash
cmake -B build -S prosper -DPROSPER_DIAGNOSTICS=ON
cmake --build build
```

Compiles ~300 lines of optional code: EventBus, DiagnosticContext, and JsonWriter.

## Usage

```cpp
#include "diagnostics/diagnostics.hpp"

// Enable recording (optional — can be enabled at runtime)
prosper::diagnostics::DiagnosticContext::instance().enable();

// Record boot phases (called from boot_program.cpp automatically)
prosper::diagnostics::record_boot_phase(prosper::diagnostics::BootPhase::PROCESS_START);

// Export as JSON
auto json = prosper::diagnostics::JsonWriter::write_events(
    prosper::diagnostics::DiagnosticContext::instance().events()
);
```

## Boot Phases

| Phase | Description |
|-------|-------------|
| `PROCESS_START` | `boot_program()` entered |
| `LINKING` | `link_program()` completed |
| `HLE_REGISTERED` | `register_builtin_hle()` complete |
| `MODULES_MAPPED` | All images mapped into guest VA |
| `STUBS_INSTALLED` | Trap handler + stubs active |
| `GUEST_INITS_RUNNING` | `run_guest_inits()` called |
| `BOOT_COMPLETE` | `boot_program()` returning true |

## Output Format

JSON array of timestamped phase events:

```json
[
  {"phase": "PROCESS_START", "timestamp_ms": 0.00},
  {"phase": "LINKING", "timestamp_ms": 1.23},
  {"phase": "BOOT_COMPLETE", "timestamp_ms": 45.67}
]
```

## Tests

Three tests verify correctness:

1. **test_diagnostics_disabled** — Verifies stub behavior (always runs)
2. **test_diagnostics_enabled** — Verifies event capture (requires `-DPROSPER_DIAGNOSTICS=ON`)
3. **test_diagnostics_json** — Verifies JSON output format (requires `-DPROSPER_DIAGNOSTICS=ON`)

Run with: `ctest --test-dir build -R diagnostics`

## Integration Point

The **only modification to existing code** is in `src/host/boot_program.cpp`, which includes `diagnostics/diagnostics.hpp` and calls `record_boot_phase()` at each boot milestone. When diagnostics are disabled, these calls are inlined no-ops with zero runtime cost.

## Design Principles

1. **Observer-only**: Does not alter program behavior; only records events
2. **Zero-cost disabled path**: Stubs compile to empty functions
3. **Thread-safe**: Mutex-protected event bus and context
4. **Opt-in**: CMake option required; off by default
5. **Minimal scope**: Boot timeline only (~300 lines)
