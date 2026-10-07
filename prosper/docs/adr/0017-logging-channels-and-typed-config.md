---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0017: Diagnostics as named log channels; settings as typed configuration

## Context

The shipping sources mention 992 distinct `"PROSPER_*"` names as string literals under `src/`,
`frontends/` and `tests/fixtures/` -- names, not a census of variables actually read -- counted by
`tools/env/check_switch_registry.py` (#4543). One mechanism carries three different things: host
capabilities and user settings, diagnostics, and temporary guest-behaviour selectors. Diagnostics
dominate, and each is its own switch with its own spelling; output goes through 1,547 direct
`fprintf(stderr, ...)` calls in `src/`, with no levels, no channel names and no single way to turn a
subsystem's output on.

Wine separates the same concerns: diagnostics are named channels with levels under one variable
(`WINEDEBUG=+relay,warn+heap`), and configuration is a typed store edited through one tool.
Structural reference only.

## Decision

1. **Diagnostics become channels.** A logging facade in `src/diagnostics/` with named channels
   (`gfx`, `agc`, `apr`, `relay`, ...) and levels (`err`, `warn`, `info`, `trace`), enabled by one
   variable, `PROSPER_DEBUG=gfx,agc:trace,-warn`. A disabled channel costs one predictable branch.
   Existing diagnostic switches migrate channel by channel; each migration deletes its switches and
   their registry rows.
2. **Settings become typed configuration.** Host-capability switches and user settings move to one
   schema (name, type, default, range, description) read from a config file, command-line flags,
   and environment overrides in that order of precedence, validated at start-up so a malformed
   value is reported, not silently ignored. The schema generates the `--help` text and the
   frontend's settings list.
3. **Selectors stay switches**, each with its retiring issue (spec `CFG-1`), because they are meant
   to disappear.
4. The switch registry (`tools/env/switch_registry.txt`, #4543) is the migration inventory: a row
   leaves when its switch does.

Adds spec rule `CFG-2`.

## Consequences

The configuration surface becomes discoverable and finite, a user report can state its whole
configuration in a few lines (ADR 0021), and enabling a subsystem's diagnostics no longer requires
knowing a dozen switch names. The cost is a long migration across hundreds of call sites; it is
incremental by design and never a flag day. Rate limits and the producer/printer split guarded by
`tools/env/check_diag_gates.py` must carry over to channels unchanged.

## Alternatives considered

- Keep per-diagnostic switches and only document them: the registry does that, and 992 entries
  remain 992 entries.
- Adopt an external logging library: possible for the formatting layer; the channel and
  configuration model is the decision here, not the library.

## Approval

Requires the project owner's acceptance; it changes how every diagnostic is enabled.
