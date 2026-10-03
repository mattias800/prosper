# IL2CPP offline tools

These tools flatten a compiled managed module for external analysis and resolve method names from
an exported symbol list. Runtime symbol loading and annotation live in `src/host/symbols`; ELF/SELF
mapping and loader behavior belong to their own subsystems.

The resolver and runtime consume the same ordered records and preserve the same selected symbol.
Use synthetic method tables to verify names, tie reporting, windows and UTF-8 ordering through the
actual interfaces, without requiring a game dump or external analysis program.
