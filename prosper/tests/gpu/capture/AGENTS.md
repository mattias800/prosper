# Capture tests

CPU contracts for capture serialization, owned replay backing and observation-only reports.
Fixtures own their bytes and use synthetic addresses; a successful metadata report cannot authorize
rendering or raw recompilation. CLI cases launch the actual replay tool with bounded synthetic inputs.

`test_rtt_dcc_codec` checks the append-only association layout, legacy prefixes, absent facts
and malformed records. These are format/metadata assertions, with no rendering authority.
