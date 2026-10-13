# `tests/gpu/pm4` — guest packets and the command processor

These CPU tests pin guest packet sizes and operands, the registered AGC builder/submit boundary,
and the command processor's real fold, retention and memory-write paths. Use synthetic streams and
owned labels; a live device or game dump belongs in the separate execution/title tests.

Diagnostic fixtures here exercise actual enqueue/overlay/drain wiring. They must retain unknown
identity and distinguish attempted effects from stores, without supplying predicate authority.
