# Architecture specification

The binding architecture rules: what code in this repository MUST and MUST NOT do, each with a
stable ID and the instrument that enforces it. Explanation of how things work today belongs in the
topic folders (`gpu/`, `subsystems/`, `architecture/`); why a rule was chosen belongs in an ADR
(`../adr/`); a dated measurement belongs in a status doc or an investigation. A sentence here is a
rule, or it is out of place.

## Format (checked by `tools/docs/check_arch_docs.py`)

- Frontmatter: `kind: spec`, `status` (`accepted` or `draft`), `owner` (an `area:` label),
  `last-verified` (date and the short SHA the claims were checked against).
- One rule per `### <ID> -- <title>` heading. IDs are `PREFIX-N` (`LAY-3`, `PERF-P1`), unique
  across this folder, and never reused: a retired rule keeps its ID with `Status: retired` text
  in the ADR that retired it, never a new meaning.
- Each rule has a `Status:` line: `accepted`, or `proposed (adr:NNNN)` naming the ADR that would
  make it binding. A proposed rule is a target, not policy, until that ADR is accepted.
- Each rule has an `Enforcement:` line built from these tokens:
  `ratchet:<rule>` (a rule in `tools/ci/check_arch_ratchet.py`), `ci:<step>`, `ctest:<name>`,
  `runtime:<signal>` (a perf-alarm rule or other always-on runtime check), `adr:NNNN`, and
  `review: (<why no tool can check this>)`. Every ratchet rule must be cited by some spec rule.
- No measurements in rule text. A figure that changes weekly belongs in the ratchet baseline or a
  status doc; cite where it lives instead of copying it.
- RFC 2119 words (MUST, MUST NOT, SHOULD) carry their RFC meaning; avoid them in prose that is not a rule.

## Files

| file | rules |
| --- | --- |
| `architecture.md` | the target shape: layers and engines, the invariants that hold across them, and the proposed rules each ADR would add |
| `layers.md` | the include order (generated from `LAYER_ORDER`) and what each layer owns |
| `performance.md` | the steady-state frame invariants P1-P6 |
