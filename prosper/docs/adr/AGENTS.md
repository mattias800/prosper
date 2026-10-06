# Architecture decision records

One file per architectural decision: the context, what was decided, its consequences, and the
alternatives rejected. The spec (`../spec/`) states the rules; an ADR is the record of why a rule
exists and who accepted it. Status docs and investigations record evidence; an ADR cites them.

## Lifecycle (checked by `tools/docs/check_arch_docs.py`)

- `proposed` -- written and open for discussion. A spec rule backed only by a proposed ADR is a
  target, not policy. **Only the project owner moves an ADR to `accepted`**; an agent never does,
  whatever its task says, because acceptance is the owner's approval of a direction (several
  moves here are explicitly gated on it).
- `accepted` -- binding. Its text is then frozen: CI rejects any change except to its `status`
  and `superseded-by` frontmatter lines, and rejects deleting it.
- `superseded` -- replaced by a later ADR named in `superseded-by: NNNN`. A changed decision is
  always a new ADR, so the history keeps saying what was decided and when.
- `rejected` -- considered and declined; kept so the question is not re-opened at full cost.

Files are `NNNN-kebab-slug.md` with an H1 of `# ADR NNNN: Title`. Take the next free number;
gaps are legal. Start from `0000-template.md`. ADRs 0001 and 0002 are backfills: they record
decisions that had already merged and are enforced by CI, and they cite the PRs that made them.
