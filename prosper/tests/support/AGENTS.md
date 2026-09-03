# tests/support — helpers for the test SUITE, not for the emulator

Small, dependency-free headers that exist to make the tests themselves behave: process hygiene,
death-test plumbing, and anything else whose subject is *how a case runs* rather than what prosper
does.

The boundary against its siblings is the important part:

- **`tests/fixtures/`** holds test *inputs and machinery under test* — `render_runner.h` is the
  production render backend despite living there, and the `gta5_*` headers are captured guest state.
  Code in `fixtures/` participates in what is being asserted.
- **`tests/shared/`** mirrors `frontends/shared/` — frontend code the tests link, not test code.
- **`tests/support/`** participates in nothing. Nothing here should ever be able to change whether
  an assertion passes; if a helper could, it belongs in `fixtures/`.

Keep these header-only and free of prosper includes, so any target can pick one up with
`target_include_directories(<target> PRIVATE tests)` and no link-order thought.
