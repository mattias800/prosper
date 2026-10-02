# tests/

Unit and integration tests. **New tests use GoogleTest** (`TEST`/`TEST_F`, `EXPECT_*`/`ASSERT_*`),
registered with `prosper_add_gtest(<target> SOURCES ... INCLUDES ... LIBRARIES ...)` from
`cmake/ProsperGTest.cmake`. Each `TEST` becomes its own ctest case (`Suite.Name`).

- Do not write a new file with its own `int main()` and a home-made `CHECK` macro. The ctest
  `gtest_policy` (`tools/ci/check_gtest_policy.py`) fails on it.
- Legacy hand-rolled tests are listed in `tools/ci/gtest_legacy_allowlist.txt`. When you migrate one,
  delete its line (the check fails if a listed file no longer has `main()`).
- Prefer one `TEST` per behaviour with a message on `EXPECT_*` (`<< "why"`), not one giant test.
- Tests that need standalone compilation (e.g. mutation scripts compiling with plain g++) may stay legacy.
- Tests that need a game dump or Vulkan keep their existing skip/gating logic; gate them in CMake, not in gtest.
