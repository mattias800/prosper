# GoogleTest (vendored)

GoogleTest and GoogleMock **v1.18.0**, vendored **verbatim** from https://github.com/google/googletest
(release tag `v1.18.0`, archive sha256 `6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5`).

- **License:** BSD-3-Clause (see `LICENSE`). Copyright 2008 Google Inc.
- **Why vendored:** the test framework for prosper's C++ tests (`prosper/cmake/ProsperGTest.cmake`,
  `prosper_add_gtest()`). Vendored rather than downloaded at configure time so offline builds,
  release builds and every CI platform configure without a network fetch. A general-purpose,
  permissively licensed library with no Sony code; a deliberate project-owner decision under the
  charter's vendoring exception.
- **Kept:** the top-level `CMakeLists.txt`, `LICENSE`, and the `googletest/` and `googlemock/` trees.
  **Removed:** upstream's own tests, samples and docs (`*/test`, `googletest/samples`, `*/docs`), Bazel
  files, CI config. Upstream gates the tests and samples behind `gtest_build_tests` /
  `gmock_build_tests` / `gtest_build_samples`, all OFF by default, so nothing here references them.

Do not edit these files; keep them byte-identical to the release so the copy stays auditable. To
update: replace the kept files from a newer release tag, record its version and archive sha256 here.
