// test_data.h -- locate prosper/tests/data from a test source, at any folder depth.
//
// Several tests read a recorded guest program out of tests/data by building the path from
// `std::filesystem::path(__FILE__).parent_path()`. That works exactly while every test sits
// directly in tests/, and silently stops the moment one moves into a folder: the path becomes
// tests/<folder>/data/..., the ifstream fails, and the test reports a fixture of zero dwords rather
// than a missing file. Three of the four that broke this way SEGFAULTed on the empty vector
// afterwards, which is a long way from "the data file moved".
//
// Walking UP to the directory that actually contains `data` is depth-independent, so a test can be
// filed anywhere under tests/ without its fixture path being part of the decision.

#pragma once

#include <filesystem>

namespace prosper::test {

inline std::filesystem::path tests_root(const std::filesystem::path& source_file) {
    std::error_code ec;
    for (auto dir = source_file.parent_path(); !dir.empty(); dir = dir.parent_path()) {
        if (std::filesystem::is_directory(dir / "data", ec)) {
            return dir;
        }
    }
    // Preserve the old behaviour's failure path rather than inventing one: callers already report a
    // readable-fixture check, and that message is more useful than an exception from here.
    return source_file.parent_path();
}

}  // namespace prosper::test
