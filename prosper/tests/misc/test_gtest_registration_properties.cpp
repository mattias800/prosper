// test_gtest_registration_properties -- prosper_add_gtest (cmake/ProsperGTest.cmake) must deliver a
// LIST-valued test property to every discovered case intact (#4682).
//
// The registration in CMakeLists.txt passes ENVIRONMENT_MODIFICATION as one quoted three-element
// list. When the helper forwarded it through gtest_discover_tests(PROPERTIES ...), the list was split
// into separate words: CMake 3.31 then applied only the first element (the other two became a bogus
// property name and value), and CTest 4.4 segfaulted while reading the tests, so no test in the tree
// ran at all. Either way this case fails: here by the missing variables, there by the whole run.
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

namespace {

std::string env_or_missing(const char* name) {
    const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe): read-only, single thread
    return value ? std::string(value) : std::string("<missing>");
}

}   // namespace

TEST(GtestRegistrationProperties, EveryListElementReachesTheTest) {
    EXPECT_EQ(env_or_missing("GTEST_REGISTRATION_PROBE_A"), "alpha");
    EXPECT_EQ(env_or_missing("GTEST_REGISTRATION_PROBE_B"), "beta")
        << "only the first ENVIRONMENT_MODIFICATION element was applied: the list was split";
    EXPECT_EQ(env_or_missing("GTEST_REGISTRATION_PROBE_C"), "gamma delta")
        << "the last element (with a space) must arrive whole";
}
