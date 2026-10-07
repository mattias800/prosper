// test_mapping_class_posix -- the non-Windows backend of mapping_class.hpp. These platforms have the
// real write watch, so every address is `Other`; the test pins that the backend answers rather than
// leaving the class unset or claiming a Windows-only class.
#include "host/platform/mapping_class.hpp"

#include <gtest/gtest.h>

TEST(MappingClassPosix, EveryAddressIsOther) {
    int local = 0;
    EXPECT_EQ(prosper::host::classify_host_mapping(reinterpret_cast<uintptr_t>(&local)),
              prosper::host::MappingClass::Other);
    EXPECT_EQ(prosper::host::classify_host_mapping(0), prosper::host::MappingClass::Other);
}
