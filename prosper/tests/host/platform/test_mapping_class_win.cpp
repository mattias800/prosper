// test_mapping_class_win -- the Windows backend of mapping_class.hpp against hand-built memory.
//
// The census is only as good as the answer to "is this address private or a section view?", and a
// control drawn from the census's own input could only show the plumbing runs. So build one of each
// kind outside it. A swap or a collapse of MEM_PRIVATE / MEM_MAPPED fails here.
#include "host/platform/mapping_class.hpp"

#include <gtest/gtest.h>
#include <windows.h>

using prosper::host::MappingClass;
using prosper::host::classify_host_mapping;

TEST(MappingClassWin, ClassifiesPrivateMemoryAndSectionViewsApart) {
    void* priv = VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    ASSERT_NE(priv, nullptr);
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT, 0, 65536, nullptr);
    ASSERT_NE(section, nullptr);
    void* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, 65536);
    ASSERT_NE(view, nullptr);

    EXPECT_EQ(classify_host_mapping(reinterpret_cast<uintptr_t>(priv)), MappingClass::Private);
    EXPECT_EQ(classify_host_mapping(reinterpret_cast<uintptr_t>(view)), MappingClass::MappedView);

    UnmapViewOfFile(view);
    CloseHandle(section);
    VirtualFree(priv, 0, MEM_RELEASE);
    // Freed memory is not committed host memory any more.
    EXPECT_EQ(classify_host_mapping(reinterpret_cast<uintptr_t>(priv)), MappingClass::Untracked);
}

TEST(MappingClassWin, ReservedButUncommittedMemoryIsNotPrivateCommitted) {
    // A reservation is MEM_PRIVATE to VirtualQuery; the classifier reports what the OS says rather than
    // guessing at commitment, which the census does not need.
    void* reserved = VirtualAlloc(nullptr, 65536, MEM_RESERVE, PAGE_NOACCESS);
    ASSERT_NE(reserved, nullptr);
    EXPECT_EQ(classify_host_mapping(reinterpret_cast<uintptr_t>(reserved)), MappingClass::Private);
    VirtualFree(reserved, 0, MEM_RELEASE);
}

TEST(MappingClassWin, ClassificationIsInformative) {
    // The census prints its class rows only where the classes distinguish something.
    EXPECT_TRUE(prosper::host::host_mapping_classification_informative());
}
