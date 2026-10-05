// test_local_fallback_import -- a module may ship a local body for a function it also imports (a
// `xor eax,eax; ret` placeholder). Its symbol then has a section index and a non-zero value, so the
// undefined-symbol test calls it an export and the linker binds the import to that placeholder:
// Tales of Graces f's sceKernelSyncOnAddressWait returned at once and every waiting thread spun.
// The classifier must keep such a symbol an import when it names a declared import library, and
// must leave real exports (export-library id, or no NID encoding) alone.
#include "self/module.hpp"
#include <gtest/gtest.h>

namespace {

prosper::Symbol make(const char* raw, int lib_id, uint64_t value, bool is_import = false) {
    prosper::Symbol s;
    s.raw = raw;
    s.nid = std::string(raw).substr(0, std::string(raw).find('#'));
    s.lib_id = lib_id;
    s.value = value;
    s.shndx = value ? 3 : 0;
    s.is_import = is_import;
    return s;
}

const std::map<int, std::string> kImportLibs = {{1, "libkernel_sync_on_address"}};

TEST(LocalFallbackImport, FallbackBodyWithDeclaredImportLibIsAnImport) {
    EXPECT_TRUE(prosper::is_relocated_local_fallback_import(make("Hc4CaR6JBL0#B#B", 1, 0x280),
                                                            kImportLibs));
}

TEST(LocalFallbackImport, RealExportLibraryIsNotReclassified) {
    // Library id 4 is not an import library of this module: a genuine export stays an export.
    EXPECT_FALSE(prosper::is_relocated_local_fallback_import(make("zlqfTyrQSPk#E#A", 4, 0x2a0),
                                                             kImportLibs));
}

TEST(LocalFallbackImport, AlreadyImportOrUnencodedSymbolIsUntouched) {
    EXPECT_FALSE(prosper::is_relocated_local_fallback_import(make("Hc4CaR6JBL0#B#B", 1, 0, true),
                                                             kImportLibs));
    prosper::Symbol plain;
    plain.raw = "memcpy";
    plain.nid = "memcpy";
    plain.value = 0x10;
    plain.lib_id = 1;
    EXPECT_FALSE(prosper::is_relocated_local_fallback_import(plain, kImportLibs));
}

}   // namespace
