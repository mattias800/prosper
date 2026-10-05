// test_local_fallback_import -- a module may ship a local body for a function it also imports (a
// `xor eax,eax; ret` placeholder). Its symbol then has a section index and a non-zero value, so the
// undefined-symbol test calls it an export and the linker binds the import to that placeholder:
// Tales of Graces f's sceKernelSyncOnAddressWait returned at once and every waiting thread spun.
// The classifier must keep such a symbol an import when it names a declared import library, and
// must leave real exports (export-library id, or no NID encoding) alone.
#include "loader/linker.hpp"
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

// One LOAD segment mapping va 0.. onto `bytes`.
prosper::Module module_with(std::vector<uint8_t> bytes) {
    prosper::Module m;
    m.file = std::move(bytes);
    prosper::Segment seg;
    seg.type = 1;
    seg.vaddr = 0;
    seg.filesz = seg.memsz = m.file.size();
    seg.file_off = 0;
    m.loads.push_back(seg);
    return m;
}

TEST(ReturnZeroPlaceholder, DirectAndTrampolinedBodiesAreRecognised) {
    // 0x00: xor eax,eax; ret    0x10: jmp 0x00    0x20: jmp 0x10
    std::vector<uint8_t> b(0x30, 0xcc);
    b[0] = 0x31;
    b[1] = 0xc0;
    b[2] = 0xc3;
    b[0x10] = 0xe9;
    b[0x11] = 0xeb;
    b[0x12] = 0xff;
    b[0x13] = 0xff;
    b[0x14] = 0xff;   // -0x15
    b[0x20] = 0xe9;
    b[0x21] = 0xeb;
    b[0x22] = 0xff;
    b[0x23] = 0xff;
    b[0x24] = 0xff;   // 0x25-0x15=0x10
    auto m = module_with(b);
    EXPECT_TRUE(prosper::is_return_zero_placeholder(m, 0x00));
    EXPECT_TRUE(prosper::is_return_zero_placeholder(m, 0x10));
    EXPECT_TRUE(prosper::is_return_zero_placeholder(m, 0x20));
}

TEST(ReturnZeroPlaceholder, RealBodiesAndLoopsAreNot) {
    std::vector<uint8_t> b(0x30, 0xcc);
    b[0] = 0xb8;
    b[1] = 0x01;
    b[5] = 0xc3;   // mov eax,1; ret
    b[0x10] = 0xe9;
    b[0x11] = 0xfb;
    b[0x12] = 0xff;
    b[0x13] = 0xff;
    b[0x14] = 0xff;   // jmp self
    auto m = module_with(b);
    EXPECT_FALSE(prosper::is_return_zero_placeholder(m, 0x00));
    EXPECT_FALSE(prosper::is_return_zero_placeholder(m, 0x10));
    EXPECT_FALSE(prosper::is_return_zero_placeholder(m, 0x1000));   // unmapped
}

}   // namespace
