// test_stat — guards the host struct stat -> guest FreeBSD SceKernelStat (0x78) translation.
// Getting this layout wrong once overran a guest stack buffer's canary (fstat crash). This locks
// in the critical offsets: st_mode@0x08 (16-bit), st_size@0x48 (64-bit), total size 0x78.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#endif

namespace prosper { void to_sce_stat(const struct stat& s, uint8_t* out); }
#ifdef _WIN32
namespace prosper { void to_sce_stat64(const struct _stat64& s, uint8_t* out); }
#endif

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_stat ==\n");
#ifndef _WIN32
    struct stat s; memset(&s, 0, sizeof s);
    s.st_mode  = S_IFREG | 0644;   // regular file, rw-r--r--
    s.st_size  = 0x123456789ULL;   // a >32-bit size to catch truncation/misplacement
    s.st_nlink = 1;
    s.st_blocks = 42;
    s.st_blksize = 4096;
#else
    struct _stat64 s; memset(&s, 0, sizeof s);
    s.st_mode  = S_IFREG | 0644;   // regular file, rw-r--r--
    s.st_size  = 0x123456789LL;    // a >32-bit size to catch truncation/misplacement
    s.st_nlink = 1;
#endif

    // Write into a canary-guarded buffer: 0x78 payload + a sentinel that must remain intact.
    uint8_t buf[0x80];
    memset(buf, 0xAB, sizeof buf);
    const uint8_t SENT = 0xAB;
#ifndef _WIN32
    prosper::to_sce_stat(s, buf);
#else
    prosper::to_sce_stat64(s, buf);
#endif

    CHECK(*(uint16_t*)(buf + 0x08) == (uint16_t)(S_IFREG | 0644), "st_mode at 0x08 (16-bit)");
    CHECK(*(int64_t*)(buf + 0x48)  == (int64_t)0x123456789ULL,   "st_size at 0x48 (full 64-bit, no truncation)");
#ifndef _WIN32
    CHECK(*(int64_t*)(buf + 0x50)  == 42,                        "st_blocks at 0x50");
    CHECK(*(uint32_t*)(buf + 0x58) == 4096,                      "st_blksize at 0x58");
#else
    CHECK(*(int64_t*)(buf + 0x50)  == ((int64_t)0x123456789LL + 511) / 512,
          "st_blocks at 0x50 synthesized from 64-bit size");
    CHECK(*(uint32_t*)(buf + 0x58) == 4096,                      "st_blksize at 0x58 synthesized");
#endif
    // Nothing past the documented 0x78 was touched (no canary smash).
    bool clean = true; for (int i = 0x78; i < 0x80; i++) if (buf[i] != SENT) clean = false;
    CHECK(clean, "no write past 0x78 (stack canary safe)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
