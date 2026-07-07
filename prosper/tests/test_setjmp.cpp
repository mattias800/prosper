// test_setjmp — regression test for the real setjmp/longjmp implementation (hle_libc.cpp).
// The guest's Boehm GC uses setjmp to flush callee-saved registers for root scanning, and the
// runtime uses it for exception unwinding; a no-op stub (the old behaviour, returning 0) breaks
// both. Our impl is naked x86-64 asm that saves the SysV callee-saved set + rsp + return address.
// Because it's entered via a plain `call` here (as it is via the HLE tail-jump stub), it behaves
// exactly like C setjmp/longjmp. This test verifies the return-twice semantics, the value
// propagation, that longjmp never returns 0, and that callee-saved registers survive the jump.
#include <cstdio>
#include <cstdint>

extern "C" uint64_t prosper_setjmp(void*);
extern "C" void     prosper_longjmp(void*, uint64_t);

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); failures++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

int main() {
    printf("== test_setjmp ==\n");

    // 1) Basic return-twice: setjmp returns 0 first, then the longjmp value.
    {
        uint64_t buf[16] = {0};
        volatile int stage = 0;
        uint64_t r = prosper_setjmp(buf);
        if (r == 0) { stage = 1; prosper_longjmp(buf, 42); printf("  [FAIL] longjmp fell through\n"); failures++; }
        else {
            CHECK(r == 42, "longjmp value (42) propagates to setjmp return");
            CHECK(stage == 1, "stack locals survive the longjmp");
        }
    }

    // 2) longjmp(buf, 0) must be rewritten to 1 (setjmp never returns 0 on a real jump).
    {
        uint64_t buf[16] = {0};
        volatile int did = 0;
        uint64_t r = prosper_setjmp(buf);
        if (r == 0) { if (did++ == 0) prosper_longjmp(buf, 0); }
        else        CHECK(r == 1, "longjmp(buf,0) is rewritten to 1");
    }

    // 3) Callee-saved register preservation: put a sentinel in rbx before setjmp, clobber it,
    //    longjmp, and confirm the value is restored (this is exactly what the GC relies on).
    {
        uint64_t buf[16] = {0};
        volatile int pass = 0;
        register uint64_t rbx_val asm("rbx");
        (void)rbx_val;
        uint64_t saved;
        asm volatile("mov $0x1234567089abcdef, %%rbx\n\t" "mov %%rbx, %0" : "=r"(saved) :: "rbx");
        uint64_t r = prosper_setjmp(buf);
        if (r == 0) {
            asm volatile("mov $0, %%rbx" ::: "rbx");   // clobber rbx
            prosper_longjmp(buf, 7);
        } else {
            uint64_t now;
            asm volatile("mov %%rbx, %0" : "=r"(now));
            pass = (now == saved);
            CHECK(pass, "callee-saved rbx restored across longjmp");
        }
    }

    if (failures) { printf("== FAIL: %d check(s) failed ==\n", failures); return 1; }
    printf("== PASS ==\n");
    return 0;
}
