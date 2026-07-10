// hle_libc.cpp — HLE implementations of the common libc functions the guest imports.
// The guest ABI == host SysV ABI, so most of these are thin thunks straight to the
// host C library. Registered by NID so the loader binds imports directly to them.
#include "dispatch.hpp"
#include "nid.hpp"
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cctype>
#include <cmath>
#include <cwchar>
#include <cerrno>
#include <atomic>
#include <mutex>
#include <condition_variable>
#ifdef _WIN32
#include <malloc.h>   // _aligned_malloc
#endif

// setjmp/longjmp — the guest's Boehm GC calls setjmp to flush callee-saved registers to a
// buffer so it can scan them as GC roots (and the runtime uses it for exception unwinding).
// A stub returning 0 silently breaks that. We implement the real thing: because our HLE
// stubs *tail-jump* into the handler (movabs rax,fn; jmp rax — rsp and all callee-saved regs
// are exactly the guest caller's), setjmp entered here sees the caller's true context. We
// save only the SysV callee-saved set + rsp + return address (64 bytes) — well within the
// guest's FreeBSD jmp_buf — and never touch the signal mask (matched pair, self-consistent).
#if defined(__linux__) && defined(__x86_64__)
extern "C" uint64_t prosper_setjmp(void*);
extern "C" void     prosper_longjmp(void*, uint64_t);
__asm__(
    ".text\n"
    ".globl prosper_setjmp\n.p2align 4\n"
    "prosper_setjmp:\n"
    "    movq (%rsp), %rax\n"        // guest return address (stub jmp'd here, so [rsp]=caller ret)
    "    movq %rbx,  0(%rdi)\n"
    "    movq %rbp,  8(%rdi)\n"
    "    movq %r12, 16(%rdi)\n"
    "    movq %r13, 24(%rdi)\n"
    "    movq %r14, 32(%rdi)\n"
    "    movq %r15, 40(%rdi)\n"
    "    leaq 8(%rsp), %rcx\n"        // caller's rsp (after our eventual ret)
    "    movq %rcx, 48(%rdi)\n"
    "    movq %rax, 56(%rdi)\n"
    "    xorl %eax, %eax\n"           // first return: 0
    "    ret\n"
    ".globl prosper_longjmp\n.p2align 4\n"
    "prosper_longjmp:\n"
    "    movq  0(%rdi), %rbx\n"
    "    movq  8(%rdi), %rbp\n"
    "    movq 16(%rdi), %r12\n"
    "    movq 24(%rdi), %r13\n"
    "    movq 32(%rdi), %r14\n"
    "    movq 40(%rdi), %r15\n"
    "    movq 48(%rdi), %rsp\n"
    "    movq %rsi, %rax\n"           // return the longjmp value...
    "    testq %rax, %rax\n"
    "    jnz 1f\n"
    "    movl $1, %eax\n"             // ...but never 0 (setjmp must see nonzero)
    "1:  jmp *56(%rdi)\n"
);
#else
extern "C" uint64_t prosper_setjmp(void*)          { return 0; }
extern "C" void     prosper_longjmp(void*, uint64_t) {}
#endif

namespace prosper {
// Portable aligned allocation (POSIX posix_memalign / Windows _aligned_malloc).
// NOTE: on Windows these need _aligned_free; the guest doesn't run on Windows yet,
// so h_free's plain free() is fine for now (Linux is the runtime target).
static void* aligned_alloc_portable(size_t align, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void* p = nullptr;
    return posix_memalign(&p, align, size) == 0 ? p : nullptr;
#endif
}
} // namespace prosper

namespace prosper {

// All handlers use the full 6-arg HLE signature; extras are ignored (SysV-safe).
#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)
#define P(x) ((void*)(uintptr_t)(x))
#define CP(x) ((const void*)(uintptr_t)(x))
#define CS(x) ((const char*)(uintptr_t)(x))

HLE(h_memcpy)  { return (uint64_t)(uintptr_t)memcpy(P(a0), CP(a1), a2); }
HLE(h_memmove) { return (uint64_t)(uintptr_t)memmove(P(a0), CP(a1), a2); }
HLE(h_memset)  { return (uint64_t)(uintptr_t)memset(P(a0), (int)a1, a2); }
HLE(h_memcmp)  { return (uint64_t)(int64_t)memcmp(CP(a0), CP(a1), a2); }
HLE(h_memchr)  { return (uint64_t)(uintptr_t)memchr(CP(a0), (int)a1, a2); }
HLE(h_strlen)  { return (uint64_t)strlen(CS(a0)); }
HLE(h_strnlen) { return (uint64_t)strnlen(CS(a0), a1); }
HLE(h_strcmp)  { return (uint64_t)(int64_t)strcmp(CS(a0), CS(a1)); }
HLE(h_strncmp) { return (uint64_t)(int64_t)strncmp(CS(a0), CS(a1), a2); }
HLE(h_strcpy)  { return (uint64_t)(uintptr_t)strcpy((char*)P(a0), CS(a1)); }
HLE(h_strncpy) { return (uint64_t)(uintptr_t)strncpy((char*)P(a0), CS(a1), a2); }
HLE(h_strcat)  { return (uint64_t)(uintptr_t)strcat((char*)P(a0), CS(a1)); }
HLE(h_strncat) { return (uint64_t)(uintptr_t)strncat((char*)P(a0), CS(a1), a2); }
HLE(h_strchr)  { return (uint64_t)(uintptr_t)strchr(CS(a0), (int)a1); }
HLE(h_strrchr) { return (uint64_t)(uintptr_t)strrchr(CS(a0), (int)a1); }
HLE(h_strstr)  { return (uint64_t)(uintptr_t)strstr(CS(a0), CS(a1)); }
// BSD strlcpy/strlcat: bounded, always NUL-terminate; return the length they tried to build.
HLE(h_strlcpy) {
    const char* s = CS(a1); size_t n = a2, sl = strlen(s);
    if (n) { size_t c = sl < n - 1 ? sl : n - 1; memcpy(P(a0), s, c); ((char*)P(a0))[c] = 0; }
    return sl;
}
HLE(h_strlcat) {
    char* d = (char*)P(a0); const char* s = CS(a1); size_t n = a2;
    size_t dl = strnlen(d, n), sl = strlen(s);
    if (dl < n) { size_t c = sl < n - dl - 1 ? sl : n - dl - 1; memcpy(d + dl, s, c); d[dl + c] = 0; }
    return dl + sl;
}
// C11 Annex-K bounded functions (errno_t return; 0 == success), arg order (dst, dstsz, src[, n]). These
// were MISSING -> the return-0 stub reported SUCCESS without copying, leaving the destination stale/
// uninitialized -> silent data corruption far from the call site. Real, bounds-checked impls that never
// write past dstsz (ERANGE=34 on overflow, zeroing the dst for the mem* forms per Annex-K).
HLE(h_memcpy_s)  { size_t ds=a1,n=a3; if (n>ds) { memset(P(a0),0,ds); return 34; } memcpy(P(a0),CP(a2),n); return 0; }
HLE(h_memmove_s) { size_t ds=a1,n=a3; if (n>ds) { memset(P(a0),0,ds); return 34; } memmove(P(a0),CP(a2),n); return 0; }
HLE(h_memset_s)  { size_t ds=a1,n=a3; size_t c=n<ds?n:ds; memset(P(a0),(int)a2,c); return n>ds?34:0; }
HLE(h_strcpy_s)  { char* d=(char*)P(a0); const char* s=CS(a2); size_t ds=a1,i=0; if (ds){ while(i<ds-1&&s[i]){d[i]=s[i];i++;} d[i]=0; } return 0; }
HLE(h_strncpy_s) { char* d=(char*)P(a0); const char* s=CS(a2); size_t ds=a1,n=a3,lim=(ds?ds-1:0); if(n<lim)lim=n; size_t i=0; if (ds){ while(i<lim&&s[i]){d[i]=s[i];i++;} d[i]=0; } return 0; }
HLE(h_strcat_s)  { char* d=(char*)P(a0); const char* s=CS(a2); size_t ds=a1,dl=strnlen(d,ds),i=0; if (dl<ds){ while(dl+i<ds-1&&s[i]){d[dl+i]=s[i];i++;} d[dl+i]=0; } return 0; }
HLE(h_strncat_s) { char* d=(char*)P(a0); const char* s=CS(a2); size_t ds=a1,n=a3,dl=strnlen(d,ds),i=0; if (dl<ds){ while(dl+i<ds-1&&i<n&&s[i]){d[dl+i]=s[i];i++;} d[dl+i]=0; } return 0; }

HLE(h_malloc)  { return (uint64_t)(uintptr_t)malloc(a0); }
HLE(h_calloc)  { return (uint64_t)(uintptr_t)calloc(a0, a1); }
HLE(h_realloc) { return (uint64_t)(uintptr_t)realloc(P(a0), a1); }
HLE(h_free)    { free(P(a0)); return 0; }
// memalign(alignment, size): aligned allocation. Normalize alignment to a valid
// power-of-two >= sizeof(void*) for posix_memalign.
HLE(h_memalign) {
    uint64_t al = a0 < sizeof(void*) ? sizeof(void*) : a0;
    if (al & (al - 1)) { uint64_t p = sizeof(void*); while (p < al) p <<= 1; al = p; } // round up to pow2
    return (uint64_t)(uintptr_t)aligned_alloc_portable(al, a1);
}
HLE(h_posix_memalign) {
    // POSIX: EINVAL if the alignment isn't a power of two AND a multiple of sizeof(void*); ENOMEM only on
    // an actual allocation failure. We returned ENOMEM for the bad-alignment case too, so a guest that
    // branches on errno==EINVAL (validating its own request) took the wrong path.
    if (a1 < sizeof(void*) || (a1 & (a1 - 1))) return 22;   // EINVAL
    void* p = aligned_alloc_portable(a1, a2);
    if (!p) return 12; // ENOMEM
    *(void**)P(a0) = p;
    return 0;
}
HLE(h_aligned_alloc)  { return (uint64_t)(uintptr_t)aligned_alloc_portable(a0, a1); }
// C++ operators new/delete (the whole IL2CPP game is C++). new -> malloc; the aligned
// forms take (size, align); nothrow forms take an extra tag arg we ignore.
HLE(h_new)         { return (uint64_t)(uintptr_t)malloc(a0 ? a0 : 1); }
HLE(h_new_align)   { return (uint64_t)(uintptr_t)aligned_alloc_portable(a1 ? a1 : 16, a0 ? a0 : 1); }
HLE(h_delete)      { free(P(a0)); return 0; }

// --- stdio ---
// v*printf receive a guest-built va_list (a pointer to __va_list_tag under the SysV
// ABI, which the host shares) — we can forward it directly.
// --- byte ops / search (integer/pointer args → plain HLE thunks) ---
HLE(h_bcmp)    { return (uint64_t)(int64_t)memcmp(CP(a0), CP(a1), a2); }   // bcmp == memcmp for equality
HLE(h_bsearch) { // (key, base, nmemb, size, compar) — compar is a guest fn ptr; SysV ABI matches host
    return (uint64_t)(uintptr_t)bsearch(CP(a0), CP(a1), a2, a3,
                                        (int (*)(const void*, const void*))(uintptr_t)a4); }
// --- integer conversion, sort, tokenize, wide/scan. All confirmed target imports that were MISSING
// (routed to the return-0 stub) -> returned 0/garbage and left endptr/output args unwritten on parsing
// paths (localization, save/config, gameplay data). Plain host thunks; guest pointers are identity-mapped
// so endptr and buffers pass through directly. (strtod/strtof return in XMM -> native thunks, below.) ---
HLE(h_strtol)   { return (uint64_t)(int64_t)strtol(CS(a0), (char**)P(a1), (int)a2); }
HLE(h_strtoll)  { return (uint64_t)(int64_t)strtoll(CS(a0), (char**)P(a1), (int)a2); }
HLE(h_strtoul)  { return (uint64_t)strtoul(CS(a0), (char**)P(a1), (int)a2); }
HLE(h_strtoull) { return (uint64_t)strtoull(CS(a0), (char**)P(a1), (int)a2); }
HLE(h_atoi)     { return (uint64_t)(int64_t)atoi(CS(a0)); }
HLE(h_atol)     { return (uint64_t)(int64_t)atol(CS(a0)); }
// rand/srand/rand_r were MISSING -> the return-0 stub made rand() a constant 0 and srand() a no-op, so any
// guest RNG routed through libc (procedural effects, shuffles, jitter, retry backoff) was degenerate.
HLE(h_rand)     { return (uint64_t)(int64_t)rand(); }
HLE(h_srand)    { srand((unsigned)a0); return 0; }
HLE(h_rand_r)   { return (uint64_t)(int64_t)rand_r((unsigned*)P(a0)); }
// qsort: the comparator is a guest fn ptr; SysV ABI matches host, callable directly (cf. h_bsearch).
HLE(h_qsort)    { qsort(P(a0), a1, a2, (int (*)(const void*, const void*))(uintptr_t)a3); return 0; }
HLE(h_strdup)   { const char* s = CS(a0); size_t n = strlen(s) + 1; void* p = malloc(n); if (p) memcpy(p, s, n); return (uint64_t)(uintptr_t)p; }
HLE(h_strtok)   { return (uint64_t)(uintptr_t)strtok((char*)P(a0), CS(a1)); }
HLE(h_strspn)   { return (uint64_t)strspn(CS(a0), CS(a1)); }
HLE(h_strcspn)  { return (uint64_t)strcspn(CS(a0), CS(a1)); }
HLE(h_strpbrk)  { return (uint64_t)(uintptr_t)strpbrk(CS(a0), CS(a1)); }
HLE(h_wcslen)   { return (uint64_t)wcslen((const wchar_t*)P(a0)); }   // host wchar_t is 32-bit == PS5/FreeBSD ABI
HLE(h_vsscanf)  { va_list ap; if (a2) memcpy(&ap, P(a2), sizeof(va_list)); return (uint64_t)(int64_t)vsscanf(CS(a0), CS(a1), ap); }
// _init_env / malloc_stats_fast: legitimately no-ops here (no PS5 process env vars; no malloc stats
// sink). Registered so they resolve as real, intentional no-ops rather than logged "unimplemented".
HLE(h_init_env)         { return 0; }
HLE(h_malloc_stats_fast){ return 0; }
// __error / __errno_location: FreeBSD/POSIX errno accessor — returns the address of the current
// thread's errno. Guest threads ARE host pthreads, so the host's thread-local &errno is exactly
// right, and it stays consistent with the errno our file/mem HLE thunks set. (A null stub here
// would make the guest deref a null errno pointer.)
HLE(h_errno_location)   { return (uint64_t)(uintptr_t)&errno; }

// --- math (float/double args + returns travel in XMM regs). The HLE stub tail-jumps preserving
// every register, so a handler with the correct native signature reads/writes the right regs and
// is an exact host thunk. Registered by name; only the ones the guest imports actually bind. ---
static float  m_sinf(float x){return sinf(x);}   static double m_sin(double x){return sin(x);}
static float  m_cosf(float x){return cosf(x);}   static double m_cos(double x){return cos(x);}
static float  m_tanf(float x){return tanf(x);}   static double m_tan(double x){return tan(x);}
static float  m_asinf(float x){return asinf(x);} static double m_asin(double x){return asin(x);}
static float  m_acosf(float x){return acosf(x);} static double m_acos(double x){return acos(x);}
static float  m_atanf(float x){return atanf(x);} static double m_atan(double x){return atan(x);}
static float  m_expf(float x){return expf(x);}   static double m_exp(double x){return exp(x);}
static float  m_exp2f(float x){return exp2f(x);} static double m_exp2(double x){return exp2(x);}
static float  m_logf(float x){return logf(x);}   static double m_log(double x){return log(x);}
static float  m_log10f(float x){return log10f(x);}static double m_log10(double x){return log10(x);}
static float  m_log2f(float x){return log2f(x);} static double m_log2(double x){return log2(x);}
static float  m_sqrtf(float x){return sqrtf(x);} static double m_sqrt(double x){return sqrt(x);}
static float  m_cbrtf(float x){return cbrtf(x);} static double m_cbrt(double x){return cbrt(x);}
static float  m_floorf(float x){return floorf(x);}static double m_floor(double x){return floor(x);}
static float  m_ceilf(float x){return ceilf(x);} static double m_ceil(double x){return ceil(x);}
static float  m_roundf(float x){return roundf(x);}static double m_round(double x){return round(x);}
static float  m_truncf(float x){return truncf(x);}static double m_trunc(double x){return trunc(x);}
static float  m_fabsf(float x){return fabsf(x);} static double m_fabs(double x){return fabs(x);}
static float  m_powf(float x,float y){return powf(x,y);}   static double m_pow(double x,double y){return pow(x,y);}
static float  m_fmodf(float x,float y){return fmodf(x,y);} static double m_fmod(double x,double y){return fmod(x,y);}
static float  m_atan2f(float x,float y){return atan2f(x,y);}static double m_atan2(double x,double y){return atan2(x,y);}
static float  m_hypotf(float x,float y){return hypotf(x,y);}static double m_hypot(double x,double y){return hypot(x,y);}
static void   m_sincosf(float x, float* s, float* c)   { *s = sinf(x); *c = cosf(x); }
static void   m_sincos (double x, double* s, double* c){ *s = sin(x);  *c = cos(x);  }
static double m_ldexp(double x,int n){return ldexp(x,n);}    static float  m_ldexpf(float x,int n){return ldexpf(x,n);}
static double m_frexp(double x,int* e){return frexp(x,e);}   static float  m_frexpf(float x,int* e){return frexpf(x,e);}
static double m_modf(double x,double* i){return modf(x,i);}  static float  m_modff(float x,float* i){return modff(x,i);}
static double m_copysign(double x,double y){return copysign(x,y);} static float m_copysignf(float x,float y){return copysignf(x,y);}
static double m_fmin(double x,double y){return fmin(x,y);}   static float  m_fminf(float x,float y){return fminf(x,y);}
static double m_fmax(double x,double y){return fmax(x,y);}   static float  m_fmaxf(float x,float y){return fmaxf(x,y);}
static double m_sinh(double x){return sinh(x);}   static float m_sinhf(float x){return sinhf(x);}
static double m_cosh(double x){return cosh(x);}   static float m_coshf(float x){return coshf(x);}
static double m_tanh(double x){return tanh(x);}   static float m_tanhf(float x){return tanhf(x);}
static double m_log1p(double x){return log1p(x);} static float m_log1pf(float x){return log1pf(x);}
static double m_expm1(double x){return expm1(x);} static float m_expm1f(float x){return expm1f(x);}
// strtod/strtof return a double/float in XMM0 (which the return-0 stub never set -> garbage float for
// every text-parsed number). Native-signature thunks land the result in the right register; endptr is a
// guest pointer, passed through. (Also fixes float.Parse / Atof / JSON numeric fields across all titles.)
static double m_strtod(const char* s, char** e){ return strtod(s, e); }
static float  m_strtof(const char* s, char** e){ return strtof(s, e); }

HLE(h_vsnprintf) { va_list ap; if (a3) memcpy(&ap, P(a3), sizeof(va_list)); return (uint64_t)(int64_t)vsnprintf((char*)P(a0), (size_t)a1, (const char*)P(a2), ap); }
HLE(h_vsprintf)  { va_list ap; if (a2) memcpy(&ap, P(a2), sizeof(va_list)); return (uint64_t)(int64_t)vsprintf((char*)P(a0), (const char*)P(a1), ap); }
// Variadic forms: REAL C variadic functions, not the old 3-int-register forward (which dropped
// %f/XMM args, all stack args, and %s past the 4th argument -> garbage output or a SIGSEGV on a
// bogus %s pointer). The import stub TAIL-JUMPS into these with the guest's SysV call frame intact
// (GP regs + XMM + AL + overflow stack), so the compiler-generated variadic prologue captures the
// full argument set and va_start/v*printf format it correctly. Guest pointers are identity-mapped
// (guest VA == host VA), so the buffer, format string, and any %s arguments are usable host
// pointers directly — no P() translation needed (the tail-jump passes the raw guest values).
// CONFIDENCE: HIGH for register + XMM args (the overwhelmingly common case, correct on both the
// tail-jmp and GUEST_FS swap-stub paths). MED only for args that spill to the STACK (>6 GP or
// >8 FP) under the GUEST_FS swap stub, whose reframing shifts the overflow area — rare for a
// format call, and still strictly better than the old register-only truncation.
static uint64_t h_snprintf(void* buf, size_t n, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); int r = vsnprintf((char*)buf, n, fmt, ap); va_end(ap);
    return (uint64_t)(int64_t)r;
}
static uint64_t h_sprintf(void* buf, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); int r = vsprintf((char*)buf, fmt, ap); va_end(ap);
    return (uint64_t)(int64_t)r;
}
static uint64_t h_printf(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); int r = vprintf(fmt, ap); va_end(ap);
    return (uint64_t)(int64_t)r;
}
// sscanf: a REAL variadic host thunk (like the *printf family) — the import stub tail-jumps with the
// guest's full SysV frame so va_start captures every arg. Output pointer args are guest pointers
// (identity-mapped), so vsscanf writes through them directly. MISSING before -> returned 0 leaving ALL
// output args uninitialized (the guest consumed uninit memory as parsed values).
static uint64_t h_sscanf(const char* s, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); int r = vsscanf(s, fmt, ap); va_end(ap);
    return (uint64_t)(int64_t)r;
}
HLE(h_puts)      { int r = fputs((const char*)P(a0), stdout); fputc('\n', stdout); return (uint64_t)(int64_t)r; }
HLE(h_putchar)   { return (uint64_t)(int64_t)putchar((int)a0); }
HLE(h_fputs)     { return (uint64_t)(int64_t)fputs((const char*)P(a0), a1 ? (FILE*)P(a1) : stdout); }

// --- locale / ctype (Dinkumware CRT: _Getpctype/_Getpt{o,}lower return table ptrs) ---
// Tables have 257 entries; element [-1] is the EOF slot, so we return base+1 and the
// guest indexes [c] for c in 0..255 (and [-1] for EOF). Classification bits follow the
// common MSVCRT/Dinkumware layout; built from the host's ctype so they're correct.
namespace {
    short g_ctype[257], g_tolow[257], g_toup[257];
    bool  g_ctype_built = false;
    void build_ctype() {
        if (g_ctype_built) return;
        g_ctype[0] = g_tolow[0] = g_toup[0] = 0;   // EOF slot
        for (int c = 0; c < 256; c++) {
            short m = 0;
            if (isupper(c)) m |= 0x01; if (islower(c)) m |= 0x02; if (isdigit(c)) m |= 0x04;
            if (isspace(c)) m |= 0x08; if (ispunct(c)) m |= 0x10; if (iscntrl(c)) m |= 0x20;
            if (c == ' ' || c == '\t') m |= 0x40; if (isxdigit(c)) m |= 0x80;
            g_ctype[c + 1] = m;
            g_tolow[c + 1] = (short)tolower(c);
            g_toup[c + 1]  = (short)toupper(c);
        }
        g_ctype_built = true;
    }
}
HLE(h_getpctype)  { build_ctype(); return (uint64_t)(uintptr_t)(g_ctype + 1); }
HLE(h_getptolow)  { build_ctype(); return (uint64_t)(uintptr_t)(g_tolow + 1); }
HLE(h_getptoup)   { build_ctype(); return (uint64_t)(uintptr_t)(g_toup + 1); }
// glibc contract: __ctype_get_mb_cur_max returns the VALUE of MB_CUR_MAX (size_t),
// not a pointer. 1 in the "C" locale we present via setlocale.
HLE(h_mbcurmax)   { return 1; }
HLE(h_setlocale)  { static char c[] = "C"; return (uint64_t)(uintptr_t)c; }

// C++/CRT lifecycle: we don't run global destructors, so registration is a no-op.
HLE(h_atexit)      { return 0; }
HLE(h_cxa_atexit)  { return 0; }
HLE(h_cxa_finalize){ return 0; }
// Itanium C++ ABI static-init guards: byte 0 of the guard = "initialized", byte 1 = an init is in
// flight. The old implementation was a non-atomic check with no blocking, so two threads racing an
// uninitialized function-local static BOTH got "you initialize" — concurrent double-construction
// (torn singletons) in the guest's 15-thread IL2CPP pool. Real contract: acquire returns 1 to
// exactly one thread and blocks the rest until release/abort. Contenders park on one global
// mutex/condvar pair (init is cold; the winner runs its constructor OUTSIDE the lock, so
// independent guards cannot deadlock each other); the fast path stays a lock-free acquire-load.
namespace { std::mutex g_guard_mx; std::condition_variable g_guard_cv; }
HLE(h_guard_acquire) {
    auto* g = (std::atomic<uint8_t>*)P(a0);
    if (g->load(std::memory_order_acquire)) return 0;      // already initialized
    std::unique_lock<std::mutex> lk(g_guard_mx);
    for (;;) {
        if (g->load(std::memory_order_acquire)) return 0;  // init finished while we waited
        uint8_t* busy = (uint8_t*)g + 1;
        if (!*busy) { *busy = 1; return 1; }               // we win: caller runs the initializer
        g_guard_cv.wait(lk);
    }
}
HLE(h_guard_release) {
    auto* g = (std::atomic<uint8_t>*)P(a0);
    { std::lock_guard<std::mutex> lk(g_guard_mx);
      ((uint8_t*)g)[1] = 0;
      g->store(1, std::memory_order_release); }
    g_guard_cv.notify_all();
    return 0;
}
HLE(h_guard_abort) {   // initializer threw: clear busy so another thread can retry
    auto* g = (std::atomic<uint8_t>*)P(a0);
    { std::lock_guard<std::mutex> lk(g_guard_mx); ((uint8_t*)g)[1] = 0; }
    g_guard_cv.notify_all();
    return 0;
}

// std::_Execute_once(once_flag&, int(*cb)(void*,void*,void**), void* arg) — the guts
// of std::call_once. It MUST invoke the callback (which runs the real one-time init),
// exactly once per flag, and return nonzero on success (call_once throws on 0). The old
// check-then-run had no synchronization: concurrent first calls ran the initializer twice
// in parallel. Flag word: 0 = not run, 2 = running, 1 = done; the callback runs OUTSIDE
// the lock so once-inits that depend on other threads' progress can't deadlock.
HLE(h_execute_once) {
    auto* flag = (std::atomic<uint32_t>*)P(a0);
    auto cb = (int (*)(void*, void*, void**))P(a1);
    if (!flag) { void* ctx = nullptr; return (uint64_t)((!cb || cb(nullptr, (void*)P(a2), &ctx)) ? 1 : 0); }
    for (;;) {
        if (flag->load(std::memory_order_acquire) == 1) return 1;   // already executed
        std::unique_lock<std::mutex> lk(g_guard_mx);
        uint32_t v = flag->load(std::memory_order_acquire);
        if (v == 1) return 1;
        if (v == 2) { g_guard_cv.wait(lk); continue; }              // another thread is running it
        flag->store(2, std::memory_order_relaxed);
        lk.unlock();
        void* ctx = nullptr;
        int r = cb ? cb((void*)P(a0), (void*)P(a2), &ctx) : 1;
        lk.lock();
        flag->store(r ? 1u : 0u, std::memory_order_release);        // failure -> retryable next call
        lk.unlock();
        g_guard_cv.notify_all();
        return (uint64_t)(r ? 1 : 0);
    }
}
// C++ exception refcounting — no-op is safe for the non-throwing boot path.
HLE(h_cxa_dec_refcount) { return 0; }
HLE(h_cxa_inc_refcount) { return 0; }

// sceLibcHeapGetTraceInfo (libSceLibcInternalExt, NID NWtTN10cJzE) — the guest allocator's
// malloc-trace coordination block. Contract (Kyty LibC.cpp:139 + shadPS4 kernel.cpp:161, identical):
// the caller passes a 32-byte struct { u64 size(=32, caller-set); u32 flag; u32 getSegmentInfo;
// u64* mspace_atomic_id_mask; u64* mstate_table } and the callee fills the two pointers with
// PERSISTENT storage (Kyty: static u64 + static u64[64]) that the guest mspace allocator then
// reads AND WRITES on allocation paths. Our previous behavior (unimplemented stub -> return 0,
// struct untouched) handed the guest two garbage stack values as pointers: every subsequent
// traced-mspace operation stored through random addresses — silent heap/stack corruption that
// surfaces only under heavy allocation (e.g. the level1 resources.assets loader thread).
// CONFIDENCE: HIGH — two independent references agree on layout and semantics.
namespace { uint64_t g_mspace_atomic_id_mask = 0; uint64_t g_mstate_table[64] = {0}; }
HLE(h_heap_get_trace_info) {
    uint8_t* info = (uint8_t*)P(a0);
    if (!info) return 0;
    // The size field is caller-set; only fill the layout we know (exactly 32 bytes — never write
    // past the caller's struct; cf. the f_fstat oversized-write lesson).
    if (*(uint64_t*)info != 32) {
        fprintf(stderr, "[prosper] sceLibcHeapGetTraceInfo: unexpected info->size=%llu (want 32) — leaving untouched\n",
                (unsigned long long)*(uint64_t*)info);
        return 0;
    }
    *(uint32_t*)(info + 0x0c)  = 0;                        // getSegmentInfo: no segment-info callback
    *(uint64_t**)(info + 0x10) = &g_mspace_atomic_id_mask; // persistent, zero-initialized
    *(uint64_t**)(info + 0x18) = g_mstate_table;
    return 0;
}

void register_builtin_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // libSceLibcInternalExt heap-trace hookup (raw NID — guaranteed match; see handler comment).
    Hle::register_fn("NWtTN10cJzE", (HleFn)h_heap_get_trace_info, "sceLibcHeapGetTraceInfo");
    R("memcpy", h_memcpy);   R("memmove", h_memmove); R("memset", h_memset);
    R("memcmp", h_memcmp);   R("memchr", h_memchr);
    R("strlen", h_strlen);   R("strnlen", h_strnlen);
    R("strcmp", h_strcmp);   R("strncmp", h_strncmp);
    R("strcpy", h_strcpy);   R("strncpy", h_strncpy);
    R("strcat", h_strcat);   R("strncat", h_strncat);
    R("strchr", h_strchr);   R("strrchr", h_strrchr); R("strstr", h_strstr);
    R("strlcpy", h_strlcpy); R("strlcat", h_strlcat);
    // Annex-K bounded (secure-CRT) variants — were MISSING -> return-0 stub = "success" without copying
    R("memcpy_s", h_memcpy_s);   R("memmove_s", h_memmove_s);   R("memset_s", h_memset_s);
    R("strcpy_s", h_strcpy_s);   R("strncpy_s", h_strncpy_s);
    R("strcat_s", h_strcat_s);   R("strncat_s", h_strncat_s);
    R("malloc", h_malloc);   R("calloc", h_calloc);   R("realloc", h_realloc); R("free", h_free);
    R("memalign", h_memalign); R("posix_memalign", h_posix_memalign); R("aligned_alloc", h_aligned_alloc);
    // operator new / new[] (+ nothrow), and aligned variants
    R("_Znwm", h_new); R("_Znam", h_new);
    R("_ZnwmRKSt9nothrow_t", h_new); R("_ZnamRKSt9nothrow_t", h_new);
    R("_ZnwmSt11align_val_t", h_new_align); R("_ZnamSt11align_val_t", h_new_align);
    R("_ZnwmSt11align_val_tRKSt9nothrow_t", h_new_align); R("_ZnamSt11align_val_tRKSt9nothrow_t", h_new_align);
    // operator delete / delete[] (+ sized, aligned, nothrow) -> free
    R("_ZdlPv", h_delete); R("_ZdaPv", h_delete);
    R("_ZdlPvm", h_delete); R("_ZdaPvm", h_delete);
    R("_ZdlPvSt11align_val_t", h_delete); R("_ZdaPvSt11align_val_t", h_delete);
    R("_ZdlPvmSt11align_val_t", h_delete); R("_ZdaPvmSt11align_val_t", h_delete);
    R("_ZdlPvRKSt9nothrow_t", h_delete); R("_ZdaPvRKSt9nothrow_t", h_delete);
    // stdio
    R("vsnprintf", h_vsnprintf); R("vsprintf", h_vsprintf);
    R("snprintf", h_snprintf);   R("sprintf", h_sprintf);   R("snprintf_s", h_snprintf);
    R("sprintf_s", h_snprintf);   // Annex-K sprintf_s(s, n, fmt, ...) has a size arg -> bounded snprintf, NOT sprintf
    R("printf", h_printf);       R("puts", h_puts);
    R("putchar", h_putchar);     R("fputs", h_fputs);
    // locale / ctype
    R("_Getpctype", h_getpctype); R("_Getptolower", h_getptolow); R("_Getptoupper", h_getptoup);
    R("__ctype_get_mb_cur_max", h_mbcurmax); R("setlocale", h_setlocale);
    R("atexit", h_atexit);   R("__cxa_atexit", h_cxa_atexit); R("__cxa_finalize", h_cxa_finalize);
    R("__cxa_guard_acquire", h_guard_acquire);
    R("__cxa_guard_release", h_guard_release);
    R("__cxa_guard_abort",   h_guard_abort);
    R("_ZSt13_Execute_onceRSt9once_flagPFiPvS1_PS1_ES1_", h_execute_once);  // std::call_once core
    R("__cxa_decrement_exception_refcount", h_cxa_dec_refcount);
    R("__cxa_increment_exception_refcount", h_cxa_inc_refcount);
    // setjmp/longjmp family (real register-saving impl; used by Boehm GC root scanning)
    R("setjmp", prosper_setjmp);   R("_setjmp", prosper_setjmp);   R("sigsetjmp", prosper_setjmp);
    R("longjmp", prosper_longjmp); R("_longjmp", prosper_longjmp);  R("siglongjmp", prosper_longjmp);
    // byte ops / search / env
    R("bcmp", h_bcmp);   R("bsearch", h_bsearch);   R("qsort", h_qsort);
    // conversion / tokenize / wide / scan (were MISSING -> 0/garbage on parsing paths; confirmed imports)
    R("strtol", h_strtol);   R("strtoll", h_strtoll);
    R("strtoul", h_strtoul); R("strtoull", h_strtoull);
    R("strtod", m_strtod);   R("strtof", m_strtof);
    R("atoi", h_atoi);       R("atol", h_atol);
    R("rand", h_rand);       R("srand", h_srand);       R("rand_r", h_rand_r);   // were MISSING -> rand()==0
    R("strdup", h_strdup);   R("strtok", h_strtok);
    R("strspn", h_strspn);   R("strcspn", h_strcspn);   R("strpbrk", h_strpbrk);
    R("wcslen", h_wcslen);
    R("sscanf", h_sscanf);   R("vsscanf", h_vsscanf);
    R("_init_env", h_init_env);   R("malloc_stats_fast", h_malloc_stats_fast);
    R("__error", h_errno_location);  R("__errno_location", h_errno_location);  R("___errno", h_errno_location);
    // math (real host thunks; float args in XMM survive the tail-jump stub)
    R("sinf", m_sinf); R("cosf", m_cosf); R("tanf", m_tanf); R("asinf", m_asinf); R("acosf", m_acosf);
    R("atanf", m_atanf); R("atan2f", m_atan2f); R("expf", m_expf); R("exp2f", m_exp2f); R("logf", m_logf);
    R("log10f", m_log10f); R("log2f", m_log2f); R("sqrtf", m_sqrtf); R("cbrtf", m_cbrtf); R("powf", m_powf);
    R("floorf", m_floorf); R("ceilf", m_ceilf); R("roundf", m_roundf); R("truncf", m_truncf);
    R("fmodf", m_fmodf); R("fabsf", m_fabsf); R("hypotf", m_hypotf);
    R("sin", m_sin); R("cos", m_cos); R("tan", m_tan); R("asin", m_asin); R("acos", m_acos);
    R("atan", m_atan); R("atan2", m_atan2); R("exp", m_exp); R("exp2", m_exp2); R("log", m_log);
    R("log10", m_log10); R("log2", m_log2); R("sqrt", m_sqrt); R("cbrt", m_cbrt); R("pow", m_pow);
    R("floor", m_floor); R("ceil", m_ceil); R("round", m_round); R("trunc", m_trunc);
    R("fmod", m_fmod); R("fabs", m_fabs); R("hypot", m_hypot);
    R("sincosf", m_sincosf); R("sincos", m_sincos);
    R("ldexp", m_ldexp); R("ldexpf", m_ldexpf); R("frexp", m_frexp); R("frexpf", m_frexpf);
    R("modf", m_modf); R("modff", m_modff); R("copysign", m_copysign); R("copysignf", m_copysignf);
    R("fmin", m_fmin); R("fminf", m_fminf); R("fmax", m_fmax); R("fmaxf", m_fmaxf);
    R("sinh", m_sinh); R("sinhf", m_sinhf); R("cosh", m_cosh); R("coshf", m_coshf);
    R("tanh", m_tanh); R("tanhf", m_tanhf); R("log1p", m_log1p); R("log1pf", m_log1pf);
    R("expm1", m_expm1); R("expm1f", m_expm1f);
    #undef R
    register_file_hle();     // file I/O (stdio + POSIX, /app0 translation)
    register_service_hle();  // PS5 system services (user/NP/mouse/appcontent/dialog)
    register_pad_hle();      // libScePad: real game-controller input (input/pad.cpp)
    register_audio_hle();    // libSceAudioOut backed by a headless/pluggable AudioSink
    register_graphics_hle(); // headless libSceAgc/libSceVideoOut placeholders (bring-up)
    register_agc_hle();      // real AGC Dcb functions (override the glog stubs for Dcb NIDs)
    register_kernel_hle();   // libkernel primitives (pthread/sync/...)
}

} // namespace prosper
