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
#include <cerrno>
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

HLE(h_vsnprintf) { va_list ap; if (a3) memcpy(&ap, P(a3), sizeof(va_list)); return (uint64_t)(int64_t)vsnprintf((char*)P(a0), (size_t)a1, (const char*)P(a2), ap); }
HLE(h_vsprintf)  { va_list ap; if (a2) memcpy(&ap, P(a2), sizeof(va_list)); return (uint64_t)(int64_t)vsprintf((char*)P(a0), (const char*)P(a1), ap); }
// Variadic forms: forward the register args best-effort (handles the common
// integer/pointer/≤4-arg case; float/stack args are a later refinement).
HLE(h_snprintf)  { return (uint64_t)(int64_t)snprintf((char*)P(a0), (size_t)a1, (const char*)P(a2), a3, a4, a5); }
HLE(h_sprintf)   { return (uint64_t)(int64_t)sprintf((char*)P(a0), (const char*)P(a1), a2, a3, a4, a5); }
HLE(h_printf)    { return (uint64_t)(int64_t)printf((const char*)P(a0), a1, a2, a3, a4, a5); }
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
HLE(h_mbcurmax)   { static int one = 1; return (uint64_t)(uintptr_t)&one; }   // __ctype_get_mb_cur_max ptr/val
HLE(h_setlocale)  { static char c[] = "C"; return (uint64_t)(uintptr_t)c; }

// C++/CRT lifecycle: we don't run global destructors, so registration is a no-op.
HLE(h_atexit)      { return 0; }
HLE(h_cxa_atexit)  { return 0; }
HLE(h_cxa_finalize){ return 0; }
// Itanium C++ ABI static-init guards: byte 0 of the guard = "initialized".
HLE(h_guard_acquire) { uint8_t* g = (uint8_t*)P(a0); return (*g) ? 0 : 1; }
HLE(h_guard_release) { uint8_t* g = (uint8_t*)P(a0); *g = 1; return 0; }
HLE(h_guard_abort)   { return 0; }

// std::_Execute_once(once_flag&, int(*cb)(void*,void*,void**), void* arg) — the guts
// of std::call_once. It MUST invoke the callback (which runs the real one-time init),
// exactly once per flag, and return nonzero on success (call_once throws on 0).
HLE(h_execute_once) {
    uint32_t* flag = (uint32_t*)P(a0);
    if (flag && *flag) return 1;                 // already executed
    auto cb = (int (*)(void*, void*, void**))P(a1);
    int r = 1;
    if (cb) { void* ctx = nullptr; r = cb((void*)P(a0), (void*)P(a2), &ctx); }
    if (flag) *flag = 1;
    return (uint64_t)(r ? 1 : 0);
}
// C++ exception refcounting — no-op is safe for the non-throwing boot path.
HLE(h_cxa_dec_refcount) { return 0; }
HLE(h_cxa_inc_refcount) { return 0; }

void register_builtin_hle() {
    #define R(str, fn) Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    R("memcpy", h_memcpy);   R("memmove", h_memmove); R("memset", h_memset);
    R("memcmp", h_memcmp);   R("memchr", h_memchr);
    R("strlen", h_strlen);   R("strnlen", h_strnlen);
    R("strcmp", h_strcmp);   R("strncmp", h_strncmp);
    R("strcpy", h_strcpy);   R("strncpy", h_strncpy);
    R("strcat", h_strcat);   R("strncat", h_strncat);
    R("strchr", h_strchr);   R("strrchr", h_strrchr); R("strstr", h_strstr);
    R("strlcpy", h_strlcpy); R("strlcat", h_strlcat);
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
    R("bcmp", h_bcmp);   R("bsearch", h_bsearch);
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
