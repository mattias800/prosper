#include "nid.hpp"
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace prosper {

// ---- SHA-1 (public-domain style compact implementation) --------------------
namespace {
struct Sha1 {
    uint32_t h[5]; uint64_t len = 0; uint8_t buf[64]; size_t n = 0;
    Sha1() { h[0]=0x67452301; h[1]=0xEFCDAB89; h[2]=0x98BADCFE; h[3]=0x10325476; h[4]=0xC3D2E1F0; }
    static uint32_t rol(uint32_t v, int b) { return (v << b) | (v >> (32 - b)); }
    void block(const uint8_t* p) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = (p[i*4]<<24)|(p[i*4+1]<<16)|(p[i*4+2]<<8)|p[i*4+3];
        for (int i = 16; i < 80; i++) w[i] = rol(w[i-3]^w[i-8]^w[i-14]^w[i-16], 1);
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | (~b & d);            k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;                     k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d);   k = 0x8F1BBCDC; }
            else             { f = b ^ c ^ d;                     k = 0xCA62C1D6; }
            uint32_t t = rol(a,5) + f + e + k + w[i];
            e = d; d = c; c = rol(b,30); b = a; a = t;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e;
    }
    void update(const void* data, size_t l) {
        const uint8_t* p = (const uint8_t*)data; len += l;
        while (l) {
            size_t take = 64 - n; if (take > l) take = l;
            memcpy(buf + n, p, take); n += take; p += take; l -= take;
            if (n == 64) { block(buf); n = 0; }
        }
    }
    void finish(uint8_t out[20]) {
        uint64_t bits = len * 8;
        uint8_t pad = 0x80; update(&pad, 1);
        uint8_t z = 0; while (n != 56) update(&z, 1);
        uint8_t lb[8]; for (int i = 0; i < 8; i++) lb[i] = (bits >> (56 - i*8)) & 0xff;
        update(lb, 8);
        for (int i = 0; i < 5; i++) { out[i*4]=h[i]>>24; out[i*4+1]=h[i]>>16; out[i*4+2]=h[i]>>8; out[i*4+3]=h[i]; }
    }
};

// base64 of raw bytes with Sony's alphabet, no padding.
std::string b64_sony(const uint8_t* p, size_t n) {
    static const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    std::string out;
    size_t i = 0;
    while (i + 3 <= n) {
        uint32_t v = (p[i]<<16)|(p[i+1]<<8)|p[i+2];
        out += A[(v>>18)&63]; out += A[(v>>12)&63]; out += A[(v>>6)&63]; out += A[v&63];
        i += 3;
    }
    size_t rem = n - i;
    if (rem == 1) { uint32_t v = p[i]<<16; out += A[(v>>18)&63]; out += A[(v>>12)&63]; }
    else if (rem == 2) { uint32_t v = (p[i]<<16)|(p[i+1]<<8); out += A[(v>>18)&63]; out += A[(v>>12)&63]; out += A[(v>>6)&63]; }
    return out;
}
} // namespace

std::string nid_hash(const std::string& name) {
    // Sony's full 16-byte NID salt (the well-known 0x518D64A635DED8C1E6B039B1C3E55230).
    static const uint8_t salt[16] = {0x51,0x8D,0x64,0xA6,0x35,0xDE,0xD8,0xC1,
                                     0xE6,0xB0,0x39,0xB1,0xC3,0xE5,0x52,0x30};
    Sha1 s; s.update(name.data(), name.size()); s.update(salt, 16);
    uint8_t d[20]; s.finish(d);
    // first 8 bytes, little-endian (reversed), then Sony-base64, take 11 chars.
    uint8_t r[8]; for (int i = 0; i < 8; i++) r[i] = d[7 - i];
    std::string b = b64_sony(r, 8);
    return b.substr(0, 11);
}

// ---- NidDb -----------------------------------------------------------------
NidDb::NidDb() {
    add_many(builtin_symbol_names());
    // Load the big known-symbol list (idc/ps4libdoc) if available, so traces name
    // the whole import surface. Env override first, then the compiled-in source path.
    if (const char* env = getenv("PROSPER_NIDDB")) { if (load_names_file(env)) return; }
#ifdef PROSPER_SOURCE_DIR
    load_names_file(PROSPER_SOURCE_DIR "/data/known_names.txt");
#endif
}
void NidDb::add(const std::string& name) { by_nid_[nid_hash(name)] = name; }
void NidDb::add_many(const std::vector<std::string>& names) { for (auto& n : names) add(n); }
size_t NidDb::load_names_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return 0;
    size_t n = 0; char line[512];
    while (fgets(line, sizeof line, f)) {
        char* p = line;
        if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF) p += 3; // BOM
        size_t len = strlen(p);
        while (len && (p[len-1] == '\n' || p[len-1] == '\r' || p[len-1] == ' ')) p[--len] = 0;
        if (len) { by_nid_[nid_hash(p)] = p; n++; }
    }
    fclose(f);
    return n;
}
const std::string& NidDb::resolve(const std::string& nid) const {
    auto it = by_nid_.find(nid);
    return it == by_nid_.end() ? empty_ : it->second;
}

const std::vector<std::string>& builtin_symbol_names() {
    static const std::vector<std::string> names = {
        // libc string/mem
        "memcpy","memmove","memset","memcmp","memchr","strlen","strnlen","strcmp","strncmp",
        "strcpy","strncpy","strcat","strncat","strchr","strrchr","strstr","strtol","strtoul",
        "strtod","strdup","snprintf","vsnprintf","sprintf","printf","fprintf","puts","putchar",
        // libc stdlib / heap
        "malloc","calloc","realloc","free","aligned_alloc","posix_memalign","abort","exit",
        "atexit","__cxa_atexit","__cxa_finalize","qsort","bsearch","getenv","rand","srand",
        // libc stdio / file
        "fopen","fclose","fread","fwrite","fseek","ftell","fflush","fgets","fputs","setvbuf",
        // libc math
        "pow","sqrt","sin","cos","tan","atan2","floor","ceil","fmod","log","exp",
        // stack protector / init
        "__stack_chk_fail","__stack_chk_guard","__memcpy_chk","__memset_chk",
        // C++ runtime
        "__cxa_guard_acquire","__cxa_guard_release","__cxa_throw","__cxa_begin_catch",
        "__cxa_end_catch","_Znwm","_Znam","_ZdlPv","_ZdaPv","__gxx_personality_v0",
        // pthread (Sony scePthread + posix names)
        "scePthreadCreate","scePthreadJoin","scePthreadExit","scePthreadMutexInit",
        "scePthreadMutexLock","scePthreadMutexUnlock","scePthreadMutexDestroy",
        "scePthreadMutexTrylock","scePthreadMutexTimedlock",
        "scePthreadMutexattrInit","scePthreadMutexattrDestroy","scePthreadMutexattrSettype",
        "scePthreadMutexattrSetprotocol","scePthreadMutexattrSetpshared","scePthreadMutexattrGettype",
        "scePthreadCondInit","scePthreadCondWait","scePthreadCondSignal","scePthreadCondBroadcast",
        "scePthreadCondDestroy","scePthreadCondTimedwait","scePthreadCondattrInit","scePthreadCondattrDestroy",
        "scePthreadCondattrSetclock","scePthreadCondattrGetclock",
        "scePthreadCondattrSetpshared","scePthreadCondattrGetpshared",
        "scePthreadSelf","scePthreadOnce","scePthreadKeyCreate","scePthreadSetspecific","scePthreadGetspecific",
        "scePthreadKeyDelete","scePthreadEqual","scePthreadYield","scePthreadDetach",
        "scePthreadAttrInit","scePthreadAttrDestroy","scePthreadAttrSetstacksize","scePthreadAttrSetdetachstate",
        "scePthreadAttrSetinheritsched","scePthreadAttrSetschedparam","scePthreadRename","scePthreadSetprio","scePthreadGetprio",
        "scePthreadRwlockInit","scePthreadRwlockDestroy","scePthreadRwlockRdlock","scePthreadRwlockWrlock","scePthreadRwlockUnlock",
        "pthread_create","pthread_join","pthread_mutex_lock","pthread_mutex_unlock",
        "pthread_cond_wait","pthread_cond_signal","pthread_condattr_setclock","pthread_condattr_getclock",
        "pthread_condattr_setpshared","pthread_condattr_getpshared","pthread_self","pthread_once",
        // libkernel memory
        "sceKernelAllocateDirectMemory","sceKernelReleaseDirectMemory","sceKernelMapDirectMemory",
        "sceKernelMapNamedFlexibleMemory","sceKernelMapFlexibleMemory","sceKernelReserveVirtualRange",
        "sceKernelMunmap","sceKernelMmap","sceKernelAvailableDirectMemorySize","sceKernelGetDirectMemorySize",
        "sceKernelVirtualQuery","sceKernelSetVirtualRangeName",
        // libkernel process / module / misc
        "sceKernelGetProcParam","sceKernelLoadStartModule","sceKernelDlsym","sceKernelGetModuleInfo",
        "sceKernelError","sceKernelGetCpumode","sceKernelIsNeoMode","sceKernelUsleep","sceKernelSleep",
        "sceKernelNanosleep","sceKernelGettimeofday","sceKernelClockGettime","sceKernelGetProcessTime",
        "sceKernelGetTscFrequency","sceKernelReadTsc","sceKernelGetSystemSwVersion",
        // libkernel threads / sync primitives
        "sceKernelCreateEqueue","sceKernelWaitEqueue","sceKernelDeleteEqueue",
        "sceKernelCreateEventFlag","sceKernelWaitEventFlag","sceKernelSetEventFlag",
        "sceKernelCreateSema","sceKernelWaitSema","sceKernelSignalSema","sceKernelDeleteSema",
        // libkernel fs
        "sceKernelOpen","sceKernelClose","sceKernelRead","sceKernelWrite","sceKernelLseek",
        "sceKernelStat","sceKernelFstat","sceKernelMkdir","sceKernelUnlink",
        "open","close","read","write","lseek","stat","fstat","mmap","munmap",
        "clock_gettime","gettimeofday","nanosleep","usleep","getpid","sched_yield",
    };
    return names;
}

} // namespace prosper
