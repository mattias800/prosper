// hle_agc.cpp — libSceAgc "Gen5" graphics-driver HLE: the Draw Command Buffer (Dcb) frontend.
//
// The AGC Dcb functions ARE the driver: each appends a PM4 packet into the game's command buffer and
// RETURNS the allocated dword pointer, which the game (and the indirect-patch helpers) then use. Our
// old stubs returned 0 -> the patch helpers wrote through a null pointer (the eboot+0x3b5ea6 fault).
//
// Ported/adapted from Kyty (../Kyty, MIT-licensed; source/emulator/src/Graphics/Graphics.cpp +
// Pm4.h) — the Dcb struct layout, PM4 encoding, and per-function packet contents are Kyty's, which
// reverse-engineered the real libSceAgc ABI. A later CommandProcessor will decode this PM4 stream to
// Vulkan (see docs/AGC_IMPL_PLAN.md). Registered by raw NID (AGC lib is undocumented).
#include "dispatch.hpp"
#include "gpu/pm4_registers.hpp"
#include "gpu/command_processor.hpp"
#include "gpu/pm4_decode.hpp"
#include "gpu/gpu_execute.hpp"
#include "gpu/videoout_present.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

namespace {

// --- PM4 encoding (Kyty Pm4.h) ---------------------------------------------------------------
constexpr uint32_t IT_NOP = 0x10, IT_INDEX_TYPE = 0x2A, IT_EVENT_WRITE = 0x46, IT_SET_SH_REG = 0x76;
// Custom sub-opcodes carried inside IT_NOP:
constexpr uint32_t R_DRAW_INDEX_AUTO = 0x04, R_DRAW_RESET = 0x05, R_WAIT_FLIP_DONE = 0x06,
                   R_SH_REGS_INDIRECT = 0x11, R_CX_REGS_INDIRECT = 0x12, R_UC_REGS_INDIRECT = 0x13,
                   R_ACQUIRE_MEM = 0x14, R_WRITE_DATA = 0x15, R_WAIT_MEM_64 = 0x16, R_FLIP = 0x17,
                   R_RELEASE_MEM = 0x18;
constexpr uint32_t R_NUM = 0x40;
inline uint32_t PM4(uint32_t len, uint32_t op, uint32_t r) {
    return 0xC0000000u | (((len - 2u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | ((r & (R_NUM - 1u)) << 2u);
}

// --- The Draw Command Buffer (the game's own struct; we cast the guest pointer). Kyty layout
// (Graphics.cpp:777): a dword ring [bottom,top) with a write cursor and a "buffer full" callback. ---
struct AgcDcb {
    uint32_t* bottom;       // +0x00 buffer base
    uint32_t* top;          // +0x08 buffer end
    uint32_t* cursor_up;    // +0x10 current write ptr
    uint32_t* cursor_down;  // +0x18 reserve limit
    bool (*callback)(AgcDcb*, uint32_t, void*);  // +0x20 buffer-full callback (guest fn)
    void*     user_data;    // +0x28
    uint32_t  reserved_dw;  // +0x30

    uint32_t available_dw() const {
        auto avail = (uint32_t)(cursor_down - cursor_up);
        return avail > reserved_dw ? avail - reserved_dw : 0;
    }
    uint32_t* allocate_dw(uint32_t n) {
        if (n == 0) return nullptr;
        if (n > available_dw()) {
            if (!callback || !callback(this, n + reserved_dw, user_data)) return nullptr;
            if (available_dw() < n) return nullptr;
        }
        uint32_t* r = cursor_up;
        cursor_up += n;
        return r;
    }
};

// Helper: allocate `n` dwords and set the header; returns cmd (or nullptr).
inline uint32_t* begin_packet(uint64_t buf, uint32_t n, uint32_t op, uint32_t r, uint32_t** out) {
    auto* dcb = (AgcDcb*)(uintptr_t)buf;
    if (!dcb) { *out = nullptr; return nullptr; }
    uint32_t* cmd = dcb->allocate_dw(n);
    if (cmd) cmd[0] = PM4(n, op, r);
    *out = cmd;
    return cmd;
}

}  // namespace

// --- Dcb append functions (a0 = Dcb*, return = the allocated packet pointer). ------------------
HLE(agc_dcb_reset_queue) {  // (buf, op=0x3ff, state=0)
    uint32_t* cmd; if (!begin_packet(a0, 2, IT_NOP, R_DRAW_RESET, &cmd)) return 0;
    cmd[1] = 0; return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_wait_safe_for_rendering) {  // (buf, video_out_handle, display_buffer_index)
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_WAIT_FLIP_DONE, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; cmd[2] = (uint32_t)a2; cmd[3] = cmd[4] = cmd[5] = cmd[6] = 0;
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_set_sh_register_direct) {  // (buf, ShaderRegister reg)  reg packed in a1: offset|value<<32
    uint32_t* cmd; if (!begin_packet(a0, 3, IT_SET_SH_REG, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffffffffu); cmd[2] = (uint32_t)(a1 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
static uint64_t set_regs_indirect(uint64_t buf, uint64_t regs, uint64_t num_regs, uint32_t r) {
    uint32_t* cmd; if (!begin_packet(buf, 4, IT_NOP, r, &cmd)) return 0;
    cmd[1] = (uint32_t)num_regs; cmd[2] = (uint32_t)(regs & 0xffffffffu); cmd[3] = (uint32_t)(regs >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_set_cx_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_CX_REGS_INDIRECT); }
HLE(agc_dcb_set_sh_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_SH_REGS_INDIRECT); }
HLE(agc_dcb_set_uc_regs_indirect) { return set_regs_indirect(a0, a1, a2, R_UC_REGS_INDIRECT); }
HLE(agc_dcb_set_index_size) {  // (buf, index_size, cache_policy)
    uint32_t* cmd; if (!begin_packet(a0, 2, IT_INDEX_TYPE, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_draw_index_auto) {  // (buf, index_count, modifier)
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_DRAW_INDEX_AUTO, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; cmd[2] = 0; return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_event_write) {  // (buf, event_type, address)
    uint32_t* cmd; if (!begin_packet(a0, 2, IT_EVENT_WRITE, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)(a1 & 0xffu); return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_acquire_mem) {  // (buf, engine, cb_db_op, gcr_cntl, ...) — 8 dw
    uint32_t* cmd; if (!begin_packet(a0, 8, IT_NOP, R_ACQUIRE_MEM, &cmd)) return 0;
    cmd[1] = (uint32_t)a2; cmd[2] = cmd[3] = cmd[4] = cmd[5] = cmd[6] = 0; cmd[7] = (uint32_t)a3;
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_cb_release_mem) {  // sceAgcCbReleaseMem(buf, action, gcr_cntl, dst, cache_policy, address, data_sel, data, …)
    // ABI pinned to Kyty GraphicsCbReleaseMem (Graphics.cpp:1763): a5=label address, then the two stack
    // args are data_sel (7th) and the 64-bit fence value (8th). SysV: fp[1]=ret, fp[2]=data_sel, fp[3]=data.
    volatile uint64_t* fp = (uint64_t*)__builtin_frame_address(0);
    uint64_t data_sel = fp[2], data = fp[3];
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] ReleaseMem action=0x%llx dst=0x%llx addr=0x%llx data_sel=0x%llx data=0x%llx\n",
            (unsigned long long)a1,(unsigned long long)a3,(unsigned long long)a5,
            (unsigned long long)data_sel,(unsigned long long)data);
    // EOP fence: stash the label address (a5), the data_sel, the full 64-bit value, and the event action
    // into the packet so the CommandProcessor performs the completion write at SUBMIT time (correct
    // end-of-pipe timing — our GPU folds synchronously, so submit == pipe drain). CONFIDENCE: HIGH — the
    // arg positions are fixed by the AGC ABI; a5-as-label was independently confirmed (WaitRegMem polls it).
    uint32_t* cmd; if (!begin_packet(a0, 7, IT_NOP, R_RELEASE_MEM, &cmd)) return 0;
    cmd[1] = (uint32_t)(a5 & 0xffffffffu); cmd[2] = (uint32_t)(a5 >> 32u);
    cmd[3] = (uint32_t)data_sel;
    cmd[4] = (uint32_t)(data & 0xffffffffu); cmd[5] = (uint32_t)(data >> 32u);
    cmd[6] = (uint32_t)a1;
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_write_data) {  // sceAgcDcbWriteData(buf, dst, cache_policy, address_or_offset, data*, num_dwords, …)
    // ABI per Kyty GraphicsDcbWriteData (Graphics.cpp:2061): a1=dst, a3=addr, a4=data ptr, a5=num_dwords.
    const uint32_t* src = (const uint32_t*)(uintptr_t)a4;
    uint32_t num = (uint32_t)a5;
    if (num > 60) num = 60;   // clamp to keep the packet within a reasonable bound (labels/fences are tiny)
    if (getenv("PROSPER_GFXLOG")) fprintf(stderr, "[agc] WriteData dst=0x%llx addr=0x%llx num=%u src=%p\n",
        (unsigned long long)a1,(unsigned long long)a3, num, (const void*)src);
    // Copy the inline data into the packet so the CommandProcessor can perform the write at submit time.
    uint32_t* cmd; if (!begin_packet(a0, 5 + num, IT_NOP, R_WRITE_DATA, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    cmd[2] = (uint32_t)(a3 & 0xffffffffu); cmd[3] = (uint32_t)(a3 >> 32u);
    cmd[4] = num;
    if (src) for (uint32_t i = 0; i < num; i++) cmd[5 + i] = src[i];
    return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_wait_reg_mem) {  // (buf, ...) — 9 dw
    if (getenv("PROSPER_GFXLOG")) {
        volatile uint64_t* fp = (uint64_t*)__builtin_frame_address(0);
        fprintf(stderr, "[agc] WaitRegMem a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx | ret=0x%llx a6=0x%llx a7=0x%llx\n",
            (unsigned long long)a1,(unsigned long long)a2,(unsigned long long)a3,(unsigned long long)a4,(unsigned long long)a5,
            (unsigned long long)fp[1],(unsigned long long)fp[2],(unsigned long long)fp[3]);
    }
    uint32_t* cmd; if (!begin_packet(a0, 9, IT_NOP, R_WAIT_MEM_64, &cmd)) return 0;
    for (int i = 1; i < 9; i++) cmd[i] = 0; return (uint64_t)(uintptr_t)cmd;
}
HLE(agc_dcb_set_flip) {  // (buf, video_out_handle, display_buffer_index, flip_mode, flip_arg) — 6 dw
    uint32_t* cmd; if (!begin_packet(a0, 6, IT_NOP, R_FLIP, &cmd)) return 0;
    cmd[1] = (uint32_t)a1; cmd[2] = (uint32_t)a2; cmd[3] = (uint32_t)a3;
    cmd[4] = (uint32_t)(a4 & 0xffffffffu); cmd[5] = (uint32_t)(a4 >> 32u);
    return (uint64_t)(uintptr_t)cmd;
}

// --- sceAgcCreateShader (NID f3dg2CSgRKY) — the shader-object constructor. ---------------------
//
// THE boot blocker's root (docs/GRAPHICS.md "register source"): Unity registers its ~36 built-in
// shaders at graphics init via eboot+0x14e74c0, which parses shader ELFs embedded in eboot rodata
// (e_machine=0xe0 EM_AMDGPU; sections .shader_header / .shader_text) and calls
// sceAgcCreateShader(&slot->shader, header, code) per shader — through the arg-validating wrapper
// eboot+0x3ae120 (rejects nulls + non-256-aligned code) that tail-jumps to this import. Our old
// stub returned 0 WITHOUT writing *dst, so every registry slot's shader stayed null — including
// [eboot+0x2048c60] (= slot base 0x2048c50 + 0x10), the "source" object whose null pointer flows
// through SetSource (eboot+0x3af400) into the register-context sub-objects and null-derefs at
// eboot+0x3b1562/0x3b5ea6. The Shader IS the register source: SetSource reads [src+0x08]
// (user_data), [src+0x28] (specials), [src+0x5a] (type) — the Shader field offsets below.
//
// Semantics per Kyty GraphicsCreateShader (Graphics.cpp:1432) — layout-verified against the real
// SDK blobs (file_header '1234', version 0x18, matching our captured gfx1030 shaders). The header's
// pointer fields are stored SELF-RELATIVE in the blob; the constructor relocates them in place
// (ptr += &ptr), binds the code pointer, and patches the shader-program base address into the
// leading SPI_SHADER_PGM_LO/HI sh-register pair. Beyond Kyty (which supports only ES/PS pairs and
// hard-aborts otherwise): we accept all five PGM pairs (PS/GS/ES/HS/CS), guard against double
// relocation, and never abort — unexpected shapes are logged and survive.
namespace {

struct AgcShaderRegister { uint32_t offset, value; };   // guest-visible (Kyty Shader.h:941)
struct AgcShaderUserData {           // guest-visible (Kyty Shader.h:911)
    uint16_t* direct_resource_offset;   // +0x00  self-relative until relocated
    void*     sharp_resource_offset[4]; // +0x08  (ShaderSharp*[4])
    uint16_t  eud_size_dw, srt_size_dw, direct_resource_count, sharp_resource_count[4];
};
struct AgcShader {                   // guest-visible (Kyty Shader.h:974)
    uint32_t           file_header;             // +0x00  '1234' = 0x34333231
    uint32_t           version;                 // +0x04  0x18
    AgcShaderUserData* user_data;               // +0x08  \.
    const void*        code;                    // +0x10   | SetSource consumes +0x08/+0x28/+0x5a
    AgcShaderRegister* cx_registers;            // +0x18   |
    AgcShaderRegister* sh_registers;            // +0x20   |
    void*              specials;                // +0x28  /
    void*              input_semantics;         // +0x30
    void*              output_semantics;        // +0x38
    uint32_t           header_size;             // +0x40
    uint32_t           shader_size;             // +0x44
    uint32_t           embedded_constant_buffer_size_dqw; // +0x48
    uint32_t           target;                  // +0x4c
    uint32_t           num_input_semantics;     // +0x50
    uint16_t           scratch_size_dw_per_thread; // +0x54
    uint16_t           num_output_semantics;    // +0x56
    uint16_t           special_sizes_bytes;     // +0x58
    uint8_t            type;                    // +0x5a
    uint8_t            num_cx_registers;        // +0x5b
    uint8_t            num_sh_registers;        // +0x5c
};
static_assert(offsetof(AgcShader, user_data) == 0x08 && offsetof(AgcShader, code) == 0x10 &&
              offsetof(AgcShader, specials) == 0x28 && offsetof(AgcShader, type) == 0x5a &&
              offsetof(AgcShader, num_sh_registers) == 0x5c, "AgcShader must match the SDK blob layout");

// AGC error codes (observed in the eboot wrapper at 0x3ae120).
constexpr uint64_t kAgcErrInvalidArg = 0x8a6c000aull;

// Shaders whose headers we already relocated (the fixup is not idempotent), and the registry of all
// created shaders — the AGC->Vulkan pipeline consumes this to find shader code by PGM base.
std::unordered_set<const void*>& agc_relocated() { static std::unordered_set<const void*> s; return s; }
std::vector<const AgcShader*>&   agc_shaders()   { static std::vector<const AgcShader*> v; return v; }

// Relocate one self-relative pointer field in place (no-op for null fields).
template <class T> inline void agc_fix_ptr(T*& m) {
    if (m) m = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(m) + reinterpret_cast<uintptr_t>(&m));
}

}  // namespace

// Exposed for tests: how many shaders the guest successfully registered.
extern "C" size_t prosper_agc_shader_count() { return agc_shaders().size(); }

// Look up a registered shader header by its bound code address (the SHADER_PGM base the executor
// recovers from the sh registers). Returns the AgcShader* (layout-compatible with gpu::AgcShaderHeader
// — file_header@0, user_data@0x08, code@0x10, type@0x5a) so the GPU executor can build the shader's
// resource table from its user_data descriptors. Null if no registered shader binds that code. The
// registry is append-only during boot and CreateShader is main-thread; reads here are on the submit
// thread — benign for a lookup (worst case a just-registered shader isn't seen yet, then it retries).
extern "C" const void* prosper_agc_shader_header_for_code(uint64_t code_addr) {
    if (!code_addr) return nullptr;
    for (const AgcShader* h : agc_shaders())
        if (h && (uint64_t)(uintptr_t)h->code == code_addr) return h;
    return nullptr;
}

// Bounded RE probe (PROSPER_PIPETRACE): log the raw pointer args of the shader/pipeline construction
// calls, so their argument/return pointers can be diffed against a later fault object (e.g. the
// eboot+0xba6e08 pipeline object's [+0x140] companion). Purely observational.
static void pipetrace(const char* fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    if (getenv("PROSPER_PIPETRACE"))
        fprintf(stderr, "[pipe] %s a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n", fn,
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
}

static void pipetrace_shader_user_data(const char* tag, const AgcShader* h) {
    if (!getenv("PROSPER_PIPETRACE")) return;
    if (!h) { fprintf(stderr, "  [ud] %s shader=(null)\n", tag); return; }
    const AgcShaderUserData* ud = h->user_data;
    if (!ud) { fprintf(stderr, "  [ud] %s shader=%p user_data=(null)\n", tag, (const void*)h); return; }

    fprintf(stderr,
            "  [ud] %s shader=%p user_data=%p eud=%u srt=%u direct_count=%u sharp_counts={%u,%u,%u,%u}\n",
            tag, (const void*)h, (const void*)ud, ud->eud_size_dw, ud->srt_size_dw,
            ud->direct_resource_count, ud->sharp_resource_count[0], ud->sharp_resource_count[1],
            ud->sharp_resource_count[2], ud->sharp_resource_count[3]);

    if (ud->direct_resource_offset && ud->direct_resource_count && ud->direct_resource_count <= 64) {
        fprintf(stderr, "    direct offsets:");
        for (uint16_t i = 0; i < ud->direct_resource_count && i < 16; i++)
            fprintf(stderr, " type%u=%04x", i, ud->direct_resource_offset[i]);
        if (ud->direct_resource_count > 16) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }

    for (int imm = 0; imm < 4; imm++) {
        const uint16_t count = ud->sharp_resource_count[imm];
        auto* raw = (const uint16_t*)ud->sharp_resource_offset[imm];
        if (!raw || count == 0 || count > 64) continue;
        const char* kind = (imm == 0) ? "texture" : (imm == 2) ? "sampler" :
                           (imm == 3) ? "storage" : "sharp1";
        fprintf(stderr, "    sharp[%d] %s:", imm, kind);
        for (uint16_t slot = 0; slot < count && slot < 16; slot++) {
            uint16_t v = raw[slot];
            fprintf(stderr, " slot%u={off=%04x,size=%u}", slot, (unsigned)(v & 0x7fffu), (unsigned)(v >> 15));
        }
        if (count > 16) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
    }
}

HLE(agc_create_shader) {  // (Shader** dst, void* header, const void* code)
    pipetrace("CreateShader", a0, a1, a2, a3, a4, a5);
    auto** dst   = (AgcShader**)(uintptr_t)a0;
    auto*  h     = (AgcShader*)(uintptr_t)a1;
    auto*  code  = (const void*)(uintptr_t)a2;
    if (!dst || !h || !code) return kAgcErrInvalidArg;

    if (h->file_header != 0x34333231u || h->version != 0x18u) {
        fprintf(stderr, "[agc] CreateShader: unexpected header magic=0x%08x version=0x%x (want '1234'/0x18)\n",
                h->file_header, h->version);
        return kAgcErrInvalidArg;   // wrong layout model -- fail loudly rather than corrupt the blob
    }

    if (agc_relocated().insert(h).second) {
        agc_fix_ptr(h->cx_registers);
        agc_fix_ptr(h->sh_registers);
        agc_fix_ptr(h->user_data);
        agc_fix_ptr(h->specials);
        agc_fix_ptr(h->input_semantics);
        agc_fix_ptr(h->output_semantics);
        if (h->user_data) {
            agc_fix_ptr(h->user_data->direct_resource_offset);
            for (auto& sharp : h->user_data->sharp_resource_offset) agc_fix_ptr(sharp);
        }
    }
    h->code = code;

    // Patch the shader-program base into the leading SPI/COMPUTE PGM_LO/HI register pair. The blob
    // pre-seeds the pair's offsets, so match any known stage rather than gating on `type` (Kyty
    // handles only ES/PS and aborts otherwise; the game also registers other stages).
    using namespace prosper::agc::Pm4;
    static constexpr uint32_t kPgmPairs[][2] = {
        {SPI_SHADER_PGM_LO_PS, SPI_SHADER_PGM_HI_PS}, {SPI_SHADER_PGM_LO_ES, SPI_SHADER_PGM_HI_ES},
        {SPI_SHADER_PGM_LO_GS, SPI_SHADER_PGM_HI_GS}, {SPI_SHADER_PGM_LO_HS, SPI_SHADER_PGM_HI_HS},
        {COMPUTE_PGM_LO,       COMPUTE_PGM_HI},
    };
    uint64_t base = (uint64_t)(uintptr_t)code;
    bool patched = false;
    if (h->sh_registers && h->num_sh_registers >= 2) {
        for (const auto& pair : kPgmPairs) {
            if (h->sh_registers[0].offset == pair[0] && h->sh_registers[1].offset == pair[1]) {
                h->sh_registers[0].value = (uint32_t)((base >> 8) & 0xffffffffu);
                h->sh_registers[1].value = (uint32_t)((base >> 40) & 0xffu);
                patched = true;
                break;
            }
        }
    }
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] CreateShader #%zu: header=%p code=%p type=%u target=0x%x size=%u "
                        "cx=%u sh=%u pgm_patched=%d\n",
                agc_shaders().size(), (void*)h, (void*)code, h->type, h->target, h->shader_size,
                h->num_cx_registers, h->num_sh_registers, (int)patched);

    // Semantic surfacing probe (PROSPER_PIPETRACE): logs the shader's input/output vertex semantics.
    // CORRECTION (2026-07-05, fresh 3-agent RE, ground-truth via eboot x86 disasm): an earlier theory
    // blamed the eboot+0xba6e08 GfxDevice fault on "empty semantics leaving the reflection table empty
    // via a VS->PS linkage predicate at eboot+0xd58710." That was WRONG. eboot+0xd58710 is Unity's
    // generic SafeBinaryRead::Transfer (asset deserialization; ~4955 call sites; references the
    // "SafeBinaryRead::BeginTransfer name mismatch" string) — unrelated to the crash. The 0xba6e08 fault
    // is a GfxDevice GC / deferred-release DRAIN pass (fn eboot+0xba6720) that, for a category-{5,9,15}
    // pipeline whose processed-flag [+0x1a0]==0, consults its GPU companion [+0x140] to decide keep-vs-
    // release — and ours is null. The companion is built eagerly on a normal PS5 path (writers in the
    // eboot 0xb3xxxx GfxDevice module); the fix is to make that companion get built (an AGC/GPU-resource
    // gap), NOT to surface semantics. See docs/GFXDEVICE_BRINGUP_PROBLEM.md §"2026-07-05 correction".
    if (getenv("PROSPER_PIPETRACE")) {
        fprintf(stderr, "  [sem] type=%u num_in=%u num_out=%u in_sem=%p out_sem=%p\n",
                h->type, h->num_input_semantics, h->num_output_semantics,
                (void*)h->input_semantics, (void*)h->output_semantics);
        auto dump = [](const char* tag, const void* arr, uint32_t n) {
            if (!arr || n == 0 || n > 64) { fprintf(stderr, "    %s: (none)\n", tag); return; }
            const uint32_t* s = (const uint32_t*)arr;
            fprintf(stderr, "    %s[%u]:", tag, n);
            for (uint32_t i = 0; i < n && i < 8; i++) fprintf(stderr, " %08x", s[i]);
            fprintf(stderr, "\n");
        };
        dump("in ", h->input_semantics,  h->num_input_semantics);
        dump("out", h->output_semantics, (uint32_t)h->num_output_semantics);
        // Raw header count region (0x50..0x5f) to confirm counts are read from the real blob, not a
        // relocation artifact: num_input_semantics u32@0x50, num_output_semantics u16@0x56.
        const uint8_t* hb = (const uint8_t*)h;
        fprintf(stderr, "    raw[0x50..0x5f]:");
        for (int i = 0x50; i < 0x60; i++) fprintf(stderr, " %02x", hb[i]);
        fprintf(stderr, "\n");
    }
    pipetrace_shader_user_data("CreateShader", h);

    agc_shaders().push_back(h);
    *dst = h;               // <- the write our old stub omitted; populates the shader-registry slots
    return 0;
}

// --- Shader-derived pipeline-state constructors + register-bank plumbing. ----------------------
//
// With CreateShader real, the eboot's register-context machinery runs its true path: it allocates
// "data packets" in the game's own Dcb and points each register-set sub-object's banks
// ([sub+0x10]/[sub+0x18]) at the packet payload. The leaf imports it needs are these four
// (semantics per Kyty Graphics.cpp:1606-1762, generalized — see each note).
namespace {

struct AgcShaderSemantic {           // guest-visible 32-bit bitfield (Kyty Shader.h:958)
    uint32_t semantic         : 8;
    uint32_t hardware_mapping : 8;
    uint32_t size_in_elements : 4;
    uint32_t is_f16           : 2;
    uint32_t is_flat_shaded   : 1;
    uint32_t is_linear        : 1;
    uint32_t is_custom        : 1;
    uint32_t static_vb_index  : 1;
    uint32_t static_attribute : 1;
    uint32_t reserved         : 1;
    uint32_t default_value    : 2;
    uint32_t default_value_hi : 2;
};
static_assert(sizeof(AgcShaderSemantic) == 4, "semantic must be one dword");

struct AgcShaderSpecialRegs {        // guest-visible (Kyty Shader.h:947); pointed to by Shader+0x28
    AgcShaderRegister ge_cntl;                  // +0x00
    AgcShaderRegister vgt_shader_stages_en;     // +0x08
    uint32_t          dispatch_modifier;        // +0x10
    uint16_t          user_data_range_start;    // +0x14  (consumed by SetSource @ eboot+0x3af400)
    uint16_t          user_data_range_end;      // +0x16
    uint64_t          draw_modifier;            // +0x18  (packed ShaderDrawModifier bitfields)
    AgcShaderRegister vgt_gs_out_prim_type;     // +0x20
    AgcShaderRegister ge_user_vgpr_en;          // +0x28
};
static_assert(offsetof(AgcShaderSpecialRegs, vgt_gs_out_prim_type) == 0x20, "specials layout");

}  // namespace

// sceAgcGetDataPacketPayloadAddress (V++UgBtQhn0) — called from INSIDE eboot's static AGC code
// (register-bank prepare, eboot+0x3af040 path, callsite +0x3af170): resolve a data packet built in
// the Dcb to its payload address, which becomes the register bank. Kyty: *addr = cmd + 2 (2-dword
// packet header), type must be 1. We accept other types with a log instead of aborting.
HLE(agc_get_data_packet_payload) {  // (uint32_t** addr, uint32_t* cmd, int type)
    auto** addr = (uint32_t**)(uintptr_t)a0;
    auto*  cmd  = (uint32_t*)(uintptr_t)a1;
    if (!addr || !cmd) return kAgcErrInvalidArg;
    if (a2 != 1 && getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] GetDataPacketPayloadAddress: unexpected type=%d\n", (int)a2);
    *addr = cmd + 2;
    return 0;
}

// sceAgcCbSetShRegisterRangeDirect (n2fD4A+pb+g) — append an IT_SET_SH_REG packet covering
// [offset, offset+num) with the given values (zeros if values==null); returns the packet pointer
// (the game patches/fills it afterwards). Kyty also emits a 2-dword NOP marker (payload 0x6875000d)
// first, mirroring the real library's output; we keep it for stream fidelity (NOP = inert).
HLE(agc_cb_set_sh_register_range_direct) {  // (buf, offset, values, num_values)
    uint32_t num = (uint32_t)a3;
    if (!a0 || !num) return 0;
    uint32_t* marker; if (begin_packet(a0, 2, IT_NOP, 0, &marker)) marker[1] = 0x6875000du;
    uint32_t* cmd;    if (!begin_packet(a0, num + 2, IT_SET_SH_REG, 0, &cmd)) return 0;
    cmd[1] = (uint32_t)a1;
    auto* values = (const uint32_t*)(uintptr_t)a2;
    if (values) memcpy(cmd + 2, values, (size_t)num * 4);
    else        memset(cmd + 2, 0, (size_t)num * 4);
    return (uint64_t)(uintptr_t)cmd;
}

// sceAgcCreatePrimState (D9sr1xGUriE) — derive the primitive-pipeline registers from the bound
// geometry shader's "specials" block. Kyty hard-asserts hs==null and exact register offsets; we
// copy the specials through and log deviations instead (tessellation would arrive via hs).
HLE(agc_create_prim_state) {  // (cx_regs, uc_regs, hs, gs, prim_type)
    pipetrace("CreatePrimState", a0, a1, a2, a3, a4, a5);
    auto* cx = (AgcShaderRegister*)(uintptr_t)a0;
    auto* uc = (AgcShaderRegister*)(uintptr_t)a1;
    auto* gs = (const AgcShader*)(uintptr_t)a3;
    if (!cx || !uc || !gs || !gs->specials) return kAgcErrInvalidArg;
    if (a2 && getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[agc] CreatePrimState: hs shader present (tessellation) — not yet modeled\n");
    const auto* sp = (const AgcShaderSpecialRegs*)gs->specials;
    using namespace prosper::agc::Pm4;
    cx[0] = sp->vgt_shader_stages_en;
    cx[1] = sp->vgt_gs_out_prim_type;
    uc[0] = sp->ge_cntl;
    uc[1] = sp->ge_user_vgpr_en;
    uc[2] = {VGT_PRIMITIVE_TYPE, (uint32_t)a4};
    return 0;
}

// sceAgcCreateInterpolantMapping (HV4j+E0MBHE) — build the 32 SPI_PS_INPUT_CNTL_* registers wiring
// PS inputs to GS/VS output parameter slots. Generalized beyond Kyty (which asserts Unity's exact
// identity layout): match each PS input to the GS output with the same semantic id and use that
// output's parameter slot, with the flat-shade bit (0x400) from the PS side. Falls back to the
// identity mapping (slot i) when there is nothing to match — which reproduces Kyty's behaviour on
// the layouts Kyty supports.
HLE(agc_create_interpolant_mapping) {  // (ShaderRegister regs[32], const Shader* gs, const Shader* ps)
    pipetrace("CreateInterpolantMapping", a0, a1, a2, a3, a4, a5);
    auto* regs = (AgcShaderRegister*)(uintptr_t)a0;
    auto* gs   = (const AgcShader*)(uintptr_t)a1;
    auto* ps   = (const AgcShader*)(uintptr_t)a2;
    if (!regs || !gs) return kAgcErrInvalidArg;
    using namespace prosper::agc::Pm4;
    const auto* gs_out = (const AgcShaderSemantic*)gs->output_semantics;
    const auto* ps_in  = ps ? (const AgcShaderSemantic*)ps->input_semantics : nullptr;
    uint32_t n = ps ? ps->num_input_semantics : (uint32_t)gs->num_output_semantics;
    if (getenv("PROSPER_PIPETRACE"))
        fprintf(stderr, "  [interp] gs.num_out=%u ps=%p ps.num_in=%u -> mapping %u interpolants\n",
                (uint32_t)gs->num_output_semantics, (void*)ps,
                ps ? ps->num_input_semantics : 0u, n);
    pipetrace_shader_user_data("CreateInterpolant.gs", gs);
    pipetrace_shader_user_data("CreateInterpolant.ps", ps);
    for (uint32_t i = 0; i < 32; i++) {
        regs[i].offset = SPI_PS_INPUT_CNTL_0 + i;
        uint32_t value = 0;
        if (i < n) {
            uint32_t slot = i; bool flat = false;
            if (ps_in && gs_out) {
                flat = ps_in[i].is_flat_shaded != 0;
                for (uint32_t j = 0; j < gs->num_output_semantics; j++)
                    if (gs_out[j].semantic == ps_in[i].semantic) { slot = gs_out[j].hardware_mapping; break; }
            } else if (gs_out) {
                slot = gs_out[i].hardware_mapping;
            }
            value = slot | (flat ? 0x400u : 0u);
        }
        regs[i].value = value;
    }
    return 0;
}

// --- sceAgcDriverSubmitDcb (AgcDriver NID UglJIZjGssM) — the GPU submission point. -------------
//
// The game hands us a Packet {dcb_addr, dw_num} (Kyty Gen5Driver::Packet, Graphics.cpp:2168). We
// replay the PM4 stream through the CommandProcessor into a persistent GpuState — register files +
// draw list — which the AGC->Vulkan translation (render_state/vk_translate) consumes. No rendering
// happens here yet; this is the semantic execution of the submit, observable via PROSPER_GFXLOG and
// prosper_agc_submit_stats(). The Dcb address is a guest pointer, valid 1:1 in-process.
namespace {
gpu::GpuState& agc_gpu_state() { static gpu::GpuState st; return st; }
uint64_t g_submit_count = 0;
}

// EOP completion signaling (hle_kernel_time.cpp): fire the GPU end-of-pipe events the game registered
// via sceGnmAddEqEvent. Our fold is synchronous, so a completed SubmitDcb == the GPU pipe having drained.
void prosper_eq_trigger_eop();

extern "C" void prosper_agc_submit_stats(uint64_t* submits, uint64_t* draws) {
    if (submits) *submits = g_submit_count;
    if (draws)   *draws   = (uint64_t)agc_gpu_state().draws.size();
}

HLE(agc_driver_submit_dcb) {  // (const Packet* packet)
    struct Packet { uint32_t* addr; uint32_t dw_num; uint8_t pad[4]; };
    const auto* p = (const Packet*)(uintptr_t)a0;
    if (!p || !p->addr || !p->dw_num) return kAgcErrInvalidArg;
    // Reset the per-submit draw list BEFORE folding this Dcb. The folded GpuState is process-lifetime and
    // its register files (cx/sh/uc) persist across submits (as on real hardware), but its `draws` vector
    // must NOT accumulate — otherwise it grows unbounded and every later frame re-renders stale geometry.
    // Clearing here (not after) keeps this submit's draws inspectable once the handler returns. (Ported
    // from PRs #31/#32.)
    agc_gpu_state().draws.clear();
    size_t applied = gpu::run_command_buffer(p->addr, p->dw_num, agc_gpu_state());
    g_submit_count++;
    // The submit has "completed" (synchronous fold): fire any registered GPU EOP events. Inert unless the
    // game called sceGnmAddEqEvent (b0xyllnVY-I); the RELEASE_MEM label write already happened in apply().
    prosper_eq_trigger_eop();
    // Stage A: if a live renderer has been registered (runtime binary wires a Vulkan device) and this
    // submit accumulated draws, execute the folded GpuState and present the frame. Inert (returns false)
    // on the pure-HLE path until a device is registered, so it never perturbs the existing boot behavior.
    // PROSPER_RENDER_EVERY=K: render only every Kth draw-carrying submit (default 1). Under llvmpipe a
    // 1080p render is ~10-20 s and execute_and_present is SYNCHRONOUS here, throttling the frame loop to
    // a crawl — so a distant scene (the cutscene, thousands of frames in) would take hours. Sampling
    // keeps the game running near full speed while still producing periodic frames of the live scene.
    static const unsigned render_every = [] {
        const char* e = getenv("PROSPER_RENDER_EVERY"); long v = e ? atol(e) : 1; return v > 0 ? (unsigned)v : 1u; }();
    static unsigned draw_submits = 0;
    if (gpu::have_submit_renderer() && !agc_gpu_state().draws.empty() && (draw_submits++ % render_every) == 0) {
        uint32_t w = gpu::present_width(), h = gpu::present_height();
        // PROSPER_RENDER_SCALE=N: render at 1/N resolution. execute_and_present is SYNCHRONOUS on the
        // guest submit thread; a full 1080p llvmpipe render blocks it ~15 s, and while it is blocked in
        // host Vulkan the guest GC's stop-the-world can't get its ack -> the collection races and aborts
        // ("Unexpected state"), so the game crashes long before reaching a distant scene. A 1/4 render
        // (~1 s) shrinks that window enough to run deep into the cutscene while still capturing a frame.
        static const uint32_t scale = [] {
            const char* e = getenv("PROSPER_RENDER_SCALE"); long v = e ? atol(e) : 1; return v > 0 ? (uint32_t)v : 1u; }();
        if (scale > 1) { w = (w / scale) & ~1u; h = (h / scale) & ~1u; if (!w) w = 2; if (!h) h = 2; }
        bool presented = gpu::execute_and_present(agc_gpu_state(), w, h);
        if (presented && getenv("PROSPER_GFXLOG"))
            fprintf(stderr, "[agc] SubmitDcb #%llu: executed %zu draws -> presented %ux%u frame\n",
                    (unsigned long long)g_submit_count, agc_gpu_state().draws.size(), w, h);
    }
    if (getenv("PROSPER_GFXLOG")) {
        fprintf(stderr, "[agc] SubmitDcb #%llu: %u dwords -> %zu packets applied (draws so far: %zu)\n",
                (unsigned long long)g_submit_count, p->dw_num, applied, agc_gpu_state().draws.size());
        // Dump each packet's kind + first payload dwords so we can see whether the game embeds a
        // completion-label write (WriteData/ReleaseMem/EventWrite) or a GPU-side wait in the stream.
        std::vector<gpu::Pm4Command> ops; gpu::decode_pm4(p->addr, p->dw_num, ops);
        static const char* kKindName[] = {"DrawReset","WaitFlipDone","SetShRegDirect","SetRegsIndirect",
            "SetIndexType","DrawIndexAuto","EventWrite","AcquireMem","WriteData","WaitRegMem","Flip",
            "ReleaseMem","Unknown"};
        for (auto& c : ops) {
            uint32_t k = (uint32_t)c.kind; const char* nm = k < 13 ? kKindName[k] : "?";
            fprintf(stderr, "[agc]   pkt op=0x%02x r=0x%02x len=%u kind=%s pl0=0x%08x pl1=0x%08x pl2=0x%08x\n",
                    c.op, c.r, c.len, nm,
                    c.len > 1 ? c.payload[0] : 0, c.len > 2 ? c.payload[1] : 0, c.len > 3 ? c.payload[2] : 0);
        }
    }
    return 0;
}

// --- Indirect-register patch helpers: modify a packet returned by a Set*RegsIndirect call.
// cmd = a0 (that returned pointer). Old stub returned 0 for Set*RegsIndirect -> these wrote to null. -
HLE(agc_patch_set_address) {  // (cmd, regs): cmd[2..3] = regs vaddr
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    cmd[2] = (uint32_t)(a1 & 0xffffffffu); cmd[3] = (uint32_t)(a1 >> 32u); return 0;
}
HLE(agc_patch_add_registers) {  // (cmd, num_regs): cmd[1] += num_regs
    auto* cmd = (uint32_t*)(uintptr_t)a0; if (!cmd) return 0;
    cmd[1] += (uint32_t)a1; return 0;
}

void register_agc_hle() {
    #define RN(nid, fn) Hle::register_fn(nid, (HleFn)(fn), nid)
    RN("f3dg2CSgRKY", agc_create_shader);   // sceAgcCreateShader — populates the shader registry
    RN("V++UgBtQhn0", agc_get_data_packet_payload);          // data packet -> register-bank payload
    RN("n2fD4A+pb+g", agc_cb_set_sh_register_range_direct);  // SET_SH_REG range packet
    RN("D9sr1xGUriE", agc_create_prim_state);                // prim registers from gs specials
    RN("HV4j+E0MBHE", agc_create_interpolant_mapping);       // SPI_PS_INPUT_CNTL_* wiring
    RN("TRO721eVt4g", agc_dcb_reset_queue);
    RN("MWiElSNE8j8", agc_dcb_wait_safe_for_rendering);
    RN("ZvwO9euwYzc", agc_dcb_set_cx_regs_indirect);
    RN("-HOOCn0JY48", agc_dcb_set_sh_regs_indirect);
    RN("hvUfkUIQcOE", agc_dcb_set_uc_regs_indirect);
    RN("GIIW2J37e70", agc_dcb_set_index_size);
    RN("Yw0jKSqop+E", agc_dcb_draw_index_auto);
    RN("aJf+j5yntiU", agc_dcb_event_write);
    RN("57labkp+rSQ", agc_dcb_acquire_mem);
    RN("wr23dPKyWc0", agc_cb_release_mem);
    RN("i1jyy49AjXU", agc_dcb_write_data);
    RN("VmW0Tdpy420", agc_dcb_wait_reg_mem);
    // Indirect-register patch helpers (SetAddress patches cmd[2..3]; AddRegisters patches cmd[1]).
    RN("vcmNN+AAXnY", agc_patch_set_address);   RN("d-6uF9sZDIU", agc_patch_add_registers);   // Cx
    RN("Qrj4c+61z4A", agc_patch_set_address);   RN("z2duB-hHQSM", agc_patch_add_registers);   // Sh
    RN("6lNcCp+fxi4", agc_patch_set_address);   RN("vRoArM9zaIk", agc_patch_add_registers);   // Uc
    RN("YUeqkyT7mEQ", agc_dcb_set_flip);        // sceAgcDcbSetFlip (Kyty LibGraphicsDriver.cpp:98)
    RN("UglJIZjGssM", agc_driver_submit_dcb);   // sceAgcDriverSubmitDcb -> CommandProcessor replay
    #undef RN
}

}  // namespace prosper
