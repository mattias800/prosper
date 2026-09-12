// test_buf_op_census_disposition — a REFUSED buffer instruction must not print the same `how=` word
// as an emitted one.
//
// Why this exists (#3579). The `[buf-op]` census exists to separate three outcomes that are
// indistinguishable in the emitted SPIR-V: an op folded away against an empty descriptor, an op that
// never reached the emitter, and an op that reached it and was REFUSED. `buf_op.how` was set to the
// optimistic word `"resolved"` the moment a V# resolved — before the format decode runs — and several
// later reject paths returned without touching it. Those refusals printed `resolved`, i.e. the census
// reported failures as successes on exactly the distinction it exists to make.
//
// The word is now `descriptor-resolved`, which is what that early assignment actually establishes, and
// each reject names itself. The arms below drive one shader per refusal through the real diagnostic
// path and read the line off stderr.
//
// THE POSITIVE CONTROL IS LOAD-BEARING: without an arm proving that a healthy op still reaches
// `descriptor-resolved` and prints no `reject-` string at all, every assertion here would also pass
// against an emitter that refused everything.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#if defined(_MSC_VER)
#include <io.h>
#define PROSPER_DUP   _dup
#define PROSPER_DUP2  _dup2
#define PROSPER_CLOSE _close
#else
#include <unistd.h>
#define PROSPER_DUP   dup
#define PROSPER_DUP2  dup2
#define PROSPER_CLOSE close
#endif
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

// Captures one recompile's stderr and PUTS STDERR BACK (#3598). Without the restore the file is
// unlinked below and every later line of the test -- including a crash message -- writes to a
// descriptor nobody can read, which is a silent loss precisely when something has gone wrong.
static std::string recompile_capturing_stderr(const uint32_t* code, size_t dwords,
                                              const ShaderResourceTable* rt,
                                              const char* scratch) {
    std::fflush(stderr);
    const int saved_stderr = PROSPER_DUP(fileno(stderr));
    if (!std::freopen(scratch, "w+", stderr)) {
        printf("  [FAIL] cannot redirect stderr\n"); fails++;
        if (saved_stderr >= 0) PROSPER_CLOSE(saved_stderr);
        return {};
    }
    (void)recompile_compute(code, dwords, rt, ComputeShaderConfig{});
    std::fflush(stderr);
    std::string text;
    if (FILE* f = std::fopen(scratch, "rb")) {
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
        std::fclose(f);
    }
    std::remove(scratch);
    if (saved_stderr >= 0) {
        std::fflush(stderr);
        PROSPER_DUP2(saved_stderr, fileno(stderr));
        PROSPER_CLOSE(saved_stderr);
    }
    return text;
}

static bool has(const std::string& h, const char* n) { return h.find(n) != std::string::npos; }

// One resource at s[8:11], whose FORMAT is the variable each arm changes. A MUBUF format op takes its
// type from the resolved V#, so the descriptor is how a test selects which decode path runs.
static ShaderResourceTable table_with_format(DataFormat fmt, uint32_t ncomp) {
    ShaderResourceTable rt;
    ShaderResource r{};
    r.cls = ResourceClass::VertexBuffer;
    r.format = fmt;
    r.num_components = ncomp;
    r.binding = 2;
    r.gpu_addr = 0x10000;
    r.size = 4096;
    r.stride = 16;
    r.sgpr_base = 8;
    rt.resources.push_back(r);
    return rt;
}

int main() {
    printf("== test_buf_op_census_disposition ==\n");
    set_test_env("PROSPER_DBG", "1");

    // buffer_load_format_xyzw v[0:3], v0, s[8:11], 0 idxen   (gfx1030 llvm-mc)
    const uint32_t load_xyzw[] = { 0xE00C2000u, 0x80020000u, 0xBF810000u };
    // buffer_store_format_xyzw v[0:3], v0, s[8:11], 0 idxen  — opcode 3 -> 7 in d0[24:18].
    const uint32_t store_xyzw[] = { 0xE01C2000u, 0x80020000u, 0xBF810000u };
    // The same store with inst_offset:1, so the packed components are not dword-aligned.
    const uint32_t store_xyzw_unaligned[] = { 0xE01C2001u, 0x80020000u, 0xBF810000u };

    // --- Positive control, first: a healthy Float32x4 fetch -------------------------------------
    // If this arm does not reach `descriptor-resolved` with no reject string, nothing below means
    // anything: the assertions would be satisfied by an emitter that refused every buffer op.
    {
        ShaderResourceTable rt = table_with_format(DataFormat::Float32, 4);
        const std::string log = recompile_capturing_stderr(
            load_xyzw, std::size(load_xyzw), &rt, "buf_op_census_control.log");
        CHECK(has(log, "[buf-op]"), "control: the census line is emitted at all under PROSPER_DBG");
        CHECK(has(log, "descriptor-resolved"),
              "control: a healthy Float32x4 format load reports descriptor-resolved");
        CHECK(!has(log, "reject-"),
              "CONTROL: a healthy format load prints NO reject- string "
              "(without this, every arm below would pass against a refuse-everything emitter)");
    }

    // --- Arm 1: a V# whose FORMAT does not decode ------------------------------------------------
    {
        ShaderResourceTable rt = table_with_format(DataFormat::Unknown, 4);
        const std::string log = recompile_capturing_stderr(
            load_xyzw, std::size(load_xyzw), &rt, "buf_op_census_unknown.log");
        CHECK(has(log, "reject-unknown-format"),
              "#3579: an undecodable buffer FORMAT names itself (was: resolved)");
        // The DISCRIMINATING half. `how` is the last field on the line, so " resolved\n" is exactly
        // what this op used to print -- a bare success word for a refused instruction. Asserting the
        // absence of the reject string alone would pass against the unfixed emitter too, since the
        // word `descriptor-resolved` simply did not exist there.
        CHECK(!has(log, " resolved\n"),
              "#3579: ...and no longer reports the refusal with a bare success word");
    }

    // --- Arm 2: a format with a size but no conversion (USCALED is a dead enumerator) ------------
    // Uscaled8 reports 1 byte per component, so `packed` is true, but it is not half, not integer and
    // has no normalization divisor -- the `[mubuf-badfmt]` shape.
    {
        ShaderResourceTable rt = table_with_format(DataFormat::Uscaled8, 4);
        const std::string log = recompile_capturing_stderr(
            load_xyzw, std::size(load_xyzw), &rt, "buf_op_census_badfmt.log");
        CHECK(has(log, "reject-badfmt"),
              "#3579: the [mubuf-badfmt] refusal names itself in the census");
    }

    // --- Arm 3: a packed store whose components are not dword-aligned ---------------------------
    {
        ShaderResourceTable rt = table_with_format(DataFormat::Unorm8, 4);
        const std::string log = recompile_capturing_stderr(
            store_xyzw_unaligned, std::size(store_xyzw_unaligned), &rt,
            "buf_op_census_unaligned.log");
        CHECK(has(log, "reject-unaligned"),
              "#3579: the [mubuf-unaligned] refusal names itself in the census");
    }

    // --- Arm 4: the packed-word typed store is now EMITTED, not refused (#3575) -------------------
    // This arm deliberately asserts the opposite of what it would have a commit ago. The refusal it
    // used to pin is gone because the store is implemented, and a census arm that preserved an
    // obsolete refusal expectation would be pinning the emitter's past rather than its behaviour.
    {
        ShaderResourceTable rt = table_with_format(DataFormat::Float10_11_11, 3);
        const std::string log = recompile_capturing_stderr(
            store_xyzw, std::size(store_xyzw), &rt, "buf_op_census_packedword.log");
        CHECK(!has(log, "reject-"),
              "#3575: a 10_11_11 typed store is emitted, not refused");
        CHECK(has(log, "descriptor-resolved"),
              "#3575: ...and the census says so");
    }

    // --- Arm 5: the FIFTH reject path, which was the headline of #3579 and had no arm ------------
    // An MTBUF whose 7-bit BUF_FMT does not decode returns four lines below the optimistic
    // assignment, inside the same `if (res)` block, and printed the bare success word exactly like
    // the other four. It is named separately from the MUBUF case because the two read the format from
    // different places -- instruction vs descriptor -- and a census that merged them could not say
    // which source was undecodable.
    //
    // tbuffer_load_format_xyzw v[32:35], v8, s[8:11], 0 idxen format:3 (a USCALED code, which
    // `rdna2_buffer_format` has no case for -- see #3585/#3586). Opcode 3 in d0[18:16], format 3 in
    // d0[25:19].
    {
        const uint32_t mtbuf_bad_fmt[] = { 0xE81B2000u, 0x80020008u, 0xBF810000u };
        // MTBUF resolves its SRSRC through the EXACT per-fetch entry rather than the SGPR base, so
        // the resource needs this instruction's own pc as provenance; without it the op rejects
        // upstream as `mubuf-unresolved` and never reaches the format decode under test.
        ShaderResourceTable rt = table_with_format(DataFormat::Float32, 4);
        rt.resources[0].fetch_pc = 0;
        const std::string log = recompile_capturing_stderr(
            mtbuf_bad_fmt, std::size(mtbuf_bad_fmt), &rt, "buf_op_census_mtbuf.log");
        CHECK(has(log, "reject-mtbuf-unknown-format"),
              "#3579: an MTBUF whose instruction-supplied BUF_FMT does not decode names ITSELF, "
              "separately from the descriptor-supplied MUBUF case");
        CHECK(!has(log, " resolved\n"),
              "#3579: ...and does not report that refusal with a bare success word");
    }

    // --- Arm 6: the partial packed-word store refusal (#3594) --------------------------------------
    // All fields share one dword, so writing a subset is a read-modify-write whose semantics are not
    // established. This arm exists so the refusal cannot be quietly widened away: the implementation
    // that landed in #3575 is one predicate change from emitting it.
    {
        // buffer_store_format_xy through a 3-component 10_11_11 -- two of three fields.
        const uint32_t store_xy[] = { 0xE0142000u, 0x80020000u, 0xBF810000u };
        ShaderResourceTable rt = table_with_format(DataFormat::Float10_11_11, 3);
        const std::string log = recompile_capturing_stderr(
            store_xy, std::size(store_xy), &rt, "buf_op_census_partial.log");
        CHECK(has(log, "reject-packed-word-partial-store"),
              "#3575/#3594: a PARTIAL packed-word store still refuses, and names the refusal");
    }

    set_test_env("PROSPER_DBG", nullptr);
    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
