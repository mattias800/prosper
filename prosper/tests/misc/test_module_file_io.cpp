// #2687: fail each sizing operation on an otherwise valid module. GNU ld wrapping keeps the
// failure bounded and independent of host filesystem behavior; no enormous allocation is needed.
#include "fixtures/synth_prx.h"
#include "fixtures/test_scratch.h"
#include "hle/dispatch/nid.hpp"
#include "loader/linker.hpp"
#include <cerrno>
#include <cstdio>
#include <exception>
#include <string>

using namespace prosper;
namespace {
enum class Failure { None, EndSeek, Tell, StartSeek };
Failure failure = Failure::None;
bool observing = false;
FILE* observed_file = nullptr;
int end_seeks = 0, tells = 0, start_seeks = 0, reads = 0, closes = 0;
int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++fails; } \
                        else std::printf("[ok] %s\n", m); } while (0)
void begin(Failure value) {
    failure = value; observing = true; observed_file = nullptr;
    end_seeks = tells = start_seeks = reads = closes = 0;
}
} // namespace

extern "C" {
int __real_fseek(FILE*, long, int);
long __real_ftell(FILE*);
size_t __real_fread(void*, size_t, size_t, FILE*);
int __real_fclose(FILE*);
int __wrap_fseek(FILE* file, long offset, int whence) {
    if (observing) {
        observed_file = file;
        if (whence == SEEK_END) {
            ++end_seeks;
            if (failure == Failure::EndSeek) { errno = ESPIPE; return -1; }
        } else if (whence == SEEK_SET) {
            ++start_seeks;
            if (failure == Failure::StartSeek) { errno = EIO; return -1; }
        }
    }
    return __real_fseek(file, offset, whence);
}
long __wrap_ftell(FILE* file) {
    if (observing && file == observed_file) {
        ++tells;
        if (failure == Failure::Tell) { errno = EIO; return -1; }
    }
    return __real_ftell(file);
}
size_t __wrap_fread(void* data, size_t size, size_t count, FILE* file) {
    if (observing && file == observed_file) ++reads;
    return __real_fread(data, size, count, file);
}
int __wrap_fclose(FILE* file) {
    if (observing && file == observed_file) { ++closes; observed_file = nullptr; }
    return __real_fclose(file);
}
}

namespace {
void finish() {
    observing = false;
    // The original negative-ftell implementation throws before closing. Clean up that red
    // control after recording the missing close, so even the failing test remains disposable.
    if (observed_file) { __real_fclose(observed_file); observed_file = nullptr; }
}
void check_failure_operations(Failure value) {
    CHECK(end_seeks == 1, "the end seek was exercised exactly once");
    CHECK(tells == (value == Failure::EndSeek ? 0 : 1), "stop sizing after the first failed operation");
    CHECK(start_seeks == (value == Failure::StartSeek ? 1 : 0), "do not seek again after sizing failure");
    CHECK(reads == 0, "a failed sizing or seek operation never reaches fread");
    CHECK(closes == 1, "close the module stream exactly once on failure");
}
} // namespace

int main() {
    const std::string path = prosper_test::test_scratch_file("module-io.prx");
    prosper_test::SynthModuleSpec spec;
    spec.exports = { nid_hash("prosperIoControl") };
    std::string err;
    CHECK(prosper_test::write_synth_prx(path, spec, &err), "write a valid synthetic module");
    if (fails) return 1;

    begin(Failure::None);
    const auto control = Module::load(path, &err);
    CHECK(control && control->file.size() == prosper_test::kSynthFileSize,
          "positive control: the same regular module parses with no injected failure");
    CHECK(end_seeks == 1 && tells == 1 && start_seeks == 1 && reads == 1 && closes == 1,
          "positive control: all operation interceptors observe the real module load");
    finish();
    Program linked;
    CHECK(link_program({{path, 0x400000000ull}}, 0x700000000ull, 0x7c0000000ull, linked, &err) &&
          linked.mods.size() == 1 && !linked.imgs[0].mem.empty(),
          "positive control: the module links into a nonempty image");

    for (const auto value : {Failure::EndSeek, Failure::Tell, Failure::StartSeek}) {
        const char* expected = value == Failure::EndSeek ? "cannot seek to end of file" :
                               value == Failure::Tell ? "cannot determine file size" :
                                                        "cannot seek to start of file";
        err.clear();
        begin(value);
        bool refused = false, threw = false;
        try { refused = !Module::load(path, &err); }
        catch (const std::exception&) { threw = true; }
        CHECK(refused && !threw && err == expected, "the failed operation returns its loader error without throwing");
        check_failure_operations(value);
        finish();

        begin(value);
        err.clear(); Program out;
        refused = false; threw = false;
        try { refused = !link_program({{path, 0x400000000ull}}, 0x700000000ull, 0x7c0000000ull, out, &err); }
        catch (const std::exception&) { threw = true; }
        CHECK(refused && !threw && err == "load " + path + ": " + expected,
              "link_program propagates the failed operation and module path to its caller");
        CHECK(out.mods.empty() && out.imgs.empty(), "a failed module load does not publish an image");
        check_failure_operations(value);
        finish();
    }
    std::printf("failures: %d\n", fails);
    return fails ? 1 : 0;
}
