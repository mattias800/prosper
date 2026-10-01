// #2687: fail each sizing operation on an otherwise valid module. GNU ld wrapping keeps the
// failure bounded and independent of host filesystem behavior; no enormous allocation is needed.
#include "fixtures/synth_prx.h"
#include "fixtures/test_scratch.h"
#include "hle/dispatch/nid.hpp"
#include "loader/linker.hpp"
#include <cerrno>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

using namespace prosper;
namespace {
enum class Failure { None, EndSeek, Tell, StartSeek };
struct IoState {
    Failure failure = Failure::None;
    bool observing = false;
    FILE* observed_file = nullptr;
    int end_seeks = 0;
    int tells = 0;
    int start_seeks = 0;
    int reads = 0;
    int closes = 0;
    int fails = 0;
};
IoState& io() { static IoState state; return state; }
#define CHECK(c, m) do { if (!(c)) { std::printf("[FAIL] %s\n", m); ++io().fails; } \
                        else std::printf("[ok] %s\n", m); } while (0)
void begin(Failure value) {
    auto& s = io();
    s.failure = value;
    s.observing = true;
    s.observed_file = nullptr;
    s.end_seeks = 0;
    s.tells = 0;
    s.start_seeks = 0;
    s.reads = 0;
    s.closes = 0;
}
} // namespace

extern "C" {
int __real_fseek(FILE*, long, int);
long __real_ftell(FILE*);
size_t __real_fread(void*, size_t, size_t, FILE*);
int __real_fclose(FILE*);
int __wrap_fseek(FILE* file, long offset, int whence) {
    if (io().observing) {
        io().observed_file = file;
        if (whence == SEEK_END) {
            ++io().end_seeks;
            if (io().failure == Failure::EndSeek) { errno = ESPIPE; return -1; }
        } else if (whence == SEEK_SET) {
            ++io().start_seeks;
            if (io().failure == Failure::StartSeek) { errno = EIO; return -1; }
        }
    }
    return __real_fseek(file, offset, whence);
}
long __wrap_ftell(FILE* file) {
    if (io().observing && file == io().observed_file) {
        ++io().tells;
        if (io().failure == Failure::Tell) { errno = EIO; return -1; }
    }
    return __real_ftell(file);
}
size_t __wrap_fread(void* data, size_t size, size_t count, FILE* file) {
    if (io().observing && file == io().observed_file) ++io().reads;
    return __real_fread(data, size, count, file);
}
int __wrap_fclose(FILE* file) {
    if (io().observing && file == io().observed_file) { ++io().closes; io().observed_file = nullptr; }
    return __real_fclose(file);
}
}

namespace {
void finish() {
    io().observing = false;
    // The original negative-ftell implementation throws before closing. Clean up that red
    // control after recording the missing close, so even the failing test remains disposable.
    if (io().observed_file) { __real_fclose(io().observed_file); io().observed_file = nullptr; }
}
void check_failure_operations(Failure value) {
    CHECK(io().end_seeks == 1, "the end seek was exercised exactly once");
    CHECK(io().tells == (value == Failure::EndSeek ? 0 : 1), "stop sizing after the first failed operation");
    CHECK(io().start_seeks == (value == Failure::StartSeek ? 1 : 0), "do not seek again after sizing failure");
    CHECK(io().reads == 0, "a failed sizing or seek operation never reaches fread");
    CHECK(io().closes == 1, "close the module stream exactly once on failure");
}
} // namespace

int main() {
    const std::string path = prosper_test::test_scratch_file("module-io.prx");
    prosper_test::SynthModuleSpec spec;
    spec.exports = { nid_hash("prosperIoControl") };
    std::string err;
    CHECK(prosper_test::write_synth_prx(path, spec, &err), "write a valid synthetic module");
    if (io().fails) return 1;

    begin(Failure::None);
    const auto control = Module::load(path, &err);
    CHECK(control && control->file.size() == prosper_test::kSynthFileSize,
          "positive control: the same regular module parses with no injected failure");
    CHECK(io().end_seeks == 1 && io().tells == 1 && io().start_seeks == 1 && io().reads == 1 && io().closes == 1,
          "positive control: all operation interceptors observe the real module load");
    finish();
    Program linked;
    CHECK(link_program({{path, 0x400000000ULL}}, 0x700000000ULL, 0x7c0000000ULL, linked, &err) &&
          linked.mods.size() == 1 && !linked.imgs[0].mem.empty(),
          "positive control: the module links into a nonempty image");

    struct TestCase { Failure failure; const char* error; };
    const TestCase cases[] = {
        {Failure::EndSeek, "cannot seek to end of file"},
        {Failure::Tell, "cannot determine file size"},
        {Failure::StartSeek, "cannot seek to start of file"},
    };
    for (const auto& test : cases) {
        err.clear();
        begin(test.failure);
        try {
            CHECK(!Module::load(path, &err) && err == test.error,
                  "the failed operation returns its loader error without throwing");
        } catch (const std::exception&) {
            CHECK(false, "Module::load threw instead of returning a loader error");
        }
        check_failure_operations(test.failure);
        finish();

        begin(test.failure);
        err.clear();
        Program out;
        try {
            CHECK(!link_program({{path, 0x400000000ULL}}, 0x700000000ULL, 0x7c0000000ULL, out, &err) &&
                  err == "load " + path + ": " + test.error,
                  "link_program propagates the failed operation and module path to its caller");
        } catch (const std::exception&) {
            CHECK(false, "link_program threw instead of propagating a loader error");
        }
        CHECK(out.mods.empty() && out.imgs.empty(), "a failed module load does not publish an image");
        check_failure_operations(test.failure);
        finish();
    }
    std::cout << "failures: " << io().fails << '\n';
    return io().fails ? 1 : 0;
}
