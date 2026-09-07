#include "shared/device/pipeline_cache_file.hpp"
#include <thread>
#include <cassert>

// Keep assertions active in Release, like the surrounding standalone policy checks.
#undef assert
#define assert(condition) do { if (!(condition)) { std::fprintf(stderr, "failed: %s\n", #condition); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    using prosper::frontend::PipelineCacheFile;
    PipelineCacheFile file;
    file.path = std::filesystem::path(argv[1]) / "cache";
    file.device.vendorID = 123;
    file.device.deviceID = 456;
    file.device.driverVersion = 789;
    file.device.pipelineCacheUUID[0] = 42;
    std::vector<uint8_t> blob(64, 0);
    PipelineCacheFile::write_le(blob.data(), 32, 4);
    PipelineCacheFile::write_le(blob.data() + 4, 1, 4);
    PipelineCacheFile::write_le(blob.data() + 8, 123, 4);
    PipelineCacheFile::write_le(blob.data() + 12, 456, 4);
    blob[16] = 42;
    assert(file.save(blob));
    assert(file.load() == blob);
    auto other = file;
    other.device.driverVersion++;
    assert(other.load().empty());
    other = file; other.device.deviceID++;
    assert(other.load().empty());
    other = file; other.device.pipelineCacheUUID[0]++;
    assert(other.load().empty());
    auto second = blob; second.back() = 17;
    bool a = false, b = false;
    std::thread first([&] { a = file.save(blob); });
    std::thread last([&] { b = file.save(second); });
    first.join(); last.join();
    assert(a && b);
    auto loaded = file.load();
    assert(loaded == blob || loaded == second);
    // A failed replacement must leave the destination intact, including non-file destinations.
    auto obstructed = file; obstructed.path += ".directory";
    std::filesystem::create_directories(obstructed.path);
    std::ofstream(obstructed.path / "marker") << "keep";
    assert(!obstructed.save(blob));
    assert(std::filesystem::exists(obstructed.path / "marker"));
    for (const auto& entry : std::filesystem::directory_iterator(file.path.parent_path()))
        assert(entry.path().filename().string().find(".tmp-") == std::string::npos);
    return 0;
}
