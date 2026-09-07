#pragma once
#include <vulkan/vulkan.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace prosper::frontend {

// Store the original driver bytes inside a checked envelope. Only those original bytes may be
// supplied to VkPipelineCacheCreateInfo; a compatible header alone cannot detect a truncated body.
struct PipelineCacheFile {
    static constexpr size_t max_bytes = 256u * 1024u * 1024u;
    std::filesystem::path path;
    VkPhysicalDeviceProperties device{};

    static uint64_t checksum(std::span<const uint8_t> bytes) {
        uint64_t h = 14695981039346656037ull;
        for (uint8_t b : bytes) { h ^= b; h *= 1099511628211ull; }
        return h;
    }
    static uint64_t read_le(const uint8_t* p, size_t n) {
        uint64_t v = 0;
        for (size_t i = 0; i < n; ++i) v |= uint64_t(p[i]) << (8 * i);
        return v;
    }
    static void write_le(uint8_t* p, uint64_t v, size_t n) {
        for (size_t i = 0; i < n; ++i) p[i] = uint8_t(v >> (8 * i));
    }
    bool compatible(std::span<const uint8_t> blob) const {
        return blob.size() >= 32 && blob.size() <= max_bytes &&
            read_le(blob.data(), 4) >= 32 && read_le(blob.data(), 4) <= blob.size() &&
            read_le(blob.data() + 4, 4) == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
            read_le(blob.data() + 8, 4) == device.vendorID &&
            read_le(blob.data() + 12, 4) == device.deviceID &&
            std::memcmp(blob.data() + 16, device.pipelineCacheUUID, VK_UUID_SIZE) == 0;
    }
    static PipelineCacheFile graphics(const VkPhysicalDeviceProperties& properties) {
        PipelineCacheFile file;
        file.device = properties;
        if (std::getenv("PROSPER_NO_DISK_PIPELINE_CACHE")) return file;
        // Explicit paths also count as opt-in, matching the existing compute policy. This is
        // deliberately not default-on: identity checks do not cure the NVIDIA blob-load crash.
        if (const char* p = std::getenv("PROSPER_GRAPHICS_PIPELINE_CACHE_PATH")) {
            file.path = p;
            return file;
        }
        if (!std::getenv("PROSPER_DISK_PIPELINE_CACHE")) return file;
#ifdef _WIN32
        const char* base = std::getenv("LOCALAPPDATA");
        if (!base || !*base) return file;
        std::filesystem::path directory(base);
#else
        const char* base = std::getenv("XDG_CACHE_HOME");
        std::filesystem::path directory;
        if (base && *base) directory = base;
        else {
            base = std::getenv("HOME");
            if (!base || !*base) return file;
            directory = std::filesystem::path(base) / ".cache";
        }
#endif
        char name[160];
        std::snprintf(name, sizeof(name), "graphics-vkcache-v1-%08x-%08x-%08x-",
                      properties.vendorID, properties.deviceID, properties.driverVersion);
        std::string identity(name);
        for (uint8_t b : properties.pipelineCacheUUID) {
            char hex[3]; std::snprintf(hex, sizeof(hex), "%02x", b); identity += hex;
        }
        file.path = directory / "prosper" / identity;
        return file;
    }
    std::vector<uint8_t> load() const {
        if (path.empty()) return {};
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) return {};
        const auto bytes = in.tellg();
        if (bytes < 64 || bytes > std::streamoff(max_bytes + 32)) return {};
        in.seekg(0);
        std::array<uint8_t, 32> header{};
        in.read(reinterpret_cast<char*>(header.data()), header.size());
        if (!in || std::memcmp(header.data(), "PRVKC01\0", 8) ||
            read_le(header.data() + 8, 4) != device.driverVersion ||
            read_le(header.data() + 12, 4) != sizeof(void*) ||
            read_le(header.data() + 16, 8) != uint64_t(bytes) - header.size()) return {};
        std::vector<uint8_t> blob(size_t(bytes) - header.size());
        in.read(reinterpret_cast<char*>(blob.data()), blob.size());
        if (!in || !compatible(blob) ||
            checksum(blob) != read_le(header.data() + 24, 8)) return {};
        return blob;
    }
    bool save(std::span<const uint8_t> blob) const {
        if (path.empty() || !compatible(blob)) return false;
        std::error_code ec;
        auto parent = path.parent_path();
        if (parent.empty()) parent = ".";
        std::filesystem::create_directories(parent, ec);
        if (ec) return false;
        // Atomically reserve a private sibling directory, including across concurrent processes.
        // No writer truncates another writer's temporary or deletes the previous good cache.
        static std::atomic<uint64_t> serial{0};
        std::filesystem::path temporary;
        for (unsigned attempt = 0; attempt < 16; ++attempt) {
            temporary = path;
            temporary += ".tmp-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                std::to_string(serial.fetch_add(1));
            if (std::filesystem::create_directory(temporary, ec)) break;
            if (ec || attempt == 15) return false;
        }
        struct Cleanup {
            std::filesystem::path directory;
            ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(directory, ignored); }
        } cleanup{temporary};
        auto payload = temporary / "cache";
        std::array<uint8_t, 32> header{};
        std::memcpy(header.data(), "PRVKC01\0", 8);
        write_le(header.data() + 8, device.driverVersion, 4);
        write_le(header.data() + 12, sizeof(void*), 4);
        write_le(header.data() + 16, blob.size(), 8);
        write_le(header.data() + 24, checksum(blob), 8);
        std::ofstream out(payload, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(header.data()), header.size());
        out.write(reinterpret_cast<const char*>(blob.data()), blob.size());
        out.close();
        if (!out) return false;
#ifdef _WIN32
        return MoveFileExW(payload.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
        std::filesystem::rename(payload, path, ec);
        return !ec;
#endif
    }
};
} // namespace prosper::frontend
