#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
namespace prosper::screenshot {
uint32_t crc32_bytes(const void* data, size_t len, uint32_t seed = 0);
bool write_png(const char* path, const uint8_t* rgba, uint32_t width, uint32_t height,
               std::string& error);
}   // namespace prosper::screenshot
