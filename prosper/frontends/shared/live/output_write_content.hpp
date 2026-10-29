// A synchronous borrowed view, visible only while the existing write notification runs.
#pragma once
#include <cstdint>
#include <span>

namespace prosper::gpu {
struct ShaderResource;
}
namespace prosper::frontend {
class OutputWriteContent {
public:
    OutputWriteContent(uint64_t advertised, const void* destination, uint64_t bytes,
                       uint64_t readable_host_bytes);
    ~OutputWriteContent();
    OutputWriteContent(const OutputWriteContent&) = delete;
    OutputWriteContent& operator=(const OutputWriteContent&) = delete;
    static std::span<const uint8_t> for_write(uint64_t address);

private:
    const OutputWriteContent* previous_;
    uint64_t advertised_, effective_;
    std::span<const uint8_t> bytes_;
};
uint64_t hosted_output_bytes(const prosper::gpu::ShaderResource*, const void* destination);
} // namespace prosper::frontend
