#pragma once
#include "gpu/diagnostics/wait_producer_trace.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace wait_producer_test {
using namespace prosper;
using namespace prosper::gpu;

inline uint64_t address(const void* p) {
    return reinterpret_cast<uintptr_t>(p);
}
inline std::string hex(uint64_t value) {
    char out[32];
    std::snprintf(out, sizeof(out), "0x%llx", static_cast<unsigned long long>(value));
    return out;
}
inline uint32_t header(uint32_t words, uint32_t sub) {
    return 0xc0000000u | ((words - 2) << 16) | (IT_NOP << 8) | (sub << 2);
}
inline std::vector<uint32_t> write(uint64_t target, uint64_t value) {
    return {header(7, R_WRITE_DATA), 0, uint32_t(target),
            uint32_t(target >> 32),  2, uint32_t(value),
            uint32_t(value >> 32)};
}
inline std::vector<uint32_t> wait(uint64_t target, uint64_t reference) {
    return {header(9, R_WAIT_MEM_64),
            uint32_t(target),
            uint32_t(target >> 32),
            UINT32_MAX,
            UINT32_MAX,
            uint32_t(reference),
            uint32_t(reference >> 32),
            3,
            400};
}
inline std::vector<uint32_t> release(uint64_t target, uint64_t value) {
    return {header(6, R_RELEASE_MEM), uint32_t(target),     uint32_t(target >> 32), 2,
            uint32_t(value),          uint32_t(value >> 32)};
}
using Record = std::unordered_map<std::string, std::string>;
inline std::vector<Record> records(const std::string& output, const char* phase) {
    std::vector<Record> result;
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        if (!line.starts_with("[wait-producer] ")) continue;
        Record fields;
        std::istringstream words(line);
        std::string word;
        while (words >> word) {
            const auto equal = word.find('=');
            if (equal != std::string::npos)
                fields.emplace(word.substr(0, equal), word.substr(equal + 1));
        }
        if (fields["phase"] == phase) result.push_back(std::move(fields));
    }
    return result;
}
inline Record at_address(const std::vector<Record>& rows, uint64_t target) {
    for (const auto& row : rows)
        if (row.at("address") == hex(target)) return row;
    return {};
}

// Hold the real import-return visibility boundary so a queued write remains available to overlay.
// Closing the scope and draining makes every pending application complete before labels go away.
class Fixture : public testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        prosper_gpu_enable_post_submit_visibility();
        prosper_gpu_drain_completion_writes();
        testing::internal::CaptureStderr();
        capture = true;
        prosper_gpu_submit_scope_begin();
        outer_scope = true;
    }
    void TearDown() override {
        close_scope();
        if (capture) testing::internal::GetCapturedStderr();
    }
    void close_scope() {
        if (!outer_scope) return;
        prosper_gpu_submit_scope_end();
        outer_scope = false;
        prosper_gpu_drain_completion_writes();
    }
    std::string finish() {
        close_scope();
        capture = false;
        return testing::internal::GetCapturedStderr();
    }
    void acb(uint64_t queue, const std::vector<uint32_t>& stream) {
        const char* nid = "gSRnr79F8tQ";
        const auto fn = Hle::lookup(nid);
        const auto hook = Hle::return_hook_of(nid);
        ASSERT_NE(fn, nullptr);
        ASSERT_NE(hook, nullptr);
        struct Packet {
            uint64_t stream;
            uint32_t dwords, flags;
        } packet{address(stream.data()), uint32_t(stream.size()), 0};
        EXPECT_EQ(fn(queue, address(&packet), packet.dwords, 0, 0, 0), 0u);
        hook();
    }
    void dcb(const std::vector<uint32_t>& stream) {
        const char* nid = "UglJIZjGssM";
        const auto fn = Hle::lookup(nid);
        const auto hook = Hle::return_hook_of(nid);
        ASSERT_NE(fn, nullptr);
        ASSERT_NE(hook, nullptr);
        struct Packet {
            uint64_t stream;
            uint32_t dwords, flags;
        } packet{address(stream.data()), uint32_t(stream.size()), 0};
        EXPECT_EQ(fn(address(&packet), 0, 0, 0, 0, 0), 0u);
        hook();
    }
    void fold(const std::vector<uint32_t>& stream) {
        EXPECT_EQ(run_command_buffer(stream.data(), stream.size(), state), 1u);
    }
    bool outer_scope = false, capture = false;
    GpuState state;
    alignas(8) uint64_t label = 0, other = 0;
};
} // namespace wait_producer_test
