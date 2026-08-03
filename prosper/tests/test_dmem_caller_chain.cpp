#include "../src/hle/dmem_caller_chain.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <string>
#include <thread>

using prosper::DmemCallerChainInterner;
using prosper::DmemCallerChainFrame;
using prosper::DmemCallerChainResult;
using prosper::DmemCallerChainState;
using prosper::dmem_caller_chain_correlates_allocation;
using prosper::format_dmem_caller_chain_definition;
using prosper::write_dmem_caller_chain_definition;

static int failures = 0;

static void check(bool ok, const char* name) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) ++failures;
}

int main() {
    DmemCallerChainInterner<2> chains;

    const DmemCallerChainResult unknown = chains.intern(0, 0);
    const DmemCallerChainResult unknown_again = chains.intern(0, 0);
    check(unknown.state == DmemCallerChainState::Unknown && unknown.id == 0 && unknown.first &&
              unknown_again.state == DmemCallerChainState::Unknown && !unknown_again.first,
          "a missing guest stack is explicitly unknown and announced once");

    const DmemCallerChainResult first = chains.intern(0x41000100, 0x41000200);
    const DmemCallerChainResult repeated = chains.intern(0x41000100, 0x41000200);
    check(first.state == DmemCallerChainState::Known && first.id == 1 && first.first &&
              repeated.state == DmemCallerChainState::Known && repeated.id == first.id &&
              !repeated.first,
          "a repeated allocation keeps the same nonzero caller-chain ID");
    check(dmem_caller_chain_correlates_allocation(first) &&
              dmem_caller_chain_correlates_allocation(repeated),
          "every enabled allocation carries correlation, including repeated chains");

    const DmemCallerChainResult second = chains.intern(0x41000100, 0x41000300);
    check(second.state == DmemCallerChainState::Known && second.id == 2 && second.first &&
              second.id != first.id,
          "a distinct first-two-frame key receives a distinct stable ID");

    const DmemCallerChainResult overflow = chains.intern(0x41000400, 0x41000500);
    const DmemCallerChainResult overflow_again = chains.intern(0x41000600, 0x41000700);
    check(overflow.state == DmemCallerChainState::Overflow && overflow.id == 0 && overflow.first &&
              overflow_again.state == DmemCallerChainState::Overflow && !overflow_again.first,
          "capacity overflow is explicit and its ceiling is announced once");

    const DmemCallerChainResult retained = chains.intern(0x41000100, 0x41000200);
    check(retained.state == DmemCallerChainState::Known && retained.id == first.id &&
              chains.size() == 2,
          "known correlations survive after the bounded registry fills");

    DmemCallerChainInterner<> concurrent;
    std::array<DmemCallerChainResult, 8> concurrent_results{};
    std::array<std::thread, 8> workers;
    for (size_t i = 0; i < workers.size(); ++i)
        workers[i] = std::thread([&, i] {
            concurrent_results[i] = concurrent.intern(0x44001000, 0x44002000);
        });
    for (auto& worker : workers) worker.join();
    size_t first_count = 0;
    bool all_same = true;
    for (const auto& result : concurrent_results) {
        first_count += result.first ? 1 : 0;
        all_same = all_same && result.state == DmemCallerChainState::Known && result.id == 1;
    }
    check(all_same && first_count == 1 && concurrent.size() == 1,
          "concurrent repeats publish one ID and one full-chain owner");

    const DmemCallerChainFrame format_frames[] = {
        {"eboot", 0x1b2454f}, {"libSceFoo", 0x7d0a00},
    };
    check(format_dmem_caller_chain_definition(
              17, 0x1000000, format_frames, std::size(format_frames), 23, 160) ==
              "[dmem-caller] caller-chain=17 alloc_main_dmem len=0x1000000 from"
              " eboot+0x1b2454f libSceFoo+0x7d0a00"
              " [scan clamped to 23/160 slots by stack top]\n",
          "a full-chain definition has one exact parseable line format");

    // Reproduce the reviewed failure shape: several guest threads discover DIFFERENT chains at
    // once. The old implementation emitted each definition through a sequence of fprintf calls,
    // so an ID prefix from one thread could be joined to frames from another. Exercise the exact
    // production formatter/writer and require one intact independently identifiable line per ID.
    constexpr size_t kDistinctChains = 64;
    FILE* definitions = std::tmpfile();
    check(definitions != nullptr, "a temporary stream is available for definition serialization");
    if (definitions) {
        std::array<std::array<std::string, 6>, kDistinctChains> module_names;
        std::array<std::array<DmemCallerChainFrame, 6>, kDistinctChains> frame_sets{};
        std::array<std::string, kDistinctChains> expected_lines;
        for (size_t chain = 0; chain < kDistinctChains; ++chain) {
            for (size_t frame = 0; frame < frame_sets[chain].size(); ++frame) {
                module_names[chain][frame] = "chain" + std::to_string(chain + 1) +
                    "_frame" + std::to_string(frame + 1) + "_" +
                    std::string(96, static_cast<char>('a' + (chain % 26)));
                frame_sets[chain][frame] = {
                    module_names[chain][frame].c_str(),
                    0x100000ull * (chain + 1) + 0x100ull * (frame + 1),
                };
            }
            expected_lines[chain] = format_dmem_caller_chain_definition(
                static_cast<uint32_t>(chain + 1), 0x1000000 + chain,
                frame_sets[chain].data(), frame_sets[chain].size(), 160, 160);
        }

        std::atomic<size_t> ready{0};
        std::atomic<bool> start{false};
        std::array<std::thread, kDistinctChains> definition_workers;
        for (size_t chain = 0; chain < kDistinctChains; ++chain) {
            definition_workers[chain] = std::thread([&, chain] {
                ready.fetch_add(1, std::memory_order_release);
                while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
                write_dmem_caller_chain_definition(
                    definitions, static_cast<uint32_t>(chain + 1), 0x1000000 + chain,
                    frame_sets[chain].data(), frame_sets[chain].size(), 160, 160);
            });
        }
        while (ready.load(std::memory_order_acquire) != kDistinctChains)
            std::this_thread::yield();
        start.store(true, std::memory_order_release);
        for (auto& worker : definition_workers) worker.join();

        std::fflush(definitions);
        std::fseek(definitions, 0, SEEK_END);
        const long output_size = std::ftell(definitions);
        std::rewind(definitions);
        std::string output(output_size > 0 ? static_cast<size_t>(output_size) : 0, '\0');
        const size_t bytes_read = output.empty()
            ? 0 : std::fread(output.data(), 1, output.size(), definitions);
        bool intact = output_size >= 0 && bytes_read == output.size() &&
            std::count(output.begin(), output.end(), '\n') == kDistinctChains;
        for (const std::string& expected : expected_lines) {
            const size_t first_match = output.find(expected);
            intact = intact && first_match != std::string::npos &&
                output.find(expected, first_match + 1) == std::string::npos;
        }
        check(intact,
              "concurrent distinct definitions are each one complete uncorrupted line");
        std::fclose(definitions);
    }

    std::printf("%s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
