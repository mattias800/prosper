#pragma once

#include <cstdint>
#include <set>

namespace prosper::gpu {
struct RegState;
struct Rdna2Inst;
struct ScalarSourceMarks;

struct CurrentWordRoots {
    uint32_t exec = 0;
    std::set<uint64_t> roots;
};
struct CurrentWordRootPublication {
    CurrentWordRoots view;
    bool atomic_result = false;
};
using WordRootReader = const std::set<uint64_t>& (*)(const RegState&, const ScalarSourceMarks&,
                                                     uint32_t);

// Called after the value emitter succeeds, before the ordinary mark transfer replaces its inputs.
// This refines roots only; it authenticates neither a missing word nor a loop route.
CurrentWordRootPublication snapshot_current_word_roots(const RegState& rs, const Rdna2Inst& in,
                                                       const ScalarSourceMarks& source,
                                                       WordRootReader read_roots);
void retain_current_word_roots(ScalarSourceMarks& result, const RegState& rs,
                               const CurrentWordRootPublication& publication,
                               WordRootReader read_roots);
const std::set<uint64_t>* current_word_roots(const RegState& rs, const ScalarSourceMarks& marks,
                                             uint32_t reader);
}  // namespace prosper::gpu
