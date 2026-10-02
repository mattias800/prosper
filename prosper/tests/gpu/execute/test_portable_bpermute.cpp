#include "fixtures/compute_runner.h"
#include "fixtures/portable_bpermute_fixture.hpp"
#include <cstdio>
#include <vector>

int main() {
  using namespace prosper::test::bpermute;
  uint32_t failures = 0, ordinal = 0;
  for (const auto &c : cases()) {
    const auto module = compile(c);
    if (module.empty()) {
      std::fprintf(stderr,
                   "[FAIL] case %u: actual compute translation refused\n",
                   ordinal++);
      ++failures;
      continue;
    }
    std::vector<uint32_t> output;
    prosper::test::run_compute(module, std::vector<float>(c.local), c.local,
                               c.local, {}, input(c), &output, c.local);
    const auto want = expected(c);
    if (output != want) {
      ++failures;
      uint32_t first = 0;
      while (first < output.size() && first < want.size() &&
             output[first] == want[first])
        ++first;
      std::fprintf(stderr,
                   "[FAIL] case %u local=%u wave=%u extent=%u first=%u "
                   "size=%zu expected=%zu\n",
                   ordinal, c.local, c.wave, c.threads, first, output.size(),
                   want.size());
      if (first < output.size() && first < want.size())
        std::fprintf(stderr, "  actual=%08x expected=%08x\n", output[first],
                     want[first]);
    }
    ++ordinal;
  }
  std::printf("portable_bpermute_execution: %u actual production storage "
              "sinks, %u failures\n",
              ordinal, failures);
  return failures ? 1 : 0;
}
