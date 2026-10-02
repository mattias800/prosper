#include "bpermute_spirv_oracle.hpp"
#include "fixtures/portable_bpermute_fixture.hpp"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {
int failures = 0, checks = 0;
void check(bool ok, const std::string &name) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "[FAIL] %s\n", name.c_str());
  }
}
uint32_t count(const std::vector<uint32_t> &m, uint32_t op) {
  uint32_t n = 0;
  for (size_t pc = 5; pc < m.size();) {
    uint32_t size = m[pc] >> 16;
    if (!size || size > m.size() - pc) {
      check(false, "module framing");
      break;
    }
    n += (m[pc] & 65535) == op;
    pc += size;
  }
  return n;
}
void dump(const std::filesystem::path &root, const std::string &name,
          const std::vector<uint32_t> &m) {
  if (root.empty() || m.empty())
    return;
  std::ofstream f(root / (name + ".spv"), std::ios::binary);
  f.write(reinterpret_cast<const char *>(m.data()),
          static_cast<std::streamsize>(m.size() * 4));
  f.close();
  check(bool(f), name + " dump");
}
} // namespace
int main(int argc, char **argv) {
  using namespace prosper::test::bpermute;
  std::filesystem::path root;
  if (argc == 3 && std::strcmp(argv[1], "--dump-directory") == 0)
    root = argv[2];
  else if (argc != 1) {
    std::fprintf(stderr,
                 "usage: test_portable_bpermute [--dump-directory DIR]\n");
    return 2;
  }
  uint32_t ordinal = 0;
  for (const auto &c : cases()) {
    const std::string name = "bpermute_" + std::to_string(ordinal++);
    const auto raw = guest(c);
    std::vector<prosper::gpu::Rdna2Inst> decoded;
    prosper::gpu::rdna2_walk(raw.data(), raw.size(), decoded);
    uint32_t events = 0;
    for (const auto &i : decoded)
      if (i.fmt == prosper::gpu::Rdna2Format::DS) {
        ++events;
        check(!i.ds_gds && i.len_dwords == 2 &&
                  i.src[0].kind == prosper::gpu::OperandKind::VGPR &&
                  i.src[0].value == 2 &&
                  i.src[1].kind == prosper::gpu::OperandKind::VGPR &&
                  (i.src[1].value == 1 || i.src[1].value == 4 ||
                   (c.different_sites && i.src[1].value == 7)),
              name + " actual DS operand decode");
        check(i.literal == c.offset || ((c.second || c.different_sites) &&
                                        i.literal == c.offset + 4),
              name + " byte OFFSET0 decode");
      }
    check(events == ((c.second || c.different_sites) ? 2u : 1u),
          name + " static gather events");
    const auto module = compile(c);
    check(!module.empty(), name + " actual production compute emitted");
    if (module.empty())
      continue;
    dump(root, name, module);
    check(count(module, 251) > 0 && count(module, 224) >= 2,
          name + " common CFG and rendezvous emitted");
    for (uint32_t op = 333; op <= 366; ++op)
      check(count(module, op) == 0,
            name + " no host subgroup operation " + std::to_string(op));
    bpermute_oracle::Interpreter vm(module);
    const auto got = vm.run(c.local, input(c));
    check(vm.error.empty(),
          name + " typed actual SOURCE execution: " + vm.error);
    check(got == expected(c),
          name + " all storage words match independent guest gather");
    if (!got.empty()) {
      auto wrong = got;
      wrong[c.local * 3 + c.threads - 1] ^= 1;
      check(wrong != expected(c), name + " last active sink is load-bearing");
    }
  }
  Case control;
  const auto clean = compile(control);
  check(!clean.empty(), "GDS refusal paired clean production control");
  auto invalid = guest(control);
  for (size_t pc = 0; pc + 1 < invalid.size(); ++pc)
    if ((invalid[pc] & 0xffff0000u) == 0xdacc0000u) {
      invalid[pc] |= 0x10000u;
      break;
    }
  const auto rt = resources(control);
  prosper::gpu::ComputeShaderConfig cfg;
  prosper::gpu::TerminalRejectCapture rejects;
  check(prosper::gpu::recompile_compute(
            invalid.data(), invalid.size(), &rt, cfg,
            {prosper::gpu::RecompileDiagnosticStage::Compute, 0x4089u})
            .empty(),
        "GDS is not an LDS lane gather");
  const auto reasons = rejects.take();
  check(std::any_of(reasons.begin(), reasons.end(),
                    [](const auto &r) {
                      return r.second.find(
                                 "ds-bpermute-native-wave-contract") !=
                             std::string::npos;
                    }),
        "GDS refusal comes from actual BPERMUTE contract guard");
  control.wave = 32;
  const auto native = compile(control, 32);
  check(!native.empty() && count(native, 345) == 2,
        "exact native straight-line path still shuffles value/EXEC");
  dump(root, "bpermute_native32", native);
  for (uint32_t wave : {32u, 64u}) {
    for (uint32_t offset : {0u, 4u}) {
      Case partial{64, wave, 16, offset, 1};
      const auto m = compile(partial, wave);
      check(!m.empty() && count(m, 345) == 0,
            "adopted native width switches to ACTIVE for an exact tail");
      if (m.empty())
        continue;
      dump(root,
           "bpermute_native_requested_tail" + std::to_string(wave) + "_offset" +
               std::to_string(offset),
           m);
      bpermute_oracle::Interpreter vm(m);
      check(vm.run(partial.local, input(partial)) == expected(partial) &&
                vm.error.empty(),
            offset == 0 ? "adopted native tail actually supplies source-off "
                          "zero and keeps padded sinks"
                        : "adopted native tail actually gathers active source "
                          "after OFFSET wrap and keeps padded sinks");
    }
  }

  // The interpreter itself must fail closed: remove all actual Workgroup
  // publication Stores while leaving their reads, and invalidate a barrier
  // scope.
  if (!clean.empty()) {
    auto no_store = clean, bad_barrier = clean;
    bpermute_oracle::Interpreter parsed(clean);
    uint32_t wg = 0;
    for (const auto &g : parsed.globals)
      if (g.a.size() >= 3 && g.a[2] == 4) {
        wg = g.a[1];
        break;
      }
    std::vector<uint32_t> pointers;
    bool changed_store = false, changed_barrier = false;
    for (size_t pc = 5; pc < clean.size();) {
      const auto n = clean[pc] >> 16, op = clean[pc] & 65535;
      if (op == 65 && n >= 5 && clean[pc + 3] == wg)
        pointers.push_back(clean[pc + 2]);
      if (op == 62 && n >= 3 &&
          std::find(pointers.begin(), pointers.end(), clean[pc + 1]) !=
              pointers.end()) {
        no_store[pc] = (n << 16);
        changed_store = true;
      }
      if (op == 224 && n == 4 && !changed_barrier) {
        bad_barrier[pc + 1] = clean[pc + 3];
        changed_barrier = true;
      }
      pc += n;
    }
    check(changed_store && changed_barrier,
          "oracle controls alter actual publication/barrier");
    bpermute_oracle::Interpreter poison(no_store), scope(bad_barrier);
    const auto poisoned = poison.run(64, input(control));
    const auto scoped = scope.run(64, input(control));
    check(!poison.error.empty() && poisoned.empty(),
          "oracle refuses wholly unpublished Workgroup poison");
    check(!scope.error.empty() && scoped.empty(),
          "oracle refuses wrong barrier scope");
  }
  std::printf("portable_bpermute_contract: %d checks, %d failures, %u portable "
              "production modules\n",
              checks, failures, ordinal);
  return failures ? 1 : 0;
}
