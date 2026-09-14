#include "gpu/capture/fold_capture.hpp"
#include "gpu/resources/fold_control_plan.hpp"
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>

using namespace prosper::gpu;
static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { std::cerr << "FAIL: " << m << '\n'; ++failures; } } while (0)
static void rejects(const std::function<void()>& f, const char* text) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    CHECK(rejected, text);
}
struct Memory : FoldReader {
    std::array<uint32_t, 4> words{0x77770000u, 16u << 16, 8u, 0u};
    std::vector<FoldProbe> probes;
    unsigned reads = 0;
    bool fail = false, canonical = false, changing = false;
    std::function<void()> nested;
    bool probe(FoldProbe kind, uint32_t, uint64_t, uint32_t) override {
        probes.push_back(kind); return !fail && (!canonical || kind == FoldProbe::Base40);
    }
    uint32_t word(uint32_t, uint64_t address) override {
        if (nested) { auto run = std::move(nested); nested = {}; run(); }
        const auto index = (address / 4) % 4;
        ++reads;
        return changing && reads > 4 && index == 2 ? 3 : words[index];
    }
    void prefix(uint32_t, uint64_t, void* destination, uint32_t bytes) override {
        CHECK(bytes == 4, "partial buffer copies exactly one in-range word");
        const uint32_t value = 0x11223344;
        std::memcpy(destination, &value, bytes);
    }
};
static FoldInputs input() {
    FoldInputs i;
    i.code = {0xF4080108u, 0xFA000000u, 0xE0002000u, 0x80010100u, 0xBF810000u};
    i.user.resize(12); i.user[8] = 0x12345000u; i.user_base = 8;
    i.code_address = 0x12340000; i.revision = "independent-fixture"; i.srt_requested = true;
    return i;
}
static void repair_checksum(std::vector<uint8_t>& bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (size_t n = 0; n + 8 < bytes.size(); ++n) { hash ^= bytes[n]; hash *= 1099511628211ull; }
    for (unsigned k = 0; k < 8; ++k) bytes[bytes.size() - 8 + k] = uint8_t(hash >> (8 * k));
}
static void put32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
    for (unsigned k = 0; k < 4; ++k) bytes.at(offset + k) = uint8_t(value >> (8 * k));
}
static int live(const char* mode) {
    auto i = input();
    alignas(16) const uint32_t descriptor[4] = {0x77770000u, 16u << 16, 8u, 0u};
    const auto pointer = uint64_t(uintptr_t(descriptor));
    i.user[8] = uint32_t(pointer); i.user[9] = uint32_t(pointer >> 32);
    std::vector<SrtUse> uses;
    const bool occupied = std::string_view(mode) == "occupied";
    if (occupied) { uses.emplace_back(); uses.back().key = 123; }
    const auto before = fold_workbench_evaluations.load();
    const auto out = resolve_dynamic_fetch(i.code.data(), i.code.size(), i.user.data(),
        static_cast<uint32_t>(i.user.size()), i.user_base, &uses);
    CHECK(out.size() == 1 && out[0].desc.base == 0x77770000 && out[0].desc.num_records == 8,
          "production hook returns the actual live descriptor even on recording failure/refusal");
    CHECK(fold_workbench_evaluations.load() == before + 1,
          "production evaluator executes exactly once across capture, refusal and I/O failure");
    CHECK(!occupied || (uses.size() == 1 && uses[0].key == 123), "refusal preserves initial SRT entries");
    return failures ? 1 : 0;
}
int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "--live") return live(argv[2]);
    Memory memory;
    auto capture = capture_fold(input(), &memory);
    CHECK(capture.complete && capture.fetches.size() == 1, "actual scalar load feeds one vertex fetch");
    if (capture.fetches.size() == 1) {
        const auto& f = capture.fetches[0];
        CHECK(f.fetch_pc == 2 && f.desc.base == 0x77770000 && f.desc.stride == 16 &&
              f.desc.num_records == 8 && !f.from_seed, "independent descriptor and provenance expectation");
    }
    CHECK(capture.events.size() == 9 && memory.reads == 8,
          "one probe and both four-word observations are retained");
    CHECK(capture.evaluated_instructions == 2 && capture.decoded_dwords == 5,
          "real evaluator and decoder counters are exercised");
    if (capture.fetches.size() != 1 || capture.events.size() != 9) return 1;
    if (argc == 3 && std::string_view(argv[1]) == "--emit") {
        write_fold_capture(argv[2], capture); return failures ? 1 : 0;
    }
    replay_fold(capture);
    const auto builds = fold_control_plan_builds.load();
    replay_fold(capture);
    CHECK(fold_control_plan_builds.load() == builds, "warm offline replay reuses actual control owner");

    auto restored = decode_fold_capture(encode_fold_capture(capture));
    CHECK(restored.fetches == capture.fetches && restored.uses == capture.uses,
          "portable format retains complete outputs");
    replay_fold(restored);
    Memory changing; changing.changing = true;
    auto changing_capture = capture_fold(input(), &changing);
    if (changing_capture.events.size() != 9) return 1;
    CHECK(changing_capture.events[3].data != changing_capture.events[7].data,
          "same address has different values at distinct consumption points");
    replay_fold(changing_capture);

    auto wrong = capture;
    wrong.events.erase(wrong.events.begin() + 1);
    rejects([&] { replay_fold(wrong); }, "missing event is not zero-filled");
    wrong = capture; std::swap(wrong.events[1], wrong.events[2]);
    rejects([&] { replay_fold(wrong); }, "wrong request order is rejected");
    wrong = capture; wrong.events.push_back(wrong.events.back());
    rejects([&] { replay_fold(wrong); }, "unconsumed event is rejected");
    wrong = capture; wrong.events[1].data[0] ^= 4;
    rejects([&] { replay_fold(wrong); }, "changed input needs explicit mutation mode");
    CHECK(replay_fold(wrong, false).fetches[0].desc.base != capture.fetches[0].desc.base,
          "explicit mutation actually evaluates changed input");
    wrong = capture; wrong.fetches[0].direct_user_data_index = 31;
    rejects([&] { replay_fold(wrong); }, "provenance-only output mismatch is rejected");
    wrong = capture; wrong.complete = false;
    rejects([&] { encode_fold_capture(wrong); }, "truncated capture cannot be written as evidence");
    wrong = capture; wrong.events[0].bytes = 65;
    rejects([&] { replay_fold(wrong); }, "oversized requests are rejected before evaluation");
    wrong = capture; wrong.events[1].data.clear();
    rejects([&] { replay_fold(wrong); }, "payload shortage cannot become an ordinary load failure");

    // Same vector storage, different code: stale decode metadata would incorrectly accept B.
    const auto original = capture.input.code[0];
    capture.input.code[0] = 0xBF810000u;
    rejects([&] { replay_fold(capture); }, "same-address code replacement invalidates decoding");
    capture.input.code[0] = original;
    replay_fold(capture);

    auto canonical_input = input();
    canonical_input.user[8] = 0x9abcdef0; canonical_input.user[9] = 0x12345678;
    Memory canonical; canonical.canonical = true;
    auto canonical_capture = capture_fold(canonical_input, &canonical);
    CHECK(canonical.probes == std::vector<FoldProbe>({FoldProbe::Raw, FoldProbe::Base48, FoldProbe::Base40}),
          "failed raw/Base48 probes precede successful Base40");
    if (canonical_capture.events.size() < 3) return 1;
    CHECK(canonical_capture.events[2].address == 0x789abcdef0ull,
          "canonicalization retains logical address instead of mapping it");
    replay_fold(canonical_capture);
    Memory unavailable; unavailable.fail = true;
    auto failed_capture = capture_fold(canonical_input, &unavailable);
    CHECK(failed_capture.fetches.empty() && failed_capture.events.size() == 3,
          "unreadable canonical candidates do not fabricate a descriptor");
    replay_fold(failed_capture);

    FoldInputs scalar;
    scalar.code = {0xF4300404u, 0xFA000060u, 0xE0002000u, 0x80040100u, 0xBF810000u};
    scalar.user = {0x12345000u, 0u, 100u, 0u}; scalar.user_base = 8; scalar.srt_requested = true;
    Memory partial;
    const auto partial_capture = capture_fold(scalar, &partial);
    CHECK(partial_capture.events.size() == 2 && partial_capture.events[0].bytes == 4 &&
          partial_capture.events[0].probe == FoldProbe::ScalarBuffer &&
          partial_capture.events[1].kind == FoldEventKind::Prefix,
          "partial OOB observes only the readable prefix, not the zero suffix");
    CHECK(partial.reads == 0, "partial-prefix snapshot is not reread through guest word accessor");
    CHECK(partial_capture.fetches.size() == 1 && partial_capture.fetches[0].desc.base == 0x11223344 &&
          partial_capture.fetches[0].desc.stride == 0 && partial_capture.fetches[0].desc.num_records == 0 &&
          partial_capture.fetches[0].desc_v3 == 0,
          "downstream fetch sees the loaded prefix and all three architectural zero words");
    replay_fold(partial_capture);
    partial.fail = true;
    const auto partial_failure = capture_fold(scalar, &partial);
    CHECK(partial_failure.events.size() == 1 && !partial_failure.events[0].readable,
          "unreadable in-range prefix retains only its failed probe");
    replay_fold(partial_failure);
    scalar.user[2] = 96;
    const auto all_oob = capture_fold(scalar, &partial);
    CHECK(all_oob.events.empty(), "fully OOB scalar load performs no resource read");
    replay_fold(all_oob);

    // Witness ranges establish source provenance by mapping probe alone, never payload copying.
    for (auto [offset, kind, span] : {std::tuple{0x58u, FoldProbe::OptionalTable, 272u},
                                    std::tuple{0x20u, FoldProbe::NullableOutput, 40u}}) {
        FoldInputs witness;
        witness.code = {0xf4040400u, 0xfa000000u | offset, 0xbf810000u};
        witness.user = {0x12345000u, 0u};
        Memory zeros; zeros.words.fill(0);
        const auto c = capture_fold(witness, &zeros);
        CHECK(c.events.size() >= 2 && c.events[1].kind == FoldEventKind::Probe &&
              c.events[1].probe == kind && c.events[1].bytes == span && c.events[1].data.empty(),
              "optional/null source witness records a probe without copying its range");
        CHECK(zeros.reads == 4, "two acquired words and two null-proof rereads only");
        replay_fold(decode_fold_capture(encode_fold_capture(c)));
    }
    auto many = input(); many.code.clear();
    constexpr unsigned loads = 7300;
    for (unsigned n = 0; n < loads; ++n) {
        many.code.push_back(0xF4080108u); many.code.push_back(0xFA000000u);
    }
    many.code.insert(many.code.end(), {0xE0002000u, 0x80010100u, 0xBF810000u});
    Memory exhausted;
    const auto limited = capture_fold(many, &exhausted);
    CHECK(!limited.complete && limited.events.size() == kFoldMaxEvents &&
          exhausted.reads == loads * 8 && limited.evaluated_instructions == loads + 1 &&
          limited.fetches.size() == 1 && limited.fetches[0].desc.base == 0x77770000,
          "exhausted recording still performs every live read and publishes the final fetch");
    rejects([&] { encode_fold_capture(limited); }, "budget exhaustion is unusable evidence");

    std::vector<uint32_t> dispatch_ps = {
        0xf4201a8cu, 0xfa000010u, // pc0: s_buffer_load_dword s106, s[24:27], 0x10
        0x816ac16au,              // pc2: s_add_i32 s106, s106, -1
        0x83ea826au,              // pc3: s_min_u32 s106, s106, 2
        0x8f6a836au,              // pc4: s_lshl_b32 s106, s106, 3
        0xbea01f00u,              // pc5: s_getpc_b64 s[32:33]
        0x802020ffu, 64u,         // pc6: add table byte delta (table starts at aligned pc22)
        0x82212180u,              // pc8: s_addc_u32 s33, 0, s33
        0xf4040890u, 0xd4000000u, // pc9: s_load_dwordx2 s[34:35], s[32:33], s106
        0xbea81f00u,              // pc11: s_getpc_b64 s[40:41]
        0x80282228u,              // pc12: s_add_u32 s40, s40, s34
        0x82292329u,              // pc13: s_addc_u32 s41, s41, s35
        0xbe802028u,              // pc14: s_setpc_b64 s[40:41]
        0x7e000280u,              // pc15 target A: v_mov_b32 v0, 0
        0xbf820001u,              // pc16: s_branch common export at pc18
        0x7e0002f2u,              // pc17 target B: v_mov_b32 v0, 1.0
        0xf800180fu, 0x00000000u, // pc18: exp mrt0 v0,v0,v0,v0
        0xbf810000u,              // pc20: s_endpgm
        0u,                       // pc21: alignment padding before the qword table
    };
    // Entries are signed byte offsets relative to the instruction after the second s_getpc (pc12).
    for (uint32_t index = 0; index < 3; ++index) {
        dispatch_ps.push_back(index == 0 ? 12u : index == 1 ? 20u : 24u); // pc15 / pc17 / merge pc18
        dispatch_ps.push_back(0u);
    }
    // Keep the jump-table/control layout, but make each selected arm produce a different V#.
    dispatch_ps[15] = 0xbe860388u; // s_mov_b32 s6, 8 records
    dispatch_ps[17] = 0xbe860384u; // s_mov_b32 s6, 4 records
    dispatch_ps[18] = 0xe0002000u; dispatch_ps[19] = 0x80010100u; // fetch s[4:7]
    FoldInputs dispatch_input;
    dispatch_input.code = dispatch_ps; dispatch_input.dispatch_target = 17;
    dispatch_input.user = {0x77770000u, 16u << 16, 0u, 0u}; dispatch_input.user_base = 4;
    dispatch_input.dispatch = rdna2_pcrel_dispatch_info(dispatch_ps.data(), dispatch_ps.size());
    CHECK(dispatch_input.dispatch.valid && dispatch_input.dispatch.required_dwords == 28 &&
          dispatch_input.dispatch.target_pcs == std::vector<uint32_t>({15, 17, 18}),
          "independent PC-relative table targets and declared tail");
    Memory no_resource;
    auto derived = capture_fold(dispatch_input, &no_resource);
    CHECK(derived.input.code.size() == 28 && derived.decoded_dwords == 21 &&
          derived.events.empty(), "derived dispatch consumes its owned code tail without guest resource reads");
    CHECK(derived.fetches.size() == 1 && derived.fetches[0].desc.base == 0x77770000 &&
          derived.fetches[0].desc.num_records == 4, "selected second arm actually constructs four-record fetch");
    replay_fold(decode_fold_capture(encode_fold_capture(derived)));
    dispatch_input.dispatch_present = true; dispatch_input.dispatch_target = 15;
    auto supplied = capture_fold(dispatch_input, &no_resource);
    dispatch_input.dispatch.target_pcs.clear(); // Capture owns metadata, not the caller's vector.
    CHECK(supplied.fetches.size() == 1 && supplied.fetches[0].desc.num_records == 8,
          "changed dispatch selection constructs the independent eight-record fetch");
    auto supplied_roundtrip = decode_fold_capture(encode_fold_capture(supplied));
    CHECK(supplied_roundtrip.input.dispatch.target_pcs == std::vector<uint32_t>({15, 17, 18}) &&
          supplied_roundtrip.input.dispatch.setup_pcs == supplied.input.dispatch.setup_pcs &&
          supplied_roundtrip.input.dispatch.selector_addend == -1,
          "supplied metadata and signed selector fields survive independent ownership and codec");
    replay_fold(supplied_roundtrip);

    Memory nested;
    nested.nested = [&] { replay_fold(canonical_capture); };
    replay_fold(capture_fold(input(), &nested));
    bool left = true, right = true;
    const auto worker = [&](bool& ok) {
        try { for (int n = 0; n < 40; ++n) replay_fold(capture); }
        catch (...) { ok = false; }
    };
    std::thread a(worker, std::ref(left)), b(worker, std::ref(right));
    for (int n = 0; n < 40; ++n) clear_shader_decode_cache();
    a.join(); b.join(); CHECK(left && right, "nested/parallel readers retain independent cursors");

    auto bytes = encode_fold_capture(capture);
    auto corrupt = bytes; corrupt.pop_back();
    rejects([&] { decode_fold_capture(corrupt); }, "truncated file is rejected");
    corrupt = bytes; put32(corrupt, 8, 999); repair_checksum(corrupt);
    rejects([&] { decode_fold_capture(corrupt); }, "unsupported schema beyond checksum guard");
    corrupt = bytes; corrupt[12] ^= 1; repair_checksum(corrupt);
    rejects([&] { decode_fold_capture(corrupt); }, "code identity mismatch beyond checksum guard");
    corrupt = bytes; put32(corrupt, 44 + capture.input.revision.size(), kFoldMaxCodeDwords + 1);
    repair_checksum(corrupt);
    rejects([&] { decode_fold_capture(corrupt); }, "oversized code count beyond checksum guard");

    // Populate every scalar/array output field with a non-default witness. Defaulted equality
    // is independent of the codec's field inventory and detects omitted serialization fields.
    auto fields = capture;
    SrtUse u;
    u.kind=2; u.key=3; u.t8.fill(4); u.descriptor_source_addr=5; u.v4.fill(6);
    u.table_record_count=7; u.table_entry_stride=8; u.table_element_offset=9; u.table_load_pc=10;
    u.table_base=11; u.bvh4.fill(12); u.instruction_format=13; u.zero_record_raw=true;
    u.scalar_oob_offset_known=true; u.scalar_oob_byte_offset=14; u.scalar_buffer_dword_count=15;
    u.optional_null_raw_load=true; u.proven_null_guarded_raw_store=true;
    u.proven_null_nullable_raw_buffer=true; u.has_samp=true; u.s4.fill(16);
    u.required_size=17; u.atomic_x2_record_count=18; u.use_pc=19; u.is_storage_image=true;
    u.mimg_dim=20; u.proven_zero_mip=true; u.is_depth_compare=true;
    fields.uses={u}; fields.fetches[0].desc.dst_sel[0]=7;
    fields.fetches[0].desc.dst_sel[1]=3; fields.fetches[0].desc.dst_sel[2]=2;
    fields.fetches[0].desc.dst_sel[3]=0;
    fields.fetches[0].desc.format=DataFormat::Uint32;
    fields.fetches[0].desc.num_components=3; fields.fetches[0].desc_v3=123;
    fields.fetches[0].srsrc=11;
    fields.fetches[0].desc.forbid_unknown_fallback=true;
    fields.fetches[0].unshifted_desc=fields.fetches[0].desc;
    fields.fetches[0].direct_user_data_index=9; fields.fetches[0].from_seed=true;
    fields.fetches[0].instruction_format=12; fields.fetches[0].index_mode=VertexFetchIndexMode::Instance;
    auto field_roundtrip=decode_fold_capture(encode_fold_capture(fields));
    CHECK(field_roundtrip.fetches==fields.fetches && field_roundtrip.uses==fields.uses,
          "all non-default output fields survive serialization");

    const auto directory = std::filesystem::temp_directory_path() /
        ("prosper-fold-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CHECK(std::filesystem::create_directory(directory), "reserve disposable capture directory");
    const auto path = directory / "fixture.prfold";
    write_fold_capture(path, capture); replay_fold(read_fold_capture(path));
    wrong=capture; wrong.complete=false;
    rejects([&] { write_fold_capture(path, wrong); }, "failed write preserves previous capture");
    replay_fold(read_fold_capture(path));
    const auto blocked=directory/"blocked"; std::filesystem::create_directory(blocked);
    std::ofstream(blocked/"keep") << "owned by test";
    rejects([&] { write_fold_capture(blocked, capture); }, "rename failure cannot replace destination directory");
    CHECK(std::filesystem::exists(blocked/"keep"), "existing destination is preserved");
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        CHECK(entry.path().filename().string().find(".tmp-")==std::string::npos, "writer cleans its private temporary");
    std::filesystem::remove_all(directory);
    std::cout << "fold capture: " << failures << " failures\n";
    return failures ? 1 : 0;
}
