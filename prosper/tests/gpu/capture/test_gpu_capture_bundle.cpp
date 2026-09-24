#include "gpu/capture/gpu_capture_bundle.hpp"
#include "gpu/timeline/gpu_timeline.hpp"     // interactive F9 whole-frame grab state machine
#include "gpu/texture/tile.hpp"

#include <chrono>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include "fixtures/test_scratch.h"
#include "fixtures/interactive_capture_wait.h"

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_gpu_capture_bundle ==\n");
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = prosper_test::test_scratch_dir() /
        ("prosper-gpu-bundle-" + std::to_string(nonce) + ".prgbundle");
    const auto corrupt_path = path.string() + ".corrupt";

    GpuCaptureFile first;
    first.metadata.width = 64; first.metadata.height = 64;
    first.metadata.submit_index = 41; first.metadata.revision = "bundle-test";
    GpuCaptureBlob blob;
    blob.guest_addr = 0x100000; blob.bytes_read = 256 * 1024;
    blob.bytes.resize(static_cast<size_t>(blob.bytes_read));
    uint32_t random = 0x12345678u;
    for (auto& byte : blob.bytes) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        byte = static_cast<uint8_t>(random);
    }
    blob.content_hash = gpu_capture_hash(blob.bytes);
    first.blobs.push_back(blob);
    GpuCapturedDmaCopy bundle_dma;
    bundle_dma.dst = 0x100000; bundle_dma.src = 0x100010; bundle_dma.bytes = 4;
    bundle_dma.command_order = 850; bundle_dma.packet_addr = 0x123400;
    bundle_dma.destination_blob_index = 0; bundle_dma.source_blob_index = 0;
    bundle_dma.source_blob_offset = 16;
    first.dma_copies.push_back(bundle_dma);
    first.operations.push_back({SubmitOperationKind::DmaCopy, 0, 850, true});
    first.operations.push_back({SubmitOperationKind::Draw, 5, 900, false});
    GpuCaptureRawShaderVersion failed_shader;
    failed_shader.words = {0xbf860001u, 0xbf810000u};
    failed_shader.has_endpgm = true;
    failed_shader.content_hash = gpu_capture_hash(
        reinterpret_cast<const uint8_t*>(failed_shader.words.data()), failed_shader.words.size() * 4);
    first.raw_shader_versions.push_back(failed_shader);
    GpuCapturedOperationFailure failed_operation;
    failed_operation.kind = SubmitOperationKind::Draw;
    failed_operation.source_index = 5;
    failed_operation.command_order = 900;
    failed_operation.reason = RealizationFailureReason::ShaderRecompile;
    GpuCapturedStageDiagnostic failed_stage;
    failed_stage.stage = ShaderProgramStage::Fragment;
    failed_stage.program_addr = 0x400000;
    failed_stage.raw_shader_index = 0;
    failed_stage.coverage.total = 1;
    failed_stage.coverage.unsupported = 1;
    failed_stage.coverage.first_bad_fmt = 2;
    failed_stage.coverage.first_bad_op = 6;
    failed_stage.coverage.first_bad_pc = 0;
    failed_operation.stages.push_back(failed_stage);
    first.failure_diagnostics.push_back(failed_operation);
    first.failure_diagnostics_available = true;
    GpuCaptureDsSeed ds_seed;
    ds_seed.depth_read_base = ds_seed.depth_write_base = 0x310000;
    ds_seed.htile_data_base = 0x300000; ds_seed.width = 2; ds_seed.height = 2;
    ds_seed.format = GpuCaptureDsFormat::D32Float;
    ds_seed.depth_valid = true; ds_seed.depth.assign(16, 0x5a);
    first.ds_seeds.push_back(ds_seed);
    GpuCaptureFile second = first;
    second.metadata.submit_index = 42;
    second.metadata.input_route = "shift-the-serialized-blob-boundary";
    second.blobs[0].guest_addr = 0x200000;

    GpuCaptureFile stale = second;
    stale.metadata.submit_index = 43;
    stale.blobs[0].bytes[17] ^= 0x80;

    GpuCaptureBundle bundle;
    std::string error;
    CHECK(append_gpu_capture_bundle(bundle, first, error), "first submit appends to bundle");
    CHECK(append_gpu_capture_bundle(bundle, second, error), "second submit appends in order");
    CHECK(bundle.version == 2 && bundle.resources.size() == 1 &&
          bundle.submits[0].blob_resource_indices == bundle.submits[1].blob_resource_indices,
          "same content at different addresses reuses one exact resource");
    CHECK(!append_gpu_capture_bundle(bundle, stale, error) &&
          error.find("content hash mismatch") != std::string::npos &&
          bundle.resources.size() == 1 && bundle.submits.size() == 2,
          "failed append rolls back and cannot reuse a stale same-address resource");
    stale.blobs[0].content_hash = gpu_capture_hash(stale.blobs[0].bytes);
    CHECK(append_gpu_capture_bundle(bundle, stale, error) && bundle.resources.size() == 2 &&
          bundle.submits[2].blob_resource_indices[0] != bundle.submits[1].blob_resource_indices[0],
          "changed content at the same address creates a distinct resource version");
    CHECK(bundle.submits.size() == 3 &&
          gpu_capture_bundle_unique_bytes(bundle) * 10 < bundle.logical_bytes * 7,
          "resource dictionary and manifest chunks reduce physical bytes");
    const GpuCaptureBundleStats stats = gpu_capture_bundle_stats(bundle);
    CHECK(stats.resource_reference_count == 3 && stats.exact_reuse_count == 1 &&
          stats.resource_logical_bytes == blob.bytes.size() * 3,
          "bundle statistics distinguish logical references from exact reuse");
    GpuCaptureBundle rolling = bundle;
    rolling.submits.erase(rolling.submits.begin(), rolling.submits.begin() + 2);
    rolling.logical_bytes = rolling.submits[0].logical_bytes;
    const uint64_t rolling_before = gpu_capture_bundle_unique_bytes(rolling);
    GpuCaptureFile rolling_restored;
    CHECK(compact_gpu_capture_bundle(rolling, error) && rolling.resources.size() == 1 &&
          gpu_capture_bundle_unique_bytes(rolling) < rolling_before &&
          materialize_gpu_capture_bundle_submit(rolling, 0, rolling_restored, error) &&
          rolling_restored.blobs[0].bytes == stale.blobs[0].bytes,
          "rolling-window compaction removes unreachable resources and chunks");
    CHECK(write_gpu_capture_bundle(path.string(), bundle, error), "bundle writes atomically");

    GpuCaptureBundle loaded;
    CHECK(read_gpu_capture_bundle(path.string(), loaded, error), "checksummed bundle reads");
    GpuCaptureFile restored_first, restored_second;
    GpuCaptureFile second_manifest;
    CHECK(materialize_gpu_capture_bundle_submit(loaded, 0, restored_first, error) &&
          materialize_gpu_capture_bundle_submit(loaded, 1, restored_second, error) &&
          restored_second.metadata.submit_index == 42 && restored_second.blobs.size() == 1 &&
          restored_second.blobs[0].guest_addr == 0x200000 && restored_second.blobs[0].bytes == blob.bytes &&
          restored_second.failure_diagnostics_available && restored_second.failure_diagnostics.size() == 1 &&
          restored_second.raw_shader_versions[0].words == failed_shader.words &&
          restored_second.dma_copies.size() == 1 &&
          restored_second.dma_copies[0].source_blob_offset == 16 &&
          restored_second.ds_seeds.size() == 1 && restored_second.ds_seeds[0].depth == ds_seed.depth,
          "bundle reconstructs exact validated capture and ordered DMA records");
    CHECK(materialize_gpu_capture_bundle_manifest(loaded, 1, second_manifest, error) &&
          second_manifest.metadata.submit_index == 42 && second_manifest.blobs.size() == 1 &&
          second_manifest.blobs[0].bytes.empty() && second_manifest.draws.size() == restored_second.draws.size() &&
          second_manifest.dma_copies.size() == 1 &&
          second_manifest.ds_seeds.size() == 1 && second_manifest.ds_seeds[0].depth == ds_seed.depth,
          "manifest-only materialization exposes DMA state without resource payload reconstruction");
    restored_first.blobs[0].bytes[0] ^= 0xff;
    CHECK(restored_second.blobs[0].bytes == blob.bytes,
          "materialized submits own independent mutable resource bytes");

    GpuCaptureFile dcc_capture;
    dcc_capture.metadata.submit_index = 51;
    dcc_capture.metadata.revision = "bundle-dcc-test";
    ShaderResource dcc_resource{};
    dcc_resource.cls = ResourceClass::Texture; dcc_resource.binding = 7;
    dcc_resource.gpu_addr = 0x500000; dcc_resource.size = 32 * 32 * 32 * 4;
    dcc_resource.format = DataFormat::Sint32; dcc_resource.num_components = 1;
    dcc_resource.img_dim = 2; dcc_resource.width = 32; dcc_resource.height = 32;
    dcc_resource.depth = 32; dcc_resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
    dcc_resource.compression_enabled = true; dcc_resource.meta_pipe_aligned = true;
    dcc_resource.metadata_addr = 0x900000;
    GpuCaptureBlob dcc_base;
    dcc_base.guest_addr = dcc_resource.gpu_addr;
    dcc_base.bytes.resize(static_cast<size_t>(gpu_capture_resource_footprint(dcc_resource)), 0x41);
    dcc_base.bytes_read = dcc_base.bytes.size();
    dcc_base.content_hash = gpu_capture_hash(dcc_base.bytes);
    GpuCaptureBlob dcc_metadata;
    dcc_metadata.guest_addr = dcc_resource.metadata_addr;
    dcc_metadata.bytes.resize(static_cast<size_t>(gpu_capture_dcc_metadata_footprint(dcc_resource)), 0x82);
    dcc_metadata.bytes_read = dcc_metadata.bytes.size();
    dcc_metadata.content_hash = gpu_capture_hash(dcc_metadata.bytes);
    dcc_capture.blobs = {dcc_base, dcc_metadata};
    GpuCapturedDraw dcc_draw;
    dcc_draw.vs = {0x07230203, 51}; dcc_draw.fs = {0x07230203, 52};
    dcc_draw.vrt.present = true;
    GpuCapturedResource captured_dcc;
    captured_dcc.resource = dcc_resource; captured_dcc.blob_index = 0;
    captured_dcc.metadata_size = dcc_metadata.bytes.size(); captured_dcc.metadata_blob_index = 1;
    dcc_draw.vrt.resources.push_back(captured_dcc);
    dcc_capture.draws.push_back(dcc_draw);
    GpuCaptureBundle dcc_bundle;
    GpuCaptureFile dcc_manifest, dcc_restored;
    GpuReplayFrame dcc_replay;
    CHECK(append_gpu_capture_bundle(dcc_bundle, dcc_capture, error) &&
          materialize_gpu_capture_bundle_manifest(dcc_bundle, 0, dcc_manifest, error) &&
          dcc_manifest.blobs.size() == 2 && dcc_manifest.blobs[1].bytes.empty() &&
          dcc_manifest.draws[0].vrt.resources[0].metadata_blob_index == 1 &&
          materialize_gpu_capture_bundle_submit(dcc_bundle, 0, dcc_restored, error) &&
          materialize_gpu_replay(dcc_restored, dcc_replay, error) &&
          dcc_replay.items[0].vrt->resources[0].dcc_metadata_host_data_size == dcc_metadata.bytes.size(),
          "bundle manifest preserves separate DCC references while payloads stay deduplicated");

    GpuCaptureBundle malformed = loaded;
    malformed.submits[0].blob_resource_indices[0] =
        static_cast<uint32_t>(malformed.resources.size());
    GpuCaptureFile rejected_capture;
    CHECK(!materialize_gpu_capture_bundle_submit(malformed, 0, rejected_capture, error) &&
          error.find("invalid resource") != std::string::npos,
          "materialization rejects an out-of-range resource reference");

    const auto v1_path = path.string() + ".v1";
    std::vector<uint8_t> first_bytes;
    CHECK(serialize_gpu_capture(first, first_bytes, error), "created a legacy capture payload");
    GpuCaptureBundle legacy;
    legacy.version = 1; legacy.chunk_bytes = 16u << 20;
    legacy.logical_bytes = first_bytes.size();
    legacy.chunks.push_back(first_bytes); legacy.chunk_hashes.push_back(gpu_capture_hash(first_bytes));
    GpuCaptureBundleSubmit legacy_submit;
    legacy_submit.submit_index = first.metadata.submit_index;
    legacy_submit.logical_bytes = legacy_submit.manifest_bytes = first_bytes.size();
    legacy_submit.chunk_indices.push_back(0); legacy.submits.push_back(legacy_submit);
    CHECK(write_gpu_capture_bundle(v1_path, legacy, error), "writer preserves the legacy v1 layout");
    GpuCaptureBundle loaded_legacy;
    GpuCaptureFile restored_legacy;
    CHECK(read_gpu_capture_bundle(v1_path, loaded_legacy, error) && loaded_legacy.version == 1 &&
          materialize_gpu_capture_bundle_submit(loaded_legacy, 0, restored_legacy, error) &&
          restored_legacy.blobs[0].bytes == blob.bytes,
          "reader and materializer remain compatible with v1 bundles");

    std::ifstream input(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    if (!bytes.empty()) bytes.back() ^= 0x80;
    std::ofstream corrupt(corrupt_path, std::ios::binary);
    corrupt.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    corrupt.close();
    GpuCaptureBundle rejected;
    CHECK(!read_gpu_capture_bundle(corrupt_path, rejected, error) &&
          error.find("checksum") != std::string::npos,
          "bundle rejects manifest corruption");

    // An interactive frame grab claims its output names when it is ARMED, so a grab that never
    // completes (the app was killed mid-capture) leaves a zero-byte .prgbundle on disk. That file must
    // be unmistakable to the tool: an empty capture is "this never finished", not "this frame had no
    // work" and not a corruption to go hunting for.
    const auto empty_path = path.string() + ".empty";
    { std::ofstream empty(empty_path, std::ios::binary); }
    GpuCaptureBundle from_empty;
    CHECK(!read_gpu_capture_bundle(empty_path, from_empty, error) &&
          error.find("empty bundle file") != std::string::npos &&
          error.find("never completed") != std::string::npos,
          "an empty bundle is rejected as an unfinished capture, by name");
    CHECK(from_empty.submits.empty(),
          "an empty bundle never loads as a capture that simply contained no submits");

    std::error_code ec; std::filesystem::remove(path, ec); std::filesystem::remove(corrupt_path, ec);
    std::filesystem::remove(empty_path, ec);
    std::filesystem::remove(v1_path, ec);

    // Interactive F9 whole-frame grab (prosper-app): the present-boundary state machine. A grab is armed
    // from the app's main thread; the NEXT present starts capturing a fresh frame, each submit appends,
    // and the following present writes the .prgbundle and disarms. Here we drive the present transitions
    // headlessly (the submit-capture + write + replay are validated on a display host via prosper-app F9)
    // to prove the on-demand contract: the per-submit hook is provably inert until F9 arms a grab.
    using prosper::gpu::interactive_capture_bundle_active;
    using prosper::gpu::request_interactive_capture_bundle;
    using prosper::gpu::record_gpu_timeline_present;
    // 1-frame window so start+end is a simple two-present sequence (cross-platform env set).
#ifdef _WIN32
    _putenv_s("PROSPER_CAPTURE_FRAMES", "1");
#else
    setenv("PROSPER_CAPTURE_FRAMES", "1", 1);
#endif
    CHECK(!interactive_capture_bundle_active(), "frame-bundle grab is inert by default");
    record_gpu_timeline_present(1, 0, 0, 1920, 1080);
    CHECK(!interactive_capture_bundle_active(), "a present with no armed grab stays inert");
    (void)request_interactive_capture_bundle(prosper_test::test_scratch_file("prosper_frame_grab_unit.prgbundle"));
    CHECK(interactive_capture_bundle_active(), "F9 arms a whole-frame grab");
    record_gpu_timeline_present(2, 0, 0, 1920, 1080);   // start capturing the next frame
    CHECK(interactive_capture_bundle_active(), "the next present starts the frame (still active)");
    record_gpu_timeline_present(3, 0, 0, 1920, 1080);   // end the frame (no submits captured here)
    CHECK(!interactive_capture_bundle_active(),
          "the following present ends the frame and disarms (no re-arm)");
    InteractiveGrabOutcome first_empty;
    CHECK(wait_for_interactive_grab(first_empty) && !first_empty.ok &&
          first_empty.bundle_path.find("prosper_frame_grab_unit.prgbundle") != std::string::npos,
          "the first empty window reports its own failure before rearming");
    CHECK(first_empty.opened_present == 2 && first_empty.closed_presents ==
              std::vector<uint64_t>({3}) && first_empty.closed_submit_counts ==
              std::vector<uint64_t>({0}) && first_empty.captured_submits.empty() &&
              first_empty.opened_source_flip == 0 &&
              first_empty.closed_source_flips == std::vector<uint64_t>({0}),
          "an empty interrupted window reports boundaries but no fabricated source identity");
    (void)request_interactive_capture_bundle(prosper_test::test_scratch_file("prosper_frame_grab_unit2.prgbundle"));
    CHECK(interactive_capture_bundle_active(), "the grab can be re-armed for another press");
    record_gpu_timeline_present(4, 0, 0, 1920, 1080);
    record_gpu_timeline_present(5, 0, 0, 1920, 1080);
    CHECK(!interactive_capture_bundle_active(), "the re-armed grab completes and disarms");

    // #1587: the user pressed a key, so every completed grab must leave a collectable outcome. When
    // the only report was a line on stderr among tens of thousands, a failed grab was
    // indistinguishable from a keystroke that never registered — several agents on a 4K title read it
    // that way. Assert the SPECIFIC outcome, not merely that one exists.
    {
        InteractiveGrabOutcome outcome;
        // Both grabs above ended with an empty window (no submits recorded between the presents).
        // That is a failure the user needs told, not silence.
        CHECK(wait_for_interactive_grab(outcome),
              "#1587: a completed grab leaves an outcome for the frontend to report");
        CHECK(!outcome.ok && !outcome.error.empty(),
              "#1587: an empty-window grab reports failure with a reason, not silence");
        CHECK(outcome.bundle_path.find("prosper_frame_grab_unit2") != std::string::npos,
              "#1587: the outcome names the exact grab it belongs to");
        CHECK(outcome.max_unique_bytes > 0,
              "#1587: the outcome carries the budget in force, so the frontend can name the remedy");
        // Take-once: a second poll must not re-report the same grab, or the frontend would re-raise
        // the same notice on every frame.
        InteractiveGrabOutcome again;
        CHECK(!take_interactive_grab_outcome(again),
              "#1587: an outcome is reported exactly once");
    }

    // The budget is settable per grab, which is the whole point of #1587: the F9 path was pinned to
    // the default with no way to raise it, and one 3840x2160 deferred frame exceeds 2 GiB.
    {
        (void)request_interactive_capture_bundle(
            prosper_test::test_scratch_file("prosper_frame_grab_budget.prgbundle"), 3072);
        record_gpu_timeline_present(6, 0, 0, 3840, 2160);
        record_gpu_timeline_present(7, 0, 0, 3840, 2160);
        InteractiveGrabOutcome budget;
        CHECK(wait_for_interactive_grab(budget) && budget.max_unique_bytes == (3072ull << 20),
              "#1587: a requested budget of 3072 MiB is the budget actually in force");
        // And the clamp still holds at the top end.
        (void)request_interactive_capture_bundle(
            prosper_test::test_scratch_file("prosper_frame_grab_clamp.prgbundle"), 99999);
        record_gpu_timeline_present(8, 0, 0, 3840, 2160);
        record_gpu_timeline_present(9, 0, 0, 3840, 2160);
        InteractiveGrabOutcome clamped;
        CHECK(wait_for_interactive_grab(clamped) && clamped.max_unique_bytes == (3072ull << 20),
              "#1587: an over-large request clamps to the 3072 MiB ceiling rather than overflowing");
        // Restore the default budget: this state is global, and leaving it at 3072 would silently
        // change the ceiling every later grab in this binary runs under.
        (void)request_interactive_capture_bundle(
            prosper_test::test_scratch_file("prosper_frame_grab_restore.prgbundle"), 2048);
        record_gpu_timeline_present(10, 0, 0, 1920, 1080);
        record_gpu_timeline_present(11, 0, 0, 1920, 1080);
        InteractiveGrabOutcome restored;
        CHECK(wait_for_interactive_grab(restored) && restored.max_unique_bytes == (2048ull << 20),
              "#1587: the budget is restored to the default for later grabs");
    }

    // Persistent depth/stencil images can predate the captured frame (for example a shadow-atlas
    // cascade retained from the previous frame). F9 must snapshot them at the capture boundary and
    // attach them to the first submit so bundle replay does not start those planes at zero (#1307).
    const auto ds_bundle_path = prosper_test::test_scratch_dir() /
        ("prosper-f9-ds-seed-" + std::to_string(nonce) + ".prgbundle");
    unsigned ds_snapshots = 0;
    set_gpu_capture_ds_seed_snapshot_reader(
        [&](std::vector<GpuCaptureDsSeed>& seeds, std::string&) {
            ++ds_snapshots; seeds = {ds_seed}; return true;
        });
    (void)request_interactive_capture_bundle(ds_bundle_path.string());
    record_gpu_timeline_present(6, 0, 0, 1920, 1080, 1006);
    GpuState empty_submit;
    record_gpu_timeline_submit(empty_submit, 77);
    record_gpu_timeline_submit(empty_submit, 78);
    record_gpu_timeline_present(7, 0, 0, 1920, 1080, 1007);
    InteractiveGrabOutcome ok_outcome;
    CHECK(wait_for_interactive_grab(ok_outcome), "owned DS bundle writer completes");
    CHECK(ok_outcome.ok && ok_outcome.opened_present == 6 &&
              ok_outcome.closed_presents == std::vector<uint64_t>({7}) &&
              ok_outcome.opened_source_flip == 1006 &&
              ok_outcome.closed_source_flips == std::vector<uint64_t>({1007}) &&
              ok_outcome.closed_submit_counts == std::vector<uint64_t>({2}) &&
              ok_outcome.captured_submits == std::vector<uint64_t>({77, 78}),
          "successful writer reports exact selected-flip IDs and serialized submit prefix");
    GpuCaptureBundle ds_bundle;
    GpuCaptureFile ds_first, ds_second;
    CHECK(ds_snapshots == 1 && read_gpu_capture_bundle(ds_bundle_path.string(), ds_bundle, error) &&
          ds_bundle.submits.size() == 2 &&
          materialize_gpu_capture_bundle_submit(ds_bundle, 0, ds_first, error) &&
          materialize_gpu_capture_bundle_submit(ds_bundle, 1, ds_second, error) &&
          ds_first.ds_seeds.size() == 1 && ds_first.ds_seeds[0].depth == ds_seed.depth &&
          ds_second.ds_seeds.empty(),
          "F9 bundle seeds only its first submit with capture-boundary depth/stencil state");
    // #1587: this is the only grab in the file that SUCCEEDS, so it is the only place the ok-path
    // publish can be covered. Every other new assertion exercises a failure outcome; without this one
    // a regression that reported success as a failure — or published nothing at all on success —
    // would pass the suite. It also restores the consume-once invariant: leaving this grab's outcome
    // uncollected would hand it to whichever later test polls next.
    {
        CHECK(ok_outcome.ok &&
              ok_outcome.error.empty() &&
              ok_outcome.bundle_path == ds_bundle_path.string(),
              "#1587: a grab that writes its bundle publishes ok with no error and the exact path");
    }
    set_gpu_capture_ds_seed_snapshot_reader({});
    std::filesystem::remove(ds_bundle_path, ec);

    // The app must pick the closing selected-front image before the async writer finishes. Hold
    // that writer deliberately: polling its outcome would already be too late to choose pixels.
    std::atomic<bool> release_writer{false};
    CHECK(set_interactive_bundle_writer_hook_for_test([&] {
        while (!release_writer.load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }), "the closure test owns an idle interactive writer");
    const auto closure_path = prosper_test::test_scratch_file("prosper_f9_closure.prgbundle");
    (void)request_interactive_capture_bundle(closure_path);
    uint64_t closed_flip = 0;
    record_gpu_timeline_present(3001, 0, 0, 1920, 1080, 4001);
    CHECK(!interactive_grab_closed_source_flip(closure_path, closed_flip),
          "opening a one-frame window cannot nominate the opening image as its screenshot");
    record_gpu_timeline_present(3002, 0, 0, 1920, 1080, 4002);
    CHECK(interactive_grab_closed_source_flip(closure_path, closed_flip) && closed_flip == 4002 &&
          !interactive_grab_closed_source_flip(closure_path + ".other", closed_flip),
          "the owned closing flip is available while the writer is held; another path cannot adopt it");
    InteractiveGrabOutcome premature;
    CHECK(!take_interactive_grab_outcome(premature),
          "the synchronous target is independent of delayed bundle serialization");
    release_writer.store(true, std::memory_order_release);
    InteractiveGrabOutcome closed_outcome;
    CHECK(wait_for_interactive_grab(closed_outcome) &&
          closed_outcome.closed_source_flips == std::vector<uint64_t>({4002}),
          "the eventual writer outcome agrees with the synchronous closing target");
    CHECK(set_interactive_bundle_writer_hook_for_test({}),
          "the delayed-writer test restores ordinary writing");

    // A two-frame bundle must nominate its final boundary, never its first closed frame. A new
    // request resets the old path's token even while retaining that previous outcome in the test.
#ifdef _WIN32
    _putenv_s("PROSPER_CAPTURE_FRAMES", "2");
#else
    setenv("PROSPER_CAPTURE_FRAMES", "2", 1);
#endif
    const auto multi_path = prosper_test::test_scratch_file("prosper_f9_multi_closure.prgbundle");
    (void)request_interactive_capture_bundle(multi_path);
    CHECK(!interactive_grab_closed_source_flip(closure_path, closed_flip),
          "a later request cannot see the earlier request's closing flip");
    record_gpu_timeline_present(3003, 0, 0, 1920, 1080, 4003);
    record_gpu_timeline_present(3004, 0, 0, 1920, 1080, 4004);
    CHECK(!interactive_grab_closed_source_flip(multi_path, closed_flip),
          "an intermediate boundary does not choose a multi-frame screenshot");
    record_gpu_timeline_present(3005, 0, 0, 1920, 1080, 4005);
    CHECK(interactive_grab_closed_source_flip(multi_path, closed_flip) && closed_flip == 4005,
          "the final selected-front flip chooses the multi-frame screenshot");
    InteractiveGrabOutcome multi_outcome;
    CHECK(wait_for_interactive_grab(multi_outcome) &&
          multi_outcome.closed_source_flips == std::vector<uint64_t>({4004, 4005}),
          "the multi-frame writer retains both exact boundaries");
#ifdef _WIN32
    _putenv_s("PROSPER_CAPTURE_FRAMES", "1");
#else
    setenv("PROSPER_CAPTURE_FRAMES", "1", 1);
#endif
    std::filesystem::remove(closure_path, ec);
    std::filesystem::remove(multi_path, ec);

    // #3828: the guest closing flip can run before the final renderer callback has reached the
    // capture hook. Outer Wilds closed flip 1648 at submit 58376, then displayed that same flip
    // with completed producer 58377. The owned GPU F9 mode must retain the later submit before
    // serialization; a guest-present-only closure fails this exact arm.
    using prosper::gpu::InteractiveGrabClosure;
    using prosper::gpu::InteractiveGrabProducerClose;
    using prosper::gpu::InteractiveGrabProducerLimits;
    using prosper::gpu::interactive_grab_pending_source_flip;
    using prosper::gpu::interactive_grab_on_host_presented_source;
    using prosper::gpu::interactive_grab_on_cpu_fallback;
    using prosper::gpu::interactive_grab_expire_producer_wait;
    const auto late_path = prosper_test::test_scratch_file("prosper_f9_late_producer.prgbundle");
    CHECK(request_interactive_capture_bundle(
              late_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer).accepted,
          "#3828: the owned GPU F9 mode arms a one-frame bundle");
    record_gpu_timeline_present(4100, 0, 0, 1920, 1080, 1647);
    record_gpu_timeline_submit(empty_submit, 58376);
    record_gpu_timeline_present(4101, 0, 0, 1920, 1080, 1648);
    uint64_t pending_flip = 0;
    CHECK(interactive_capture_bundle_active() &&
              interactive_grab_pending_source_flip(late_path, pending_flip) &&
              pending_flip == 1648 &&
              !interactive_grab_closed_source_flip(late_path, closed_flip),
          "#3828: guest flip 1648 nominates, but cannot serialize before its producer arrives");
    record_gpu_timeline_submit(empty_submit, 58377);
    CHECK(interactive_grab_on_host_presented_source(late_path, 1648, true, 58377) ==
              InteractiveGrabProducerClose::Captured,
          "#3828: exact host-presented flip closes only after producer 58377 reaches the hook");
    InteractiveGrabOutcome late_outcome;
    CHECK(wait_for_interactive_grab(late_outcome) && late_outcome.ok &&
              late_outcome.closed_source_flips == std::vector<uint64_t>({1648}) &&
              late_outcome.closed_submit_counts == std::vector<uint64_t>({2}) &&
              late_outcome.captured_submits == std::vector<uint64_t>({58376, 58377}),
          "#3828: serialized bundle contains the exact completed producer after the guest flip");
    GpuCaptureBundle late_bundle;
    CHECK(read_gpu_capture_bundle(late_path, late_bundle, error) &&
              late_bundle.submits.size() == 2 &&
              late_bundle.submits.back().submit_index == 58377,
          "#3828: on-disk submit membership agrees with the closure outcome");
    std::filesystem::remove(late_path, ec);

    const auto unknown_path = prosper_test::test_scratch_file("prosper_f9_unknown_producer.prgbundle");
    CHECK(request_interactive_capture_bundle(
              unknown_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer).accepted,
          "#3828: unknown-producer refusal arms");
    record_gpu_timeline_present(4200, 0, 0, 1920, 1080, 2001);
    record_gpu_timeline_submit(empty_submit, 58400);
    record_gpu_timeline_present(4201, 0, 0, 1920, 1080, 2002);
    CHECK(interactive_grab_on_host_presented_source(unknown_path, 2002, false, 0) ==
              InteractiveGrabProducerClose::Refused,
          "#3828: the exact flip without known producer cannot certify the bundle");
    InteractiveGrabOutcome unknown_outcome;
    CHECK(wait_for_interactive_grab(unknown_outcome) && !unknown_outcome.ok &&
              unknown_outcome.error.find("without known producer") != std::string::npos,
          "#3828: unknown lineage produces an explicit incomplete result");

    const auto skipped_path = prosper_test::test_scratch_file("prosper_f9_skipped_flip.prgbundle");
    CHECK(request_interactive_capture_bundle(
              skipped_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer).accepted,
          "#3828: skipped-flip refusal arms");
    record_gpu_timeline_present(4300, 0, 0, 1920, 1080, 3001);
    record_gpu_timeline_submit(empty_submit, 58410);
    record_gpu_timeline_present(4301, 0, 0, 1920, 1080, 3002);
    CHECK(interactive_grab_on_host_presented_source(skipped_path, 3003, true, 58410) ==
              InteractiveGrabProducerClose::Refused,
          "#3828: a newer host-presented flip cannot inherit the target's producer");
    InteractiveGrabOutcome skipped_outcome;
    CHECK(wait_for_interactive_grab(skipped_outcome) && !skipped_outcome.ok &&
              skipped_outcome.error.find("skipped") != std::string::npos,
          "#3828: skipped host presentation is explicit and does not serialize");

    const auto expiry_path = prosper_test::test_scratch_file("prosper_f9_producer_expiry.prgbundle");
    CHECK(request_interactive_capture_bundle(
              expiry_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer,
              InteractiveGrabProducerLimits{1, 2, 2}).accepted,
          "#3828: bounded never-presented test arms");
    record_gpu_timeline_present(4400, 0, 0, 1920, 1080, 4001);
    record_gpu_timeline_submit(empty_submit, 58420);
    record_gpu_timeline_present(4401, 0, 0, 1920, 1080, 4002);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(interactive_grab_expire_producer_wait(expiry_path),
          "#3828: a target never shown by the host expires without another guest event");
    InteractiveGrabOutcome expired_outcome;
    CHECK(wait_for_interactive_grab(expired_outcome) && !expired_outcome.ok &&
              expired_outcome.error.find("wall-clock budget") != std::string::npos,
          "#3828: deadline expiry reports an explicit incomplete result");

    const auto submit_limit_path = prosper_test::test_scratch_file("prosper_f9_submit_limit.prgbundle");
    CHECK(request_interactive_capture_bundle(
              submit_limit_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer,
              InteractiveGrabProducerLimits{30'000, 1, 2}).accepted,
          "#3828: post-flip submit-budget test arms");
    record_gpu_timeline_present(4500, 0, 0, 1920, 1080, 5001);
    record_gpu_timeline_submit(empty_submit, 58430);
    record_gpu_timeline_present(4501, 0, 0, 1920, 1080, 5002);
    record_gpu_timeline_submit(empty_submit, 58431);
    record_gpu_timeline_submit(empty_submit, 58432);
    CHECK(interactive_grab_expire_producer_wait(submit_limit_path),
          "#3828: a late producer cannot silently extend the bounded submit window");
    InteractiveGrabOutcome submit_limit_outcome;
    CHECK(wait_for_interactive_grab(submit_limit_outcome) && !submit_limit_outcome.ok &&
              submit_limit_outcome.error.find("submit budget") != std::string::npos &&
              submit_limit_outcome.captured_submits == std::vector<uint64_t>({58430, 58431}),
          "#3828: the submit-budget refusal names only work actually retained");

    const auto present_limit_path = prosper_test::test_scratch_file("prosper_f9_present_limit.prgbundle");
    CHECK(request_interactive_capture_bundle(
              present_limit_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer,
              InteractiveGrabProducerLimits{30'000, 4, 2}).accepted,
          "#3828: post-target present-budget test arms");
    record_gpu_timeline_present(4600, 0, 0, 1920, 1080, 6001);
    record_gpu_timeline_submit(empty_submit, 58440);
    record_gpu_timeline_present(4601, 0, 0, 1920, 1080, 6002);
    record_gpu_timeline_present(4602, 0, 0, 1920, 1080, 6003);
    record_gpu_timeline_present(4603, 0, 0, 1920, 1080, 6004);
    CHECK(interactive_grab_expire_producer_wait(present_limit_path),
          "#3828: unrelated guest flips cannot keep the target window open forever");
    InteractiveGrabOutcome present_limit_outcome;
    CHECK(wait_for_interactive_grab(present_limit_outcome) && !present_limit_outcome.ok &&
              present_limit_outcome.error.find("guest-present budget") != std::string::npos &&
              present_limit_outcome.closed_source_flips == std::vector<uint64_t>({6002}),
          "#3828: present-budget refusal retains the original target, not a later flip");

    // CPU fallback may be the first successfully shown image before target nomination. The app
    // writes its BMP immediately, then clears its pending screenshot path. The bundle therefore
    // cannot rely on that path for closure; it must downgrade to bounded ordinary guest closure.
    const auto fallback_path = prosper_test::test_scratch_file("prosper_f9_cpu_fallback.prgbundle");
    CHECK(request_interactive_capture_bundle(
              fallback_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer).accepted,
          "#3828: fallback-before-target test arms producer closure");
    record_gpu_timeline_present(4700, 0, 0, 1920, 1080, 7001);
    record_gpu_timeline_submit(empty_submit, 58450);
    CHECK(interactive_grab_on_cpu_fallback(fallback_path) &&
              !interactive_grab_pending_source_flip(fallback_path, pending_flip),
          "#3828: successful CPU fallback before nomination changes the closure policy");
    record_gpu_timeline_present(4701, 0, 0, 1920, 1080, 7002);
    InteractiveGrabOutcome fallback_outcome;
    CHECK(wait_for_interactive_grab(fallback_outcome) && fallback_outcome.ok &&
              fallback_outcome.closed_source_flips == std::vector<uint64_t>({7002}) &&
              fallback_outcome.captured_submits == std::vector<uint64_t>({58450}),
          "#3828: CPU BMP path closes a real bundle without any later GPU producer callback");
    std::filesystem::remove(fallback_path, ec);

    const auto fallback_after_path =
        prosper_test::test_scratch_file("prosper_f9_cpu_fallback_after_target.prgbundle");
    CHECK(request_interactive_capture_bundle(
              fallback_after_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer).accepted,
          "#3828: fallback-after-target test arms producer closure");
    record_gpu_timeline_present(4750, 0, 0, 1920, 1080, 7501);
    record_gpu_timeline_submit(empty_submit, 58455);
    record_gpu_timeline_present(4751, 0, 0, 1920, 1080, 7502);
    CHECK(interactive_grab_on_cpu_fallback(fallback_after_path),
          "#3828: CPU fallback after the target guest flip closes the retained prefix");
    InteractiveGrabOutcome fallback_after_outcome;
    CHECK(wait_for_interactive_grab(fallback_after_outcome) && fallback_after_outcome.ok &&
              fallback_after_outcome.closed_source_flips == std::vector<uint64_t>({7502}) &&
              fallback_after_outcome.captured_submits == std::vector<uint64_t>({58455}),
          "#3828: late CPU fallback preserves a real serialized bundle for unknown_source");
    std::filesystem::remove(fallback_after_path, ec);

    const auto fallback_expiry_path =
        prosper_test::test_scratch_file("prosper_f9_cpu_fallback_expiry.prgbundle");
    CHECK(request_interactive_capture_bundle(
              fallback_expiry_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer,
              InteractiveGrabProducerLimits{1, 2, 2}).accepted &&
              interactive_grab_on_cpu_fallback(fallback_expiry_path),
          "#3828: CPU fallback while the grab is still armed starts a bounded wait");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(interactive_grab_expire_producer_wait(fallback_expiry_path),
          "#3828: the armed CPU fallback expires even after the BMP path is cleared");
    InteractiveGrabOutcome fallback_expired;
    CHECK(prosper::gpu::take_interactive_grab_outcome(fallback_expired) &&
              !fallback_expired.ok &&
              fallback_expired.error.find("CPU fallback") != std::string::npos,
          "#3828: armed fallback refusal publishes an explicit owned outcome");

    // F9 requests the producer mode even when an operator asks for a multi-frame bundle. That
    // request must downgrade to the established guest-present contract, which does not require
    // one GPU producer callback for every additional frame.
#ifdef _WIN32
    _putenv_s("PROSPER_CAPTURE_FRAMES", "2");
#else
    setenv("PROSPER_CAPTURE_FRAMES", "2", 1);
#endif
    const auto producer_multi_path =
        prosper_test::test_scratch_file("prosper_f9_producer_multi.prgbundle");
    CHECK(request_interactive_capture_bundle(
              producer_multi_path, 0, 0, InteractiveGrabClosure::PresentedGpuProducer).accepted,
          "#3828: an F9 producer-mode request with two frames arms");
    record_gpu_timeline_present(4800, 0, 0, 1920, 1080, 8001);
    record_gpu_timeline_submit(empty_submit, 58460);
    record_gpu_timeline_present(4801, 0, 0, 1920, 1080, 8002);
    CHECK(interactive_capture_bundle_active(),
          "#3828: the first of two requested frames does not close the window");
    record_gpu_timeline_present(4802, 0, 0, 1920, 1080, 8003);
    InteractiveGrabOutcome producer_multi_outcome;
    CHECK(wait_for_interactive_grab(producer_multi_outcome) && producer_multi_outcome.ok &&
              producer_multi_outcome.closed_source_flips ==
                  std::vector<uint64_t>({8002, 8003}),
          "#3828: multi-frame F9 closes at guest boundary without a producer callback");
#ifdef _WIN32
    _putenv_s("PROSPER_CAPTURE_FRAMES", "1");
#else
    setenv("PROSPER_CAPTURE_FRAMES", "1", 1);
#endif
    std::filesystem::remove(producer_multi_path, ec);

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
