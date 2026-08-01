// library_media.cpp — see library_media.hpp for the threading contract, which this file is written to
// and must not drift from (#1630).
#include "library_media.hpp"

#include "hle/atrac9_decode.hpp"

#include "imgui.h"
#include "imgui_impl_vulkan.h"

// The implementation lives in library_ui.cpp; this translation unit only needs the declarations. The
// same STBI_NO_STDIO is defined there, so both see the identical set of prototypes.
#define STBI_NO_STDIO
#include "stb_image.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace prosper::frontend {
namespace {

// How much audio to keep queued. Deliberately short: audio already handed to SDL cannot be re-gained,
// so the queue depth is the worst-case lag between a focus change and the music starting to duck. At
// 60 ms it is a small fraction of the 200 ms fade-out and still three or four frames of slack against
// a 16.7 ms UI frame.
constexpr uint64_t kAudioQueueTargetMs = 60;

// Frames generated per top-up step. Short enough that the gain ramp across a fade is smooth rather
// than stepped (512 frames is ~10.7 ms at 48 kHz).
constexpr uint32_t kAudioChunkFrames = 512;

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

std::string read_file_bytes(const std::string& path) {
    std::string out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    if (size > 0) {
        out.resize(static_cast<size_t>(size));
        std::fseek(f, 0, SEEK_SET);
        const size_t got = std::fread(&out[0], 1, out.size(), f);
        out.resize(got);
    }
    std::fclose(f);
    return out;
}

} // namespace

LibraryMedia::~LibraryMedia() { shutdown(); }

bool LibraryMedia::init(VkPhysicalDevice phys, VkDevice device, VkQueue queue, uint32_t queue_family,
                        VkSampler sampler, bool music_enabled, float music_gain) {
    phys_ = phys; device_ = device; queue_ = queue; qfamily_ = queue_family; sampler_ = sampler;
    musicGain_ = music_gain;

    // BC7 needs both the device feature and the format itself usable as a sampled image. Checking only
    // the feature bit would still let a driver reject the format, and the fallback exists precisely so
    // that case degrades to pic0.png instead of to a broken image.
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(phys_, &feats);
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(phys_, VK_FORMAT_BC7_UNORM_BLOCK, &fp);
    bcSupported_ = feats.textureCompressionBC &&
                   (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
    if (!bcSupported_)
        fprintf(stderr, "[library] BC7 unavailable; backgrounds fall back to pic0.png\n");

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = qfamily_;
    if (vkCreateCommandPool(device_, &cpi, nullptr, &cmdPool_) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &cai, &cmd_) != VK_SUCCESS) { shutdown(); return false; }
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};   // created unsignalled: nothing in flight
    if (vkCreateFence(device_, &fi, nullptr, &uploadFence_) != VK_SUCCESS) { shutdown(); return false; }

    if (music_enabled) musicOn_ = ensure_audio();

    worker_ = std::thread([this] { worker_main(); });
    ready_ = true;
    return true;
}

// Open the audio device on first use. Failure is reported once and is not an error: a machine with no
// sound card gets a silent library, exactly like a title with no snd0.at9.
bool LibraryMedia::ensure_audio() {
    if (stream_) return true;
    if (!audioInit_) {
        if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
            fprintf(stderr, "[library] no audio subsystem (%s); the library stays silent\n", SDL_GetError());
            return false;
        }
        audioInit_ = true;
    }
    SDL_AudioSpec spec{};
    spec.format = SDL_AUDIO_S16;
    spec.channels = audioChannels_;
    spec.freq = audioRate_;
    // No callback: audio is pushed from update() on the UI thread, so no second thread ever reads the
    // decoded PCM and there is no lock taken inside an audio callback.
    stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
    if (!stream_) {
        fprintf(stderr, "[library] could not open audio (%s); the library stays silent\n", SDL_GetError());
        return false;
    }
    SDL_ResumeAudioStreamDevice(stream_);
    return true;
}

void LibraryMedia::set_music_enabled(bool on, uint64_t now_ms) {
    if (!ready_ || on == musicOn_) return;
    if (!on) {
        musicOn_ = false;
        // Drop what is queued as well as what is decoded: leaving up to 60 ms of audio in the device
        // would play on after the user asked for silence.
        if (stream_) SDL_ClearAudioStream(stream_);
        current_.reset();
        outgoing_.reset();
        recent_.reset();
        return;
    }
    musicOn_ = ensure_audio();
    if (!musicOn_) return;
    // Fetch the focused title's track, which was never requested while music was off. The background
    // is already cached, so this asks the worker for the audio alone.
    if (!shownRoot_.empty()) {
        LoadRequest req;
        req.generation = ++generation_;
        req.root = shownRoot_;
        req.want_music = true;
        req.want_background = cache_.find(shownRoot_) == cache_.end();   // usually already resident
        ++loadsStarted_;
        {
            std::lock_guard<std::mutex> lk(reqMutex_);
            request_ = std::move(req);
            hasRequest_ = true;
        }
        reqCv_.notify_one();
        // No transition: the background is unchanged, and the user just asked for music by clicking,
        // so the track starting promptly is the expected response rather than something to ease in.
        (void)now_ms;
    }
}

void LibraryMedia::shutdown() {
    if (worker_.joinable()) {
        { std::lock_guard<std::mutex> lk(reqMutex_); quit_ = true; }
        reqCv_.notify_all();
        worker_.join();
    }
    { std::lock_guard<std::mutex> lk(resMutex_); result_.reset(); }
    deferred_.reset();   // plain bytes, no GPU handles, but do not hold it past shutdown

    if (stream_) { SDL_DestroyAudioStream(stream_); stream_ = nullptr; }
    if (audioInit_) { SDL_QuitSubSystem(SDL_INIT_AUDIO); audioInit_ = false; }
    musicOn_ = false;
    current_.reset();
    outgoing_.reset();

    if (device_) {
        // An upload may still be in flight; its images and staging memory cannot be freed under it.
        if (pending_.active && uploadFence_)
            vkWaitForFences(device_, 1, &uploadFence_, VK_TRUE, UINT64_MAX);
        vkDeviceWaitIdle(device_);
        if (pending_.active) {
            if (pending_.staging)    vkDestroyBuffer(device_, pending_.staging, nullptr);
            if (pending_.stagingMem) vkFreeMemory(device_, pending_.stagingMem, nullptr);
            destroy_background(pending_.bg);
            pending_ = PendingUpload{};
        }
        collect_retired(true);   // the device is idle here, so anything still retired can go now
        destroy_all_backgrounds();
        if (uploadFence_) { vkDestroyFence(device_, uploadFence_, nullptr); uploadFence_ = VK_NULL_HANDLE; }
        if (cmdPool_) { vkDestroyCommandPool(device_, cmdPool_, nullptr); cmdPool_ = VK_NULL_HANDLE; cmd_ = VK_NULL_HANDLE; }
    }
    ready_ = false;
}

// --- worker ---------------------------------------------------------------------------------------

void LibraryMedia::worker_main() {
    for (;;) {
        LoadRequest req;
        {
            std::unique_lock<std::mutex> lk(reqMutex_);
            reqCv_.wait(lk, [this] { return hasRequest_ || quit_; });
            if (quit_) return;
            req = std::move(request_);
            hasRequest_ = false;
        }
        auto res = std::make_unique<LoadResult>();
        res->generation = req.generation;
        res->root = req.root;
        load_one(req, bcSupported_, *res);
        {
            // Overwrite any result the UI has not claimed: generations only increase, so an unclaimed
            // older one is already superseded and dropping it here saves the UI the work of doing so.
            std::lock_guard<std::mutex> lk(resMutex_);
            result_ = std::move(res);
        }
    }
}

void LibraryMedia::load_one(const LoadRequest& req, bool bc, LoadResult& out) {
    const std::string& root = req.root;
    for (const BackgroundCandidate& cand : req.want_background ? background_candidates(root, bc)
                                                               : std::vector<BackgroundCandidate>{}) {
        const std::string bytes = read_file_bytes(cand.path);
        if (bytes.empty()) continue;
        if (cand.block_compressed) {
            const DdsBc7Image img = parse_dds_bc7(bytes);
            if (!img.ok) {
                fprintf(stderr, "[library] %s: %s\n", cand.path.c_str(), img.reason);
                continue;   // try the PNG rather than giving up on a background entirely
            }
            out.image_bytes.assign(bytes.begin() + static_cast<ptrdiff_t>(img.data_offset),
                                   bytes.begin() + static_cast<ptrdiff_t>(img.data_offset + img.data_size));
            out.image_w = img.width;
            out.image_h = img.height;
            out.image_format = img.srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
            out.image_ok = true;
            break;
        }
        int w = 0, h = 0, ch = 0;
        stbi_uc* px = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()),
                                            static_cast<int>(bytes.size()), &w, &h, &ch, 4);
        if (!px || w <= 0 || h <= 0) { if (px) stbi_image_free(px); continue; }
        out.image_bytes.assign(px, px + static_cast<size_t>(w) * h * 4);
        stbi_image_free(px);
        out.image_w = static_cast<uint32_t>(w);
        out.image_h = static_cast<uint32_t>(h);
        out.image_format = VK_FORMAT_R8G8B8A8_UNORM;
        out.image_ok = true;
        break;
    }

    if (!req.want_music) return;
    const std::string bytes = read_file_bytes(music_path_for(root));
    if (bytes.empty()) return;      // 17 of 30 dumps: absent is normal, and silent is the right answer
    const At9Clip clip = parse_at9_riff(bytes);
    if (!clip.ok) {
        fprintf(stderr, "[library] %s/sce_sys/snd0.at9: %s\n", root.c_str(), clip.reason);
        return;
    }
    Atrac9Decoder dec;
    if (!dec.init(clip.config)) {
        fprintf(stderr, "[library] %s: ATRAC9 config rejected\n", root.c_str());
        return;
    }
    // The container and the codec must agree about the superframe, or the cursor arithmetic below is
    // built on a number the decoder does not share. This has held on all 13 local tracks.
    if (dec.superframe_bytes() != static_cast<int>(clip.block_align) ||
        dec.superframe_samples() != static_cast<int>(clip.samples_per_block)) {
        fprintf(stderr, "[library] %s: superframe mismatch (%d/%u bytes, %d/%u samples)\n", root.c_str(),
                dec.superframe_bytes(), clip.block_align, dec.superframe_samples(), clip.samples_per_block);
        return;
    }

    const size_t blocks = clip.data_size / clip.block_align;
    const size_t perBlock = static_cast<size_t>(dec.superframe_samples()) * dec.channels();
    out.pcm.resize(blocks * perBlock);
    for (size_t b = 0; b < blocks; b++) {
        const uint8_t* in = reinterpret_cast<const uint8_t*>(bytes.data()) + clip.data_offset +
                            b * clip.block_align;
        // Bound every read by what is actually left in the file: LibAtrac9's bit reader takes no input
        // length and will parse past the end of a corrupt block otherwise.
        const int avail = static_cast<int>(bytes.size() - (clip.data_offset + b * clip.block_align));
        if (dec.decode_superframe(in, out.pcm.data() + b * perBlock, avail) != dec.superframe_samples()) {
            // Keep what decoded rather than dropping the track: a damaged tail costs the end of the
            // loop, silence costs the whole feature for that title.
            fprintf(stderr, "[library] %s: ATRAC9 decode stopped at superframe %zu/%zu\n",
                    root.c_str(), b, blocks);
            out.pcm.resize(b * perBlock);
            break;
        }
    }
    if (out.pcm.empty()) return;
    out.clip = clip;
    out.music_ok = true;
}

// --- focus / requests -------------------------------------------------------------------------------

void LibraryMedia::set_focus(const std::string& app0_root, uint64_t now_ms) {
    if (!ready_ || app0_root == focusRoot_) return;
    focusRoot_ = app0_root;
    focusChangedMs_ = now_ms;
    focusPending_ = true;
}

void LibraryMedia::start_load(const std::string& root, uint64_t now_ms) {
    ++generation_;
    ++loadsStarted_;
    if (const char* s = SDL_getenv("PROSPER_LIBRARY_STATS"))
        if (*s && s[0] != '0')
            fprintf(stderr, "[library] load #%llu gen=%llu %s\n", (unsigned long long)loadsStarted_,
                    (unsigned long long)generation_, root.c_str());

    // Begin the transition NOW, in parallel with the load, so the load latency hides behind the
    // fade-out instead of adding to it. Worst measured decode is ~103 ms against a 200 ms fade.
    //
    // The start time is BACK-DATED when a transition is already running, so that evaluating the fade at
    // `now_ms` reproduces the alpha currently on screen. Restarting from a clean `now` would snap the
    // outgoing asset back to full opacity — a visible flash when the user scrubs. Back-dating makes an
    // interruption continuous by construction, without special-casing which stage it interrupted, and
    // without changing the pure fade function.
    float visible = 1.0f;
    if (transitionActive_) {
        const FadeState s = fade_state_at(transitionMs_, now_ms, incomingReady_, readyMs_);
        visible = s.old_alpha > 0.0f ? s.old_alpha : s.new_alpha;
        // Whatever is on screen becomes the thing that fades out.
        outgoingRoot_ = s.old_alpha > 0.0f ? outgoingRoot_ : shownRoot_;
        if (outgoing_ == nullptr || s.old_alpha <= 0.0f) outgoing_ = std::move(current_);
    } else {
        outgoingRoot_ = shownRoot_;
        outgoing_ = std::move(current_);
    }
    current_.reset();
    // Returning to a title whose track is still held? Resume it at its own cursor rather than decoding
    // it again and replaying its opening. This is what stops repeated stepping between two titles from
    // sounding like the same short tone over and over.
    bool haveTrack = false;
    if (musicOn_ && recent_ && recent_->root == root) {
        current_ = std::move(recent_);
        haveTrack = true;
    }
    if (visible < 0.0f) visible = 0.0f;
    if (visible > 1.0f) visible = 1.0f;
    const uint64_t backdate = static_cast<uint64_t>((1.0f - visible) * static_cast<float>(kFadeOutMs));
    transitionMs_ = now_ms >= backdate ? now_ms - backdate : 0;
    transitionActive_ = true;
    incomingReady_ = false;
    shownRoot_ = root;

    // A cached background is available immediately; only the music still has to be fetched.
    const bool cached = cache_.find(root) != cache_.end();
    if (cached) { incomingReady_ = true; readyMs_ = transitionMs_ + kFadeOutMs; }

    LoadRequest req;
    req.generation = generation_;
    req.root = root;
    req.want_music = musicOn_ && !haveTrack;   // resumed tracks need no decode
    req.want_background = !cached;
    // Everything this title needs is already in hand.
    if (cached && !req.want_music) return;
    {
        std::lock_guard<std::mutex> lk(reqMutex_);
        request_ = std::move(req);
        hasRequest_ = true;   // overwrites any request not yet picked up: newest focus wins
    }
    reqCv_.notify_one();
}

void LibraryMedia::claim_result(uint64_t now_ms) {
    std::unique_ptr<LoadResult> res;
    {
        std::lock_guard<std::mutex> lk(resMutex_);
        res = std::move(result_);
    }
    // A result parked by an earlier frame goes first: it is older, and retrying it is what stops a
    // focus change inside a previous upload's window from silently losing its background.
    if (deferred_) {
        std::unique_ptr<LoadResult> d = std::move(deferred_);
        apply_result(std::move(d), now_ms);
    }
    if (!res) return;
    apply_result(std::move(res), now_ms);
}

void LibraryMedia::apply_result(std::unique_ptr<LoadResult> res, uint64_t now_ms) {
    if (!res) return;
    if (res->generation != generation_) {
        ++loadsDiscarded_;   // superseded while in flight; never reaches the upload path
        return;
    }
    install_track(*res);
    if (res->image_ok && cache_.find(res->root) == cache_.end()) {
        const UploadStart r = begin_upload(*res);
        if (r == UploadStart::busy) {
            // Transient: an upload is still in flight. Park it and try again on a later frame rather
            // than dropping a decoded image and leaving this title with no background at all.
            deferred_ = std::move(res);
            return;
        }
        if (r == UploadStart::failed) {
            // Genuinely no background for this title. The transition must still finish, or the view
            // would hold at nothing forever.
            incomingReady_ = true;
            readyMs_ = now_ms;
        }
        return;
    }
    incomingReady_ = true;
    readyMs_ = now_ms;
}

// --- upload -----------------------------------------------------------------------------------------

LibraryMedia::UploadStart LibraryMedia::begin_upload(LoadResult& res) {
    // One upload in flight at a time. This is BUSY, not a failure: the caller must retry it later.
    if (pending_.active) return UploadStart::busy;

    Background bg;
    bg.width = res.image_w;
    bg.height = res.image_h;

    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = res.image_format;
    ii.extent = {res.image_w, res.image_h, 1};
    ii.mipLevels = 1; ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device_, &ii, nullptr, &bg.image) != VK_SUCCESS) return UploadStart::failed;

    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(device_, bg.image, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_memory_type(phys_, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(device_, &ai, nullptr, &bg.memory) != VK_SUCCESS ||
        vkBindImageMemory(device_, bg.image, bg.memory, 0) != VK_SUCCESS) {
        destroy_background(bg);
        return UploadStart::failed;
    }

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = res.image_bytes.size();
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bool ok = vkCreateBuffer(device_, &bi, nullptr, &staging) == VK_SUCCESS;
    if (ok) {
        VkMemoryRequirements bmr{};
        vkGetBufferMemoryRequirements(device_, staging, &bmr);
        VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        bai.allocationSize = bmr.size;
        bai.memoryTypeIndex = find_memory_type(phys_, bmr.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        ok = bai.memoryTypeIndex != UINT32_MAX &&
             vkAllocateMemory(device_, &bai, nullptr, &stagingMem) == VK_SUCCESS &&
             vkBindBufferMemory(device_, staging, stagingMem, 0) == VK_SUCCESS;
    }
    if (ok) {
        void* mapped = nullptr;
        ok = vkMapMemory(device_, stagingMem, 0, res.image_bytes.size(), 0, &mapped) == VK_SUCCESS;
        if (ok) {
            std::memcpy(mapped, res.image_bytes.data(), res.image_bytes.size());
            vkUnmapMemory(device_, stagingMem);
        }
    }
    if (ok) {
        vkResetCommandBuffer(cmd_, 0);
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_, &cbi);
        VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image = bg.image;
        toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toDst);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {bg.width, bg.height, 1};
        vkCmdCopyBufferToImage(cmd_, staging, bg.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier toRead = toDst;
        toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toRead);
        vkEndCommandBuffer(cmd_);

        vkResetFences(device_, 1, &uploadFence_);
        VkSubmitInfo su{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        su.commandBufferCount = 1; su.pCommandBuffers = &cmd_;
        // Submitted and left to run. Waiting here is what the cover path does and is exactly the hitch
        // this feature cannot afford at 8 MB; poll_upload() picks it up when the fence signals.
        ok = vkQueueSubmit(queue_, 1, &su, uploadFence_) == VK_SUCCESS;
    }
    if (!ok) {
        if (staging)    vkDestroyBuffer(device_, staging, nullptr);
        if (stagingMem) vkFreeMemory(device_, stagingMem, nullptr);
        destroy_background(bg);
        return UploadStart::failed;
    }
    pending_.active = true;
    pending_.root = res.root;
    pending_.bg = bg;
    pending_.format = res.image_format;
    pending_.staging = staging;
    pending_.stagingMem = stagingMem;
    return UploadStart::started;
}

void LibraryMedia::poll_upload(uint64_t now_ms) {
    if (!pending_.active) return;
    if (vkGetFenceStatus(device_, uploadFence_) != VK_SUCCESS) return;   // still copying; try next frame

    if (pending_.staging)    vkDestroyBuffer(device_, pending_.staging, nullptr);
    if (pending_.stagingMem) vkFreeMemory(device_, pending_.stagingMem, nullptr);
    pending_.staging = VK_NULL_HANDLE;
    pending_.stagingMem = VK_NULL_HANDLE;

    Background bg = pending_.bg;
    VkImageViewCreateInfo ivi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivi.image = bg.image;
    ivi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    // Exactly the format the image was created with, carried on the pending upload rather than
    // re-derived here, so a BC7 upload and a PNG fallback cannot borrow each other's.
    ivi.format = pending_.format;
    ivi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    bool ok = vkCreateImageView(device_, &ivi, nullptr, &bg.view) == VK_SUCCESS;
    if (ok) {
        bg.set = ImGui_ImplVulkan_AddTexture(sampler_, bg.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        ok = bg.set != VK_NULL_HANDLE;
    }
    if (!ok) {
        fprintf(stderr, "[library] could not publish the background for %s\n", pending_.root.c_str());
        destroy_background(bg);
    } else {
        insert_background(pending_.root, bg);
    }
    const std::string root = pending_.root;
    pending_ = PendingUpload{};

    if (root == shownRoot_) { incomingReady_ = true; readyMs_ = now_ms; }
    if (const char* s = SDL_getenv("PROSPER_LIBRARY_STATS"))
        if (*s && s[0] != '0')
            // "the background for X is now on screen" — the only line that means a capture taken after
            // it will actually contain that title's art.
            fprintf(stderr, "[library] showing %s\n", root.c_str());
}

void LibraryMedia::insert_background(const std::string& root, const Background& bg) {
    cache_[root] = bg;
    for (const std::string& gone : lru_touch(lru_, root, kBackgroundCacheCapacity)) {
        auto it = cache_.find(gone);
        if (it == cache_.end()) continue;
        // Retired, not destroyed: an in-flight frame may still sample it. Freeing it here would need a
        // device wait, which is a measurable stall on exactly the frame the user moved the selection.
        retired_.push_back({it->second, frameCounter_});
        cache_.erase(it);
    }
}

// Free anything retired long enough ago that no submitted frame can still reference it.
void LibraryMedia::collect_retired(bool force) {
    for (size_t i = 0; i < retired_.size();) {
        if (force || frameCounter_ - retired_[i].frame >= kRetireFrames) {
            destroy_background(retired_[i].bg);
            retired_.erase(retired_.begin() + static_cast<ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
}

void LibraryMedia::destroy_background(Background& bg) {
    if (bg.set)    ImGui_ImplVulkan_RemoveTexture(bg.set);
    if (bg.view)   vkDestroyImageView(device_, bg.view, nullptr);
    if (bg.image)  vkDestroyImage(device_, bg.image, nullptr);
    if (bg.memory) vkFreeMemory(device_, bg.memory, nullptr);
    bg = Background{};
}

void LibraryMedia::destroy_all_backgrounds() {
    for (auto& kv : cache_) destroy_background(kv.second);
    cache_.clear();
    lru_.clear();
}

// --- audio -------------------------------------------------------------------------------------------

void LibraryMedia::install_track(LoadResult& res) {
    if (!musicOn_) return;
    if (!res.music_ok) return;    // no snd0.at9, or it did not parse: this title is simply silent
    if (current_ && current_->root == res.root) return;   // already resumed from the retained track
    auto t = std::make_unique<Track>();
    t->pcm = std::move(res.pcm);
    t->clip = res.clip;
    t->cursor = 0;
    t->root = res.root;
    current_ = std::move(t);
}

float LibraryMedia::gain_at(uint64_t play_ms, bool outgoing) const {
    if (!transitionActive_) return outgoing ? 0.0f : musicGain_;
    const FadeState s = fade_state_at(transitionMs_, play_ms, incomingReady_, readyMs_);
    return (outgoing ? s.old_alpha : s.new_alpha) * musicGain_;
}

void LibraryMedia::pump_audio(uint64_t now_ms) {
    if (!musicOn_ || !stream_) return;
    const int bytesPerFrame = audioChannels_ * 2;
    const uint64_t targetFrames = static_cast<uint64_t>(kAudioQueueTargetMs) * audioRate_ / 1000;
    int queuedBytes = SDL_GetAudioStreamQueued(stream_);
    if (queuedBytes < 0) queuedBytes = 0;
    uint64_t queuedFrames = static_cast<uint64_t>(queuedBytes) / bytesPerFrame;

    std::vector<int16_t> buf;
    while (queuedFrames < targetFrames) {
        const uint32_t want = static_cast<uint32_t>(
            std::min<uint64_t>(kAudioChunkFrames, targetFrames - queuedFrames));

        // The gain is evaluated at the time this audio will actually be HEARD, not the time it is
        // generated, so the music fade stays locked to the visual one despite the queue.
        const uint64_t playMs = now_ms + queuedFrames * 1000 / audioRate_;
        const uint64_t endMs = now_ms + (queuedFrames + want) * 1000 / audioRate_;

        // Only one track is ever audible: the fade is sequential, so whichever side has gain owns the
        // chunk. Ties cannot happen — fade_state_at guarantees the two are never both non-zero.
        const float outA0 = gain_at(playMs, true), outA1 = gain_at(endMs, true);
        const float inA0 = gain_at(playMs, false), inA1 = gain_at(endMs, false);
        const bool useOutgoing = (outA0 > 0.0f || outA1 > 0.0f);
        Track* track = useOutgoing ? outgoing_.get() : current_.get();
        const float a0 = useOutgoing ? outA0 : inA0;
        const float a1 = useOutgoing ? outA1 : inA1;

        buf.assign(static_cast<size_t>(want) * audioChannels_, 0);
        if (track && !track->pcm.empty()) {
            const uint64_t total = track->clip.total_frames();
            const size_t haveFrames = track->pcm.size() / audioChannels_;
            for (uint32_t i = 0; i < want; i++) {
                const float t = want > 1 ? static_cast<float>(i) / static_cast<float>(want - 1) : 1.0f;
                const float g = a0 + (a1 - a0) * t;
                const uint64_t f = track->cursor;
                if (f < haveFrames) {
                    for (int c = 0; c < audioChannels_; c++) {
                        const int32_t s = track->pcm[f * audioChannels_ + c];
                        buf[static_cast<size_t>(i) * audioChannels_ + c] =
                            static_cast<int16_t>(s * g);
                    }
                }
                // Honours the smpl loop when the track declared a usable one, so a title with a 25 s
                // intro does not replay it every pass and one with a 12 s ending never reaches it.
                track->cursor = at9_next_frame(track->clip, f);
                if (track->cursor >= total && total > 0) track->cursor = 0;
            }
        }
        if (!SDL_PutAudioStreamData(stream_, buf.data(),
                                    static_cast<int>(buf.size() * sizeof(int16_t))))
            break;
        queuedFrames += want;
    }
}

// --- per-frame --------------------------------------------------------------------------------------

void LibraryMedia::update(uint64_t now_ms) {
    if (!ready_) return;
    lastUpdateMs_ = now_ms;
    ++frameCounter_;
    collect_retired(false);

    // Start a load only once focus has rested. Holding an arrow key steps far faster than the debounce,
    // so a sweep across the grid starts exactly one load: the one the user stopped on.
    if (focusPending_ && focus_load_due(focusChangedMs_, now_ms)) {
        focusPending_ = false;
        if (focusRoot_ != shownRoot_) start_load(focusRoot_, now_ms);
    }

    claim_result(now_ms);
    poll_upload(now_ms);

    if (transitionActive_) {
        const FadeState s = fade_state_at(transitionMs_, now_ms, incomingReady_, readyMs_);
        if (s.done) {
            transitionActive_ = false;
            outgoingRoot_.clear();
            // Retain the faded-out track WITH its cursor instead of freeing it, so coming back to that
            // title continues its music rather than replaying the first fraction of a second. Exactly
            // one is held, so this costs one decoded track, not a growing cache.
            if (outgoing_) recent_ = std::move(outgoing_);
            outgoing_.reset();
        }
    }
    pump_audio(now_ms);

    // Ground truth for verification: what is actually on screen right now, logged when it changes.
    // The "showing" line above only says an upload finished, which is not the same thing — the fade may
    // still be displaying the previous title, and a capture taken on that line alone can catch the
    // wrong art. This is the line a screenshot should be synchronized to.
    if (const char* s = SDL_getenv("PROSPER_LIBRARY_STATS")) {
        if (*s && s[0] != '0') {
            const Backdrop bd = backdrop();
            std::string drawn;
            if (bd.texture) {
                for (const auto& kv : cache_)
                    if (kv.second.set == bd.texture) { drawn = kv.first; break; }
            }
            if (drawn != lastDrawn_) {
                lastDrawn_ = drawn;
                fprintf(stderr, "[library] drawing %s\n", drawn.empty() ? "(nothing)" : drawn.c_str());
            }
        }
    }
}

Backdrop LibraryMedia::backdrop() const {
    Backdrop b;
    if (!ready_) return b;

    std::string root = shownRoot_;
    float alpha = 1.0f;
    if (transitionActive_) {
        const FadeState s = fade_state_at(transitionMs_, lastUpdateMs_, incomingReady_, readyMs_);
        if (s.old_alpha > 0.0f) { root = outgoingRoot_; alpha = s.old_alpha; }
        else                    { root = shownRoot_;    alpha = s.new_alpha; }
    }
    if (root.empty() || alpha <= 0.0f) return b;
    const auto it = cache_.find(root);
    if (it == cache_.end()) return b;
    b.texture = it->second.set;
    b.alpha = alpha;
    b.width = it->second.width;
    b.height = it->second.height;
    return b;
}

} // namespace prosper::frontend
