#pragma once
// library_media.hpp — the runtime behind the library view's per-title background art and focus music
// (#1630). launcher_media.hpp holds the pure parsing and policy; this owns the thread, the Vulkan
// images and the audio stream that act on it.
//
// THREADING CONTRACT, which the rest of this file depends on:
//
//   * The worker thread does file I/O, container validation, PNG decode and the whole ATRAC9 decode.
//     It makes NO Vulkan call, touches no SDL object, and never sees an ImGui type. It communicates
//     only through the two mailboxes below.
//   * Every Vulkan and SDL call here happens on the UI thread, inside update(). VkQueue submission and
//     VkDescriptorPool allocation are externally synchronized, and command pools are owned by one
//     thread; the library's queue is also its present queue. Uploading from the worker would mean
//     locking the present path and would save nothing, because the expensive work (an 8 MB read and up
//     to ~103 ms of audio decode) is already off-thread.
//   * update() never blocks. The upload is submitted with its own fence and polled with
//     vkGetFenceStatus on later frames; the texture is published only once that fence signals. There
//     is deliberately no vkQueueWaitIdle — the cover path has one, which is tolerable for a 512x512
//     icon and a guaranteed hitch for an 8 MB background.
//
// Superseded work is discarded rather than cancelled: each request carries a generation, the mailbox
// holds a single slot that a newer request overwrites, and a result whose generation is stale is freed
// on arrival without ever reaching the upload path. Scrubbing the grid therefore costs at most one
// in-flight load plus one queued one, and no GPU work at all for titles passed over.

#include "launcher_media.hpp"

#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace prosper::frontend {

// What the UI should paint behind the grid this frame.
struct Backdrop {
    VkDescriptorSet texture = VK_NULL_HANDLE;
    float           alpha   = 0.0f;   // 0 when nothing should be drawn
    uint32_t        width   = 0;
    uint32_t        height  = 0;
};

class LibraryMedia {
public:
    ~LibraryMedia();

    // Call once per instance; a second call without an intervening shutdown() leaks this layer's pool
    // and fence and reassigns the still-joinable worker thread, which terminates the process.
    // `sampler` is borrowed from the library UI (same filtering as the covers). Returns false only if
    // the Vulkan objects this layer owns could not be created. A machine with no audio device is NOT a
    // failure — it yields a silent library, the same outcome as a title with no snd0.at9 — so check
    // music_enabled() afterwards for the effective audio state rather than this return value.
    bool init(VkPhysicalDevice phys, VkDevice device, VkQueue queue, uint32_t queue_family,
              VkSampler sampler, bool music_enabled, float music_gain);

    // Stop the worker, close the audio stream, destroy every image. Safe without a successful init and
    // safe to call twice. Called before a guest boots, so nothing here outlives the library view.
    void shutdown();

    bool ready() const { return ready_; }

    // Whether this device can sample BC7 at all. False falls the backgrounds back to pic0.png.
    bool bc_supported() const { return bcSupported_; }

    // Tell the media layer which title is focused. Cheap and idempotent: a repeat call with the same
    // root does nothing, and a change only starts the debounce, never a load.
    void set_focus(const std::string& app0_root, uint64_t now_ms);

    // Service the debounce, the mailboxes, the upload fence and the audio queue. Never blocks.
    void update(uint64_t now_ms);

    // Turn launcher music on or off while the library is up. Turning it off silences and releases the
    // decoded tracks immediately; turning it on opens the audio device if that has not happened yet and
    // fetches the focused title's track. The audio device is opened lazily so a run that never enables
    // music never touches the host's audio hardware at all.
    void set_music_enabled(bool on, uint64_t now_ms);
    bool music_enabled() const { return musicOn_; }

    Backdrop backdrop() const;

    // Diagnostics for the frame-time histogram (PROSPER_LIBRARY_STATS=1).
    uint64_t loads_started() const { return loadsStarted_; }
    uint64_t loads_discarded() const { return loadsDiscarded_; }

private:
    struct LoadRequest {
        uint64_t    generation = 0;
        std::string root;
        bool        want_music = false;
        // False when the background is already resident, so stepping back to a cached title does not
        // re-read and re-parse 8 MB of BC7 that is already on the GPU. Without this, moving between two
        // titles re-reads both files on every step even though neither image is ever rebuilt.
        bool        want_background = true;
    };

    // Everything the worker produces. Plain bytes only — no GPU or SDL object crosses the mailbox.
    struct LoadResult {
        uint64_t    generation = 0;
        std::string root;

        std::vector<uint8_t> image_bytes;    // BC7 blocks, or RGBA8 when the PNG fallback was used
        uint32_t             image_w = 0, image_h = 0;
        VkFormat             image_format = VK_FORMAT_UNDEFINED;
        bool                 image_ok = false;

        std::vector<int16_t> pcm;            // interleaved S16, fully decoded
        At9Clip              clip;
        bool                 music_ok = false;
    };

    // A background living on the GPU.
    struct Background {
        VkImage         image  = VK_NULL_HANDLE;
        VkDeviceMemory  memory = VK_NULL_HANDLE;
        VkImageView     view   = VK_NULL_HANDLE;
        VkDescriptorSet set    = VK_NULL_HANDLE;
        uint32_t        width = 0, height = 0;
    };

    // An upload in flight: submitted, not yet signalled. The staging buffer must outlive the copy, and
    // the format is carried along so the view is created with the one the image was made with — a BC7
    // upload and a PNG fallback must not be able to borrow each other's.
    struct PendingUpload {
        bool           active   = false;
        std::string    root;
        Background     bg;
        VkFormat       format   = VK_FORMAT_UNDEFINED;
        VkBuffer       staging  = VK_NULL_HANDLE;
        VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    };

    // A decoded track and where playback has reached, in sample-frames. The root is carried so a track
    // can be recognised when focus returns to the title it belongs to.
    struct Track {
        std::vector<int16_t> pcm;
        At9Clip              clip;
        uint64_t             cursor = 0;
        std::string          root;
    };

    void worker_main();
    // `rate`/`channels` are the opened stream's format, so a track that disagrees is refused
    // rather than played through the wrong interleave or at the wrong pitch.
    static void load_one(const LoadRequest& req, bool bc, int rate, int channels, LoadResult& out);

    void start_load(const std::string& root, uint64_t now_ms);
    void claim_result(uint64_t now_ms);
    void poll_upload(uint64_t now_ms);

    // Why an upload did not start. `busy` is transient and MUST be retried — conflating it with
    // `failed` throws away a decoded image and leaves that title permanently without a background.
    enum class UploadStart { started, busy, failed };
    UploadStart begin_upload(LoadResult& res);
    void        apply_result(std::unique_ptr<LoadResult> res, uint64_t now_ms);
    void install_track(LoadResult& res);
    void pump_audio(uint64_t now_ms);
    float gain_at(uint64_t play_ms, bool outgoing) const;
    bool ensure_audio();

    void mark_incoming_ready(uint64_t now_ms);
    void retire_evicted(const std::vector<std::string>& gone);
    void insert_background(const std::string& root, const Background& bg);
    void collect_retired(bool force);
    void destroy_background(Background& bg);
    void destroy_all_backgrounds();

    // --- Vulkan, UI thread only -------------------------------------------------------------------
    VkPhysicalDevice phys_    = VK_NULL_HANDLE;
    VkDevice         device_  = VK_NULL_HANDLE;
    VkQueue          queue_   = VK_NULL_HANDLE;
    uint32_t         qfamily_ = 0;
    VkSampler        sampler_ = VK_NULL_HANDLE;
    VkCommandPool    cmdPool_ = VK_NULL_HANDLE;   // this layer's own, never the UI's
    VkCommandBuffer  cmd_     = VK_NULL_HANDLE;
    VkFence          uploadFence_ = VK_NULL_HANDLE;
    PendingUpload    pending_;

    std::map<std::string, Background> cache_;
    std::vector<std::string>          lru_;

    // Evicted images awaiting destruction. An evicted background may still be referenced by a frame the
    // UI has already submitted, so it cannot be freed the moment the LRU drops it. Waiting for the
    // device to go idle would be correct but blocks the UI thread — it was measured at ~42 ms on the
    // frame an eviction happened. Instead the image is retired and destroyed a few frames later:
    // LibraryUi waits on its own in-flight fence at the top of every frame, so a frame submitted before
    // the eviction has certainly completed by then.
    struct Retired { Background bg; uint64_t frame = 0; };
    std::vector<Retired> retired_;
    uint64_t             frameCounter_ = 0;
    static constexpr uint64_t kRetireFrames = 3;

    // --- focus / transition state, UI thread only --------------------------------------------------
    std::string focusRoot_;         // what the grid is on now
    uint64_t    focusChangedMs_ = 0;
    bool        focusPending_   = false;   // focus moved; the debounce has not expired yet

    std::string shownRoot_;         // the background currently displayed (the incoming asset)
    std::string outgoingRoot_;      // the one fading out
    uint64_t    transitionMs_ = 0;  // when the current transition began (may be back-dated)
    uint64_t    readyMs_      = 0;  // when the incoming asset became available
    bool        incomingReady_ = true;
    bool        transitionActive_ = false;
    // The clock update() last saw. backdrop() is const and called during drawing, after update(), so it
    // evaluates the fade at the same instant the rest of the frame did rather than re-sampling a clock
    // that has moved on — otherwise the alpha the audio ramped to and the one drawn could disagree.
    uint64_t    lastUpdateMs_ = 0;
    std::string lastDrawn_;         // last background reported by the PROSPER_LIBRARY_STATS ground truth

    // --- audio, UI thread only (the stream is pushed, not pulled) ----------------------------------
    SDL_AudioStream* stream_    = nullptr;
    bool             audioInit_ = false;
    bool             musicOn_   = false;
    float            musicGain_ = kDefaultMusicGain;
    std::unique_ptr<Track> current_;    // the incoming/settled track
    std::unique_ptr<Track> outgoing_;   // still fading out
    // The most recently displaced track, kept WITH ITS CURSOR after its fade-out finishes.
    //
    // Without this, every settled focus change restarts a track at sample 0, so stepping between two
    // titles replays the same opening over and over — byte-identical each time, a fraction of a second
    // long, and cut off by the next change. Game themes commonly open on a chime or a swell, so what
    // that actually sounds like is a notification tone repeating, not music. Holding one track back
    // means returning to a title RESUMES it where it left off instead of replaying its first moment.
    // Two decoded tracks is the ceiling this adds: ~46 MB of host RAM at the longest observed length.
    std::unique_ptr<Track> recent_;
    int              audioRate_ = 48000;
    int              audioChannels_ = 2;

    // --- worker + mailboxes -------------------------------------------------------------------------
    std::thread             worker_;
    std::mutex              reqMutex_;
    std::condition_variable reqCv_;
    LoadRequest             request_;
    bool                    hasRequest_ = false;
    bool                    quit_       = false;

    std::mutex                  resMutex_;
    std::unique_ptr<LoadResult> result_;      // single slot; a newer result overwrites an unclaimed one
    // A claimed result whose upload could not start because one was already in flight. Held on the UI
    // thread and retried on later frames rather than discarded, so a focus change that lands inside the
    // few-frame window of a previous upload still gets its background.
    std::unique_ptr<LoadResult> deferred_;

    uint64_t generation_     = 0;
    uint64_t loadsStarted_   = 0;
    uint64_t loadsDiscarded_ = 0;
    bool     bcSupported_    = false;
    bool     ready_          = false;
};

} // namespace prosper::frontend
