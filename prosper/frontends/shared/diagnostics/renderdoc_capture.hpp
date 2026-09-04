#pragma once

// RenderDoc in-application capture, aimed by frame ordinal instead of by keypress.
//
// Why this exists (#3321). RenderDoc is installed on the development box and is the strongest
// frame debugger available to this project -- it answers "which draw wrote this pixel" and "what
// is in this render target at draw N", which are exactly the questions a black-world bug asks and
// exactly the ones prosper's own instruments answer worst. It was nevertheless unusable here,
// for one mechanical reason recorded in docs/GPU_PROFILING_EXTERNAL.md: `renderdoccmd capture`
// triggers on a KEYPRESS. Every agent on this project runs headless, so the best instrument in the
// toolbox was again the one nobody could aim -- the same failure the F8/F9 schedulers fixed for
// prosper's own captures (#2233), one tool over.
//
// RenderDoc's answer to this is its in-application API: the library exports exactly one symbol,
// `RENDERDOC_GetAPI`, which hands back a struct of function pointers including StartFrameCapture /
// EndFrameCapture. Those delimit a capture EXPLICITLY, so they need neither a keypress nor a
// present -- which is what makes this work under SDL_VIDEODRIVER=offscreen, where there is no
// window to press a key into and, on some routes, no swapchain present to delimit a frame at all.
//
// This header only LOADS the API. The frame triggers that aim it live with their F8/F9 siblings in
// prosper-app/main.cpp, so all four schedulable captures read as one family.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#define PROSPER_RENDERDOC_SUPPORTED 1
#else
// Windows RenderDoc uses GetModuleHandle("renderdoc.dll") rather than dlopen. Not wired up yet;
// the trigger compiles to a no-op that says so rather than silently doing nothing.
#define PROSPER_RENDERDOC_SUPPORTED 0
#endif

#include "renderdoc_app.h"

namespace prosper::frontend {

// Runtime binding to librenderdoc.so. Never linked -- the library is loaded by RenderDoc's Vulkan
// layer, and this class only asks the already-loaded copy for its API table.
class RenderDocCapture {
public:
    static RenderDocCapture& instance() {
        static RenderDocCapture inst;
        return inst;
    }

    bool available() const { return api_ != nullptr; }
    const std::string& unavailable_reason() const { return reason_; }

    // Where captures are written. This is NOT optional politeness: RenderDoc's default template is
    // under /tmp, which on this project's Linux box is a RAM-backed tmpfs with a per-user quota
    // shared by every concurrent agent -- and a single capture of a prosper frame is large enough
    // to exhaust it, which does not merely fail the write but kills the Bash tool for every agent
    // on the machine (see CLAUDE.md). So the trigger always sets this from PROSPER_CAPTURE_DIR.
    void set_path_template(const std::string& path_template) {
        if (!api_) return;
        api_->SetCaptureFilePathTemplate(path_template.c_str());
    }

    bool capturing() const { return api_ && api_->IsFrameCapturing() != 0; }

    // Stamp provenance INTO the capture file. RenderDoc displays these comments when the file is
    // opened, which is the only place a warning about the artifact survives: a reader opening a
    // .rdc weeks later never sees the stderr line that described how it was made.
    //
    // This matters most for a shutdown-truncated capture, which is the same hazard as the
    // wrong-device one this whole header exists for -- it opens perfectly, reads as a whole frame,
    // and is simply missing the draws that had not happened yet. A warning in stderr does not
    // travel with the file; this does.
    void set_comments(const std::string& path, const std::string& comments) {
        if (!api_ || path.empty()) return;
        api_->SetCaptureFileComments(path.c_str(), comments.c_str());
    }

    // WHICH DEVICE. This is the whole game, and getting it wrong produces a capture that looks
    // perfectly valid and contains no draws -- measured on The Messenger before this parameter
    // existed (#3321): 56 chunks, every one of them presentation.
    //
    // On the default headless route prosper runs TWO Vulkan instances. prosper-app owns one for
    // presentation (main.cpp), and the live renderer owns another (tests/fixtures/render_runner.h --
    // the directory name is a misnomer; that header IS the renderer), which it publishes through
    // prosper::gpu::set_shared_vulkan_context, and every guest draw and dispatch is on the
    // renderer's. Two is this route's count and not an invariant: live_compute.cpp can create a
    // third private instance when it declines to adopt the renderer's, and present unification
    // (#1270) can collapse the two into one. Hence the caller passes the device rather than the
    // trigger assuming a topology. Passing NULL lets RenderDoc pick the device, and RenderDoc documents that choice
    // as arbitrary; here it took the one holding the swapchain -- the presentation device, whose
    // entire contribution is a blit and a present. So the rule is to pass the device ALWAYS, not to
    // rely on the observed pick: a choice that is arbitrary and happens to be right will not stay
    // right.
    //
    // So the caller passes the renderer's VkInstance, converted with RenderDoc's own macro. NULL
    // remains meaningful: it captures whatever RenderDoc would have chosen, which is the right
    // behaviour when the renderer has not started and is worth keeping reachable rather than
    // asserting against.
    bool begin(void* device_pointer) {
        if (!api_) return false;
        api_->StartFrameCapture(device_pointer, nullptr);
        return true;
    }

    // Same argument as begin(): the span must be closed on the device it was opened on.

    // Returns the written capture's path, or an empty string if RenderDoc discarded the capture.
    // The distinction matters: EndFrameCapture returning 0 means the capture was ABANDONED (most
    // often because no API work was recorded between the two calls), and reporting a filename in
    // that case would send a reader to a file that does not exist.
    std::string end(void* device_pointer) {
        if (!api_) return {};
        if (api_->EndFrameCapture(device_pointer, nullptr) == 0) return {};
        const uint32_t n = api_->GetNumCaptures();
        if (n == 0) return {};
        uint32_t pathlen = 0;
        if (api_->GetCapture(n - 1, nullptr, &pathlen, nullptr) == 0 || pathlen == 0) return {};
        std::string path(pathlen, '\0');
        if (api_->GetCapture(n - 1, path.data(), &pathlen, nullptr) == 0) return {};
        if (!path.empty() && path.back() == '\0') path.pop_back();
        return path;
    }

private:
    RenderDocCapture() {
#if !PROSPER_RENDERDOC_SUPPORTED
        reason_ = "RenderDoc triggers are not wired up on this platform yet";
#else
        // RTLD_NOLOAD is the load-bearing flag, not an optimisation. RenderDoc can only capture
        // Vulkan work if librenderdoc.so was present when the instance was created -- it hooks at
        // layer-load time. By the time this runs, prosper's Vulkan instance is long since built, so
        // a fresh dlopen would hand back a perfectly valid API table attached to nothing, and every
        // capture would silently produce an empty file. NOLOAD asks the strictly correct question:
        // "is RenderDoc already in this process?" A null answer means the layer was not enabled,
        // which is a configuration error worth naming rather than a capture worth attempting.
        void* lib = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
        if (!lib) {
            reason_ = "librenderdoc.so is not loaded in this process -- run with "
                      "ENABLE_VULKAN_RENDERDOC_CAPTURE=1 so the Vulkan layer loads it before the "
                      "instance is created (enabling it later cannot work: RenderDoc hooks at "
                      "layer-load time)";
            return;
        }
        auto get_api = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(lib, "RENDERDOC_GetAPI"));
        if (!get_api) {
            reason_ = "librenderdoc.so is loaded but exports no RENDERDOC_GetAPI";
            return;
        }
        // Request 1.4.1 rather than the newest available. RenderDoc versions this struct precisely
        // so a consumer can pin the subset it uses; asking for the oldest version that has these
        // five calls means an older RenderDoc still works, and nothing here needs anything newer.
        if (get_api(eRENDERDOC_API_Version_1_4_1, reinterpret_cast<void**>(&api_)) != 1 || !api_) {
            api_ = nullptr;
            reason_ = "RENDERDOC_GetAPI refused version 1.4.1 (installed RenderDoc is too old)";
            return;
        }
        int maj = 0, min = 0, patch = 0;
        api_->GetAPIVersion(&maj, &min, &patch);
        std::fprintf(stderr, "[renderdoc] in-application API %d.%d.%d bound\n", maj, min, patch);
#endif
    }

    RENDERDOC_API_1_4_1* api_ = nullptr;
    std::string reason_;
};

}  // namespace prosper::frontend
