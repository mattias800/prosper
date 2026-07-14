#include "video_backend.hpp"

namespace prosper::video {
namespace { VideoBackend* g_backend = nullptr; }

void          set_backend(VideoBackend* b) { g_backend = b; }
VideoBackend* backend()                    { return g_backend; }

} // namespace prosper::video
