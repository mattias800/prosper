#include "ajm_decoder.hpp"

#include <atomic>

namespace prosper::ajm {
namespace {
std::atomic<DecoderBackend*> g_backend{nullptr};
}

void set_decoder_backend(DecoderBackend* backend) { g_backend.store(backend); }
DecoderBackend* decoder_backend() { return g_backend.load(); }

} // namespace prosper::ajm
