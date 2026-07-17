#include "media_foundation_backend.hpp"

#ifndef _WIN32
#error "The Media Foundation backend is Windows-only"
#endif

#include <windows.h>
#include <codecapi.h>
#include <combaseapi.h>
#include <d3d10.h>
#include <d3d11.h>
#include <dxgi.h>
#include <icodecapi.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mftransform.h>
#include <propvarutil.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prosper::video {
namespace {

constexpr size_t kVideoQueueCapacity = 8;
constexpr size_t kAudioQueueCapacity = 32;

bool avp_log() { return std::getenv("PROSPER_AVPLOG") != nullptr; }
bool software_decode_allowed() { return std::getenv("PROSPER_AVP_ALLOW_SOFTWARE") != nullptr; }

void log_hresult(const char* operation, HRESULT hr, const std::string& path = {}) {
    if (!avp_log()) return;
    if (path.empty()) {
        std::fprintf(stderr, "[avp-mf] %s failed: 0x%08lx\n", operation,
                     static_cast<unsigned long>(hr));
    } else {
        std::fprintf(stderr, "[avp-mf] %s path='%s' failed: 0x%08lx\n", operation,
                     path.c_str(), static_cast<unsigned long>(hr));
    }
}

template <typename T>
class ComPtr {
public:
    ComPtr() = default;
    ~ComPtr() { reset(); }
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    ComPtr(ComPtr&& other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}
    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) { reset(); ptr_ = std::exchange(other.ptr_, nullptr); }
        return *this;
    }

    T* get() const { return ptr_; }
    T** put() { reset(); return &ptr_; }
    T* operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    void reset(T* replacement = nullptr) {
        if (ptr_) ptr_->Release();
        ptr_ = replacement;
    }

private:
    T* ptr_ = nullptr;
};

std::wstring utf8_to_wide(const std::string& input) {
    if (input.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input.data(),
                                           static_cast<int>(input.size()), nullptr, 0);
    if (count <= 0) return {};
    std::wstring output(static_cast<size_t>(count), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, input.data(),
                            static_cast<int>(input.size()), output.data(), count) != count) return {};
    return output;
}

bool is_uncompressed_video(REFGUID subtype) {
    return IsEqualGUID(subtype, MFVideoFormat_NV12) || IsEqualGUID(subtype, MFVideoFormat_YUY2) ||
           IsEqualGUID(subtype, MFVideoFormat_RGB32) || IsEqualGUID(subtype, MFVideoFormat_ARGB32);
}

struct VideoPacket {
    std::vector<uint8_t> bytes;
    uint32_t width = 0, height = 0, stride = 0;
    uint64_t pts_us = 0;
};

struct AudioPacket {
    std::vector<int16_t> pcm;
    uint32_t channels = 0, sample_rate = 0;
    uint64_t pts_us = 0;
};

struct D3DDecodeContext {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> immediate;
    ComPtr<IMFDXGIDeviceManager> manager;
    UINT reset_token = 0;
};

struct Session {
    std::string path;
    bool allow_software = false;

    std::mutex mutex;
    std::condition_variable cv;
    bool initialized = false;
    bool init_ok = false;
    HRESULT init_error = E_FAIL;
    bool stopping = false;
    bool decode_done = false;
    bool decode_error = false;
    bool video_eof = false;
    bool audio_eof = false;
    bool hardware_video = false;

    DWORD video_stream = MF_SOURCE_READER_INVALID_STREAM_INDEX;
    DWORD audio_stream = MF_SOURCE_READER_INVALID_STREAM_INDEX;
    StreamInfo stream_info;
    uint32_t video_stride = 0;

    std::deque<VideoPacket> video_queue;
    std::deque<AudioPacket> audio_queue;
    VideoPacket last_video;
    AudioPacket last_audio;
    std::jthread thread;
};

bool read_presentation_duration(IMFSourceReader* reader, uint64_t& duration_us) {
    PROPVARIANT value;
    PropVariantInit(&value);
    const HRESULT hr = reader->GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE,
                                                         MF_PD_DURATION, &value);
    if (FAILED(hr)) return false;
    bool ok = false;
    if (value.vt == VT_UI8) { duration_us = value.uhVal.QuadPart / 10; ok = true; }
    else if (value.vt == VT_I8 && value.hVal.QuadPart >= 0) {
        duration_us = static_cast<uint64_t>(value.hVal.QuadPart) / 10; ok = true;
    }
    PropVariantClear(&value);
    return ok;
}

bool discover_streams(IMFSourceReader* reader, Session& session, GUID& native_video_subtype) {
    if (FAILED(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE))) return false;
    for (DWORD stream = 0;; ++stream) {
        ComPtr<IMFMediaType> type;
        const HRESULT hr = reader->GetNativeMediaType(stream, 0, type.put());
        if (hr == MF_E_INVALIDSTREAMNUMBER) break;
        if (FAILED(hr)) continue;
        GUID major{};
        if (FAILED(type->GetGUID(MF_MT_MAJOR_TYPE, &major))) continue;
        if (session.video_stream == MF_SOURCE_READER_INVALID_STREAM_INDEX &&
            IsEqualGUID(major, MFMediaType_Video)) {
            session.video_stream = stream;
            type->GetGUID(MF_MT_SUBTYPE, &native_video_subtype);
        } else if (session.audio_stream == MF_SOURCE_READER_INVALID_STREAM_INDEX &&
                   IsEqualGUID(major, MFMediaType_Audio)) {
            session.audio_stream = stream;
        }
    }
    return session.video_stream != MF_SOURCE_READER_INVALID_STREAM_INDEX;
}

bool configure_video(IMFSourceReader* reader, Session& session) {
    if (FAILED(reader->SetStreamSelection(session.video_stream, TRUE))) return false;
    ComPtr<IMFMediaType> requested;
    if (FAILED(MFCreateMediaType(requested.put())) ||
        FAILED(requested->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) ||
        FAILED(requested->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12)) ||
        FAILED(reader->SetCurrentMediaType(session.video_stream, nullptr, requested.get()))) return false;

    ComPtr<IMFMediaType> current;
    if (FAILED(reader->GetCurrentMediaType(session.video_stream, current.put()))) return false;
    UINT32 width = 0, height = 0;
    if (FAILED(MFGetAttributeSize(current.get(), MF_MT_FRAME_SIZE, &width, &height)) ||
        width == 0 || height == 0) return false;
    UINT32 numerator = 0, denominator = 0;
    MFGetAttributeRatio(current.get(), MF_MT_FRAME_RATE, &numerator, &denominator);
    UINT32 stride_bits = width;
    current->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride_bits);
    const int32_t signed_stride = static_cast<int32_t>(stride_bits);
    if (signed_stride <= 0 || static_cast<uint32_t>(signed_stride) < width) return false;

    session.stream_info.width = width;
    session.stream_info.height = height;
    session.stream_info.fps = denominator ? static_cast<float>(numerator) / denominator : 0.0f;
    session.video_stride = static_cast<uint32_t>(signed_stride);
    return true;
}

bool configure_audio(IMFSourceReader* reader, Session& session) {
    if (session.audio_stream == MF_SOURCE_READER_INVALID_STREAM_INDEX) return true;
    if (FAILED(reader->SetStreamSelection(session.audio_stream, TRUE))) return false;
    ComPtr<IMFMediaType> requested;
    if (FAILED(MFCreateMediaType(requested.put())) ||
        FAILED(requested->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio)) ||
        FAILED(requested->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_PCM)) ||
        FAILED(requested->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 16)) ||
        FAILED(reader->SetCurrentMediaType(session.audio_stream, nullptr, requested.get()))) return false;

    ComPtr<IMFMediaType> current;
    UINT32 channels = 0, rate = 0, bits = 0;
    if (FAILED(reader->GetCurrentMediaType(session.audio_stream, current.put())) ||
        FAILED(current->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &channels)) ||
        FAILED(current->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &rate)) ||
        FAILED(current->GetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, &bits)) ||
        channels == 0 || rate == 0 || bits != 16) return false;
    session.stream_info.has_audio = true;
    session.stream_info.audio_channels = channels;
    session.stream_info.audio_rate = rate;
    return true;
}

bool create_d3d_decode_context(IMFAttributes* reader_attributes, D3DDecodeContext& d3d) {
    D3D_FEATURE_LEVEL feature_level{};
    const UINT flags = D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, nullptr, 0,
                                   D3D11_SDK_VERSION, d3d.device.put(), &feature_level,
                                   d3d.immediate.put());
    if (FAILED(hr)) return false;
    // Media Foundation's decoder uses the device on internal threads while this worker copies
    // decoded surfaces through the immediate context. Microsoft explicitly recommends enabling
    // D3D multithread protection for this shared decoder-device scenario.
    ComPtr<ID3D10Multithread> multithread;
    hr = d3d.device->QueryInterface(IID_ID3D10Multithread,
                                    reinterpret_cast<void**>(multithread.put()));
    if (FAILED(hr)) { d3d.immediate.reset(); d3d.device.reset(); return false; }
    multithread->SetMultithreadProtected(TRUE);
    hr = MFCreateDXGIDeviceManager(&d3d.reset_token, d3d.manager.put());
    if (FAILED(hr)) { d3d.immediate.reset(); d3d.device.reset(); return false; }
    hr = d3d.manager->ResetDevice(d3d.device.get(), d3d.reset_token);
    if (FAILED(hr)) {
        d3d.manager.reset(); d3d.immediate.reset(); d3d.device.reset(); return false;
    }
    hr = reader_attributes->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, d3d.manager.get());
    if (FAILED(hr)) {
        d3d.manager.reset(); d3d.immediate.reset(); d3d.device.reset(); return false;
    }
    return true;
}

bool transform_has_hardware_url(IMFTransform* transform) {
    ComPtr<IMFAttributes> attributes;
    if (FAILED(transform->GetAttributes(attributes.put()))) return false;
    UINT32 length = 0;
    return SUCCEEDED(attributes->GetStringLength(MFT_ENUM_HARDWARE_URL_Attribute, &length)) && length > 0;
}

bool find_video_decoder(IMFSourceReader* reader, DWORD video_stream,
                        ComPtr<IMFTransform>& decoder, bool& hardware_url, bool& d3d11_aware) {
    ComPtr<IMFSourceReaderEx> extended;
    if (FAILED(reader->QueryInterface(IID_IMFSourceReaderEx,
                                      reinterpret_cast<void**>(extended.put())))) return false;
    for (DWORD index = 0;; ++index) {
        GUID category{};
        ComPtr<IMFTransform> transform;
        const HRESULT hr = extended->GetTransformForStream(video_stream, index, &category,
                                                            transform.put());
        if (hr == MF_E_INVALIDINDEX || hr == MF_E_INVALIDSTREAMNUMBER) break;
        if (FAILED(hr)) break;
        if (IsEqualGUID(category, MFT_CATEGORY_VIDEO_DECODER)) {
            hardware_url = transform_has_hardware_url(transform.get());
            ComPtr<IMFAttributes> attributes;
            UINT32 aware = FALSE;
            if (SUCCEEDED(transform->GetAttributes(attributes.put())))
                attributes->GetUINT32(MF_SA_D3D11_AWARE, &aware);
            d3d11_aware = aware != FALSE;
            decoder = std::move(transform);
            return true;
        }
    }
    return false;
}

int decoder_dxva_mode(IMFTransform* decoder) {
    if (!decoder) return eAVDecVideoDXVAMode_NOTPLAYING;
    ComPtr<ICodecAPI> codec;
    if (FAILED(decoder->QueryInterface(IID_ICodecAPI,
                                       reinterpret_cast<void**>(codec.put()))))
        return eAVDecVideoDXVAMode_NOTPLAYING;
    VARIANT value;
    VariantInit(&value);
    const HRESULT hr = codec->GetValue(&CODECAPI_AVDecVideoDXVAMode, &value);
    int mode = eAVDecVideoDXVAMode_NOTPLAYING;
    if (SUCCEEDED(hr)) {
        if (value.vt == VT_I4) mode = value.lVal;
        else if (value.vt == VT_UI4) mode = static_cast<int>(value.ulVal);
    }
    VariantClear(&value);
    return mode;
}

bool copy_dxgi_video_sample(IMFMediaBuffer* buffer, D3DDecodeContext& d3d,
                            Session& session, VideoPacket& packet) {
    ComPtr<IMFDXGIBuffer> dxgi_buffer;
    if (FAILED(buffer->QueryInterface(IID_IMFDXGIBuffer,
                                      reinterpret_cast<void**>(dxgi_buffer.put())))) return false;
    ComPtr<ID3D11Texture2D> texture;
    if (FAILED(dxgi_buffer->GetResource(IID_ID3D11Texture2D,
                                        reinterpret_cast<void**>(texture.put())))) return false;
    UINT subresource = 0;
    if (FAILED(dxgi_buffer->GetSubresourceIndex(&subresource))) return false;
    D3D11_TEXTURE2D_DESC source_desc{};
    texture->GetDesc(&source_desc);
    if (source_desc.Format != DXGI_FORMAT_NV12 || source_desc.Width < session.stream_info.width ||
        source_desc.Height < session.stream_info.height) return false;

    D3D11_TEXTURE2D_DESC staging_desc = source_desc;
    staging_desc.MipLevels = 1;
    staging_desc.ArraySize = 1;
    staging_desc.SampleDesc.Count = 1;
    staging_desc.SampleDesc.Quality = 0;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    ComPtr<ID3D11Texture2D> staging;
    if (FAILED(d3d.device->CreateTexture2D(&staging_desc, nullptr, staging.put()))) return false;
    d3d.immediate->CopySubresourceRegion(staging.get(), 0, 0, 0, 0, texture.get(), subresource,
                                         nullptr);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(d3d.immediate->Map(staging.get(), 0, D3D11_MAP_READ, 0, &mapped))) return false;

    const uint32_t width = session.stream_info.width;
    const uint32_t height = session.stream_info.height;
    const uint32_t uv_rows = (height + 1u) / 2u;
    bool ok = mapped.pData && mapped.RowPitch >= width;
    if (ok) {
        packet.bytes.resize(static_cast<size_t>(width) * (height + uv_rows));
        const auto* source = static_cast<const uint8_t*>(mapped.pData);
        for (uint32_t row = 0; row < height; ++row)
            std::memcpy(packet.bytes.data() + static_cast<size_t>(row) * width,
                        source + static_cast<size_t>(row) * mapped.RowPitch, width);
        const auto* source_uv = source + static_cast<size_t>(mapped.RowPitch) * height;
        auto* destination_uv = packet.bytes.data() + static_cast<size_t>(width) * height;
        for (uint32_t row = 0; row < uv_rows; ++row)
            std::memcpy(destination_uv + static_cast<size_t>(row) * width,
                        source_uv + static_cast<size_t>(row) * mapped.RowPitch, width);
        packet.width = width;
        packet.height = height;
        packet.stride = width;
    }
    d3d.immediate->Unmap(staging.get(), 0);
    return ok;
}

bool copy_cpu_video_sample(IMFMediaBuffer* buffer, Session& session, VideoPacket& packet) {
    BYTE* data = nullptr;
    DWORD maximum = 0, current = 0;
    if (FAILED(buffer->Lock(&data, &maximum, &current))) return false;
    const size_t y_bytes = static_cast<size_t>(session.video_stride) * session.stream_info.height;
    const size_t uv_bytes = static_cast<size_t>(session.video_stride) *
                            ((session.stream_info.height + 1u) / 2u);
    const size_t required = y_bytes + uv_bytes;
    bool ok = current >= required;
    if (ok) {
        const uint32_t width = session.stream_info.width;
        const uint32_t height = session.stream_info.height;
        const uint32_t uv_rows = (height + 1u) / 2u;
        packet.bytes.resize(static_cast<size_t>(width) * (height + uv_rows));
        for (uint32_t row = 0; row < height; ++row)
            std::memcpy(packet.bytes.data() + static_cast<size_t>(row) * width,
                        data + static_cast<size_t>(row) * session.video_stride, width);
        const auto* source_uv = data + static_cast<size_t>(session.video_stride) * height;
        auto* destination_uv = packet.bytes.data() + static_cast<size_t>(width) * height;
        for (uint32_t row = 0; row < uv_rows; ++row)
            std::memcpy(destination_uv + static_cast<size_t>(row) * width,
                        source_uv + static_cast<size_t>(row) * session.video_stride, width);
        packet.width = width;
        packet.height = height;
        packet.stride = width;
    }
    buffer->Unlock();
    return ok;
}

bool copy_video_sample(IMFSample* sample, LONGLONG timestamp, D3DDecodeContext& d3d,
                       Session& session, VideoPacket& packet, bool& dxgi_backed) {
    ComPtr<IMFMediaBuffer> buffer;
    if (FAILED(sample->GetBufferByIndex(0, buffer.put()))) return false;
    bool copied = false;
    dxgi_backed = false;
    if (d3d.device) {
        copied = copy_dxgi_video_sample(buffer.get(), d3d, session, packet);
        dxgi_backed = copied;
    }
    if (!copied) {
        buffer.reset();
        if (FAILED(sample->ConvertToContiguousBuffer(buffer.put()))) return false;
        copied = copy_cpu_video_sample(buffer.get(), session, packet);
    }
    if (copied) packet.pts_us = timestamp > 0 ? static_cast<uint64_t>(timestamp) / 10 : 0;
    return copied;
}

bool copy_audio_sample(IMFSample* sample, LONGLONG timestamp, Session& session, AudioPacket& packet) {
    ComPtr<IMFMediaBuffer> buffer;
    if (FAILED(sample->ConvertToContiguousBuffer(buffer.put()))) return false;
    BYTE* data = nullptr;
    DWORD maximum = 0, current = 0;
    if (FAILED(buffer->Lock(&data, &maximum, &current))) return false;
    const size_t frame_bytes = static_cast<size_t>(session.stream_info.audio_channels) * sizeof(int16_t);
    const bool ok = frame_bytes != 0 && current >= frame_bytes && current % frame_bytes == 0;
    if (ok) {
        packet.pcm.resize(current / sizeof(int16_t));
        std::memcpy(packet.pcm.data(), data, current);
        packet.channels = session.stream_info.audio_channels;
        packet.sample_rate = session.stream_info.audio_rate;
        packet.pts_us = timestamp > 0 ? static_cast<uint64_t>(timestamp) / 10 : 0;
    }
    buffer->Unlock();
    return ok;
}

void finish_initialization(Session& session, bool ok, HRESULT error) {
    std::lock_guard<std::mutex> lock(session.mutex);
    session.initialized = true;
    session.init_ok = ok;
    session.init_error = error;
    if (!ok) session.decode_done = true;
    session.cv.notify_all();
}

void decode_session_com(Session& session) {
    D3DDecodeContext d3d;
    ComPtr<IMFAttributes> attributes;
    ComPtr<IMFSourceReader> reader;
    ComPtr<IMFTransform> video_decoder;
    GUID native_video_subtype{};
    bool hardware_url = false;
    bool d3d11_aware = false;
    HRESULT hr = MFCreateAttributes(attributes.put(), 3);
    if (SUCCEEDED(hr)) hr = attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
    const bool d3d_ready = SUCCEEDED(hr) && create_d3d_decode_context(attributes.get(), d3d);
    const std::wstring wide_path = utf8_to_wide(session.path);
    if (SUCCEEDED(hr) && wide_path.empty()) hr = E_INVALIDARG;
    if (SUCCEEDED(hr)) hr = MFCreateSourceReaderFromURL(wide_path.c_str(), attributes.get(), reader.put());
    if (SUCCEEDED(hr) && !discover_streams(reader.get(), session, native_video_subtype)) hr = MF_E_INVALIDMEDIATYPE;
    if (SUCCEEDED(hr) && !configure_video(reader.get(), session)) hr = MF_E_INVALIDMEDIATYPE;
    if (SUCCEEDED(hr) && !configure_audio(reader.get(), session)) hr = MF_E_INVALIDMEDIATYPE;
    if (SUCCEEDED(hr)) read_presentation_duration(reader.get(), session.stream_info.duration_us);

    const bool compressed_video = !is_uncompressed_video(native_video_subtype);
    if (SUCCEEDED(hr) && compressed_video &&
        !find_video_decoder(reader.get(), session.video_stream, video_decoder,
                            hardware_url, d3d11_aware)) hr = MF_E_TOPO_CODEC_NOT_FOUND;
    if (SUCCEEDED(hr) && compressed_video && !session.allow_software &&
        (!d3d_ready || (!d3d11_aware && !hardware_url))) hr = MF_E_TOPO_CODEC_NOT_FOUND;
    if (FAILED(hr)) {
        log_hresult("open/configure", hr, session.path);
        finish_initialization(session, false, hr);
        return;
    }

    auto announce_open = [&] {
        if (avp_log()) {
            std::fprintf(stderr,
                         "[avp-mf] opened '%s': %ux%u %.3f fps, audio=%u/%u, decoder=%s\n",
                         session.path.c_str(), session.stream_info.width, session.stream_info.height,
                         session.stream_info.fps, session.stream_info.audio_channels,
                         session.stream_info.audio_rate,
                         session.hardware_video ? "hardware DXVA MFT" :
                         (compressed_video ? "software (explicit)" : "uncompressed"));
        }
        finish_initialization(session, true, S_OK);
    };
    if (!compressed_video) announce_open();

    while (true) {
        {
            std::lock_guard<std::mutex> lock(session.mutex);
            if (session.stopping) break;
        }
        DWORD actual_stream = MF_SOURCE_READER_INVALID_STREAM_INDEX;
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        ComPtr<IMFSample> sample;
        hr = reader->ReadSample(MF_SOURCE_READER_ANY_STREAM, 0, &actual_stream, &flags,
                                &timestamp, sample.put());
        if (FAILED(hr)) {
            log_hresult("ReadSample", hr, session.path);
            if (!session.initialized) {
                finish_initialization(session, false, hr);
            } else {
                std::lock_guard<std::mutex> lock(session.mutex);
                session.decode_error = true;
                session.decode_done = true;
                session.cv.notify_all();
            }
            break;
        }

        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            bool all_eof = false;
            {
                std::lock_guard<std::mutex> lock(session.mutex);
                if (actual_stream == session.video_stream) session.video_eof = true;
                if (actual_stream == session.audio_stream) session.audio_eof = true;
                all_eof = session.video_eof && (!session.stream_info.has_audio || session.audio_eof);
                if (all_eof) {
                    session.decode_done = true;
                    session.cv.notify_all();
                }
            }
            if (all_eof && !session.initialized)
                finish_initialization(session, false, MF_E_END_OF_STREAM);
            if (all_eof) break;
            continue;
        }
        if (!sample) continue;

        if (actual_stream == session.video_stream) {
            VideoPacket packet;
            bool dxgi_backed = false;
            if (!session.initialized && avp_log())
                std::fprintf(stderr, "[avp-mf] received first decoded video sample\n");
            if (!copy_video_sample(sample.get(), timestamp, d3d, session, packet, dxgi_backed)) {
                if (!session.initialized) {
                    finish_initialization(session, false, MF_E_INVALIDMEDIATYPE);
                } else {
                    std::lock_guard<std::mutex> lock(session.mutex);
                    session.decode_error = true;
                    session.decode_done = true;
                    session.cv.notify_all();
                }
                break;
            }
            if (!session.initialized) {
                const int dxva_mode = decoder_dxva_mode(video_decoder.get());
                // A decoder-owned DXGI NV12 sample with no video processor in the topology is direct
                // evidence that the D3D11-aware decoder accepted the device manager. Some inbox MFTs
                // leave CODECAPI_AVDecVideoDXVAMode at NOTPLAYING even after their first output, so
                // do not reject that stronger sample-level evidence merely because the legacy query
                // remains inconclusive.
                session.hardware_video = hardware_url || dxva_mode >= eAVDecVideoDXVAMode_MC ||
                                         (dxgi_backed && d3d_ready && d3d11_aware);
                if (avp_log())
                    std::fprintf(stderr,
                                 "[avp-mf] decoder preflight: DXVA mode=%d hardware-url=%d "
                                 "d3d11-aware=%d dxgi-manager=%d dxgi-sample=%d\n",
                                 dxva_mode, static_cast<int>(hardware_url),
                                 static_cast<int>(d3d11_aware), static_cast<int>(d3d_ready),
                                 static_cast<int>(dxgi_backed));
                if (!session.hardware_video && !session.allow_software) {
                    finish_initialization(session, false, MF_E_TOPO_CODEC_NOT_FOUND);
                    break;
                }
                announce_open();
            }
            std::unique_lock<std::mutex> lock(session.mutex);
            session.cv.wait(lock, [&] {
                return session.stopping || session.video_queue.size() < kVideoQueueCapacity;
            });
            if (session.stopping) break;
            session.video_queue.push_back(std::move(packet));
            session.cv.notify_all();
        } else if (actual_stream == session.audio_stream) {
            AudioPacket packet;
            if (!copy_audio_sample(sample.get(), timestamp, session, packet)) {
                std::lock_guard<std::mutex> lock(session.mutex);
                session.decode_error = true;
                session.decode_done = true;
                session.cv.notify_all();
                break;
            }
            std::lock_guard<std::mutex> lock(session.mutex);
            if (session.stopping) break;
            // A title may render video while routing or disabling audio elsewhere. Do not let an
            // unconsumed audio queue stall Source Reader before it can produce the next video frame.
            if (session.audio_queue.size() == kAudioQueueCapacity)
                session.audio_queue.pop_front();
            session.audio_queue.push_back(std::move(packet));
            session.cv.notify_all();
        }
    }

    if (avp_log()) std::fprintf(stderr, "[avp-mf] decode loop stopped '%s'\n", session.path.c_str());
    // Release the Source Reader pipeline while its D3D device manager is still alive. Letting the
    // declaration-order destructors drop D3D first can strand the hardware MFT during shutdown.
    video_decoder.reset();
    if (avp_log()) std::fprintf(stderr, "[avp-mf] decoder reference released '%s'\n", session.path.c_str());
    reader.reset();
    if (avp_log()) std::fprintf(stderr, "[avp-mf] source reader released '%s'\n", session.path.c_str());
    attributes.reset();
    d3d.manager.reset();
    d3d.immediate.reset();
    d3d.device.reset();
}

void decode_session(Session& session) {
    const HRESULT com_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninitialize_com = SUCCEEDED(com_hr);
    if (FAILED(com_hr) && com_hr != RPC_E_CHANGED_MODE) {
        finish_initialization(session, false, com_hr);
        return;
    }

    // Keep every Media Foundation/D3D COM object inside the helper's scope. They must be released
    // before CoUninitialize tears down the worker apartment, or hardware decoder shutdown can hang.
    decode_session_com(session);
    if (avp_log()) std::fprintf(stderr, "[avp-mf] decode worker exiting '%s'\n", session.path.c_str());
    if (uninitialize_com) CoUninitialize();
}

void stop_session(const std::shared_ptr<Session>& session) {
    if (!session) return;
    if (avp_log()) std::fprintf(stderr, "[avp-mf] stopping decode worker '%s'\n", session->path.c_str());
    {
        std::lock_guard<std::mutex> lock(session->mutex);
        session->stopping = true;
        session->cv.notify_all();
    }
    session->thread.request_stop();
    if (session->thread.joinable()) session->thread.join();
    if (avp_log()) std::fprintf(stderr, "[avp-mf] decode worker stopped '%s'\n", session->path.c_str());
}

} // namespace

struct MediaFoundationBackend::Impl {
    HRESULT startup = E_FAIL;
    std::atomic<int> next_id{1};
    std::mutex mutex;
    std::unordered_map<int, std::shared_ptr<Session>> sessions;

    Impl() : startup(MFStartup(MF_VERSION, MFSTARTUP_FULL)) {}
    ~Impl() {
        std::vector<std::shared_ptr<Session>> pending;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto& [id, session] : sessions) pending.push_back(std::move(session));
            sessions.clear();
        }
        for (const auto& session : pending) stop_session(session);
        if (SUCCEEDED(startup)) MFShutdown();
    }

    std::shared_ptr<Session> get(int id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = sessions.find(id);
        return it == sessions.end() ? nullptr : it->second;
    }
};

MediaFoundationBackend::MediaFoundationBackend() : impl_(std::make_unique<Impl>()) {
    if (FAILED(impl_->startup)) log_hresult("MFStartup", impl_->startup);
}

MediaFoundationBackend::~MediaFoundationBackend() = default;

bool MediaFoundationBackend::available() const { return impl_ && SUCCEEDED(impl_->startup); }

int MediaFoundationBackend::open(const std::string& host_path) {
    if (!available() || host_path.empty()) return -1;
    auto session = std::make_shared<Session>();
    session->path = host_path;
    session->allow_software = software_decode_allowed();
    session->thread = std::jthread([session] { decode_session(*session); });
    {
        std::unique_lock<std::mutex> lock(session->mutex);
        session->cv.wait(lock, [&] { return session->initialized; });
        if (!session->init_ok) {
            lock.unlock();
            stop_session(session);
            return -1;
        }
    }
    const int id = impl_->next_id.fetch_add(1);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->sessions.emplace(id, std::move(session));
    return id;
}

bool MediaFoundationBackend::info(int id, StreamInfo& out) {
    auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    out = session->stream_info;
    return session->init_ok;
}

bool MediaFoundationBackend::next_video(int id, VideoFrame& out) {
    auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->video_queue.empty()) return false;
    session->last_video = std::move(session->video_queue.front());
    session->video_queue.pop_front();
    session->cv.notify_all();
    const size_t y_bytes = static_cast<size_t>(session->last_video.stride) *
                           session->last_video.height;
    out.y = session->last_video.bytes.data();
    out.uv = session->last_video.bytes.data() + y_bytes;
    out.width = session->last_video.width;
    out.height = session->last_video.height;
    out.y_stride = session->last_video.stride;
    out.uv_stride = session->last_video.stride;
    out.pts_us = session->last_video.pts_us;
    return true;
}

bool MediaFoundationBackend::next_audio(int id, AudioFrame& out) {
    auto session = impl_->get(id);
    if (!session) return false;
    std::lock_guard<std::mutex> lock(session->mutex);
    if (session->audio_queue.empty()) return false;
    session->last_audio = std::move(session->audio_queue.front());
    session->audio_queue.pop_front();
    session->cv.notify_all();
    out.pcm = session->last_audio.pcm.data();
    out.channels = session->last_audio.channels;
    out.samples = static_cast<uint32_t>(session->last_audio.pcm.size() /
                                        session->last_audio.channels);
    out.sample_rate = session->last_audio.sample_rate;
    out.pts_us = session->last_audio.pts_us;
    return true;
}

bool MediaFoundationBackend::eof(int id) {
    auto session = impl_->get(id);
    if (!session) return true;
    std::lock_guard<std::mutex> lock(session->mutex);
    // The guest may route or disable audio without ever pulling it from AvPlayer. The bounded
    // audio queue intentionally does not stall decode, so it must not stall playback completion
    // after the final video frame either. Queued/last audio storage remains owned by the session
    // until close, preserving successful-pull pointer lifetime.
    return session->decode_done && session->video_queue.empty();
}

void MediaFoundationBackend::close(int id) {
    std::shared_ptr<Session> session;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        auto it = impl_->sessions.find(id);
        if (it == impl_->sessions.end()) return;
        session = std::move(it->second);
        impl_->sessions.erase(it);
    }
    stop_session(session);
}

MediaFoundationBackend& shared_media_foundation_backend() {
    static MediaFoundationBackend backend;
    return backend;
}

bool install_media_foundation_backend() {
    auto& media_foundation = shared_media_foundation_backend();
    if (!media_foundation.available()) return false;
    set_backend(&media_foundation);
    return true;
}

void uninstall_media_foundation_backend() {
    if (backend() == &shared_media_foundation_backend()) set_backend(nullptr);
}

} // namespace prosper::video
