#include "mf_access_unit_decoder.hpp"

#include "mf_com_ptr.hpp"
#include "vp9_uncompressed_header.hpp"

#ifndef _WIN32
#error "The Media Foundation access-unit decoder is Windows-only"
#endif

#include <windows.h>
#include <combaseapi.h>
#include <d3d10.h>
#include <d3d11.h>
#include <codecapi.h>
#include <icodecapi.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mftransform.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace prosper::video::mf_au {
namespace {

using AuResult = VideoBackend::AuResult;
using AuPicture = VideoBackend::AuPicture;

// PROSPER_VDEC2_MF_SOFTWARE=1 (default off): do not hand the decoder a D3D11 device, so it decodes
// into system memory. The test uses it to cover the CPU-buffer copy path on a host whose decoder
// would otherwise take DXVA, the same way the VA-API test forces its software pass.
bool software_forced() { return std::getenv("PROSPER_VDEC2_MF_SOFTWARE") != nullptr; }

// Every Media Foundation / D3D11 call for one decoder runs on that decoder's own thread.
//
// Not for speed, and not because the objects are apartment-bound (they are free-threaded). Because
// decode_au is called from a GUEST thread: its stack is the guest's, sized by the title, and the
// system decoders are opaque code that may use as much of it as they like and may raise and catch
// their own exceptions on it. A dedicated host thread gives them an ordinary host stack and an
// ordinary MTA, and costs one handoff per access unit. The AvPlayer path makes the same choice for
// the same reason: its Source Reader runs on its own worker too.
class ComWorker {
public:
    ComWorker() : thread_([this] { loop(); }) {}
    ~ComWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            cv_.notify_all();
        }
        if (thread_.joinable()) thread_.join();
    }
    ComWorker(const ComWorker&) = delete;
    ComWorker& operator=(const ComWorker&) = delete;

    // Runs `job` on the worker and returns when it has finished. Not re-entrant; the caller
    // serialises (Decoder::call_mutex).
    void run(std::function<void()> job) {
        std::unique_lock<std::mutex> lock(mutex_);
        job_ = std::move(job);
        pending_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return !pending_; });
    }

private:
    void loop() {
        const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        // MFStartup is reference counted, so this worker holds its own reference for exactly as
        // long as it can touch a Media Foundation object -- independent of whether the AvPlayer
        // backend's reference is still alive during shutdown.
        const HRESULT mf = MFStartup(MF_VERSION, MFSTARTUP_LITE);
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return pending_ || stop_; });
                if (!pending_) break;
                job = std::move(job_);
            }
            job();
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = nullptr;
            pending_ = false;
            cv_.notify_all();
        }
        if (SUCCEEDED(mf)) MFShutdown();
        if (SUCCEEDED(com)) CoUninitialize();
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::function<void()> job_;
    bool pending_ = false;
    bool stop_ = false;
    std::thread thread_;
};

// One decoded picture, already packed as the guest wants it: w*h luma, then ceil(h/2) rows of
// 2*ceil(w/2) interleaved chroma bytes -- exactly nv12_bytes(w, h).
struct Picture {
    std::vector<uint8_t> nv12;
    uint32_t width = 0, height = 0;
    bool hardware = false;
};

// A Media Foundation decoder may hold several pictures and release them together, or refuse input
// until its output is drained. The guest contract is at most one picture per access unit, so the
// surplus waits here and goes out on the following calls. Bounded so a decoder that out-produces
// its input (it should not) cannot grow without limit; the overflow is reported, not hidden.
constexpr size_t kMaxPendingPictures = 16;

struct Decoder;
void release_all(Decoder& d);

struct Decoder {
    std::mutex call_mutex;   // one decode/reset/close at a time per decoder
    uint32_t codec = 0;
    GUID subtype{};
    const char* name = "";

    // ---- everything below is touched only on `worker` ----
    ComPtr<IMFTransform> mft;
    std::wstring mft_name;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> immediate;
    ComPtr<IMFDXGIDeviceManager> manager;
    ComPtr<ID3D11Texture2D> staging;
    D3D11_TEXTURE2D_DESC staging_desc{};
    bool d3d = false;
    bool configured = false;      // input + output types set, streaming started
    bool checked_first_au = false;
    bool described_output = false;
    // Current output type.
    uint32_t coded_w = 0, coded_h = 0;           // allocated surface (MF_MT_FRAME_SIZE)
    uint32_t show_x = 0, show_y = 0, show_w = 0, show_h = 0;   // display aperture inside it
    int32_t default_stride = 0;
    bool mft_provides_samples = false;
    DWORD output_bytes = 0;
    ComPtr<IMFSample> output_sample;             // caller-allocated output, until it is filled
    LONGLONG next_time = 0;
    std::deque<Picture> pending;
    uint64_t pictures_dropped = 0;

    ComWorker worker;   // LAST: destroyed first, so its thread is joined before the rest goes

    // A decoder never closed (process exit with a decoder live) still releases its COM objects on
    // its own worker, before that worker leaves its apartment.
    ~Decoder() { worker.run([this] { release_all(*this); }); }
};

std::mutex g_registry_mutex;
std::map<int, std::shared_ptr<Decoder>> g_decoders;
int g_next_id = 1;

std::shared_ptr<Decoder> find(int id) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_decoders.find(id);
    return it == g_decoders.end() ? nullptr : it->second;
}

void release_all(Decoder& d) {
    d.pending.clear();
    d.output_sample.reset();
    d.staging.reset();
    d.mft.reset();          // before the device manager it was handed
    d.manager.reset();
    d.immediate.reset();
    d.device.reset();
}

bool create_d3d(Decoder& d) {
    D3D_FEATURE_LEVEL level{};
    const UINT flags = D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, nullptr, 0,
                                 D3D11_SDK_VERSION, d.device.put(), &level, d.immediate.put())))
        return false;
    // The decoder uses the device on its own threads while this worker copies surfaces through the
    // immediate context; Microsoft requires multithread protection for that sharing.
    ComPtr<ID3D10Multithread> multithread;
    if (FAILED(d.device->QueryInterface(IID_ID3D10Multithread,
                                        reinterpret_cast<void**>(multithread.put()))))
        return false;
    multithread->SetMultithreadProtected(TRUE);
    UINT token = 0;
    if (FAILED(MFCreateDXGIDeviceManager(&token, d.manager.put()))) return false;
    if (FAILED(d.manager->ResetDevice(d.device.get(), token))) return false;
    return true;
}

// Find and instantiate a synchronous decoder MFT for the codec. Asynchronous MFTs (the hardware
// vendor ones) use an event-driven protocol this driver does not implement; the inbox and Store
// decoders are synchronous and do DXVA themselves when handed a device manager, so that is not a
// loss of hardware decode on the hosts measured (see the PR for #2983).
bool create_mft(Decoder& d) {
    MFT_REGISTER_TYPE_INFO input{MFMediaType_Video, d.subtype};
    IMFActivate** activates = nullptr;
    UINT32 count = 0;
    const HRESULT hr = MFTEnumEx(MFT_CATEGORY_VIDEO_DECODER,
                                 MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT |
                                     MFT_ENUM_FLAG_SORTANDFILTER,
                                 &input, nullptr, &activates, &count);
    bool ok = false;
    for (UINT32 i = 0; SUCCEEDED(hr) && i < count; ++i) {
        if (!ok) {
            ComPtr<IMFTransform> mft;
            if (SUCCEEDED(activates[i]->ActivateObject(IID_IMFTransform,
                                                       reinterpret_cast<void**>(mft.put())))) {
                WCHAR* friendly = nullptr;
                UINT32 length = 0;
                if (SUCCEEDED(activates[i]->GetAllocatedString(MFT_FRIENDLY_NAME_Attribute,
                                                               &friendly, &length)) && friendly) {
                    d.mft_name = friendly;
                    CoTaskMemFree(friendly);
                }
                d.mft = std::move(mft);
                ok = true;
            }
        }
        activates[i]->Release();
    }
    CoTaskMemFree(activates);
    if (!ok) {
        std::fprintf(stderr,
                     "[vdec2] no Media Foundation %s decoder is installed on this host (MFTEnumEx "
                     "hr=0x%08lx, %u candidates)%s -- refusing the codec (#2983)\n",
                     d.name, static_cast<unsigned long>(hr), count,
                     IsEqualGUID(d.subtype, MFVideoFormat_VP90)
                         ? "; Windows supplies VP9 through the 'VP9 Video Extensions' package"
                         : "");
    }
    return ok;
}

// H.264 only: ask the decoder to output each picture as soon as the stream's own reordering allows,
// instead of waiting for its decoded-picture buffer to fill. Without it the inbox decoder holds up
// to a level-derived DPB's worth of pictures before the first comes out -- measured: 12 access units
// of the committed 128x96 stream produced ZERO pictures -- and since sceVideodec2Flush does not
// drain (#2562), those pictures would never reach the guest at all.
void request_low_latency(Decoder& d) {
    if (!IsEqualGUID(d.subtype, MFVideoFormat_H264)) return;
    ComPtr<ICodecAPI> codec;
    if (FAILED(d.mft->QueryInterface(IID_ICodecAPI, reinterpret_cast<void**>(codec.put())))) return;
    VARIANT value;
    VariantInit(&value);
    value.vt = VT_UI4;
    value.ulVal = TRUE;
    const HRESULT hr = codec->SetValue(&CODECAPI_AVLowLatencyMode, &value);
    if (FAILED(hr))
        std::fprintf(stderr, "[vdec2] Media Foundation H.264 decoder refused low-latency mode "
                             "(hr=0x%08lx); pictures may be held back (#2983)\n",
                     static_cast<unsigned long>(hr));
}

bool attach_d3d(Decoder& d) {
    if (software_forced()) return false;
    ComPtr<IMFAttributes> attributes;
    UINT32 aware = FALSE;
    if (FAILED(d.mft->GetAttributes(attributes.put())) ||
        FAILED(attributes->GetUINT32(MF_SA_D3D11_AWARE, &aware)) || !aware)
        return false;
    if (!create_d3d(d)) { d.manager.reset(); d.immediate.reset(); d.device.reset(); return false; }
    if (FAILED(d.mft->ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER,
                                     reinterpret_cast<ULONG_PTR>(d.manager.get())))) {
        d.manager.reset(); d.immediate.reset(); d.device.reset();
        return false;
    }
    return true;
}

// Pick the NV12 output type and record its geometry. Called at configuration and again whenever the
// decoder announces MF_E_TRANSFORM_STREAM_CHANGE (the first key frame usually does, once it knows
// the real size).
bool negotiate_output(Decoder& d) {
    for (DWORD index = 0;; ++index) {
        ComPtr<IMFMediaType> type;
        const HRESULT hr = d.mft->GetOutputAvailableType(0, index, type.put());
        if (FAILED(hr)) break;
        GUID subtype{};
        if (FAILED(type->GetGUID(MF_MT_SUBTYPE, &subtype)) ||
            !IsEqualGUID(subtype, MFVideoFormat_NV12)) continue;
        if (FAILED(d.mft->SetOutputType(0, type.get(), 0))) continue;

        ComPtr<IMFMediaType> current;
        if (FAILED(d.mft->GetOutputCurrentType(0, current.put()))) return false;
        UINT32 w = 0, h = 0;
        if (FAILED(MFGetAttributeSize(current.get(), MF_MT_FRAME_SIZE, &w, &h))) w = h = 0;
        d.coded_w = w; d.coded_h = h;
        d.show_x = d.show_y = 0; d.show_w = w; d.show_h = h;
        // The picture the guest gets is the DISPLAY aperture, not the allocated surface: a decoder
        // pads its surfaces to its block alignment (a 160x90 VP9 stream comes back in a taller
        // surface), and libavcodec -- the Linux backend's reference -- reports the cropped size.
        MFVideoArea area{};
        UINT32 area_bytes = 0;
        if (SUCCEEDED(current->GetBlob(MF_MT_MINIMUM_DISPLAY_APERTURE,
                                       reinterpret_cast<UINT8*>(&area), sizeof(area),
                                       &area_bytes)) && area_bytes == sizeof(area) &&
            area.Area.cx > 0 && area.Area.cy > 0 && area.OffsetX.value >= 0 &&
            area.OffsetY.value >= 0 &&
            static_cast<uint64_t>(area.OffsetX.value) + area.Area.cx <= w &&
            static_cast<uint64_t>(area.OffsetY.value) + area.Area.cy <= h) {
            d.show_x = static_cast<uint32_t>(area.OffsetX.value);
            d.show_y = static_cast<uint32_t>(area.OffsetY.value);
            d.show_w = static_cast<uint32_t>(area.Area.cx);
            d.show_h = static_cast<uint32_t>(area.Area.cy);
        }
        UINT32 stride = 0;
        d.default_stride = SUCCEEDED(current->GetUINT32(MF_MT_DEFAULT_STRIDE, &stride))
                               ? static_cast<int32_t>(stride) : static_cast<int32_t>(w);

        MFT_OUTPUT_STREAM_INFO info{};
        if (FAILED(d.mft->GetOutputStreamInfo(0, &info))) return false;
        d.mft_provides_samples = (info.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES |
                                                  MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES)) != 0;
        d.output_bytes = info.cbSize;
        d.output_sample.reset();
        d.described_output = false;
        return true;
    }
    return false;
}

bool configure(Decoder& d, const uint8_t* au, size_t bytes) {
    ComPtr<IMFMediaType> input;
    if (FAILED(MFCreateMediaType(input.put())) ||
        FAILED(input->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)) ||
        FAILED(input->SetGUID(MF_MT_SUBTYPE, d.subtype)) ||
        FAILED(input->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)))
        return false;
    // VP9 has no in-band parameter sets for a decoder to discover its size from before it is
    // configured, so hand it the size the first key frame declares. H.264 carries its SPS in-band
    // and the decoder reads it itself.
    uint32_t w = 0, h = 0;
    if (IsEqualGUID(d.subtype, MFVideoFormat_VP90) && vp9_key_frame_size(au, bytes, w, h))
        MFSetAttributeSize(input.get(), MF_MT_FRAME_SIZE, w, h);
    HRESULT hr = d.mft->SetInputType(0, input.get(), 0);
    if (FAILED(hr)) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 4)
            std::fprintf(stderr, "[vdec2] Media Foundation %s decoder refused its input type "
                                 "(hr=0x%08lx, size %ux%u); no picture yet (#2983)\n",
                         d.name, static_cast<unsigned long>(hr), w, h);
        return false;
    }
    if (!negotiate_output(d)) {
        std::fprintf(stderr, "[vdec2] Media Foundation %s decoder offers no NV12 output (#2983)\n",
                     d.name);
        return false;
    }
    d.mft->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
    d.mft->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
    d.configured = true;
    return true;
}

// Pack the display aperture of one NV12 image into `out`. `luma` and `chroma` point at the top-left
// of the full surface planes, each `pitch` bytes per row.
void pack_nv12(const uint8_t* luma, const uint8_t* chroma, size_t pitch, const Decoder& d,
               Picture& out) {
    const uint32_t w = d.show_w, h = d.show_h;
    out.width = w;
    out.height = h;
    out.nv12.resize(static_cast<size_t>(nv12_bytes(w, h)));
    for (uint32_t row = 0; row < h; ++row)
        std::memcpy(out.nv12.data() + static_cast<size_t>(row) * w,
                    luma + (static_cast<size_t>(d.show_y) + row) * pitch + d.show_x, w);
    const uint32_t chroma_rows = (h + 1) / 2, chroma_bytes = 2 * ((w + 1) / 2);
    uint8_t* uv = out.nv12.data() + static_cast<size_t>(w) * h;
    // NV12 chroma is subsampled 2x2, so an aperture offset maps to half that offset in chroma rows
    // and to the same even byte offset in a row. Decoders report offset 0 in practice.
    const size_t cx = (d.show_x / 2) * 2, cy = d.show_y / 2;
    for (uint32_t row = 0; row < chroma_rows; ++row)
        std::memcpy(uv + static_cast<size_t>(row) * chroma_bytes,
                    chroma + (cy + row) * pitch + cx, chroma_bytes);
}

bool copy_dxgi(Decoder& d, IMFMediaBuffer* buffer, Picture& out) {
    ComPtr<IMFDXGIBuffer> dxgi;
    if (FAILED(buffer->QueryInterface(IID_IMFDXGIBuffer, reinterpret_cast<void**>(dxgi.put()))))
        return false;
    ComPtr<ID3D11Texture2D> texture;
    UINT subresource = 0;
    if (FAILED(dxgi->GetResource(IID_ID3D11Texture2D, reinterpret_cast<void**>(texture.put()))) ||
        FAILED(dxgi->GetSubresourceIndex(&subresource)))
        return false;
    D3D11_TEXTURE2D_DESC desc{};
    texture->GetDesc(&desc);
    if (desc.Format != DXGI_FORMAT_NV12 || desc.Width < d.show_x + d.show_w ||
        desc.Height < d.show_y + d.show_h)
        return false;
    if (!d.staging || d.staging_desc.Width != desc.Width || d.staging_desc.Height != desc.Height) {
        D3D11_TEXTURE2D_DESC staging = desc;
        staging.MipLevels = 1;
        staging.ArraySize = 1;
        staging.SampleDesc.Count = 1;
        staging.SampleDesc.Quality = 0;
        staging.Usage = D3D11_USAGE_STAGING;
        staging.BindFlags = 0;
        staging.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        staging.MiscFlags = 0;
        d.staging.reset();
        if (FAILED(d.device->CreateTexture2D(&staging, nullptr, d.staging.put()))) return false;
        d.staging_desc = staging;
    }
    d.immediate->CopySubresourceRegion(d.staging.get(), 0, 0, 0, 0, texture.get(), subresource,
                                       nullptr);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    if (FAILED(d.immediate->Map(d.staging.get(), 0, D3D11_MAP_READ, 0, &mapped))) return false;
    bool ok = mapped.pData && mapped.RowPitch >= desc.Width;
    if (ok) {
        const auto* base = static_cast<const uint8_t*>(mapped.pData);
        // A mapped NV12 texture's chroma plane starts after the TEXTURE's full height of luma rows.
        pack_nv12(base, base + static_cast<size_t>(mapped.RowPitch) * desc.Height, mapped.RowPitch,
                  d, out);
    }
    d.immediate->Unmap(d.staging.get(), 0);
    return ok;
}

bool copy_system_memory(Decoder& d, IMFSample* sample, Picture& out) {
    ComPtr<IMFMediaBuffer> buffer;
    if (FAILED(sample->ConvertToContiguousBuffer(buffer.put()))) return false;
    // Prefer the 2D lock: it states the real pitch, which a decoder may pad beyond the default
    // stride the media type advertises.
    ComPtr<IMF2DBuffer> buffer2d;
    if (SUCCEEDED(buffer->QueryInterface(IID_IMF2DBuffer,
                                         reinterpret_cast<void**>(buffer2d.put())))) {
        BYTE* scan0 = nullptr;
        LONG pitch = 0;
        if (SUCCEEDED(buffer2d->Lock2D(&scan0, &pitch))) {
            const bool ok = scan0 && pitch > 0 && static_cast<uint32_t>(pitch) >= d.coded_w;
            if (ok) pack_nv12(scan0, scan0 + static_cast<size_t>(pitch) * d.coded_h,
                              static_cast<size_t>(pitch), d, out);
            buffer2d->Unlock2D();
            return ok;
        }
    }
    BYTE* data = nullptr;
    DWORD maximum = 0, current = 0;
    if (FAILED(buffer->Lock(&data, &maximum, &current))) return false;
    const int32_t stride = d.default_stride;
    const uint64_t required = stride > 0
        ? static_cast<uint64_t>(stride) * (d.coded_h + (d.coded_h + 1) / 2) : 0;
    const bool ok = data && stride > 0 && static_cast<uint32_t>(stride) >= d.coded_w &&
                    current >= required;
    if (ok) pack_nv12(data, data + static_cast<size_t>(stride) * d.coded_h,
                      static_cast<size_t>(stride), d, out);
    buffer->Unlock();
    return ok;
}

bool copy_picture(Decoder& d, IMFSample* sample, Picture& out) {
    ComPtr<IMFMediaBuffer> first;
    if (FAILED(sample->GetBufferByIndex(0, first.put()))) return false;
    // THE OUTCOME, not the request (#2586). A picture counts as hardware only when it came back as a
    // decoder-owned D3D11 surface; handing over a device manager merely asked for that.
    // CONFIDENCE: MED that a D3D11 surface implies DXVA decode -- it is the same evidence the
    // AvPlayer path accepts (media_foundation_backend.cpp, "dxgi-sample"), and a decoder that
    // decoded in software and then uploaded would defeat it.
    if (d.device && copy_dxgi(d, first.get(), out)) { out.hardware = true; return true; }
    out.hardware = false;
    return copy_system_memory(d, sample, out);
}

void describe_output_once(Decoder& d, const Picture& pic) {
    if (d.described_output) return;
    d.described_output = true;
    std::fprintf(stderr,
                 "[vdec2] Media Foundation %s output: surface %ux%u, picture %ux%u at (%u,%u), "
                 "%s (#2983)\n",
                 d.name, d.coded_w, d.coded_h, pic.width, pic.height, d.show_x, d.show_y,
                 pic.hardware ? "D3D11 surface (DXVA)" : "system memory (software)");
}

// Pull everything the decoder is ready to hand back into `pending`.
void drain(Decoder& d) {
    for (int guard = 0; guard < 64; ++guard) {
        MFT_OUTPUT_DATA_BUFFER output{};
        output.dwStreamID = 0;
        if (!d.mft_provides_samples) {
            if (!d.output_sample) {
                ComPtr<IMFSample> sample;
                ComPtr<IMFMediaBuffer> buffer;
                const DWORD size = d.output_bytes
                    ? d.output_bytes : static_cast<DWORD>(nv12_bytes(d.coded_w, d.coded_h));
                if (FAILED(MFCreateSample(sample.put())) ||
                    FAILED(MFCreateMemoryBuffer(size, buffer.put())) ||
                    FAILED(sample->AddBuffer(buffer.get())))
                    return;
                d.output_sample = std::move(sample);
            }
            output.pSample = d.output_sample.get();
        }
        DWORD status = 0;
        const HRESULT hr = d.mft->ProcessOutput(0, 1, &output, &status);
        if (output.pEvents) output.pEvents->Release();
        ComPtr<IMFSample> produced;
        if (d.mft_provides_samples) produced.reset(output.pSample);   // we own the MFT's sample
        if (hr == MF_E_TRANSFORM_NEED_MORE_INPUT) return;
        if (hr == MF_E_TRANSFORM_STREAM_CHANGE) {
            if (!negotiate_output(d)) {
                std::fprintf(stderr, "[vdec2] Media Foundation %s decoder changed format and "
                                     "offers no NV12 output (#2983)\n", d.name);
                return;
            }
            continue;
        }
        if (FAILED(hr)) {
            static std::atomic<int> warned{0};
            if (warned.fetch_add(1) < 8)
                std::fprintf(stderr, "[vdec2] Media Foundation %s ProcessOutput failed: "
                                     "hr=0x%08lx (#2983)\n",
                             d.name, static_cast<unsigned long>(hr));
            return;
        }
        IMFSample* sample = d.mft_provides_samples ? produced.get() : d.output_sample.get();
        if (!sample) continue;
        Picture pic;
        if (!copy_picture(d, sample, pic) || !pic.width || !pic.height) {
            static std::atomic<int> warned{0};
            if (warned.fetch_add(1) < 8)
                std::fprintf(stderr, "[vdec2] Media Foundation %s produced a sample prosper could "
                                     "not read as NV12 -- no picture (#2983)\n", d.name);
            continue;
        }
        describe_output_once(d, pic);
        // Never hand a filled sample back. The inbox H.264 decoder answers E_FAIL to every
        // ProcessOutput after the first when it is given the sample it already filled -- measured on
        // the software path: 1 picture from 12 access units reusing it, 12 of 12 with a fresh one.
        d.output_sample.reset();
        if (d.pending.size() >= kMaxPendingPictures) {
            d.pending.pop_front();
            if (d.pictures_dropped++ == 0)
                std::fprintf(stderr, "[vdec2] Media Foundation %s decoder produced more pictures "
                                     "than access units; dropping the oldest (#2983)\n", d.name);
        }
        d.pending.push_back(std::move(pic));
    }
}

bool submit(Decoder& d, const uint8_t* au, size_t bytes) {
    ComPtr<IMFSample> sample;
    ComPtr<IMFMediaBuffer> buffer;
    if (FAILED(MFCreateSample(sample.put())) ||
        FAILED(MFCreateMemoryBuffer(static_cast<DWORD>(bytes), buffer.put())))
        return false;
    BYTE* data = nullptr;
    if (FAILED(buffer->Lock(&data, nullptr, nullptr))) return false;
    std::memcpy(data, au, bytes);
    buffer->Unlock();
    buffer->SetCurrentLength(static_cast<DWORD>(bytes));
    sample->AddBuffer(buffer.get());
    // Synthetic, monotonic, never shown to the guest: Videodec2 carries its own PTS through the
    // HLE, and some decoders want every input sample to carry a time.
    constexpr LONGLONG kTick = 333333;   // 100 ns units; the value is irrelevant, the order is not
    sample->SetSampleTime(d.next_time);
    sample->SetSampleDuration(kTick);
    d.next_time += kTick;

    HRESULT hr = d.mft->ProcessInput(0, sample.get(), 0);
    if (hr == MF_E_NOTACCEPTING) {
        drain(d);
        hr = d.mft->ProcessInput(0, sample.get(), 0);
    }
    static std::atomic<unsigned> sends{0}, failures{0};
    const unsigned n = ++sends;
    if (FAILED(hr)) ++failures;
    if (n <= 4 || n % 64 == 0 || (FAILED(hr) && failures.load() <= 8))
        std::fprintf(stderr, "[vdec2] mf au#%u bytes=%zu ProcessInput=0x%08lx | "
                             "input_failures=%u\n",
                     n, bytes, static_cast<unsigned long>(hr), failures.load());
    return SUCCEEDED(hr);
}

} // namespace

int open_decoder(uint32_t codec) {
    // The codec VALUES are the ones the Linux backend identified, and the evidence for each is
    // written down there (VaapiBackend::open_decoder): 1 is AVC/H.264 (profile_idc 100 on two
    // titles), 2382845 is VP9 (read off Sonic Racing: CrossWorlds' key-frame sync code).
    // CONFIDENCE: MED on 2382845 meaning VP9 in general -- one title; decode_au warns if a stream
    // opened as VP9 does not look like one.
    auto d = std::make_shared<Decoder>();
    d->codec = codec;
    if (codec == 1)            { d->subtype = MFVideoFormat_H264; d->name = "H.264"; }
    else if (codec == 2382845) { d->subtype = MFVideoFormat_VP90; d->name = "VP9"; }
    else {
        std::fprintf(stderr, "[vdec2] open_decoder: codec=%u is not a bitstream format we have "
                             "identified (1=AVC/H.264, 2382845=VP9); refusing rather than guessing "
                             "(#2270)\n", codec);
        return -1;
    }
    bool ok = false;
    d->worker.run([&] {
        ok = create_mft(*d);
        if (ok) request_low_latency(*d);
        if (ok) d->d3d = attach_d3d(*d);
        if (!ok) release_all(*d);
    });
    if (!ok) return -1;
    int id;
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        id = g_next_id++;
        g_decoders[id] = d;
    }
    // REQUESTED, not "using": whether DXVA actually decodes is known only from the first picture's
    // surface (AuPicture::hardware), which describe_output_once reports.
    std::fprintf(stderr, "[vdec2] access-unit %s decoder opened (id=%d, Media Foundation '%ls', "
                         "DXVA %s)\n",
                 d->name, id, d->mft_name.empty() ? L"?" : d->mft_name.c_str(),
                 d->d3d ? "requested" : (software_forced() ? "disabled by PROSPER_VDEC2_MF_SOFTWARE"
                                                           : "unavailable -- software"));
    return id;
}

AuResult decode_au(int id, const uint8_t* au, size_t bytes, uint8_t* dst, uint64_t dst_bytes,
                   AuPicture& out) {
    auto d = find(id);
    if (!d || !au || !bytes) return AuResult::NoPicture;
    std::lock_guard<std::mutex> call(d->call_mutex);
    AuResult result = AuResult::NoPicture;
    d->worker.run([&] {
        if (!d->mft) return;
        // Same once-per-decoder falsification check as the Linux backend: a stream opened as VP9
        // whose first unit carries no VP9 frame marker / key-frame sync code means the codec
        // mapping is wrong for this title, and that must be loud rather than look like a decoder
        // quietly producing nothing. Warns only; refusing here would turn a diagnostic into a
        // regression.
        if (!d->checked_first_au) {
            d->checked_first_au = true;
            if (IsEqualGUID(d->subtype, MFVideoFormat_VP90) && bytes >= 4) {
                const bool marker = (au[0] & 0xC0) == 0x80;
                const bool sync = au[1] == 0x49 && au[2] == 0x83 && au[3] == 0x42;
                const bool key = (au[0] & 0x04) == 0;
                if (!marker || (key && !sync))
                    std::fprintf(stderr,
                                 "[vdec2] WARNING: opened as VP9 but the first access unit does "
                                 "not look like VP9 (head=%02x %02x %02x %02x). The codec->format "
                                 "mapping may be wrong for this title (#2270, #2983).\n",
                                 au[0], au[1], au[2], au[3]);
            }
        }
        if (!d->configured && !configure(*d, au, bytes)) return;
        if (submit(*d, au, bytes)) drain(*d);
        if (d->pending.empty()) return;
        Picture pic = std::move(d->pending.front());
        d->pending.pop_front();
        out.width = pic.width;
        out.height = pic.height;
        out.y_stride = pic.width;
        out.uv_stride = pic.width;
        out.nv12_bytes = nv12_bytes(pic.width, pic.height);
        out.hardware = pic.hardware;
        if (!dst || dst_bytes < out.nv12_bytes || pic.nv12.size() != out.nv12_bytes) {
            result = AuResult::FrameTooSmall;
            return;
        }
        // Straight into the caller's buffer, on the worker, while decode_au still holds this
        // decoder's call lock -- so a concurrent close_decoder cannot free anything mid-copy.
        std::memcpy(dst, pic.nv12.data(), pic.nv12.size());
        result = AuResult::Decoded;
    });
    return result;
}

// sceVideodec2Reset: discard buffered pictures and references, keep the decoder (#2585).
// MFT_MESSAGE_COMMAND_FLUSH is Media Foundation's discard: the MFT drops every input it holds and
// every output it has not delivered, and stays configured with its current media types. The
// test's reset arm checks what that means for decoding after it.
bool reset_decoder(int id) {
    auto d = find(id);
    if (!d) return false;
    std::lock_guard<std::mutex> call(d->call_mutex);
    bool ok = false;
    d->worker.run([&] {
        if (!d->mft) return;
        d->pending.clear();
        ok = !d->configured || SUCCEEDED(d->mft->ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0));
    });
    return ok;
}

void close_decoder(int id) {
    std::shared_ptr<Decoder> d;
    {
        std::lock_guard<std::mutex> lock(g_registry_mutex);
        auto it = g_decoders.find(id);
        if (it == g_decoders.end()) return;
        d = std::move(it->second);
        g_decoders.erase(it);
    }
    {
        // Waits out a decode_au already in flight on this decoder.
        std::lock_guard<std::mutex> call(d->call_mutex);
        // COM objects are released on the thread that created them, before it leaves its MTA.
        d->worker.run([&] { release_all(*d); });
    }
}

} // namespace prosper::video::mf_au
