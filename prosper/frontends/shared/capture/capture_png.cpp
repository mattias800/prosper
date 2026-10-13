#include "shared/capture/capture_png.hpp"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <wincodec.h>
#include <objbase.h>
#else
#include <zlib.h>
#endif
namespace prosper::screenshot {
uint32_t crc32_bytes(const void* data, size_t len, uint32_t seed) {
    uint32_t crc = ~seed;
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

void put_chunk(FILE* f, const char* type, const uint8_t* data, size_t len) {
    uint8_t hdr[4] = {(uint8_t)(len >> 24), (uint8_t)(len >> 16), (uint8_t)(len >> 8),
                      (uint8_t)len};
    fwrite(hdr, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (len) fwrite(data, 1, len, f);
    uint32_t crc = crc32_bytes(type, 4);
    if (len) crc = crc32_bytes(data, len, crc);
    uint8_t crcb[4] = {(uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8),
                       (uint8_t)crc};
    fwrite(crcb, 1, 4, f);
}

// Write w*h RGBA8 as an 8-bit RGBA PNG.
bool write_png(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h, std::string& error) {
#ifdef _WIN32
    const HRESULT init_hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninitialize = SUCCEEDED(init_hr);
    if (FAILED(init_hr) && init_hr != RPC_E_CHANGED_MODE) {
        char detail[64];
        snprintf(detail, sizeof detail, "COM init failed (0x%08lx)", (unsigned long)init_hr);
        error = detail;
        return false;
    }

    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapEncoder* encoder = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* properties = nullptr;
    HRESULT hr = S_OK;
    bool ok = false;
    do {
        hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                              IID_PPV_ARGS(&factory));
        if (FAILED(hr)) break;
        hr = factory->CreateStream(&stream);
        if (FAILED(hr)) break;
        const std::wstring wide_path = std::filesystem::path(path).wstring();
        hr = stream->InitializeFromFilename(wide_path.c_str(), GENERIC_WRITE);
        if (FAILED(hr)) break;
        hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
        if (FAILED(hr)) break;
        hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
        if (FAILED(hr)) break;
        hr = encoder->CreateNewFrame(&frame, &properties);
        if (FAILED(hr)) break;
        hr = frame->Initialize(properties);
        if (FAILED(hr)) break;
        hr = frame->SetSize(w, h);
        if (FAILED(hr)) break;
        WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
        hr = frame->SetPixelFormat(&format);
        if (FAILED(hr) || !IsEqualGUID(format, GUID_WICPixelFormat32bppBGRA)) {
            if (SUCCEEDED(hr)) hr = WINCODEC_ERR_UNSUPPORTEDPIXELFORMAT;
            break;
        }
        if (w == 0 || h == 0 || w > UINT32_MAX / 4) {
            hr = E_INVALIDARG;
            break;
        }
        const UINT stride = w * 4;
        if (h > UINT32_MAX / stride) {
            hr = E_INVALIDARG;
            break;
        }
        const UINT byte_count = stride * h;
        std::vector<uint8_t> bgra(byte_count);
        for (size_t i = 0; i < byte_count; i += 4) {
            bgra[i + 0] = rgba[i + 2];
            bgra[i + 1] = rgba[i + 1];
            bgra[i + 2] = rgba[i + 0];
            bgra[i + 3] = rgba[i + 3];
        }
        hr = frame->WritePixels(h, stride, byte_count, bgra.data());
        if (FAILED(hr)) break;
        hr = frame->Commit();
        if (FAILED(hr)) break;
        hr = encoder->Commit();
        if (FAILED(hr)) break;
        ok = true;
    } while (false);

    if (properties) properties->Release();
    if (frame) frame->Release();
    if (encoder) encoder->Release();
    if (stream) stream->Release();
    if (factory) factory->Release();
    if (uninitialize) CoUninitialize();
    if (!ok) {
        char detail[64];
        snprintf(detail, sizeof detail, "WIC PNG encode failed (0x%08lx)", (unsigned long)hr);
        error = detail;
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
    }
    return ok;
#else
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (1 + (size_t)w * 4));
    for (uint32_t y = 0; y < h; y++) {
        raw.push_back(0);   // filter: None
        raw.insert(raw.end(), rgba + (size_t)y * w * 4, rgba + (size_t)(y + 1) * w * 4);
    }
    uLongf clen = compressBound((uLong)raw.size());
    std::vector<uint8_t> comp(clen);
    const int zrc = compress2(comp.data(), &clen, raw.data(), (uLong)raw.size(), Z_BEST_SPEED);
    if (zrc != Z_OK) {
        error = "zlib compress2 failed (" + std::to_string(zrc) + ")";
        return false;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        error = std::string("open failed: ") + strerror(errno);
        return false;
    }
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13] = {(uint8_t)(w >> 24),
                        (uint8_t)(w >> 16),
                        (uint8_t)(w >> 8),
                        (uint8_t)w,
                        (uint8_t)(h >> 24),
                        (uint8_t)(h >> 16),
                        (uint8_t)(h >> 8),
                        (uint8_t)h,
                        8,   // bit depth
                        6,   // colour type: RGBA
                        0,
                        0,
                        0};
    put_chunk(f, "IHDR", ihdr, 13);
    put_chunk(f, "IDAT", comp.data(), clen);
    put_chunk(f, "IEND", nullptr, 0);
    if (ferror(f)) {
        error = std::string("write failed: ") + strerror(errno);
        fclose(f);
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        return false;
    }
    if (fclose(f) != 0) {
        error = std::string("close failed: ") + strerror(errno);
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        return false;
    }
    return true;
#endif
}

} // namespace prosper::screenshot
