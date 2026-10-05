// libSceNgs2 waveform surface, re-derived from libSceNgs2.native.sprx.
//
// Why the native module: PS5 titles bind libSceNgs2.native.sprx, not libSceNgs2.sprx. Both export
// the library name libSceNgs2, but sceNgs2SystemResetOption (AQkj7C0f3PY) exists only in the
// native module and every local importer of these three NIDs imports it, so the native module is
// the one these handlers must match. The plain module differs (0x804a04xx errors, 32-byte block).
//
// Offsets below are FILE offsets into libSceNgs2.native.sprx (first LOAD: vaddr = file - 0x4000):
// ParseWaveformData 0x10ed0 -> core 0x10f50, chunk finder 0x12560, chunk reader 0x124a0,
// CalcWaveformBlock 0x12060, SystemResetOption 0x21090. The parse core is shared by the file,
// callback and memory readers; the guest entry point always selects the memory reader (type 1),
// so every bound below is that reader's: a read of [off, off+n) succeeds only when off+n <= size.
// CONFIDENCE: HIGH on everything stated as "native" (read from the disassembly and exercised by
// test_ngs2_waveform); the places where prosper must differ because the module itself would fault
// or loop forever are marked "DIVERGES" and fail visibly.
#include "hle/audio/ngs2_waveform.hpp"

#include "host/memory/guest_memory_copy.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

namespace prosper::ngs2_waveform {
namespace {

void put_u32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}
void put_u64(uint8_t* p, uint64_t v) {
    put_u32(p, (uint32_t)v);
    put_u32(p + 4, (uint32_t)(v >> 32));
}
uint32_t get_u16(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
uint32_t get_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t get_u64(const uint8_t* p) { return (uint64_t)get_u32(p) | ((uint64_t)get_u32(p + 4) << 32); }
uint32_t get_u32be(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}
constexpr uint32_t fourcc(char a, char b, char c, char d) {
    return (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) | ((uint32_t)(uint8_t)c << 16) |
           ((uint32_t)(uint8_t)d << 24);
}

// One-line notice for the places prosper cannot follow the module (it would fault or spin there).
void diverge_note(const char* what) {
    static std::atomic<uint32_t> notes{0};
    if (notes.fetch_add(1) < 16) fprintf(stderr, "[ngs2] waveform: %s\n", what);
}

// Bytes per sample for types 0x10..0x1b, rodata 0x54d10 (u64 entries).
constexpr uint32_t kPcmBytesPerSample[12] = {1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 8, 8};
// ATRAC9 samples per frame by sampling-rate index (config bits 20..23), rodata 0x4e520 (u16).
constexpr uint32_t kAtrac9FrameSamples[16] = {0, 64, 0, 0, 128, 0, 0, 256, 0, 0, 0, 0, 0, 0, 0, 0};

// WAVE_FORMAT_EXTENSIBLE SubFormat GUIDs the module compares at fmt+0x18 (rodata 0x70820..0x7084f).
constexpr uint8_t kGuidPcm[16] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
                                  0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71};
constexpr uint8_t kGuidAtrac9[16] = {0xd2, 0x42, 0xe1, 0x47, 0xba, 0x36, 0x8d, 0x4d,
                                     0x88, 0xfc, 0x61, 0x65, 0x4f, 0x8c, 0x83, 0x6c};
constexpr uint8_t kGuidExt43[16] = {0x1c, 0xe0, 0x10, 0x9e, 0xcc, 0x8b, 0xd9, 0x47,
                                    0x8f, 0x3b, 0xe3, 0x0b, 0x8b, 0xdc, 0x35, 0x29};

// The module's memory reader over the guest image [base, base+size).
struct MemReader {
    uint64_t base = 0, size = 0;
    bool read(uint64_t off, void* dst, size_t n) const {
        if (off > size || n > size - off) return false;
        return host::guest_read_exact(base + off, dst, n);
    }
};

// A located RIFF chunk: body offset, DECLARED size (never clamped), and up to 0x4c field bytes.
struct Chunk {
    uint32_t body_off = 0;
    uint32_t size = 0;
    uint8_t bytes[0x4c]{};
};

// Chunk finder, 0x12560. Walks from file offset 12 with NO "WAVE" check. Ends the walk (not found)
// on a zero-sized non-matching chunk or once the next offset reaches riff_size-4 (u64 compare, so
// riff_size < 4 never ends the walk by size); a riff_size in [4,16] finds nothing. Advances by the
// even-padded size in u32. On a match it reads min(max, size) body bytes through the reader (which
// must fit in `size`) and hands back a pointer INTO the image, so field reads past that count see
// the following file bytes; those are read best-effort here and zero where unreadable.
bool find_chunk(const MemReader& r, uint32_t id, uint32_t max, Chunk& c) {
    uint8_t hdr[12];
    if (!r.read(0, hdr, sizeof hdr)) return false;
    const uint64_t limit = (uint64_t)get_u32(hdr + 4) - 4;   // u64 wrap, as native (0x1264a)
    if (limit < 13) return false;
    uint32_t off = 12;
    for (;;) {
        uint8_t ch[8];
        if (!r.read(off, ch, sizeof ch)) return false;
        const uint32_t body = off + 8;
        const uint32_t size = get_u32(ch + 4);
        if (get_u32(ch) == id) {
            c.body_off = body;
            c.size = size;
            std::memset(c.bytes, 0, sizeof c.bytes);
            const uint32_t count = std::min(max, size);
            if (count) {
                if (!r.read(body, c.bytes, count)) return false;
                if (count < max) host::guest_read_prefix(r.base + body + count, c.bytes + count,
                                                         max - count);
            }
            return true;
        }
        if (size == 0) return false;
        const uint32_t next = ((size + 1u) & ~1u) + body;
        // DIVERGES: a size that wraps the u32 offset back to or below this chunk makes the native
        // walk revisit chunks forever; prosper ends it as "not found".
        if (next <= off) return false;
        off = next;
        if (!(limit > (uint64_t)next)) return false;
    }
}

// Parsed description, host order. Field names follow the info header they are written to.
struct Parsed {
    uint32_t type = 0, channels = 0, rate = 0, config = 0;
    uint32_t data_offset = 0, data_size = 0;
    uint32_t loop_begin = 0, loop_end = 0, samples = 0;
    uint32_t unit_size = 0, unit_samples = 0, units_per_frame = 0, delay = 0;
    // Type 0x43 only: fields of its "loop" chunk that place the blocks (0x117b9..0x117fd).
    uint32_t l4 = 0, l8 = 0, lc = 0, l14 = 0, l18 = 0, l1c = 0;
};

// VAGp, 0x11174..0x11f04. Big-endian header; data at 0x30; frames of 16*ch bytes, 28 samples.
uint32_t parse_vag(const MemReader& r, Parsed& w) {
    uint8_t h[0x30];
    if (!r.read(0, h, sizeof h) || std::memcmp(h, "VAGp", 4) != 0) return kNgs2NativeErrInvalidData;
    w.type = kNgs2NativeTypeVag;
    w.channels = h[0x1e] ? h[0x1e] : 1;
    w.rate = get_u32be(h + 0x10);
    w.data_offset = 0x30;
    w.data_size = get_u32be(h + 0x0c);   // declared, unclamped
    const uint32_t frame = 16u * w.channels;
    w.unit_size = frame;
    w.unit_samples = 28;
    w.units_per_frame = 1;
    w.samples = (w.data_size / frame) * 28u;
    // The flag byte of the frame at 0x30+frame decides: bit 1 clear -> one frame fewer, no loop;
    // set -> scan every frame's channel-0 flag (3 = loop end, 6 = loop start).
    uint8_t row[16];
    if (!r.read(0x30 + (uint64_t)frame, row, sizeof row)) return kNgs2NativeErrInvalidData;
    if (!(row[1] & 2)) {
        w.samples -= 28u;
        return 0;
    }
    if (w.data_size == 0) return 0;
    const uint64_t last = (uint64_t)((w.data_size - 1u) / frame) * frame;
    // The native scan fails at the first frame past `size` and writes nothing, so checking the
    // last frame first gives the same result. DIVERGES: a data size within one frame of 4 GiB
    // wraps the native u32 cursor and scans forever; prosper stops at the last frame.
    if (last + 0x40 > r.size) return kNgs2NativeErrInvalidData;
    const uint64_t rows = last / frame + 1;
    const uint64_t per = std::max<uint64_t>(1, 65536 / frame);
    std::vector<uint8_t> buf((size_t)(per * frame + 16));
    for (uint64_t k0 = 0; k0 < rows; k0 += per) {
        const uint64_t k1 = std::min(rows, k0 + per);
        if (!r.read(0x30 + k0 * frame, buf.data(), (size_t)((k1 - k0 - 1) * frame + 16)))
            return kNgs2NativeErrInvalidData;
        for (uint64_t k = k0; k < k1; ++k) {
            const uint8_t flag = buf[(size_t)((k - k0) * frame + 1)] & 0xf;
            if (flag == 3) w.loop_end = (uint32_t)k * 28u + 28u;
            else if (flag == 6) w.loop_begin = (uint32_t)k * 28u;
        }
    }
    return 0;
}

// RIFF/WAVE, 0x1106e..0x1198a.
uint32_t parse_riff(const MemReader& r, Parsed& w) {
    Chunk fmt;
    if (!find_chunk(r, fourcc('f', 'm', 't', ' '), 0x34, fmt)) return kNgs2NativeErrInvalidData;
    const uint8_t* f = fmt.bytes;
    const uint32_t tag = get_u16(f);
    const uint32_t channels = get_u16(f + 2);
    const uint32_t bits = get_u16(f + 0xe);
    // Channels and rate come from the fmt header for every codec, ATRAC9 included (0x110c4/0x1138e).
    w.channels = channels;
    w.rate = get_u32(f + 4);
    w.unit_samples = 1;
    w.units_per_frame = 1;
    enum class Route { Pcm, Float, Atrac9, Ext43 } route;
    if (tag == 0xfffe) {
        if (fmt.size < 0x28) return kNgs2NativeErrUnsupportedFormat;
        const uint8_t* guid = f + 0x18;
        if (std::memcmp(guid, kGuidPcm, 16) == 0) route = Route::Pcm;   // -> tag-1 ladder, 0x11c20
        else if (std::memcmp(guid, kGuidAtrac9, 16) == 0 || fmt.size == 0x34) route = Route::Atrac9;
        else if (std::memcmp(guid, kGuidExt43, 16) == 0) route = Route::Ext43;
        else return kNgs2NativeErrUnsupportedFormat;
    } else if (tag == 3) {
        route = Route::Float;
    } else if (tag == 1) {
        route = Route::Pcm;
    } else {
        return kNgs2NativeErrUnsupportedFormat;
    }
    bool atrac9 = false, ext43 = false;
    switch (route) {
        case Route::Pcm:   // 0x110eb: bits 8/16/24/32 -> 0x11/0x12/0x14/0x16
            if (bits == 8) w.type = 0x11;
            else if (bits == 16) w.type = 0x12;
            else if (bits == 24) w.type = 0x14;
            else if (bits == 32) w.type = 0x16;
            else return kNgs2NativeErrUnsupportedFormat;
            w.unit_size = channels * (bits / 8);
            break;
        case Route::Float:   // 0x1144b: bits 32/64 -> 0x18/0x1a
            if (bits == 32) w.type = 0x18;
            else if (bits == 64) w.type = 0x1a;
            else return kNgs2NativeErrUnsupportedFormat;
            w.unit_size = channels * (bits / 8);
            break;
        case Route::Atrac9: {   // 0x11ce2: config assembled big-endian from fmt+0x2c..0x2f
            w.type = kNgs2NativeTypeAtrac9;
            w.config = get_u32be(f + 0x2c);
            w.unit_size = ((w.config >> 5) & 0x7ff) + 1;                    // frame bytes
            w.unit_samples = kAtrac9FrameSamples[(w.config >> 20) & 0xf];   // frame samples
            w.units_per_frame = 1u << ((w.config >> 3) & 3);                // frames per superframe
            w.delay = w.unit_samples;
            atrac9 = true;
            break;
        }
        case Route::Ext43:   // 0x113e2/0x11f09: no unit geometry; config/loops from its own chunks
            w.type = kNgs2NativeTypeExt43;
            ext43 = true;
            break;
    }

    uint32_t declared_samples = 0;
    Chunk fact;
    if (find_chunk(r, fourcc('f', 'a', 'c', 't'), 0xc, fact)) {
        declared_samples = get_u32(fact.bytes);
        if (ext43) w.delay = get_u32(fact.bytes + 4);
    } else if (atrac9 || ext43) {
        return kNgs2NativeErrInvalidData;   // 0x1162b -> 0x1192a / 0x11649
    }

    // Loop points from a sampler chunk with at least one loop: begin = start, end = end + 1
    // (exclusive); ATRAC9 shifts both by the decoder delay (0x116cd..0x11729).
    Chunk smpl;
    if (find_chunk(r, fourcc('s', 'm', 'p', 'l'), 0x3c, smpl) && get_u32(smpl.bytes + 0x1c) != 0) {
        const uint32_t start = get_u32(smpl.bytes + 0x2c);
        const uint32_t end = get_u32(smpl.bytes + 0x30);
        w.loop_begin = atrac9 ? start - w.delay : start;
        w.loop_end = (atrac9 ? end - w.delay : end) + 1u;
    }

    if (ext43) {
        Chunk cmap;
        if (find_chunk(r, fourcc('c', 'm', 'a', 'p'), 0xb, cmap)) w.config = cmap.bytes[0];
        Chunk loop;
        if (find_chunk(r, fourcc('l', 'o', 'o', 'p'), 0x4c, loop) && get_u32(loop.bytes) != 0) {
            w.l4 = get_u32(loop.bytes + 0x04);
            w.l8 = get_u32(loop.bytes + 0x08);
            w.lc = get_u32(loop.bytes + 0x0c);
            w.loop_begin = get_u32(loop.bytes + 0x10);
            w.l14 = get_u32(loop.bytes + 0x14);
            w.l18 = get_u32(loop.bytes + 0x18);
            w.l1c = get_u32(loop.bytes + 0x1c);
            w.loop_end = get_u32(loop.bytes + 0x20);
        }
    }

    Chunk data;
    if (!find_chunk(r, fourcc('d', 'a', 't', 'a'), 0, data)) return kNgs2NativeErrInvalidData;
    w.data_offset = data.body_off;
    w.data_size = data.size;   // declared, unclamped; the payload is never read

    if (w.type >= kNgs2NativeTypePcmFirst && w.type <= kNgs2NativeTypePcmLast && !declared_samples) {
        const uint32_t frame_bytes = (channels * bits) >> 3;   // 0x118bc, raw fmt fields
        if (!frame_bytes) {
            // DIVERGES: the module divides by zero here (zero channels and no fact chunk).
            diverge_note("PCM fmt with zero channels and no fact chunk (native faults); refused");
            return kNgs2NativeErrInvalidData;
        }
        w.samples = w.data_size / frame_bytes;
    } else {
        w.samples = declared_samples;
    }
    return 0;
}

// Block list, 0x11a20..0x11cdd. Ordinary types build 1-3 blocks with CalcWaveformBlock itself
// (format = the info header), marking the loop block numRepeats = -1; type 0x43 places its
// blocks from its loop chunk. Every block offset is then made absolute with data_offset.
uint32_t build_blocks(const Parsed& w, uint8_t (*b)[kNgs2NativeWaveformBlockBytes]) {
    uint32_t count = 0;
    const uint32_t lb = w.loop_begin, le = w.loop_end;
    auto calc = [&](uint32_t pos, uint32_t num) {
        calc_block(w.type, w.channels, w.config, pos, num, b[count]);   // result ignored, as native
        return b[count++];
    };
    if (w.type == kNgs2NativeTypeExt43) {
        if (le <= lb) {
            std::memset(b[0], 0, kNgs2NativeWaveformBlockBytes);
            put_u64(b[0] + 0x08, w.data_size);
            put_u32(b[0] + 0x14, w.delay);
            put_u32(b[0] + 0x18, w.samples);
            count = 1;
        } else {
            if (lb != 0) {
                std::memset(b[count], 0, kNgs2NativeWaveformBlockBytes);
                put_u64(b[count] + 0x08, w.l8);
                put_u32(b[count] + 0x14, w.delay);
                put_u32(b[count] + 0x18, lb - w.delay);
                ++count;
            }
            std::memset(b[count], 0, kNgs2NativeWaveformBlockBytes);
            put_u64(b[count] + 0x00, w.l4);
            put_u64(b[count] + 0x08, (uint32_t)(w.l18 - w.l4));
            put_u32(b[count] + 0x10, 0xffffffffu);
            put_u32(b[count] + 0x14, w.lc);
            put_u32(b[count] + 0x18, le - lb);
            ++count;
            if (w.samples > le) {
                std::memset(b[count], 0, kNgs2NativeWaveformBlockBytes);
                put_u64(b[count] + 0x00, w.l14);
                put_u64(b[count] + 0x08, (uint32_t)(w.data_size - w.l14));
                put_u32(b[count] + 0x14, w.l1c);
                put_u32(b[count] + 0x18, w.samples - le);
                ++count;
            }
        }
    } else if (le <= lb) {
        calc(0, w.samples);
    } else {
        if (lb != 0) calc(0, lb);
        put_u32(calc(lb, le - lb) + 0x10, 0xffffffffu);
        if (w.samples != le) calc(le, w.samples - le);
    }
    for (uint32_t i = 0; i < count; ++i) put_u64(b[i], get_u64(b[i]) + w.data_offset);
    return count;
}

}   // namespace

uint32_t calc_block(uint32_t type, uint32_t channels, uint32_t config, uint32_t pos, uint32_t num,
                    uint8_t block[kNgs2NativeWaveformBlockBytes]) {
    std::memset(block, 0, kNgs2NativeWaveformBlockBytes);
    if (type >= kNgs2NativeTypePcmFirst && type <= kNgs2NativeTypePcmLast) {
        // 0x120a5: stride = channels * bytes-per-sample; offset and end computed modulo 2^32.
        const uint64_t stride = (uint64_t)channels * kPcmBytesPerSample[type - kNgs2NativeTypePcmFirst];
        const uint32_t offset = (uint32_t)((uint64_t)pos * stride);
        const uint32_t end = (pos + num) * (uint32_t)stride;
        put_u64(block + 0x00, offset);
        put_u64(block + 0x08, (uint32_t)(end - offset));
        put_u32(block + 0x18, num);
        return 0;
    }
    if (type == kNgs2NativeTypeVag) {
        // 0x1219b: whole 28-sample frames; numSamples is the frame-rounded span, skip 0.
        const uint32_t frame = 16u * channels;
        const uint32_t first = pos / 28u;
        const uint32_t past = (pos + num + 27u) / 28u;
        const uint32_t offset = (uint32_t)((uint64_t)first * frame);
        put_u64(block + 0x00, offset);
        put_u64(block + 0x08, (uint32_t)(past * frame - offset));
        put_u32(block + 0x18, (past - first) * 28u);
        return 0;
    }
    if (type == kNgs2NativeTypeAtrac9) {
        // 0x1210f: superframe-aligned; skip = decoder delay + position inside the superframe.
        const uint32_t frame_bytes = ((config >> 5) & 0x7ff) + 1;
        const uint32_t sf_shift = (config >> 3) & 3;
        const uint64_t sf_bytes = (uint64_t)frame_bytes << sf_shift;
        const uint32_t frame_samples = kAtrac9FrameSamples[(config >> 20) & 0xf];
        const uint32_t sf_samples = frame_samples << sf_shift;
        if (!sf_samples) {
            // DIVERGES: an unknown sampling-rate index has 0 frame samples and the module divides
            // by zero; prosper refuses with the unsupported-format code instead.
            diverge_note("ATRAC9 config with an unknown sampling-rate index (native faults); refused");
            return kNgs2NativeErrUnsupportedFormat;
        }
        const uint32_t first = pos / sf_samples;
        const uint32_t lead = frame_samples + pos;
        const uint32_t past = (num + lead + sf_samples - 1u) / sf_samples;
        const uint32_t offset = (uint32_t)((uint64_t)first * sf_bytes);
        put_u64(block + 0x00, offset);
        put_u64(block + 0x08, (uint32_t)(past * (uint32_t)sf_bytes - offset));
        put_u32(block + 0x14, lead - first * sf_samples);
        put_u32(block + 0x18, num);
        return 0;
    }
    return kNgs2NativeErrUnsupportedFormat;
}

uint32_t parse_waveform_data(uint64_t data, uint64_t size, uint64_t info) {
    // 0x10ef7: the first 8 bytes of info are zeroed before anything is checked, failures included.
    if (info) {
        static const uint8_t zero[8]{};
        // DIVERGES: the module faults on an unwritable info pointer; prosper reports it as NULL.
        if (!host::guest_write_exact(info, zero, sizeof zero)) return kNgs2NativeErrNullOut;
    }
    if (!info) return kNgs2NativeErrNullOut;                       // 0x10f86
    if (size < 4 || !data) return kNgs2NativeErrInvalidData;       // 0x11154 / 0x11169
    const MemReader r{data, size};
    uint8_t magic[4];
    if (!r.read(0, magic, sizeof magic)) return kNgs2NativeErrInvalidData;
    Parsed w;
    uint32_t rc;
    if (std::memcmp(magic, "VAGp", 4) == 0) rc = parse_vag(r, w);
    else if (std::memcmp(magic, "RIFF", 4) == 0) rc = parse_riff(r, w);
    else return kNgs2NativeErrInvalidData;   // 0x11037/0x11053: anything else, no WAVE check
    if (rc) return rc;

    // Only the header and the blocks actually built are written (0x119c0..0x11cdd); the unused
    // tail of the 0xe8-byte info keeps whatever the guest had there.
    uint8_t out[kNgs2NativeWaveformInfoHeaderBytes +
                kNgs2NativeWaveformMaxBlocks * kNgs2NativeWaveformBlockBytes]{};
    put_u32(out + 0x00, w.type);
    put_u32(out + 0x04, w.channels);
    put_u32(out + 0x08, w.rate);
    put_u32(out + 0x0c, w.config);
    put_u32(out + 0x18, w.data_offset);
    put_u32(out + 0x1c, w.data_size);
    put_u32(out + 0x20, w.loop_begin);
    put_u32(out + 0x24, w.loop_end);
    put_u32(out + 0x28, w.samples);
    put_u32(out + 0x2c, w.unit_size);
    put_u32(out + 0x30, w.unit_samples);
    put_u32(out + 0x34, w.units_per_frame);
    put_u32(out + 0x38, w.unit_size * w.units_per_frame);
    put_u32(out + 0x3c, w.units_per_frame * w.unit_samples);
    put_u32(out + 0x40, w.delay);
    auto* blocks = reinterpret_cast<uint8_t(*)[kNgs2NativeWaveformBlockBytes]>(
        out + kNgs2NativeWaveformInfoHeaderBytes);
    const uint32_t count = build_blocks(w, blocks);
    put_u32(out + 0x44, count);
    const size_t bytes = kNgs2NativeWaveformInfoHeaderBytes + (size_t)count * kNgs2NativeWaveformBlockBytes;
    return host::guest_write_exact(info, out, bytes) ? 0 : kNgs2NativeErrNullOut;
}

uint32_t calc_waveform_block(uint64_t format, uint32_t sample_pos, uint32_t num_samples,
                             uint64_t block) {
    if (!block) return kNgs2NativeErrNullOut;   // 0x1206b, before the block is touched
    uint8_t out[kNgs2NativeWaveformBlockBytes]{};
    // 0x12084: the block is zeroed before the format is examined, so every failure below leaves
    // it zero. DIVERGES: an unwritable block faults natively; prosper reports it as NULL.
    if (!host::guest_write_exact(block, out, sizeof out)) return kNgs2NativeErrNullOut;
    if (!format) return kNgs2NativeErrNullFormat;   // 0x12089
    // The module reads type@0, channels@4 and (ATRAC9 only) config@0xc.
    uint8_t f[16]{};
    const size_t got = host::guest_read_prefix(format, f, sizeof f);
    if (got < 8 || (get_u32(f) == kNgs2NativeTypeAtrac9 && got < 16)) {
        diverge_note("CalcWaveformBlock: unreadable format (native faults); refused");
        return kNgs2NativeErrNullFormat;
    }
    const uint32_t rc = calc_block(get_u32(f), get_u32(f + 4), get_u32(f + 0xc), sample_pos,
                                   num_samples, out);
    if (rc) return rc;
    return host::guest_write_exact(block, out, sizeof out) ? 0 : kNgs2NativeErrNullOut;
}

uint32_t system_reset_option(uint64_t option) {
    if (!option) return kNgs2NativeErrNullOut;   // 0x2109a
    // 0x2109f: 0x90 bytes copied from the template at rodata 0x58800.
    uint8_t out[kNgs2NativeSystemOptionBytes]{};
    put_u32(out + 0x00, kNgs2NativeSystemOptionBytes);   // size
    put_u32(out + 0x6c, 512);
    put_u32(out + 0x70, 256);
    put_u32(out + 0x74, 48000);
    put_u32(out + 0x78, 8);
    // DIVERGES: an unwritable option faults natively; prosper reports it as NULL.
    return host::guest_write_exact(option, out, sizeof out) ? 0 : kNgs2NativeErrNullOut;
}

}   // namespace prosper::ngs2_waveform
