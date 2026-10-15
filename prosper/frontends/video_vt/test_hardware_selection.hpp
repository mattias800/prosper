// Test-fixture setup must observe the backend's hardware-only session. A separate decoder's
// failed first-picture probe does not prove that the backend's next session must fail too.
#pragma once

namespace prosper::video::test {

enum class DecodeMode { Hardware, AllowSoftware, Refuse };

template <typename Open, typename Hardware, typename Close>
DecodeMode select_decode_mode(bool independent_probe_decoded, Open open, Hardware hardware,
                              Close close) {
    const int id = open();
    if (id < 0) return independent_probe_decoded ? DecodeMode::Refuse : DecodeMode::AllowSoftware;

    const int observed_hardware = hardware(id);
    close(id);
    return observed_hardware == 1 ? DecodeMode::Hardware : DecodeMode::Refuse;
}

inline bool decoder_matches_mode(DecodeMode mode, int observed_hardware) {
    return mode != DecodeMode::Refuse &&
           (observed_hardware == 1 ||
            (mode == DecodeMode::AllowSoftware && observed_hardware == 0));
}

}   // namespace prosper::video::test
