// A transient independent decode failure must not reject a later genuine hardware session or
// permit an unrequested software session. These synthetic sessions require no native codec API.
#include "test_hardware_selection.hpp"

#include <gtest/gtest.h>

#include <initializer_list>

namespace {

using prosper::video::test::DecodeMode;
using prosper::video::test::decoder_matches_mode;
using prosper::video::test::select_decode_mode;

struct Session {
    int opened = 41;
    int hardware = 1;
    int opens = 0, queries = 0, closes = 0;

    DecodeMode select(bool probe_decoded) {
        return select_decode_mode(
            probe_decoded,
            [&] {
                ++opens;
                return opened;
            },
            [&](int id) {
                EXPECT_EQ(id, 41);
                ++queries;
                return hardware;
            },
            [&](int id) {
                EXPECT_EQ(id, 41);
                ++closes;
            });
    }
};

TEST(VideoHardwareSelection, FailedIndependentDecodeThenGenuineHardwareOpenStaysHardwareOnly) {
    Session session;
    EXPECT_EQ(session.select(false), DecodeMode::Hardware);
    EXPECT_EQ(session.opens, 1);
    EXPECT_EQ(session.queries, 1) << "the actual session must supply the hardware proof";
    EXPECT_EQ(session.closes, 1) << "setup must drain its own session";
}

TEST(VideoHardwareSelection, SuccessfulIndependentDecodeAndBackendOpenStayHardwareOnly) {
    Session session;
    EXPECT_EQ(session.select(true), DecodeMode::Hardware);
    EXPECT_EQ(session.opens, 1);
    EXPECT_EQ(session.queries, 1);
    EXPECT_EQ(session.closes, 1);
}

TEST(VideoHardwareSelection, BothDecodeAttemptsRefuseBeforeSoftwareCanBeExplicitlyAllowed) {
    Session session;
    session.opened = -1;
    EXPECT_EQ(session.select(false), DecodeMode::AllowSoftware);
    EXPECT_EQ(session.opens, 1);
    EXPECT_EQ(session.queries, 0);
    EXPECT_EQ(session.closes, 0);
}

TEST(VideoHardwareSelection, BackendRefusalAfterSuccessfulIndependentDecodeFailsSetup) {
    Session session;
    session.opened = -1;
    EXPECT_EQ(session.select(true), DecodeMode::Refuse);
    EXPECT_EQ(session.opens, 1);
    EXPECT_EQ(session.queries, 0);
    EXPECT_EQ(session.closes, 0);
}

TEST(VideoHardwareSelection, SoftwareWithoutOverrideFailsSetupEvenAfterAFailedProbe) {
    for (const bool probe_decoded : {false, true}) {
        Session session;
        session.hardware = 0;
        EXPECT_EQ(session.select(probe_decoded), DecodeMode::Refuse);
        EXPECT_EQ(session.queries, 1);
        EXPECT_EQ(session.closes, 1);
    }
}

TEST(VideoHardwareSelection, UnknownDecoderStateFailsSetupEvenAfterAFailedProbe) {
    for (const bool probe_decoded : {false, true}) {
        Session session;
        session.hardware = -1;
        EXPECT_EQ(session.select(probe_decoded), DecodeMode::Refuse);
        EXPECT_EQ(session.queries, 1);
        EXPECT_EQ(session.closes, 1);
    }
}

TEST(VideoHardwareSelection, ExplicitSoftwareAllowanceAlsoAdmitsRecoveredHardware) {
    EXPECT_TRUE(decoder_matches_mode(DecodeMode::AllowSoftware, 0));
    EXPECT_TRUE(decoder_matches_mode(DecodeMode::AllowSoftware, 1));
    EXPECT_FALSE(decoder_matches_mode(DecodeMode::AllowSoftware, -1));
}

TEST(VideoHardwareSelection, OnlyExplicitSoftwareAllowsASoftwareSession) {
    EXPECT_TRUE(decoder_matches_mode(DecodeMode::Hardware, 1));
    EXPECT_FALSE(decoder_matches_mode(DecodeMode::Hardware, 0));
    EXPECT_FALSE(decoder_matches_mode(DecodeMode::Hardware, -1));
    EXPECT_FALSE(decoder_matches_mode(DecodeMode::Refuse, 0));
    EXPECT_FALSE(decoder_matches_mode(DecodeMode::Refuse, 1));
}

}   // namespace
