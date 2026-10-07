// test_writer_provenance_env -- writer_provenance_enabled() is asked once per draw and reads four
// environment names. Each read is a full scan of the environment block when the name is absent,
// which is the normal state of a diagnostic (a sampled Dragon Quest VII load spent measurable CPU
// there). It is therefore memoised for the duration of one submit, and must stay a live read
// outside one.
//
// Two-sided on purpose, like test_env_submit: a memo that is too sticky (process lifetime) makes a
// test that arms the switch between submits go vacuous, and a memo that never holds is the live
// getenv it replaced. So the arm made INSIDE a scope must NOT be seen until the next scope, and the
// arm made OUTSIDE every scope must be seen at once.
#include "diagnostics/env_submit.hpp"
#include "gpu/capture/writer_provenance.hpp"

#include <gtest/gtest.h>
#include <cstdlib>
#ifdef _WIN32
#include <stdlib.h>
#endif

namespace {

void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

void clear_provenance_env() {
    set_test_env("PROSPER_PROVENANCE_DIM", nullptr);
    set_test_env("PROSPER_RESOURCE_HASH_DIM", nullptr);
    set_test_env("PROSPER_GPU_TIMELINE_DEPTH_HASH_DIM", nullptr);
    set_test_env("PROSPER_WRITER_PROVENANCE", nullptr);
}

}  // namespace

TEST(WriterProvenanceEnv, LiveOutsideASubmit) {
    clear_provenance_env();
    ASSERT_EQ(prosper::diag::submit_env_window(), 0u);
    EXPECT_FALSE(prosper::gpu::writer_provenance_enabled());
    set_test_env("PROSPER_PROVENANCE_DIM", "3200x1800");
    EXPECT_TRUE(prosper::gpu::writer_provenance_enabled())
        << "outside a submit the answer must track the environment immediately";
    set_test_env("PROSPER_PROVENANCE_DIM", nullptr);
    EXPECT_FALSE(prosper::gpu::writer_provenance_enabled());
    clear_provenance_env();
}

TEST(WriterProvenanceEnv, FrozenWithinASubmitVisibleAtTheNext) {
    clear_provenance_env();
    {
        const prosper::diag::SubmitEnvScope scope;
        EXPECT_FALSE(prosper::gpu::writer_provenance_enabled());
        set_test_env("PROSPER_RESOURCE_HASH_DIM", "1920x1080");
        EXPECT_FALSE(prosper::gpu::writer_provenance_enabled())
            << "inside one submit the answer is memoised, not re-read per draw";
    }
    {
        const prosper::diag::SubmitEnvScope scope;
        EXPECT_TRUE(prosper::gpu::writer_provenance_enabled())
            << "an arm made since the last submit must be visible at the next one";
        set_test_env("PROSPER_RESOURCE_HASH_DIM", nullptr);
        EXPECT_TRUE(prosper::gpu::writer_provenance_enabled())
            << "a disarm made inside a submit is likewise not seen until the next";
    }
    {
        const prosper::diag::SubmitEnvScope scope;
        EXPECT_FALSE(prosper::gpu::writer_provenance_enabled());
    }
    clear_provenance_env();
}
