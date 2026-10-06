// Which descriptor-table uses build_stage_table publishes (srt_publication_dedupe.hpp). A keyed
// constant buffer is published once per key, because its consumers resolve by srt_offset; once its
// key clashes it is published key-less, so each consumer needs its own entry. Kena's title pixel
// shader reads one V# at two PCs with its key already held: before this rule the second PC had no
// entry and the whole program was refused as unresolved-cbuf.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/srt_publication_dedupe.hpp"
#include <gtest/gtest.h>

using namespace prosper::gpu;

namespace {

SrtUse use(int kind, uint32_t key, uint32_t pc, uint32_t format = UINT32_MAX) {
    SrtUse u;
    u.kind = kind;
    u.key = key;
    u.use_pc = pc;
    u.instruction_format = format;
    return u;
}

TEST(SrtPublicationDedupe, AKeyedBufferIsPublishedOncePerKey) {
    SrtPublicationDedupe d;
    EXPECT_TRUE(d.admit(use(1, 0x40, 5)));
    EXPECT_FALSE(d.admit(use(1, 0x40, 7))) << "a second consumer resolves by the same key";
    EXPECT_TRUE(d.admit(use(1, 0x50, 9))) << "another key is another resource";
}

TEST(SrtPublicationDedupe, AClashedKeyIsPublishedPerConsumer) {
    SrtPublicationDedupe d;
    const SrtUse first = use(1, 0x130, 5);
    ASSERT_TRUE(d.admit(first));
    d.note_clash(first);   // the key was already held: published key-less, by fetch_pc 5
    EXPECT_TRUE(d.admit(use(1, 0x130, 7))) << "the second consumer needs its own fetch_pc entry";
    EXPECT_FALSE(d.admit(use(1, 0x130, 7))) << "...once";
    EXPECT_FALSE(d.admit(use(1, 0x40, 9)) && d.admit(use(1, 0x40, 11)))
        << "an unclashed key keeps per-key publication";
}

TEST(SrtPublicationDedupe, PerInstructionUsesAreUnchanged) {
    SrtPublicationDedupe d;
    EXPECT_TRUE(d.admit(use(0, 0x20, 3)));
    EXPECT_TRUE(d.admit(use(0, 0x20, 4))) << "textures sharing a key are per instruction";
    EXPECT_TRUE(d.admit(use(1, 0xFFFFFFFFu, 6)));
    EXPECT_TRUE(d.admit(use(1, 0xFFFFFFFFu, 8))) << "key-less buffers are per instruction";
    EXPECT_TRUE(d.admit(use(1, 0x60, 10, 4)));
    EXPECT_TRUE(d.admit(use(1, 0x60, 12, 4))) << "exact MTBUF uses are per instruction";
    EXPECT_FALSE(d.admit(use(0, 0x20, 3))) << "and still deduplicated per instruction";
}

TEST(SrtPublicationDedupe, PcIdentitiesDoNotCollideWithKeys) {
    SrtPublicationDedupe d;
    EXPECT_TRUE(d.admit(use(1, 7, 100)));            // keyed: identity is the key 7
    EXPECT_TRUE(d.admit(use(1, 0xFFFFFFFFu, 7)));    // key-less at pc 7: a different namespace
    EXPECT_TRUE(d.admit(use(0, 0x10, 7)));           // texture at pc 7: a different kind
}

TEST(SrtPublicationDedupe, AKeyLessClashRecordsNothing) {
    SrtPublicationDedupe d;
    const SrtUse keyless = use(1, 0xFFFFFFFFu, 2);
    ASSERT_TRUE(d.admit(keyless));
    d.note_clash(keyless);
    EXPECT_TRUE(d.admit(use(1, 0x40, 4)));
    EXPECT_FALSE(d.admit(use(1, 0x40, 6))) << "an unrelated keyed buffer is still per key";
}

}  // namespace
