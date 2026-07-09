#!/bin/bash
# Brute-force candidate names for the NpTrophy2 NIDs via the nid test tool if it supports hashing,
# else compile a one-off hasher against nid.cpp.
cd /mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-ae005ba42de93cd40/prosper
cat > /tmp/nidhash.cpp << "EOF"
#include <cstdio>
#include <string>
namespace prosper { std::string nid_hash(const std::string& name); }
int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++)
        printf("%-52s %s\n", argv[i], prosper::nid_hash(argv[i]).c_str());
    return 0;
}
EOF
g++ -O1 -std=c++17 -Isrc /tmp/nidhash.cpp src/hle/nid.cpp -o /tmp/nidhash 2>/tmp/nidhash.err || { cat /tmp/nidhash.err | head -5; exit 1; }
/tmp/nidhash \
  sceNpTrophy2GetTrophySetInfo sceNpTrophy2GetTrophySetInfoInGroup \
  sceNpTrophy2GetTrophyInfo sceNpTrophy2GetTrophyInfoList sceNpTrophy2GetTrophyInfos \
  sceNpTrophy2GetGroupInfo sceNpTrophy2GetGroupInfoList sceNpTrophy2GetGroupInfos \
  sceNpTrophy2GetTrophyList sceNpTrophy2GetTrophies sceNpTrophy2GetTrophyDetails \
  sceNpTrophy2CreateContext sceNpTrophy2DestroyContext sceNpTrophy2RegisterContext \
  sceNpTrophy2AbortHandle sceNpTrophy2CreateHandle sceNpTrophy2DestroyHandle \
  sceNpTrophy2UnlockTrophy sceNpTrophy2GetUnlockedTrophies \
  sceNpTrophy2GetTrophyGameDetails sceNpTrophy2GetTrophyGameData \
  sceNpTrophy2SetInfoGetTrophyNum sceNpTrophy2GetTrophyNum \
  sceNpTrophy2ShowTrophyList sceNpTrophy2GetGameInfo \
  sceNpTrophy2SystemGetTrophyGameInfo sceNpTrophy2SystemGetTrophyInfo
