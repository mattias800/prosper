#!/usr/bin/env python3
# Guess NID names: sha1(name + Sony salt), first 8 bytes reversed, Sony b64, 11 chars.
import hashlib, sys

SALT = bytes([0x51,0x8D,0x64,0xA6,0x35,0xDE,0xD8,0xC1,0xE6,0xB0,0x39,0xB1,0xC3,0xE5,0x52,0x30])
A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"

def nid(name):
    d = hashlib.sha1(name.encode() + SALT).digest()
    r = d[7::-1]
    out = ""
    v = 0
    bits = 0
    for b in r:
        v = (v << 8) | b
        bits += 8
        while bits >= 6:
            bits -= 6
            out += A[(v >> bits) & 63]
    if bits:
        out += A[(v << (6 - bits)) & 63]
    return out[:11]

targets = {
    "ts6GlZOKRrE": "libScePlayGo", "M1Gma1ocrGE": "libScePlayGo",
    "1n37q1Bvc5Y": "libSceSystemService",
    "uW4vfTwMQVo": "sd", "sDCBrmc61XU": "sd", "ie7qhZ4X0Cc": "sd",
    "gjRZNnw0JPE": "sd", "ZP4e7rlzOUk": "sd", "TywrFKCoLGY": "sd",
    "vYU8P9Td2Zo": "kern", "rqwFKI4PAiM": "kern", "nu4a0-arQis": "kern",
    "bBfz7kMF2Ho": "kern", "6xVpy0Fdq+I": "kern",
    "sUXGfNMalIo": "trophy2", "bIDov3wBu5Q": "trophy2", "Gz1rmUZpROM": "trophy2",
    "U8IfNl6-Css": "voiceqos", "hdpVEUDFW3s": "ssl", "RnDibcGCPKw": "videodec2",
    "nBDD66kiFW8": "share", "T64o-315wbg": "share", "ORspsWDXPps": "share",
    "WV1GwM32NgY": "npwebapi2", "+o9816YQhqQ": "npwebapi2",
    "tpFJ8LIKvPw": "npuds", "sjaobBgqeB4": "npuds", "hT0IAEvN+M0": "npuds", "5zBnau1uIEo": "npuds",
    "Xs9hdiD7sAA": "posix",
}

cands = []
for base in ["scePlayGo"]:
    for fn in ["Initialize","Terminate","Open","Close","GetLocus","SetToDoList","GetToDoList",
               "GetProgress","GetInstallSpeed","SetInstallSpeed","Prefetch","GetEta",
               "GetLanguageMask","SetLanguageMask","GetChunkId","RequestInstall",
               "GetInstallStatus","GetTotalProgress","Snapshot","GetDiscStatus"]:
        cands.append(base + fn)
for base in ["sceSystemService"]:
    for fn in ["GetStatus","HideSplashScreen","ReceiveEvent","ParamGetInt","ParamGetString",
               "GetDisplaySafeAreaInfo","ReportAbnormalTermination","LoadExec","EnableMusicPlayer",
               "DisableMusicPlayer","GetAppSecurityZone","IsScreenSaverOn","ActivateChatApp",
               "GetLocalProcessStatusList","KillLocalProcess","LaunchApp","GetAppStatus",
               "GetAppIdOfMiniApp","GetAppIdOfBigApp","AcquireBgmCpuBudget","ReleaseBgmCpuBudget",
               "AcquireBgmCpu","ReleaseBgmCpu"]:
        cands.append(base + fn)
for base in ["sceSaveData"]:
    for fn in ["Initialize","Initialize2","Initialize3","Terminate","Mount","Mount2","Mount5",
               "Umount","UmountWithBackup","DirNameSearch","Delete","SetupSaveDataMemory",
               "SetupSaveDataMemory2","GetSaveDataMemory","GetSaveDataMemory2","SetSaveDataMemory",
               "SetSaveDataMemory2","SyncSaveDataMemory","GetEventResult","GetSaveDataRootDir",
               "GetMountInfo","GetParam","SetParam","LoadIcon","SaveIcon","DirNameSearchInternal",
               "CheckBackupData","RestoreBackupData","GetProgress","ClearProgress",
               "TransferringMount","GetSaveDataCount",
               # broader PS4 SDK surface
               "Backup","CheckSaveDataBroken","CheckSaveDataVersion","CheckSaveDataVersionLatest",
               "DebugCleanMount","DebugCompiledSdkVersion","DebugCreateSaveDataRoot",
               "DebugGetThreadId","DebugRemoveSaveDataRoot","DebugSetSaveDataRoot",
               "DeleteAllUser","DeleteCloudData","DeleteUser","DirNameSearch2",
               "DownloadCloudData","GetAllSize","GetAppLaunchedUser","GetAutoUploadConditions",
               "GetAutoUploadRequestInfo","GetAutoUploadSetting","GetBoundPsnAccountCount",
               "GetClientThreadPriority","GetCloudQuotaInfo","GetDataBaseFilePath",
               "GetEventInfo","GetFormat","GetMountedSaveDataCount","GetSaveDataCount2",
               "GetSaveDataMemorySync","GetUpdatedDataCount","GetUserSaveDataCount",
               "IsDeletingUsbData","IsMounted","IsUsbAccessible","LoadIconAsync",
               "RegisterEventCallback","RestoreBackupDataForCdlg","RestoreLoadSaveDataMemory",
               "SaveIconAsync","SetAutoUploadSetting","SetEventInfo","SetSaveDataLibraryUser",
               "ShrinkSaveData","SyncCloudList","UnregisterEventCallback","UploadCloudData",
               # event-queue style (PS5)
               "GetEvent","ClearEvent","AbortEvent","WaitEvent","GetEventStatus",
               "CreateEventQueue","DeleteEventQueue","RegisterEventQueue","UnregisterEventQueue",
               "MountPs4","MountPs5","MountReadOnly","MountRdwr","MountCreate",
               "GetMountPoint","GetMountPointByName","GetMountStatus","MountAuto"]:
        cands.append(base + fn)
for base in ["sceNpTrophy2"]:
    for fn in ["CreateContext","DestroyContext","CreateHandle","DestroyHandle","RegisterContext",
               "AbortHandle","GetGameInfo","GetGroupInfo","GetTrophyInfo","GetGameIcon",
               "UnlockTrophy","ShowTrophyList","GetTrophyUnlockState","SetPlayedWithUsers"]:
        cands.append(base + fn)
for fn in ["sceKernelIsProspero","sceKernelGetSystemSwVersion","sceKernelGetCpuFrequency",
           "sceKernelGetCpuTemperature","sceKernelGetSocSensorTemperature","sceKernelSetGPO",
           "sceKernelGetGPI","sceKernelGetProcessType","sceKernelIsTestKit","sceKernelIsDevKit",
           "sceKernelSetFsstParam","sceKernelSetCompressionAttribute","sceKernelSetMemoryPstate",
           "sceKernelGetCurrentCpu","sceKernelMprotect","sceKernelGetProcessName",
           "sceKernelSetVirtualRangeName","sceKernelDebugOutText","sceKernelGetAppInfo",
           "sceKernelSetProcessProperty","sceKernelSetAppInfo","get_page_table_stats",
           "sceKernelInternalMemoryGetAvailableSize","sceKernelAvailableFlexibleMemorySize",
           "sceKernelGetPrtAperture","sceKernelSetPrtAperture","sceKernelGetExtLibcHandle",
           "sceKernelGetSanitizerMallocReplace","sceKernelGetSanitizerNewReplace",
           "sceKernelGetSanitizerMallocReplaceExternal","sceKernelIsAddressSanitizerEnabled",
           "sceKernelNotifyAppStateChanged","sceKernelGetAppState","scePthreadSetAffinity",
           "scePthreadGetAffinity","sceKernelGetCurrentFrequency","sceKernelGetTraceMemorySize"]:
    cands.append(fn)
for base in ["sceShare"]:
    for fn in ["Initialize","Terminate","OpenParameter","CloseParameter","SetParameter",
               "GetCurrentStatus","EnableScreenshot","DisableScreenshot","EnableVideoRecording",
               "DisableVideoRecording","NotifyEventFinish"]:
        cands.append(base + fn)

hit = 0
for c in cands:
    h = nid(c)
    if h in targets:
        print("%s = %s (%s)" % (h, c, targets[h]))
        hit += 1
print("matched %d/%d targets" % (hit, len(targets)))
