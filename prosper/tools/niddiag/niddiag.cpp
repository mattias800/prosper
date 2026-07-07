// niddiag — diagnose the NID algorithm: verify SHA1, then try scheme variants
// against the game's real import NIDs to discover the correct one.
#include "../../src/self/module.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <set>
#include <vector>

// self-contained SHA1
struct Sha1 {
    uint32_t h[5]; uint64_t len=0; uint8_t buf[64]; size_t n=0;
    Sha1(){h[0]=0x67452301;h[1]=0xEFCDAB89;h[2]=0x98BADCFE;h[3]=0x10325476;h[4]=0xC3D2E1F0;}
    static uint32_t rol(uint32_t v,int b){return (v<<b)|(v>>(32-b));}
    void block(const uint8_t*p){uint32_t w[80];for(int i=0;i<16;i++)w[i]=(p[i*4]<<24)|(p[i*4+1]<<16)|(p[i*4+2]<<8)|p[i*4+3];
        for(int i=16;i<80;i++)w[i]=rol(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
        for(int i=0;i<80;i++){uint32_t f,k;
            if(i<20){f=(b&c)|(~b&d);k=0x5A827999;}else if(i<40){f=b^c^d;k=0x6ED9EBA1;}
            else if(i<60){f=(b&c)|(b&d)|(c&d);k=0x8F1BBCDC;}else{f=b^c^d;k=0xCA62C1D6;}
            uint32_t t=rol(a,5)+f+e+k+w[i];e=d;d=c;c=rol(b,30);b=a;a=t;}
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;}
    void update(const void*data,size_t l){const uint8_t*p=(const uint8_t*)data;len+=l;
        while(l){size_t t=64-n;if(t>l)t=l;memcpy(buf+n,p,t);n+=t;p+=t;l-=t;if(n==64){block(buf);n=0;}}}
    void finish(uint8_t o[20]){uint64_t bits=len*8;uint8_t pad=0x80;update(&pad,1);uint8_t z=0;
        while(n!=56)update(&z,1);uint8_t lb[8];for(int i=0;i<8;i++)lb[i]=(bits>>(56-i*8))&0xff;update(lb,8);
        for(int i=0;i<5;i++){o[i*4]=h[i]>>24;o[i*4+1]=h[i]>>16;o[i*4+2]=h[i]>>8;o[i*4+3]=h[i];}}
};
static std::string b64(const uint8_t*p,size_t n){
    static const char*A="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    std::string o;size_t i=0;while(i+3<=n){uint32_t v=(p[i]<<16)|(p[i+1]<<8)|p[i+2];
        o+=A[(v>>18)&63];o+=A[(v>>12)&63];o+=A[(v>>6)&63];o+=A[v&63];i+=3;}
    size_t r=n-i;if(r==1){uint32_t v=p[i]<<16;o+=A[(v>>18)&63];o+=A[(v>>12)&63];}
    else if(r==2){uint32_t v=(p[i]<<16)|(p[i+1]<<8);o+=A[(v>>18)&63];o+=A[(v>>12)&63];o+=A[(v>>6)&63];}
    return o;}

static std::string hex(const uint8_t*d,int n){std::string s;char b[3];for(int i=0;i<n;i++){snprintf(b,3,"%02x",d[i]);s+=b;}return s;}

// value-based base64: emit 11 chars, 6 bits each, either high-group-first or low-first
static std::string b64val(uint64_t v,bool lowfirst){
    static const char*A="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    std::string o;
    for(int i=0;i<11;i++){int sh=lowfirst?(i*6):(58-i*6);
        int idx = sh>=0 ? (int)((v>>sh)&0x3f) : (int)((v<<(-sh))&0x3f); o+=A[idx];}
    return o;
}
// mode: 0 bytes[0..7]+std b64, 1 reversed+std b64, 2 LE64 low-first, 3 BE64 low-first,
//       4 LE64 high-first, 5 BE64 high-first, 6 bytes[12..19]+std, 7 rev[12..19]+std
static std::string nid(const std::string&name,const uint8_t*salt,int saltlen,int mode){
    Sha1 s;s.update(name.data(),name.size());if(saltlen)s.update(salt,saltlen);
    uint8_t d[20];s.finish(d);
    auto le=[&](const uint8_t*p){uint64_t v=0;for(int i=0;i<8;i++)v|=(uint64_t)p[i]<<(8*i);return v;};
    auto be=[&](const uint8_t*p){uint64_t v=0;for(int i=0;i<8;i++)v=(v<<8)|p[i];return v;};
    switch(mode){
        case 0:{return b64(d,8).substr(0,11);}
        case 1:{uint8_t r[8];for(int i=0;i<8;i++)r[i]=d[7-i];return b64(r,8).substr(0,11);}
        case 2:return b64val(le(d),true);
        case 3:return b64val(be(d),true);
        case 4:return b64val(le(d),false);
        case 5:return b64val(be(d),false);
        case 6:{return b64(d+12,8).substr(0,11);}
        case 7:{uint8_t r[8];for(int i=0;i<8;i++)r[i]=d[19-i];return b64(r,8).substr(0,11);}
    }
    return "";
}
namespace prosper{const std::vector<std::string>&builtin_symbol_names();}

int main(int argc,char**argv){
    const char*path=argc>=2?argv[1]:"../../PPSA24651-app0/eboot.bin";
    // 1) verify SHA1
    Sha1 s;s.update("abc",3);uint8_t d[20];s.finish(d);
    std::string got=hex(d,20), want="a9993e364706816aba3e25717850c26c9cd0d89d";
    printf("SHA1(\"abc\") = %s  [%s]\n", got.c_str(), got==want?"OK":"WRONG");

    std::string er;auto mo=prosper::Module::load(path,&er);
    if(!mo){printf("load fail: %s\n",er.c_str());return 1;}
    // Ground truth: for libc.prx use EXPORTS (defined syms) — the C library names.
    // For eboot use imports. Collect both; sweep against whichever is richer.
    std::set<std::string> impset, expset;
    for(auto&i:mo->imports) impset.insert(i.nid);
    for(auto&s:mo->symbols) if(!s.is_import && !s.nid.empty()) expset.insert(s.nid);
    printf("distinct import NIDs: %zu | export NIDs: %zu\n", impset.size(), expset.size());
    std::set<std::string>& imp = expset.size() > impset.size() ? expset : impset;
    printf("sweeping against %zu %s NIDs\n", imp.size(), (&imp==&expset)?"EXPORT":"IMPORT");

    // The FULL salt is 16 bytes, not 8 (I had truncated it).
    uint8_t salt16[16]={0x51,0x8D,0x64,0xA6,0x35,0xDE,0xD8,0xC1,0xE6,0xB0,0x39,0xB1,0xC3,0xE5,0x52,0x30};
    const auto& dict = prosper::builtin_symbol_names();
    printf("dictionary size: %zu names\n", dict.size());
    const char* modeName[]={"bytes[0..7]+stdb64","rev[0..7]+stdb64","LE-lowfirst","BE-lowfirst",
                            "LE-highfirst","BE-highfirst","bytes[12..19]","rev[12..19]"};
    for(int mode=0;mode<8;mode++){
        int matches=0;
        for(auto&nm:dict) if(imp.count(nid(nm,salt16,16,mode))) matches++;
        printf("salt16 mode=%d %-18s : %4d / %zu dict names match  (memcpy=%s)\n",
               mode, modeName[mode], matches, dict.size(), nid("memcpy",salt16,16,mode).c_str());
    }
    return 0;
}
