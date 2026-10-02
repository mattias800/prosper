#!/usr/bin/env python3
"""#4066 actual CPU raw/tap/retry routes retain producing profile, never infer stored markers."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

REPLAY=os.environ.get("PROSPER_GPU_REPLAY_BIN")
FIXTURE=os.environ.get("PROSPER_FLOAT_TRANSPORT_FIXTURE_BIN")
if not REPLAY or not Path(REPLAY).is_file() or not FIXTURE or not Path(FIXTURE).is_file():
    print("required replay/fixture binary unavailable -- skipping")
    sys.exit(77)
failures=[]
def check(ok,label):
    print("[%s] %s" % ("ok" if ok else "FAIL",label))
    if not ok: failures.append(label)

def facts(data):
    if len(data)<20 or len(data)%4: raise ValueError("invalid module span")
    words=struct.unpack("<%dI" % (len(data)//4),data)
    if words[0]!=0x07230203: raise ValueError("invalid module magic")
    instructions=[]; at=5
    while at<len(words):
        count,op=words[at]>>16,words[at]&65535
        if not count or at+count>len(words): raise ValueError("invalid instruction span")
        instructions.append((op,words[at+1:at+count])); at+=count
    f32=set(); i32=set(); vectors=set(); pointers={}; values={}; results={}; extensions=set()
    cap=False; szi=False; decorations=set(); default=False
    for op,a in instructions:
        if op==22 and len(a)==2 and a[1]==32: f32.add(a[0])
        if op==21 and len(a)==3 and a[1]==32: i32.add(a[0])
        if op==23 and len(a)==3 and a[1] in f32 and 2<=a[2]<=4: vectors.add(a[0])
        if op==32 and len(a)==3: pointers[a[0]]=a[1]
        if op==17 and a==(6029,): cap=True
        if op==10:
            extensions.add(struct.pack("<%dI" % len(a),*a).split(b"\0",1)[0])
        if op==16 and len(a)==3 and a[1:]==(4461,32): szi=True
        if op in (16,331) and len(a)>=2 and a[1]==6028: default=True
        if op==71 and len(a)>=2 and a[1]==40:
            if len(a)!=3 or a[2]!=0 or a[0] in decorations: raise ValueError("noncanonical None decoration")
            decorations.add(a[0])
        if op in (41,42,43,44,46,48,49,50,51,52,55,59) and len(a)>=2: values[a[1]]=a[0]
    in_function=False
    no_result={62,63,64,99,218,219,220,221,224,225,228,246,247,248,249,250,251,252,253,254,255}
    for op,a in instructions:
        if op==54: in_function=True
        elif op==56: in_function=False
        elif in_function and op not in no_result and len(a)>=2:
            values[a[1]]=a[0]; results[a[1]]=(op,a)
    eligible=set()
    for ident,(op,a) in results.items():
        if op==124 and len(a)==3:
            operand_type=values.get(a[2])
            if (a[0] in f32 and operand_type in i32) or (a[0] in i32 and operand_type in f32): eligible.add(ident)
        if op==61 and len(a)==3 and a[0] in f32|vectors and pointers.get(values.get(a[2]))==1: eligible.add(ident)
    if decorations-eligible: raise ValueError("None targets arithmetic or unsupported operations")
    return cap,szi,b"SPV_KHR_float_controls2" in extensions,decorations,eligible,default

VAL=os.environ.get("PROSPER_SPIRV_VAL_BIN") or shutil.which("spirv-val")
if not VAL and os.environ.get("VULKAN_SDK"):
    candidate=Path(os.environ["VULKAN_SDK"])/"Bin/spirv-val.exe"
    if candidate.is_file(): VAL=str(candidate)
check(bool(VAL),"strict SPIR-V validator available; absence is not a pass")
scratch=Path(os.environ.get("PROSPER_TEST_SCRATCH_DIR") or Path.cwd()/"prosper-test-scratch")
scratch.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix="float-transport-",dir=scratch) as directory:
    directory=Path(directory)
    env={k:v for k,v in os.environ.items() if not k.startswith("PROSPER_")}
    fixture_env=dict(env,PROSPER_TEST_SCRATCH_DIR=str(directory))
    fixture=subprocess.run([FIXTURE,"--write-fixture",str(directory)],env=fixture_env,capture_output=True,text=True,timeout=120)
    check(fixture.returncode==0,"actual realization/collector fixture succeeds without Vulkan")
    if fixture.returncode: print(fixture.stdout+fixture.stderr)
    for failed in (False,True):
        path=directory/("explicit-nonfinite32"+("-failed" if failed else "")+".prgcap")
        data=path.read_bytes()
        tail=(struct.pack("<III",0,0,1)+b"\x02"+struct.pack("<I",1)+b"\x00" if failed else
              struct.pack("<I",1)+b"\x02"+struct.pack("<II",0,0))
        check(struct.unpack_from("<I",data,8)[0]==65 and data.endswith(tail),"genuine v65 exact profile tail")
        legacy=bytearray(data[:-len(tail)]); struct.pack_into("<I",legacy,8,64)
        (directory/("legacy"+("-failed" if failed else "")+".prgcap")).write_bytes(legacy)
    states=("unknown","implicit","explicit-nonfinite32","legacy")
    modules=0
    for state in states:
        for route in ("recompile-raw","fs-tap","retry-failed-stage"):
            capture=directory/(state+("-failed" if route=="retry-failed-stage" else "")+".prgcap")
            output=directory/(state+"-"+route+".spv")
            route_env=dict(env)
            if route=="fs-tap": route_env["PROSPER_FS_TAP"]="7:0"
            args=(["--retry-failed-stage","0:0","--retry-failed-stage-spv",str(output),str(capture)]
                  if route=="retry-failed-stage" else
                  ["--inspect-only"]+(["--recompile-raw"] if route=="recompile-raw" else [])+
                  ["--dump-shader","7:fs",str(output),str(capture)])
            done=subprocess.run([REPLAY]+args,env=route_env,capture_output=True,text=True,timeout=120)
            label=state+" "+route
            check(done.returncode==0 and output.is_file(),label+" actual CPU regeneration succeeds")
            if done.returncode or not output.is_file():
                print(done.stdout+done.stderr); continue
            expected="unknown" if state=="legacy" else state
            check("float-transport="+expected+" source=" in done.stderr,label+" exact producing profile visible")
            try:
                cap,szi,extension,decorations,eligible,default=facts(output.read_bytes())
                check(bool(eligible) and not default,label+" live transport slice exists without FPDefault")
                if state=="explicit-nonfinite32":
                    check(cap and szi and extension and decorations==eligible,
                          label+" actual None covers exactly all eligible transports; stored opposite marker ignored")
                else:
                    check(not cap and not extension and not decorations and "nonfinite raw/Input transport semantics UNVERIFIED" in done.stderr,
                          label+" unknown/implicit stays visibly unverified, no inferred feature authority")
            except ValueError as error: check(False,label+" fail-closed structural oracle: "+str(error))
            if VAL:
                valid=subprocess.run([VAL,"--target-env","vulkan1.3",str(output)],capture_output=True,text=True,timeout=30)
                check(valid.returncode==0,label+" strict source valid")
                if valid.returncode: print(valid.stdout+valid.stderr)
            modules+=1
        if state!="legacy":
            stored=directory/(state+"-stored.spv")
            done=subprocess.run([REPLAY,"--inspect-only","--dump-shader","7:fs",str(stored),
                                 str(directory/(state+".prgcap"))],env=env,capture_output=True,text=True,timeout=120)
            check(done.returncode==0 and stored.is_file(),state+" stored module remains usable offline")
            if stored.is_file(): check(facts(stored.read_bytes())[0] == (state!="explicit-nonfinite32"),
                                      state+" opposing stored capability is dormant producer authority")
    original=(directory/"explicit-nonfinite32.prgcap").read_bytes(); start=len(original)-13
    corrupt={"draw-count":original[:start]+struct.pack("<I",2)+original[start+4:],
             "draw-tag":original[:start+4]+b"\x03"+original[start+5:],
             "compute-count":original[:start+5]+struct.pack("<I",1)+original[start+9:],
             "failure-count":original[:-4]+struct.pack("<I",1),
             "truncated":original[:-1],"trailing":original+b"\x00",
             "version-only64":original[:8]+struct.pack("<I",64)+original[12:]}
    for name,data in corrupt.items():
        capture=directory/(name+".prgcap"); capture.write_bytes(data)
        output=directory/"sentinel.spv"; output.write_bytes(b"unchanged")
        done=subprocess.run([REPLAY,"--inspect-only","--recompile-raw","--dump-shader","7:fs",
                             str(output),str(capture)],env=env,capture_output=True,text=True,timeout=120)
        check(done.returncode==2 and "fragment-wave=" not in done.stderr and output.read_bytes()==b"unchanged",
              name+" refuses before regeneration/dump")
    check(modules==12,"all 12 profile/route source modules executed, no missing matrix arms")
print("float transport CLI: %d source modules, %d failures" % (modules,len(failures)))
sys.exit(1 if failures else 0)
