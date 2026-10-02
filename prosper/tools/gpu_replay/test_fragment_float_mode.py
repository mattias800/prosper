#!/usr/bin/env python3
"""#4056 actual offline raw/tap/retry routes: producing mode, not width override or SPV inference."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

REPLAY = os.environ.get("PROSPER_GPU_REPLAY_BIN")
FIXTURE = os.environ.get("PROSPER_FRAGMENT_FLOAT_MODE_FIXTURE_BIN")
if not REPLAY or not Path(REPLAY).is_file() or not FIXTURE or not Path(FIXTURE).is_file():
    print("required replay/fixture binary unavailable -- skipping")
    sys.exit(77)
failures = []
def check(ok, name):
    print("[%s] %s" % ("ok" if ok else "FAIL", name))
    if not ok:
        failures.append(name)

class Module:
    """Typed fail-closed actual MRT0 SSA evaluator. Unsupported live vocabulary is an error."""
    def __init__(self, data):
        if len(data) < 20 or len(data) % 4:
            raise ValueError("incomplete SPIR-V")
        words = struct.unpack("<%dI" % (len(data)//4), data)
        if words[0] != 0x07230203:
            raise ValueError("bad SPIR-V magic")
        self.defs, self.types, self.locations, self.stores, self.ops = {}, {}, {}, [], []
        pc = 5
        while pc < len(words):
            n, op = words[pc] >> 16, words[pc] & 65535
            if not n or n > len(words)-pc:
                raise ValueError("bad instruction boundary")
            a = words[pc+1:pc+n]
            self.ops.append(op)
            if op in (20,21,22,23,32):
                self.types[a[0]] = (op,a)
            elif op in (41,42,43,59,80,83,124,128,132,168,169,170,171,174,199):
                self.defs[a[1]] = (op,a)
            elif op == 71 and len(a)==3 and a[1]==30:
                self.locations[a[0]]=a[2]
            elif op == 62:
                self.stores.append(a)
            pc += n
        outputs = []
        for pointer, value in self.stores:
            op, a = self.defs.get(pointer,(None,()))
            if op==59 and a[2]==3 and self.locations.get(pointer)==0:
                outputs.append(value)
        if len(outputs)!=1:
            raise ValueError("expected exactly one live MRT0 output")
        self.sink=outputs[0]

    def output(self):
        memo, active = {}, set()
        def evaluate(ident):
            if ident in memo:
                return memo[ident]
            if ident in active or ident not in self.defs:
                raise ValueError("unknown/cyclic live definition")
            active.add(ident)
            op,a=self.defs[ident]
            t,ta=self.types.get(a[0],(None,()))
            if op==43 and len(a)==3 and t in (21,22) and ta[1]==32:
                result=a[2]
            elif op in (41,42) and t==20:
                result=int(op==41)
            elif op==80 and t==23:
                result=tuple(evaluate(x) for x in a[2:])
            elif op in (83,124):
                if op==124 and t not in (21,22):
                    raise ValueError("unsupported bitcast destination")
                result=evaluate(a[2])
            elif op==169:
                result=evaluate(a[3] if evaluate(a[2]) else a[4])
            elif op==168 and t==20:
                result=int(not evaluate(a[2]))
            elif op in (128,132,199) and t==21 and ta[1:]==(32,0):
                lhs,rhs=evaluate(a[2]),evaluate(a[3])
                result=((lhs+rhs) if op==128 else (lhs*rhs) if op==132 else (lhs&rhs)) & 0xffffffff
            elif op in (170,171,174) and t==20:
                lhs,rhs=evaluate(a[2]),evaluate(a[3])
                result=int(lhs==rhs if op==170 else lhs!=rhs if op==171 else lhs>=rhs)
            else:
                raise ValueError("unsupported live opcode %s" % op)
            active.remove(ident)
            memo[ident]=result
            return result
        return evaluate(self.sink)

VAL=os.environ.get("PROSPER_SPIRV_VAL_BIN") or shutil.which("spirv-val")
if not VAL and os.environ.get("VULKAN_SDK"):
    candidate=Path(os.environ["VULKAN_SDK"])/"Bin/spirv-val.exe"
    if candidate.is_file():
        VAL=str(candidate)
check(bool(VAL), "strict SPIR-V validator available; missing validation is not a pass")
def run(args, tap=False, width=None):
    env={k:v for k,v in os.environ.items() if not k.startswith("PROSPER_")}
    if tap:
        env["PROSPER_FS_TAP"]="7:3" # after actual VCNDMASK destination v1
    if width is not None:
        env["PROSPER_GPU_REPLAY_FRAGMENT_WAVE_SIZE"]=width
    return subprocess.run([REPLAY]+args,env=env,capture_output=True,text=True,timeout=120)
def args(capture,path,route):
    if route=="retry-failed-stage":
        return ["--retry-failed-stage","0:0","--retry-failed-stage-spv",str(path),str(capture)]
    prefix=["--inspect-only"] + (["--recompile-raw"] if route=="recompile-raw" else [])
    return prefix+["--dump-shader","7:fs",str(path),str(capture)]

scratch=Path(os.environ.get("PROSPER_TEST_SCRATCH_DIR") or Path.cwd()/"prosper-test-scratch")
scratch.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix="fragment-mode-",dir=scratch) as directory:
    directory=Path(directory)
    env={k:v for k,v in os.environ.items() if not k.startswith("PROSPER_")}
    env["PROSPER_TEST_SCRATCH_DIR"]=str(directory)
    fixture=subprocess.run([FIXTURE,"--write-fixture",str(directory)],env=env,capture_output=True,text=True,timeout=120)
    check(fixture.returncode==0,"actual realization/collector fixture succeeds without GPU")
    if fixture.returncode:
        print(fixture.stdout+fixture.stderr)
    states=[("mode0",0,(0,0),0),("mode16",16,(0,0),16<<12),("unknown",None,None,None)]
    states.extend(("mode16-i%d-d%d" % (ieee,dx10),16,(ieee,dx10),(16<<12)|(ieee<<23)|(dx10<<21))
                  for ieee in (0,1) for dx10 in (0,1))
    states.extend([("mode16-raw-a",16,(0,0),(16<<12)|(1<<29)),
                   ("mode16-raw-b",16,(0,0),(16<<12)|(1<<29)|(1<<22))])
    for name in ("mode16","mode16-failed"):
        data=(directory/(name+".prgcap")).read_bytes()
        launch=b"\x01\x00\x00\x01"+struct.pack("<I",16<<12)
        flags_tail=(struct.pack("<I",1)+launch+struct.pack("<I",0) if name=="mode16" else
                    struct.pack("<II",0,1)+launch)
        check(struct.unpack_from("<I",data,8)[0]==68 and data.endswith(flags_tail+bytes(4)),
              name+" exact official67 flags followed by resource-free nested68 count")
        data=bytearray(data[:-4]); struct.pack_into("<I",data,8,67)
        check(data.endswith(flags_tail),name+" genuine independent known-clear-flags v67 tail")
        data=bytearray(data[:-len(flags_tail)])
        struct.pack_into("<I",data,8,66)
        (directory/("official66"+("-failed" if name.endswith("failed") else "")+".prgcap")).write_bytes(data)
        transport_tail=(struct.pack("<I",1)+b"\x00"+struct.pack("<II",0,0) if name=="mode16" else
                        struct.pack("<III",0,0,1)+b"\x00"+struct.pack("<I",1)+b"\x00")
        check(struct.unpack_from("<I",data,8)[0]==66 and data.endswith(transport_tail),
              name+" genuine official unknown-transport v66 tail")
        data=bytearray(data[:-len(transport_tail)])
        struct.pack_into("<I",data,8,65)
        expected_tail=(struct.pack("<I",1)+b"\x01\x10"+struct.pack("<I",0) if name=="mode16" else
                       struct.pack("<I",0)+struct.pack("<I",1)+b"\x01\x10")
        combined=(struct.unpack_from("<I",data,8)[0]==65 and
                  data.endswith(expected_tail+struct.pack("<I",0)))
        check(combined,name+" combined v65 retains canonical mode before zero owned obligations")
        if not combined:
            continue
        official=bytearray(data[:-4])
        struct.pack_into("<I",official,8,64)
        check(official.endswith(expected_tail),name+" genuine official v64 producing mode tail")
        (directory/("official64"+("-failed" if name.endswith("failed") else "")+".prgcap")).write_bytes(official)
        legacy=bytearray(official[:-len(expected_tail)])
        struct.pack_into("<I",legacy,8,63)
        (directory/("legacy"+("-failed" if name.endswith("failed") else "")+".prgcap")).write_bytes(legacy)
    states.extend([("official66",16,None,None),("official64",16,None,None),("legacy",None,None,None)])
    for state,mode,flags,raw_word in states:
        for route in ("recompile-raw","fs-tap","retry-failed-stage"):
            capture=directory/(state+("-failed" if route=="retry-failed-stage" else "")+".prgcap")
            for width in (None,"32","64"):
                path=directory/(state+"-"+route+"-"+str(width)+".spv")
                done=run(args(capture,path,route),route=="fs-tap",width)
                label="%s %s width=%s" % (state,route,width)
                check(done.returncode==0 and path.is_file(),label+" actual CPU compile route succeeds")
                if not path.is_file():
                    print(done.stdout+done.stderr)
                    continue
                if VAL:
                    valid=subprocess.run([VAL,"--target-env","vulkan1.1",str(path)],capture_output=True,text=True,timeout=30)
                    check(valid.returncode==0,label+" strict SPIR-V valid")
                    if valid.returncode:
                        print(valid.stdout+valid.stderr)
                module=Module(path.read_bytes())
                if flags is not None:
                    check("fragment-float-flags ieee-mode=%d dx10-clamp=%d source=captured" % flags in done.stderr,
                          label+" actual independent producing flags are visible")
                else:
                    provenance="captured" if state=="unknown" else "legacy-unknown"
                    check("fragment-float-flags=unavailable source="+provenance in done.stderr,
                          label+" missing flags remain visibly unknown, never known-clear")
                if raw_word is not None:
                    check("fragment-rsrc1-ps=0x%08x source=captured" % raw_word in done.stderr,
                          label+" exact observed full word remains visible evidence, not inferred policy")
                else:
                    provenance="captured" if state=="unknown" else "legacy-unknown"
                    check("fragment-rsrc1-ps=unavailable source="+provenance in done.stderr,
                          label+" missing raw word stays unavailable independently of known MODE")
                if mode is None:
                    check("fragment-float-mode=unavailable" in done.stderr and
                          "FLOAT_MODE=unavailable" in done.stderr and
                          "host-dependent legacy FP path" in done.stderr and
                          "exact guest semantics unverified" in done.stderr and 180 in module.ops,
                          label+" visibly unknown and actual eligible live FP fallback, no guessed integer proof")
                else:
                    check("fragment-float-mode=0x%02x source=captured" % mode in done.stderr and
                          "FLOAT_MODE=unavailable" not in done.stderr and 180 not in module.ops,
                          label+" exact producing mode reported and consumed, independent of width override")
                    try:
                        actual=module.output()
                        # Independent category semantics: the raw word1 is a nonzero subnormal.
                        # Flush-input becomes zero/equal; preserve-input remains nonzero/not equal.
                        red=0x3f800000 if mode==0 else 0
                        expected=(red,0,0,0) if route=="fs-tap" else (red,0,0,0x3f800000)
                        check(actual==expected,label+" actual live MRT0 matches independent subnormal category semantics")
                    except ValueError as error:
                        check(False,label+" fail-closed output oracle: "+str(error))
    original=(directory/"mode16.prgcap").read_bytes()
    # MODE precedes v65 owners, v66 transport, v67 launch flags and zero nested68 count.
    transport_tail_size=13
    flags_tail_size=16
    flags_start=len(original)-4-flags_tail_size
    start=flags_start-transport_tail_size-4-10
    malformed={
        "count":original[:start]+struct.pack("<I",0)+original[start+4:],
        "tag":original[:start+4]+b"\x02"+original[start+5:],
        "unknown-value":original[:start+4]+b"\x00\x10"+original[start+6:],
        "failure-count":original[:start+6]+struct.pack("<I",1)+original[start+10:],
        "truncated-mode":original[:start+9],
        "truncated-owned":original[:flags_start-transport_tail_size-1],
        "truncated-transport":original[:flags_start-1],
        "truncated-flags":original[:-5],
        "truncated-nested":original[:-1],
        "flags-count":original[:flags_start]+struct.pack("<I",0)+original[flags_start+4:],
        "flags-tag":original[:flags_start+4]+b"\x02"+original[flags_start+5:],
        "flags-unknown-value":original[:flags_start+4]+b"\x00\x01\x00"+original[flags_start+7:],
        "raw-tag":original[:flags_start+7]+b"\x02"+original[flags_start+8:],
        "raw-unknown-value":original[:flags_start+7]+b"\x00"+struct.pack("<I",1)+original[flags_start+12:],
        "trailing":original+b"\x00",
        "relabel-only63":original[:8]+struct.pack("<I",63)+original[12:],
    }
    for name,data in malformed.items():
        check(data != original,name+" corruption changes the actual capture")
        capture=directory/(name+".prgcap"); capture.write_bytes(data)
        for route in ("recompile-raw","fs-tap"):
            path=directory/"sentinel.spv"; path.write_bytes(b"unchanged")
            done=run(args(capture,path,route),route=="fs-tap")
            check(done.returncode==2 and "fragment-float-mode=" not in done.stderr and
                  "fragment-float-flags" not in done.stderr and
                  "fragment-rsrc1-ps" not in done.stderr and
                  path.read_bytes()==b"unchanged",name+" "+route+" rejects before regeneration/dump")
print("== %s (%d failures) ==" % ("FAIL" if failures else "PASS",len(failures)))
sys.exit(1 if failures else 0)
