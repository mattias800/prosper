#!/bin/bash
# Did the giant submit contain the warmup fence packets (ReleaseMem with hi-addr 0x11)?
L=/root/d232f.log
n=$(grep -n "SubmitDcb #174" "$L" | head -1 | cut -d: -f1)
echo "giant submit at line $n"
# packet dump follows the submit line; count pkt kinds and hi-addr pl1 values within the next 13000 lines
sed -n "$((n+1)),$((n+13000))p" "$L" | grep "^\[agc\]   pkt" > /tmp/pkts174.txt
wc -l /tmp/pkts174.txt
echo "=== kinds ==="
awk "{for(i=1;i<=NF;i++) if(\$i ~ /^kind=/) print \$i}" /tmp/pkts174.txt | sort | uniq -c | sort -rn
echo "=== ReleaseMem pl1 (addr hi) values ==="
grep "kind=ReleaseMem" /tmp/pkts174.txt | awk "{for(i=1;i<=NF;i++) if(\$i ~ /^pl1=/) print \$i}" | sort | uniq -c
echo "=== WriteData pl2 (addr hi) values ==="
grep "kind=WriteData" /tmp/pkts174.txt | awk "{for(i=1;i<=NF;i++) if(\$i ~ /^pl2=/) print \$i}" | sort | uniq -c | head
echo "=== does any pkt carry pl>=0x80f0 (warmup label lo bytes)? ==="
grep -c "80f0" /tmp/pkts174.txt
