#!/bin/bash
# Print the MAIN thread (GameThread) section from each gwj sample + a few named threads.
for i in 1 2 3; do
  echo "=== SAMPLE $i main ==="
  awk -v pid="$1" '$0 ~ "lwp="pid" " {f=1; print; next} f&&/^== thread/{exit} f{print}' /root/gwj_$i.txt
done
for n in "SlateLoadingThr" "RenderThread 0" "RHIThread" "AgcSubmissionTh" "IoDispatcher" "FMediaTicker" "CriManaDecodeTh" "CRI Server Mana"; do
  echo "=== $n (sample 1) ==="
  awk -v nm="$n" 'index($0,"name="nm)>0 {f=1; print; next} f&&/^== thread/{exit} f{print}' /root/gwj_1.txt
done
