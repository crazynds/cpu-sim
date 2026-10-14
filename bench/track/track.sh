#!/bin/bash
# Performance tracking: compiles and runs every program of apps.txt with one
# build of the simulator, on that build's daemon, and writes one CSV line per
# program: name, compile seconds, binary KB, ok/FAIL, kernel ns (the sum of
# the program's kernel times, CPUSIM_KERNEL_TIMES; the least of REPS runs),
# wall seconds (all runs).
#   track.sh <build dir> <label> [apps.txt]     (HECBENCH: the HeCBench src dir)
# Results in results/<label>.csv; compare two with compare.py.
set -u
BUILD=$(realpath "$1"); LABEL=$2; APPS=${3:-$(dirname "$0")/apps.txt}
SRC=${HECBENCH:-$HOME/fire/src}
OUT=$(dirname "$(realpath "$0")")/results; mkdir -p "$OUT"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
CSV=$OUT/$LABEL.csv
echo "app,compile_s,size_kb,status,kernel_ns,wall_s" > "$CSV"
# This build's daemon.
"$BUILD/cpusimd" --stop >/dev/null 2>&1; pkill -x cpusimd; sleep 1
(cd "$BUILD" && setsid nohup numactl --interleave=all ./cpusimd --daemon >/dev/null 2>&1 < /dev/null &)
sleep 2
grep -v '^#' "$APPS" | while read -r app args; do
  [ -z "$app" ] && continue
  rm -rf "$WORK/$app"; cp -r "$SRC/$app-cuda" "$WORK/$app"; [ -d "$SRC/include" ] && cp -r "$SRC/include" "$WORK/include"
  cd "$WORK/$app" || continue
  [ -z "$args" ] && args=$(grep -A1 "^run:" Makefile | tail -1 | sed 's/.*\.\/$(program) *//')
  make -s clean >/dev/null 2>&1
  t0=$(date +%s.%N)
  timeout 900 make -s CC="$BUILD/cucc" ARCH=sm_70 >build.log 2>&1
  t1=$(date +%s.%N)
  if [ ! -x main ]; then echo "$app,$(echo "$t1 - $t0" | bc),0,BUILD,0,0" >> "$CSV"; continue; fi
  size=$(( $(stat -c %s main) / 1024 ))
  # REPS runs (3 by default), the least kernel time: one run of a short or
  # memory-bound program varies by 20% or more.
  ns=; status=ok
  t2=$(date +%s.%N)
  for r in $(seq "${REPS:-3}"); do
    CPUSIM_KERNEL_TIMES=1 timeout 120 ./main $args > run.log 2>&1; rc=$?
    { [ $rc -ne 0 ] || grep -qE "FAIL|INCORRECT|[Mm]ismatch" run.log; } && status=FAIL
    n=$(sed -n 's/^cpusim-times-total //p' run.log)
    [ -n "$n" ] && { [ -z "$ns" ] || [ "$n" -lt "$ns" ]; } && ns=$n
  done
  t3=$(date +%s.%N)
  printf "%s,%.1f,%d,%s,%s,%.2f\n" "$app" "$(echo "$t1 - $t0" | bc)" "$size" "$status" "${ns:-0}" "$(echo "$t3 - $t2" | bc)" >> "$CSV"
done
echo "results: $CSV"
