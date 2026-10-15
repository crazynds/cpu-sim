#!/bin/bash
# GPU (nvcc), cucc and host OpenMP on the same HeCBench programs: build
# times, run times (best of RUNS) and each run's own verdict.
#   run.sh <cucc build dir> <src dir> <out dir> app...
# <src dir> holds <app>-cuda and <app>-omp (unmodified HeCBench). Arguments
# are the ones in each Makefile's run target. One line per app and form in
# <out dir>/results.csv; the programs' output in <out dir>/<app>.<form>.txt.
set -u
BUILD=$(realpath "$1"); SRC=$(realpath "$2"); OUT=$(realpath -m "$3"); shift 3
RUNS=${RUNS:-2}; TMO=${TMO:-120}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}; OMPCC=${OMPCC:-$HOME/hec/clang-omp}
mkdir -p "$OUT"; CSV=$OUT/results.csv
[ -f "$CSV" ] || echo "app,form,build,build_s,size_kb,runs_ok,best_s,verdict" > "$CSV"
# cucc's programs run on a daemon of their own.
export CPUSIM_SOCKET=$OUT/driver.sock
(cd "$BUILD" && setsid nohup numactl --interleave=all ./cpusimd --daemon >/dev/null 2>&1 < /dev/null &)
sleep 2
export OMP_NUM_TEAMS=1 OMP_NUM_THREADS=$(nproc) OMP_PROC_BIND=spread

args_of() { grep -A1 "^run:" Makefile | tail -1 | sed 's/.*\.\/\$(program) *//; s/.*\.\/main *//'; }
verdict() {  # what the program says of its own result
  if grep -qiE "fail|error|mismatch|incorrect|wrong" "$1"; then echo FAIL
  elif grep -qiE "pass|success|correct|verif|match" "$1"; then echo PASS
  else echo "-"; fi
}
for app in "$@"; do
  for form in gpu cucc omp; do
    if [ $form = omp ]; then dir=$SRC/$app-omp; cc=$OMPCC; extra=""
    else dir=$SRC/$app-cuda; extra="ARCH=sm_90"; [ $form = gpu ] && cc=$NVCC || { cc=$BUILD/cucc; extra="ARCH=sm_70"; }; fi
    cd "$dir" || continue
    make -s clean >/dev/null 2>&1; rm -f main
    t0=$(date +%s.%N)
    timeout 1800 make -s CC="$cc" $extra > "$OUT/$app.$form.build.log" 2>&1; rc=$?
    t1=$(date +%s.%N)
    bs=$(echo "$t1 - $t0" | bc)
    if [ $rc != 0 ] || [ ! -x main ]; then
      echo "$app,$form,FAIL,$bs,0,0,," >> "$CSV"; continue
    fi
    size=$(( $(stat -c %s main) / 1024 ))
    args=$(args_of); best=""; ok=0
    for r in $(seq $RUNS); do
      t0=$(date +%s.%N)
      timeout $TMO ./main $args > "$OUT/$app.$form.txt" 2>&1; rc=$?
      t1=$(date +%s.%N)
      [ $rc = 0 ] || { echo "rc=$rc" >> "$OUT/$app.$form.txt"; break; }
      ok=$((ok + 1)); s=$(echo "$t1 - $t0" | bc)
      [ -z "$best" ] || [ $(echo "$s < $best" | bc) = 1 ] && best=$s
    done
    echo "$app,$form,ok,$bs,$size,$ok,$best,$(verdict "$OUT/$app.$form.txt")" >> "$CSV"
    mv main "$OUT/$app.$form.bin" 2>/dev/null
  done
done
CPUSIM_SOCKET=$CPUSIM_SOCKET "$BUILD/cpusimd" --stop >/dev/null 2>&1
