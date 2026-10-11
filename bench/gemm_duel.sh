#!/bin/bash
# The fp32 GEMM duel (bench/gemm_fp32.cu): builds the contenders and runs them
# size by size in pairs (cucc and OpenMP back to back, alternating which goes
# first), with a pause before each run so both start from a similar thermal
# state. The GPU (cuBLAS) runs once at the start if nvcc is available.
#
#   bench/gemm_duel.sh [sizes...]          default: 1536 3072 4608 6144
#   PAUSE=25 PAIRS=2 bench/gemm_duel.sh    pause in seconds, pairs per size
#
# The cucc contender runs on the device the driver (cpusimd) or the
# configuration defines; OpenMP uses OMP_NUM_THREADS threads (default: all).
# Output: one line per run (contender, n, kernel ms, total ms, GFLOPS, check),
# then the mean per size with the OpenMP / cucc time ratios.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
OUT=${BENCH_OUT:-$ROOT/build/bench}
PAUSE=${PAUSE:-25}
PAIRS=${PAIRS:-2}
SIZES=${*:-"1536 3072 4608 6144"}
mkdir -p "$OUT"

state() {  # temperature and clock of the device, when the driver is running
    "$ROOT/build/cpu-smi" --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader 2>/dev/null |
        sed 's/^/device temperature, clock: /' || true
}

echo "building..."
"$ROOT/build/cucc" -O3 -std=c++20 -fopenmp "$HERE/gemm_fp32.cu" -o "$OUT/gemm_cpu"
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
if [ -x "$NVCC" ]; then
    "$NVCC" -O3 -std=c++20 -arch=native -Xcompiler -fopenmp "$HERE/gemm_fp32.cu" -lcublas -o "$OUT/gemm_gpu" 2>/dev/null
    "$OUT/gemm_gpu" cuda $SIZES
fi

results=$(mktemp)
trap 'rm -f "$results"' EXIT
run() {  # contender n
    sleep "$PAUSE"
    "$OUT/gemm_cpu" "$1" "$2" | tail -1 | sed "s/^/$1 /" | tee -a "$results"
}
state
echo "== CPU, $PAIRS pairs per size, ${PAUSE}s pause before each run"
echo "contender  n  kernel(ms)  total(ms)  GFLOPS  check"
turn=0
for n in $SIZES; do
    for _ in $(seq "$PAIRS"); do
        if [ $((turn % 2)) = 0 ]; then run cuda "$n"; run omp "$n"; else run omp "$n"; run cuda "$n"; fi
        turn=$((turn + 1))
    done
done
state

echo "== mean per size (cuda = cucc)"
awk '{k[$1" "$2]+=$3; t[$1" "$2]+=$4; c[$1" "$2]++; if ($6 != "ok") bad=1; seen[$2]=1}
     END {
       printf "%6s  %12s %12s  %12s  %10s %10s\n", "n", "cucc kernel", "cucc total", "omp total", "kernel x", "total x"
       for (n in seen) {
         ck=k["cuda "n]/c["cuda "n]; ct=t["cuda "n]/c["cuda "n]; ot=t["omp "n]/c["omp "n]
         printf "%6d  %12.2f %12.2f  %12.2f  %10.2f %10.2f\n", n, ck, ct, ot, ot/ck, ot/ct
       }
       if (bad) print "SOME CHECK FAILED"
     }' "$results" | sort -n
