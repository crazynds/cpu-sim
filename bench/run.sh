#!/bin/sh
# Builds the kernel benchmarks for the GPU (nvcc) and the CPU (cucc) and runs them.
#   bench/run.sh [gpu|cpu|both|native] [bench names...]
# native: the plain C++ versions in bench/native, single core (1 thread) and
# multi core ($NATIVE_THREADS, default 28 like the simulator's workers).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
OUT=${BENCH_OUT:-$ROOT/build/bench}
WHICH=${1:-both}
[ $# -gt 0 ] && shift
NAMES=${*:-"matmul mandelbrot transpose heat2d scan bitonic_sort"}
mkdir -p "$OUT"
if [ "$WHICH" = native ]; then
    for n in $NAMES; do
        g++ -O3 -march=native -ffp-contract=off -fopenmp -std=c++20 "$HERE/native/$n.cpp" -o "$OUT/$n.native"
        for t in 1 "${NATIVE_THREADS:-28}"; do
            OMP_NUM_THREADS=$t "$OUT/$n.native" > "$OUT/$n.native$t.txt"
            sed -n 's/^NATIVE /CPU-C++ /p' "$OUT/$n.native$t.txt"
            if [ -f "$OUT/$n.gpu.txt" ]; then
                grep '^CHECK' "$OUT/$n.native$t.txt" | while read -r _ name hash; do
                    want=$(grep "^CHECK $name " "$OUT/$n.gpu.txt" | awk '{print $3}')
                    if [ "$want" = "$hash" ]; then echo "OK   $name ($t threads)"; else echo "DIFF $name ($t threads): gpu $want, c++ $hash"; fi
                done
            fi
        done
    done
    exit 0
fi
for n in $NAMES; do
    if [ "$WHICH" != cpu ]; then
        /usr/local/cuda/bin/nvcc -O3 -std=c++20 -fmad=false -arch=native -I "$ROOT/samples/common" \
            "$HERE/$n.cu" -o "$OUT/$n.gpu" 2>/dev/null
        "$OUT/$n.gpu" > "$OUT/$n.gpu.txt"
        sed -n 's/^BENCH /GPU /p' "$OUT/$n.gpu.txt"
    fi
    if [ "$WHICH" != gpu ]; then
        "$ROOT/build/cucc" -std=c++20 -fmad=false -I "$ROOT/samples/common" "$HERE/$n.cu" -o "$OUT/$n.cpu"
        "$OUT/$n.cpu" > "$OUT/$n.cpu.txt"  # on the driver (cpusimd) when it is running
        sed -n 's/^BENCH /CPU /p' "$OUT/$n.cpu.txt"
        # Results must match the GPU's bit for bit.
        if [ -f "$OUT/$n.gpu.txt" ]; then
            grep '^CHECK' "$OUT/$n.gpu.txt" | while read -r _ name hash; do
                got=$(grep "^CHECK $name " "$OUT/$n.cpu.txt" | awk '{print $3}')
                if [ "$got" = "$hash" ]; then echo "OK   $name"; else echo "DIFF $name (gpu $hash, cpu $got)"; fi
            done
        fi
    fi
done
