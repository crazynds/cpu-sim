#!/bin/bash
# The stress suite: every program in tests/stress, built with cucc and run on
# the driver (cpusimd), optionally also built with nvcc and run on the GPU.
#   tests/stress/run.sh [cpu|gpu|both] [programs...]     (default: cpu, all)
# Output in build/stress/: <name>.<cpu|gpu>.log (the program's output),
# <name>.cpu.build.log (the compiler's), and summary.tsv, one line per case:
#   program  case  status  kernel_ms  ref_ms  gpu_ms
# status: PASS, FAIL (wrong result), COMPILE (cucc rejected the program),
# ERROR (a CUDA call failed), CRASH (killed by a signal), TIMEOUT, DAEMON (the
# driver died), MISSING (the case ran on the GPU, but the program stopped
# before it on the CPU).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BUILD=${STRESS_BUILD:-$ROOT/build}  # another build (and CPUSIM_SOCKET for its driver) to compare
OUT=${STRESS_OUT:-$BUILD/stress}
GPU_OUT=$ROOT/build/stress  # the GPU runs, shared by every build compared
WHICH=${1:-cpu}
[ $# -gt 0 ] && shift
TIMEOUT=${STRESS_TIMEOUT:-120}
# The OpenMP references wait passively: by default libgomp's threads keep
# spinning for milliseconds after each parallel region, on every CPU, and the
# kernel timed right after would share its workers' CPUs with them.
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive}
mkdir -p "$OUT" "$GPU_OUT"

if [ $# -gt 0 ]; then
    PROGRAMS=("$@")
else
    PROGRAMS=()
    for f in "$HERE"/*.cu; do PROGRAMS+=("$(basename "$f" .cu)"); done
fi

daemon_alive() { "$BUILD/cpusimd" --status >/dev/null 2>&1; }
ensure_daemon() { daemon_alive || "$BUILD/cpusimd" --daemon >/dev/null; }

# Runs a binary with a timeout; prints the status word for a program that did
# not end normally.
run_one() {  # binary log
    timeout -s KILL "$TIMEOUT" "$1" >"$2" 2>&1
    local rc=$?
    if [ $rc -eq 137 ]; then echo TIMEOUT
    elif [ $rc -gt 128 ]; then echo CRASH
    elif grep -q '^ERROR' "$2"; then echo ERROR
    elif ! grep -q '^DONE' "$2"; then echo CRASH
    else echo OK
    fi
}

if [ "$WHICH" != cpu ]; then
    for p in "${PROGRAMS[@]}"; do
        echo "== gpu $p"
        if ! /usr/local/cuda/bin/nvcc -O3 -std=c++20 -arch=native -Xcompiler -fopenmp,-march=native -I "$HERE" \
            "$HERE/$p.cu" -o "$GPU_OUT/$p.gpu" >"$GPU_OUT/$p.gpu.build.log" 2>&1; then
            echo "   nvcc failed: see $GPU_OUT/$p.gpu.build.log"
            continue
        fi
        st=$(run_one "$GPU_OUT/$p.gpu" "$GPU_OUT/$p.gpu.log")
        echo "   $st, $(grep -c '^CASE .* PASS' "$GPU_OUT/$p.gpu.log") passed, $(grep -c '^CASE .* FAIL' "$GPU_OUT/$p.gpu.log") failed"
    done
fi

[ "$WHICH" = gpu ] && exit 0

SUMMARY=$OUT/summary.tsv
printf 'program\tcase\tstatus\tkernel_ms\tref_ms\tgpu_ms\n' >"$SUMMARY"
ensure_daemon
for p in "${PROGRAMS[@]}"; do
    echo "== cpu $p"
    gpu_log=$GPU_OUT/$p.gpu.log
    cases_from() { [ -f "$1" ] && sed -n 's/^CASE \([^ ]*\) .*/\1/p' "$1"; }
    gpu_ms() { [ -f "$gpu_log" ] && sed -n "s/^CASE $1 [A-Z]* kernel_ms=\([^ ]*\) .*/\1/p" "$gpu_log" | head -1; }
    if ! "$BUILD/cucc" -O3 -std=c++20 -fopenmp -Xcompiler -march=native -I "$HERE" "$HERE/$p.cu" -o "$OUT/$p.cpu" \
        >"$OUT/$p.cpu.build.log" 2>&1; then
        echo "   COMPILE: $(grep -m1 -E 'error|cpusim-devc' "$OUT/$p.cpu.build.log" | cut -c1-200)"
        names=$(cases_from "$gpu_log")
        [ -z "$names" ] && names="(all)"
        for c in $names; do printf '%s\t%s\tCOMPILE\t\t\t%s\n' "$p" "$c" "$(gpu_ms "$c")" >>"$SUMMARY"; done
        continue
    fi
    ensure_daemon
    st=$(run_one "$OUT/$p.cpu" "$OUT/$p.cpu.log")
    if ! daemon_alive; then st=DAEMON; fi
    # Cases that ran.
    while read -r _ c res k r _; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$p" "$c" "$res" "${k#kernel_ms=}" "${r#ref_ms=}" "$(gpu_ms "$c")" >>"$SUMMARY"
    done < <(grep '^CASE ' "$OUT/$p.cpu.log")
    # Cases that did not (the program stopped first).
    if [ "$st" != OK ]; then
        missing=0
        for c in $(cases_from "$gpu_log"); do
            grep -q "^CASE $c " "$OUT/$p.cpu.log" && continue
            printf '%s\t%s\t%s\t\t\t%s\n' "$p" "$c" "$( [ $missing = 0 ] && echo "$st" || echo MISSING)" "$(gpu_ms "$c")" >>"$SUMMARY"
            missing=1
        done
        [ $missing = 0 ] && printf '%s\t%s\t%s\t\t\t\n' "$p" "(program)" "$st" >>"$SUMMARY"
    fi
    echo "   $st, $(grep -c '^CASE .* PASS' "$OUT/$p.cpu.log") passed, $(grep -c '^CASE .* FAIL' "$OUT/$p.cpu.log") failed"
done

echo
echo "Summary: $SUMMARY"
cut -f3 "$SUMMARY" | tail -n +2 | sort | uniq -c
