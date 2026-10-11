#!/bin/sh
# Reproducibility: compiles a CUDA test twice (the binaries must be identical)
# and runs it in every execution mode, repeating the automatic one; every
# "DIGEST <kernel> <hash>" line must be the same in all runs.
#   cuda_repro_test.sh <cucc> <cpusimd> <test.cu> [cucc flags...]
set -eu
CUCC=$1 CPUSIMD=$2 SRC=$3
shift 3
DIR=$(mktemp -d)
NAME=$(basename "$SRC" .cu)
export CPUSIM_SOCKET="$DIR/driver.sock" CPUSIM_VRAM=2GiB CPUSIM_CONFIG="$DIR/none.toml"
trap '"$CPUSIMD" --stop >/dev/null 2>&1 || true; rm -rf "$DIR"' EXIT

"$CUCC" -std=c++20 -I "$(dirname "$SRC")" "$@" "$SRC" -o "$DIR/$NAME"
"$CUCC" -std=c++20 -I "$(dirname "$SRC")" "$@" "$SRC" -o "$DIR/$NAME.again"
if cmp -s "$DIR/$NAME" "$DIR/$NAME.again"; then
    echo "compiler determinism: PASS"
else
    echo "compiler determinism: FAIL (two compilations of $SRC differ)"
fi

# run <label> [VAR=value...]: runs the test, keeps its digests.
run() {
    label=$1
    shift
    echo "== $label"
    env CPUSIM_BACKEND=inprocess "$@" "$DIR/$NAME" | tee "$DIR/out"
    grep '^DIGEST ' "$DIR/out" > "$DIR/digests.$label"
}

run auto
run auto2
run auto3
for lanes in 32 64 64s 128 128s 128c 256 256c 512; do run "lanes-$lanes" CPUSIM_LANES=$lanes; done
run no-jit CPUSIM_JIT=0
run one-core CPUSIM_CORES=1
run three-cores CPUSIM_CORES=3
run linear-order CPUSIM_BLOCK_ORDER=linear
run no-barrier-pass CPUSIM_NO_BARRIER_PASS=1
run tiny-trials CPUSIM_TRIAL_BLOCKS=1

echo "== daemon backend"
"$CPUSIMD" --daemon >/dev/null
CPUSIM_BACKEND=daemon "$DIR/$NAME" | tee "$DIR/out"
grep '^DIGEST ' "$DIR/out" > "$DIR/digests.daemon"

status=PASS
[ -s "$DIR/digests.auto" ] || status=FAIL
for f in "$DIR"/digests.*; do
    if ! cmp -s "$DIR/digests.auto" "$f"; then
        echo "digests differ: ${f##*/digests.} vs auto"
        diff "$DIR/digests.auto" "$f" || true
        status=FAIL
    fi
done
echo "$NAME reproducibility: $status"
