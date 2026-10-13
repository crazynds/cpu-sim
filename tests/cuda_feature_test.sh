#!/bin/sh
# Compiles a self-checking CUDA test with cucc and runs it in every execution
# mode that can change the result: each lane width the runtime may pick
# (forced with CPUSIM_LANES; widths that do not apply to a kernel fall back to
# 32), the generic module without JIT, and the daemon backend.
#   cuda_feature_test.sh <cucc> <cpusimd> <test.cu> [cucc flags...]
# The program prints "<name>: PASS" or "<name>: FAIL" and exits non-zero on failure.
# FEATURE_ENVS="VAR=value ...": one more in-process run with each setting.
set -eu
CUCC=$1 CPUSIMD=$2 SRC=$3
shift 3
DIR=$(mktemp -d)
NAME=$(basename "$SRC" .cu)
export CPUSIM_SOCKET="$DIR/driver.sock" CPUSIM_VRAM=2GiB CPUSIM_CONFIG="$DIR/none.toml"
trap '"$CPUSIMD" --stop >/dev/null 2>&1 || true; rm -rf "$DIR"' EXIT

"$CUCC" -std=c++20 -I "$(dirname "$SRC")" "$@" "$SRC" -o "$DIR/$NAME"

for lanes in auto 32 32r 64 64s 128 128s 128c 256 256c 512; do
    echo "== in-process, lanes $lanes"
    if [ "$lanes" = auto ]; then
        CPUSIM_BACKEND=inprocess "$DIR/$NAME"
    else
        CPUSIM_BACKEND=inprocess CPUSIM_LANES=$lanes "$DIR/$NAME"
    fi
done

for setting in ${FEATURE_ENVS:-}; do
    echo "== in-process, $setting"
    env CPUSIM_BACKEND=inprocess "$setting" "$DIR/$NAME"
done

echo "== in-process, generic module (no JIT)"
CPUSIM_BACKEND=inprocess CPUSIM_JIT=0 "$DIR/$NAME"

echo "== daemon backend"
"$CPUSIMD" --daemon >/dev/null
CPUSIM_BACKEND=daemon "$DIR/$NAME"
