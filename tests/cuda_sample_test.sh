#!/bin/sh
# Compiles an unmodified CUDA sample with cucc and runs it on both backends.
#   cuda_sample_test.sh <cucc> <cpusimd> <sample.cu> [cucc flags...]
set -eu
CUCC=$1 CPUSIMD=$2 SRC=$3
shift 3
DIR=$(mktemp -d)
NAME=$(basename "$SRC" .cu)
export CPUSIM_SOCKET="$DIR/driver.sock" CPUSIM_VRAM=6GiB CPUSIM_CONFIG="$DIR/none.toml"
trap '"$CPUSIMD" --stop >/dev/null 2>&1 || true; rm -rf "$DIR"' EXIT

"$CUCC" -std=c++20 -I "$(dirname "$SRC")/common" "$@" "$SRC" -o "$DIR/$NAME"

echo "== in-process backend"
CPUSIM_BACKEND=inprocess "$DIR/$NAME"

echo "== daemon backend"
"$CPUSIMD" --daemon >/dev/null
CPUSIM_BACKEND=daemon "$DIR/$NAME"
