#!/bin/sh
# Starts a private driver, runs the runtime test through it and checks cpu-smi.
#   daemon_test.sh <cpusimd> <test_runtime> <cpu-smi>
set -eu
CPUSIMD=$1 TEST=$2 SMI=$3
DIR=$(mktemp -d)
export CPUSIM_SOCKET="$DIR/driver.sock" CPUSIM_VRAM=1GiB CPUSIM_CORES=2 CPUSIM_CONFIG="$DIR/none.toml"
trap '"$CPUSIMD" --stop >/dev/null 2>&1 || true; rm -rf "$DIR"' EXIT

"$CPUSIMD" --daemon >/dev/null
CPUSIM_BACKEND=daemon "$TEST"
"$SMI" >/dev/null
"$SMI" -q | grep -q "Kernels Launched"
test "$("$SMI" --query-gpu=memory.total --format=csv,noheader,nounits)" = 1024
test "$("$SMI" --query-gpu=memory.used --format=csv,noheader,nounits)" = 0
echo "daemon backend: PASS"
