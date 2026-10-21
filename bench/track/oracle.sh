#!/bin/bash
# How close the automatic width selection is to the best width: every program
# of apps.txt compiled once, then run on a daemon of its own per mode, the
# automatic selection and every width forced (CPUSIM_LANES, read by the
# daemon). One line per mode, program and kernel: the least of REPS runs.
#   oracle.sh <build dir> <label> [apps.txt]     (HECBENCH: the HeCBench src dir)
# Results in results/<label>.oracle.tsv; read them with oracle.py. The modes
# of a program run back to back, so that they see the same load.
# (A width a kernel has no variant for falls back to 32: such a line repeats
# the 32-lane one.)
set -u
BUILD=$(realpath "$1"); LABEL=$2; APPS=${3:-$(dirname "$0")/apps.txt}
SRC=${HECBENCH:-$HOME/fire/src}
OUT=$(dirname "$(realpath "$0")")/results; mkdir -p "$OUT"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
TSV=$OUT/$LABEL.oracle.tsv
MODES=${MODES:-auto 32 64 128 256 512 64s 128s 32r}
printf "mode\tapp\tkernel\tlaunches\tns\n" > "$TSV"

declare -A ARGS
while read -r app args; do
  [ -z "$app" ] && continue
  rm -rf "$WORK/$app"; cp -r "$SRC/$app-cuda" "$WORK/$app"; [ -d "$SRC/include" ] && cp -r "$SRC/include" "$WORK/include"
  (cd "$WORK/$app" && make -s clean >/dev/null 2>&1; timeout 900 make -s CC="$BUILD/cucc" ARCH=sm_70 >build.log 2>&1)
  [ -z "$args" ] && args=$(grep -A1 "^run:" "$WORK/$app/Makefile" | tail -1 | sed 's/.*\.\/$(program) *//')
  ARGS[$app]=$args
done < <(grep -v '^#' "$APPS")

export CPUSIM_SOCKET=$WORK/driver.sock
for mode in $MODES; do rm -f "$OUT/$LABEL.oracle.$mode.widths"; done
# Program by program, every mode back to back, each on a fresh driver: the
# machine's load changes over minutes (another user's work), and the modes of
# one program must see the same.
for app in "${!ARGS[@]}"; do
  [ -x "$WORK/$app/main" ] || continue
  for mode in $MODES; do
    # A mode may repeat as <mode>#<n> (auto#2: the automatic selection again).
    setting=${mode%%#*}
    if [ "$setting" = auto ]; then unset CPUSIM_LANES; else export CPUSIM_LANES=$setting; fi
    # The driver's choices (CPUSIM_VERBOSE) go to <socket>.log, kept per mode.
    rm -f "$CPUSIM_SOCKET.log"
    (cd "$BUILD" && CPUSIM_VERBOSE=1 setsid nohup numactl --interleave=all ./cpusimd --daemon >/dev/null 2>&1 < /dev/null &)
    sleep 1
    cd "$WORK/$app"
    for r in $(seq "${REPS:-3}"); do
      CPUSIM_KERNEL_TIMES=1 timeout 120 ./main ${ARGS[$app]} 2>&1 | sed -n 's/^cpusim-times \([^ ]*\) \([0-9]*\) \([0-9]*\)$/\1 \2 \3/p'
    done | sort -k1,1 -k3,3n | awk -v m="$mode" -v a="$app" '$1 != last { printf "%s\t%s\t%s\t%s\t%s\n", m, a, $1, $2, $3; last = $1 }' >> "$TSV"
    "$BUILD/cpusimd" --stop >/dev/null 2>&1; sleep 1
    grep 'cpusim: width of' "$CPUSIM_SOCKET.log" >> "$OUT/$LABEL.oracle.$mode.widths" 2>/dev/null
  done
done
unset CPUSIM_LANES
echo "results: $TSV"
