#!/bin/bash
# Data for predicting lane widths: for every program of apps.txt, the
# kernels' features (cpusim-devc, CPUSIM_DEVC_FEATURES) and the times the
# runtime measured for each width (daemon log, CPUSIM_VERBOSE).
#   collect_widths.sh <build dir> <label> [apps.txt]
# Writes results/<label>.features and results/<label>.widths.
set -u
BUILD=$(realpath "$1"); LABEL=$2; APPS=${3:-$(dirname "$0")/apps.txt}
SRC=${HECBENCH:-$HOME/fire/src}
OUT=$(dirname "$(realpath "$0")")/results; mkdir -p "$OUT"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
FEAT=$OUT/$LABEL.features; WID=$OUT/$LABEL.widths; : > "$FEAT"; : > "$WID"
LOG=$WORK/daemon.log
pkill -x cpusimd; sleep 1
(cd "$BUILD" && CPUSIM_VERBOSE=1 setsid nohup numactl --interleave=all ./cpusimd --daemon > "$LOG" 2>&1 < /dev/null &)
sleep 2
grep -v '^#' "$APPS" | while read -r app args; do
  [ -z "$app" ] && continue
  rm -rf "$WORK/$app"; cp -r "$SRC/$app-cuda" "$WORK/$app"; [ -d "$SRC/include" ] && cp -r "$SRC/include" "$WORK/include"
  cd "$WORK/$app" || continue
  [ -z "$args" ] && args=$(grep -A1 "^run:" Makefile | tail -1 | sed 's/.*\.\/$(program) *//')
  make -s clean >/dev/null 2>&1
  CPUSIM_DEVC_FEATURES=1 timeout 900 make -s CC="$BUILD/cucc" ARCH=sm_70 2>&1 | grep '^cpusim-features' | sed "s/^/$app /" >> "$FEAT"
  [ -x main ] || continue
  before=$(wc -l < "$LOG")
  timeout 120 ./main $args > /dev/null 2>&1
  tail -n +$((before + 1)) "$LOG" | grep 'cpusim: width of' | sed "s/^/$app /" >> "$WID"
done
pkill -x cpusimd
echo "features: $(wc -l < "$FEAT"), width measurements: $(wc -l < "$WID")"
