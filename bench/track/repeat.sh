#!/bin/bash
# Runs HeCBench programs again and again on one build's daemon, to catch
# failures that come now and then: counts the failing runs and keeps each
# failing run's output.
#   repeat.sh <build dir> <runs> app...     (HECBENCH: the HeCBench src dir)
# Arguments come from apps.txt when the program is there, else from its
# Makefile's run target. Logs in results/repeat/<app>.<n>.log.
set -u
BUILD=$(realpath "$1"); RUNS=$2; shift 2
SRC=${HECBENCH:-$HOME/fire/src}
HERE=$(dirname "$(realpath "$0")")
OUT=$HERE/results/repeat; mkdir -p "$OUT"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
export CPUSIM_SOCKET=$WORK/driver.sock
(cd "$BUILD" && setsid nohup numactl --interleave=all ./cpusimd --daemon >/dev/null 2>&1 < /dev/null &)
sleep 2
for app in "$@"; do
  rm -rf "$WORK/$app"; cp -r "$SRC/$app-cuda" "$WORK/$app"; [ -d "$SRC/include" ] && cp -r "$SRC/include" "$WORK/include"
  cd "$WORK/$app" || continue
  make -s clean >/dev/null 2>&1; timeout 900 make -s CC="$BUILD/cucc" ARCH=sm_70 >build.log 2>&1
  [ -x main ] || { echo "$app: does not build"; continue; }
  args=$(awk -v a="$app" '$1 == a { $1 = ""; print }' "$HERE/apps.txt")
  [ -z "$args" ] && args=$(grep -A1 "^run:" Makefile | tail -1 | sed 's/.*\.\/$(program) *//')
  fails=0
  for n in $(seq "$RUNS"); do
    timeout 120 ./main $args > run.log 2>&1; rc=$?
    if [ $rc -ne 0 ] || grep -qE "FAIL|INCORRECT|[Mm]ismatch" run.log; then
      fails=$((fails + 1)); { echo "rc=$rc"; cat run.log; } > "$OUT/$app.$n.log"
    fi
  done
  echo "$app: $fails of $RUNS runs failed"
done
"$BUILD/cpusimd" --stop >/dev/null 2>&1
