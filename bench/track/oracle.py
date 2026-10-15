#!/usr/bin/env python3
"""Reads results/<label>.oracle.tsv (oracle.sh): per kernel, the time of the
automatic width selection against the best forced width.
   oracle.py <label> [--min-ms N]
Prints every kernel where the selection is more than 10% off the best, then
how many kernels fall in each band of auto / best. Kernels whose best time is
under --min-ms (default 1 ms over all launches) are left out: there the
measurement itself costs more than any width saves, and noise decides."""
import csv, collections, sys

label = sys.argv[1]
min_ms = float(sys.argv[sys.argv.index("--min-ms") + 1]) if "--min-ms" in sys.argv else 1.0
path = label if label.endswith(".tsv") else "%s/results/%s.oracle.tsv" % (sys.path[0], label)
t = collections.defaultdict(dict)  # (app, kernel) -> mode -> ns
for r in csv.DictReader(open(path), delimiter="\t"):
    t[(r["app"], r["kernel"])][r["mode"]] = int(r["ns"])

bands = [(0, 1.05, "within 5%"), (1.05, 1.1, "5-10%"), (1.1, 1.25, "10-25%"), (1.25, 1.5, "25-50%"),
         (1.5, 2, "1.5-2x"), (2, 1e9, "over 2x")]
count = collections.Counter()
worst = []
for (app, kernel), m in sorted(t.items()):
    autos = [v for k, v in m.items() if k.split("#")[0] == "auto"]
    if not autos:
        continue
    auto = min(autos)  # (the automatic selection may run several times: auto#2, ...)
    forced = {k: v for k, v in m.items() if k.split("#")[0] != "auto"}
    if not forced:
        continue
    best_mode = min(forced, key=forced.get)
    best = forced[best_mode]
    if best < min_ms * 1e6:
        continue
    r = auto / best
    for lo, hi, name in bands:
        if lo <= r < hi:
            count[name] += 1
    if r > 1.1:
        worst.append((r, app, kernel[:60], best_mode, best / 1e6, auto / 1e6))
for r, app, kernel, mode, b, a in sorted(worst, reverse=True):
    print("%5.2fx  %-18s %-60s best %-5s %9.2f ms  auto %9.2f ms" % (r, app, kernel, mode, b, a))
n = sum(count.values())
print("\n%d kernels (best time >= %g ms):" % (n, min_ms))
for _, _, name in bands:
    print("  auto vs best %-10s %3d" % (name, count[name]))
