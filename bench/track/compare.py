#!/usr/bin/env python3
"""Compares two tracking runs (bench/track/results/<label>.csv):
   compare.py <before> <after>
Kernel time ratio after/before per program (below 1: faster), compile time,
size, and failures; then the geometric means."""
import csv, math, sys


def load(label):
    path = label if label.endswith(".csv") else "%s/results/%s.csv" % (sys.path[0], label)
    return {r["app"]: r for r in csv.DictReader(open(path))}


a, b = load(sys.argv[1]), load(sys.argv[2])
logs = {"kernel": [], "compile": [], "size": []}
print("%-20s %10s %10s %7s %8s %8s %8s  %s" % ("app", "kernel ms", "after", "ratio", "compile", "after", "size MB", "status"))
for app in a:
    if app not in b:
        continue
    x, y = a[app], b[app]
    kx, ky = int(x["kernel_ns"]) / 1e6, int(y["kernel_ns"]) / 1e6
    r = ky / kx if kx > 0 and ky > 0 else float("nan")
    flag = "" if x["status"] == y["status"] == "ok" else "%s -> %s" % (x["status"], y["status"])
    if r == r:
        logs["kernel"].append(math.log(r))
        flag += "  <-- slower" if r > 1.1 else "  <-- faster" if r < 0.9 else ""
    cx, cy = float(x["compile_s"]), float(y["compile_s"])
    if cx > 0 and cy > 0:
        logs["compile"].append(math.log(cy / cx))
    sx, sy = int(x["size_kb"]), int(y["size_kb"])
    if sx > 0 and sy > 0:
        logs["size"].append(math.log(sy / sx))
    print("%-20s %10.2f %10.2f %7.3f %8.1f %8.1f %4d->%-4d %s" % (app, kx, ky, r, cx, cy, sx // 1024, sy // 1024, flag))
for k, v in logs.items():
    if v:
        print("geomean %s ratio: %.3f (%d programs)" % (k, math.exp(sum(v) / len(v)), len(v)))
