#!/usr/bin/env python3
"""Which lane widths win, and what a rule that builds only a few would lose.
   widths.py <label>          (results/<label>.features and .widths, from collect_widths.sh)
For each kernel and block shape measured: the time of every width tried
(us per 1000 blocks), the winner, the kernel's features; then, for each
rule, the slowdown of the best width the rule keeps against the best
overall (geometric mean and worst)."""
import collections
import math
import re
import sys

label = sys.argv[1]
base = "%s/results/%s" % (sys.path[0], label)

features = {}  # kernel -> dict
for line in open(base + ".features"):
    p = line.split()
    if len(p) < 3:
        continue
    kernel = p[2]
    features[kernel] = {k: int(v) for k, v in (x.split("=") for x in p[3:])}

WIDTH = re.compile(r"cpusim: width of (\S+), block (\d+)x(\d+)x(\d+), .*?\)\s*(.*?) -> ")
cases = {}  # (kernel, block) -> {variant: us}
for line in open(base + ".widths"):
    m = WIDTH.search(line)
    if not m:
        continue
    kernel, block = m.group(1), (int(m.group(2)), int(m.group(3)), int(m.group(4)))
    times = {}
    for item in m.group(5).split():
        v, t = item.rsplit(":", 1)
        times[v] = float(t)
    cases[(kernel, block)] = times


def lanes(v):
    return int(re.match(r"\d+", v).group(0))


def rule_all(f, block, variants):
    return variants


def rule_warp_only(f, block, variants):
    return [v for v in variants if v == "32"]


def make_pressure_rule(budget, n):
    """Up to n widths: the widest plain ones whose pressure (live units *
    lanes / 16 zmm) fits the budget, plus the warp entry; with barriers,
    the widest kinds that fit first."""
    def rule(f, block, variants):
        live = f.get("live", 0)
        fits = [v for v in variants if live * lanes(v) / 16 <= budget or v == "32"]
        fits.sort(key=lambda v: (-lanes(v), v))
        keep = ["32"] if "32" in variants else []
        for v in fits:
            if len(keep) >= n:
                break
            if v not in keep:
                keep.append(v)
        return keep
    return rule


rules = {"all": rule_all, "32 only": rule_warp_only}
for budget in (24, 32, 48, 64):
    for n in (2, 3):
        rules["pressure<=%d, %d widths" % (budget, n)] = make_pressure_rule(budget, n)

print("%-60s %-12s %-50s %s" % ("kernel", "block", "us per 1000 blocks", "features"))
wins = collections.Counter()
for (kernel, block), times in sorted(cases.items()):
    best = min(times, key=times.get)
    wins[best] += 1
    f = features.get(kernel, {})
    ts = " ".join("%s:%.0f" % (v, t) for v, t in sorted(times.items(), key=lambda x: (lanes(x[0]), x[0])))
    print("%-60s %-12s %-50s best %-5s live=%s insts=%s bar=%s" % (
        kernel[:60], "%dx%dx%d" % block, ts[:50], best, f.get("live"), f.get("insts"), f.get("barriers")))
print("\nwins by variant:", dict(wins))
print("\n%-28s %10s %10s %8s" % ("rule", "geomean", "worst", "widths"))
for name, rule in rules.items():
    logs, worst, built = [], 1.0, 0
    for (kernel, block), times in cases.items():
        f = features.get(kernel, {})
        keep = [v for v in rule(f, block, list(times)) if v in times] or list(times)
        built += len(keep)
        r = min(times[v] for v in keep) / min(times.values())
        logs.append(math.log(r))
        worst = max(worst, r)
    print("%-28s %10.3f %10.3f %8.2f" % (name, math.exp(sum(logs) / len(logs)), worst, built / len(cases)))
