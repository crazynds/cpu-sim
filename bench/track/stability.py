#!/usr/bin/env python3
"""How stable the width selection is: for oracle.sh runs whose modes are the
automatic selection on fresh daemons (MODES="auto auto#2 auto#3 ...",
REPS=1), how many different widths each kernel and block shape got.
   stability.py <label>
Prints the kernels that got more than one, then the counts."""
import collections, glob, re, sys

label = sys.argv[1]
here = sys.path[0]
choices = collections.defaultdict(list)  # (kernel, block) -> widths, one per daemon
files = sorted(glob.glob("%s/results/%s.oracle.auto*.widths" % (here, label)))
for path in files:
    for line in open(path):
        m = re.search(r"width of (\S+), block (\S+),.*-> (\S+) lanes", line)
        if m:
            choices[(m.group(1), m.group(2))].append(m.group(3))
unstable = 0
for (kernel, block), w in sorted(choices.items()):
    if len(set(w)) > 1:
        unstable += 1
        print("%-60s %-10s %s" % (kernel[:60], block, " ".join(w)))
print("\n%d daemons; %d kernel shapes, %d with more than one width" % (len(files), len(choices), unstable))
