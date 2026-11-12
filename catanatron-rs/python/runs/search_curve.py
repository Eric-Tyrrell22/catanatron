"""Search curve for the first pick on the expert boards: scores search_pick's
output (root visits per node at several simulation budgets) against the forced
win rates in runs/expert_picks2/forced (8 labelled nodes per board) and the
expert's pick. Per budget: mean regret of the argmax (points of forced win
rate below the best labelled node), agreement with the expert, and how many
argmaxes fell on an unlabelled node (excluded from regret, reported).
The policy prior's own argmax (regret_pol in the value track) is the zero-search
reference.

    runs/search_curve.py <search_pick output> [<search_pick output> ...]
"""
import collections
import glob
import os
import re
import sys

forced = collections.defaultdict(dict)
for fn in glob.glob("runs/expert_picks2/forced/*.txt"):
    for l in open(fn):
        m = re.search(r"board (\d+) forced node\s+(\d+): A \(first seat\) wins ([\d.]+)%", l)
        if m:
            forced[int(m.group(1))][int(m.group(2))] = float(m.group(3))
experts = {}
for fn in glob.glob("runs/expert_picks/*.txt"):
    for l in open(fn):
        if l.startswith("#policy expert"):
            items = l.split()[2:]
            experts[int(os.path.basename(fn)[:-4])] = int(max(items, key=lambda it: float(it.split(":")[1])).split(":")[0])

for path in sys.argv[1:]:
    by_sims = collections.defaultdict(list)
    for l in open(path):
        m = re.match(r"board (\d+) sims (\d+): (.*)", l)
        if not m:
            continue
        board, sims = int(m.group(1)), int(m.group(2))
        picks = [(int(a), int(b)) for a, b in (x.split(":") for x in m.group(3).split())]
        by_sims[sims].append((board, picks))
    print(f"== {os.path.basename(path)}")
    print(f"{'sims':>6} {'boards':>6} {'regret':>7} {'agree':>6} {'unlabelled':>10} {'top share':>9}  regret (0-10 pts)")
    for sims in sorted(by_sims):
        regret, agree, unl, share = [], 0, 0, []
        for board, picks in by_sims[sims]:
            node, v = picks[0]
            total = sum(x for _, x in picks)
            share.append(v / total if total else 0)
            agree += experts.get(board) == node
            fw = forced.get(board, {})
            if not fw:
                continue
            if node in fw:
                regret.append(max(fw.values()) - fw[node])
            else:
                unl += 1
        r = sum(regret) / max(1, len(regret))
        print(f"{sims:>6} {len(by_sims[sims]):>6} {r:>6.1f}p {agree:>6} {unl:>10} {100 * sum(share) / len(share):>8.0f}%  {'#' * int(round(r * 3))}")
    print()
