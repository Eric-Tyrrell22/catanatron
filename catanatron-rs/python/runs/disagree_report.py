import csv, sys
from collections import defaultdict
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t")]
def table(key, title):
    agg = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        a = agg[key(r)]; a[0] += 1; a[1] += int(r["agree"]); a[2] += float(r["regret"])
    tot = sum(a[2] for a in agg.values())
    print(f"\n{title:14s} {'decisions':>9s} {'agree':>6s} {'regret/dec':>10s} {'share of regret':>15s}")
    for k, a in sorted(agg.items(), key=lambda x: -x[1][2]):
        print(f"{k:14s} {a[0]:9d} {a[1]/a[0]:6.0%} {a[2]/a[0]:10.3f} {a[2]/tot:15.0%}")
print(f"{len(rows)} decisions over {len({r['game'] for r in rows})} games; overall agreement {sum(int(r['agree']) for r in rows)/len(rows):.0%}, total regret/game {sum(float(r['regret']) for r in rows)/len({r['game'] for r in rows}):.2f}")
table(lambda r: r["phase"], "phase")
table(lambda r: r["kind"], "move type")
table(lambda r: f"{r['phase']}/{r['kind']}", "phase/type")
