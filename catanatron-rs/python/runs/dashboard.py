"""Local training dashboard. Usage: python runs/dashboard.py [port]   (default 8787)

Serves runs/dashboard.html plus JSON endpoints that re-read the live run files on every request:
  /api/runs            runs that have log.jsonl, newest first
  /api/run?name=X      downsampled per-iteration series + vs-Ham points + args for run X
  /api/pig             summary of every runs/pig_*.jsonl benchmark file
  /api/panel_boards?run=X  per-board panel lines of run X + runs/panel100/*.json baselines + sp3 specialist strength
Standard library only.
"""
import json
import math
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

try:  # board drawings on /specialists only; the page omits them when the engine module is unavailable
    import catanatron_rs
except ImportError:
    catanatron_rs = None

RUNS = os.path.dirname(os.path.abspath(__file__))
MAX_POINTS = 2000
SECONDS_PER_SHARD = {"ls": 4.6}  # 100-sim run; every other run is ~400 sims
SECONDS_PER_SHARD_400 = 15.3
GPH_WINDOW = 10  # iterations used for the games/hour rate
SCALARS = ["surprise", "loss_pi", "loss_v", "loss_aux", "loss_aux2", "entropy", "target_entropy",
           "decisions_per_game", "endgame_share", "forks", "remote_shards", "remote_lag",
           "remote_dropped", "remote_games", "luck_per_event", "random_share"]
VS_HAM_RE = re.compile(r"snap_(\d+)\.ctnn:\s*([\d.]+)% vs Ham \(±([\d.]+)\)")
VS_GAZ_RE = re.compile(r"snap_(\d+)\.ctnn:\s*([\d.]+)% vs Gaz \(±([\d.]+)\)")
PANEL_RE = re.compile(r"^\d\d:\d\d (.+?): mean ([\d.]+)% \|(.*)$")
# runs/panel100.txt lines: "HH:MM label: mean P% (±S over N boards, G games each) | min A max B | Ts"
PANEL100_RE = re.compile(r"^\d\d:\d\d (.+?): mean ([\d.]+)% \(±([\d.]+) over (\d+) boards, (\d+) games each\) \| min ([\d.]+) max ([\d.]+)")
PANEL_BOARD_RE = re.compile(r"(\d+):([\d.]+)%")
CONTROL_RE = re.compile(r"^(.+)@(\d+)k$")  # baseline label like ac@50k: run ac at 50k games
SURPRISE_DIST_FIELDS = ("gt20", "gt69", "mean_gt05", "p90")
HEADLINE_WINDOW = 10  # iterations averaged for the headline tiles of per-iteration series
ACTIONS = ["settlement", "road", "city", "buy_dev", "knight", "road_building", "yop", "monopoly",
           "robber", "discard", "trade"]


def list_runs():
    out = []
    for name in os.listdir(RUNS):
        log = os.path.join(RUNS, name, "log.jsonl")
        if os.path.isfile(log):
            st = os.stat(log)
            out.append({"name": name, "mtime": st.st_mtime, "size": st.st_size,
                        "has_vs_ham": os.path.isfile(os.path.join(RUNS, name + ".vs_ham"))})
    out.sort(key=lambda r: -r["mtime"])
    return out


def read_log(name):
    rows = []
    with open(os.path.join(RUNS, name, "log.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line being written right now
                continue
    return rows


def parse_panel(path):
    """[{label, mean, boards: {board: pct}}] for each well-formed line; [] if the file is missing."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            m = PANEL_RE.match(line.strip())
            if m:
                out.append({"label": m.group(1), "mean": float(m.group(2)),
                            "boards": {b: float(p) for b, p in PANEL_BOARD_RE.findall(m.group(3))}})
    return out


def parse_panel100(path):
    """Baselines of the 100-board panel (runs/panel100_score.py output): [{label, mean, se, min, max, boards: {}}]."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            m = PANEL100_RE.match(line.strip())
            if m:
                out.append({"label": m.group(1), "mean": float(m.group(2)), "se": float(m.group(3)), "n_boards": int(m.group(4)),
                            "min": float(m.group(6)), "max": float(m.group(7)), "boards": {}})
    return out


def panel_baselines():
    """Fixed-net lines of the 100-board panel (runs/panel100.txt). Labels like ac@356k also get iter/actor_hours from that run's log."""
    out, logs = [], {}
    for b in parse_panel100(os.path.join(RUNS, "panel100.txt")):
        m = CONTROL_RE.match(b["label"])
        b["control_run"], b["games"], b["iter"], b["actor_hours"] = None, None, None, None
        if m:
            b["control_run"], b["games"] = m.group(1), int(m.group(2)) * 1000
            if os.path.isfile(os.path.join(RUNS, m.group(1), "log.jsonl")):
                if m.group(1) not in logs:
                    logs[m.group(1)] = run_series(m.group(1), downsample=False)["series"]
                s = logs[m.group(1)]
                i = next((i for i, g in enumerate(s.get("games", [])) if g is not None and g >= b["games"]), None)
                if i is not None:
                    b["iter"] = s["iter"][i]
                    b["actor_hours"] = s["actor_hours"][i] if "actor_hours" in s else None
        out.append(b)
    return out


SNAP_RE = re.compile(r"snap_(\d+)$")


def panel_path(name):
    """runs/<name>.panel100 if present, else the old 4-board runs/<name>.panel, else None (same choice as run_series)."""
    return next((os.path.join(RUNS, p) for p in (name + ".panel100", name + ".panel") if os.path.isfile(os.path.join(RUNS, p))), None)


def panel_boards(run, path):
    """Everything the per-board drill-down page needs, straight from the files:
    lines: the run's panel lines [{label, iter (from snap_NNNNNN, None if the label has none), mean, boards}];
    baselines: runs/panel100/*.json as stored; specialists: {seed: {run, outcome, best_iter, best_win}} from runs/sp3/results.jsonl,
    best_* taken as the eval with the highest win (win = the specialist's win rate vs its base net on its board)."""
    lines = []
    for e in parse_panel(path):
        m = SNAP_RE.search(e["label"])
        lines.append({"label": e["label"], "iter": int(m.group(1)) if m else None, "mean": e["mean"], "boards": e["boards"]})
    bdir = os.path.join(RUNS, "panel100")
    baselines = []
    for fn in sorted(os.listdir(bdir)):
        if fn.endswith(".json"):
            with open(os.path.join(bdir, fn)) as f:
                d = json.load(f)
            baselines.append({k: d[k] for k in ("label", "net", "games", "mean", "se", "boards")})
    specialists = {}
    for d in read_jsonl(os.path.join(RUNS, "sp3", "results.jsonl")):
        best = max(d["evals"], key=lambda e: e["win"]) if d["evals"] else None
        specialists[str(d["seed"])] = {"run": d["run"], "outcome": d["outcome"],
                                       "best_iter": best["iter"] if best else None, "best_win": best["win"] if best else None}
    return {"run": run, "panel_file": os.path.basename(path), "panel_kind": 100 if path.endswith(".panel100") else 4,
            "lines": lines, "baselines": baselines, "specialists": specialists}


SPEC_EVAL_RE = re.compile(r"^(\d\d:\d\d) (\w+)-(\d+) it (\d+): ([\d.]+)% vs base")
DRAW_TOP = 12
RUNNING_STALE_S = 900  # a run dir with no results line and no log write for 15 min is an abandoned attempt


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def last_log_row(run):
    p = os.path.join(RUNS, run, "log.jsonl")
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as f:  # only the tail matters
        f.seek(max(0, os.path.getsize(p) - 8192))
        lines = [l for l in f.read().decode(errors="ignore").splitlines() if l.strip()]
    for l in reversed(lines):
        try:
            return json.loads(l)
        except json.JSONDecodeError:
            continue
    return None


def board_layout(seed):
    tiles, _ports = catanatron_rs.board_layout(f"seed:{seed}")
    return [{"q": x, "r": z, "t": terrain, "n": number} for (x, _y, z), terrain, number in tiles]


def specialists_batch(batch, seeds_file=None):
    """One specialist batch: finished boards from <batch>/results.jsonl, the running board(s) from run dirs
    without a results line, and per-eval win rates from <batch>.out (they appear before results.jsonl)."""
    results = read_jsonl(os.path.join(RUNS, batch, "results.jsonl"))
    total = len(json.load(open(os.path.join(RUNS, seeds_file)))) if seeds_file else len(results)
    out_evals, current = {}, None
    out_path = os.path.join(RUNS, batch + ".out")
    if os.path.isfile(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.rstrip()
                if line:
                    current = line
                m = SPEC_EVAL_RE.match(line)
                if m and m.group(2) == batch:
                    out_evals.setdefault(int(m.group(3)), {})[int(m.group(4))] = float(m.group(5)) / 100

    def row(seed, run, evals, status, minutes):
        last = last_log_row(run) or {}
        return {"seed": seed, "run": run, "status": status, "win10": evals.get(10), "win20": evals.get(20),
                "evals": evals, "minutes": minutes if minutes is not None else num(last.get("minutes")),
                "games": num(last.get("games")), "iter": num(last.get("iter"))}

    finished = []
    for r in results:
        ev = {e["iter"]: e["win"] for e in r.get("evals", []) if "iter" in e and "win" in e}
        fr = row(r["seed"], r.get("run", f"{batch}-{r['seed']}"), ev, r.get("outcome", "finished"), num(r.get("minutes")))
        fr["final"] = ev[max(ev)] if ev else r.get("best_win")  # the last evaluation (iteration 20 for sp3, 24 for sp4)
        fr["best"] = r.get("best_win")
        finished.append(fr)
    done = {r["seed"] for r in results}
    running = []
    for name in sorted(os.listdir(RUNS)):
        m = re.fullmatch(re.escape(batch) + r"-(\d+)", name)
        if m and int(m.group(1)) not in done and os.path.isfile(os.path.join(RUNS, name, "log.jsonl")):
            live = time.time() - os.path.getmtime(os.path.join(RUNS, name, "log.jsonl")) < RUNNING_STALE_S
            running.append(row(int(m.group(1)), name, out_evals.get(int(m.group(1)), {}),
                               "running" if live else "unfinished", None))
    mins = sorted(r["minutes"] for r in finished if r["minutes"] is not None)
    median = (mins[len(mins) // 2] if len(mins) % 2 else (mins[len(mins) // 2 - 1] + mins[len(mins) // 2]) / 2) if mins else None
    # Against the earlier batch on the same boards: the gap difference (this specialist's win vs its base minus
    # sp3's vs o3-187) and the head-to-head results from runs/<batch>/h2h.jsonl (sp4_h2h.py follower).
    if batch != "sp3":
        sp3_final = {}
        for r in read_jsonl(os.path.join(RUNS, "sp3", "results.jsonl")):
            ev = {e["iter"]: e["win"] for e in r.get("evals", []) if "iter" in e and "win" in e}
            sp3_final[r["seed"]] = ev[max(ev)] if ev else r.get("best_win")
        h2h = {d["seed"]: d for d in read_jsonl(os.path.join(RUNS, batch, "h2h.jsonl"))}
        for fr in finished:
            fr["sp3_final"] = sp3_final.get(fr["seed"])
            fr["delta"] = fr["final"] - fr["sp3_final"] if fr["final"] is not None and fr["sp3_final"] is not None else None
            h = h2h.get(fr["seed"], {})
            fr["h2h_nn"], fr["h2h_search"] = h.get("nn"), h.get("search")
    finals = [r["final"] for r in finished if r["final"] is not None]
    if catanatron_rs is not None:
        draw = sorted((r for r in finished if r["final"] is not None), key=lambda r: -r["final"])[:DRAW_TOP] + \
            [r for r in running if r["status"] == "running"]
        for r in draw:
            r["layout"] = board_layout(r["seed"])
    return {"batch": batch, "total": total, "finished": finished, "running": running, "current_line": current,
            "median_minutes": median,
            "eta_minutes": median * (total - len(finished)) if median is not None and seeds_file else None,
            "final_mean": sum(finals) / len(finals) if finals else None,
            "above60": sum(f > 0.6 for f in finals), "above70": sum(f > 0.7 for f in finals),
            "delta_mean": (lambda v: sum(v) / len(v) if v else None)([r["delta"] for r in finished if r.get("delta") is not None]),
            "h2h_nn_mean": (lambda v: sum(v) / len(v) if v else None)([r["h2h_nn"] for r in finished if r.get("h2h_nn") is not None]),
            "h2h_search_mean": (lambda v: sum(v) / len(v) if v else None)([r["h2h_search"] for r in finished if r.get("h2h_search") is not None]),
            "h2h_n": sum(1 for r in finished if r.get("h2h_nn") is not None),
            "drawings": catanatron_rs is not None}


LEDGERS = ("experiments", "optimizations")  # /<name> renders runs/<name>.json with ledger.html


def ledger(name):
    """runs/<name>.json, read fresh. Cards with runs and lane "running" get live numbers from their first run as
    card["live"]; cards with a score_file get its last lines as card["score_tail"]. Errors come back as {"error"}
    (a half-written file from a concurrent editor is just a transient error; the page keeps its last good board)."""
    path = os.path.join(RUNS, name + ".json")
    if not os.path.isfile(path):
        return {"error": f"runs/{name}.json not found"}
    try:
        with open(path) as f:
            d = json.load(f)
    except json.JSONDecodeError as e:
        return {"error": f"runs/{name}.json is not valid JSON: {e}"}
    if not isinstance(d, dict) or not isinstance(d.get("cards"), list):
        return {"error": f"runs/{name}.json must be an object with a \"cards\" list"}
    base = os.path.dirname(RUNS)  # score_file paths are relative to python/
    for c in d["cards"]:
        if not isinstance(c, dict):
            continue
        sf = c.get("score_file")
        if isinstance(sf, str) and c.get("lane") in ("running", "done"):
            p = os.path.realpath(os.path.join(base, sf))
            if p.startswith(os.path.realpath(base) + os.sep) and os.path.isfile(p):
                with open(p, errors="replace") as f:
                    c["score_tail"] = [l.rstrip("\n") for l in f if l.strip()][-3:]
        if c.get("lane") != "running":
            continue
        runs = [r for r in (c.get("runs") or []) if isinstance(r, str)]  # run names only; other ledgers may store records here
        if not runs:
            continue
        run = runs[0]
        if not os.path.isfile(os.path.join(RUNS, run, "log.jsonl")):
            c["live"] = {"run": run, "error": f"runs/{run}/log.jsonl not found"}
            continue
        r = run_series(run)
        L = r["latest"]
        c["live"] = {"run": run, "iter": L["iter"], "games": L["games"], "vs_ham": L["vs_ham"],
                     "best_vs_ham": L["best_vs_ham"], "surprise20": L["surprise20"],
                     "games_per_hour": L["games_per_hour"],
                     "panel_mean": r["panel"]["panel_mean"][-1] if r["panel"] else None}
    return d


def num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def fork_merge(fp, base):
    """n-weighted merge of the fork-pair kinds `base` and its uniform-draw twin: "open" with
    "open_uniform", "open2" with "open_uniform2" (the ordinal of a settlement fork trails the kind).
    Stats present in every merged kind are averaged; older logs have no regret."""
    ph = base.rstrip("0123456789")
    tag = base[len(ph):]
    parts = [fp.get(k) for k in (base, f"{ph}_uniform{tag}")]
    parts = [d for d in parts if isinstance(d, dict) and num(d.get("n"))]
    n = sum(d["n"] for d in parts)
    if not n:
        return {}
    out = {"n": n}
    for f in ("gap", "alt_p", "regret"):
        vals = [(d["n"], num(d.get(f))) for d in parts]
        if all(v is not None for _, v in vals):
            out[f] = sum(w * v for w, v in vals) / n
    return out


def window_pair(xs):
    """Headline tile for a per-iteration series: mean of the last HEADLINE_WINDOW rows and of the window before it
    (None for a window with no values)."""
    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None
    w = HEADLINE_WINDOW
    return {"value": mean(xs[-w:]), "prev": mean(xs[-2 * w:-w]) if len(xs) > w else None, "window": w}


def point_pair(xs, iters, back):
    """Latest non-null value and the value `back` rows earlier (already a windowed rate, so no averaging)."""
    idx = [i for i, x in enumerate(xs) if x is not None]
    if not idx:
        return {"value": None, "prev": None, "iter": None, "prev_iter": None}
    i = idx[-1]
    j = i - back
    ok = j >= 0 and xs[j] is not None
    return {"value": xs[i], "iter": iters[i], "prev": xs[j] if ok else None, "prev_iter": iters[j] if ok else None}


KINDS = ['road', 'robber', 'settlement', 'city', 'maritime', 'buy_dev', 'knight', 'end_turn', 'discard', 'year_of_plenty', 'monopoly', 'road_building']  # mid-game action kinds charted


def run_series(name, downsample=True):
    rows = read_log(name)
    sps = SECONDS_PER_SHARD.get(name, SECONDS_PER_SHARD_400)
    wall = 0.0          # cumulative wall-clock minutes; "minutes" restarts from 0 on resume
    prev_min = None
    shards = 0
    walls = []
    full = {k: [] for k in ["iter", "games", "actor_hours", "wall_min", "games_per_hour"] + SCALARS +
            [f"fp_{ph}_{f}" for ph in ("open", "mid") for f in ("gap", "alt_p", "n", "regret")] +
            [f"fp_regret_s{i}" for i in range(4)] + [f"fork_w{i}" for i in range(4)] +
            ["open_loss_pi", "open_loss_v", "open_steps", "open_replay"] +
            [f"sd_{ph}_{f}" for ph in ("mid", "open") for f in SURPRISE_DIST_FIELDS] +
            [f"lk_{k}" for k in KINDS] + [f"sk_{k}" for k in KINDS]}
    for i, r in enumerate(rows):
        m = num(r.get("minutes"))
        if m is not None:
            wall += (m - prev_min) if prev_min is not None and m >= prev_min else m
            prev_min = m
        walls.append(wall)
        rs = num(r.get("remote_shards"))
        if rs is not None:
            shards += rs
        full["iter"].append(num(r.get("iter")) if num(r.get("iter")) is not None else i + 1)
        full["games"].append(num(r.get("games")))
        full["actor_hours"].append(shards * sps / 3600 if "remote_shards" in r else None)
        full["wall_min"].append(wall if m is not None else None)
        j = i - GPH_WINDOW
        g, gj = num(r.get("games")), num(rows[j].get("games")) if j >= 0 else None
        dt = wall - walls[j] if j >= 0 else 0
        full["games_per_hour"].append((g - gj) / dt * 60 if g is not None and gj is not None and dt > 0 else None)
        for k in SCALARS:
            full[k].append(num(r.get(k)))
        # Fork pairs are logged per kind: open/mid, a _uniform suffix when the alternative was drawn
        # uniformly or from the pips margin, and a trailing settlement ordinal for settlement forks.
        # The charts merge the draw kinds per phase and per ordinal (the regret-weighted forks in
        # az.py --regret-forks follow the per-ordinal regret).
        fp = r.get("fork_pairs") if isinstance(r.get("fork_pairs"), dict) else {}
        for ph in ("open", "mid"):
            d = fork_merge(fp, ph)
            for f in ("gap", "alt_p", "n", "regret"):
                full[f"fp_{ph}_{f}"].append(d.get(f))
        for i in range(4):
            full[f"fp_regret_s{i}"].append(fork_merge(fp, f"open{i}").get("regret"))
        fw = r.get("fork_weights") if isinstance(r.get("fork_weights"), list) else []
        for i in range(4):
            full[f"fork_w{i}"].append(num(fw[i]) if i < len(fw) else None)
        # The separate opening network (az.py --opening-net), when the run has one.
        op = r.get("opening") if isinstance(r.get("opening"), dict) else {}
        for f in ("loss_pi", "loss_v", "steps", "replay"):
            full[f"open_{f}"].append(num(op.get(f)))
        # Surprise distribution per phase (runs started after Nov 11 18:15): shares of rows whose
        # surprise exceeds 0.2 / 0.69 nats, mean surprise among rows above 0.05, and the 90th percentile.
        sd = r.get("surprise_dist") if isinstance(r.get("surprise_dist"), dict) else {}
        for ph in ("mid", "open"):
            d = sd.get(ph) if isinstance(sd.get(ph), dict) else {}
            for f in SURPRISE_DIST_FIELDS:
                full[f"sd_{ph}_{f}"].append(num(d.get(f)))
        # Per kind of action played (mid-game rows; runs started after Nov 12 13:00): the policy loss of the
        # training batches' rows of that kind, and the mean search surprise of the iteration's new rows.
        lk = r.get("loss_pi_kind") if isinstance(r.get("loss_pi_kind"), dict) else {}
        sk = r.get("surprise_kind") if isinstance(r.get("surprise_kind"), dict) else {}
        for k in KINDS:
            full[f"lk_{k}"].append(num(lk.get(k)))
            full[f"sk_{k}"].append(num(sk[k].get("mean")) if isinstance(sk.get(k), dict) else None)
    n = len(rows)
    keep = list(range(n))
    if downsample and n > MAX_POINTS:
        step = n / MAX_POINTS
        keep = sorted({int(i * step) for i in range(MAX_POINTS)} | {n - 1})
    series = {k: [v[i] for i in keep] for k, v in full.items() if any(x is not None for x in v)}

    by_iter = {it: i for i, it in enumerate(full["iter"])}
    vs_ham = []
    p = os.path.join(RUNS, name + ".vs_ham")
    if os.path.isfile(p):
        with open(p) as f:
            for line in f:
                mt = VS_HAM_RE.search(line)
                if not mt:
                    continue
                it = int(mt.group(1))
                i = by_iter.get(it)
                vs_ham.append({"iter": it, "pct": float(mt.group(2)), "pm": float(mt.group(3)),
                               "games": full["games"][i] if i is not None else None,
                               "actor_hours": full["actor_hours"][i] if i is not None else None})

    vs_gaz = []
    p = os.path.join(RUNS, name + ".vs_gaz")
    if os.path.isfile(p):
        with open(p) as f:
            for line in f:
                mt = VS_GAZ_RE.search(line)
                if not mt:
                    continue
                it = int(mt.group(1))
                i = by_iter.get(it)
                vs_gaz.append({"iter": it, "pct": float(mt.group(2)), "pm": float(mt.group(3)),
                               "games": full["games"][i] if i is not None else None,
                               "actor_hours": full["actor_hours"][i] if i is not None else None})

    # The board panel: the 100-specialist follower file when the run has one, else the old 4-board file.
    panel_file = next((p for p in (name + ".panel100", name + ".panel") if os.path.isfile(os.path.join(RUNS, p))), name + ".panel100")
    # Opening trackers: runs/<run>.agree (first-pick agreement with the 100 experts) and
    # runs/<run>.valuetrack (opening value head vs forced win rates), one line per snapshot.
    tracks = {"agree": [], "spearman": [], "regret_pol": [], "regret_val": []}
    def track_point(it, value):
        i = by_iter.get(it)
        return {"iter": it, "pct": value, "games": full["games"][i] if i is not None else None,
                "actor_hours": full["actor_hours"][i] if i is not None else None}
    p = os.path.join(RUNS, name + ".agree")
    if os.path.isfile(p):
        with open(p) as f:
            for line in f:
                mt = re.search(r"snap_(\d+): agree (\d+)/(\d+)", line)
                if mt:
                    tracks["agree"].append(track_point(int(mt.group(1)), 100.0 * int(mt.group(2)) / int(mt.group(3))))
    p = os.path.join(RUNS, name + ".valuetrack")
    if os.path.isfile(p):
        with open(p) as f:
            for line in f:
                mt = re.search(r"snap_(\d+): spearman ([\d.]+) regret_pol ([\d.]+) regret_val ([\d.]+)", line)
                if mt:
                    it = int(mt.group(1))
                    tracks["spearman"].append(track_point(it, float(mt.group(2))))
                    tracks["regret_pol"].append(track_point(it, float(mt.group(3))))
                    tracks["regret_val"].append(track_point(it, float(mt.group(4))))

    snaps = []
    for e in parse_panel(os.path.join(RUNS, panel_file)):
        mt = re.search(r"snap_(\d+)", e["label"])
        if mt:
            snaps.append((int(mt.group(1)), e))
    boards = sorted({b for _, e in snaps for b in e["boards"]})
    panel = {"iter": [], "games": [], "actor_hours": [], "panel_mean": []}
    panel.update({"panel_" + b: [] for b in boards})
    for it, e in snaps:
        i = by_iter.get(it)
        panel["iter"].append(it)
        panel["games"].append(full["games"][i] if i is not None else None)
        panel["actor_hours"].append(full["actor_hours"][i] if i is not None else None)
        panel["panel_mean"].append(e["mean"])
        for b in boards:
            panel["panel_" + b].append(e["boards"].get(b))

    last50 = [x for x in full["surprise"][-50:] if x is not None]
    last20 = [x for x in full["surprise"][-20:] if x is not None]
    gph = [x for x in full["games_per_hour"] if x is not None]
    last = rows[-1] if rows else {}
    latest = {"iter": full["iter"][-1] if rows else None, "games": num(last.get("games")),
              "surprise50": sum(last50) / len(last50) if last50 else None,
              "surprise20": sum(last20) / len(last20) if last20 else None,
              "games_per_hour": gph[-1] if gph else None,
              "loss_pi": num(last.get("loss_pi")), "loss_v": num(last.get("loss_v")),
              "vs_ham": vs_ham[-1]["pct"] if vs_ham else None,
              "best_vs_ham": max(v["pct"] for v in vs_ham) if vs_ham else None,
              "wall_hours": wall / 60, "actor_hours": full["actor_hours"][-1] if rows else None,
              "windows": {k: window_pair(full[k]) for k in ("loss_v", "surprise")},
              "gph_pair": point_pair(full["games_per_hour"], full["iter"], HEADLINE_WINDOW)}
    args = None
    ap = os.path.join(RUNS, name, "args.json")
    if os.path.isfile(ap):
        with open(ap) as f:
            args = json.load(f)
    return {"name": name, "n": n, "seconds_per_shard": sps, "series": series, "vs_ham": vs_ham, "vs_gaz": vs_gaz, "tracks": tracks,
            "latest": latest, "args": args,
            "panel": panel if snaps else None, "panel_boards": boards, "panel_kind": 100 if panel_file.endswith("100") else 4}


def pig_games(fn=None):
    """One Pig match: its games (runs/<fn>, a pig_*.jsonl file; default = the file a running pig_match.py writes to,
    else the newest), whether that match is still running, the planned game count and the last log lines of its .out."""
    import subprocess
    procs = subprocess.run("ps -eo args", shell=True, capture_output=True, text=True).stdout.splitlines()
    live = {}
    for p in procs:
        m = re.search(r"pig_match\.py (\d+) .*--out (\S+)", p)
        if m:
            live[os.path.basename(m.group(2))] = int(m.group(1))
    files = sorted((f for f in os.listdir(RUNS) if f.startswith("pig_") and f.endswith(".jsonl")), key=lambda f: os.path.getmtime(os.path.join(RUNS, f)))
    if not fn:
        fn = next((f for f in files if f in live), files[-1] if files else None)
    if not fn or "/" in fn or not (fn.startswith("pig_") and fn.endswith(".jsonl")) or not os.path.isfile(os.path.join(RUNS, fn)):
        return {"file": fn, "games": [], "running": False, "planned": None, "log": []}
    games = [g for g in read_jsonl(os.path.join(RUNS, fn))]
    log_path = os.path.join(RUNS, fn[:-6] + ".out")
    log = []
    if os.path.isfile(log_path):
        with open(log_path) as f:
            log = [l.rstrip()[:200] for l in f.readlines()[-6:]]
    return {"file": fn, "games": games, "running": fn in live, "planned": live.get(fn), "log": log}


def pig_summary():
    out = []
    for fn in sorted(os.listdir(RUNS)):
        if not (fn.startswith("pig_") and fn.endswith(".jsonl")):
            continue
        games = []
        with open(os.path.join(RUNS, fn)) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        games.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        def wr(gs):
            n = len(gs)
            if not n:
                return {"n": 0, "wr": None, "se": None}
            p = sum(1 for g in gs if g.get("we_won")) / n
            return {"n": n, "wr": p, "se": math.sqrt(p * (1 - p) / n)}

        def mean(xs):
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else None

        def pts(g, ours):
            p, s = g.get("points"), g.get("our_seat")
            return p[s if ours else 1 - s] if isinstance(p, list) and len(p) == 2 and s in (0, 1) else None

        with_counts = [g for g in games if isinstance(g.get("counts"), list) and len(g["counts"]) == 2
                       and g.get("our_seat") in (0, 1)]
        profile = None
        if with_counts:
            profile = {"n": len(with_counts), "actions": ACTIONS,
                       "us": [mean([g["counts"][g["our_seat"]].get(a) for g in with_counts]) for a in ACTIONS],
                       "pig": [mean([g["counts"][1 - g["our_seat"]].get(a) for g in with_counts]) for a in ACTIONS]}
        out.append({"file": fn, "all": wr(games),
                    "first": wr([g for g in games if g.get("our_seat") == 0]),
                    "second": wr([g for g in games if g.get("our_seat") == 1]),
                    "turns": mean([num(g.get("turns")) for g in games]),
                    "our_points": mean([pts(g, True) for g in games]),
                    "pig_points": mean([pts(g, False) for g in games]),
                    "our_seconds": mean([num(g.get("our_seconds")) for g in games]),
                    "pig_seconds": mean([num(g.get("pig_seconds")) for g in games]),
                    "boards": sorted({str(g["board"]) for g in games if g.get("board")}),
                    "profile": profile})
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, allow_nan=False), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(RUNS, "dashboard.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/runs":
            return self._json(list_runs())
        if u.path == "/api/run":
            name = parse_qs(u.query).get("name", [""])[0]
            if not name or "/" in name or name.startswith(".") or \
                    not os.path.isfile(os.path.join(RUNS, name, "log.jsonl")):
                return self._json({"error": f"unknown run {name!r}"}, 404)
            return self._json(run_series(name))
        if u.path == "/panel":
            with open(os.path.join(RUNS, "panel_boards.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/panel_boards":
            name = parse_qs(u.query).get("run", [""])[0]
            path = panel_path(name) if name and "/" not in name and not name.startswith(".") else None
            if path is None:
                return self._json({"error": f"no runs/{name}.panel100 or runs/{name}.panel"}, 404)
            return self._json(panel_boards(name, path))
        if u.path == "/api/panel":
            return self._json(panel_baselines())
        if u.path == "/specialists":
            with open(os.path.join(RUNS, "specialists.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/specialists":
            batch = (parse_qs(u.query).get("batch") or ["sp3"])[0]
            if not re.fullmatch(r"sp\d+", batch):
                return self._json({"error": "unknown batch"}, 400)
            base = {"sp3": "o3-187", "sp4": "on6 (snap 800)"}.get(batch, "?")
            out = {"batch": batch, "base": base, "main": specialists_batch(batch, "boards/panel100.json")}
            if batch == "sp3":
                out["sp2"] = specialists_batch("sp2")
            return self._json(out)
        if u.path.lstrip("/") in LEDGERS:
            with open(os.path.join(RUNS, "ledger.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path.startswith("/api/") and u.path[5:] in LEDGERS:
            d = ledger(u.path[5:])
            return self._json(d, 500 if "error" in d else 200)
        if u.path == "/api/pig":
            return self._json(pig_summary())
        if u.path == "/pig":
            with open(os.path.join(RUNS, "pig.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/pig_games":
            return self._json(pig_games((parse_qs(u.query).get("file") or [None])[0]))
        self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"dashboard on http://0.0.0.0:{port}/  (runs dir {RUNS})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
