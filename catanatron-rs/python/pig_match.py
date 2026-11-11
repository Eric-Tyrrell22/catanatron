"""Our bots against the colonizing-pig bot ("Oversized Pig",
github.com/TheWeiHu/colonizing-pig), played in its own engine (Colonist 1v1
rules: 15 points, balanced dice, friendly robber below 3 public points,
discards above 9, bank and port trades only).

The game runs in the pig engine through bench/pig/pigd (see pigd.cpp). Our
side keeps a `catanatron_rs.Mirror` in step with every action and chance
outcome, as for OpenHex, and searches it for our moves. The pig board is
placed in the mirror's (OpenHex) frame by the hex-grid symmetry that puts
its harbors on our harbor slots. Every step is checked: on our turns our
legal moves must be exactly the pig engine's, and after every action both
engines must agree on hands, bank, public points, dev card counts, robber
and awards. Any disagreement stops the match (`OutOfStep`).

Only the pig bot's own view reaches it (its engine's `observe`); ours sees
the opponent's dev card purchases as unknown cards.

    python pig_match.py <games> <our bot key> [--parallel N] [--pig-workers W] [--seed S]
                        [--out results.jsonl]

Each game is reported (with the running tally) as it finishes, so a remote
run can be followed in its logs; with --out each game is also appended to a
JSON-lines file.

e.g. python pig_match.py 100 "MCTS:4000,c1,f0.25,m0.1,s0.3/150:openhex_models/ham.ctnn" --parallel 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import catanatron_rs

HERE = os.path.dirname(os.path.abspath(__file__))
# Where the pig checkout (its weights/) and the driver are; --pig-repo and
# --pigd change them (the actor image has them under /app/pig).
PIG_REPO = os.path.abspath(os.path.join(HERE, "../../../external/colonizing-pig"))
PIGD = os.path.abspath(os.path.join(HERE, "../bench/pig/pigd"))

# The pig engine's geometry (engine/catan.cpp): land hexes in axial (q, r),
# neighbour directions, and for each hex corner the two neighbours sharing it.
LAND = [(0, -2), (1, -2), (2, -2), (2, -1), (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
        (-2, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, 0)]
DIRS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
SHARED = [((5, 4), (4, 2)), ((5, 3), (0, 5)), ((0, 4), (1, 0)), ((1, 5), (2, 1)), ((2, 0), (3, 2)), ((3, 1), (4, 3))]
YOP_PAIRS = [(a, b) for a in range(5) for b in range(a, 5)]
# Action ids (engine/catan.hpp).
ROLL, END, BUY, KNIGHT, ROAD_BUILDING, YOP, MONOPOLY, ROBBER, DISCARD, TRADE = 180, 181, 182, 183, 184, 185, 200, 205, 224, 229
# Phases.
PLACE_SETTLEMENT, PLACE_ROAD, PRE_ROLL, P_ROLL, P_DISCARD, MOVE_ROBBER, STEAL, MAIN, DEV_DRAW, P_ROAD_BUILDING = range(10)

RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
TERRAIN = ("forest", "hills", "pasture", "fields", "mountains")
DEV = ("knight", "victoryPoint", "roadBuilding", "yearOfPlenty", "monopoly")
COLORS = ("red", "blue")  # pig player 0 moves first


class OutOfStep(RuntimeError):
    """The two engines disagree: stop rather than play on."""


class Pig:
    """One pigd process."""

    def __init__(self, workers: int, pigd: str, repo: str):
        self.p = subprocess.Popen([pigd, str(workers)], cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def __call__(self, cmd: str) -> dict:
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        out = json.loads(self.p.stdout.readline())
        if "error" in out:
            raise OutOfStep(f"pig engine: {out['error']} (after {cmd!r})")
        return out

    def close(self):
        self.p.stdin.write("quit\n")
        self.p.stdin.flush()
        self.p.wait()


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _openhex_vertex_hexes():
    """OpenHex vertex -> the three hexes around it (cube), as a frozenset key map."""
    corners = {}
    for x in range(-4, 5):
        for y in range(-4, 5):
            z = -x - y
            if max(abs(x), abs(y), abs(z)) > 4:
                continue
            for v in catanatron_rs_hex_vertices((x, y, z)):
                corners.setdefault(v, set()).add((x, y, z))
    return {frozenset(h): v for v, h in corners.items() if len(h) == 3}


def catanatron_rs_hex_vertices(h):
    """OpenHex's corners of hex h (catanatron::openhex::hex_vertices)."""
    x, y, z = h
    return [(x + 1, y, z + 1), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z), (x, y + 1, z + 1), (x, y, z + 1)]


VERTEX_OF_HEXES = _openhex_vertex_hexes()
SYMMETRIES = [(perm, sign) for perm in itertools.permutations(range(3)) for sign in (1, -1)]


class Geometry:
    """Pig ids <-> OpenHex-frame coordinates under one hex-grid symmetry."""

    def __init__(self, topo: dict, symmetry):
        perm, sign = symmetry
        def cube(axial):
            q, r = axial
            c = (q, r, -q - r)
            return tuple(sign * c[i] for i in perm)
        self.cube = cube
        self.tile_hex = [cube(h) for h in LAND]
        self.hex_tile = {h: i for i, h in enumerate(self.tile_hex)}
        self.node_vertex = {}
        for tile, corners in enumerate(topo["tile_nodes"]):
            for c, n in enumerate(corners):
                around = frozenset(cube(h) for h in (LAND[tile], _add(LAND[tile], DIRS[SHARED[c][0][0]]), _add(LAND[tile], DIRS[SHARED[c][1][0]])))
                v = VERTEX_OF_HEXES[around]
                if self.node_vertex.setdefault(n, v) != v:
                    raise ValueError(f"pig node {n} at two vertices")
        self.vertex_node = {v: n for n, v in self.node_vertex.items()}
        self.edge_nodes = [tuple(e) for e in topo["edge_nodes"]]
        self.nodes_edge = {frozenset(e): i for i, e in enumerate(self.edge_nodes)}


def pig_board_command(topo: dict, seed: int, board_spec: str):
    """The pigd `board` command that sets up our board `board_spec` in the pig
    engine: our layout in pig's tile and port order, under the hex symmetry
    and port layout that put pig's harbor slots on ours."""
    tiles, ports = catanatron_rs.board_layout(board_spec)
    our_tiles = {tuple(h): (-1 if t == "desert" else TERRAIN.index(t), n or 0) for h, t, n in tiles}
    our_ports = {frozenset((tuple(a), tuple(b))): -1 if r is None else RESOURCES.index(r) for a, b, r in ports}
    for sym in SYMMETRIES:
        geo = Geometry(topo, sym)
        if any(h not in our_tiles for h in geo.tile_hex):
            continue
        for alt in (0, 1):
            pairs = [frozenset((geo.node_vertex[a], geo.node_vertex[b])) for a, b in topo["port_nodes"][alt]]
            if all(p in our_ports for p in pairs):
                res = [our_tiles[h][0] for h in geo.tile_hex]
                nums = [our_tiles[h][1] for h in geo.tile_hex]
                return f"board {seed} {' '.join(map(str, res))} {' '.join(map(str, nums))} {' '.join(str(our_ports[p]) for p in pairs)} {alt}", sym
    raise OutOfStep(f"{board_spec}: no pig port layout matches our harbor slots")


def _parse_v(s: str):
    return tuple(int(x) for x in s.split(","))


class Game:
    """One game: the pig engine (truth) and our mirror of it."""

    def __init__(self, pig: Pig, topo: dict, seed: int, our_seat: int, key: str, check_legal: bool = True, board_spec=None):
        self.pig, self.seed, self.us, self.key, self.check_legal = pig, seed, our_seat, key, check_legal
        self.board_spec = board_spec
        syms = SYMMETRIES
        if board_spec is None:
            board = pig(f"new {seed}")
        else:
            # Mirror under the same symmetry the board went in with, so the
            # mirror's board is exactly ours (not a turned copy).
            cmd, sym = pig_board_command(topo, seed, board_spec)
            board = pig(cmd)
            syms = [sym] + [x for x in SYMMETRIES if x != sym]
        self.board = board
        tiles_axial = list(zip(board["resources"], board["numbers"]))
        self.mirror = None
        for sym in syms:
            geo = Geometry(topo, sym)
            tiles = [(geo.tile_hex[i], "desert" if r < 0 else TERRAIN[r], n or None) for i, (r, n) in enumerate(tiles_axial)]
            ports = [(geo.node_vertex[a], geo.node_vertex[b], None if r < 0 else RESOURCES[r])
                     for (a, b), r in zip(board["port_nodes"], board["port_resources"])]
            try:
                self.mirror = catanatron_rs.Mirror(tiles, ports, list(COLORS), vps_to_win=15, discard_limit=9,
                                                   friendly_robber=True, openhex_dice=True, seed=seed,
                                                   colonist=True)
            except ValueError:
                continue
            self.geo = geo
            break
        if self.mirror is None:
            raise OutOfStep(f"board {seed}: no symmetry puts its harbors on our slots")
        self.discards = {c: [] for c in COLORS}
        self.decisions = 0
        # Per player, how many actions of each kind the game saw (behavioural profile).
        self.counts = [{k: 0 for k in ("settlement", "road", "city", "buy_dev", "knight", "road_building", "yop", "monopoly", "robber", "discard", "trade")} for _ in COLORS]

    # -- pig events -> mirror ---------------------------------------------------

    def apply(self, events):
        m, i = self.mirror, 0
        while i < len(events):
            player, a, phase, info = events[i]
            who = COLORS[player]
            hidden = player != self.us
            if a >= 0:
                k = ("settlement" if a < 54 else "road" if a < 126 else "city" if a < 180 else None if a < 182 else "buy_dev" if a == 182
                     else "knight" if a == 183 else "road_building" if a == 184 else "yop" if a < 200 else "monopoly" if a < 205
                     else "robber" if a < 224 else "discard" if a < 229 else "trade")
                if k:
                    self.counts[player][k] += 1
            nxt = events[i + 1] if i + 1 < len(events) else None
            chance = nxt if nxt and nxt[1] == -1 else None
            i += 1
            if a == -1:
                # A roll on its own: after a dev card played before rolling, the
                # pig engine rolls at once (ours offers only the roll then).
                if phase != P_ROLL:
                    raise OutOfStep(f"unexpected chance event {events[i - 1]}")
                total = info + 2
                d1 = min(6, total - 1)
                m.roll(who, d1, total - d1)
                continue
            if a < 54:
                m.settlement(who, self.geo.node_vertex[a])
            elif a < 126:
                x, y = self.geo.edge_nodes[a - 54]
                m.road(who, self.geo.node_vertex[x], self.geo.node_vertex[y])
            elif a < 180:
                m.city(who, self.geo.node_vertex[a - 126])
            elif a == ROLL:
                if not chance or chance[2] != P_ROLL:
                    raise OutOfStep("a roll without its outcome")
                total = chance[3] + 2
                d1 = min(6, total - 1)
                m.roll(who, d1, total - d1)
                i += 1
            elif a == END:
                if phase == P_ROAD_BUILDING:
                    raise OutOfStep("ending road building early has no move in our engine")
                m.end_turn(who)
            elif a == BUY:
                if not chance or chance[2] != DEV_DRAW:
                    raise OutOfStep("a purchase without its card")
                m.buy(who, None if hidden else DEV[chance[3]])
                i += 1
            elif a == KNIGHT:
                if hidden:
                    m.ensure_dev(who, "knight")
                m.knight(who)
            elif a == ROAD_BUILDING:
                if hidden:
                    m.ensure_dev(who, "roadBuilding")
                m.road_building(who)
            elif a < MONOPOLY:
                if hidden:
                    m.ensure_dev(who, "yearOfPlenty")
                x, y = YOP_PAIRS[a - YOP]
                m.year_of_plenty(who, RESOURCES[x], RESOURCES[y])
            elif a < ROBBER:
                if hidden:
                    m.ensure_dev(who, "monopoly")
                m.monopoly(who, RESOURCES[a - MONOPOLY])
            elif a < DISCARD:
                hexc = self.geo.tile_hex[a - ROBBER]
                if chance and chance[2] == STEAL:
                    m.robber(who, hexc, COLORS[1 - player], RESOURCES[chance[3]])
                    i += 1
                else:
                    self._robber_without_steal(who, hexc)
            elif a < TRADE:
                self.discards[who].append(RESOURCES[a - DISCARD])
                self._drain_discards()
            else:
                give, k = divmod(a - TRADE, 4)
                get = k + (k >= give)
                m.bank_trade(who, RESOURCES[give], info, RESOURCES[get])
        self._drain_discards()

    def _robber_without_steal(self, who, hexc):
        """The pig engine had no steal: our move is the one without a victim if
        it has one, else the one naming a victim with nothing to take."""
        legal = self.mirror.legal_moves()
        h = f"{hexc[0]},{hexc[1]},{hexc[2]}"
        options = [args for kind, args in legal if kind == "robber" and args[0] == h]
        if [h] in options:
            self.mirror.robber(who, hexc)
        else:
            raise OutOfStep(f"robber to {h} without a steal: our engine only has {options}")

    def _drain_discards(self):
        m = self.mirror
        while True:
            who, prompt = m.prompt()
            if prompt != "Discard" or not self.discards[who]:
                return
            m.discard(who, self.discards[who].pop(0))

    # -- our moves -> pig ids -----------------------------------------------------

    def to_pig(self, move) -> int:
        kind, args = move
        g = self.geo
        if kind == "settlement":
            return g.vertex_node[_parse_v(args[0])]
        if kind == "city":
            return 126 + g.vertex_node[_parse_v(args[0])]
        if kind == "road":
            a, b = (g.vertex_node[_parse_v(x)] for x in args)
            return 54 + g.nodes_edge[frozenset((a, b))]
        if kind == "roll":
            return ROLL
        if kind == "end":
            return END
        if kind == "buy":
            return BUY
        if kind == "knight":
            return KNIGHT
        if kind == "roadBuilding":
            return ROAD_BUILDING
        if kind == "yearOfPlenty":
            if len(args) != 2:
                raise OutOfStep(f"year of plenty for one card has no pig move: {move}")
            x, y = sorted(RESOURCES.index(r) for r in args)
            return YOP + YOP_PAIRS.index((x, y))
        if kind == "monopoly":
            return MONOPOLY + RESOURCES.index(args[0])
        if kind == "robber":
            return ROBBER + g.hex_tile[_parse_v(args[0])]
        if kind == "discard":
            return DISCARD + RESOURCES.index(args[0])
        if kind == "bankTrade":
            give, rate, get = RESOURCES.index(args[0]), int(args[1]), RESOURCES.index(args[2])
            return TRADE + give * 4 + (get if get < give else get - 1), rate
        raise OutOfStep(f"no pig move for {move}")

    def _pig_id(self, move) -> int:
        a = self.to_pig(move)
        return a[0] if isinstance(a, tuple) else a

    # -- checks -------------------------------------------------------------------

    def check_moves(self, legal: list[int]):
        """On our turn: our legal moves are exactly the pig engine's (trades at
        the same rates)."""
        ours = {}
        for mv in self.mirror.legal_moves():
            a = self.to_pig(mv)
            if isinstance(a, tuple):
                a, rate = a
                ours[a] = rate
            else:
                ours[a] = None
        if set(ours) != set(legal):
            extra, missing = sorted(set(ours) - set(legal)), sorted(set(legal) - set(ours))
            raise OutOfStep(f"legal moves differ: ours only {extra}, pig only {missing}")

    def pending_discards(self) -> bool:
        """After a 7 the pig engine takes the player to move's discards first,
        ours takes seats in order: until both are in, hands and legal moves
        legitimately differ."""
        return any(self.discards.values())

    def check_state(self):
        s = self.pig("state")
        if self.pending_discards():
            return s
        snap = self.mirror.snapshot()
        problems = []
        for p in range(2):
            c = COLORS[p]
            pl = s["players"][p]
            if snap[f"{c}.hand"] != ",".join(map(str, pl["hand"])):
                problems.append(f"{c} hand {snap[f'{c}.hand']} vs {pl['hand']}")
            if int(snap[f"{c}.publicVP"]) != pl["public"]:
                problems.append(f"{c} public points {snap[f'{c}.publicVP']} vs {pl['public']}")
            if int(snap[f"{c}.devCards"]) != sum(pl["dev"]):
                problems.append(f"{c} dev cards {snap[f'{c}.devCards']} vs {sum(pl['dev'])}")
            if int(snap[f"{c}.knights"]) != pl["played"][0]:
                problems.append(f"{c} knights {snap[f'{c}.knights']} vs {pl['played'][0]}")
        if snap["bank"] != ",".join(map(str, s["bank"])):
            problems.append(f"bank {snap['bank']} vs {s['bank']}")
        h = self.geo.tile_hex[s["robber"]]
        if snap["robber"] != f"{h[0]},{h[1]},{h[2]}":
            problems.append(f"robber {snap['robber']} vs {h}")
        for key, field in (("longestRoad", "longest_road"), ("largestArmy", "largest_army")):
            pig = COLORS[s[field]] if s[field] >= 0 else ""
            if snap[key] != pig:
                problems.append(f"{key} {snap[key]!r} vs {pig!r}")
        if problems:
            raise OutOfStep("; ".join(problems))
        return s

    # -- play -----------------------------------------------------------------------

    def play(self) -> dict:
        t_us = t_pig = 0.0
        while True:
            s = self.check_state()
            if s["done"]:
                break
            legal = self.pig("legal")
            if legal["cur"] == self.us:
                who = COLORS[self.us]
                mirror_who, prompt = self.mirror.prompt()
                if self.check_legal and mirror_who == who and not self.pending_discards():
                    self.check_moves(legal["legal"])
                t = time.time()
                moves = self.mirror.decide(who, self.key, self.seed * 7919 + self.decisions)
                t_us += time.time() - t
                self.decisions += 1
                for k, mv in enumerate(moves):
                    if k:
                        later = self.pig("legal")["legal"]
                        if not later:
                            # The pig engine ends the game as soon as a knight wins
                            # largest army; our mirror still has the robber move to make.
                            break
                        if self.check_legal and not self.pending_discards() and self.mirror.prompt()[0] == who:
                            # Later moves of one decision (a knight's robber move,
                            # more discards): checked against the pig engine too.
                            self.check_moves(later)
                    out = self.pig(f"act {self._pig_id(mv)}")
                    self.apply(out["events"])
            else:
                t = time.time()
                out = self.pig("bot")
                t_pig += time.time() - t
                self.apply(out["events"])
        winner = s["winner"]
        return {"seed": self.seed, "board": self.board_spec, "our_seat": self.us, "winner": winner, "we_won": winner == self.us,
                "turns": s["turn"], "points": [p["total"] for p in s["players"]], "counts": self.counts,
                "our_seconds": round(t_us, 1), "pig_seconds": round(t_pig, 1)}


def play_one(args):
    seed, our_seat, key, workers, topo, pigd, repo, board_spec = args
    pig = Pig(workers, pigd, repo)
    try:
        return Game(pig, topo, seed, our_seat, key, board_spec=board_spec).play()
    except OutOfStep as e:
        return {"seed": seed, "our_seat": our_seat, "error": str(e)}
    finally:
        pig.close()


def tally(results):
    """(line) for the games so far: wins, rate, error, Elo."""
    done = [r for r in results if "error" not in r]
    errors = len(results) - len(done)
    if not done:
        return f"0 games{f', {errors} errors' if errors else ''}"
    wins = sum(r["we_won"] for r in done)
    p = wins / len(done)
    se = math.sqrt(max(p * (1 - p), 0.25 / len(done)) / len(done))
    elo = -400 * math.log10(1 / min(max(p, 1e-3), 1 - 1e-3) - 1)
    by_seat = []
    for seat in (0, 1):
        g = [r for r in done if r["our_seat"] == seat]
        if g:
            by_seat.append(f"{'first' if seat == 0 else 'second'} {sum(r['we_won'] for r in g)}/{len(g)}")
    return (f"{wins}/{len(done)} = {p:.1%} ±{se:.1%}, {elo:+.0f} Elo ({', '.join(by_seat)})"
            + (f", {errors} errors" if errors else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", type=int)
    ap.add_argument("key", help="our bot (catanatron::players::from_key), e.g. MCTS:4000,c1,f0.25,m0.1,s0.3/150:<model.ctnn>")
    ap.add_argument("--parallel", type=int, default=4, help="games at once")
    ap.add_argument("--pig-workers", type=int, default=1, help="search threads per pig bot")
    ap.add_argument("--seed", type=int, default=1, help="first board seed (each board is played from both seats)")
    ap.add_argument("--out", help="append one JSON line per game here")
    ap.add_argument("--board", help="play every game on this board of ours (e.g. seed:1010194); --seed then only seeds the dice")
    ap.add_argument("--pig-repo", default=PIG_REPO, help="the colonizing-pig checkout (the bot loads weights/ from it)")
    ap.add_argument("--pigd", default=PIGD, help="the driver binary (bench/pig/pigd)")
    args = ap.parse_args()
    key = args.key
    if ":" in key:
        head, path = key.rsplit(":", 1)
        key = f"{head}:{os.path.abspath(path)}"
    topo_pig = Pig(1, args.pigd, args.pig_repo)
    topo = topo_pig("topo")
    topo_pig.close()
    jobs = [(args.seed + g // 2, g % 2, key, args.pig_workers, topo, args.pigd, args.pig_repo, args.board) for g in range(args.games)]
    results = []
    print(f"{time.strftime('%H:%M')} {args.games} games, {args.parallel} at once: {args.key}", flush=True)
    # Processes, not threads: our search holds the GIL while it thinks, so
    # threads would take turns on one core.
    with ProcessPoolExecutor(args.parallel) as pool:
        for fut in as_completed([pool.submit(play_one, j) for j in jobs]):
            r = fut.result()
            results.append(r)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
            if "error" in r:
                print(f"{time.strftime('%H:%M')} board {r['seed']} seat {r['our_seat']}: ERROR {r['error']} | {tally(results)}", flush=True)
                continue
            print(f"{time.strftime('%H:%M')} board {r['seed']} seat {r['our_seat']}: {'win ' if r['we_won'] else 'loss'} "
                  f"{r['points'][r['our_seat']]}-{r['points'][1 - r['our_seat']]} in {r['turns']} turns "
                  f"(us {r['our_seconds']}s, pig {r['pig_seconds']}s) | {tally(results)}", flush=True)
    print(f"{time.strftime('%H:%M')} done: {tally(results)}", flush=True)


if __name__ == "__main__":
    main()
