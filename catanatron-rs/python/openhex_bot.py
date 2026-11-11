"""Play OpenHex 1v1 (github.com/bovard/open-hex) with our engine's bots.

A `MirrorBot` keeps a local copy of the game (`catanatron_rs.Mirror`) in step
with the server: it applies the server's events in order, copies hands and
bank from each state snapshot (in 1v1 the opponent's hand is what is in
neither the bank nor ours), checks the copy against the snapshot's public
facts, and turns our bot's moves into OpenHex actions.

Everything here speaks the live API's JSON (hexes as {"x","y","z"} under
"hex", edges as two vertices under "edge"); actions are sent with the
internal names ("location", "vertices"), which the server also accepts.
"""

from __future__ import annotations

import catanatron_rs

RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")


class OutOfStep(RuntimeError):
    """The mirror disagrees with the server: stop rather than play on."""


def _xyz(d):
    return (d["x"], d["y"], d["z"])


def _parse(s):
    x, y, z = map(int, s.split(","))
    return {"x": x, "y": y, "z": z}


def _txt(d):
    return f"{d['x']},{d['y']},{d['z']}"


def _canon(a, b):
    """OpenHex's edge order: the x+y+z = 1 vertex first."""
    return (a, b) if sum(a) == 1 else (b, a)


def _edge(e):
    return e.get("edge") or e.get("vertices")


def _hex(e):
    return e.get("hex") or e.get("location")


class MirrorBot:
    """Plays one game as `me` with bot `key` (see catanatron_rs.Mirror.decide).
    `discard_key`, if given, decides discards: our engine discards one card
    per decision, so a time-budgeted key would spend its budget per card."""

    def __init__(self, me: str, key: str, seed: int = 0, log=None, discard_key: str | None = None):
        self.me, self.key, self.seed = me, key, seed
        self.discard_key = discard_key
        self.mirror = None
        self.buffer = []          # events not yet applied (a robber move waits for its steal)
        self.discards = {}        # color -> resources still to discard, in our engine's order
        self.log = log or (lambda *a: None)
        self.decisions = 0
        self.finished = False     # the server announced a winner

    # -- setup ---------------------------------------------------------------

    def start(self, state: dict):
        board = state["board"]
        tiles = [(_xyz(_hex(t)), t["terrain"], t.get("number")) for t in board["tiles"]]
        ports = []
        for p in board["ports"]:
            a, b = _edge(p)
            ports.append((_xyz(a), _xyz(b), p.get("resource")))
        cfg = state["config"]
        if cfg.get("numPlayers", len(state["turnOrder"])) != 2:
            raise ValueError("MirrorBot plays 1v1 only")
        self.mirror = catanatron_rs.Mirror(
            tiles, ports, state["turnOrder"],
            vps_to_win=cfg["victoryPoints"], discard_limit=cfg["discardLimit"],
            friendly_robber=cfg["robberMode"] == "friendly", openhex_dice=cfg["diceMode"] == "balanced",
            seed=self.seed)

    # -- events --------------------------------------------------------------

    def apply_events(self, events: list[dict], state: dict):
        """Applies events in order, `state` being the snapshot taken after
        them. A robber move is held until its steal has happened: it is the
        last event so far and the server is still waiting for a target."""
        self.buffer.extend(events)
        steal_pending = (state.get("pendingAction") or {}).get("type") == "stealFromPlayer"
        m = self.mirror
        while self.buffer:
            e = self.buffer[0]
            t, who = e["type"], e.get("player")
            if t in ("knightPlayed", "robberMoved"):
                # Award events can sit between a knight and its steal.
                k = 1
                while k < len(self.buffer) and self.buffer[k]["type"] in ("largestArmyAwarded", "longestRoadAwarded"):
                    k += 1
                if k >= len(self.buffer):
                    if steal_pending:
                        return  # its steal is still to come
                    nxt = {"type": None}
                else:
                    nxt = self.buffer[k]
                victim = stolen = None
                n = k
                if nxt["type"] == "resourceStolen" and nxt.get("player") == who:
                    victim, stolen, n = nxt["target"], nxt.get("resource"), k + 1
                    if stolen is None:
                        raise OutOfStep("a steal in 1v1 must be visible to both seats")
                elif nxt["type"] == "nothingToSteal" and nxt.get("player") == who:
                    n = k + 1
                if t == "knightPlayed":
                    m.ensure_dev(who, "knight")
                    m.knight(who)
                # A knight that wins ends the game before its robber move (in
                # the opponent's mirror only once `gameWon` reveals its
                # hidden victory point cards).
                if not m.is_over() and nxt["type"] != "gameWon":
                    m.robber(who, _xyz(_hex(e)), victim, stolen)
                del self.buffer[:n]
                continue
            self.buffer.pop(0)
            if t in ("setupSettlementPlaced", "settlementBuilt"):
                m.settlement(who, _xyz(e["vertex"]))
            elif t in ("setupRoadPlaced", "roadBuilt", "freeRoadPlaced"):
                a, b = _edge(e)
                m.road(who, _xyz(a), _xyz(b))
            elif t == "cityBuilt":
                m.city(who, _xyz(e["vertex"]))
            elif t == "diceRolled":
                m.roll(who, e["die1"], e["die2"])
            elif t == "devCardBought":
                m.buy(who, e.get("card"))
            elif t == "roadBuildingPlayed":
                m.ensure_dev(who, "roadBuilding")
                m.road_building(who)
            elif t == "yearOfPlentyPlayed":
                m.ensure_dev(who, "yearOfPlenty")
                rs = e["resources"]
                m.year_of_plenty(who, rs[0], rs[1] if len(rs) > 1 else None)
            elif t == "monopolyPlayed":
                m.ensure_dev(who, "monopoly")
                m.monopoly(who, e["resource"])
            elif t == "resourcesDiscarded":
                cards = [r for r in RESOURCES for _ in range(e["resources"].get(r, 0))]
                self.discards.setdefault(who, []).extend(cards)
                self._drain_discards()
            elif t == "bankTradeExecuted":
                (give, rate), = e["offering"].items()
                (get, _), = e["requesting"].items()
                m.bank_trade(who, give, rate, get)
            elif t == "turnEnded":
                m.end_turn(who)
            elif t == "gameWon":
                self.finished = True
            elif t == "tradeExecuted":
                raise OutOfStep("player trades are not supported by the mirror")
            elif t in ("setupResourcesGranted", "resourcesProduced", "resourceStolen", "nothingToSteal",
                       "longestRoadAwarded", "largestArmyAwarded", "tradeProposed",
                       "tradeAccepted", "tradeRejected", "tradeAnswerWithdrawn", "tradeCancelled"):
                pass  # follows from what the mirror already did, or doesn't change it
            else:
                raise OutOfStep(f"unhandled event {t}")
        self._drain_discards()

    def _drain_discards(self):
        """Our engine takes discards one card at a time in seat order; the
        server takes each seat's cards at once in any order."""
        m = self.mirror
        while True:
            who, prompt = m.prompt()
            if prompt != "Discard" or not self.discards.get(who):
                return
            m.discard(who, self.discards[who].pop(0))

    # -- snapshot ------------------------------------------------------------

    def sync_and_check(self, state: dict):
        """Copies hands and bank from the snapshot, then checks every public
        fact the mirror tracks. Raises OutOfStep on any difference."""
        if self.buffer:
            return  # mid-robber: the snapshot is ahead of what we applied
        m = self.mirror
        if m.is_over() or self.finished:
            return  # nothing left to play (a winning knight's robber move is skipped)
        if state.get("isSetupPhase"):
            # Nothing is hidden in setup, so the mirror is exact; and our
            # engine pays a second settlement's resources when it is placed,
            # OpenHex only after its road, so a sync now would take them away.
            return
        me = next(p for p in state["players"] if p["color"] == self.me)
        hand = {r: me.get("hand", {}).get(r, 0) for r in RESOURCES}
        bank = {r: state["bank"].get(r, 0) for r in RESOURCES}
        before = m.snapshot()
        mine_before = ",".join(str(hand[r]) for r in RESOURCES)
        if before[f"{self.me}.hand"] != mine_before:
            self.log(f"note: our hand was {before[f'{self.me}.hand']}, server says {mine_before} (resynced)")
        m.sync(self.me, hand, bank)
        snap = m.snapshot()
        problems = []
        board = state["board"]
        robber = board.get("robberHex") or board.get("robber")
        if robber and snap["robber"] != _txt(robber):
            problems.append(f"robber {snap['robber']} vs {_txt(robber)}")
        for p in state["players"]:
            c = p["color"]
            # The state has no points field: count them from the board and awards.
            public = (sum(1 for b in board.get("settlements", []) if b["player"] == c)
                      + 2 * sum(1 for b in board.get("cities", []) if b["player"] == c)
                      + 2 * (state.get("longestRoadPlayer") == c) + 2 * (state.get("largestArmyPlayer") == c))
            if int(snap[f"{c}.publicVP"]) != public:
                problems.append(f"{c} public VP {snap[f'{c}.publicVP']} vs {public}")
            # Our own cards come as `devCards` (this turn's purchases included),
            # the opponent's as a count.
            devs = p.get("devCardCount")
            if devs is None:
                devs = sum(p.get("devCards", {}).values())
            if int(snap[f"{c}.devCards"]) != devs:
                problems.append(f"{c} dev cards {snap[f'{c}.devCards']} vs {devs}")
            if int(snap[f"{c}.knights"]) != p.get("knightsPlayed", 0):
                problems.append(f"{c} knights {snap[f'{c}.knights']} vs {p.get('knightsPlayed')}")
            for kind in ("settlements", "cities"):
                server = sorted(_txt(b["vertex"]) for b in board.get(kind, []) if b["player"] == c)
                if snap[f"{c}.{kind}"] != ";".join(server):
                    problems.append(f"{c} {kind} differ")
            roads = []
            for r in board.get("roads", []):
                if r["player"] == c:
                    a, b = _canon(_xyz(_edge(r)[0]), _xyz(_edge(r)[1]))
                    roads.append(f"{a[0]},{a[1]},{a[2]}|{b[0]},{b[1]},{b[2]}")
            if snap[f"{c}.roads"] != ";".join(sorted(roads)):
                problems.append(f"{c} roads differ")
        deck = state.get("devCardDeck", {})
        remaining = deck.get("remaining", sum(deck.values()))
        if int(snap["devDeck"]) != remaining:
            problems.append(f"dev deck {snap['devDeck']} vs {remaining}")
        dice = state.get("balancedDice")
        if dice and "remaining" in dice:
            # Both decks refill before the next draw once 13 or fewer cards
            # remain (OpenHex starts empty, ours full): compare the deck the
            # next roll comes from.
            nxt = lambda n: 36 if n <= 13 else n
            if nxt(int(snap["diceDeck"])) != nxt(dice["remaining"]):
                problems.append(f"dice deck {snap['diceDeck']} vs {dice['remaining']}")
        for key, field in (("longestRoad", "longestRoadPlayer"), ("largestArmy", "largestArmyPlayer")):
            if snap[key] != (state.get(field) or ""):
                problems.append(f"{key} {snap[key]!r} vs {state.get(field)!r}")
        if problems:
            raise OutOfStep("; ".join(problems))

    # -- acting ----------------------------------------------------------------

    def must_act(self, state: dict) -> bool:
        pa = state.get("pendingAction")
        if pa and pa.get("type") == "discardResources":
            return self.me in (pa.get("players") or {})
        return state.get("currentPlayer") == self.me and not state.get("winner")

    def actions(self, state: dict) -> list[dict]:
        """OpenHex actions for our next decision (send them in order; a
        steal target comes back as {"type": "stealFromPlayer"} to send only
        if the server then asks for one)."""
        discarding = (state.get("pendingAction") or {}).get("type") == "discardResources"
        key = self.discard_key if discarding and self.discard_key else self.key
        moves = self.mirror.decide(self.me, key, self.seed + self.decisions)
        self.decisions += 1
        if not moves:
            raise OutOfStep(f"no move for {self.me} although the server waits on it "
                            f"(mirror {self.mirror.prompt()}, over {self.mirror.is_over()})")
        setup = state.get("isSetupPhase")
        pending = (state.get("pendingAction") or {}).get("type")
        # The live server only lets a knight be played before the roll (its
        # GitHub code allows any card): roll first instead.
        if pending == "rollDice" and moves and moves[0][0] in ("yearOfPlenty", "monopoly", "roadBuilding"):
            self.log(f"pre-roll {moves[0][0]} not allowed live: rolling first")
            return [{"type": "rollDice"}]
        out = []
        i = 0
        while i < len(moves):
            kind, args = moves[i]
            if kind == "settlement":
                out.append({"type": "placeInitialSettlement" if setup else "buildSettlement", "vertex": _parse(args[0])})
            elif kind == "city":
                out.append({"type": "buildCity", "vertex": _parse(args[0])})
            elif kind == "road":
                t = "placeInitialRoad" if setup else ("placeFreeRoad" if pending == "placeFreeRoad" else "buildRoad")
                out.append({"type": t, "vertices": [_parse(args[0]), _parse(args[1])]})
            elif kind == "roll":
                out.append({"type": "rollDice"})
            elif kind == "end":
                out.append({"type": "endTurn"})
            elif kind == "buy":
                out.append({"type": "buyDevCard"})
            elif kind == "roadBuilding":
                out.append({"type": "playRoadBuilding"})
            elif kind == "monopoly":
                out.append({"type": "playMonopoly", "resource": args[0]})
            elif kind == "yearOfPlenty":
                out.append({"type": "playYearOfPlenty", "resources": list(args)})
            elif kind == "bankTrade":
                give, rate, get = args
                out.append({"type": "tradeWithBank", "offering": {give: int(rate)}, "requesting": {get: 1}})
            elif kind == "knight":
                rkind, rargs = moves[i + 1]
                assert rkind == "robber", moves
                out.append({"type": "playKnight", "location": _parse(rargs[0])})
                if len(rargs) > 1:
                    out.append({"type": "stealFromPlayer", "target": rargs[1]})
                i += 1
            elif kind == "robber":
                out.append({"type": "moveRobber", "location": _parse(args[0])})
                if len(args) > 1:
                    out.append({"type": "stealFromPlayer", "target": args[1]})
            elif kind == "discard":
                counts = {}
                for k, a in moves[i:]:
                    assert k == "discard", moves
                    counts[a[0]] = counts.get(a[0], 0) + 1
                out.append({"type": "discardResources", "resources": counts})
                break
            else:
                raise OutOfStep(f"no OpenHex action for {kind}")
            i += 1
        return out
