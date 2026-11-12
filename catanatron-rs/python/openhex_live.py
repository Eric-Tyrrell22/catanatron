"""Run one bot on the OpenHex ladder (https://open-hex.web.app), 1v1.

    .venv/bin/python openhex_live.py register <name>        # once: prints nothing secret, stores the key
    .venv/bin/python openhex_live.py play <name> <bot key>  # loop: queue, play, repeat

API keys live in openhex_keys.json next to this file (never printed). The bot
key is a catanatron_rs bot key, e.g.
    MCTS:100000,t10000,c1,f0.25,m0.1,s0.1/400:/path/to/model.ctnn
(t<ms> = thinking time per move). Each process plays one bot; the mirror
(catanatron_rs.Mirror) lives on the thread that made it.

If the mirror ever disagrees with the server, the game is abandoned (no
more moves are sent) and the process stops, rather than playing on a wrong
picture of the game.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

from openhex_bot import MirrorBot, OutOfStep

API = "https://open-hex.web.app"
KEYS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openhex_keys.json")
UA = {"User-Agent": "catan-research-client"}
PAUSE_BETWEEN_GAMES_S = 3  # seconds between a finished game and the next queue join (was 60)


def log(name, msg):
    print(f"{time.strftime('%H:%M:%S')} [{name}] {msg}", flush=True)


def load_keys():
    if not os.path.exists(KEYS):
        return {}
    with open(KEYS) as f:
        return json.load(f)


def http(method, path, headers, body=None, params=None):
    """(status, parsed JSON or text) for one request."""
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={**headers, **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status, text = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, text = e.code, e.read().decode()
    try:
        return status, json.loads(text)
    except ValueError:
        return status, text


def register(name):
    keys = load_keys()
    if name in keys:
        raise SystemExit(f"{name} is already registered (key in {KEYS})")
    status, data = http("POST", "/api/bots", UA, {"name": name, "mode": "1v1"})
    if status != 200:
        raise SystemExit(f"registration failed: {status} {data}")
    keys[name] = {"api_key": data["api_key"], "bot_id": data["bot"]["id"]}
    tmp = KEYS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(keys, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, KEYS)
    print(f"registered {name} (bot id {data['bot']['id']}); key stored in {KEYS}")


class Client:
    def __init__(self, name):
        entry = load_keys().get(name)
        if not entry:
            raise SystemExit(f"{name} is not registered; run: openhex_live.py register {name}")
        self.name = name
        self.h = {"Authorization": f"Bearer {entry['api_key']}", **UA}

    def _req(self, method, path, body=None, params=None):
        for attempt in range(6):
            try:
                status, data = http(method, path, self.h, body, params)
                if status >= 500:
                    raise ConnectionError(f"server {status}")
                return status, data
            except (ConnectionError, urllib.error.URLError, TimeoutError) as e:
                delay = 2 ** attempt
                log(self.name, f"network error ({e}); retry in {delay}s")
                time.sleep(delay)
        raise RuntimeError("network keeps failing")

    def get(self, path, **params):
        status, data = self._req("GET", path, params=params or None)
        if status != 200:
            raise RuntimeError(f"GET {path}: {status} {data}")
        return data

    def act(self, game_id, action, duration_ms):
        status, data = self._req("POST", f"/api/games/{game_id}/actions", {"action": action, "duration_ms": duration_ms})
        if status != 200:
            raise RuntimeError(f"action {action['type']}: {status} {data}")
        return data


def discard_key_for(key: str) -> str:
    """The bot key with at most 1 s per card for discards (see MirrorBot)."""
    parts = key.split(":")
    if not parts[0] == "MCTS":
        return key
    opts = [o for o in parts[1].split(",") if not o.startswith("t")] + ["t1000"]
    return ":".join([parts[0], ",".join(opts)] + parts[2:])


def play_game(c: Client, game_id: str, key: str):
    g = c.get(f"/api/games/{game_id}/state")
    me = g["your_color"]
    state = g["state"]
    if not state["board"].get("tiles"):
        board = c.get(f"/api/games/{game_id}/board")["board"]
        state["board"]["tiles"], state["board"]["ports"] = board["tiles"], board["ports"]
    bot = MirrorBot(me, key, seed=int(time.time()), log=lambda m: log(c.name, m), discard_key=discard_key_for(key))
    bot.start(state)
    last_seq = -1
    log(c.name, f"game {game_id}: playing {me}")
    while True:
        events = c.get(f"/api/games/{game_id}/events", after=last_seq)["events"]
        g = c.get(f"/api/games/{game_id}/state")
        state = g["state"]
        # Events up to the snapshot only (more may have arrived in between).
        new = c.get(f"/api/games/{game_id}/events", after=last_seq)["events"]
        if len(new) != len(events):
            continue  # the game moved while we read it: read again
        if events:
            last_seq = events[-1]["seq"]
        bot.apply_events([e["event"] for e in events], state)
        if g["game"]["status"] != "active" or state.get("winner"):
            w = (g["game"].get("winner") or {}).get("name", state.get("winner"))
            log(c.name, f"game {game_id} over; winner {w}")
            return
        bot.sync_and_check(state)
        if not bot.must_act(state):
            time.sleep(1.0)
            continue
        t0 = time.time()
        actions = bot.actions(state)
        took = int((time.time() - t0) * 1000)
        for a in actions:
            if a["type"] == "stealFromPlayer":
                pending = (c.get(f"/api/games/{game_id}/state")["state"].get("pendingAction") or {}).get("type")
                if pending != "stealFromPlayer":
                    continue
            res = c.act(game_id, a, took)
            if not res.get("success"):
                raise OutOfStep(f"server rejected {a}: {res.get('error')}")
            if any(e.get("event", {}).get("type") == "gameWon" for e in res.get("events", [])):
                break


def play_forever(name, key):
    c = Client(name)
    log(name, f"starting with {key.split(':')[0]}")
    while True:
        try:
            games = c.get("/api/games", status="active")
            if games:
                play_game(c, games[0]["id"], key)
                time.sleep(PAUSE_BETWEEN_GAMES_S)
                continue
            status, data = c._req("POST", "/api/queue/join", {"mode": "1v1"})
            if status not in (200, 409):
                log(name, f"queue join: {status} {data}")
            time.sleep(10)
        except OutOfStep as e:
            log(name, f"STOPPING: mirror out of step: {e}")
            traceback.print_exc()
            c._req("POST", "/api/queue/leave", {"mode": "1v1"})
            raise SystemExit(2)
        except KeyboardInterrupt:
            c._req("POST", "/api/queue/leave", {"mode": "1v1"})
            log(name, "stopped")
            return


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "register":
        register(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "play":
        play_forever(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
