"""Local viewer for our bots' OpenHex games.

    .venv/bin/python openhex_viewer.py [port] [host]   # then open http://<host>:8765

`host` defaults to 127.0.0.1 (this machine only); 0.0.0.0 serves the local
network too.

Serves a page that redraws the board every few seconds, for every bot in
openhex_keys.json. A picker lists each bot's active and recent games; by
default the page follows a live game. The server reads games with the bots'
API keys; the keys never reach the browser.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from openhex_live import Client, load_keys

RES = ("wood", "brick", "sheep", "wheat", "ore")
FULL_DEV = {"knight": 14, "victoryPoint": 5, "yearOfPlenty": 2, "monopoly": 2, "roadBuilding": 2}


class Feed:
    """Game lists and snapshots for every bot, read on demand and cached
    briefly, so many browser polls cost few API reads. Finished games are
    cached for good."""

    LIST_TTL, LIVE_TTL = 10.0, 2.5

    def __init__(self, bots):
        self.clients = {b: Client(b) for b in bots}
        self.lock = threading.Lock()
        self.lists = (0.0, None)
        self.snaps = {}   # (bot, game id) -> (time, snapshot)
        self.boards = {}  # game id -> board

    def games(self):
        with self.lock:
            at, cached = self.lists
            if cached is not None and time.time() - at < self.LIST_TTL:
                return cached
        out = []
        for bot, c in self.clients.items():
            for status, limit in (("active", None), ("completed", 10)):
                for g in c.get("/api/games", status=status)[:limit]:
                    opp = next((p["name"] for p in g["players"] if p["name"] != bot), "?")
                    out.append({"bot": bot, "id": g["id"], "status": g["status"], "opponent": opp,
                                "turn": g.get("turn_number"), "winner": (g.get("winner") or {}).get("name")})
        with self.lock:
            self.lists = (time.time(), out)
        return out

    def state(self, bot, gid):
        if bot not in self.clients:
            raise KeyError(f"unknown bot {bot}")
        with self.lock:
            hit = self.snaps.get((bot, gid))
        if hit and (hit[1]["status"] != "active" or time.time() - hit[0] < self.LIVE_TTL):
            return hit[1]
        snap = self.read(bot, gid)
        with self.lock:
            self.snaps[(bot, gid)] = (time.time(), snap)
        return snap

    def read(self, bot, gid):
        client = self.clients[bot]
        if gid not in self.boards:
            self.boards[gid] = client.get(f"/api/games/{gid}/board")["board"]
        board = self.boards[gid]
        g = client.get(f"/api/games/{gid}/state")
        events = client.get(f"/api/games/{gid}/events", after=-1)["events"]
        s, info, me = g["state"], g["game"], g["your_color"]
        b = s["board"]
        names = {p["color"]: p["name"] for p in info["players"]}
        mine = next(p for p in s["players"] if p["color"] == me)
        hand = {r: mine.get("hand", {}).get(r, 0) for r in RES}
        players = []
        played = {k: 0 for k in FULL_DEV}
        for p in s["players"]:
            for k, v in p.get("playedDevCards", {}).items():
                played[k] = played.get(k, 0) + v
        for p in s["players"]:
            col = p["color"]
            vis = (sum(1 for x in b.get("settlements", []) if x["player"] == col)
                   + 2 * sum(1 for x in b.get("cities", []) if x["player"] == col)
                   + 2 * (s.get("longestRoadPlayer") == col) + 2 * (s.get("largestArmyPlayer") == col))
            if col == me:
                h = hand
                hidden_vp = p.get("devCards", {}).get("victoryPoint", 0)
                dev = p.get("devCards", {})
            else:
                h = {r: 19 - s["bank"].get(r, 0) - hand[r] for r in RES}
                hidden_vp = None
                dev = {"unplayed": p.get("devCardCount", 0)}
            players.append({
                "color": col, "name": names.get(col, col), "me": col == me, "visible_vp": vis,
                "hidden_vp": hidden_vp, "hand": h, "dev": dev, "played": p.get("playedDevCards", {}),
                "knights": p.get("knightsPlayed", 0), "road_length": p.get("longestRoadLength", 0),
                "longest_road": s.get("longestRoadPlayer") == col, "largest_army": s.get("largestArmyPlayer") == col,
            })
        unseen = {k: FULL_DEV[k] - played.get(k, 0) - mine.get("devCards", {}).get(k, 0) for k in FULL_DEV}
        return {
            "status": info["status"], "game_id": gid, "turn": s.get("turnNumber"), "current": s.get("currentPlayer"),
            "pending": (s.get("pendingAction") or {}).get("type"), "dice": s.get("dice"),
            "winner": (info.get("winner") or {}).get("name"), "me": me, "bot": bot,
            "tiles": board["tiles"], "ports": board["ports"],
            "robber": b.get("robberHex"), "settlements": b.get("settlements", []), "cities": b.get("cities", []),
            "roads": b.get("roads", []), "players": players, "bank": s["bank"],
            "dev_deck": s.get("devCardDeck", {}).get("remaining"), "unseen_dev": unseen,
            "dice_deck": (s.get("balancedDice") or {}).get("remaining"),
            "events": [e["event"] for e in events[-14:]], "updated": time.strftime("%H:%M:%S"),
        }


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Game Viewer</title>
<style>
:root { --bg:#f4f1ea; --panel:#fffdf8; --ink:#2b2a27; --muted:#7a766d; --line:#d9d3c5; --water:#bcd6e6;
  --wood:#3f7d3a; --brick:#b5562f; --sheep:#9cc56b; --wheat:#e3bd4a; --ore:#8a8f99; --desert:#e8d8b0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#1d1e20; --panel:#26282b; --ink:#ece8df; --muted:#a39e93; --line:#3a3c40; --water:#284456; } }
:root[data-theme="dark"] { --bg:#1d1e20; --panel:#26282b; --ink:#ece8df; --muted:#a39e93; --line:#3a3c40; --water:#284456; }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui,sans-serif }
header { display:flex; gap:16px; align-items:baseline; padding:12px 16px; flex-wrap:wrap }
header h1 { font-size:18px; margin:0 } .muted { color:var(--muted) }
select { font:inherit; color:var(--ink); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:3px 6px; max-width:100% }
main { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; padding:0 16px 16px }
@media (max-width:900px) { main { grid-template-columns:1fr } }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px }
svg { width:100%; height:auto; display:block }
.player { border-left:6px solid; padding-left:10px; margin-bottom:14px }
.vp { font-size:28px; font-weight:700 } .row { display:flex; gap:6px; flex-wrap:wrap; margin-top:4px }
.chip { display:inline-flex; gap:4px; align-items:center; padding:1px 7px; border-radius:999px; background:var(--bg); border:1px solid var(--line); font-size:12px }
.dot { width:10px; height:10px; border-radius:2px; display:inline-block }
ol { margin:0; padding-left:18px; font-size:12px } li { margin:1px 0 } .turnflag { font-weight:700 }
</style></head><body>
<header><h1>Game Viewer</h1><select id="pick" aria-label="Game"></select><span id="meta" class="muted">loading…</span></header>
<main><div class="card"><svg id="board" viewBox="-430 -400 860 800" role="img" aria-label="Board"></svg></div>
<div><div class="card" id="players"></div><div class="card" style="margin-top:12px" id="decks"></div>
<div class="card" style="margin-top:12px"><b>Recent events</b><ol id="events"></ol></div></div></main>
<script>
const RES = {forest:"wood", hills:"brick", pasture:"sheep", fields:"wheat", mountains:"ore", desert:"desert"};
const PCOL = {red:"#d0342c", blue:"#2f6fd0", white:"#e9e6df", orange:"#e8862b"};
const S = 62, R3 = Math.sqrt(3);
const css = n => getComputedStyle(document.documentElement).getPropertyValue("--" + n).trim();
// Hexes: cube (x,y,z) with x+y+z=0, pointy-top, oriented as OpenHex's
// directions: East (+1,0,-1) to the right, NE (+1,-1,0) up and right.
const hexXY = (x, y, z) => [S * R3 * (x + y / 2), S * 1.5 * y];
// A vertex is the meeting point of its three hexes (OpenHex SPEC section 3).
function vXY(v) {
  const {x, y, z} = v, a = x + y + z === 1
    ? [[x-1,y,z],[x,y-1,z],[x,y,z-1]] : [[x-1,y,z-1],[x-1,y-1,z],[x,y-1,z-1]];
  const p = a.map(h => hexXY(...h));
  return [(p[0][0]+p[1][0]+p[2][0])/3, (p[0][1]+p[1][1]+p[2][1])/3];
}
const pips = n => n ? 6 - Math.abs(7 - n) : 0;
function el(tag, attrs, text) { const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; }
function draw(d) {
  const svg = document.getElementById("board"); svg.replaceChildren();
  svg.append(el("rect", {x:-430, y:-400, width:860, height:800, fill:css("water"), rx:20}));
  for (const t of d.tiles) {
    const [cx, cy] = hexXY(t.hex.x, t.hex.y, t.hex.z);
    const pts = [...Array(6)].map((_, i) => { const a = Math.PI / 180 * (60 * i - 30); return `${cx + S * Math.cos(a)},${cy + S * Math.sin(a)}`; }).join(" ");
    svg.append(el("polygon", {points: pts, fill: css(RES[t.terrain]), stroke: css("panel"), "stroke-width": 3}));
    if (t.number) {
      svg.append(el("circle", {cx, cy, r: 19, fill: css("panel"), opacity: .92}));
      svg.append(el("text", {x: cx, y: cy + 6, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: (t.number === 6 || t.number === 8) ? "#c0392b" : css("ink")}, t.number));
      svg.append(el("text", {x: cx, y: cy + 16, "text-anchor": "middle", "font-size": 9, fill: css("muted")}, "•".repeat(pips(t.number))));
    }
    if (d.robber && d.robber.x === t.hex.x && d.robber.y === t.hex.y && d.robber.z === t.hex.z)
      svg.append(el("circle", {cx: cx + 26, cy: cy - 22, r: 11, fill: "#222", stroke: "#fff", "stroke-width": 2}));
  }
  for (const p of d.ports) {
    const [a, b] = p.edge.map(vXY), mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, len = Math.hypot(mx, my), ox = mx / len * 26, oy = my / len * 26;
    svg.append(el("line", {x1: a[0], y1: a[1], x2: mx + ox, y2: my + oy, stroke: css("muted"), "stroke-width": 2}));
    svg.append(el("line", {x1: b[0], y1: b[1], x2: mx + ox, y2: my + oy, stroke: css("muted"), "stroke-width": 2}));
    svg.append(el("rect", {x: mx + ox - 22, y: my + oy - 11, width: 44, height: 22, rx: 6, fill: p.resource ? css(p.resource) : css("panel"), stroke: css("ink")}));
    svg.append(el("text", {x: mx + ox, y: my + oy + 5, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: p.resource ? "#fff" : css("ink")}, `${p.rate}:1`));
  }
  for (const r of d.roads) { const [a, b] = r.edge.map(vXY);
    svg.append(el("line", {x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: "#111", "stroke-width": 11, "stroke-linecap": "round"}));
    svg.append(el("line", {x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: PCOL[r.player], "stroke-width": 7, "stroke-linecap": "round"})); }
  for (const s of d.settlements) { const [x, y] = vXY(s.vertex);
    svg.append(el("rect", {x: x - 10, y: y - 10, width: 20, height: 20, rx: 3, fill: PCOL[s.player], stroke: "#111", "stroke-width": 2})); }
  for (const c of d.cities) { const [x, y] = vXY(c.vertex);
    svg.append(el("circle", {cx: x, cy: y, r: 15, fill: PCOL[c.player], stroke: "#111", "stroke-width": 3}));
    svg.append(el("circle", {cx: x, cy: y, r: 5, fill: "#111"})); }
}
const chip = (label, n, color) => `<span class="chip">${color ? `<span class="dot" style="background:${color}"></span>` : ""}${label} <b>${n}</b></span>`;
function panel(d) {
  const meta = d.status === "active"
    ? `${d.bot} as ${d.me} · turn ${d.turn} · ${d.current} to act${d.pending ? " (" + d.pending + ")" : ""}${d.dice ? " · last roll " + (d.dice.die1 + d.dice.die2) : ""}`
    : `${d.status}${d.winner ? " · winner " + d.winner : ""}`;
  document.getElementById("meta").textContent = `${meta} · updated ${d.updated}${d.error ? " · " + d.error : ""}`;
  document.getElementById("players").innerHTML = d.players.map(p => {
    const vp = p.hidden_vp != null ? `${p.visible_vp + p.hidden_vp} <span class="muted" style="font-size:13px">(${p.visible_vp} + ${p.hidden_vp} hidden)</span>` : `${p.visible_vp}+ <span class="muted" style="font-size:13px">visible</span>`;
    const hand = Object.entries(p.hand).map(([r, n]) => chip(r, n, css(r))).join("");
    const dev = Object.entries(p.dev).map(([k, n]) => chip(k, n)).join("") || '<span class="muted">none</span>';
    const awards = (p.longest_road ? chip("Longest Road", "+2") : "") + (p.largest_army ? chip("Largest Army", "+2") : "");
    return `<div class="player" style="border-color:${PCOL[p.color]}">
      <div><b>${p.name}</b> <span class="muted">${p.color}${p.me ? " · us" : ""}</span>${d.current === p.color && d.status === "active" ? ' <span class="turnflag">◀ to act</span>' : ""}</div>
      <div class="vp">${vp}</div><div class="row">${awards}${chip("knights", p.knights)}${chip("road", p.road_length)}</div>
      <div class="row">${hand}</div><div class="row"><span class="muted">dev:</span> ${dev}</div></div>`;
  }).join("");
  document.getElementById("decks").innerHTML = `<b>Decks</b><div class="row">${chip("dev deck", d.dev_deck)}${chip("dice deck", d.dice_deck ?? "–")}</div>
    <div class="row"><span class="muted">unseen dev:</span> ${Object.entries(d.unseen_dev).filter(([, n]) => n > 0).map(([k, n]) => chip(k, n)).join("")}</div>
    <div class="row"><span class="muted">bank:</span> ${Object.entries(d.bank).map(([r, n]) => chip(r, n, css(r))).join("")}</div>`;
  document.getElementById("events").innerHTML = d.events.slice().reverse().map(e => {
    const extra = e.die1 ? ` ${e.die1 + e.die2}` : e.card ? ` ${e.card}` : e.resource ? ` ${e.resource}` : e.resources ? " " + JSON.stringify(e.resources).replace(/[{}"]/g, "") : "";
    return `<li><span style="color:${PCOL[e.player] || "inherit"}">●</span> ${e.type}${extra}</li>`; }).join("");
}
// Selection: "live" follows an active game (any bot); "<bot>/<game id>" pins one.
let choice = decodeURIComponent(location.hash.slice(1)) || "live";
let games = [];
const pick = document.getElementById("pick");
const label = g => `${g.bot} vs ${g.opponent} · ${g.status === "active" ? "LIVE, turn " + g.turn
  : (g.winner ? (g.winner === g.bot ? "won" : "lost") : g.status) + ", " + g.turn + " turns"} · ${g.id.slice(0, 6)}`;
async function loadGames() {
  try { games = await (await fetch("/games")).json(); } catch (e) { return; }
  const opts = [`<option value="live">Follow a live game</option>`]
    .concat(games.map(g => `<option value="${g.bot}/${g.id}">${label(g)}</option>`));
  pick.innerHTML = opts.join(""); pick.value = choice;
  if (pick.value !== choice) pick.value = "live";
}
pick.onchange = () => { choice = pick.value; location.hash = choice === "live" ? "" : choice; tick(); };
function target() {
  if (choice !== "live") { const [bot, id] = choice.split("/"); return {bot, id}; }
  const g = games.find(g => g.status === "active") || games[0];
  return g && {bot: g.bot, id: g.id};
}
async function tick() {
  const t = target();
  if (!t) { document.getElementById("meta").textContent = "no games yet"; return; }
  try { const d = await (await fetch(`/state?bot=${encodeURIComponent(t.bot)}&game=${encodeURIComponent(t.id)}`)).json();
    if (d.tiles) { draw(d); panel(d); } else document.getElementById("meta").textContent = d.error || d.status; }
  catch (e) { document.getElementById("meta").textContent = "viewer server not reachable"; }
}
loadGames().then(tick); setInterval(tick, 3000); setInterval(loadGames, 15000);
</script></body></html>"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    bots = list(load_keys())
    feed = Feed(bots)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/games":
                    body, ctype = json.dumps(feed.games()).encode(), "application/json"
                elif url.path == "/state":
                    body, ctype = json.dumps(feed.state(q["bot"], q["game"])).encode(), "application/json"
                elif url.path in ("/", "/index.html"):
                    body, ctype = PAGE.encode(), "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
            except Exception as e:  # report to the page; the server keeps running
                body, ctype = json.dumps({"status": "error", "error": str(e)}).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"viewing {', '.join(bots)} at http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
