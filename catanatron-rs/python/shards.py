"""Self-play shards: compact, compressed batches of examples for moving
self-play between machines (remote actors -> S3 -> the learner).

A shard stores exactly what az.py's replay buffer keeps: observations and
policy targets in half precision (as the buffer does), sparse; the legal-move
mask as bits; values and auxiliary targets as float32. ~100 bytes/example.

S3 layout (bucket from the CDK stack, see ../infra):
    models/control.json     what actors should play: model key, pool keys, self-play kwargs
    models/<run>/<key>.ctnn networks
    shards/<run>/<model iter>/<actor>-<seq>.npz.xz
"""

import io
import json
import lzma

import numpy as np

CONTROL_KEY = "models/control.json"


def encode(d, meta):
    """Shard bytes from a catanatron_rs.self_play result `d` (plus `meta`)."""
    n, A, X = d["n"], meta["action_size"], d["aux_size"]
    obs = np.frombuffer(d["obs"], "<f4").reshape(n, meta["obs_size"]).astype(np.float16)
    pi = np.frombuffer(d["pi"], "<f4").reshape(n, A).astype(np.float16)
    mask = np.frombuffer(d["mask"], np.uint8).reshape(n, A)
    arrays = {"mask": np.packbits(mask.astype(bool), axis=1)}
    for name, m in (("obs", obs), ("pi", pi)):
        nz = m != 0
        arrays[f"{name}_count"] = nz.sum(1).astype(np.uint16)
        arrays[f"{name}_idx"] = np.nonzero(nz)[1].astype(np.uint16)
        arrays[f"{name}_val"] = m[nz]
    for k in ("z", "z_luck", "q", "surprise"):
        arrays[k] = np.frombuffer(d[k], "<f4")
    arrays["aux"] = np.frombuffer(d["aux"], "<f4").reshape(n, X)
    arrays["endgame"] = np.frombuffer(d["endgame"], np.uint8)
    if "kind" in d:  # the kind of the action played (selfplay::KIND_NAMES), for per-kind losses
        arrays["kind"] = np.frombuffer(d["kind"], np.uint8)
    if "horizon" in d:  # the mid-game horizon value label (NaN where not computed)
        arrays["horizon"] = np.frombuffer(d["horizon"], "<f4")
    arrays["opening"] = np.frombuffer(d["opening"], np.uint8)
    arrays["focus"] = np.frombuffer(d["focus"], np.uint8)
    arrays["game"] = np.frombuffer(d["game"], "<u4")
    stats = {k: d[k] for k in ("n", "results", "decisions", "escalated", "branches", "forks",
                               "pool_games", "luck_abs", "luck_events")}
    # Fork pairs summarised per kind: count, mean (main - fork) result for the fork's mover, mean alternative
    # probability, mean regret (how much the alternative beat the main pick, floored at 0). A settlement fork is
    # also counted under its settlement ordinal ("open0".."open3", "open_uniform0"..).
    pairs = {}
    for main_v, fork_v, p_alt, kind in d.get("fork_pairs", []):
        for name in fork_kind_names(kind):
            c = pairs.setdefault(name, [0, 0.0, 0.0, 0.0])
            c[0] += 1; c[1] += main_v - fork_v; c[2] += p_alt; c[3] += max(0.0, fork_v - main_v)
    pairs = {k: [n, s_ / n, p_ / n, r_ / n] for k, (n, s_, p_, r_) in pairs.items()}
    header = json.dumps({**meta, "aux_size": X, "stats": stats, "boards": [[int(b), float(e)] for b, e in d.get("boards", [])],
                         "fork_pairs": pairs}).encode()
    arrays["header"] = np.frombuffer(header, np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return lzma.compress(buf.getvalue(), preset=1)  # fast: at low sims a shard plays in ~5 s


def fork_kind_names(kind):
    """The aggregate kind name of a fork pair and, for a settlement fork, its per-ordinal name."""
    base = ("open", "mid", "open_uniform", "mid_uniform")[kind & 3]
    tag = kind >> 2
    return (base, f"{base}{tag - 1}") if tag else (base,)


def _dense(count, idx, val, shape):
    out = np.zeros(shape, np.float16)
    rows = np.repeat(np.arange(shape[0]), count.astype(np.int64))
    out[rows, idx] = val
    return out


def decode(blob):
    """(header dict, arrays): obs/pi float16 [n][...], mask bool, z, z_luck,
    q, surprise float32 [n], aux float32 [n][aux_size], endgame and opening uint8."""
    with np.load(io.BytesIO(lzma.decompress(blob)), allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    header = json.loads(a.pop("header").tobytes())
    n, obs_size, A = header["stats"]["n"], header["obs_size"], header["action_size"]
    out = {
        "obs": _dense(a["obs_count"], a["obs_idx"], a["obs_val"], (n, obs_size)),
        "pi": _dense(a["pi_count"], a["pi_idx"], a["pi_val"], (n, A)),
        "mask": np.unpackbits(a["mask"], axis=1, count=A).astype(bool),
    }
    for k in ("z", "z_luck", "q", "surprise", "aux", "endgame", "opening"):
        out[k] = a[k]
    # Shards from before the game tag: every example its own game (weight 1).
    out["game"] = a["game"] if "game" in a else np.arange(n, dtype=np.uint32)
    # Shards from before the focus tag have none: no focus decisions.
    out["focus"] = a["focus"] if "focus" in a else np.zeros(n, np.uint8)
    # Shards from before the kind tag: unknown kind (255).
    out["kind"] = a["kind"] if "kind" in a else np.full(n, 255, np.uint8)
    out["horizon"] = a["horizon"] if "horizon" in a else np.full(n, np.nan, np.float32)
    return header, out
