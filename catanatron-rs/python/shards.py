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
    arrays["opening"] = np.frombuffer(d["opening"], np.uint8)
    arrays["focus"] = np.frombuffer(d["focus"], np.uint8)
    arrays["game"] = np.frombuffer(d["game"], "<u4")
    stats = {k: d[k] for k in ("n", "results", "decisions", "escalated", "branches", "forks",
                               "pool_games", "luck_abs", "luck_events")}
    # Fork pairs summarised per kind: count, mean (main - fork) result for the fork's mover, mean alternative probability.
    pairs = {}
    for main_v, fork_v, p_alt, kind in d.get("fork_pairs", []):
        c = pairs.setdefault(("open", "mid", "open_uniform", "mid_uniform")[kind], [0, 0.0, 0.0])
        c[0] += 1; c[1] += main_v - fork_v; c[2] += p_alt
    pairs = {k: [n, s_ / n, p_ / n] for k, (n, s_, p_) in pairs.items()}
    header = json.dumps({**meta, "aux_size": X, "stats": stats, "boards": [[int(b), float(e)] for b, e in d.get("boards", [])],
                         "fork_pairs": pairs}).encode()
    arrays["header"] = np.frombuffer(header, np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return lzma.compress(buf.getvalue(), preset=1)  # fast: at low sims a shard plays in ~5 s


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
    return header, out
