"""Rebuilds a learner checkpoint (weights + meta, fresh optimizer) from a CTNN network file, which stores
exact fp32 weights: for recovering a run from a snapshot when model.pt is lost or corrupted.
Handles version 1 (flat), 3 (opening spot head) and 4 (board heads).

    PYTHONPATH=<branch>/target/pylib:<branch>/python .venv/bin/python runs/ctnn_to_pt.py <net.ctnn> <out.pt> <meta from a checkpoint .pt>
"""
import struct, sys
import numpy as np, torch
sys.path.insert(0, "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs/python")
from model import Policy, SPOT_HEAD_TABLES, BOARD_HEAD_TABLES

src, out, meta_src = sys.argv[1], sys.argv[2], sys.argv[3]
b = open(src, "rb").read(); pos = 0
def u32():
    global pos; v = struct.unpack_from("<I", b, pos)[0]; pos += 4; return v
def tensors(n):
    global pos; out = []
    for _ in range(n):
        r, c = u32(), u32(); t = np.frombuffer(b, "<f4", r * c, pos).reshape(r, c).copy(); pos += 4 * r * c; out.append(torch.tensor(t))
    return out
def table():
    global pos; nd = u32(); dims = [u32() for _ in range(nd)]; n = int(np.prod(dims)); t = np.frombuffer(b, "<u4", n, pos).reshape(dims).astype(np.int64); pos += 4 * n; return t
assert b[:4] == b"CTNN"; pos = 4
version, obs_size, action_size = u32(), u32(), u32()
body = tensors(u32())
layers = len(body) // 2 - 2
state = {}
for i in range(layers):
    state[f"trunk.{2 * i}.weight"], state[f"trunk.{2 * i}.bias"] = body[2 * i], body[2 * i + 1].reshape(-1)
state["pi.weight"], state["pi.bias"] = body[2 * layers], body[2 * layers + 1].reshape(-1)
state["v.weight"], state["v.bias"] = body[2 * layers + 2], body[2 * layers + 3].reshape(-1)
hidden = state["trunk.0.weight"].shape[0]
spot_head = board_heads = None; spot_width = board_width = None
if version == 3:
    spot_head = {k: table() for k in SPOT_HEAD_TABLES}
    t = tensors(u32()); spot_width = t[0].shape[0]
    for name, (w, bias) in zip(("enc", "ctx", "mix", "out"), zip(t[::2], t[1::2])):
        state[f"spot_head.{name}.weight"] = w
        if name != "ctx": state[f"spot_head.{name}.bias"] = bias.reshape(-1)
elif version == 4:
    board_heads = {k: table() for k in BOARD_HEAD_TABLES}
    t = tensors(u32()); board_width = t[0].shape[0]
    for name, (w, bias) in zip(("enc", "ctx", "mix", "settle_out", "city_out", "edge_enc", "edge_out", "tile_enc", "tile_out"), zip(t[::2], t[1::2])):
        state[f"board_heads.{name}.weight"] = w
        if name != "ctx": state[f"board_heads.{name}.bias"] = bias.reshape(-1)
assert pos == len(b), f"{len(b) - pos} trailing bytes"
meta = dict(torch.load(meta_src, map_location="cpu")["meta"])
model = Policy(obs_size, action_size, hidden=hidden, layers=layers, aux=meta.get("aux", 0), spot_head=spot_head, spot_width=spot_width or 64, board_heads=board_heads, board_width=board_width or 32)
missing = model.load_state_dict(state, strict=False)
allowed = lambda k: k.startswith("aux.") or k.endswith(("count1", "count2", "tile_count")) or k.startswith(("spot_head.head_", "board_heads.head_"))
assert not missing.unexpected_keys and all(allowed(k) for k in missing.missing_keys), missing
meta.update({"obs_size": obs_size, "action_size": action_size, "hidden": hidden, "layers": layers, "spot_head": spot_head is not None, "board_heads": board_heads is not None, "restored_from": src})
torch.save({"model": model.state_dict(), "meta": meta}, out)
print(f"wrote {out}: version {version}, {layers}x{hidden}, obs {obs_size}, actions {action_size}, heads: spot={spot_head is not None} board={board_heads is not None}; aux head {'kept random' if meta.get('aux') else 'none'}; meta iters {meta.get('iters')} games {meta.get('games')}")
