"""Export a train.py checkpoint to the CTNN format read by Rust (src/nn.rs).

    .venv/bin/python export.py runs/long1/model.pt runs/long1/model.ctnn
"""

import struct
import sys

import torch

from model import BOARD_HEAD_TABLES, SPOT_HEAD_TABLES, SPOT_TABLES, Policy, board_head_tables_of, board_head_width_of, spot_head_tables_of, spot_head_width_of, spot_tables_of


def _tensors(linears):
    """Weight and bias (zeros if the layer has none) of each layer."""
    out = []
    for lin in linears:
        w = lin.full_weight() if hasattr(lin, "full_weight") else lin.weight
        out.append(w.detach().float().cpu())
        bias = lin.bias if lin.bias is not None else torch.zeros(lin.out_features)
        out.append(bias.detach().float().cpu().reshape(1, -1))
    return out


def _pack(tensors):
    parts = [struct.pack("<I", len(tensors))]
    for t in tensors:
        parts.append(struct.pack("<II", *t.shape))
        parts.append(t.contiguous().numpy().astype("<f4").tobytes())
    return parts


def to_bytes(model, obs_size, action_size):
    """CTNN bytes for a Policy (weights copied to CPU float32): version 1, or
    2 with the spot branch (format in src/nn.rs)."""
    linears = [m for m in model.trunk if isinstance(m, torch.nn.Linear) or hasattr(m, "full_weight")] + [model.pi, model.v]
    extras = sum(x is not None for x in (model.spots, model.spot_head, model.board_heads))
    if extras > 1:
        raise ValueError("only one of the spot branch, the spot head and the board heads is exportable per network")
    version = 2 if model.spots is not None else 3 if model.spot_head is not None else 4 if model.board_heads is not None else 1
    parts = [b"CTNN", struct.pack("<III", version, obs_size, action_size)] + _pack(_tensors(linears))
    if model.spots is not None:
        sp = model.spots
        for name in SPOT_TABLES:
            t = getattr(sp, name).cpu().numpy()
            parts.append(struct.pack("<I", t.ndim) + struct.pack(f"<{t.ndim}I", *t.shape))
            parts.append(t.astype("<u4").tobytes())
        parts += _pack(_tensors([sp.tile, sp.node1, sp.node_tiles_in, sp.node2, sp.join, sp.tower, sp.tower_out]))
        parts.append(struct.pack("<f", float(model.flat_value_scale)))
    if model.spot_head is not None:
        sh = model.spot_head
        for name in SPOT_HEAD_TABLES:
            t = getattr(sh, name).cpu().numpy()
            parts.append(struct.pack("<I", t.ndim) + struct.pack(f"<{t.ndim}I", *t.shape))
            parts.append(t.astype("<u4").tobytes())
        parts += _pack(_tensors([sh.enc, sh.ctx, sh.mix, sh.out]))
    if model.board_heads is not None:
        bh = model.board_heads
        for name in BOARD_HEAD_TABLES:
            t = getattr(bh, name).cpu().numpy()
            parts.append(struct.pack("<I", t.ndim) + struct.pack(f"<{t.ndim}I", *t.shape))
            parts.append(t.astype("<u4").tobytes())
        parts += _pack(_tensors([bh.enc, bh.ctx, bh.mix, bh.settle_out, bh.city_out, bh.edge_enc, bh.edge_out, bh.tile_enc, bh.tile_out]))
    return b"".join(parts)


def export(ckpt_path, out_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    meta = ck["meta"]
    # Checkpoints from before auxiliary heads have no "aux" entry.
    # Checkpoints from before --hidden/--layers are 2 layers of 512.
    model = Policy(meta["obs_size"], meta["action_size"], hidden=meta.get("hidden", 512), aux=meta.get("aux", 0), extra_in=meta.get("extra_in", 0),
                   layers=meta.get("layers", 2), spots=spot_tables_of(ck["model"]),
                   spot_head=spot_head_tables_of(ck["model"]), spot_width=spot_head_width_of(ck["model"]),
                   board_heads=board_head_tables_of(ck["model"]), board_width=board_head_width_of(ck["model"]))
    model.load_state_dict(ck["model"])
    data = to_bytes(model, meta["obs_size"], meta["action_size"])
    with open(out_path + ".tmp", "wb") as f:
        f.write(data)
    import os
    os.replace(out_path + ".tmp", out_path)
    return model


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]}")
