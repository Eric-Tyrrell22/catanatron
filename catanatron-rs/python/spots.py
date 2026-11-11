"""Index tables for the spot branch (model.SpotBranch): which observation
positions describe each land node ("spot"), each land tile and each land
edge, and which are global (everything else). Built from the observation's
names and the board topology, so the network (Python) and its export (Rust,
which only gathers by these indices) never hard-code the layout.

Per node: its own features (buildings, port, the spots block), the summed
road features of its edges, and its tiles (summed after a shared tile
encoder). A node has 2 or 3 edges and 1 to 3 tiles; missing ones point at a
padding slot (index `obs_size` of an observation with one zero appended, or
the zero row after the tiles).
"""
import re

import numpy as np

import catanatron_rs


def _group(names, pattern):
    """{key: [positions]} for names matching `pattern` (its group 1 is the key)."""
    out = {}
    for i, n in enumerate(names):
        m = re.fullmatch(pattern, n)
        if m:
            out.setdefault(m.group(1), []).append(i)
    return out


def spot_tables(names, map_type="BASE"):
    """Arrays (int64) describing the branch's inputs for an observation with
    these names; see the module docstring."""
    obs_size = len(names)
    node_tiles, edges = catanatron_rs.board_topology(map_type)
    num_nodes, num_tiles = len(node_tiles), 1 + max(t for ts in node_tiles for t in ts)

    node = _group(names, r"NODE(\d+)_.*")
    spot = _group(names, r"SPOT(\d+)_.*")
    tile = _group(names, r"TILE(\d+)_.*")
    edge = _group(names, r"EDGE(\d+_\d+)_.*")
    if len(node) != num_nodes or (spot and len(spot) != num_nodes) or len(tile) != num_tiles or len(edge) != len(edges):
        raise ValueError("observation layout doesn't match the board topology")

    node_feat = np.array([node[str(n)] + spot.get(str(n), []) for n in range(num_nodes)], dtype=np.int64)
    tile_feat = np.array([tile[str(t)] for t in range(num_tiles)], dtype=np.int64)
    edge_feat = np.array([edge[f"{a}_{b}"] for a, b in edges], dtype=np.int64)
    # Node -> its edges / tiles, padded (edge index len(edges) and tile index
    # num_tiles are padding).
    by_node = [[] for _ in range(num_nodes)]
    for e, (a, b) in enumerate(edges):
        by_node[a].append(e)
        by_node[b].append(e)
    node_edges = np.array([es + [len(edges)] * (3 - len(es)) for es in by_node], dtype=np.int64)
    node_tile = np.array([ts + [num_tiles] * (3 - len(ts)) for ts in node_tiles], dtype=np.int64)
    used = set(node_feat.ravel()) | set(tile_feat.ravel()) | set(edge_feat.ravel())
    global_feat = np.array([i for i in range(obs_size) if i not in used], dtype=np.int64)
    # Edge features gathered per node: [node, 3 edges, edge features], padding
    # edges pointing at the zero slot `obs_size`.
    edge_pad = np.vstack([edge_feat, np.full((1, edge_feat.shape[1]), obs_size, dtype=np.int64)])
    node_edge_feat = edge_pad[node_edges]
    return {
        "obs_size": np.array(obs_size),
        "node_feat": node_feat,              # [54, F_node]
        "node_edge_feat": node_edge_feat,    # [54, 3, F_edge]
        "tile_feat": tile_feat,              # [19, F_tile]
        "node_tiles": node_tile,             # [54, 3] (19 = padding)
        "global_feat": global_feat,          # [F_global]
    }


def tables_for(obs_size, map_type="BASE", **obs_flags):
    """`spot_tables` for the OpenHex 1v1 layout an `obs_size`-input network
    was trained on (obs_flags: VecEnv's opponent_hands etc.)."""
    from catanatron_env import OPENHEX_1V1
    env = catanatron_rs.VecEnv(1, ["R"], map_type=map_type, **obs_flags, **OPENHEX_1V1)
    names = env.observation_names()
    if len(names) != obs_size:
        raise ValueError(f"layout has {len(names)} inputs, the network {obs_size}")
    return spot_tables(names, map_type)
