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


def spot_head_tables(names, map_type="BASE"):
    """Index tables for model.SpotHead from an observation's names: per land node
    the positions of its NODE/SPOT/STRUCT/REL entries (every node must have the
    same count), its neighbours at road distance 1 (padded to 3 with N) and 2
    (padded to the widest with N), and its settlement action slot."""
    obs_size = len(names)
    node_tiles, edges = catanatron_rs.board_topology(map_type)
    num_nodes = len(node_tiles)
    per_node = [[] for _ in range(num_nodes)]
    for i, name in enumerate(names):
        m = re.fullmatch(r"(?:NODE|SPOT|STRUCT|REL)(\d+)_.*", name)
        if m and int(m.group(1)) < num_nodes:
            per_node[int(m.group(1))].append(i)
    widths = {len(p) for p in per_node}
    if len(widths) != 1 or not per_node[0]:
        raise ValueError(f"per-node entry counts differ across nodes: {sorted(widths)}")
    adj1 = [set() for _ in range(num_nodes)]
    for a, b in edges:
        adj1[a].add(b)
        adj1[b].add(a)
    adj2 = [set(k for m in adj1[n] for k in adj1[m] if k != n and k not in adj1[n]) for n in range(num_nodes)]
    k2 = max(len(s) for s in adj2)
    pad = lambda s, w: sorted(s) + [num_nodes] * (w - len(s))
    return {
        "head_node_feat": np.array(per_node, dtype=np.int64),                      # [N, F]
        "head_adj1": np.array([pad(s, 3) for s in adj1], dtype=np.int64),           # [N, 3]
        "head_adj2": np.array([pad(s, k2) for s in adj2], dtype=np.int64),          # [N, K]
        "head_slot": np.array(catanatron_rs.settlement_slots(map_type), dtype=np.int64),  # [N]
    }


def board_head_tables(names, map_type="BASE"):
    """Index tables for model.BoardHeads (mid-game settlement/city, road and robber scores from one node
    encoder): the spot head's node tables plus, per land edge (board_topology order), its two nodes and its
    EDGE entries; per land tile, its adjacent nodes (padded with N) and its TILE entries; and the action
    slots of every settlement, city, road and robber move (tile x victim)."""
    t = spot_head_tables(names, map_type)
    obs_size = len(names)
    node_tiles, edges = catanatron_rs.board_topology(map_type)
    num_nodes, num_tiles = len(node_tiles), 1 + max(x for ts in node_tiles for x in ts)
    edge = _group(names, r"EDGE(\d+_\d+)_.*")
    tile = _group(names, r"TILE(\d+)_.*")
    edge_feat = [edge[f"{a}_{b}"] for a, b in edges]
    tile_feat = [tile[str(k)] for k in range(num_tiles)]
    if len({len(f) for f in edge_feat}) != 1 or len({len(f) for f in tile_feat}) != 1:
        raise ValueError("edge or tile entry counts differ")
    by_tile = [[] for _ in range(num_tiles)]
    for n, ts in enumerate(node_tiles):
        for k in ts:
            by_tile[k].append(n)
    w = max(len(ns) for ns in by_tile)
    settle, city, road, robber = catanatron_rs.board_slots(map_type)
    if len(road) != len(edges) or len(robber) != num_tiles:
        raise ValueError("slot tables don't match the topology")
    t.update({
        "head_edge_nodes": np.array(edges, dtype=np.int64),                                    # [E, 2]
        "head_edge_feat": np.array(edge_feat, dtype=np.int64),                                 # [E, Fe]
        "head_tile_nodes": np.array([ns + [num_nodes] * (w - len(ns)) for ns in by_tile], dtype=np.int64),  # [T, 6]
        "head_tile_feat": np.array(tile_feat, dtype=np.int64),                                 # [T, Ft]
        "head_city_slot": np.array(city, dtype=np.int64),                                      # [N]
        "head_road_slot": np.array(road, dtype=np.int64),                                      # [E]
        "head_robber_slot": np.array(robber, dtype=np.int64),                                  # [T, 4]
    })
    return t


def tables_for(obs_size, map_type="BASE", **obs_flags):
    """`spot_tables` for the OpenHex 1v1 layout an `obs_size`-input network
    was trained on (obs_flags: VecEnv's opponent_hands etc.)."""
    from catanatron_env import OPENHEX_1V1
    env = catanatron_rs.VecEnv(1, ["R"], map_type=map_type, **obs_flags, **OPENHEX_1V1)
    names = env.observation_names()
    if len(names) != obs_size:
        raise ValueError(f"layout has {len(names)} inputs, the network {obs_size}")
    return spot_tables(names, map_type)
