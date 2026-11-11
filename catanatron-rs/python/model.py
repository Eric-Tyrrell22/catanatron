"""Policy/value network shared by train.py, export.py and eval.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MASKED = -1e9
SPOT_TABLES = ("node_feat", "node_edge_feat", "tile_feat", "node_tiles", "global_feat")


class SpotBranch(nn.Module):
    """Board structure: the same small network for every land node ("spot"),
    pooled over the board (spots.py has the index tables).

    Per tile, a shared encoder of its features; per node, a shared two-layer
    network of its own features, its edges' road features (summed) and its
    tiles' encodings (summed); then the mean and max over the 54 nodes. The
    pooled summary joins the trunk's second layer (`join`, before its ReLU)
    and feeds a value tower (with the global features) whose output is added
    to the trunk's value. `join` and the tower's output start at zero, so a
    network given this branch first computes exactly what it did without it.
    """

    def __init__(self, tables, hidden, tile_width=32, node_width=64, tower=128):
        super().__init__()
        for k in SPOT_TABLES:
            self.register_buffer(k, torch.as_tensor(tables[k], dtype=torch.long))
        f_node = self.node_feat.shape[1] + self.node_edge_feat.shape[2]
        self.tile = nn.Linear(self.tile_feat.shape[1], tile_width)
        self.node1 = nn.Linear(f_node, node_width)
        self.node_tiles_in = nn.Linear(tile_width, node_width, bias=False)
        self.node2 = nn.Linear(node_width, node_width)
        self.join = nn.Linear(2 * node_width, hidden, bias=False)
        self.tower = nn.Linear(2 * node_width + len(self.global_feat), tower)
        self.tower_out = nn.Linear(tower, 1)
        nn.init.zeros_(self.join.weight)
        nn.init.zeros_(self.tower_out.weight)
        nn.init.zeros_(self.tower_out.bias)

    def pool(self, obs):
        """(pooled node summary [B, 2 * node_width], global features)."""
        x = F.pad(obs, (0, 1))  # position obs_size: the zero padding slot
        tiles = F.relu(self.tile(x[:, self.tile_feat]))
        tiles = F.pad(tiles, (0, 0, 0, 1))  # tile index 19: padding
        own = torch.cat([x[:, self.node_feat], x[:, self.node_edge_feat].sum(2)], -1)
        h = F.relu(self.node1(own) + self.node_tiles_in(tiles[:, self.node_tiles].sum(2)))
        h = F.relu(self.node2(h))
        return torch.cat([h.mean(1), h.amax(1)], -1), x[:, self.global_feat]

    def value(self, pooled, glob):
        return self.tower_out(F.relu(self.tower(torch.cat([pooled, glob], -1))))


class SplitLinear(nn.Module):
    """A Linear whose trailing `in_extra` inputs have their own weight tensor,
    zero-initialised, so a new input block can get its own learning rate while
    the trained weights keep theirs. Exports as one (out, in_main + in_extra) matrix."""

    def __init__(self, in_main, in_extra, out_features):
        super().__init__()
        self.in_main, self.in_extra, self.out_features = in_main, in_extra, out_features
        lin = nn.Linear(in_main, out_features)
        self.weight, self.bias = lin.weight, lin.bias
        self.extra_weight = nn.Parameter(torch.zeros(out_features, in_extra))

    def forward(self, x):
        return F.linear(x[:, :self.in_main], self.weight, self.bias) + F.linear(x[:, self.in_main:], self.extra_weight)

    def full_weight(self):
        return torch.cat([self.weight, self.extra_weight], 1)


class Policy(nn.Module):
    def __init__(self, obs_size, action_size, hidden=512, aux=0, layers=2, spots=None, extra_in=0):
        """`spots`: the spot branch's index tables (spots.spot_tables), or None.
        `extra_in`: the trailing inputs routed through their own weight tensor
        (`SplitLinear`), for a new input block on a trained network."""
        super().__init__()
        if extra_in and spots is not None:
            raise ValueError("extra inputs and the spot branch together are not supported")
        first = SplitLinear(obs_size - extra_in, extra_in, hidden) if extra_in else nn.Linear(obs_size, hidden)
        trunk = [first, nn.ReLU()]
        for _ in range(layers - 1):
            trunk += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*trunk)
        self.pi = nn.Linear(hidden, action_size)
        self.v = nn.Linear(hidden, 1)
        # Optional training-only head for auxiliary targets (not exported).
        self.aux = nn.Linear(hidden, aux) if aux else None
        # Small initial policy weights => near-uniform over legal moves.
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        if spots is not None and layers < 2:
            raise ValueError("the spot branch joins the trunk's second layer")
        self.spots = SpotBranch(spots, hidden) if spots is not None else None
        if self.spots is not None:
            # The trunk's value is scaled by this (training fades it from 1 to
            # 0, leaving the value to the spot branch's tower); exported.
            self.register_buffer("flat_value_scale", torch.tensor(1.0))

    def body(self, obs):
        """(last hidden layer, value) for a batch of observations."""
        if self.spots is None:
            h = self.trunk(obs)
            return h, self.v(h).float().squeeze(-1)
        pooled, glob = self.spots.pool(obs)
        h, k = obs, 0
        for m in self.trunk:
            h = m(h)
            if isinstance(m, nn.Linear):
                if k == 1:
                    h = h + self.spots.join(pooled)
                k += 1
        v = self.flat_value_scale * self.v(h) + self.spots.value(pooled, glob)
        return h, v.float().squeeze(-1)

    def forward(self, obs, mask):
        h, v = self.body(obs)
        logits = self.pi(h).float().masked_fill(~mask, MASKED)
        return logits, v

    def forward_aux(self, obs, mask):
        """(logits, value, auxiliary predictions)."""
        h, v = self.body(obs)
        logits = self.pi(h).float().masked_fill(~mask, MASKED)
        return logits, v, self.aux(h).float()


def spot_tables_of(state):
    """The spot branch's index tables in a Policy state dict, or None."""
    if "spots.node_feat" not in state:
        return None
    return {k: state[f"spots.{k}"].cpu().numpy() for k in SPOT_TABLES}


def trunk_shape(state):
    """(hidden, layers) of a Policy state dict."""
    weights = [k for k in state if k.startswith("trunk.") and k.endswith(".weight")]
    return state["trunk.0.weight"].shape[0], len(weights)


@torch.no_grad()
def grow_state(state, model):
    """`state` (a smaller Policy's weights) mapped into `model`'s larger shape,
    computing exactly the same function:
      - extra hidden units get the model's fresh random input weights and zero
        weights out of them (into later layers and the heads), so they change
        nothing at first but receive gradients;
      - extra trunk layers are the identity: their inputs are ReLU outputs
        (non-negative), so ReLU(I h) = h.
    Returns the new state dict."""
    old_h, old_layers = trunk_shape(state)
    new = {k: v.clone() for k, v in model.state_dict().items()}
    new_h, new_layers = trunk_shape(new)
    if new_h < old_h or new_layers < old_layers:
        raise ValueError(f"can only grow: {old_layers}x{old_h} -> {new_layers}x{new_h}")
    # First layer: old units keep their weights; new units keep fresh ones.
    w = state["trunk.0.weight"]
    new["trunk.0.weight"][:old_h, :w.shape[1]] = w
    new["trunk.0.weight"][:old_h, w.shape[1]:] = 0
    new["trunk.0.bias"][:old_h] = state["trunk.0.bias"]
    # Later layers: old -> old as before, new units feed nothing old.
    for k in range(1, new_layers):
        name = f"trunk.{2 * k}"
        if k < old_layers:
            new[f"{name}.weight"][:old_h, :old_h] = state[f"{name}.weight"]
            new[f"{name}.weight"][:old_h, old_h:] = 0
            new[f"{name}.bias"][:old_h] = state[f"{name}.bias"]
        else:
            new[f"{name}.weight"].copy_(torch.eye(new_h))
            new[f"{name}.bias"].zero_()
    # Heads read only the old units.
    for head in ("pi", "v", "aux"):
        if f"{head}.weight" in state and f"{head}.weight" in new:
            new[f"{head}.weight"].zero_()
            new[f"{head}.weight"][:, :old_h] = state[f"{head}.weight"]
            new[f"{head}.bias"].copy_(state[f"{head}.bias"])
    return new


@torch.no_grad()
def revive_dead(model, obs, seed=0):
    """Dormant-unit reset, keeping the network's function: every trunk unit
    that never fires on `obs` (a [n, obs_size] float tensor of real positions)
    gets fresh random input weights (and a zero bias) and zero weights out of
    it (into the next trunk layer, or the heads after the last one). Outputs
    are unchanged; the revived units fire on some positions, so their
    outgoing weights get gradients again. Returns the number revived per
    layer."""
    gen = torch.Generator().manual_seed(seed)
    linears = [m for m in model.trunk if isinstance(m, nn.Linear)]
    heads = [h for h in (model.pi, model.v, model.aux) if h is not None]
    h, counts = obs, []
    for k, lin in enumerate(linears):
        pre = h @ lin.weight.T + lin.bias
        dead = (pre.clamp(min=0).amax(0) == 0)
        if dead.any():
            fresh = torch.empty(int(dead.sum()), lin.in_features)
            nn.init.kaiming_uniform_(fresh, a=5 ** 0.5, generator=gen)
            lin.weight[dead] = fresh.to(lin.weight)
            lin.bias[dead] = 0
            for nxt in ([linears[k + 1]] if k + 1 < len(linears) else heads):
                nxt.weight[:, dead] = 0
        counts.append(int(dead.sum()))
        h = (h @ lin.weight.T + lin.bias).clamp(min=0)
    return counts
