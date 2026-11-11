"""Thin numpy wrapper over the Rust `catanatron_rs.VecEnv`."""

from dataclasses import dataclass

import numpy as np

import catanatron_rs


@dataclass
class Step:
    obs: np.ndarray        # (N, obs_size) float32
    mask: np.ndarray       # (N, action_size) bool, True = legal
    reward: np.ndarray     # (N,) float32
    done: np.ndarray       # (N,) bool; obs is then the next game's first
    won: np.ndarray        # (N,) bool, meaningful where done
    truncated: np.ndarray  # (N,) bool, turn limit hit with no winner


class VecEnv:
    """Batched games with one learner seat.

    Opponent key "SELF" makes a seat caller-controlled: pass
    `opponent_fn(obs, mask) -> actions` to `step`/`observe` to play it
    (e.g. with a frozen copy of the policy).
    """

    def __init__(self, num_envs, opponents, seed=0, threads=0, map_type="BASE",
                 vps_to_win=10, vp_reward=0.0, discard_limit=7, friendly_robber="off",
                 dice="fair", numbers="spiral", opponent_hands=False, opponent_devs=False, extra_info=False,
                 spot_features=False, reach_features=False, structure_features=False):
        self._env = catanatron_rs.VecEnv(num_envs, list(opponents), seed, threads,
                                         map_type, vps_to_win, vp_reward, discard_limit,
                                         friendly_robber, dice, numbers, opponent_hands, opponent_devs,
                                         extra_info, spot_features, reach_features, structure_features)
        self.num_envs = self._env.num_envs
        self.obs_size = self._env.obs_size
        self.action_size = self._env.action_size
        self.has_external = "SELF" in opponents

    def _obs(self, raw, rows):
        obs, mask = raw
        return (np.frombuffer(obs, dtype="<f4").reshape(rows, self.obs_size),
                np.frombuffer(mask, dtype=np.uint8).reshape(rows, self.action_size).astype(bool))

    def _resolve(self, opponent_fn):
        """Plays external seats until every env waits on the learner."""
        while True:
            idxs = self._env.pending_opponents()
            if not idxs:
                return
            if opponent_fn is None:
                raise ValueError("env has SELF seats; pass opponent_fn")
            obs, mask = self._obs(self._env.observe_envs(idxs), len(idxs))
            actions = opponent_fn(obs, mask)
            self._env.act_opponents(idxs, [int(a) for a in actions])

    def observe(self, opponent_fn=None) -> Step:
        self._resolve(opponent_fn)
        obs, mask = self._obs(self._env.observe(), self.num_envs)
        n = self.num_envs
        z = np.zeros(n, dtype=bool)
        return Step(obs, mask, np.zeros(n, dtype=np.float32), z, z.copy(), z.copy())

    def step(self, actions, opponent_fn=None) -> Step:
        actions = [int(a) for a in actions]
        if not self.has_external:
            obs, mask, reward, done, won, truncated = self._env.step(actions)
            n = self.num_envs
            obs, mask = self._obs((obs, mask), n)
            return Step(obs, mask, np.frombuffer(reward, dtype="<f4"),
                        np.frombuffer(done, dtype=np.uint8).astype(bool),
                        np.frombuffer(won, dtype=np.uint8).astype(bool),
                        np.frombuffer(truncated, dtype=np.uint8).astype(bool))
        self._env.act(actions)
        self._resolve(opponent_fn)
        reward, done, won, truncated = self._env.finish()
        self._resolve(opponent_fn)  # new games may open with an external seat
        obs, mask = self._obs(self._env.observe(), self.num_envs)
        return Step(obs, mask, np.frombuffer(reward, dtype="<f4"),
                    np.frombuffer(done, dtype=np.uint8).astype(bool),
                    np.frombuffer(won, dtype=np.uint8).astype(bool),
                    np.frombuffer(truncated, dtype=np.uint8).astype(bool))

    def set_policy(self, ctnn_bytes):
        """Network the Rust side samples learner moves from in `rollout`."""
        self._env.set_policy(ctnn_bytes)

    def rollout_buffers(self, steps, empty=np.empty):
        """Output arrays for `rollout`; `empty(shape, dtype)` allocates each
        (e.g. to hand out numpy views of pinned torch tensors)."""
        return rollout_buffers(self.num_envs, steps, self.obs_size, self.action_size, empty)

    def rollout(self, steps, out):
        """Plays `steps` learner moves per env entirely in Rust, writing into
        `out` (from `rollout_buffers`). Arrays are (num_envs, steps, ...);
        `last_value` is (num_envs,). Boolean fields are uint8 0/1."""
        self._env.rollout(steps, *(out[k] for k in ROLLOUT_FIELDS))
        return out

    def set_opponents(self, opponents):
        self._env.set_opponents(list(opponents))
        self.has_external = "SELF" in opponents

    def observation_names(self):
        return self._env.observation_names()


COLONIST_1V1 = {"vps_to_win": 15, "discard_limit": 9, "friendly_robber": "public", "dice": "deck", "numbers": "spiral"}
# OpenHex's 1v1 ladder (github.com/bovard/open-hex, SPEC.md section 12 and dice.py).
OPENHEX_1V1 = {"vps_to_win": 15, "discard_limit": 9, "friendly_robber": "public", "dice": "openhex",
               "numbers": "spiral-random-corner"}


def board_spec(board):
    """An OpenHex board ({"tiles": [...], "ports": [...]}, as the live API
    gives it) in the compact text form the engine takes as `board=`."""
    v = lambda d: f"{d['x']},{d['y']},{d['z']}"
    tiles = ";".join(f"{v(t['hex'])},{t['terrain']},{t.get('number') or '-'}" for t in board["tiles"])
    ports = ";".join(f"{v(p['edge'][0])},{v(p['edge'][1])},{p.get('resource') or '-'}" for p in board["ports"])
    return f"{tiles}|{ports}"


def add_rule_args(ap):
    """Game-rule flags shared by the training and evaluation scripts."""
    g = ap.add_argument_group("rules")
    g.add_argument("--map", default="BASE", choices=["BASE", "MINI"], help="MINI: 7-tile board for quick experiments")
    g.add_argument("--vps-to-win", type=int, default=10)
    g.add_argument("--discard-limit", type=int, default=7, help="discard on a 7 when holding more than this")
    g.add_argument("--friendly-robber", default="off", choices=["off", "actual", "public"],
                   help="actual: catanatron's (points incl. hidden VP cards < 3); public: Colonist's (visible points < 3)")
    g.add_argument("--dice", default="fair", choices=["fair", "deck", "openhex"],
                   help="deck: each combination once per 36 rolls; openhex: OpenHex's balanced dice")
    g.add_argument("--balanced-dice", action="store_true", help="same as --dice deck (older runs' flag)")
    g.add_argument("--numbers", default="spiral", choices=["spiral", "spiral-random-corner", "random"],
                   help="number tokens: official spiral from a fixed corner, from a random corner, or shuffled")
    g.add_argument("--colonist-1v1", action="store_true",
                   help="Colonist ranked 1v1: 15 points, discard above 9, public friendly robber, deck dice")
    g.add_argument("--openhex-1v1", action="store_true",
                   help="OpenHex 1v1 ladder: 15 points, discard above 9, public friendly robber, OpenHex dice, "
                        "spiral from a random corner")


def rules_from_args(args):
    """Keyword arguments for VecEnv / play_bots / self_play."""
    if args.colonist_1v1 and args.openhex_1v1:
        raise SystemExit("--colonist-1v1 and --openhex-1v1 are different rule sets; pick one")
    if args.colonist_1v1:
        return {"map_type": args.map, **COLONIST_1V1}
    if args.openhex_1v1:
        return {"map_type": args.map, **OPENHEX_1V1}
    if args.balanced_dice and args.dice != "fair":
        raise SystemExit("--balanced-dice is --dice deck; don't combine it with --dice")
    return {"map_type": args.map, "vps_to_win": args.vps_to_win, "discard_limit": args.discard_limit,
            "friendly_robber": args.friendly_robber, "dice": "deck" if args.balanced_dice else args.dice,
            "numbers": args.numbers}


ROLLOUT_FIELDS = ("obs", "mask", "action", "logp", "value", "reward", "done", "won", "truncated", "last_value")


def rollout_buffers(num_envs, steps, obs_size, action_size, empty=np.empty):
    """Arrays `VecEnv.rollout` writes into, for `num_envs` envs. Slicing the
    first axis gives the buffers for a subset of envs."""
    n, t = num_envs, steps
    return {
        "obs": empty((n, t, obs_size), np.float32),
        "mask": empty((n, t, action_size), np.uint8),
        "action": empty((n, t), np.uint32),
        "logp": empty((n, t), np.float32),
        "value": empty((n, t), np.float32),
        "reward": empty((n, t), np.float32),
        "done": empty((n, t), np.uint8),
        "won": empty((n, t), np.uint8),
        "truncated": empty((n, t), np.uint8),
        "last_value": empty((n,), np.float32),
    }


play_bots = catanatron_rs.play_bots
