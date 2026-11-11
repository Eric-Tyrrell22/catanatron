"""AlphaZero-style training: self-play with search, learn from the search.

    .venv/bin/python az.py --run az1 --init runs/long2/model.pt --minutes 600 --device cuda --bf16
    .venv/bin/python az.py --run az1 --resume --minutes 600 --device cuda --bf16

Each iteration plays --games self-play games in Rust (both seats search with
the current network; see src/selfplay.rs), adds the fully searched positions
to a replay buffer on the device, and trains the network toward the search's
completed-Q policy target and the game result, optionally mixed with the
search's value of the position (--value-q-weight). Every
--eval-every iterations the network plays F without search, and with
--eval-sims sims of search. The replay buffer is not saved: a resumed run
refills it.

With --opening-net a second network takes the initial placements: it starts
as a copy of the main one, learns from the placement decisions only (its own
replay and optimiser, omodel.pt / oreplay.pt), is exported next to each
snapshot as osnap_*.ctnn (a player key "snap.ctnn@osnap.ctnn" uses both) and
published to the remote actors as control["opening_model"].
"""

import argparse
import glob
import json
import os
import signal
import time

# numpy's idle BLAS threads would spin on cores the Rust games need.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

import catanatron_rs
import shards
from catanatron_env import VecEnv, add_rule_args, board_spec, rules_from_args
from export import to_bytes
import spots
from model import spot_tables_of, Policy, grow_state, revive_dead, trunk_shape

# OBS depends on the rules (balanced dice adds deck features); set in main().
OBS, ACTIONS = None, None  # set from a probe environment with the run's rules
AUX = 28  # catanatron::selfplay::AUX_TARGETS, checked against each self-play batch
AUX_BASE = 8  # the game-level targets; the rest are per action type (--aux-extra-weight)


def enough_space(path, need_bytes):
    """False (with a warning) when the filesystem holding `path` can't take `need_bytes` plus a margin:
    a failed write mid-save would otherwise kill the learner (it did once the disk filled with replay buffers)."""
    import shutil
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
    if free < need_bytes + 512 * 2**20:
        print(f"not saving {path}: {free / 2**30:.1f} GB free, need {need_bytes / 2**30:.1f} GB (free the disk: runs/free_disk.sh)", flush=True)
        return False
    return True


def save(path, model, opt, meta):
    tmp = path + ".tmp"
    if not enough_space(path, 4 * sum(p.numel() for p in model.parameters()) * 3):  # weights + Adam moments
        return
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "meta": meta}, tmp)
    os.replace(tmp, path)


class Replay:
    """Fixed-capacity ring buffer of examples, kept on `device` (the training
    device, or the CPU for a buffer larger than the GPU's memory: minibatches
    then move to `train_device` as they're sampled)."""

    FIELDS = ("obs", "mask", "pi", "z", "surprise", "aux", "opening", "focus", "vw")

    def save(self, path):
        """Writes the filled part of the buffer as it lies (no reordered copy:
        the buffer may be most of the machine's memory) with the index of its
        oldest example, atomically."""
        data = {k: getattr(self, k)[:self.size] for k in self.FIELDS}
        data["oldest"] = self.next if self.size == self.capacity else 0
        if not enough_space(path, sum(t.numel() * t.element_size() for t in data.values() if torch.is_tensor(t))):
            return
        tmp = path + ".tmp"
        torch.save(data, tmp)
        os.replace(tmp, path)

    def load(self, path):
        # Memory-mapped: the file is read field by field into the buffer.
        data = torch.load(path, map_location="cpu", mmap=True)
        if data["obs"].shape[1] != self.obs.shape[1] or data["aux"].shape[1] != self.aux.shape[1]:
            raise SystemExit(f"{path} has a different observation or aux size than this run")
        total, oldest = data["z"].shape[0], data["oldest"]
        n = min(total, self.capacity)
        # Oldest first: src[oldest:] then src[:oldest]; the oldest `total - n`
        # examples don't fit and are skipped.
        order = [(oldest, total), (0, oldest)]
        for k in self.FIELDS:
            buf, src = getattr(self, k), data[k]
            skip, at = total - n, 0
            for lo, hi in order:
                drop = min(skip, hi - lo)
                lo, skip = lo + drop, skip - drop
                buf[at:at + hi - lo] = src[lo:hi].to(buf.device, buf.dtype)
                at += hi - lo
        self.size, self.next = n, n % self.capacity
        self.opening_idx = torch.nonzero(self.opening[:self.size]).squeeze(1)
        self.focus_idx = torch.nonzero(self.focus[:self.size]).squeeze(1)

    def __init__(self, capacity, device, train_device):
        self.capacity, self.size, self.next = capacity, 0, 0
        self.train_device = train_device
        self.obs = torch.zeros((capacity, OBS), dtype=torch.float16, device=device)
        self.mask = torch.zeros((capacity, ACTIONS), dtype=torch.bool, device=device)
        # Half precision: a target's probabilities don't need more, and it
        # roughly halves the buffer (so more iterations fit on the GPU).
        self.pi = torch.zeros((capacity, ACTIONS), dtype=torch.float16, device=device)
        self.z = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.surprise = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.aux = torch.zeros((capacity, AUX), dtype=torch.float32, device=device)
        self.opening = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.opening_idx = torch.zeros(0, dtype=torch.long, device=device)
        self.focus = torch.zeros(capacity, dtype=torch.bool, device=device)
        # Value-loss weight per example (--value-positions: each game counts about that many positions).
        self.vw = torch.ones(capacity, dtype=torch.float16, device=device)
        self.focus_idx = torch.zeros(0, dtype=torch.long, device=device)

    def add(self, obs, mask, pi, z, surprise, aux, opening, focus, vw):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.obs.device) + self.next) % self.capacity
        self.obs[idx] = obs.to(self.obs.device, torch.float16)
        self.mask[idx] = mask.to(self.mask.device).bool()
        self.pi[idx] = pi.to(self.pi.device, self.pi.dtype)
        self.z[idx] = z.to(self.z.device)
        self.surprise[idx] = surprise.to(self.surprise.device)
        self.aux[idx] = aux.to(self.aux.device)
        self.opening[idx] = opening.to(self.opening.device).bool()
        self.focus[idx] = focus.to(self.focus.device).bool()
        self.vw[idx] = vw.to(self.vw.device, torch.float16)
        self.next = (self.next + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.opening_idx = torch.nonzero(self.opening[:self.size]).squeeze(1)
        self.focus_idx = torch.nonzero(self.focus[:self.size]).squeeze(1)

    def sample(self, n, sym=None, surprise_share=0.0, opening_share=0.0, focus_share=0.0):
        """A random minibatch; a `surprise_share` of it is drawn in proportion
        to how much search changed the network's policy (KataGo's policy
        surprise weighting), an `opening_share` uniformly from the opening
        decisions (initial placements, ~2% of examples otherwise), the rest
        uniformly. With `sym` = (obs perms, action perms), each example gets a
        random board symmetry."""
        k = round(n * surprise_share)
        o = round(n * opening_share) if len(self.opening_idx) else 0
        f = round(n * focus_share) if len(self.focus_idx) else 0
        i = torch.randint(0, self.size, (n - k - o - f,), device=self.obs.device)
        if f > 0:
            j = self.focus_idx[torch.randint(0, len(self.focus_idx), (f,), device=self.obs.device)]
            i = torch.cat([i, j])
        if k > 0:
            j = torch.multinomial(self.surprise[:self.size] + 1e-6, k, replacement=True)
            i = torch.cat([i, j])
        if o > 0:
            j = self.opening_idx[torch.randint(0, len(self.opening_idx), (o,), device=self.obs.device)]
            i = torch.cat([i, j])
        dev = self.train_device
        obs, mask, pi, z, aux, vw = (self.obs[i].to(dev).float(), self.mask[i].to(dev), self.pi[i].to(dev).float(),
                                     self.z[i].to(dev), self.aux[i].to(dev), self.vw[i].to(dev).float())
        if sym is not None:
            k = torch.randint(0, sym[0].shape[0], (n,), device=obs.device)
            obs = obs.gather(1, sym[0][k])
            mask, pi = mask.gather(1, sym[1][k]), pi.gather(1, sym[1][k])
        return obs, mask, pi, z, aux, vw


def evaluate(run_dir, model, iteration, args, board, omodel=None):
    path = os.path.abspath(os.path.join(run_dir, f"snap_{iteration:06d}.ctnn"))
    if omodel is not None:
        # Written before the snapshot, so a scorer that sees the snapshot finds its opening network.
        opath = os.path.abspath(os.path.join(run_dir, f"osnap_{iteration:06d}.ctnn"))
        with open(opath, "wb") as f:
            f.write(to_bytes(omodel, OBS, ACTIONS))
    with open(path, "wb") as f:
        f.write(to_bytes(model, OBS, ACTIONS))
    if omodel is not None:
        path = f"{path}@{opath}"
    out = {}
    # Search values finished games the way training does.
    utility = (f",m{args.margin_weight}" if args.margin_weight else "") + \
              (f",s{args.speed_weight}/{args.speed_turns}" if args.speed_weight else "")
    search = f"MCTS:{args.eval_sims}{utility}:{path}"
    for label, key, opponent, games in [("net", f"NN:{path}", "F", args.eval_games),
                                        (f"mcts{args.eval_sims}", search, "F", args.eval_search_games),
                                        ("vsR", search, "R", args.eval_random_games)]:
        if games <= 0:
            continue
        (wins, _), capped = catanatron_rs.play_bots([key, opponent], games, 777_000, args.threads, board=board,
                                                    **rules_from_args(args))
        out[label] = wins / games
        out[label + "_capped"] = capped / games
    return out


class Remote:
    """Self-play on remote actors (actor.py) through S3: publishes the network
    each iteration and collects the shards they upload."""

    def __init__(self, bucket, run, games_per_core, max_lag, max_shards, teachers=(), teacher_share=0.0, control_key=shards.CONTROL_KEY):
        self.control_key = control_key
        import boto3
        self.s3 = boto3.client("s3")
        self.bucket, self.run, self.games_per_core, self.max_lag = bucket, run, games_per_core, max_lag
        self.max_shards = max_shards
        self.uploaded = set()  # snapshot keys already in the bucket
        self.model_key = None
        self.opening_key = None
        # Teachers (board specialists) play a teacher_share of actor rounds on
        # their boards; their networks are uploaded once.
        self.teacher_share = teacher_share
        self.teachers = []
        for t in teachers:
            k = f"models/{run}/teacher_{t['board'].replace(':', '_')}.ctnn"
            with open(t["path"], "rb") as f:
                self.s3.put_object(Bucket=bucket, Key=k, Body=f.read())
            self.teachers.append({"key": k, "board": t["board"]})

    def publish(self, iteration, net, pool_paths, pool_share, self_play_kwargs, opening=None):
        """`opening`: the opening network's bytes (--opening-net), which the actors attach to `net`."""
        key = f"models/{self.run}/model_{iteration:06d}.ctnn"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=net)
        okey = None
        if opening is not None:
            okey = f"models/{self.run}/opening_{iteration:06d}.ctnn"
            self.s3.put_object(Bucket=self.bucket, Key=okey, Body=opening)
        pool = []
        for p in pool_paths:
            k = f"models/{self.run}/{os.path.basename(p)}"
            if k not in self.uploaded:
                with open(p, "rb") as f:
                    self.s3.put_object(Bucket=self.bucket, Key=k, Body=f.read())
                self.uploaded.add(k)
            pool.append(k)
        control = {"run": self.run, "iter": iteration, "model": key, "pool": pool, "pool_share": pool_share,
                   "games_per_core": self.games_per_core, "obs_size": OBS, "action_size": ACTIONS,
                   "self_play": self_play_kwargs, "teachers": self.teachers, "teacher_share": self.teacher_share}
        if okey is not None:
            control["opening_model"] = okey
        self.s3.put_object(Bucket=self.bucket, Key=self.control_key, Body=json.dumps(control).encode())
        # Actors that are mid-round already hold the old networks.
        if self.model_key is not None and self.model_key != key:
            self.s3.delete_object(Bucket=self.bucket, Key=self.model_key)
        self.model_key = key
        if self.opening_key is not None and self.opening_key != okey:
            self.s3.delete_object(Bucket=self.bucket, Key=self.opening_key)
        self.opening_key = okey

    def collect(self, iteration):
        """Downloads (and deletes) the newest max_shards shards of this run;
        shards from a network more than max_lag iterations old are deleted
        unread. The rest wait for later iterations (a backlog, e.g. from
        actors playing while the learner was stopped, is worked off a few
        shards at a time, and what grows too old is dropped).
        Returns (list of (header, arrays), number dropped)."""
        got, dropped, fresh = [], 0, []
        pages = self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=f"shards/{self.run}/")
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if iteration - int(key.split("/")[2]) <= self.max_lag:
                    fresh.append((obj["LastModified"], key))
                else:
                    self.s3.delete_object(Bucket=self.bucket, Key=key)
                    dropped += 1
        for _, key in sorted(fresh, reverse=True)[:self.max_shards]:
            header, arrays = shards.decode(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())
            if (header["obs_size"], header["action_size"], header["aux_size"]) != (OBS, ACTIONS, AUX):
                raise SystemExit(f"{key}: shard sizes {header['obs_size']}/{header['action_size']}/"
                                 f"{header['aux_size']} don't match this run's {OBS}/{ACTIONS}/{AUX}")
            got.append((header, arrays))
            self.s3.delete_object(Bucket=self.bucket, Key=key)
        return got, dropped


def main():
    global OBS, ACTIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", help="PPO or AZ checkpoint to start from (model.pt)")
    ap.add_argument("--from-scratch", action="store_true", help="start a new run from random weights")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--games", type=int, default=512, help="self-play games per iteration")
    ap.add_argument("--sims-full", type=int, default=800)
    ap.add_argument("--sims-fast", type=int, default=32)
    ap.add_argument("--p-full", type=float, default=0.25)
    ap.add_argument("--sims-placement", type=int, default=0,
                    help="sims for initial placements, always searched fully and recorded (0 = --sims-full)")
    ap.add_argument("--temp-moves", type=int, default=15)
    ap.add_argument("--root-noise", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--fpu", type=float, default=0.1, help="first-play urgency reduction")
    ap.add_argument("--value-positions", type=float, default=0.0,
                    help="value loss weights each example by min(1, this / examples in its game), so a game's outcome "
                         "counts about this many positions (AlphaGo used 1 per game); 0 = every position counts")
    ap.add_argument("--value-q-weight", type=float, default=0.0,
                    help="value target = (1-w) * game result + w * search value of the position")
    ap.add_argument("--replay", type=int, default=300_000, help="replay buffer capacity (examples)")
    ap.add_argument("--replay-device", help="where the replay buffer lives (default: --device); "
                                            "'cpu' for a buffer larger than the GPU's memory")
    ap.add_argument("--reuse", type=float, default=4.0, help="training samples per new example")
    ap.add_argument("--opening-share", type=float, default=0.0,
                    help="share of each batch drawn from opening decisions (initial settlements and roads)")
    ap.add_argument("--surprise-share", type=float, default=0.0,
                    help="share of each batch sampled in proportion to policy surprise, KL(target || prior)")
    ap.add_argument("--escalate-gap", type=float, default=0.0,
                    help="redo a cheap search fully (and record it) when its top two moves' values are this close (0 = off)")
    ap.add_argument("--endgame-vp", type=int, default=0,
                    help="branch each game at the first position where a player has this many points (0 = off)")
    ap.add_argument("--endgame-branches", type=int, default=0,
                    help="extra continuations played from that position, with fresh dice and dev deck")
    ap.add_argument("--opening-branches", type=int, default=0,
                    help="extra continuations from the position after the initial placements; placements get the average result")
    ap.add_argument("--placement-forks", type=int, default=0,
                    help="replays per game from one of its settlement placements with a different settlement")
    ap.add_argument("--placement-temp", type=float, default=1.0,
                    help="temperature for the forced settlement (higher = more unusual openings)")
    ap.add_argument("--scarce-ows-share", type=float, default=0.0,
                    help="share of self-play games on real boards with scarce ore, wheat and sheep (map::BoardStyle)")
    ap.add_argument("--port-start-share", type=float, default=0.0,
                    help="share of self-play games on real boards with a strong 2:1 port spot (map::BoardStyle)")
    ap.add_argument("--teachers", default=None,
                    help="board specialists to distill: a specialists.py results.jsonl; each successful specialist's "
                         "best snapshot plays a share of the remote actors' rounds on its own board (needs --remote-bucket)")
    ap.add_argument("--teacher-share", type=float, default=0.25,
                    help="share of remote actor rounds that are teacher rounds")
    ap.add_argument("--fixed-board", default=None,
                    help="every self-play and evaluation game on this board: seed:<n> (the board a game with seed n "
                         "draws, as examples/board_screen numbers them), seeds:<n>,<m>,... (each game on one of those), "
                         "group:<resource> (random boards whose scarcest resource is that one), a JSON file holding an OpenHex board, or a "
                         "saved OpenHex game with one under \"board\" (recorded in the run as board.json)")
    ap.add_argument("--random-share", type=float, default=0.0,
                    help="share of game slots played as a seat-swapped pair against a noisy opponent (not recorded)")
    ap.add_argument("--random-eps", type=float, default=1.0,
                    help="how often the noisy opponent plays a random legal move instead of searching (1 = random bot)")
    ap.add_argument("--eval-random-games", type=int, default=200,
                    help="evaluation games with search against the random bot R (0 = none)")
    ap.add_argument("--move-fork-temp", type=float, default=1.0,
                    help="temperature for a move fork's alternative (1 = the search target: mostly runner-up moves)")
    ap.add_argument("--move-fork-uniform", type=float, default=0.0, help="share of move forks drawn uniformly from the other legal moves")
    ap.add_argument("--move-forks", type=int, default=0,
                    help="replay each game this many times from a searched mid-game road or robber decision with a "
                         "different choice (drawn with --placement-temp / --placement-uniform): outcome data for alternatives")
    ap.add_argument("--placement-uniform", type=float, default=0.0,
                    help="share of placement forks whose settlement or road is any other legal choice, uniformly")
    ap.add_argument("--margin-weight", type=float, default=0.0,
                    help="finished games are worth result + this * final point margin / points to win (search and targets)")
    ap.add_argument("--speed-weight", type=float, default=0.0,
                    help="winner's bonus for a quick win: this * exp(-turns / --speed-turns), taken from the loser")
    ap.add_argument("--speed-turns", type=float, default=100.0,
                    help="quick-win bonus decay: speed weight * exp(-turns / speed turns)")
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="loss weight of an auxiliary head predicting point margin, points, turns left, "
                         "road/army holders and later search values (0 = no head)")
    ap.add_argument("--aux-extra-weight", type=float, default=0.0,
                    help="loss weight of the per-action-type auxiliary targets (future roads, settlements, cities, "
                         "dev buys, knights, maritime trades and robber moves of both sides; final settlements, "
                         "cities and VP cards); 0 = those outputs untrained")
    ap.add_argument("--pool-share", type=float, default=0.0,
                    help="share of game slots played as a seat-swapped pair against a past snapshot of this run")
    ap.add_argument("--pool-size", type=int, default=6, help="the most recent this-many evaluated snapshots form the pool")
    ap.add_argument("--pool-min-iter", type=int, default=20, help="snapshots before this iteration stay out of the pool")
    ap.add_argument("--luck", action="store_true",
                    help="value target is result - c * (luck of later dice, draws and steals), with c fit each "
                         "iteration as cov(result, luck) / var(luck) (the variance-minimizing control-variate weight)")
    ap.add_argument("--symmetry", action="store_true",
                    help="train on random rotations/reflections of the board (12 symmetries)")
    ap.add_argument("--min-replay", type=int, default=100_000, help="examples gathered before training starts")
    ap.add_argument("--entropy-budget", action="store_true",
                    help="self-play spends its full searches where the network's policy entropy is highest (same "
                         "share of full searches as --p-full, placed by uncertainty instead of at random)")
    ap.add_argument("--focus-sims", type=int, default=0,
                    help="a fully searched road/settlement/robber/trade decision is searched again with this many "
                         "simulations and recorded with the deeper target (0 = off)")
    ap.add_argument("--focus-share", type=float, default=0.0,
                    help="share of each training batch drawn from focus decisions (like --opening-share)")
    ap.add_argument("--kl-escalate", action="store_true",
                    help="self-play runs the fast search first and escalates to the full search when the search's "
                         "KL from the network's prior is high (threshold steered to keep --p-full of decisions full)")
    ap.add_argument("--admit-kl", type=float, default=0.0,
                    help="also record fast-search decisions whose search-vs-prior KL exceeds this (0 = off)")
    ap.add_argument("--spot-branch", action="store_true",
                    help="add the board-structure branch (model.SpotBranch) to the --init network, joined through zero "
                         "weights so it starts out computing exactly the --init network")
    ap.add_argument("--spot-lr", type=float, default=5e-4, help="learning rate of the spot branch's parameters")
    ap.add_argument("--value-fade", type=int, default=0, metavar="GAMES",
                    help="with the spot branch: fade the trunk's value from 1 to 0 over this many games, leaving the "
                         "value to the branch's tower (0 = keep both)")
    ap.add_argument("--revive-dead", type=int, default=0, metavar="GAMES",
                    help="new run: reset trunk units that never fire on positions from this many quick games "
                         "(fresh input weights, zero output weights: same outputs to start)")
    ap.add_argument("--hidden", type=int, default=512, help="trunk width (a new run's --init network is grown to it)")
    ap.add_argument("--layers", type=int, default=2, help="trunk layers (likewise grown from --init)")
    ap.add_argument("--opening-net", action="store_true",
                    help="a separate network takes the initial placements: it starts as a copy of the main network, learns "
                         "from the placement decisions only (own replay and optimiser; omodel.pt), is exported beside each "
                         "snapshot as osnap_*.ctnn and plays the openings in local and remote self-play")
    ap.add_argument("--opening-hidden", type=int, default=0, help="opening network's trunk width (0 = the main network's; grown from its copy)")
    ap.add_argument("--opening-layers", type=int, default=0, help="opening network's trunk layers (0 = the main network's)")
    ap.add_argument("--opening-lr", type=float, default=0.0, help="opening network's learning rate (0 = --lr)")
    ap.add_argument("--opening-replay", type=int, default=50_000, help="opening replay capacity (placement decisions)")
    ap.add_argument("--opening-batch", type=int, default=512)
    ap.add_argument("--opening-reuse", type=float, default=8.0, help="opening training samples per new placement decision")
    ap.add_argument("--opening-min-replay", type=int, default=2_000,
                    help="placement decisions gathered before the opening network trains (it plays as the copy until then)")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--vf", type=float, default=1.0, help="value loss weight")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=2000)
    ap.add_argument("--eval-sims", type=int, default=100)
    ap.add_argument("--eval-search-games", type=int, default=500)
    ap.add_argument("--threads", type=int, default=0, help="Rust threads (0 = all cores)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opponent-hands", action="store_true",
                    help="network sees opponents' hands by resource (public knowledge in 1v1)")
    ap.add_argument("--opponent-devs", action="store_true",
                    help="network sees development-card tracking (unseen pool, expected opponent cards, holding streaks)")
    ap.add_argument("--spot-features", action="store_true",
                    help="network sees per-node settlement spots (pips, legality, road closeness) and port discounts")
    ap.add_argument("--reach-lr-mult", type=float, default=0.0,
                    help="with --reach-features: route the reach block through its own zero-initialised weight tensor "
                         "trained at this multiple of --lr (0 = plain zero-padded columns at --lr)")
    ap.add_argument("--reach-features", action="store_true",
                    help="network sees per-node road reach for both sides, secure/contested spots, far spots, a greedy "
                         "settle plan and affordability (the relational quantities an expansion plan is made of)")
    ap.add_argument("--structure-features", action="store_true",
                    help="network sees the board's pips per resource and, per spot, its share of each resource's supply, "
                         "its road/city/settlement resource pairs, 6s and 8s, duplicate numbers, tile count and coast "
                         "(what the board specialists' first picks track; board-constant)")
    ap.add_argument("--opening-init", default=None,
                    help="with --opening-net: start the opening network from this omodel.pt (inputs widened with zero weights "
                         "for new feature blocks) instead of from a copy of the main network")
    ap.add_argument("--sibling-labels", action="store_true",
                    help="a position's value target is the mean result over every continuation sharing it (main game, "
                         "opening branches, placement and move forks from the search target), luck-adjusted each")
    ap.add_argument("--hard-boards", type=int, default=0,
                    help="prioritised board sampling: keep this many board seeds with the largest opening value error "
                         "(|luck-adjusted result - search value| at placements) and play --hard-share of games on them")
    ap.add_argument("--hard-share", type=float, default=0.25, help="share of games on the prioritised boards")
    ap.add_argument("--extra-info", action="store_true",
                    help="network sees tile numbers (+ rolls left), the first player, opponents' this-turn dev activity")
    ap.add_argument("--remote-bucket", help="also learn from remote actors' self-play (actor.py) through this S3 bucket")
    ap.add_argument("--remote-control-key", default=shards.CONTROL_KEY,
                    help="S3 key of this learner's control file (actors started with --control <key> serve it)")
    ap.add_argument("--remote-games-per-core", type=int, default=8, help="games per actor core in each remote shard")
    ap.add_argument("--remote-max-lag", type=int, default=4,
                    help="remote shards played by a network more than this many iterations old are dropped")
    ap.add_argument("--remote-max-shards", type=int, default=12,
                    help="at most this many remote shards (the newest) are learned from per iteration")
    add_rule_args(ap)
    args = ap.parse_args()

    probe = VecEnv(1, ["R"], opponent_hands=args.opponent_hands, opponent_devs=args.opponent_devs,
                   extra_info=args.extra_info, spot_features=args.spot_features, reach_features=args.reach_features,
                   structure_features=args.structure_features, **rules_from_args(args))
    OBS, ACTIONS = probe.obs_size, probe.action_size
    extra_in = 0
    if args.reach_lr_mult > 0:
        if not args.reach_features:
            raise SystemExit("--reach-lr-mult needs --reach-features")
        if args.structure_features:
            raise SystemExit("--reach-lr-mult routes the trailing inputs; the structure block comes after the reach block")
        extra_in = OBS - VecEnv(1, ["R"], opponent_hands=args.opponent_hands, opponent_devs=args.opponent_devs,
                                extra_info=args.extra_info, spot_features=args.spot_features, **rules_from_args(args)).obs_size
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = os.path.join("runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    board = None
    if args.fixed_board and args.fixed_board.startswith(("seed:", "seeds:", "group:")):
        board = args.fixed_board
        with open(os.path.join(run_dir, "board.json"), "w") as f:
            json.dump({"spec": board}, f)
        kind = {"seed": "on board", "seeds": f"on a board from a pool of {board.count(',') + 1}:",
                "group": "on random boards of group"}[board.split(":")[0]]
        print(f"every game {kind} {board[:60]}{'...' if len(board) > 60 else ''}", flush=True)
    elif args.fixed_board:
        with open(args.fixed_board) as f:
            loaded = json.load(f)
        loaded = loaded.get("board", loaded)
        board = board_spec(loaded)
        with open(os.path.join(run_dir, "board.json"), "w") as f:
            json.dump(loaded, f)
        print(f"every game on the board in {args.fixed_board}", flush=True)
    ckpt_path = os.path.join(run_dir, "model.pt")

    if args.resume:
        # A resumed run keeps its own network shape.
        ck = torch.load(ckpt_path, map_location=device)
        args.hidden, args.layers = ck["meta"].get("hidden", 512), ck["meta"].get("layers", 2)
    if args.resume:
        spot_tables = spot_tables_of(ck["model"])
    elif args.spot_branch:
        spot_tables = spots.tables_for(OBS, map_type=args.map, opponent_hands=args.opponent_hands,
                                       opponent_devs=args.opponent_devs, extra_info=args.extra_info,
                                       spot_features=args.spot_features)
    else:
        spot_tables = None
    if args.value_fade and spot_tables is None:
        raise SystemExit("--value-fade needs the spot branch")
    model = Policy(OBS, ACTIONS, hidden=args.hidden, aux=AUX if args.aux_weight > 0 or args.aux_extra_weight > 0 else 0, layers=args.layers,
                   spots=spot_tables, extra_in=extra_in).to(device)
    # The spot branch (new, starting from zero joins) and the extra input block get their own rates.
    branch = [p for n, p in model.named_parameters() if n.startswith("spots.")]
    extra = [p for n, p in model.named_parameters() if n.endswith("extra_weight")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("spots.") and not n.endswith("extra_weight")]
    groups, group_lrs = [{"params": rest, "lr": args.lr}], [args.lr]
    if extra:
        groups.append({"params": extra, "lr": args.lr * args.reach_lr_mult}); group_lrs.append(args.lr * args.reach_lr_mult)
        print(f"reach block: {extra_in} inputs through their own weights at lr {args.lr * args.reach_lr_mult:g}", flush=True)
    if branch:
        groups.append({"params": branch, "lr": args.spot_lr}); group_lrs.append(args.spot_lr)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    meta = {"iters": 0, "games": 0, "examples": 0, "obs_size": OBS, "action_size": ACTIONS,
            "aux": AUX if args.aux_weight > 0 or args.aux_extra_weight > 0 else 0, "hidden": args.hidden, "layers": args.layers,
            "spot_branch": spot_tables is not None, "extra_in": extra_in,
            "value_fade": {"games": args.value_fade, "start": 0} if args.value_fade else None}
    if args.resume:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        for g, lr in zip(opt.param_groups, group_lrs):  # the saved state would override the rates
            g["lr"] = lr
        meta = ck["meta"]
        print(f"resumed {args.run} at iteration {meta['iters']}, {meta['games']:,} games", flush=True)
    else:
        if os.path.exists(ckpt_path):
            raise SystemExit(f"{ckpt_path} exists: use --resume or a new --run")
        if args.from_scratch == bool(args.init):
            raise SystemExit("a new run needs exactly one of --init or --from-scratch")
    if not args.resume and args.from_scratch:
        meta["init"] = "scratch"
        print(f"starting {args.run} from random weights", flush=True)
    elif not args.resume:
        ck = torch.load(args.init, map_location=device)
        state = dict(ck["model"])
        w = state["trunk.0.weight"]
        if w.shape[1] < OBS - extra_in:
            # New input features (e.g. the balanced-dice deck) get zero
            # weights, so the network starts out playing exactly as before.
            state["trunk.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], OBS - extra_in - w.shape[1])], 1)
            print(f"widened inputs {w.shape[1]} -> {OBS - extra_in} (new weights zero)", flush=True)
        old_shape = trunk_shape(state)
        if old_shape != (args.hidden, args.layers):
            # A bigger network computing exactly the --init one's function.
            if "aux.weight" not in state and model.aux is not None:
                raise SystemExit("growing a network needs the --init checkpoint's auxiliary head")
            state = grow_state(state, model)
            print(f"grew the network {old_shape[1]}x{old_shape[0]} -> {args.layers}x{args.hidden} "
                  f"(same outputs to start)", flush=True)
        if model.aux is not None and "aux.weight" in state and state["aux.weight"].shape[0] < model.aux.out_features:
            # New auxiliary targets: their outputs start at zero.
            old_n, w = state["aux.weight"].shape[0], state["aux.weight"]
            state["aux.weight"] = torch.cat([w, w.new_zeros(model.aux.out_features - old_n, w.shape[1])])
            state["aux.bias"] = torch.cat([state["aux.bias"], w.new_zeros(model.aux.out_features - old_n)])
            print(f"extended the auxiliary head {old_n} -> {model.aux.out_features} targets (new outputs zero)", flush=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        new_head = {k for k in model.state_dict() if k.startswith(("aux.", "spots.")) or k == "flat_value_scale" or k.endswith("extra_weight")}
        if unexpected or set(missing) - new_head:
            raise SystemExit(f"--init checkpoint doesn't match the model: missing {missing}, unexpected {unexpected}")
        if any(k.startswith("aux.") for k in missing):
            print("new auxiliary head (randomly initialized)", flush=True)
        if model.spots is not None:
            print(f"added the spot branch ({sum(p.numel() for p in branch):,} parameters, zero joins: same outputs "
                  f"to start){f'; trunk value fades out over {args.value_fade:,} games' if args.value_fade else ''}", flush=True)
        meta["init"] = {"path": args.init, "meta": ck.get("meta")}
        print(f"starting {args.run} from {args.init}", flush=True)
        if args.revive_dead > 0:
            # Real positions from a few quick games with the network as it is.
            model.eval()
            probe_d = catanatron_rs.self_play(to_bytes(model, OBS, ACTIONS), args.revive_dead, 9_999, args.threads,
                                              sims_full=50, sims_fast=16, **rules_from_args(args))
            probe = torch.from_numpy(np.frombuffer(probe_d["obs"], "<f4").reshape(probe_d["n"], OBS).copy())
            counts = revive_dead(model.cpu(), probe)
            model.to(device)
            meta["revived"] = counts
            print(f"revived dead units per trunk layer {counts} (found on {probe_d['n']} positions; outputs unchanged)", flush=True)

    # ---- The opening network (--opening-net): its own model, optimiser and checkpoint.
    omodel, oopt, ometa, ockpt_path = None, None, None, os.path.join(run_dir, "omodel.pt")
    if args.opening_net:
        ohidden, olayers = args.opening_hidden or args.hidden, args.opening_layers or args.layers
        ock, oinit = None, None
        if args.resume and os.path.exists(ockpt_path):
            ock = torch.load(ockpt_path, map_location=device)
            ohidden, olayers = ock["meta"]["hidden"], ock["meta"]["layers"]
        elif args.opening_init:
            oinit = torch.load(args.opening_init, map_location=device)
            ohidden, olayers = oinit["meta"].get("hidden", ohidden), oinit["meta"].get("layers", olayers)
        omodel = Policy(OBS, ACTIONS, hidden=ohidden, layers=olayers, aux=0, spots=spot_tables, extra_in=extra_in).to(device)
        olr = args.opening_lr or args.lr
        oextra = [p for n, p in omodel.named_parameters() if n.endswith("extra_weight")]
        orest = [p for n, p in omodel.named_parameters() if not n.endswith("extra_weight")]
        ogroups, ogroup_lrs = [{"params": orest, "lr": olr}], [olr]
        if oextra:
            ogroups.append({"params": oextra, "lr": olr * args.reach_lr_mult}); ogroup_lrs.append(olr * args.reach_lr_mult)
        oopt = torch.optim.AdamW(ogroups, lr=olr, weight_decay=args.weight_decay)
        if ock is not None:
            omodel.load_state_dict(ock["model"])
            oopt.load_state_dict(ock["opt"])
            for g, lr in zip(oopt.param_groups, ogroup_lrs):
                g["lr"] = lr
            ometa = ock["meta"]
            print(f"resumed the opening network ({olayers}x{ohidden}) at {ometa['steps']:,} steps", flush=True)
        elif oinit is not None:
            # An earlier run's opening network; new trailing input blocks get zero weights (same outputs to start).
            state = {k: v for k, v in oinit["model"].items() if not k.startswith("aux.")}
            w = state["trunk.0.weight"]
            if w.shape[1] < OBS - extra_in:
                state["trunk.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], OBS - extra_in - w.shape[1])], 1)
                print(f"opening network: widened inputs {w.shape[1]} -> {OBS - extra_in} (new weights zero)", flush=True)
            omodel.load_state_dict(state)
            ometa = {"steps": oinit["meta"].get("steps", 0), "examples": oinit["meta"].get("examples", 0), "obs_size": OBS,
                     "action_size": ACTIONS, "aux": 0, "hidden": ohidden, "layers": olayers, "spot_branch": spot_tables is not None,
                     "extra_in": extra_in, "init": args.opening_init}
            print(f"opening network from {args.opening_init} ({olayers}x{ohidden}, {ometa['steps']:,} steps so far), lr {olr:g}", flush=True)
        else:
            # A copy of the main network as it is now (a resumed run without one
            # gets a copy of its current network), so the openings start unchanged.
            state = {k: v for k, v in model.state_dict().items() if not k.startswith("aux.")}
            if trunk_shape(state) != (ohidden, olayers):
                if spot_tables is not None or extra_in:
                    raise SystemExit("the opening network can only be grown without the spot branch and --reach-lr-mult")
                state = grow_state(state, omodel)
            omodel.load_state_dict(state)
            ometa = {"steps": 0, "examples": 0, "obs_size": OBS, "action_size": ACTIONS, "aux": 0, "hidden": ohidden,
                     "layers": olayers, "spot_branch": spot_tables is not None, "extra_in": extra_in}
            print(f"opening network: {olayers}x{ohidden}, a copy of the main network, lr {olr:g}, "
                  f"trains from {args.opening_min_replay:,} placement decisions", flush=True)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    replay = Replay(args.replay, torch.device(args.replay_device or args.device), device)
    replay_path = os.path.join(run_dir, "replay.pt")
    if args.resume and os.path.exists(replay_path):
        replay.load(replay_path)
        print(f"reloaded {replay.size:,} replay examples", flush=True)
    oreplay, oreplay_path = None, os.path.join(run_dir, "oreplay.pt")
    if omodel is not None:
        oreplay = Replay(args.opening_replay, torch.device(args.replay_device or args.device), device)
        if args.resume and os.path.exists(oreplay_path):
            oreplay.load(oreplay_path)
            print(f"reloaded {oreplay.size:,} opening replay examples", flush=True)

    def opening_bytes():
        return to_bytes(omodel, OBS, ACTIONS) if omodel is not None else None
    # SIGTERM/SIGINT: finish the current iteration, save, then stop.
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
        print(f"stop requested (signal {signum}); finishing this iteration", flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    sym = None
    if args.symmetry:
        perms = catanatron_rs.board_symmetries(OBS, args.map)
        sym = (torch.tensor([p[0] for p in perms], device=device), torch.tensor([p[1] for p in perms], device=device))
        print(f"training with {len(perms)} board symmetries", flush=True)

    # Everything self-play needs besides the networks, seed and threads
    # (remote actors get the same).
    self_play_kwargs = dict(
        sims_full=args.sims_full, sims_fast=args.sims_fast, p_full=args.p_full, sims_placement=args.sims_placement,
        temp_moves=args.temp_moves, root_noise=args.root_noise, dirichlet_alpha=args.alpha, c_puct=args.c_puct,
        fpu_reduction=args.fpu, escalate_gap=args.escalate_gap, entropy_budget=args.entropy_budget, kl_escalate=args.kl_escalate, admit_kl=args.admit_kl, focus_sims=args.focus_sims, endgame_vp=args.endgame_vp,
        endgame_branches=args.endgame_branches, opening_branches=args.opening_branches,
        placement_forks=args.placement_forks, placement_temp=args.placement_temp, placement_uniform=args.placement_uniform, move_forks=args.move_forks, move_fork_temp=args.move_fork_temp, move_fork_uniform=args.move_fork_uniform, sibling_labels=args.sibling_labels, pool_board_share=args.hard_share, random_share=args.random_share, random_eps=args.random_eps, scarce_ows_share=args.scarce_ows_share, port_start_share=args.port_start_share, margin_weight=args.margin_weight,
        speed_weight=args.speed_weight, speed_turns=args.speed_turns, board=board, **rules_from_args(args))
    # Random-opponent games stop for good once search beats the random bot in
    # every evaluation game with none capped (recorded in meta, so a resumed
    # run keeps them off).
    if meta.get("random_off_at") is not None:
        self_play_kwargs["random_share"] = 0.0
        print(f"random-opponent games off since iteration {meta['random_off_at']}", flush=True)

    hard_boards = {}  # board seed -> smoothed opening value error (--hard-boards)

    def pool_paths():
        if args.pool_share <= 0:
            return []
        snaps = sorted(glob.glob(os.path.join(run_dir, "snap_*.ctnn")))
        snaps = [p for p in snaps if args.pool_min_iter <= int(os.path.basename(p)[5:11]) < meta["iters"]]
        return snaps[-args.pool_size:]

    teachers = []
    if args.teachers:
        if not args.remote_bucket:
            raise SystemExit("--teachers plays on the remote actors: needs --remote-bucket")
        if not 0.0 < args.teacher_share <= 1.0:
            raise SystemExit("--teacher-share must be in (0, 1]")
        with open(args.teachers) as f:
            for line in f:
                r = json.loads(line)
                if r["outcome"] == "success":
                    teachers.append({"board": f"seed:{r['seed']}",
                                     "path": os.path.join("runs", r["run"], f"snap_{r['best_iter']:06d}.ctnn")})
        if not teachers:
            raise SystemExit(f"no successful specialists in {args.teachers}")
        print(f"distilling {len(teachers)} teachers in {args.teacher_share:.0%} of actor rounds", flush=True)

    remote = None
    if args.remote_bucket:
        remote = Remote(args.remote_bucket, args.run, args.remote_games_per_core, args.remote_max_lag,
                        args.remote_max_shards, teachers, args.teacher_share, control_key=args.remote_control_key)
        remote.publish(meta["iters"], to_bytes(model, OBS, ACTIONS), pool_paths(), args.pool_share, self_play_kwargs, opening_bytes())
        print(f"remote actors: publishing to s3://{args.remote_bucket}", flush=True)

    pool_cache = {}
    start = time.time()
    it = 0
    while time.time() - start < args.minutes * 60 and not stop["requested"]:
        it += 1
        t0 = time.time()

        # ---- Self-play with the current network (and a pool of past ones).
        pool = []
        for p in pool_paths():
            if p not in pool_cache:
                with open(p, "rb") as f:
                    pool_cache[p] = f.read()
            pool.append(pool_cache[p])
        if args.hard_boards and not args.fixed_board:
            # Prioritised boards: the seeds with the largest opening value error so far.
            if len(hard_boards) >= args.hard_boards:
                top = sorted(hard_boards, key=hard_boards.get, reverse=True)[:args.hard_boards]
                self_play_kwargs["board"] = "seeds:" + ",".join(map(str, top))
            else:
                self_play_kwargs["board"] = None
        d = catanatron_rs.self_play(
            to_bytes(model, OBS, ACTIONS), args.games, args.seed * 1_000_003 + meta["iters"], args.threads,
            pool=pool, pool_share=args.pool_share if pool else 0.0, opening_net=opening_bytes(), **self_play_kwargs)
        for seed_b, err in d["boards"]:
            hard_boards[int(seed_b)] = 0.5 * hard_boards.get(int(seed_b), err) + 0.5 * err
        # Fork pairs this iteration (local + remote): count, sum of (main - fork), sum of alt prob, per kind.
        fork_acc = {k: [0, 0.0, 0.0] for k in ("open", "mid", "open_uniform", "mid_uniform")}
        for main_v, fork_v, p_alt, kind in d["fork_pairs"]:
            c = fork_acc[("open", "mid", "open_uniform", "mid_uniform")[kind]]
            c[0] += 1; c[1] += main_v - fork_v; c[2] += p_alt
        if d["aux_size"] != AUX:
            raise SystemExit(f"self-play gives {d['aux_size']} auxiliary targets, az.py expects {AUX}")
        n_local = d["n"]
        batches = [{
            "obs": np.frombuffer(d["obs"], "<f4").reshape(n_local, OBS).astype(np.float16),
            "mask": np.frombuffer(d["mask"], np.uint8).reshape(n_local, ACTIONS).astype(bool),
            "pi": np.frombuffer(d["pi"], "<f4").reshape(n_local, ACTIONS).astype(np.float16),
            "aux": np.frombuffer(d["aux"], "<f4").reshape(n_local, AUX),
            **{k: np.frombuffer(d[k], "<f4") for k in ("z", "z_luck", "q", "surprise")},
            "opening": np.frombuffer(d["opening"], np.uint8).astype(bool),
            "focus": np.frombuffer(d["focus"], np.uint8).astype(bool),
            "game": np.frombuffer(d["game"], "<u4"),
        }]
        remote_games, remote_lag, remote_dropped, teacher_shards = 0, [], 0, 0
        shard_stats = {"decisions": 0, "escalated": 0, "branches": 0, "forks": 0, "pool_games": 0, "luck_abs": 0.0, "luck_events": 0, "results": [0, 0, 0], "endgame": 0, "n": 0}
        if remote is not None:
            got, remote_dropped = remote.collect(meta["iters"])
            for header, arrays in got:
                batches.append({**arrays, "opening": arrays["opening"].astype(bool), "focus": arrays["focus"].astype(bool)})
                remote_games += header["games"]
                remote_lag.append(meta["iters"] - header["model_iter"])
                for seed_b, err in header.get("boards", []):
                    hard_boards[int(seed_b)] = 0.5 * hard_boards.get(int(seed_b), err) + 0.5 * err
                for kind, (cnt, gap, p_alt) in header.get("fork_pairs", {}).items():
                    c = fork_acc[kind]
                    c[0] += cnt; c[1] += gap * cnt; c[2] += p_alt * cnt
                teacher_shards += header.get("teacher") is not None
                # Game statistics of the shard's self-play, so the log reflects remote games too (not just the local ones).
                for k in ("decisions", "escalated", "branches", "forks", "pool_games", "luck_abs", "luck_events"):
                    shard_stats[k] += header.get("stats", {}).get(k, 0)
                shard_stats["results"] = [a + b for a, b in zip(shard_stats["results"], header.get("stats", {}).get("results", [0, 0, 0]))]
                shard_stats["endgame"] += int(arrays["endgame"].sum()); shard_stats["n"] += len(arrays["endgame"])
        batch = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
        n = len(batch["z"])
        if n == 0:
            # Nothing new this round (no local games and no shard arrived): wait
            # instead of counting an empty iteration and re-saving snapshots.
            time.sleep(5)
            continue
        z = batch["z"].astype(np.float64)
        luck_c = 0.0
        if args.luck:
            # Unbiased for any c, since every luck term has mean zero.
            later_luck = z - batch["z_luck"]
            luck_c = float(np.clip(np.cov(z, later_luck)[0, 1] / later_luck.var(), 0.0, 1.0))
            z_target = (z - luck_c * later_luck).astype(np.float32)
        else:
            z_target = z.astype(np.float32)
        # Value targets stay within what a finished game can be worth (the
        # luck adjustment can push them past it, and a network trained on
        # values beyond a win can't tell winning moves apart).
        bound = 1.0 + args.margin_weight + args.speed_weight
        value_target = np.clip((1 - args.value_q_weight) * z_target + args.value_q_weight * batch["q"], -bound, bound)
        if args.value_positions > 0:
            _, inv, counts = np.unique(batch["game"], return_inverse=True, return_counts=True)
            vw = np.minimum(1.0, args.value_positions / counts[inv]).astype(np.float32)
        else:
            vw = np.ones(len(batch["z"]), np.float32)
        replay.add(torch.from_numpy(batch["obs"]), torch.from_numpy(batch["mask"]), torch.from_numpy(batch["pi"]),
                   torch.from_numpy(value_target.astype(np.float32)),
                   torch.from_numpy(batch["surprise"]), torch.from_numpy(batch["aux"]),
                   torch.from_numpy(batch["opening"]), torch.from_numpy(batch["focus"]), torch.from_numpy(vw))
        n_open = int(batch["opening"].sum())
        if oreplay is not None and n_open:
            # The placement decisions, each counting fully (the sibling labels already average their continuations).
            o = batch["opening"]
            sel = lambda a: torch.from_numpy(a[o])
            oreplay.add(sel(batch["obs"]), sel(batch["mask"]), sel(batch["pi"]), sel(value_target.astype(np.float32)),
                        sel(batch["surprise"]), sel(batch["aux"]), sel(batch["opening"]), sel(batch["focus"]),
                        torch.ones(n_open))
        endgame_share = (float(np.frombuffer(d["endgame"], np.uint8).sum()) + shard_stats["endgame"]) / max(1, n_local + shard_stats["n"])
        luck = (d["luck_abs"] + shard_stats["luck_abs"]) / max(1, d["luck_events"] + shard_stats["luck_events"])
        games_seen = args.games + remote_games
        mean_surprise = float(batch["surprise"].mean()) if n else 0.0
        t_play = time.time() - t0

        # ---- Learn toward the search's visit counts and the results.
        t1 = time.time()
        steps = max(1, round(n * args.reuse / args.batch)) if replay.size >= args.min_replay else 0
        stats = [torch.zeros(6, device=device)] if steps == 0 else []
        model.train()
        for _ in range(steps):
            obs, mask, pi, z, aux, vw = replay.sample(args.batch, sym, args.surprise_share, args.opening_share, args.focus_share)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                if model.aux is not None:
                    logits, value, aux_pred = model.forward_aux(obs, mask)
                else:
                    logits, value = model(obs, mask)
            logp = torch.log_softmax(logits, -1)
            loss_pi = -(pi * logp).sum(-1).mean()
            loss_v = ((value - z) ** 2 * vw).sum() / vw.sum().clamp(min=1.0)
            if model.aux is not None:
                err = (aux_pred - aux) ** 2
                loss_aux, loss_aux2 = err[:, :AUX_BASE].mean(), err[:, AUX_BASE:].mean()
            else:
                loss_aux = loss_aux2 = torch.zeros((), device=device)
            loss = loss_pi + args.vf * loss_v + args.aux_weight * loss_aux + args.aux_extra_weight * loss_aux2
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # Policy entropy and the target's, to watch for collapse.
            stats.append(torch.stack([loss_pi.detach(), loss_v.detach(), loss_aux.detach(), loss_aux2.detach(),
                                      -(logp.exp() * logp.clamp(min=-1e4)).sum(-1).mean().detach(),
                                      -(pi * pi.clamp(min=1e-12).log()).sum(-1).mean()]))
        loss_pi, loss_v, loss_aux, loss_aux2, ent, target_ent = torch.stack(stats).mean(0).tolist()

        # ---- The opening network learns from the placement decisions alone.
        osteps, oloss_pi, oloss_v = 0, 0.0, 0.0
        if omodel is not None and n_open and oreplay.size >= args.opening_min_replay:
            osteps = max(1, round(n_open * args.opening_reuse / args.opening_batch))
            ostats = []
            omodel.train()
            for _ in range(osteps):
                obs, mask, pi, z, _, vw = oreplay.sample(args.opening_batch, sym)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    logits, value = omodel(obs, mask)
                logp = torch.log_softmax(logits, -1)
                o_pi = -(pi * logp).sum(-1).mean()
                o_v = ((value - z) ** 2 * vw).sum() / vw.sum().clamp(min=1.0)
                oopt.zero_grad()
                (o_pi + args.vf * o_v).backward()
                torch.nn.utils.clip_grad_norm_(omodel.parameters(), 1.0)
                oopt.step()
                ostats.append(torch.stack([o_pi.detach(), o_v.detach()]))
            oloss_pi, oloss_v = torch.stack(ostats).mean(0).tolist()
            ometa["steps"] += osteps
        if ometa is not None:
            ometa["examples"] += n_open
        t_learn = time.time() - t1

        meta["iters"] += 1
        meta["games"] += args.games + remote_games
        if meta.get("value_fade"):
            fade = meta["value_fade"]
            model.flat_value_scale.fill_(max(0.0, 1.0 - (meta["games"] - fade["start"]) / fade["games"]))
        meta["examples"] += n
        save(ckpt_path, model, opt, meta)
        if omodel is not None:
            save(ockpt_path, omodel, oopt, ometa)

        entry = {"iter": meta["iters"], "games": meta["games"], "examples": meta["examples"],
                 "minutes": (time.time() - start) / 60, "new_examples": n, "replay": replay.size,
                 "results": [a + b for a, b in zip(d["results"], shard_stats["results"])],
                 "decisions_per_game": (d["decisions"] + shard_stats["decisions"]) / max(1, games_seen),
                 "escalated": d["escalated"] + shard_stats["escalated"], "branches": d["branches"] + shard_stats["branches"], "forks": d["forks"] + shard_stats["forks"],
                 "pool_games": d["pool_games"] + shard_stats["pool_games"], "pool_size": len(pool), "random_games": d["random_games"], "endgame_share": endgame_share,
                 "surprise": mean_surprise, "luck_per_event": luck, "luck_c": luck_c,
                 "loss_pi": loss_pi, "loss_v": loss_v, "loss_aux": loss_aux, "loss_aux2": loss_aux2, "entropy": ent, "target_entropy": target_ent,
                 "fork_pairs": {k: {"n": c[0], "gap": c[1] / c[0], "alt_p": c[2] / c[0]} for k, c in fork_acc.items() if c[0]},
                 "train_steps": steps, "play_s": t_play, "learn_s": t_learn}
        if omodel is not None:
            entry["opening"] = {"new": n_open, "replay": oreplay.size, "steps": osteps, "total_steps": ometa["steps"],
                                "loss_pi": oloss_pi, "loss_v": oloss_v}
        if remote is not None:
            entry.update({"remote_shards": len(remote_lag), "remote_games": remote_games, "teacher_shards": teacher_shards,
                          "remote_examples": n - n_local, "remote_dropped": remote_dropped,
                          "remote_lag": float(np.mean(remote_lag)) if remote_lag else None})
        if meta["iters"] % args.eval_every == 0:
            model.eval()
            entry["vs_F"] = evaluate(run_dir, model, meta["iters"], args, board, omodel)
            r = entry["vs_F"]
            if self_play_kwargs["random_share"] > 0 and r.get("vsR") == 1.0 and r.get("vsR_capped") == 0.0:
                self_play_kwargs["random_share"] = 0.0
                meta["random_off_at"] = meta["iters"]
                save(ckpt_path, model, opt, meta)
                print(f"beat the random bot in every evaluation game: random-opponent games off from iteration {meta['iters']}", flush=True)
        entry["random_share"] = self_play_kwargs["random_share"]
        if remote is not None:  # after the eval, so a new snapshot can join the pool
            remote.publish(meta["iters"], to_bytes(model, OBS, ACTIONS), pool_paths(), args.pool_share, self_play_kwargs, opening_bytes())
        log.write(json.dumps(entry) + "\n")
        log.flush()
        ev = "  vs F " + " ".join(f"{k} {v:.1%}" for k, v in entry["vs_F"].items()) if "vs_F" in entry else ""
        print(f"it {meta['iters']:4d}  games {meta['games']:>8,}  ex {n:5d}  replay {replay.size:>7,}  "
              f"pi {loss_pi:.3f} v {loss_v:.3f} aux {loss_aux:.3f}/{loss_aux2:.3f} ent {ent:.2f}/{target_ent:.2f}  "
              + "".join(f"fork-{k} {c[1] / c[0]:+.3f}@{c[2] / c[0]:.2f} " for k, c in fork_acc.items() if c[0])
              + f"surp {mean_surprise:.2f} luck {luck:.3f} c {luck_c:.2f} esc {d['escalated'] + shard_stats['escalated']} br {d['branches'] + shard_stats['branches']} fk {d['forks'] + shard_stats['forks']} pool {d['pool_games'] + shard_stats['pool_games']} rnd {d['random_games']} end {endgame_share:.0%}  "
              f"play {t_play:.1f}s learn {t_learn:.2f}s{ev}"
              + (f"  open {n_open} ex replay {oreplay.size:,} pi {oloss_pi:.3f} v {oloss_v:.3f} steps {osteps}" if omodel is not None else "")
              + (f"  remote {entry['remote_shards']} shards {entry['remote_examples']} ex "
                 f"lag {entry['remote_lag']} drop {remote_dropped}" if remote is not None else "")
              + (f" teacher {teacher_shards}" if teachers else ""), flush=True)

    replay.save(replay_path)
    if oreplay is not None:
        oreplay.save(oreplay_path)
    print(f"done: {meta['iters']} iterations, {meta['games']:,} games -> {ckpt_path} (replay {replay.size:,} saved)"
          + (f"; opening network {ometa['steps']:,} steps -> {ockpt_path}" if omodel is not None else ""))


if __name__ == "__main__":
    main()
