"""PPO training: a bot curriculum, then self-play mixed with games vs F.

    .venv/bin/python train.py --run my-run --minutes 60
    .venv/bin/python train.py --run my-run --resume --minutes 60
    .venv/bin/python train.py --run my-run --device cuda --bf16     # GPU machine

Games are played entirely in Rust (`VecEnv.rollout`): the learner's moves
are sampled from an exported copy of the network, and self-play opponents
are Rust bots ("NNS:<snapshot>") that sample from frozen past copies.
PyTorch only does the learning step.

Phase 1 trains against each bot in --curriculum until it wins --advance-at
of recent games. Phase 2 plays a --f-fraction share of games against F (the
strength benchmark) and the rest against past copies of itself.
Checkpoints and a JSON-lines log go to runs/<run>/. Evaluate with eval.py.
"""

import argparse
import glob
import json
import os
import random
import time
from collections import deque

# numpy's idle BLAS threads would spin on cores the Rust games need.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

from catanatron_env import VecEnv, add_rule_args, rollout_buffers, rules_from_args
from export import to_bytes

from model import Policy


def save(path, model, opt, meta):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "meta": meta}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--curriculum", default="R,W", help="bots to beat before self-play")
    ap.add_argument("--advance-at", type=float, default=0.75, help="win rate to move on")
    ap.add_argument("--f-fraction", type=float, default=0.25,
                    help="share of self-play phase games vs the fixed opponent (--fixed-opponent)")
    ap.add_argument("--fixed-opponent", default="F",
                    help="bot key for that share, e.g. F or NNS:<model.ctnn> (a network, sampling its moves)")
    ap.add_argument("--fixed-label", default="F", help="name of the fixed opponent in the logs")
    ap.add_argument("--snapshot-every", type=int, default=10, help="iterations between pool snapshots")
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--vp-reward", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-threads", type=int, default=0, help="Rust threads (0 = all cores)")
    ap.add_argument("--torch-threads", type=int, default=0, help="PyTorch threads for updates (0 = all cores)")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:1, ...")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast for the update (good on GPUs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--opponent-hands", action="store_true",
                    help="network sees opponents' hands by resource (public knowledge in 1v1)")
    ap.add_argument("--opponent-devs", action="store_true",
                    help="network sees development-card tracking (unseen pool, expected opponent cards, holding streaks)")
    ap.add_argument("--extra-info", action="store_true",
                    help="network sees tile numbers (+ rolls left), the first player, opponents' this-turn dev activity")
    ap.add_argument("--spot-features", action="store_true",
                    help="network sees per-node settlement spots (pips, legality, road closeness) and port discounts")
    add_rule_args(ap)
    args = ap.parse_args()
    rules = rules_from_args(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads or os.cpu_count())
    device = torch.device(args.device)
    run_dir = os.path.join("runs", args.run)
    pool_dir = os.path.join(run_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "model.pt")
    curriculum = args.curriculum.split(",")
    N, T = args.num_envs, args.rollout

    def env(n, opponent, seed):
        return VecEnv(n, [opponent], seed=seed, threads=args.env_threads, vp_reward=args.vp_reward,
                      opponent_hands=args.opponent_hands, opponent_devs=args.opponent_devs,
                      extra_info=args.extra_info, spot_features=args.spot_features, **rules)

    def stage_name(stage):
        return curriculum[stage] if stage < len(curriculum) else f"{args.fixed_label}+SELF"

    stage, total_steps, iters_done = 0, 0, 0
    probe = env(1, "R", 0)
    model = Policy(probe.obs_size, probe.action_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        stage, total_steps = ck["meta"]["stage"], ck["meta"]["total_steps"]
        iters_done = ck["meta"].get("iters", 0)
        print(f"resumed at {stage_name(stage)}, {total_steps:,} steps", flush=True)
    sizes = (probe.obs_size, probe.action_size)

    # ---- Opponent pool: frozen past selves as CTNN files.
    def snapshot():
        path = os.path.abspath(os.path.join(pool_dir, f"snap_{total_steps:012d}.ctnn"))
        with open(path + ".tmp", "wb") as f:
            f.write(to_bytes(model, *sizes))
        os.replace(path + ".tmp", path)
        for old in sorted(glob.glob(os.path.join(pool_dir, "snap_*.ctnn")))[:-args.pool_size]:
            os.remove(old)

    def pick_opponent():
        snaps = sorted(glob.glob(os.path.abspath(os.path.join(pool_dir, "snap_*.ctnn"))))
        # Half the time the newest snapshot, otherwise any past one.
        return "NNS:" + (snaps[-1] if random.random() < 0.5 else random.choice(snaps))

    def make_groups(stage, seed):
        """[(label, VecEnv, first env, end env)]: rows of the shared buffers."""
        if stage < len(curriculum):
            return [(curriculum[stage], env(N, curriculum[stage], seed), 0, N)]
        n_f = max(1, round(N * args.f_fraction))
        if not glob.glob(os.path.join(pool_dir, "snap_*.ctnn")):
            snapshot()
        return [(args.fixed_label, env(n_f, args.fixed_opponent, seed), 0, n_f),
                ("SELF", env(N - n_f, pick_opponent(), seed + 1), n_f, N)]

    # Rust writes rollouts straight into these, reused every iteration.
    # Observations and masks are most of the data: pin them for fast GPU copies.
    pin = device.type == "cuda"

    def empty(shape, dtype):
        if pin and dtype in (np.float32, np.uint8):
            return torch.empty(shape, dtype=torch.float32 if dtype == np.float32 else torch.uint8,
                               pin_memory=True).numpy()
        return np.empty(shape, dtype)

    buf = rollout_buffers(N, T, *sizes, empty)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    groups = make_groups(stage, args.seed + 1000 * stage + iters_done)
    recent = {}  # label -> deque of 1/0 learner wins
    start = time.time()
    it = 0
    while time.time() - start < args.minutes * 60:
        it += 1
        t0 = time.time()

        # ---- Play games in Rust with the current weights.
        weights = to_bytes(model, *sizes)
        done_np, won_np = buf["done"].view(bool), buf["won"].view(bool)
        for label, e, lo, hi in groups:
            if label == "SELF":
                e.set_opponents([pick_opponent()])  # applies as games reset
            e.set_policy(weights)
            e.rollout(T, {k: v[lo:hi] for k, v in buf.items()})
            recent.setdefault(label, deque(maxlen=400)).extend(won_np[lo:hi][done_np[lo:hi]].astype(int).tolist())
        t_play = time.time() - t0

        obs = torch.from_numpy(buf["obs"]).to(device, non_blocking=True)          # (N, T, O)
        mask = torch.from_numpy(buf["mask"]).to(device, non_blocking=True).bool()
        act = torch.from_numpy(buf["action"].astype(np.int64)).to(device)
        old_logp = torch.from_numpy(buf["logp"]).to(device)
        val = torch.from_numpy(buf["value"])
        rew = torch.from_numpy(buf["reward"])
        done = torch.from_numpy(done_np.astype(np.float32))
        last_val = torch.from_numpy(buf["last_value"])

        # ---- Advantages (GAE) along each env's steps.
        adv = torch.zeros_like(rew)
        last = torch.zeros(rew.shape[0])
        for t in reversed(range(T)):
            nonterminal = 1.0 - done[:, t]
            nv = last_val if t == T - 1 else val[:, t + 1]
            delta = rew[:, t] + args.gamma * nv * nonterminal - val[:, t]
            last = delta + args.gamma * args.lam * nonterminal * last
            adv[:, t] = last
        ret = (adv + val).to(device)
        adv = adv.to(device)

        # ---- PPO update.
        t1 = time.time()
        B = N * T
        b_obs, b_mask = obs.reshape(B, -1), mask.reshape(B, -1)
        b_act, b_logp, b_adv, b_ret = act.reshape(B), old_logp.reshape(B), adv.reshape(B), ret.reshape(B)
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(B, device=device)
            for i in range(0, B, args.minibatch):
                idx = perm[i:i + args.minibatch]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                    logits, value = model(b_obs[idx], b_mask[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vf = 0.5 * (value - b_ret[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + args.vf * vf - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                stats.append(torch.stack([pg.detach(), vf.detach(), ent.detach(),
                                          ((ratio - 1).abs() > args.clip).float().mean().detach()]))
        # Also waits for the GPU, so `buf` is free to be overwritten next iteration.
        pg, vf, ent, clipfrac = torch.stack(stats).mean(0).tolist()
        t_update = time.time() - t1

        total_steps += B
        iters_done += 1
        if stage >= len(curriculum) and it % args.snapshot_every == 0:
            snapshot()

        rates = {k: float(np.mean(v)) for k, v in recent.items() if v}
        dt = time.time() - t0
        entry = {
            "iter": iters_done, "steps": total_steps, "minutes": (time.time() - start) / 60,
            "phase": stage_name(stage), "win_rates": rates,
            "episodes_window": {k: len(v) for k, v in recent.items()},
            "pg_loss": pg, "v_loss": vf, "entropy": ent, "clipfrac": clipfrac,
            "steps_per_s": B / dt, "play_s": t_play, "update_s": t_update,
        }
        log.write(json.dumps(entry) + "\n")
        log.flush()
        if it % 5 == 0:
            wr = "  ".join(f"vs {k} {v:5.1%} (n={len(recent[k])})" for k, v in sorted(rates.items()))
            print(f"it {iters_done:5d}  steps {total_steps:>11,}  {wr}  ent {ent:.2f}  clip {clipfrac:.3f}  "
                  f"{B / dt:,.0f} steps/s (play {t_play:.3f}s, learn {t_update:.3f}s)", flush=True)

        meta = {"stage": stage, "total_steps": total_steps, "iters": iters_done, "obs_size": sizes[0],
                "action_size": sizes[1], "phase": stage_name(stage)}
        if it % 20 == 0:
            save(ckpt_path, model, opt, meta)

        # ---- Curriculum: beat each bot, then switch to self-play + F.
        if stage < len(curriculum):
            w = recent.get(curriculum[stage], [])
            if len(w) == 400 and np.mean(w) >= args.advance_at:
                save(os.path.join(run_dir, f"model_beat_{curriculum[stage]}.pt"), model, opt, meta)
                stage += 1
                print(f"==> phase {stage_name(stage)}", flush=True)
                if stage >= len(curriculum):
                    snapshot()
                groups = make_groups(stage, args.seed + 1000 * stage + iters_done)
                recent.clear()

    save(ckpt_path, model, opt, {"stage": stage, "total_steps": total_steps, "iters": iters_done,
                                  "obs_size": sizes[0], "action_size": sizes[1],
                                  "phase": stage_name(stage)})
    print(f"done: {total_steps:,} steps in {(time.time() - start) / 60:.1f} min -> {ckpt_path}")



if __name__ == "__main__":
    main()
