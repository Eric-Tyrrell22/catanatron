"""Distill a 4x512 network into smaller 4-layer students by matching its
outputs (soft policy via KL, value via MSE) on stored positions with board
symmetries, then play each student against the teacher network-only.

    python runs/distill_small.py <teacher.pt> <replay.pt> <hidden> [<hidden> ...] [--steps N]
Writes runs/distill/h<hidden>.ctnn (+ .pt) and prints held-out agreement,
10k-game match vs the teacher, and forward-pass speed.
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
import catanatron_rs
import model as M
from export import to_bytes
from catanatron_env import OPENHEX_1V1

args = [a for a in sys.argv[1:] if not a.startswith("--")]
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 20000
teacher_path, replay_path, sizes = args[0], args[1], [int(h) for h in args[2:]]
dev = torch.device("cuda")
ck = torch.load(teacher_path, map_location="cpu")
OBS, ACT = ck["meta"]["obs_size"], ck["meta"]["action_size"]
teacher = M.Policy(OBS, ACT, 512, ck["meta"].get("aux", 0), 4).to(dev); teacher.load_state_dict(ck["model"]); teacher.eval()
os.makedirs("runs/distill", exist_ok=True)
with open("runs/distill/teacher.ctnn", "wb") as f:
    f.write(to_bytes(teacher, OBS, ACT))

d = torch.load(replay_path, map_location="cpu", mmap=True)
n = d["z"].shape[0]
obs_all, mask_all = d["obs"], d["mask"]
hold = torch.arange(n - 50_000, n)
perms = catanatron_rs.board_symmetries(OBS, "BASE")
sym_obs = torch.tensor([p[0] for p in perms], device=dev); sym_act = torch.tensor([p[1] for p in perms], device=dev)
rules = {"map_type": "BASE", **OPENHEX_1V1}


def batch(idx, sym):
    o, m = obs_all[idx].to(dev).float(), mask_all[idx].to(dev)
    if sym:
        k = torch.randint(0, sym_obs.shape[0], (len(idx),), device=dev)
        o, m = o.gather(1, sym_obs[k]), m.gather(1, sym_act[k])
    return o, m


@torch.no_grad()
def teach(o, m):
    lg, v = teacher(o, m)
    return F.log_softmax(lg, -1), v


def losses(student, o, m, tlp, tv):
    lg, v = student(o, m)
    slp = F.log_softmax(lg, -1)
    kl = (tlp.exp() * (tlp - slp)).masked_fill(~m, 0).sum(-1).mean()
    return kl, F.mse_loss(v, tv)


@torch.no_grad()
def held_out(student):
    kl = mse = agree = 0.0
    for i in range(0, len(hold), 8192):
        o, m = batch(hold[i:i + 8192], sym=False)
        tlp, tv = teach(o, m)
        k, e = losses(student, o, m, tlp, tv)
        lg, _ = student(o, m)
        kl += k.item() * len(o); mse += e.item() * len(o); agree += (lg.argmax(-1) == tlp.argmax(-1)).float().sum().item()
    return kl / len(hold), mse / len(hold), agree / len(hold)


def speed(path):
    """Network evaluations per second, single thread, via play_bots on random boards."""
    t = time.time()
    catanatron_rs.play_bots([f"MCTS:400,c1,f0.25:{path}", f"MCTS:400,c1,f0.25:{path}"], 2, 4242, 1, **rules)
    return time.time() - t


base_t = speed(os.path.abspath("runs/distill/teacher.ctnn"))
print(f"teacher 4x512: 2 games at 400 sims, 1 thread: {base_t:.1f}s", flush=True)
for H in sizes:
    torch.manual_seed(0)
    student = M.Policy(OBS, ACT, H, 0, 4).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    t = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, n - 50_000, (4096,)).sort().values
        o, m = batch(idx, sym=True)
        tlp, tv = teach(o, m)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            kl, mse = losses(student, o, m, tlp, tv)
        loss = kl + mse
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 5000 == 0 or step == STEPS:
            student.eval(); hk, hm, ha = held_out(student); student.train()
            print(f"{time.strftime('%H:%M')} 4x{H} step {step}: held-out KL {hk:.4f} value MSE {hm:.4f} top-1 agreement {ha:.1%} ({time.time() - t:.0f}s)", flush=True)
    student.eval()
    path = os.path.abspath(f"runs/distill/h{H}.ctnn")
    with open(path, "wb") as f:
        f.write(to_bytes(student.cpu(), OBS, ACT))
    torch.save({"model": student.state_dict(), "meta": {**ck["meta"], "hidden": H, "layers": 4, "aux": 0}}, f"runs/distill/h{H}.pt")
    st = speed(path)
    (w, l), _ = catanatron_rs.play_bots(["NN:" + path, "NN:" + os.path.abspath("runs/distill/teacher.ctnn")], 10000, 70_000, 6, **rules)
    print(f"{time.strftime('%H:%M')} 4x{H}: vs teacher network-only {w / 100:.1f}% (±0.5, 10k games); search speed {base_t / st:.2f}x the teacher's", flush=True)
