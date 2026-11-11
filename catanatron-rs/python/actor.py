"""Remote self-play actor: plays games with the learner's latest network and
uploads them to S3 as shards (see shards.py). Runs until stopped.

    SHARD_BUCKET=<bucket> python actor.py

The learner (az.py --remote-bucket) publishes models/control.json; each
round the actor reads it, plays `games_per_core` games per core with that
network (and its opponent pool), and uploads one shard.

With teachers in the control file (az.py --teachers), a `teacher_share` of
rounds are teacher rounds instead: one teacher (a board specialist) plays
itself on its own board, and the shard (marked with the teacher's board) is
training data for the learner's network: distillation.
"""

import argparse
import json
import os
import queue
import random
import signal
import threading
import time
import traceback
import uuid

import boto3
import catanatron_rs

import shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("SHARD_BUCKET"))
    ap.add_argument("--control", default=shards.CONTROL_KEY, help="the learner's control file key (one learner per key)")
    ap.add_argument("--threads", type=int, default=0, help="Rust threads (0 = all cores)")
    ap.add_argument("--rounds", type=int, default=0, help="stop after this many shards (0 = run until stopped)")
    args = ap.parse_args()
    if not args.bucket:
        raise SystemExit("no bucket: pass --bucket or set SHARD_BUCKET")
    s3 = boto3.client("s3")
    actor = uuid.uuid4().hex[:12]
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True
        print(f"stop requested (signal {signum})", flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    cores = args.threads or os.cpu_count()
    nets = {}  # key -> bytes

    # Compressing and uploading a shard takes seconds of one core; a worker
    # does it while the next batch plays. At most one batch waits.
    pending = queue.Queue(maxsize=1)

    def uploader():
        while True:
            item = pending.get()
            if item is None:
                return
            d, meta, key, label = item
            try:
                blob = shards.encode(d, meta)
                s3.put_object(Bucket=args.bucket, Key=key, Body=blob)
            except BaseException:
                # The main loop would block on the full queue forever: end the
                # whole actor so the task shows as failed.
                traceback.print_exc()
                os._exit(1)
            print(f"{key}: {meta['games']} games{label}, {d['n']} examples, {len(blob) / 1e6:.2f} MB, "
                  f"play {meta['play_s']:.0f}s", flush=True)
    worker = threading.Thread(target=uploader, daemon=True)
    worker.start()

    def net(key):
        if key not in nets:
            nets[key] = s3.get_object(Bucket=args.bucket, Key=key)["Body"].read()
        return nets[key]

    print(f"actor {actor}: {cores} cores, bucket {args.bucket}", flush=True)
    seq = 0
    while not stop["requested"] and (args.rounds == 0 or seq < args.rounds):
        try:
            control = json.loads(s3.get_object(Bucket=args.bucket, Key=args.control)["Body"].read())
        except s3.exceptions.NoSuchKey:
            print("no control file yet; waiting for the learner", flush=True)
            time.sleep(30)
            continue
        teachers = control.get("teachers", [])
        teacher = random.choice(teachers) if teachers and random.random() < control["teacher_share"] else None
        # Only networks still in use stay cached.
        wanted = {control["model"], *control["pool"], *(t["key"] for t in teachers)}
        for k in list(nets):
            if k not in wanted:
                del nets[k]
        games = control["games_per_core"] * cores
        seed = int.from_bytes(os.urandom(8), "little") >> 1
        t0 = time.time()
        if teacher is None:
            d = catanatron_rs.self_play(
                net(control["model"]), games, seed, args.threads,
                pool=[net(k) for k in control["pool"]], pool_share=control["pool_share"] if control["pool"] else 0.0,
                **control["self_play"])
        else:
            d = catanatron_rs.self_play(net(teacher["key"]), games, seed, args.threads,
                                        **{**control["self_play"], "board": teacher["board"]})
        t_play = time.time() - t0
        meta = {"run": control["run"], "model_iter": control["iter"], "games": games, "actor": actor,
                "obs_size": control["obs_size"], "action_size": control["action_size"], "play_s": t_play,
                "teacher": teacher["board"] if teacher else None}
        key = f"shards/{control['run']}/{control['iter']:06d}/{actor}-{seq:05d}.npz.xz"
        label = f" (teacher {teacher['board']})" if teacher else ""
        pending.put((d, meta, key, label))
        seq += 1
    # Finish the last upload before exiting.
    pending.put(None)
    worker.join()


if __name__ == "__main__":
    main()
