"""PPO (train.py) on an actor host, with the run kept in S3.

    SHARD_BUCKET=<bucket> python ppo_remote.py <run> [train.py args...]

The run directory (runs/<run>: model.pt, log.jsonl, pool/, model_beat_*.pt)
lives at s3://<bucket>/ppo/<run>/. A run already there is downloaded and
resumed (so a replaced spot instance carries on); the directory is uploaded
every SYNC_SECONDS and when train.py exits. train.py checkpoints every 20
iterations, so an interruption loses at most that much.
"""

import os
import signal
import subprocess
import sys
import time

import boto3

SYNC_SECONDS = 120


def main():
    bucket = os.environ.get("SHARD_BUCKET")
    if not bucket or len(sys.argv) < 2:
        raise SystemExit("usage: SHARD_BUCKET=<bucket> python ppo_remote.py <run> [train.py args...]")
    run, train_args = sys.argv[1], sys.argv[2:]
    s3 = boto3.client("s3")
    prefix = f"ppo/{run}/"
    local = os.path.join("runs", run)
    os.makedirs(os.path.join(local, "pool"), exist_ok=True)

    # Restore a run from S3.
    restored = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            path = os.path.join(local, obj["Key"][len(prefix):])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            s3.download_file(bucket, obj["Key"], path)
            restored += 1
    resume = os.path.exists(os.path.join(local, "model.pt"))
    print(f"ppo_remote: run {run}, restored {restored} files from s3://{bucket}/{prefix}"
          f"{' (resuming)' if resume else ''}", flush=True)

    uploaded = {}  # relative path -> mtime uploaded

    def sync():
        present = set()
        for root, _, files in os.walk(local):
            for name in files:
                if name.endswith(".tmp"):
                    continue
                path = os.path.join(root, name)
                rel = os.path.relpath(path, local)
                present.add(rel)
                mtime = os.path.getmtime(path)
                if uploaded.get(rel) != mtime:
                    s3.upload_file(path, bucket, prefix + rel)
                    uploaded[rel] = mtime
        # Pool snapshots train.py pruned locally are pruned in S3 too.
        for rel in [r for r in uploaded if r not in present]:
            s3.delete_object(Bucket=bucket, Key=prefix + rel)
            del uploaded[rel]

    cmd = [sys.executable, "-u", "train.py", "--run", run] + (["--resume"] if resume else []) + train_args
    proc = subprocess.Popen(cmd)
    stopping = {"requested": False}

    def stop(signum, frame):
        stopping["requested"] = True
        proc.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    last = time.time()
    while proc.poll() is None:
        time.sleep(5)
        if time.time() - last >= SYNC_SECONDS:
            sync()
            last = time.time()
    sync()
    print(f"ppo_remote: train.py exited with {proc.returncode}; run synced to s3://{bucket}/{prefix}", flush=True)
    sys.exit(0 if stopping["requested"] else proc.returncode)


if __name__ == "__main__":
    main()
