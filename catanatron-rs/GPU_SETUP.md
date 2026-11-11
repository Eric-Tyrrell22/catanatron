# Continuing training on a GPU machine

This folder contains the Rust engine, the Python training code, and the
`long2` run (checkpoint + self-play opponent pool) so training can resume
exactly where it stopped (72.7M steps, ~47% vs F network-only, ~53% with
search).

Requirements: an NVIDIA GPU (any RTX 30/40 series works; bf16 needs Ampere
or newer), a recent NVIDIA driver, and a multi-core CPU (the games run on
the CPU; 8+ cores keeps the GPU busy).

## 1. Install tools

Linux:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust
curl -LsSf https://astral.sh/uv/install.sh | sh                   # uv (Python env manager)
nvidia-smi                                                        # driver check: should list your GPU
```

Windows: install Rust from https://rustup.rs (accept the Visual Studio
Build Tools prompt), uv from https://docs.astral.sh/uv/, and check the
driver with `nvidia-smi` in PowerShell.

## 2. Build

From this folder (`catanatron-rs/`):

```sh
cargo build --release          # engine + CLI; also checks the toolchain
cargo test --release           # optional, ~1 minute

cd python
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy maturin
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/python -c "import torch, catanatron_rs; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line should print `True` and your GPU's name. On Windows use
`.venv\Scripts\python` instead of `.venv/bin/python`, and run maturin with
`$env:VIRTUAL_ENV="$PWD\.venv"; .venv\Scripts\maturin develop --release`.

`.cargo/config.toml` compiles for the CPU doing the build
(`target-cpu=native`). That's what you want: build on the machine that
runs it. Delete that file if you ever need a binary for a different CPU.

## 3. Resume training

```sh
cd python
nohup .venv/bin/python train.py --run long2 --resume --minutes 600 \
    --device cuda --bf16 --torch-threads 4 >> runs/long2.out 2>&1 &
tail -f runs/long2.out
```

(Windows: run the same `train.py` command without `nohup`/`&`, in its own
terminal.)

- `--device cuda --bf16`: the learning step on the GPU in bfloat16.
- `--torch-threads 4`: PyTorch barely needs the CPU now; leave cores to the
  Rust game engine (`--env-threads 0` = all cores, the default).
- Each printed line shows `play` (games, CPU) and `learn` (GPU) seconds per
  iteration. If `play` is much larger than `learn`, the CPU is the limit.
  More envs per iteration (`--num-envs 256`) gives the GPU bigger batches;
  it's safe to change on resume.
- Checkpoints go to `runs/long2/model.pt` every 20 iterations; after a crash
  or reboot, run the same command again (it resumes from the checkpoint).

## 4. Check strength

```sh
cd python
.venv/bin/python eval.py --run long2                    # 200 games vs F, with and without search
.venv/bin/python eval.py --run long2 --games 1000 --sims 0   # tighter estimate, network only
```

Results append to `runs/long2/eval.jsonl` (history from the laptop is
included). 200 games gives about ±7%; use 1000 for decisions.

## What's in the run folder

- `runs/long2/model.pt`: weights + optimizer state + step counter.
- `runs/long2/pool/`: the 20 most recent frozen snapshots used as self-play
  opponents (needed so self-play resumes with the same opponents).
- `runs/long2/log.jsonl`, `eval.jsonl`: training and evaluation history.
- `runs/long1/model.pt`: the earlier run `long2` continued from.
