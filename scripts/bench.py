"""Isolated training throughput (tokens/s) per arm config; run when the GPU is otherwise idle."""
import json
import sys
import time

import torch

from train import Trainer, get_args


def bench(scale, arm, steps=40, warm=15):
    t = Trainer(get_args(["--scale", scale, "--arm", arm, "--run_name", f"bench_{scale}_{arm}", "--micro", "24" if scale == "S" else "16",
                          "--results_dir", "runs/bench", "--runs_dir", "runs/bench"]))
    for s in range(warm):
        t.model.update_masks(s)
        t.train_step(s)
    torch.cuda.synchronize()
    t0 = time.time()
    for s in range(warm, steps):
        t.model.update_masks(s)
        t.train_step(s)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return (steps - warm) * t.tokens_per_step / dt


if __name__ == "__main__":
    scale = sys.argv[1] if len(sys.argv) > 1 else "S"
    out = {}
    for arm in ("scratch", "grow_uniform", "openelm", "base_half"):
        out[arm] = bench(scale, arm)
        print(scale, arm, f"{out[arm]:.0f} tok/s", flush=True)
    json.dump(out, open(f"results/bench_{scale}.json", "w"), indent=1)
