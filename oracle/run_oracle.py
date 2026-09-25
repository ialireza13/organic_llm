"""Stage 1 oracle: from a checkpoint, grow one candidate unit (or nothing = control), continue training
with the same data order and LR schedule, and record validation loss.

One process handles all candidates for one (checkpoint, seed), reusing the compiled model.
"""
import argparse
import json
import os
import time

import torch

from growth.operators import grow_unit
from train import Trainer, get_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0, help="new-unit init seed")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ramp", type=int, default=30)
    ap.add_argument("--eval_tokens", type=int, default=4_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", default=None, help="comma-separated subset, e.g. control,ffn:0")
    ap.add_argument("--micro", default="24")
    ap.add_argument("--eval_at", default="", help="extra intermediate eval points (steps after growth)")
    a = ap.parse_args()

    sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    targs = sd["args"]
    argv = ["--scale", targs["scale"], "--arm", targs["arm"], "--seed", str(targs["seed"]), "--lr", str(targs["lr"]),
            "--micro", a.micro, "--run_name", "oracle_tmp", "--results_dir", "runs/oracle_tmp", "--runs_dir", "runs/oracle_tmp"]
    if targs.get("tokens"):
        argv += ["--tokens", str(targs["tokens"])]
    t = Trainer(get_args(argv))
    assert t.total_steps == sd["total_steps"]
    L = t.cfg.n_layer
    cands = ["control"] + [f"ffn:{l}" for l in range(L)] + [f"head:{l}" for l in range(L)]
    if a.candidates:
        cands = a.candidates.split(",")
    s0 = sd["step"]
    results = {"ckpt": a.ckpt, "seed": a.seed, "step0": s0, "steps": a.steps, "ramp": a.ramp,
               "eval_tokens": a.eval_tokens, "runs": {}}
    if os.path.exists(a.out):
        results = json.load(open(a.out))
    for c in cands:
        if c in results["runs"]:
            continue
        t0 = time.time()
        t.load_state(sd)
        t.model.cfg.ramp_steps = a.ramp
        if c != "control":
            kind, layer = c.split(":")
            gen = torch.Generator(device="cuda")
            gen.manual_seed(1000 + a.seed)
            grow_unit(t.model, t.opt, (kind, int(layer)), s0, gen)
        losses = []
        inter = {}
        eval_at = {int(x) for x in a.eval_at.split(",") if x}
        for s in range(s0, s0 + a.steps):
            if s - s0 in eval_at:
                t.model.update_masks(s)
                inter[s - s0] = t.evaluate(a.eval_tokens)
            t.model.update_masks(s)
            loss, _, _ = t.train_step(s)
            if (s - s0) % 20 == 0:
                losses.append(float(loss))
        t.model.update_masks(s0 + a.steps)
        vl = t.evaluate(a.eval_tokens)
        inter[a.steps] = vl
        results["runs"][c] = {"val_loss": vl, "val_at": inter, "train_losses": losses, "seconds": time.time() - t0,
                              "flops_per_token": t.model.flops_per_token()}
        print(f"{c}: val {vl:.5f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
